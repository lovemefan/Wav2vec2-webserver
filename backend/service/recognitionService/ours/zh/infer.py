# -*- coding: utf-8 -*-
# @Time  : 2022/5/26 17:26
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : infer.py.py
import os
import sys


import math
import time
import ctc_segmentation
from pip import main
import logging
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils, pdb
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from sanic.request import File

from backend.decorator.singleton import singleton
from backend.utils.AudioReader import AudioReader
from fairseq_lib.examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_asr_eval_argument(parser):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument("--data", default=None, help="dict")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonaryoutput units",
    )
    try:
        parser.add_argument(
            "--lm-weight",
            "--lm_weight",
            type=float,
            default=0.2,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    parser.add_argument(
        "--w2l-decoder",
        choices=["viterbi", "kenlm", "fairseqlm"],
        help="use a w2l decoder",
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--unit-lm", action="store_true", help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    return parser


def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
            not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
            args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"

def load_models_and_criterions(
        filenames, data_path, arg_overrides=None, task=None, model_state=None
):
    models = []
    criterions = []

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides["wer_args"] = None
    arg_overrides["data"] = data_path

    if filenames is None:
        assert model_state is not None
        filenames = [0]
    else:
        filenames = filenames.split(":")

    for filename in filenames:
        if model_state is None:
            if not os.path.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
        else:
            state = model_state

        if "cfg" in state:
            cfg = state["cfg"]
        else:
            cfg = convert_namespace_to_omegaconf(state["args"])

        if task is None:
            if hasattr(cfg.task, 'data'):
                cfg.task.data = data_path
            task = tasks.setup_task(cfg.task)

        model = task.build_model(cfg.model)
        model.load_state_dict(state["model"], strict=True)
        models.append(model)

        criterion = task.build_criterion(cfg.criterion)
        # if "criterion" in state:
        #     criterion.load_state_dict(state["criterion"], strict=True)
        criterions.append(criterion)
    return models, criterions, task


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


@singleton
class Inference:
    def __init__(self, args):
        self.args = args
        self.sample_rate = 16000
        self.models = None
        self.criterions = None
        self.task = None
        self.init_model()

    def init_model(self):
        logger.info("| loading model(s) from {}".format(self.args.path))
        use_cuda = torch.cuda.is_available() and not self.args.cpu
        models, criterions, task = load_models_and_criterions(
            self.args.path,
            data_path=self.args.data,
            arg_overrides=eval(self.args.model_overrides),  # noqa
            task=None,
            model_state=None,
        )
        logger.info("| loading model(s) from {} finished".format(self.args.path))
        optimize_models(self.args, use_cuda, models)
        self.models = models
        self.criterions = criterions
        self.task = task
        self.generator = W2lViterbiDecoder(self.args, self.task.target_dictionary)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    def get_sample(self, path):
        # if the path is the object of sanic `File`, it will convert the bytes into array of numpy
        # else if path is Pathlike,it will read the path of audio using soundfile library
        if isinstance(path, File):
            wav, curr_sample_rate = AudioReader.read_pcm16(path.body)
        else:
            wav, curr_sample_rate = sf.read(path, dtype="float32")
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return feats

    def get_samples(self, paths):
        sources = [self.get_sample(path) for path in paths]
        sizes = [len(s) for s in sources]
        target_size = max(sizes)
        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False)
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
        input = {"source": collated_sources}
        input["padding_mask"] = padding_mask
        out = dict()
        out["net_input"] = input
        return out

    def infer(self, task=None, model_state=None, path=None):
        if self.args.max_tokens is None and self.args.batch_size is None:
            self.args.max_tokens = 4000000
        # logger.info(args)

        use_cuda = torch.cuda.is_available() and not self.args.cpu

        logger.info(f"use_cuda: {use_cuda}, using {'cpu' if self.args.cpu else 'gpu'} decoding")

        logger.info("| decoding with criterion {}".format(self.args.criterion))

        # Load ensemble
        if self.args.load_emissions:
            models, criterions = [], []
            task = tasks.setup_task(self.args)
        else:
            if self.models is None:
                logger.info("| loading model(s) from {}".format(self.args.path))
                models, criterions, task = load_models_and_criterions(
                    self.args.path,
                    data_path=self.args.data,
                    arg_overrides=eval(self.args.model_overrides),  # noqa
                    task=task,
                    model_state=model_state,
                )
                optimize_models(self.args, use_cuda, models)
                self.models = models
                self.criterions = criterions
                self.task = task

        # Set dictionary
        tgt_dict = self.task.target_dictionary

        # hack to pass transitions to W2lDecoder
        if self.args.criterion == "asg_loss":
            trans = self.criterions[0].asg.trans.data
            self.args.asg_transitions = torch.flatten(trans).tolist()

        # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task


        num_sentences = 0

        if self.args.results_path is not None and not os.path.exists(self.args.results_path):
            os.makedirs(self.args.results_path)

        max_source_pos = (
            utils.resolve_max_positions(
                self.task.max_positions(), *[model.max_positions() for model in self.models]
            ),
        )

        if max_source_pos is not None:
            max_source_pos = max_source_pos[0]
            if max_source_pos is not None:
                max_source_pos = max_source_pos[0] - 1

        sample = self.get_samples([path])
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        prefix_tokens = None
        if self.args.prefix_size > 0:
            prefix_tokens = sample["target"][:, : self.args.prefix_size]

        start = time.time()

        # flashlight decoder
        # with torch.no_grad():
        #     hypos = self.generator.generate(self.models, sample, prefix_tokens=prefix_tokens)

        encoder_input = {
                    k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
                }
        start = time.time()
        emissions = self.generator.get_emissions(self.models, encoder_input)
        end = time.time()
        logging.info(f"mode forward takes {end - start} seconds")
        # hypos = self.generator.decode(emissions)
        tokens = emissions.argmax(dim=-1)
        tokens = torch.masked_select(tokens, tokens != torch.zeros_like(tokens))
        hypos = tokens.unsqueeze(0)
        # hypos = self.task.inference_step(generator, self.models, sample, prefix_tokens)
        hyp_pieces = tgt_dict.string(hypos.int().cpu())
        # hyp_pieces = []
        # for index, hypo in enumerate(hypos):
        #     for item in hypo[: min(len(hypos), self.args.nbest)]:
        #         hyp_pieces = tgt_dict.string(item["tokens"].int().cpu())
        end = time.time()
        logger.info(f"it takes {end - start} seconds")
        logger.info(hyp_pieces)

        return hyp_pieces

    def get_segment(self, path, text):
        use_cuda = torch.cuda.is_available() and not self.args.cpu
        logger.info(f"use_cuda: {use_cuda}, using {'cpu' if self.args.cpu else 'gpu'} decoding")
        logger.info("| decoding with criterion {}".format(self.args.criterion))

        sample = self.get_samples([path])
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        with torch.no_grad():
            emissions = self.generator.get_emissions(self.models, encoder_input)
            softmax = torch.nn.LogSoftmax(dim=-1)
            lpz = softmax(emissions)[0].cpu().numpy()

        index_duration = sample["net_input"]["source"].shape[1] / lpz.shape[0] / 16000
        config = ctc_segmentation.CtcSegmentationParameters(char_list=self.task.target_dictionary.symbols)
        config.index_duration = index_duration
        config.min_window_size = 6400
        # config.score_min_mean_over_L = 10
        # CTC segmentation
        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat)
        segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)

        results = []
        for word, segment in zip(text, segments):
            results.append({
                "start": segment[0],
                "end": segment[1],
                "score": segment[2],
                "text": word
            })
            print(f"{segment[0]:.3f} {segment[1]:.3f} {segment[2]:3.4f} {word}")
        return results


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    check_args(args)
    task = Inference(args)
    # print(task.infer(path='/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0406.wav'))
    print(task.get_segment('/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0406.wav', ['好莱坞当红明星之前曾被盛传将扮演斯诺登']))
    # print(task.infer(path='/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0311.wav'))
    # print(task.infer(path='/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0263.wav'))


