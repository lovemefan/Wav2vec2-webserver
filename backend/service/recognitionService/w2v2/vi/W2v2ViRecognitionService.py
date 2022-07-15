# -*- coding: utf-8 -*-
# @Time  : 2022/7/14 14:45
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : W2v2ViRecognitionService.py
import torch
import ctc_segmentation
from sanic.request import File
import soundfile as sf
from backend.config.Config import Config
from backend.decorator.singleton import singleton
from backend.service.recognitionService.TransformersBase import TransformersBase
from backend.utils.AudioReader import AudioReader
from backend.utils.logger import logger
from fairseq_lib.fairseq import utils


class W2v2ViRecognitionService(TransformersBase):
    def __init__(self):
        self.enable = True if Config.get_instance().get('wav2vec-vi-transformers.enable', 'false') \
                              in ['true', 'True', 'TRUE'] else False

        self.cpu = True if Config.get_instance().get('wav2vec-vi-transformers.cpu', 'true') \
                           in ['true', 'True', 'TRUE'] else False

        self.model = Config.get_instance().get('wav2vec-vi-transformers.model', 'nguyenvulebinh/wav2vec2-base-vietnamese-250h')
        self.device = 'cpu' if self.cpu else 'cuda'
        super().__init__(self.model, self.device)

    def get_sample(self, path):
        # if the path is the object of sanic `File`, it will convert the bytes into array of numpy
        # else if path is Pathlike,it will read the path of audio using soundfile library
        if isinstance(path, File):
            wav, curr_sample_rate = AudioReader.read_pcm16(path.body)
        else:
            wav, curr_sample_rate = sf.read(path, dtype="float32")
        feats = torch.from_numpy(wav).float()
        return feats

    def get_segment(self, path, text):
        use_cuda = torch.cuda.is_available() and not self.device
        logger.info(f"use_cuda: {use_cuda}, using {self.device} decoding")

        sample = self.get_sample(path)

        with torch.no_grad():
            emissions = self.get_logits(sample)
            softmax = torch.nn.LogSoftmax(dim=-1)
            lpz = softmax(emissions)[0].cpu().numpy()

        index_duration = sample.shape[0] / lpz.shape[0] / 16000
        config = ctc_segmentation.CtcSegmentationParameters(char_list=list(self.processor.tokenizer.encoder.keys()))
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

if __name__ == '__main__':
    vi = W2v2ViRecognitionService()
    text = vi.infer('/dataset/speech/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R122.wav')
    print(text)
    text = vi.get_segment('/dataset/speech/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R122.wav', ['CŨNG KHIẾN CHO HỌ DÈ DẶT'.lower()])
    print(text)
