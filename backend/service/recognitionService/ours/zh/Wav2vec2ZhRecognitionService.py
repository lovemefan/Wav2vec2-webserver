#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 下午2:18
# @Author  : lovemefan
# @File    : UserService.py
import re
import regex
from backend.config.Config import Config
from backend.decorator.singleton import singleton
from backend.service.recognitionService.ours.zh.infer import make_parser, check_args, Inference
from backend.utils.common import split_text


class Wav2vec2ZhRecognitionService:
    """user @singleton to avoid create amount of same instance, improve the efficiency
    """

    def update_config(self, args):
        """
        update configure form config.ini
        :param args:
        :return:
        """
        args.cpu = True if Config.get_instance().get('wav2vec2-zh-fairseq.cpu', 'true') == 'true' else False
        args.task = Config.get_instance().get('wav2vec2-zh-fairseq.task', 'audio_finetuning')
        args.path = Config.get_instance().get('wav2vec2-zh-fairseq.path', '/root/model/checkpoint_best.pt')
        args.nbest = int(Config.get_instance().get('wav2vec2-zh-fairseq.nbest', 1))
        args.w2l_decoder = Config.get_instance().get('wav2vec2-zh-fairseq.w2l-decoder', 'viterbi')
        args.data = Config.get_instance().get('wav2vec2-zh-fairseq.data', '/root/model')
        args.lm_weight = int(Config.get_instance().get('wav2vec2-zh-fairseq.lm-weight', 2))
        args.word_score = int(Config.get_instance().get('wav2vec2-zh-fairseq.word-score', -1))
        args.sil_weight = int(Config.get_instance().get('wav2vec2-zh-fairseq.sil-weight', 0))
        args.criterion = Config.get_instance().get('wav2vec2-zh-fairseq.criterion', 'ctc')
        args.labels = Config.get_instance().get('wav2vec2-zh-fairseq.labels', 'ltr')
        args.max_tokens = int(Config.get_instance().get('wav2vec2-zh-fairseq.max-tokens', 1000000))
        args.post_process = Config.get_instance().get('wav2vec2-zh-fairseq.post-process', 'letter')

    def __init__(self):
        parser = make_parser()
        args = parser.parse_args()
        self.update_config(args)
        check_args(args)
        self.inference = Inference(args)

    def infer(self, path: str):
        """
        get transcription of audio
        :param path:
        :return:
        """
        return self.inference.infer(path=path)

    def get_segment(self, path: str, text: str):
        """
        get segment information of audio with text
        :param path:
        :return:
        """
        return self.inference.get_segment(path=path, text=split_text(text))


if __name__ == '__main__':

    ins1 = Wav2vec2ZhRecognitionService()
    ins1.get_segment('/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0406.wav', '好莱坞当红明星之前曾被盛传将扮演斯诺登')


