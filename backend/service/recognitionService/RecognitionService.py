#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 下午2:18
# @Author  : lovemefan
# @File    : UserService.py
from backend.config.Config import Config
from backend.decorator.singleton import singleton
from backend.service.recognitionService.infer import make_parser, check_args, Inference
from fairseq import options

# @singleton
class RecognitionService:
    """user @singleton to avoid create amount of same instance, improve the efficiency
    """

    def update_config(self, args):
        """
        update configure form config.ini
        :param args:
        :return:
        """
        args.cpu = True if Config.get_instance().get('wav2vec2.cpu', 'true') == 'true' else False
        args.task = Config.get_instance().get('wav2vec2.task', 'audio_finetuning')
        args.path = Config.get_instance().get('wav2vec2.path', '/root/model/checkpoint_best.pt')
        args.nbest = int(Config.get_instance().get('wav2vec2.nbest', 1))
        args.w2l_decoder = Config.get_instance().get('wav2vec2.w2l-decoder', 'viterbi')
        args.data = Config.get_instance().get('wav2vec2.data', '/root/model')
        args.lm_weight = int(Config.get_instance().get('wav2vec2.lm-weight', 2))
        args.word_score = int(Config.get_instance().get('wav2vec2.word-score', -1))
        args.sil_weight = int(Config.get_instance().get('wav2vec2.sil-weight', 0))
        args.criterion = Config.get_instance().get('wav2vec2.criterion', 'ctc')
        args.labels = Config.get_instance().get('wav2vec2.labels', 'ltr')
        args.max_tokens = int(Config.get_instance().get('wav2vec2.max-tokens', 1000000))
        args.post_process = Config.get_instance().get('wav2vec2.post-process', 'letter')

    def __init__(self):
        parser = make_parser()
        args = parser.parse_args()
        self.update_config(args)
        check_args(args)
        self.inference = Inference(args)

    def infer(self, path: str):
        """
        :param path:
        :return:
        """
        return self.inference.infer(path=path)




if __name__ == '__main__':

    ins1 = RecognitionService()
    ins1.infer(path='/C2_519.wav')

