# -*- coding: utf-8 -*-
# @Time  : 2022/7/14 22:02
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : RecognitionService.py
from backend.config.Config import Config
from backend.decorator.singleton import singleton
from backend.exception.SpeechException import MissParameters
from backend.service.recognitionService.ours.zh.Wav2vec2ZhRecognitionService import Wav2vec2ZhRecognitionService
from backend.service.recognitionService.w2v2.vi.W2v2ViRecognitionService import W2v2ViRecognitionService
from backend.utils.common import split_text


class RecognitionService:
    def __init__(self):
        self.language_list = Config.get_instance().get('common.language-list', '').split(';')
        self.vi_asr = W2v2ViRecognitionService()
        self.zh_asr = Wav2vec2ZhRecognitionService()

    def infer(self, speech, language):
        if language not in self.language_list:
            raise MissParameters(f'language `{language}` is not in {",".join(self.language_list)}')

        if language == 'zh':
            return self.zh_asr.infer(speech)
        elif language == 'vi':
            return self.vi_asr.infer(speech)
        else:
            raise ValueError(f'language is {language}')

    def get_segment(self, speech, text, language):
        if language not in self.language_list:
            raise MissParameters(f'language `{language}` is not in {",".join(self.language_list)}')
        text = split_text(text.lower())
        if language == 'zh':
            return self.zh_asr.get_segment(speech, text)
        elif language == 'vi':
            return self.vi_asr.get_segment(speech, text)
        else:
            raise ValueError(f'language is {language}')