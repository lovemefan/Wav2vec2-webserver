# -*- coding: utf-8 -*-
# @Time  : 2022/7/14 15:34
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : W2v2ZhRecognitionService.py
from backend.config.Config import Config
from backend.service.recognitionService.TransformersBase import TransformersBase
import soundfile as sf
import torch


class W2v2ZhRecognitionService(TransformersBase):
    def __init__(self):
        self.enable = True if Config.get_instance().get('wav2vec-zh-transformers.enable', 'false') \
                              in ['true', 'True', 'TRUE'] else False

        self.cpu = True if Config.get_instance().get('wav2vec-zh-transformers.cpu', 'true') \
                           in ['true', 'True', 'TRUE'] else False

        self.model = Config.get_instance().get('wav2vec-zh-transformers.model',
                                               'TencentGameMate/chinese-wav2vec2-large')
        device = 'cpu' if self.cpu else 'cuda'
        super().__init__(self.model, device)

    def infer(self, speech):
        if isinstance(speech, str):
            speech, _ = sf.read(speech)

        # retrieve logits
        logits = self.get_logits(speech)
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription

    def get_logits(self, speech):
        if isinstance(speech, str):
            speech, _ = sf.read(speech)

        input_values = self.feature_extractor(speech, return_tensors="pt").input_values
        input_values = input_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values)

            logits = outputs.logits
        return logits


if __name__ == '__main__':
    zh = W2v2ZhRecognitionService()
    text = zh.get_logits('/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0453.wav')
    print(text)
