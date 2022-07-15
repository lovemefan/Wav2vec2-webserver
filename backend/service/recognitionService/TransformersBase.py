# -*- coding: utf-8 -*-
# @Time  : 2022/7/14 15:15
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : TransformersBase.py
from sanic.request import File
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from datasets import load_dataset
import soundfile as sf
import torch

from backend.utils.AudioReader import AudioReader
from backend.utils.logger import logger


class TransformersBase:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        logger.info(f'loading {model} model')
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model)
        except OSError:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to(device)
        self.model.eval()

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
        if isinstance(speech, File):
            wav, curr_sample_rate = AudioReader.read_pcm16(speech.body)

        # tokenize
        input_values = self.processor(speech, return_tensors="pt", padding="longest").input_values.to(self.device)
        # retrieve logits
        with torch.no_grad():
            logits = self.model(input_values).logits
        return logits
