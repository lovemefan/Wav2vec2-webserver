#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 下午2:18
# @Author  : lovemefan
# @File    : UserService.py
import re
import regex
from backend.config.Config import Config
from backend.decorator.singleton import singleton
from backend.service.recognitionService.infer import make_parser, check_args, Inference
from fairseq import options


def split_text(text: str):
    """
    break down the text roughly into sentences. each sentence will be a separate item of list.
    :param text: input text
    :return: list of sentence
    """
    # remove extra space
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'(\.+)', '. ', text)
    # remove (*)
    text = re.sub(r'\(.*?\)', '', text)
    zh_unicode = '\u4E00-\u9FA5'

    # find phrases in quotes
    with_quotes = re.finditer(r'“[A-Za-z ?]+.*?”', text)
    sentences = []
    last_idx = 0
    for m in with_quotes:
        match = m.group()
        match_idx = m.start()
        if last_idx < match_idx:
            sentences.append(text[last_idx:match_idx])
        sentences.append(match)
        last_idx = m.end()
    sentences.append(text[last_idx:])
    sentences = [s.strip() for s in sentences if s.strip()]

    split_pattern = f"(?<!\w\.\w.)(?<![A-Z][A-Z{zh_unicode}][a-z][a-z{zh_unicode}]\.)(?<=\.|\。|\?|\？|\!|\！|\n|\.”|\?”\!”)"
    new_sentences = []
    for sent in sentences:
        new_sentences.extend(regex.split(split_pattern, sent))
    sentences = [s.strip() for s in new_sentences if s.strip()]
    return sentences

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

    ins1 = RecognitionService()
    ins1.infer(path='/C2_519.wav')

