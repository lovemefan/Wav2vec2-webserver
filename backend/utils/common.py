# -*- coding: utf-8 -*-
# @Time  : 2022/7/14 15:01
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : util.py
import re

from regex import regex


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

if __name__ == '__main__':
    print(split_text("Theo tin Đài chúng tôi: Ngày 13/7,"))
    print(split_text("你好。"))