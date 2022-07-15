# -*- coding: utf-8 -*-
# @Time  : 2022/7/9 20:55
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : cut_files.py
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor


def get_second_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    """
    音频切片，获取部分音频，单位秒
    :param main_wav_path: 原音频文件路径
    :param start_time: 截取的开始时间
    :param end_time: 截取的结束时间
    :param part_wav_path: 截取后的音频路径
    :return:
    """
    start_time = start_time * 1000
    end_time = end_time * 1000

    sound = AudioSegment.from_wav(main_wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")

def split_into_files(audio_files):



def multi_process_magicData_corpus():
    with ProcessPoolExecutor(max_workers=10) as executor:
        pass