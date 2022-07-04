#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 下午3:35
# @Author  : lovemefan
# @File    : UserException.py
from sanic.exceptions import SanicException


class MissParameters(SanicException):
    pass


class UserNotExist(Exception):
    def __init__(self, username):
        self.username = username

    def __str__(self):
        return f"User '{self.username}' is not exist"


class SpeechSampleRateException(SanicException):
    pass

