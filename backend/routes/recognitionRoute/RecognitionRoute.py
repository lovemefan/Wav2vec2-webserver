#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/05/27 0:42
# @Author  : lovemefan
# @File    : RecognitionRoute.py
from sanic import Blueprint
from sanic.response import json

from backend.exception.SpeechException import SpeechSampleRateException, MissParameters
from backend.model.ResponseBody import ResponseBody
from backend.service.recognitionService.RecognitionService import RecognitionService
from backend.utils.StatusCode import StatusCode
from backend.utils.logger import logger

recognition_route = Blueprint('speech', url_prefix='/api/speech', version=1)
recongnitionService = RecognitionService()

@recognition_route.exception(SpeechSampleRateException)
async def speech_sample_rate_exception(request, exception):
    response = {
        "reasons": [str(exception)],
        "exception": StatusCode.SAMPLE_RATE_ERROR.name
    }
    return json(response, 408)


@recognition_route.post('/recognition')
async def recognition(request):
    """add  a user  ,need administrator identify"""
    audio_file = request.files.get('audio', None)
    if not audio_file:
        raise MissParameters('audio is empty')
    result = recongnitionService.infer(audio_file).replace('|', '').replace(' ', '')
    return json(
        ResponseBody(message=f'Success',
                     status_code=StatusCode.RECOGNITION_FINISHED.name,
                     data=result).__dict__,
        200)


@recognition_route.get('/hello')
async def recognition(request):
    print(request)
    return json({'message': 'hello'})