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
    audio_file = request.files.get('audio', None)
    language = request.form.get('language', None)
    if not language:
        raise MissParameters('language is empty')

    if not audio_file:
        raise MissParameters('audio is empty')
    result = recongnitionService.infer(audio_file, language).replace('|', '')
    return json(
        ResponseBody(message=f'Success',
                     status_code=StatusCode.RECOGNITION_FINISHED.name,
                     data=result).__dict__,
        200)


@recognition_route.post('/segment')
async def segment(request):
    audio_file = request.files.get('audio', None)
    text = request.form.get('text', None)
    language = request.form.get('language', None)
    if not language:
        raise MissParameters('language is empty')
    if not audio_file:
        raise MissParameters('audio is missing')
    if not text:
        raise MissParameters('text is missing')

    result = recongnitionService.get_segment(audio_file, text, language)
    return json(
        ResponseBody(message=f'Success',
                     status_code=StatusCode.RECOGNITION_FINISHED.name,
                     data=result).__dict__,
        200)


@recognition_route.get('/hello')
async def recognition(request):
    print(request)
    return json({'message': 'hello'})