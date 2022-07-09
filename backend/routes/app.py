#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 下午3:28
# @Author  : lovemefan
# @File    : app.py
import os
from time import sleep

import sys
sys.path.append('/wav2vec2-webserver')
from sanic import Sanic
from sanic.exceptions import RequestTimeout, NotFound
from sanic.response import json
from sanic_openapi import swagger_blueprint

from backend.exception.SpeechException import MissParameters
from backend.model.ResponseBody import ResponseBody
from backend.routes.recognitionRoute.RecognitionRoute import recognition_route

from backend.utils.StatusCode import StatusCode


from backend.config.Config import Config

app = Sanic(__name__)
app.blueprint(swagger_blueprint)


@app.exception(RequestTimeout)
async def timeout(request, exception):
    response = {
        "reasons": ['Request Timeout'],
        "exception": StatusCode.REQUEST_TIMEOUT.name
    }
    return json(response, 408)


@app.exception(NotFound)
async def notfound(request, exception):
    response = {
        "reasons": [f'Requested URL {request.url} not found'],
        "exception": StatusCode.NOT_FOUND.name
    }

    return json(response, 404)


@app.exception(MissParameters)
async def notfound(request, exception):
    response = {
        "reasons": [str(exception)],
        "exception": StatusCode.MISSPARAMETERS.name
    }

    return json(response, 404)

def load_banner():
    """load the banner"""
    with open('./banner.txt', 'r', encoding='utf-8') as f:
        banner = f.read()

    print(banner)


app.blueprint(recognition_route)
if __name__ == '__main__':
    load_banner()
    port = int(Config.get_instance().get('http.port', 80))
    # if env $port is none ,get the config port or default port
    port = os.environ.get('PORT', port)
    app.run(host="0.0.0.0", port=port)

