# -*- coding: utf-8 -*-
# @Time  : 2022/7/14 15:36
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : HubertZhRecognitionService.py
from backend.config.Config import Config
from backend.service.recognitionService.TransformersBase import TransformersBase


class HubertZhRecognitionService(TransformersBase):
    def __init__(self):
        self.enable = True if Config.get_instance().get('hubert-zh-transformers.enable', 'false') \
                              in ['true', 'True', 'TRUE'] else False

        self.cpu = True if Config.get_instance().get('hubert-zh-transformers.cpu', 'true') \
                           in ['true', 'True', 'TRUE'] else False

        self.model = Config.get_instance().get('hubert-zh-transformers.model', 'TencentGameMate/chinese-hubert-large')
        device = 'cpu' if self.cpu else 'cuda'
        super.__init__(self.model, device=device)