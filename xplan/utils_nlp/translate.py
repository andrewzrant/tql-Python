#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'translate'
__author__ = 'JieYuan'
__mtime__ = '19-3-1'
"""
import requests

"""
https://www.cnblogs.com/fanyang1/p/9414088.html
"""


class Translate(object):

    def __init__(self):
        pass

    def get_zh(self, text, api='google'):
        params = {'q': text}
        if api == 'google':
            url = "http://translate.google.cn/translate_a/single?client=gtx&dt=t&dj=1&ie=UTF-8&sl=auto&tl=zh"
            requests.get(url, params)

    def get_en(self):
        pass
