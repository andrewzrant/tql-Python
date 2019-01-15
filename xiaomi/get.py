#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'get'
__author__ = 'JieYuan'
__mtime__ = '19-1-15'
"""

import requests
from retry import retry


class Get(object):

    def __init__(self):
        self.url_antiporn = "http://dev.web.du.algo.browser.miui.srv/processarticle"

    @retry(tries=3, delay=2)
    def score_antiporn(self, title):
        _ = requests.get(self.url_antiporn, {'scenario': 'antiporn', 'title': title}).json()
        return _[0]['Prob']
