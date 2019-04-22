#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '19-3-22'
"""
from .BaiduPost import BaiduPost
from gensim.models import fasttext
model = fasttext.load_facebook_model('/home/yuanjie/desktop/南京小米算法共享/WordVectors/comment.skipgram')


