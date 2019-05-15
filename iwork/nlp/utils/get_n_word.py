#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
__title__ = 'get_n_word'
__author__ = 'JieYuan'
__mtime__ = '2019-05-15'
"""
import jieba.posseg as jp

def get_n_word(sent):
    for p in jp.cut(sent):
        if 'n' in p.flag and len(p.word) > 1:
            yield p.word