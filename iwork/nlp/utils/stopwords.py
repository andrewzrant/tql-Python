#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'StopWords'
__author__ = 'JieYuan'
__mtime__ = '2019-05-16'
"""

with open('./stop_words.txt') as f:
    words = set(f.read().split())