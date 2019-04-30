#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
__title__ = 'examples'
__author__ = 'JieYuan'
__mtime__ = '18-12-28'
"""

from yuan.models import OOF

from sklearn.datasets import make_classification
import lightgbm as lgb

X, y = make_classification(10000)

data = lgb.Dataset(X, y)
lgb.cv({}, data)