#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'model_save'
__author__ = 'JieYuan'
__mtime__ = '19-1-31'
"""
from datetime import datetime
from sklearn.externals import joblib


def model_save(model, path='.'):
    prefix = datetime.today().strftime("%Y%m%d %H:%M:%S")
    model_name = model.__str__().split('(')[0]
    path = f'{path}/{prefix} | {model_name}'
    print("Model saved in", path)
    joblib.dump(model, path)
