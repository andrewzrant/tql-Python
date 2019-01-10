#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '18-12-17'
"""
import os
import numpy as np
import wrapt
import resource

from .download_file import DownloadFile

base_dir = os.path.dirname(os.path.realpath('__file__'))
get_module_path = lambda path, file: os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(file), path))


def get_weight(y):
    class_weight = dict(enumerate(len(y) / (2 * np.bincount(y))))
    sample_weight = [class_weight[i] for i in y]
    return class_weight, sample_weight


def feval(multiclass=None, is_bigger_better=True, model='lgb'):
    """example
    @feval(3)
    def f1_score(y_pred, y_true):
        '注意入参顺序'
        return f1_score(y_true, y_pred, average='macro')
    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        y_pred, y_true = args
        y_true = y_true.get_label()
        if model == 'lgb':
            if multiclass:
                y_pred = np.array(y_pred).reshape(multiclass, -1).argmax(0)
            return wrapped.__name__, wrapped(y_pred, y_true), is_bigger_better
        elif is_bigger_better:
            """xgb评估指标默认越小越好"""
            return '-' + wrapped.__name__, - wrapped(y_pred, y_true)
        else:
            return wrapped.__name__, wrapped(y_pred, y_true)

    return wrapper


def limit_memory(res_mem="120G"):
    rsrc = resource.RLIMIT_AS
    # res_mem=os.environ["RESOURCE_MEM"]
    memlimit = float(res_mem[:-1]) * 1024 ** 3
    resource.setrlimit(rsrc, (memlimit, memlimit))
    soft, hard = resource.getrlimit(rsrc)
    print("memory limit as:", res_mem, soft, hard)
