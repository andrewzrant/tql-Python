#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'get_label'
__author__ = 'JieYuan'
__mtime__ = '19-1-28'
"""
import numpy as np
import pandas as pd

def get_label(pred, n_pos):
    _ = (pd.DataFrame(pred, columns=['pred'])
         .assign(pred_rank=lambda df: df.pred.rank(method='first'))
         .assign(label=lambda df: df.pred_rank.apply(lambda x: np.where(x > n_pos, 1, 0))))
    print('Threshold: %f' % _[lambda x: x.pred_rank == n_pos].pred[0])
    return _
