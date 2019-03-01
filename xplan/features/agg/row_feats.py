#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'row_feats'
__author__ = 'JieYuan'
__mtime__ = '19-2-25'
"""
from tqdm import tqdm


class RowFeats(object):

    def __init__(self):
        self.funcs = ['min', 'mean', 'median', 'max', 'sum', 'std', 'sem', 'skew', 'kurt']

    def get_row_feats(self, df):
        df = df.copy()

        for func in tqdm(self.funcs):
            df[func] = df.apply(func, 1)
        df['p1'] = df.quantile(0.25, 1)
        df['p3'] = df.quantile(0.75, 1)
        df['iqr'] = df['p3'] - df['p1']
        df['ptp'] = df['max'] - df['min']
        df['cv'] = df['std'] / (df['mean'] + 10 ** -8)
        return df