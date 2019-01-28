#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'AggFeat'
__author__ = 'JieYuan'
__mtime__ = '19-1-15'
"""
from ... import tqdm
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from .Funcs import Funcs


class AggFeat(object):

    def __init__(self, df, cat_cols, num_cols):
        self.df = df
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.num_funcs = ['min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem',
                          'skew'] + Funcs().num_funcs
        self.cat_funcs = ['nunique', 'max', 'min'] + Funcs().cat_funcs

    def run(self, max_workers=4):
        with ProcessPoolExecutor(min(max_workers, len(self.cat_cols))) as pool:
            for _df in pool.map(self._get_agg_feats, tqdm(self.cat_cols, 'agg ...')):
                self.df = pd.merge(self.df, _df, 'left')
            return self.df

    def _get_agg_feats(self, key_cols):
        if isinstance(key_cols, str):
            key_cols = [key_cols]
        num_feats = self.num_cols
        cat_feats = list(set(self.cat_cols) - set(key_cols))

        gr = self.df.groupby(key_cols)
        trans_dict = dict(zip(num_feats + cat_feats + key_cols,
                              [self.num_cols] * len(num_feats) + [self.cat_cols] * len(cat_feats) + ['count']))
        df = gr.agg(trans_dict)
        df.columns = ['&'.join(key_cols) + '_' + '_'.join(i) for i in df.columns]
        return df.reset_index()
