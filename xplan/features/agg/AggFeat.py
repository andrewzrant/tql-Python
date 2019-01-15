#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'AggFeat'
__author__ = 'JieYuan'
__mtime__ = '19-1-15'
"""
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from .funcs import Funcs


class AggFeat(object):

    def __init__(self, df, cat_cols, num_cols):
        self.df = df
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.num_funcs = ['count', 'min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem',
                          'skew'] + Funcs().num_funcs
        self.cat_funcs = ['count', 'nunique', 'max', 'min'] + Funcs().cat_funcs

    def df_agg(self):
        with ProcessPoolExecutor(5) as pool:
            for _df in pool.map(self._agg, self.cat_cols):
                self.df = pd.merge(self.df, _df)
            return self.df

    def _agg(self, cat_cols):
        if isinstance(cat_cols, str):
            cat_cols = [cat_cols]
        agg_num = self.num_cols
        agg_cat = list(set(self.cat_cols) - set(cat_cols))
        gr = self.df.groupby(cat_cols)
        df = pd.concat((gr[agg_num].agg(self.num_funcs),
                        gr[agg_cat].agg(self.cat_funcs)), 1)
        prefix = '&'.join(cat_cols) + '_'
        df.columns = [prefix + '_'.join(i) for i in df.columns]
        return df.reset_index()
