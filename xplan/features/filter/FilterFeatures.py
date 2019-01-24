#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'FilterFeatures'
__author__ = 'JieYuan'
__mtime__ = '19-1-24'
"""
from ... import tqdm
from ...utils.timer import timer

import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor


class FilterFeatures(object):
    """粗筛
    高缺失率：
    低方差（高度重复值）：0.5%~99.5%分位数内方差为0的初筛
    高相关：特别高的初筛，根据重要性细筛
    低重要性：
    召回高IV：
    """

    def __init__(self, df: pd.DataFrame, exclude=None):
        self.df = df
        if exclude:
            self.feats = df.columns.difference(exclude)
        else:
            self.feats = df.columns.tolist()

    def run(self):
        df = self.df.copy()

        with timer('干掉高缺失'):
            to_drop = self.filter_missing()
            if to_drop:
                df.drop(to_drop, 1, inplace=True)

        with timer('干掉低方差'):
            to_drop = self.filter_variance()
            if to_drop:
                df.drop(to_drop, 1, inplace=True)

        with timer('干掉高相关'):
            pass

        return df

    def filter_missing(self, feats=None, threshold=0.95):
        """
        :param feat_cols:
        :param threshold:
        :param as_na: 比如把-99当成缺失值
        :return:
        """
        if feats is None:
            feats = self.feats

        to_drop = (self.df[feats].isna().sum() / len(self.df))[lambda x: x > threshold].index.tolist()
        print('%d features with greater than %0.2f missing values.' % (len(to_drop), threshold))
        return to_drop

    def _filter_variance(self, feat, df):
        var = df[feat][lambda x: x.between(x.quantile(0.005), x.quantile(0.995))].var()
        return '' if var else feat

    def filter_variance(self, feats=None):
        if feats is None:
            feats = self.feats

        _filter_variance = partial(self._filter_variance, df=self.df)
        with ProcessPoolExecutor(min(5, len(feats))) as pool:
            to_drop = pool.map(_filter_variance, tqdm(feats, 'Filter Variance ...'))
            to_drop = [feat for feat in to_drop if feat]
        print('%d features with 0 variance in 0.5 ~ 99.5 quantile.' % len(to_drop))
        return to_drop

    # def filter_correlation(self, feat_cols=None, threshold=0.98):
    #     if feat_cols is None:
    #         feat_cols = self.feats
    #
    #     print('Compute Corr Matrix ...')
    #     corr_matrix = self.df[feat_cols].corr().abs()
    #
    #     # Extract the upper triangle of the correlation matrix
    #     upper = pd.DataFrame(np.triu(corr_matrix, 1), feat_cols, feat_cols)
    #
    #     # Select the features with correlations above the threshold
    #     # Need to use the absolute value
    #     to_drop = [column for column in tqdm(upper.columns, 'Correlation Filter') if any(upper[column] > threshold)]
    #
    #     self.to_drop_correlation = to_drop
    #
    #     # Dataframe to hold correlated pairs
    #     # Iterate through the columns to drop to record pairs of correlated features
    #     corr_record = pd.DataFrame()
    #     for column in tqdm(to_drop, 'Correlation DataFrame'):
    #         cond = upper[column] > threshold
    #         corr_features = list(upper.index[cond])  # Find the correlated features
    #         corr_values = list(upper[column][cond])  # Find the correlated values
    #         drop_features = [column for _ in range(len(corr_features))]
    #         df_tmp = pd.DataFrame({'drop_feature': drop_features,
    #                                'corr_feature': corr_features,
    #                                'corr_value': corr_values})
    #         corr_record = corr_record.append(df_tmp, ignore_index=True, sort=False)
    #
    #     self.corr_record = corr_record
    #     print('%d features with a  correlation coefficient greater than %0.2f.' % (len(to_drop), threshold))
    #
    # @property
    # def to_drop_all(self):
    #     return self.to_drop_missing + self.to_drop_unique + self.to_drop_variance + self.to_drop_correlation + self.to_drop_zero_importance + self.to_drop_low_importance
