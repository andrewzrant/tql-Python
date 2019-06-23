#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : BaseOOF
# @Time         : 2019-06-23 20:07
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :
from abc import abstractmethod

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold

# ToDo
"""
1. 基模型: 
2. 数据n折拆分: 根据y: 判断分类、回归 train + test
3. train:交叉验证结果存储
4. test: 指标计算、存储
5. report
"""
from lightgbm.sklearn import LGBMClassifier


class BaseOOF(object):

    def __init__(self, estimator=LGBMClassifier(), cv=5, random_state=None, n_repeats=None):
        self.estimator = estimator
        if n_repeats:
            self._kf = RepeatedStratifiedKFold(cv, True, random_state)
            self._num_preds = cv * n_repeats
        else:
            self._kf = StratifiedKFold(cv, True, random_state)
            self._num_preds = cv

    @abstractmethod
    def _fit(self, eval_set):
        raise NotImplementedError

    def fit(self, X, y, X_test, feval=None):
        self.y_true = y

        # Cross validation model
        # Create arrays and dataframes to store results
        self.oof_train = np.zeros(len(X))
        self.oof_test = np.zeros((len(X_test), self._num_preds))
        for n_fold, (train_index, valid_index) in enumerate(self._kf.split(X, y)):
            print("\n\033[94mFold %s started at %s\033[0m" % (n_fold, time.ctime()))
            X_train, y_train = X[train_index], y[train_index]
            X_valid, y_valid = X[valid_index], y[valid_index]
            eval_set = [(X_train, y_train), (X_valid, y_valid)]

            # 重写fit
            self.estimator = self._fit(eval_set)

            # TODO: 多分类需要修改
            if hasattr(self.estimator, 'predict_proba'):
                self.oof_train[valid_index] = self.estimator.predict_proba(X_valid)[:, 1]
                self.oof_test[:, n_fold] = self.estimator.predict_proba(X_test)[:, 1]
            else:
                self.oof_train[valid_index] = self.estimator.predict(X_valid)
                self.oof_test[:, n_fold] = self.estimator.predict(X_test)

        # 输出需要的结果
        self.oof_test_rank = pd.DataFrame(self.oof_test).rank().mean(1) / len(self.oof_test)
        self.oof_test = self.oof_test.mean(1)
        if feval:
            if hasattr(feval, '__repr__'):
                metric_name = feval.__repr__().split()[1].title()
            else:
                metric_name = "Score"
            score = feval(y, self.oof_test)
            print("\n\033[94mCV %s: %s ended at %s\033[0m" % (metric_name, score, time.ctime()))

    def oof_save(self, file=None):
        if file is None:
            file = self.estimator.__str__().split('(')[0][:32]
            file = '%s👍%s.csv' % (file, time.ctime())
        assert isinstance(file, str)
        pd.DataFrame(np.append(self.oof_train, self.oof_test), columns='oof: train+test') \
            .to_csv(file, index=False)
