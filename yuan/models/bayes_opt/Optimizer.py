#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Optimizer'
__author__ = 'JieYuan'
__mtime__ = '19-3-18'
"""
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn import clone
import numpy as np


class Optimizer(object):

    def __init__(self, estimator, pbounds, X, y):
        self.estimator = estimator
        self.pbounds = pbounds  # 参数边界 (type, (1, 100))

        self.X = X
        self.y = y

        """缩小区间长度（边界）
         Notice how we transform between regular and log scale. While this
         is not technically necessary, it greatly improves the performance
         of the optimizer.
        """
        self.scaled_pbounds, self.scaled_params = self._scaler(self.pbounds)
        print("Scaled pbounds: %s" % self.scaled_pbounds)

    def maximize(self, n_iter=5, opt_seed=2019):
        self.optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.scaled_pbounds,
            random_state=opt_seed,
            verbose=2
        )

        # gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3}
        self.optimizer.maximize(n_iter=n_iter)  # self.optimizer.maximize() 可以接着上一次优化继续下一轮优化
        print("Final result:", self.optimizer.max)

    def objective(self, **params):
        """cv_score: 核心函数
        This function will instantiate a SVC classifier with parameters C and
        gamma. Combined with data and targets this will in turn be used to perform
        cross validation. The result of cross validation is returned.
        Our goal is to find combinations of C and gamma that maximizes the roc_auc
        metric.
        """
        estimator = clone(self.estimator)

        for p in self.scaled_params:  # TODO: 支持多种参数类型，默认float
            params[p] = 10 ** params[p]
        estimator.set_params(**params)

        cv_score = cross_val_score(estimator, self.X, self.y, scoring='roc_auc', cv=5).mean()
        return cv_score

    def _scaler(self, pbounds):
        scaled_params = []
        for k, v in pbounds.items():
            if not -1 < np.log10(np.ptp(v)) < 1:  # update pbounds
                scaled_params.append(k)
                pbounds[k] = np.log10(v[0] if v[0] else 1e-6), np.log10(v[1])

        return pbounds, scaled_params
