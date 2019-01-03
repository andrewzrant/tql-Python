#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'opt'
__author__ = 'JieYuan'
__mtime__ = '19-1-3'
"""
import warnings

warnings.filterwarnings('ignore')
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


class BayesOptLGB(object):
    def __init__(self, X, y):
        self.data = lgb.Dataset(X, y)

    def run(self, n_iter=10, save_log=False):
        logger = JSONLogger(path="./opt_lgb_logs.json")

        BoParams = {
            'learning_rate': (0.003, 0.05),
            'max_depth': (6, 12),
            'min_split_gain': (0, 1),
            'min_child_weight': (0, 10),
            'subsample': (0.6, 1),
            'colsample_bytree': (0.6, 1),
            'reg_alpha': (0, 100),
            'reg_lambda': (0, 100),
        }
        optimizer = BayesianOptimization(self.__evaluator, BoParams)
        if save_log:
            optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        optimizer.maximize(init_points=3, n_iter=n_iter, acq='ucb', kappa=2.576, xi=0.0, **gp_params)
        return optimizer.max

    def __evaluator(self, learning_rate, max_depth, min_split_gain, min_child_weight, subsample, colsample_bytree,
                    reg_alpha, reg_lambda):
        params = dict(
            boosting_type='gbdt',
            objective='binary',
            max_depth=-1,
            num_leaves=2 ** int(max_depth) - 1,
            learning_rate=learning_rate,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            subsample=subsample,
            subsample_freq=8,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=1,
            random_state=None,
            n_jobs=8
        )

        metric = 'auc'
        cv_rst = lgb.cv(params, self.data, num_boost_round=3000, nfold=5, metrics=metric, early_stopping_rounds=100,
                        verbose_eval=None, show_stdv=False)

        _ = cv_rst['%s-mean' % metric]
        print('\nBest Iter: %s' % len(_))
        return _[-1]
