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
    """
    opt_lgb = BayesOptLGB(X, y, categorical_feature=cats)
    opt_lgb.run(3)
    opt_lgb.get_best_model()
    """

    def __init__(self, X, y, categorical_feature='auto'):
        self.data = lgb.Dataset(X, y, categorical_feature=categorical_feature, free_raw_data=False)

    def run(self, n_iter=10, save_log=False):
        logger = JSONLogger(path="./opt_lgb_logs.json")

        BoParams = {
            'num_leaves': (2 ** 5, 2 ** 16),
            'min_split_gain': (0.01, 1),
            'min_child_weight': (0, 0.01),  # 0.001可以考虑不调?
            'min_child_samples': (8, 32),
            'subsample': (0.6, 1),
            'colsample_bytree': (0.6, 1),
            'reg_alpha': (0, 1),
            'reg_lambda': (0, 1),
        }
        optimizer = BayesianOptimization(self.__evaluator, BoParams)
        if save_log:
            optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        optimizer.maximize(init_points=3, n_iter=n_iter, acq='ucb', kappa=2.576, xi=0.0, **gp_params)
        self.best_params = optimizer.max
        self.__get_params()

    def get_best_model(self, best_iter):
        return lgb.train(self.params, self.data, best_iter)

    def __evaluator(self, num_leaves, min_split_gain, min_child_weight, min_child_samples, subsample, colsample_bytree,
                    reg_alpha, reg_lambda):
        params = dict(
            boosting_type='gbdt',
            objective='binary',
            max_depth=-1,
            num_leaves=int(num_leaves),
            learning_rate=0.01,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=int(min_child_samples),
            subsample=subsample,
            subsample_freq=8,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=1,
            random_state=None,
            n_jobs=-1
        )

        metric = 'auc'
        cv_rst = lgb.cv(params, self.data, num_boost_round=3000, nfold=5, metrics=metric, early_stopping_rounds=100,
                        verbose_eval=None, show_stdv=False)

        _ = cv_rst['%s-mean' % metric]
        print('\nBest Iter: %s' % len(_))
        return _[-1]

    def __get_params(self):
        params = {'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'max_depth': -1,
                  'num_leaves': 127,
                  'learning_rate': 0.01,
                  'min_split_gain': 0.0,
                  'min_child_weight': 0.001,
                  'min_child_samples': 20,
                  'subsample': 0.8,
                  'subsample_freq': 8,
                  'colsample_bytree': 0.8,
                  'reg_alpha': 0.0,
                  'reg_lambda': 0.0,
                  'scale_pos_weight': 1,
                  'random_state': None,
                  'n_jobs': -1}
        params.update(self.best_params['params'])
        params['num_leaves'] = int(params['num_leaves'])
        params['min_child_samples'] = int(params['min_child_samples'])
        self.params = {k: float('%.3f' % v) if isinstance(v, float) else v for k, v in params.items()}
        self.params_sk = self.params.copy()
