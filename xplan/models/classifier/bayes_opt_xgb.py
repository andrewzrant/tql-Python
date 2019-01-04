#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'bayes_xgb'
__author__ = 'JieYuan'
__mtime__ = '19-1-3'
"""
import warnings

warnings.filterwarnings('ignore')
import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


class BayesOptXGB(object):
    """
    opt_xgb = BayesOptXGB(X, y)
    opt_xgb.run(3)
    opt_xgb.get_best_model()
    """

    def __init__(self, X, y, missing=None):
        self.data = xgb.DMatrix(X, y, missing=missing)

    def run(self, n_iter=10, save_log=False):
        logger = JSONLogger(path="./opt_xgb_logs.json")

        BoParams = {
            'max_depth': (5, 16),
            'min_child_weight': (1, 10),
            'gamma': (0.01, 1),
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

    def get_best_model(self, best_iter):
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'max_depth': 7,
                  'gamma': 0.0,
                  'min_child_weight': 1,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'colsample_bylevel': 0.8,
                  'scale_pos_weight': 1,
                  'random_state': None,
                  'n_jobs': 8,
                  'silent': True,
                  'eta': 0.01,
                  'alpha': 0.0,
                  'lambda': 0.0}
        _params = self.best_params['params']
        _params['alpha'] = _params.pop('reg_alpha')
        _params['lambda'] = _params.pop('reg_lambda')

        params.update(self.best_params['params'])
        params['max_depth'] = int(params['max_depth'])
        params = {k: float('%.3f' % v) if isinstance(v, float) else v for k, v in params.items()}

        return xgb.train(params, self.data, best_iter)

    def __evaluator(self, max_depth, gamma, min_child_weight, subsample, colsample_bytree,
                    reg_alpha, reg_lambda):
        params = dict(
            silent=True,
            booster='gbtree',
            objective='binary:logistic',
            max_depth=int(max_depth),
            learning_rate=0.01,
            gamma=gamma,  # min_split_gain
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bylevel=0.8,  # 每一层的列数
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=1,
            random_state=None,
            n_jobs=8
        )
        params['eta'] = params.pop('learning_rate')
        params['alpha'] = params.pop('reg_alpha')
        params['lambda'] = params.pop('reg_lambda')

        metric = 'auc'
        cv_rst = xgb.cv(params, self.data, num_boost_round=3600, nfold=5, metrics=metric, early_stopping_rounds=100,
                        verbose_eval=None, stratified=True, show_stdv=False, as_pandas=False)

        _ = cv_rst['test-%s-mean' % metric]
        print('\nBest Iter: %s' % len(_))
        return _[-1]
