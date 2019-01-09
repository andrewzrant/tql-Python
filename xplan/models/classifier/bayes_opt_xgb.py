#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'bayes_xgb'
__author__ = 'JieYuan'
__mtime__ = '19-1-3'
"""
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
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

def __init__(self, X, y, topk=10, missing=None, metric='auc', fix_params={}):
    self.data = xgb.DMatrix(X, y, missing=missing)
    self.topk = topk
    self.metric = metric
    self.fix_params = fix_params  # 固定不需要调节的参数
    self.params_ls = []
    self.params_ls_sk = []
    self.params_best = {}
    self.params_best_sk = {}
    self.params_opt = None
    self.__iteration = {}

    if self.fix_params:
        print('\033[94m%s\033[0m\n' % " Fix min_child_weight ...")


@property
def best_model(self):
    if self.params_best:
        return xgb.train(dtrain=self.data, **self.params_best)
    else:
        print('\033[94m%s\033[0m\n' % "Please Run !")

def run(self, n_iter=10, save_log=False):
    logger = JSONLogger(path="./opt_xgb_logs.json")

    BoParams = {
        'max_depth': (5, 16),
        'min_child_weight': (1, 10),
        'gamma': (0, 1),
        'subsample': (0.6, 1),
        'colsample_bytree': (0.6, 1),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1),
    }
    optimizer = BayesianOptimization(self.__evaluator, BoParams)
    if save_log:
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    if self.fix_params:
        optimizer.set_bounds({k: (v, v) for k, v in self.fix_params.items()})

    optimizer.probe(
        {'max_depth': 7,
         'min_child_weight': 1,
         'gamma': 0,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'reg_alpha': 0.01,
         'reg_lambda': 1})

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3}
    optimizer.maximize(init_points=3, n_iter=n_iter, acq='ucb', kappa=2.576, xi=0.0, **gp_params)

    self.params_opt = (
        pd.concat([pd.DataFrame(self.__iteration), pd.DataFrame(optimizer.res)], 1)
            .sort_values('target', ascending=False)
            .reset_index(drop=True)[:self.topk])
    self.__get_params()


def __evaluator(self, max_depth, gamma, min_child_weight, subsample, colsample_bytree,
                reg_alpha, reg_lambda):
    self.__params_sk = dict(
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
        random_state=0,
        n_jobs=-1
    )
    params = self.__params_sk.copy()
    params['eta'] = params.pop('learning_rate')
    params['alpha'] = params.pop('reg_alpha')
    params['lambda'] = params.pop('reg_lambda')
    # params['eval_metric'] = self.metric

    cv_rst = xgb.cv(params, self.data, num_boost_round=3600, nfold=5, early_stopping_rounds=100,
                    metrics=self.metric, stratified=True, show_stdv=False, as_pandas=False)
    _ = cv_rst['test-%s-mean' % self.metric]
    self.__iteration.setdefault('best_iteration', []).append(len(_))
    return _[-1]


def __get_params(self):
    for _, (i, p, _) in self.params_opt.iterrows():
        params_sk = {**p, **{'n_estimators': i}, **self.__params_sk}
        params_sk['max_depth'] = int(params_sk['max_depth'])
        params_sk = {k: float('%.3f' % v) if isinstance(v, float) else v for k, v in params_sk.items()}
        self.params_ls_sk.append(params_sk)

        params = params_sk.copy()
        params['alpha'] = params.pop('reg_alpha')
        params['lambda'] = params.pop('reg_lambda')
        num_boost_round = params.pop('n_estimators')
        self.params_ls.append({'params': params, 'num_boost_round': num_boost_round})

    self.params_best = self.params_ls[0]
    self.params_best_sk = self.params_ls_sk[0]
