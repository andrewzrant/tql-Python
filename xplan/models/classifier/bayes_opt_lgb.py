#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'opt'
__author__ = 'JieYuan'
__mtime__ = '19-1-3'
"""
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
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

    def __init__(self, X, y, topk=10, categorical_feature='auto', metric='auc', fix_params={'min_child_weight': 0.001}):
        self.data = lgb.Dataset(X, y, categorical_feature=categorical_feature, free_raw_data=False)
        self.topk = topk
        self.metric = metric
        self.fix_params = fix_params
        self.params_ls = []
        self.params_ls_sk = []
        self.params_best = {}
        self.params_best_sk = {}
        self.params_opt = None
        self.__iteration = {}

        if self.fix_params:
            print('\033[94m%s\033[0m\n' % " Fix min_child_weight ...")

    @property
    def get_best_model(self, best_iter):
        return lgb.train(self.params, self.data, best_iter)

    def run(self, n_iter=10, save_log=False):
        logger = JSONLogger(path="./opt_lgb_logs.json")

        BoParams = {
            'num_leaves': (2 ** 5, 2 ** 16),
            'min_split_gain': (0.01, 1),
            'min_child_weight': (0, 0.01),
            'min_child_samples': (8, 32),
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
            {'num_leaves': 2 ** 7 - 1,
             'min_split_gain': 0,
             'min_child_weight': 0.001,
             'min_child_samples': 6,
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             'reg_alpha': 0.01,
             'reg_lambda': 1})

        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        optimizer.maximize(init_points=3, n_iter=n_iter, acq='ucb', kappa=2.576, xi=0.0, **gp_params)
        self.best_params = optimizer.max
        self.__get_params()

        self.params_opt = (
            pd.concat([pd.DataFrame(self.__iteration), pd.DataFrame(optimizer.res)], 1)
                .sort_values('target', ascending=False)
                .reset_index(drop=True)[:self.topk])
        self.__get_params()

    def __evaluator(self, num_leaves, min_split_gain, min_child_weight, min_child_samples, subsample, colsample_bytree,
                    reg_alpha, reg_lambda):
        self.__params_sk = dict(
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
            random_state=0,
            n_jobs=-1
        )
        params = self.__params_sk.copy()
        # params['metric'] = self.metric

        cv_rst = lgb.cv(params, self.data, num_boost_round=3000, nfold=5, early_stopping_rounds=100,
                        metrics=self.metric, verbose_eval=None, show_stdv=False)

        _ = cv_rst['%s-mean' % self.metric]
        self.__iteration.setdefault('best_iteration', []).append(len(_))
        return _[-1]

    def __get_params(self):
        for _, (i, p, _) in self.params_opt.iterrows():
            params_sk = {**p, **{'n_estimators': i}, **self.__params_sk}
            params_sk['num_leaves'] = int(params_sk['num_leaves'])
            params_sk['min_child_samples'] = int(params_sk['min_child_samples'])
            params_sk = {k: float('%.3f' % v) if isinstance(v, float) else v for k, v in params_sk.items()}
            self.params_ls_sk.append(params_sk)

            params = params_sk.copy()
            num_boost_round = params.pop('n_estimators')
            self.params_ls.append({'params': params, 'num_boost_round': num_boost_round})

        self.params_best = self.params_ls[0]
        self.params_best_sk = self.params_ls_sk[0]
