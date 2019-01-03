#!/usr/bin/env python
# -*- coding= utf-8 -*-
"""
__title__ =  lgb 
__author__ =  JieYuan 
__mtime__ =  19-1-2 
"""

import lightgbm as lgb


class BaselineLGB(object):

    def __init__(self, X, y, learning_rate=0.1, metrics='auc', feval=None, objective='binary', scale_pos_weight=1,
                 seed=None, n_jobs=8):
        """
        :param objective:
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        :param metrics: string, list of strings or None, optional (default=None)
            binary: 'auc', 'binary_error', 'binary_logloss'
            multiclass: 'multi_error', 'multi_logloss'
            https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
        :param feval:
            def feval(y_pred, y_true):
                y_true = y_true.get_label()
                return '1 / (1 + rmse)', 1 /(rmse(y_true, y_pred) + 1), True
        :param scale_pos_weight:
        """
        self.lgb_data = lgb.Dataset(X, y, weight=None, init_score=None)  # init_score初始分(例如常值回归的得分)
        self.objective = objective
        self.metrics = metrics
        self.feval = feval
        self.best_iter = None
        # sklearn params
        self.params_sk = dict(
            boosting_type='gbdt',
            objective=objective,
            max_depth=-1,
            num_leaves=2 ** 7 - 1,
            learning_rate=learning_rate,

            min_split_gain=0.0,  # 描述分裂的最小 gain, 控制树的有用的分裂
            min_child_weight=0.001,  # 决定最小叶子节点样本权重和, 孩子节点中最小的样本权重和, 避免过拟合, 如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束

            subsample=0.8,
            subsample_freq=8,
            colsample_bytree=0.8,

            reg_alpha=0.0,
            reg_lambda=0.0,

            scale_pos_weight=scale_pos_weight,

            random_state=seed,
            n_jobs=n_jobs
        )
        self.params = self.params_sk.copy()

        if self.objective == 'multiclass':
            self.num_class = len(set(y))
            self.params['objective'] = self.objective
            self.params['num_class'] = self.num_class

    def cv(self, return_model=False, nfold=5, early_stopping_rounds=50, verbose_eval=50):

        print("LGB CV ...\n")
        try:
            cv_rst = lgb.cv(
                self.params,
                self.lgb_data,
                metrics=self.metrics,
                feval=self.feval,
                nfold=nfold,
                num_boost_round=2500,
                stratified=False if 'reg' in self.objective else True,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )
        except TypeError:
            print("Please: self.lgb_data = lgb.Dataset(X, y, weight=None, init_score=None)")

        if isinstance(self.metrics, str):
            _ = cv_rst['%s-mean' % self.metrics]
            self.best_iter = len(_)
            print('\nBest Iter: %s' % self.best_iter)
            print('Best Score: %s ' % _[-1])
        else:
            _ = cv_rst['%s-mean' % self.metrics[0]]
            self.best_iter = len(_)
            print('\nBest Iter: %s' % self.best_iter)
            print('Best Score: %s ' % _[-1])

        self.params_sk['n_estimators'] = self.best_iter

        if return_model:
            print("\nReturning Model ...\n")
            return lgb.train(self.params, self.lgb_data, self.best_iter)
