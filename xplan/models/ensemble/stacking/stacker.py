#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'stacker'
__author__ = 'JieYuan'
__mtime__ = '19-2-18'
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score


class Stacker(object):

    def __init__(self, X, y, X_test, params_sk=None):
        self.X = X
        self.X_test = X_test
        self.y = y
        self.params_sk = params_sk

    def get_oof(self, f_eval=roc_auc_score, num_folds=5, stratified=False, feats_exlude=[], plot=False):
        """
        :param f_eval: f_eval(y_true, y_pred)
        :param num_folds:
        :param stratified:
        :param feats_exlude:
        :return: oof_preds, sub_preds
        """
        feats = list(filter(lambda feat: feat not in feats_exlude, self.X.columns))
        X, X_test = self.X[feats], self.X_test[feats]
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2019)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=2019)

        folds = folds.split(X, self.y)

        # Create arrays and dataframes to store results
        oof_preds = np.zeros(len(X))
        sub_preds = np.zeros(len(X_test))
        self.feature_importance_df = pd.DataFrame()

        for n_fold, (train_idx, valid_idx) in enumerate(folds, 1):
            X_train, y_train = X.iloc[train_idx], self.y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], self.y.iloc[valid_idx]

            clf = self._fit(X_train, y_train, X_valid, y_valid)

            oof_preds[valid_idx] = clf.predict_proba(X_valid)[:, 1]
            sub_preds += clf.predict_proba(X_test)[:, 1] / num_folds

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = np.log1p(clf.feature_importances_)
            fold_importance_df["fold"] = n_fold
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], 0)
        if plot:
            self.plot_importances(self.feature_importance_df)

        print('Bagging Score: %s' % f_eval(self.y, oof_preds))
        return oof_preds, sub_preds

    def plot_importances(self, topk=40):
        """Display/plot feature importance"""
        data = (self.feature_importance_df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .reset_index()
                .sort_values("importance", 0, False))[:topk]
        plt.figure(figsize=(8, 10))
        sns.barplot(x="importance", y="feature", data=data)
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')

    def _fit(self, X_train, y_train, X_valid, y_valid):
        if self.params_sk is None:
            self.params_sk = {'boosting_type': 'gbdt',
                              'colsample_bytree': 0.8,
                              'learning_rate': 0.03,
                              'max_depth': -1,
                              'metric': 'auc',
                              'min_child_samples': 20,
                              'min_child_weight': 0.001,
                              'min_split_gain': 0.0,
                              'n_estimators': 30000,
                              'importance_type': 'gain',
                              'n_jobs': 16,
                              'num_leaves': 33,
                              'objective': 'binary',
                              'random_state': 2019,
                              'reg_alpha': 0.0,
                              'reg_lambda': 0.0,
                              'scale_pos_weight': 1,
                              'subsample': 0.8,
                              'subsample_freq': 3,
                              'verbosity': -1}

        self.clf = LGBMClassifier(**self.params_sk)
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf.fit(X_train, y_train, eval_set=eval_set, eval_metric='auc', early_stopping_rounds=100, verbose=100)
        return self.clf
