#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '19-1-2'
"""

import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score


class OOF(object):
    """Out of flod prediction
    TODO: 目前仅支持二分类问题,未来将支持回归问题
    """
    _params = {'metric': 'auc',
               'learning_rate': 0.01,
               'n_estimators': 30000,
               'subsample': 0.8,
               'colsample_bytree': 0.8,
               'class_weight': 'balanced',  ##
               'scale_pos_weight': 1,  ##
               'random_state': 2019,
               'verbosity': -1}

    def __init__(self, clf=None, folds=None):
        self.clf = clf if clf else LGBMClassifier(**self._params)
        self.folds = folds if folds else StratifiedKFold(5, True, 2019)  # 支持 RepeatedStratifiedKFold

        self.model_type = self.clf.__repr__().split('(')[0]
        # self.clf_agrs = self.getfullargspec(self.clf.fit).args if hasattr(self.clf, 'fit') else None

    def fit(self, X, y, X_test, exclude_columns=None):
        """
        :param f_eval: f_eval(y_true, y_pred)
        :param num_folds:
        :param stratified:
        :param exclude_columns:
        :return: oof_preds, sub_preds
        """
        # 移除不需要的特征
        if exclude_columns:
            feats = X.columns.difference(exclude_columns)
        else:
            feats = X.columns

        X, X_test = X[feats], X_test[feats]

        if hasattr(self.folds, 'n_splits'):
            num_folds = self.folds.n_splits
        else:
            num_folds = self.folds.cvargs['n_splits']

        # Cross validation model
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(len(X))
        sub_preds = np.zeros(len(X_test))
        self.feature_importance_df = pd.DataFrame()

        for n_fold, (train_idx, valid_idx) in enumerate(self.folds.split(X, y), 1):
            print("\033[94m Fold %s Started At %s \033[0m\n" % (n_fold, time.ctime()))
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

            if not hasattr(self.clf, 'fit'):
                print("该算法无fit方法")
                break
            else:
                if self.model_type == 'LGBMClassifier':
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]

                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 categorical_feature='auto',  # TODO: 类别型的支持
                                 eval_metric='auc',
                                 early_stopping_rounds=100,
                                 verbose=100)
                elif self.model_type == 'XGBClassifier':
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 eval_metric='auc',
                                 early_stopping_rounds=100,
                                 verbose=100)
                elif self.model_type == 'CatBoostClassifier':
                    # CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss',  eval_metric='AUC', **params)
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 cat_features=[],  # Categ columns indices # TODO: 类别型的支持
                                 use_best_model=True,
                                 early_stopping_rounds=100,
                                 verbose=100)  # verbose_eval?
                else:
                    self.clf.fit(X, y)

                oof_preds[valid_idx] = self.clf.predict_proba(X_valid)[:, 1]
                sub_preds += self.clf.predict_proba(X_test)[:, 1] / num_folds

                if hasattr(self.clf, 'feature_importances_'):
                    fold_importance_df = pd.DataFrame()
                    fold_importance_df["feature"] = feats
                    fold_importance_df["importance"] = self.clf.feature_importances_
                    fold_importance_df["fold"] = n_fold
                    self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], 0)

        print('OOF AUC: %s' % roc_auc_score(y, oof_preds))

        if hasattr(self.clf, 'feature_importances_'):
            self.plot_importances(self.feature_importance_df)

        self.oof_preds = oof_preds
        self.test_preds = sub_preds
        return oof_preds, sub_preds

    def plot_importances(self, topk=40):
        """Display/plot feature importance"""
        data = (self.feature_importance_df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .reset_index()
                .sort_values("importance", 0, False))[:topk]
        plt.figure(figsize=(8, 16))
        sns.barplot(x="importance", y="feature", data=data)
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')
