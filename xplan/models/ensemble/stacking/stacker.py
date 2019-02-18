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


class SklearnHelper(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.clf = LGBMClassifier(n_estimators=3000)
        self.clf.fit(X_train, y_train,
                     eval_set=[(X_valid, y_valid)],
                     eval_metric='auc',
                     early_stopping_rounds=100,
                     verbose=50)
        return self.clf

    def predict(self, X):
        try:
            predict = lambda x: self.clf.__getattribute__('predict_proba')(x, verbose=100)[:, 1]
        except:
            predict = self.clf.__getattribute__('preidict')
        return predict(X)

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_


class Stacker(object):

    def __init__(self, X, y, X_test):
        """重写SklearnHelper
        class NewSK(SklearnHelper):
            def fit(self, X_train, y_train, X_valid, y_valid):
                b = BayesOptLGB(X_train, y_train, 'l1', 'regression')
                b.run(False)
                self.clf = LGBMRegressor(**b.params_best_sk)
                self.clf.fit(X_train, y_train,
                             eval_set=[(X_valid, y_valid)],
                             eval_metric='l1',
                             early_stopping_rounds=100,
                             verbose=50)
                return self.clf

            def fit(self, X_train, y_train, X_valid, y_valid):
                self.clf = LGBMRegressor(**p)
                self.clf.fit(X_train, y_train,
                             eval_set=[(X_valid, y_valid)],
                             eval_metric='l1',
                             early_stopping_rounds=100,
                             verbose=100)
                return self.clf
            blending = Blending(X, y, X_test)
            blending.model = NewSK()
            blending.get_oof(f_eval)
        """
        self.X = X
        self.X_test = X_test
        self.y = y
        self.model = SklearnHelper()  # 重写SklearnHelper

    def get_oof(self, f_eval, num_folds=5, stratified=False, feats_exlude=[], plot=False):
        """
        :param f_eval: f_eval(y_true, y_pred)
        :param num_folds:
        :param stratified:
        :param feats_exlude:
        :return:
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

            model = self.model.fit(X_train, y_train, X_valid, y_valid)

            oof_preds[valid_idx] = model.predict(X_valid)
            sub_preds += model.predict(X_test) / num_folds

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = np.log1p(model.feature_importances_)
            fold_importance_df["fold"] = n_fold
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], 0)
            print('Fold %s: Metric: %.6f\n' % (n_fold, f_eval(y_valid, oof_preds[valid_idx])))
        if plot:
            self.plot_importances(self.feature_importance_df)

        print('Blending Score: %s' % f_eval(self.y, oof_preds))
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
