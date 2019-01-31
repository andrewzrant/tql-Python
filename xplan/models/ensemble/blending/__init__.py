#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '19-1-30'
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns


class SklearnHelper(object):
    def __init__(self, algo=LGBMClassifier, params={'n_estimators': 3000}):
        self.clf = algo(**params)

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.model = self.clf.fit(X_train, y_train,
                                  eval_set=[(X_valid, y_valid)],
                                  eval_metric='auc',
                                  early_stopping_rounds=100,
                                  verbose=50)
        return self.model

    def predict(self, X):
        try:
            predict = lambda x: self.clf.__getattribute__('predict_proba')(x, verbose=100)[:, 1]
        except:
            predict = self.clf.__getattribute__('preidict')
        return predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class Blending(object):

    def __init__(self, df_train, df_test, target_name):
        """重写SklearnHelper
        class SklearnHelper(object):
            def __init__(self, algo=LGBMClassifier, params={'n_estimators': 3000}):
                self.clf = algo(**params)

            def fit(self, X_train, y_train, X_valid, y_valid):
                self.model = self.clf.fit(X_train, y_train,
                                          eval_set=[(X_valid, y_valid)],
                                          eval_metric='auc',
                                          early_stopping_rounds=100,
                                          verbose=50)
                return self.model

            def predict(self, X):
                try:
                    predict = lambda x: self.clf.__getattribute__('predict_proba')(x, verbose=100)[:, 1]
                except:
                    predict = self.clf.__getattribute__('preidict')
                return predict(X)

            @property
            def feature_importances_(self):
                return self.model.feature_importances_
        """
        self.df_train = df_train
        self.df_test = df_test
        self.target_name = target_name
        self.model = SklearnHelper()  # 重写SklearnHelper

    def get_oof(self, f_eval, num_folds=5, stratified=False, feats_exlude=[], plot=False):
        """
        :param f_eval: f_eval(y_true, y_pred)
        :param num_folds:
        :param stratified:
        :param feats_exlude:
        :return:
        """

        feats = [f for f in self.df_train.columns if f not in feats_exlude + [self.target_name]]
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2019).split()
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=2019)

        folds = folds.split(self.df_train[feats], self.df_train[self.target_name])

        # Create arrays and dataframes to store results
        oof_preds = np.zeros(len(self.df_train))
        sub_preds = np.zeros(len(self.df_test))
        self.feature_importance_df = pd.DataFrame()

        for n_fold, (train_idx, valid_idx) in enumerate(folds, 1):
            X_train, y_train = self.df_train[feats].iloc[train_idx], self.df_train[self.target_name].iloc[train_idx]
            X_valid, y_valid = self.df_train[feats].iloc[valid_idx], self.df_train[self.target_name].iloc[valid_idx]

            model = self.model.fit(X_train, y_train, X_valid, y_valid)

            oof_preds[valid_idx] = model.predict(X_valid)
            sub_preds += model.predict(self.df_test[feats]) / num_folds

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = np.log1p(model.feature_importances_)
            fold_importance_df["fold"] = n_fold
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)
            print('Fold %s: Metric: %.6f\n' % (n_fold, f_eval(y_valid, oof_preds[valid_idx])))
        if plot:
            self.plot_importances(self.feature_importance_df)
        return oof_preds, sub_preds

    def plot_importances(self, feature_importance_df, topk=40):
        """Display/plot feature importance"""
        data = (feature_importance_df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .reset_index()
                .sort_values("importance", 0, False))[:topk]
        plt.figure(figsize=(8, 10))
        sns.barplot(x="importance", y="feature", data=data)
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')


if __name__ == '__main__':
    class SklearnHelper:
        def __init__(self, algo=LGBMRegressor, params={'n_estimators': 3000}):
            self.clf = algo(**params)

        def fit(self, X_train, y_train, X_valid, y_valid):
            self.model = self.clf.fit(X_train, y_train,
                                      eval_set=[(X_valid, y_valid)],
                                      eval_metric='l1',
                                      early_stopping_rounds=100,
                                      verbose=50)
            return self.model

        def predict(self, X):
            try:
                predict = lambda x: self.clf.__getattribute__('predict_proba')(x, verbose=100)[:, 1]
            except:
                predict = self.clf.__getattribute__('preidict')
            return predict(X)

        @property
        def feature_importances_(self):
            return self.model.feature_importances_


    blending = Blending(df_train, df_train, '信用分')
    blending.model = SklearnHelper()
    blending.get_oof(mean_absolute_error, feats_exlude=['用户编码'], plot=True)
