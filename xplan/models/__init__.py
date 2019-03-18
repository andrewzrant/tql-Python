#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '19-1-2'
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.api import GLM, families
from xgboost import XGBClassifier

"""
# TODO: cats
https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
"""


class OOF(object):
    """Out of flod prediction
    # TODO 支持回归

    lightGBM一个一个地建立节点; XGboost一层一层地建立节点
    https://blog.csdn.net/friyal/article/details/82758532
    Catboost总是使用完全二叉树。它的节点是镜像的(对称树)。Catboost称对称树有利于避免overfit，增加可靠性，并且能大大加速预测等等。
        计算某个category出现的频率，加上超参数，生成新的numerical features
    # https://blog.csdn.net/linxid/article/details/80723811
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
    lgb = LGBMClassifier(n_jobs=16, **_params)
    xgb = XGBClassifier()
    cat = CatBoostClassifier(n_estimators=20000, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC',
                             random_state=2019)

    def __init__(self, clf=None, folds=None, early_stopping_rounds=300, verbose=100):
        self.clf = clf if clf else self.lgb
        self.folds = folds if folds else StratifiedKFold(5, True, 2019)  # 支持 RepeatedStratifiedKFold
        self.model_type = self.clf.__repr__()

        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        # self.clf_agrs = self.getfullargspec(self.clf.fit).args if hasattr(self.clf, 'fit') else None

    def fit(self, X, y, X_test, feval=None, cat_feats=None, exclude_columns=None, epochs=16, batch_size=128,
            oof2csv=False):
        """
        # TODO: Rank 融合
        :param X:
        :param y:
        :param X_test:
        :param feval: roc_auc_score(y_true, y_score)
        :param cat_feats: 类别特征索引
        :param exclude_columns:
        仅针对 nn
        :param epochs:
        :param batch_size:
        :return:
        """
        # 判断输入数据是否是数据框
        is_df = isinstance(X, pd.DataFrame)

        # oof评估函数
        feval = feval if feval else roc_auc_score

        # 移除不需要的特征
        if is_df:
            if exclude_columns:
                feats = X.columns.difference(exclude_columns)
            else:
                feats = X.columns

            X, X_test = X[feats], X_test[feats]
        else:
            feats = range(X.shape[1])

        # if hasattr(self.folds, 'n_splits'):
        #     num_folds = self.folds.n_splits
        # else:
        #     num_folds = self.folds.cvargs['n_splits']

        # Score
        if hasattr(feval, '__repr__'):
            score_name = feval.__repr__().split()[1]
        else:
            score_name = None

        # Cross validation model
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(X.shape[0])
        sub_preds = np.zeros(X_test.shape)
        self.feature_importance_df = pd.DataFrame()

        for n_fold, (train_idx, valid_idx) in enumerate(self.folds.split(X, y), 1):
            print("\n\033[94mFold %s started at %s\033[0m" % (n_fold, time.ctime()))

            if is_df:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            else:
                X_train, y_train = X[train_idx], y[train_idx]
                X_valid, y_valid = X[valid_idx], y[valid_idx]

            if not hasattr(self.clf, 'fit'):
                print("该算法无fit方法")
                break
            else:
                if 'LGBMClassifier' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 categorical_feature=cat_feats if cat_feats else 'auto',
                                 eval_metric='auc',
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 verbose=self.verbose)
                elif 'LGBMRegressor' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 categorical_feature=cat_feats if cat_feats else 'auto',
                                 eval_metric='l2',
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 verbose=self.verbose)

                elif 'XGBClassifier' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 eval_metric='auc',
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 verbose=self.verbose)
                elif 'XGBRegressor' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 eval_metric='rmse',
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 verbose=self.verbose)

                elif 'CatBoostClassifier' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 cat_features=cat_feats,
                                 use_best_model=True,
                                 plot=True,
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 verbose=self.verbose)
                elif 'CatBoostRegressor' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 cat_features=cat_feats,
                                 use_best_model=True,
                                 plot=True,
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 verbose=self.verbose)

                elif 'RGFClassifier' in self.model_type:
                    pass
                elif 'RGFRegressor' in self.model_type:
                    pass

                # https://www.cnblogs.com/flyu6/p/7691106.html
                elif 'KerasClassifier' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=eval_set)
                elif 'KerasRegressor' in self.model_type:
                    eval_set = [(X_train, y_train), (X_valid, y_valid)]
                    self.clf.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=eval_set)

                elif self.model_type == 'GLM':
                    # TODO: 其他模型的支持
                    self.clf = GLM(y_train, X_train, family=families.Binomial())
                    self.clf = self.clf.fit().predict(X)
                else:
                    # sklearn 原生模型
                    self.clf.fit(X, y)

                # 计算并保存 preds
                # TODO: 多分类需要修改
                if hasattr(self.clf, 'predict_proba'):
                    oof_preds[valid_idx] = self.clf.predict_proba(X_valid)[:, 1]
                    sub_preds[:, n_fold - 1] = self.clf.predict_proba(X_test)[:, 1]
                else:
                    oof_preds[valid_idx] = self.clf.predict(X_valid)
                    sub_preds[:, n_fold - 1] = self.clf.predict(X_test)

            if hasattr(self.clf, 'feature_importances_'):
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = feats
                fold_importance_df["importance"] = self.clf.feature_importances_
                fold_importance_df["fold"] = n_fold
                self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], 0)

        # 输出需要的结果
        self.oof_preds = oof_preds
        self.sub_preds = sub_preds.mean(1)
        self.sub_preds_rank = pd.DataFrame(sub_preds).rank().mean(1) / sub_preds.shape[0]

        try:
            score = feval(y, self.oof_preds)
        except Exception as e:
            score = 0
            print('Error feval:', e)

        print("\n\033[94mCV Score %s: %s ended at %s\033[0m" % (score_name, score, time.ctime()))

        # 保存的普通平均的得分
        if oof2csv:
            pd.Series(self.oof_preds.tolist() + self.sub_preds.tolist(), name='oof') \
                .to_csv('OOF %s %.4f.csv' % (time.ctime(), score), index=False)

        if hasattr(self.clf, 'feature_importances_'):
            self.plot_importances(self.feature_importance_df)

    def plot_importances(self, df, topk=64):
        """Display/plot feature importance"""
        assert "feature" in df.columns and "importance" in df.columns, '无["feature", "importance"]'

        data = (df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .reset_index()
                .sort_values("importance", 0, False))[:topk]

        plt.figure(figsize=(12, topk // 4))
        sns.barplot(x="importance", y="feature", data=data.assign(feature='col_' + data.feature.astype(str)))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')
