#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'stacker'
__author__ = 'JieYuan'
__mtime__ = '19-1-14'
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


class Stacker(object):
    """二分类"""

    def __init__(self, clf, cv=3, seed=None):
        self._clf = clf
        self.seed = seed  # 不同模型的种子（是否必要？）
        self.cv = cv
        self.clfs = []
        self.clf_by_all = None

        self.oof = None

    def fit(self, X, y):
        flods = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=666).split(X, y)
        _fit = partial(self._fit_and_score, clf=self._clf, X=X, y=y)

        with ProcessPoolExecutor(self.cv) as pool:
            _ = pool.map(_fit, tqdm(flods, self._clf.__repr__().split('(')[0]))

        preds = []
        scores = []
        for clf, pred, score in _:
            self.clfs.append(clf)
            preds.append(pred)
            scores.append(score)
        print('Score CV : %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))

        preds = np.row_stack(preds)
        self.oof = preds[np.argsort(preds[:, 0])][:, -1]  # 按照X原顺序排序并去掉索引
        print('Score OOF : %.4f' % roc_auc_score(y, self.oof))

        del scores, preds

    def _fit_and_score(self, iterable, clf, X, y):
        train_index, test_index = iterable
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        #################自定义最优模型###################
        clf.fit(X_train, y_train)
        test_pred = np.column_stack((test_index, clf.predict_proba(X_test)))
        score = roc_auc_score(y_test, test_pred[:, -1])
        ###############################################
        return clf, test_pred, score

    def predict_meta_features(self, X):
        _ = np.column_stack([clf.predict_proba(X)[:, -1] for clf in self.clfs]).mean(1)
        return _
