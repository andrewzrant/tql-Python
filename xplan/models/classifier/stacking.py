#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'stacking'
__author__ = 'JieYuan'
__mtime__ = '19-1-7'
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm


class Stacking(object):
    def __init__(self, cv, stacker, base_models):
        self.cv = cv
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(self.cv, True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in tqdm(enumerate(self.base_models)):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)

                y_pred = clf.predict_proba(X_holdout)[:, 1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:, 1]
        return y_pred
