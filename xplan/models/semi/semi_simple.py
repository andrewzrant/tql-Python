#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'semi_simple'
__author__ = 'JieYuan'
__mtime__ = '19-1-11'
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from ...pipe import tqdm
from ...utils import Cprint
from ..classifier import BayesOptLGB


class SemiSimple(object):

    def __init__(self, subsample=0.05, n_iter=3):
        self.subsample = subsample
        self.n_iter = n_iter
        _ = "X_train will stack %.2f% of X_test" % ((1 - (1 - 2 * self.subsample) ** n_iter) * 100)
        Cprint().cprint(_)

    def fit(self, X_train, y_train, X_test):
        X_test = np.asarray(X_test)
        for _ in tqdm(range(self.n_iter)):
            ##############可以定义其他模型#################
            # self.clf.fit(X_train, y_train)
            bo = BayesOptLGB(X_train, y_train)
            bo.run()
            self.clf = LGBMClassifier(**bo.params_best_sk)
            ############################################
            pred = (
                pd.Series(self.clf.predict_proba(X_test)[:, 1])
                    .mask(lambda x: x < x.quantile(self.subsample), 0)
                    .mask(lambda x: x > x.quantile(1 - self.subsample), 1)
            )
            pred_ = pred[lambda x: x.isin([0, 1])]

            pseudo_label_idx = pred_.index
            no_pseudo_label_idx = pred.index.difference(pseudo_label_idx)
            X_train = np.row_stack((X_train, X_test[pseudo_label_idx]))
            y_train = np.hstack((y_train, pred_))
            X_test = X_test[no_pseudo_label_idx]
        return self.clf
