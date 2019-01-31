#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'sklearn_helper'
__author__ = 'JieYuan'
__mtime__ = '19-1-31'
"""


class SklearnHelper(object):
    def __init__(self, clf, params={}):
        self.clf = clf(**params)

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    def predict(self, X):
        try:
            predict = lambda x: self.clf.__getattribute__('predict_proba')(x)[:, 1]
        except:
            predict = self.clf.__getattribute__('preidict')

        return predict(X)

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_
