#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'isolation_forest'
__author__ = 'JieYuan'
__mtime__ = '19-1-3'
"""


class Outlier(object):

    def isolation_forest(self, random_state=None):
        from sklearn.ensemble import IsolationForest
        ilf = IsolationForest(
            n_estimators=256,
            max_samples='auto',
            contamination='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=8,
            behaviour='new',
            random_state=random_state,
            verbose=0)
        return ilf
