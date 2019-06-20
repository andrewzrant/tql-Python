#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : DataGenerator
# @Time         : 2019-06-20 17:22
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 
from sklearn.utils import shuffle


class DataGenerator(object):

    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.steps = (len(X) + batch_size - 1) // batch_size

    def __len__(self):
        return self.steps

    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        while 1:
            for i in range(self.steps):
                idx_s = i * self.batch_size
                idx_e = (i + 1) * self.batch_size
                yield self.X[idx_s:idx_e], self.y[idx_s:idx_e]

    def _iter_bert(self):
        pass
