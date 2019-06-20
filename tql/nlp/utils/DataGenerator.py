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

    def __init__(self, X, y, batch_size=32, maxlen=None, is_train=1, mapper=lambda *args: args):
        """

        :param X:
        :param y:
        :param batch_size:
        :param maxlen:
        :param is_train:
        :param mapper:
            # bert keras
            from functools import partial
            from keras.preprocessing.sequence import pad_sequences
            from keras_bert import load_trained_model_from_checkpoint, Tokenizer

            func = partial(pad_sequences, maxlen=self._maxlen)
            def mapper(X, y):
                X = list(map(mapper, zip(*map(tokenizer.encode, X))))
                return X, y

        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.steps = (len(X) + batch_size - 1) // batch_size
        self._maxlen = maxlen if maxlen else max(map(len, X), 1024)
        self._is_train = is_train

        assert callable(mapper) is True
        self.mapper = mapper

    def __len__(self):
        return self.steps

    def __iter__(self):
        if self._is_train:
            self.X, self.y = shuffle(self.X, self.y)
        while 1:
            for i in range(self.steps):
                idx_s = i * self.batch_size
                idx_e = (i + 1) * self.batch_size

                X = self.X[idx_s:idx_e]
                y = self.y[idx_s:idx_e]

                yield self.mapper(X, y)
