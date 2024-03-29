#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : DataIter
# @Time         : 2019-06-21 14:48
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :

import numpy as np
from tensorflow.python.keras.utils import Sequence
from sklearn.utils import shuffle


class DataIter(Sequence):

    def __init__(self, x, y, batch_size=128, data_processor=lambda *args: args, data_shuffle=False):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle

        assert callable(data_processor) is True
        self.data_processor = data_processor

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))  # steps

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return self.data_processor(batch_x, batch_y)

    # TODO:
    # @abstractmethod
    # def data_processor(self, batch_x, batch_y):
    #     pass

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        if self.data_shuffle:
            self.x, self.y = shuffle(self.x, self.y)
        else:
            pass
