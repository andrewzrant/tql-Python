#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : BaseModel
# @Time         : 2019-06-22 22:21
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from abc import abstractmethod
from tensorflow.python.keras.utils import plot_model as _plot_model
from pathlib import Path


class BaseModel(object):

    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, plot_model=None, dir='.', **kwargs):
        model = self.get_model()
        model.summary()
        if plot_model:
            image_file = Path(dir) / ('%s.png' % self._class_name)
            image_file = image_file.absolute().__str__()
            try:
                _plot_model(model, to_file=image_file, show_shapes=True, dpi=256)
            except Exception as e:
                print(e)
                print("brew install graphviz or apt-get install graphviz")
        return model

    @property
    def _class_name(self):
        return str(self).split(maxsplit=1)[0][10:]

    @abstractmethod
    def get_model(self):
        pass


if __name__ == '__main__':
    bm = BaseModel()
    print(bm._class_name)
