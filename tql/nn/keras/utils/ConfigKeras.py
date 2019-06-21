#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : ConfigKeras
# @Time         : 2019-06-21 15:28
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import os
import numpy as np
import tensorflow as tf
import random
from keras import backend as K


class ConfigKeras(object):
    """
    https://www.cnblogs.com/wuliytTaotao/p/10883749.html
    """

    def __init__(self, seed=2019):
        self.seed = seed

    def set_seed(self):
        os.environ["PYTHONHASHSEED"] = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
