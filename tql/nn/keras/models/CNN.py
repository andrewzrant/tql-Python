#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : CNN
# @Time         : 2019-06-21 16:07
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adadelta


class CNN(object):
    def __init__(self, X, y, optimizer=Adadelta(), batch_size=128, nb_epoch=10):
        self.optimizer = optimizer
        self.input_shape = X.shape[1:]
        self.out_dim = y.shape[1]

        print(f"Input Dim: {self.input_shape}")
        print(f"Out Dim: {self.out_dim}\n")

    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.out_dim, activation='softmax'))

        # self.model.load_weights(self.best_model_weight, by_name=True)
        self.model.compile(self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
