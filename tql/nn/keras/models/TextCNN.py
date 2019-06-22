#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : TextCNN
# @Time         : 2019-06-21 18:56
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, Dense, Concatenate, Dropout
from tensorflow.python.keras.layers import Conv1D, GlobalMaxPool1D, GlobalAvgPool1D
from .BaseModel import BaseModel


class TextCNN(BaseModel):
    """ TextCNN:
    1. embedding layers,
    2. convolution layer,
    3. max-pooling,
    4. softmax layer.
    数据量较大：可以直接随机初始化embeddings，然后基于语料通过训练模型网络来对embeddings进行更新和学习。
    数据量较小：可以利用外部语料来预训练(pre-train)词向量，然后输入到Embedding层，用预训练的词向量矩阵初始化embeddings。（通过设置weights=[embedding_matrix]）。
    静态(static)方式：训练过程中不再更新embeddings。实质上属于迁移学习，特别是在目标领域数据量比较小的情况下，采用静态的词向量效果也不错。（通过设置trainable=False）
    非静态(non-static)方式：在训练过程中对embeddings进行更新和微调(fine tune)，能加速收敛。（通过设置trainable=True）
    """

    def __init__(self, max_tokens, maxlen, num_class=1, embedding_size=None, weights=None, kernel_size_list=(3, 4, 5)):
        """

        :param embedding_size: 类别/实体嵌入时可不指定

        model = TextCNN(max_token, maxlen, num_class=1)()
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        model.fit_generator(DataIter(X, y), epochs=5)
        """
        self.maxlen = maxlen
        self.class_num = num_class
        self.max_tokens = max_tokens
        self.last_activation = 'softmax' if num_class > 1 else 'sigmoid'
        self.embedding_size = embedding_size if embedding_size else min(50, (max_tokens + 1) // 2)
        self.weights = weights
        self.kernel_size_list = kernel_size_list

    def get_model(self):
        input = Input((self.maxlen,))
        # Embedding part can try multichannel as same as origin paper
        if self.weights:
            e = Embedding(*self.weights.shape, input_length=self.maxlen, weights=self.weights, trainable=False)(input)
        else:
            e = Embedding(self.max_tokens, self.embedding_size, input_length=self.maxlen)(input)
        convs = []
        for kernel_size in self.kernel_size_list:
            c = Conv1D(128, kernel_size, activation='relu')(e)  # 卷积
            # c = Dropout(0.5)(c)
            p = GlobalMaxPool1D()(c)  # 池化
            # p = GlobalAvgPool1D()(c)
            convs.append(p)
        x = Concatenate()(convs)
        output = Dense(self.class_num, activation=self.last_activation)(x)

        model = Model(inputs=input, outputs=output)
        return model


if __name__ == '__main__':
    TextCNN(1000, 10)()
