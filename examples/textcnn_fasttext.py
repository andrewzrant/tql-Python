#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : textcnn_fasttext
# @Time         : 2019-06-23 19:10
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


from tql.pipe import *
from tql.nlp.utils import Text2SequenceByFastText
from tql.nn.keras.utils import DataIter
from tql.nn.keras.models import TextCNN
from gensim.models.fasttext import load_facebook_model
jieba.initialize()

fasttext = load_facebook_model('fasttext.model')
df = pd.read_csv('../../../fds/data/sentiment.tsv.zip', '\t')
ts = Text2SequenceByFastText(fasttext_model=fasttext, tokenizer=jieba.cut)