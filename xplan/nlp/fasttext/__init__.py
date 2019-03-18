#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '19-3-15'
"""
"""
https://radimrehurek.com/gensim/models/fasttext.html#module-gensim.models.fasttext
"""
import jieba

jieba.lcut('测试')
from gensim.models import FastText
from tqdm import tqdm_notebook


# FastText.load_fasttext_format 可加载C++版模型

def corpus_iter(file=None, tokenize=jieba.lcut):
    with open(file, encoding='utf-8') as f:
        for line in tqdm_notebook(f):
            yield tokenize(line)


model = FastText(
    size=128,
    alpha=0.025,
    window=10,
    min_count=10,
    negative=5,
    min_n=1,
    max_n=6,
    workers=8)

file = './demo.txt'
model.build_vocab(corpus_iter(file))
model.train(sentences=corpus_iter(file), total_examples=model.corpus_count, epochs=20)
