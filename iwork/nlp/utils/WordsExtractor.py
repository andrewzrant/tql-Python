#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'StopWords'
__author__ = 'JieYuan'
__mtime__ = '2019-05-16'
"""
import jieba.posseg as jp


class WordsExtractor(object):

    def words(self, sent, flags=None):
        """
        :flags mode: ['v', 'vn']
        """
        for p in jp.cut(sent):
            if p.flag in flags:
                yield p.word

    def noun(self, sent):
        stopwords = self.stopwords
        for p in jp.cut(sent):
            if 'n' in p.flag and len(p.word) > 1 and p.word not in stopwords:
                yield p.word

    @property
    def stopwords(self):
        with open('./stop_words.txt') as f:
            return set(f.read().split())


if __name__ == '__main__':
    we = WordsExtractor()
    _ = we.noun('我是中国人')
    print(list(_))