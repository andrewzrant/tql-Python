#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : TextClean
# @Time         : 2019-06-21 18:19
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


# TODO:
# def remove_special_characters(text):
#     tokens = tokenize_text(text)
#     pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
#     filtered_tokens = filter(None, [re.sub(pattern=pattern, repl="", string=token) for token in tokens])
#     filtered_text = ''.join(filtered_tokens)
#     return filtered_text
#
#
# def normalize_corpus(corpus):
#     normalized_corpus = []
#     for text in corpus:
#         text = remove_special_characters(text)
#         text = remove_stopwords(text)
#         normalized_corpus.append(text)
#     return normalized_corpus
