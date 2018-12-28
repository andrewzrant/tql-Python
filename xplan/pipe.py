#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'iter'
__author__ = 'JieYuan'
__mtime__ = '18-12-14'
"""
from .utils.x import X

import json
import jieba
import numpy as np
import pandas as pd

from functools import reduce
from pprint import pprint
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .utils_eda import DataFrameSummary

try:
    from IPython import get_ipython

    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("console")
except:
    from tqdm import tqdm

else:
    from tqdm import tqdm_notebook as tqdm

# 统计函数: 待补充groupby.agg
xsummary = X(lambda iterable: DataFrameSummary(iterable | xDataframe)['iterable'])
xvalue_counts = X(lambda iterable, bins=None: pd.value_counts(iterable, bins=bins))

__funcs = [sum, min, max, abs, len, np.mean, np.median]
xsum, xmin, xmax, xabs, xlen, xmean, xmedian = [X(i) for i in __funcs]

xnorm = X(lambda iterable, ord=2: np.linalg.norm(iterable, ord))
xcount = X(lambda iterable: Counter(iterable))

xunique = X(lambda iterable: list(OrderedDict.fromkeys(iterable)))  # 移除列表中的重复元素(保持有序)
xsort = X(lambda iterable, reverse=False, key=None: sorted(iterable, key=key, reverse=reverse))

xmax_index = X(lambda x: max(range(len(x)), key=x.__getitem__))  # 列表中最小和最大值的索引
xmin_index = X(lambda x: min(range(len(x)), key=x.__getitem__))  # 列表中最小和最大值的索引
xmost_freq = X(lambda x: max(set(x), key=x.count))  # 查找列表中频率最高的值, key作用于set(x), 可类推出其他用法

# print
xprint = X(pprint)
xtqdm = X(lambda iterable, desc=None: tqdm(iterable, desc))

# base types
xtuple, xlist, xset = X(tuple), X(list), X(set)

# string
xjoin = X(lambda string, sep=' ': sep.join(string))
xcut = X(lambda string, cut_all=False: jieba.cut(string, cut_all=cut_all))

# dict
@X
def xjson(dict_):
    _ = json.dumps(dict_, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)
    return _

@X
def xSeries(iterable, name='iterable'):
    if isinstance(iterable, pd.Series):
        return iterable
    else:
        return pd.Series(iterable, name=name)


@X
def xDataframe(iterable, name='iterable'):
    if isinstance(iterable, pd.DataFrame):
        return iterable
    else:
        return pd.DataFrame({name: iterable})


# 高阶函数
xmap = X(lambda iterable, func: map(func, iterable))
xreduce = X(lambda iterable, func: reduce(func, iterable))
xfilter = X(lambda iterable, func: filter(func, iterable))


# multiple
@X
def xThreadPoolExecutor(iterable, func, max_workers=5):
    with ThreadPoolExecutor(max_workers) as pool:
        return pool.map(func, iterable)


@X
def xProcessPoolExecutor(iterable, func, max_workers=5):
    with ProcessPoolExecutor(max_workers) as pool:
        return pool.map(func, iterable)