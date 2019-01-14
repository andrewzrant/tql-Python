#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'iter'
__author__ = 'JieYuan'
__mtime__ = '18-12-14'
"""
from .utils.xx import xx

import json
import pickle
import jieba
import numpy as np
import pandas as pd

from functools import reduce
from pprint import pprint
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .utils_eda import DataFrameSummary
from .utils import Cprint

try:
    from IPython import get_ipython

    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("console")
except:
    from tqdm import tqdm

else:
    from tqdm import tqdm_notebook as tqdm


# 序列化
# df.to_hdf('./data.h5', 'w', complib='blosc', complevel=8)
def read(fname='./tmp.txt', mode='r'):
    with open(fname, mode) as f:
        for l in f:
            yield l


@xx
def xwrite(iterable, fname, mode='w', glue='\n'):
    with open(fname, mode) as f:
        for item in iterable:
            f.write(str(item) + glue)


@xx
def xpickle_dump(obj, file='tmp.pkl'):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


@xx
def xpickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


# 统计函数: 待补充groupby.agg
xsummary = xx(lambda iterable: DataFrameSummary(list(iterable) | xDataframe)['iterable'])
xvalue_counts = xx(
    lambda iterable, normalize=False, bins=None: pd.value_counts(list(iterable), normalize=normalize, bins=bins))

__funcs = [sum, min, max, abs, len, np.mean, np.median]
xsum, xmin, xmax, xabs, xlen, xmean, xmedian = [xx(i) for i in __funcs]

xnorm = xx(lambda iterable, ord=2: np.linalg.norm(list(iterable), ord))
xcount = xx(lambda iterable: Counter(list(iterable)))

xunique = xx(lambda iterable: list(OrderedDict.fromkeys(list(iterable))))  # 移除列表中的重复元素(保持有序)
xsort = xx(lambda iterable, reverse=False, key=None: sorted(list(iterable), key=key, reverse=reverse))

xmax_index = xx(lambda x: max(range(len(x)), key=x.__getitem__))  # 列表中最小和最大值的索引
xmin_index = xx(lambda x: min(range(len(x)), key=x.__getitem__))  # 列表中最小和最大值的索引
xmost_freq = xx(lambda x: max(set(x), key=x.count))  # 查找列表中频率最高的值, key作用于set(x), 可类推出其他用法

# print
xprint = xx(lambda obj, bg='blue': Cprint().cprint(obj, bg))
xtqdm = xx(lambda iterable, desc=None: tqdm(iterable, desc))

# base types
xtuple, xlist, xset = xx(tuple), xx(list), xx(set)

# string
xjoin = xx(lambda string, sep=' ': sep.join(string))
xcut = xx(lambda string, cut_all=False: jieba.cut(string, cut_all=cut_all))

# list transform
xgroup_by_step = xx(lambda ls, step=3: [ls[idx: idx + step] for idx in range(0, len(ls), step)])


# dict
@xx
def xjson(dict_):
    _ = json.dumps(dict_, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)
    return _


@xx
def xSeries(iterable, name='iterable'):
    if isinstance(iterable, pd.Series):
        return iterable
    else:
        return pd.Series(iterable, name=name)


@xx
def xDataframe(iterable, name='iterable'):
    if isinstance(iterable, pd.DataFrame):
        return iterable
    else:
        return pd.DataFrame({name: iterable})


# 高阶函数
xmap = xx(lambda iterable, func: map(func, iterable))
xreduce = xx(lambda iterable, func: reduce(func, iterable))
xfilter = xx(lambda iterable, func: filter(func, iterable))


# multiple
@xx
def xThreadPoolExecutor(iterable, func, max_workers=5):
    with ThreadPoolExecutor(max_workers) as pool:
        return pool.map(func, iterable)


@xx
def xProcessPoolExecutor(iterable, func, max_workers=5):
    with ProcessPoolExecutor(max_workers) as pool:
        return pool.map(func, iterable)
