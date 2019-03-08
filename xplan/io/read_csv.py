#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'read_csv'
__author__ = 'JieYuan'
__mtime__ = '19-3-8'
"""

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from functools import partial


def read_files(files, **kwargs):
    """
    换行符 lineterminator='\n'
    :param files:
    :param kwargs:
    :return:
    """
    read_func = partial(pd.read_csv, **kwargs)
    return list(map(read_func, tqdm(files)))
