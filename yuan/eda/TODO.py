#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'TODO'
__author__ = 'JieYuan'
__mtime__ = '2019/4/17'
"""

import pandas as pd

df = pd.DataFrame()

# 缺失值占比倒排
df.isnull().sum().sort_values(0, False) / df.shape[0]

# 方差/去重类别数
df.nunique(dropna=False).sort_values(0, False)

# 相关性
