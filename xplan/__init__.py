#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '18-12-14'
"""

# pd.set_option('display.max_rows', 1024)
# pd.set_option('display.max_columns', 128)
# pd.set_option('max_colwidth', 128)  # 列宽
# pd.set_option('expand_frame_repr', False)  # 允许换行显示

import re
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, IsolationForest
from sklearn.metrics import *
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Simhei']  # 中文乱码的处理
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 负号
plt.rcParams["text.usetex"] = False
plt.rcParams["legend.numpoints"] = 1
plt.rcParams["figure.figsize"] = (12, 6)  # (8, 6)
plt.rcParams["figure.dpi"] = 128
plt.rcParams["savefig.dpi"] = plt.rcParams["figure.dpi"]
plt.rcParams["font.size"] = 10
plt.rcParams["pdf.fonttype"] = 42

import seaborn as sns
sns.set(style="darkgrid") # darkgrid, whitegrid, dark, white,和ticks
# sns.plotting_context()
# sns.axes_style()




import warnings

warnings.filterwarnings("ignore")
try:
    from IPython import get_ipython

    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("console")
except:
    from tqdm import tqdm

else:
    from tqdm import tqdm_notebook as tqdm
