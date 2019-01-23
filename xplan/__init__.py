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
