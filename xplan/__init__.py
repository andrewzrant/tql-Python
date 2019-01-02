#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '18-12-14'
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, train_test_split

try:
    from IPython import get_ipython

    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("console")
except:
    from tqdm import tqdm

else:
    from tqdm import tqdm_notebook as tqdm
