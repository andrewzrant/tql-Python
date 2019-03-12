# -*- coding: utf-8 -*-
"""
__title__ = 'time_utils'
__author__ = 'JieYuan'
__mtime__ = '2018/7/27'
"""
import time
import datetime
import pandas as pd


# t = pd.datetime.now().timestamp()
# pd.datetime.today().timestamp()
# pd.read_csv(parse_dates)


# 时间戳 转 时间字符串
def timestamp2str(timestamp, format='%Y-%m-%d %H:%M:%S'):
    """
    t = pd.datetime.now().timestamp()
    ts = pd.Series([t]*10, name='t')

    # 时间戳 转 时间字符串
    ts = ts.map(timestamp2str)

    # 时间字符串 转 时间
    ts = pd.to_datetime(ts, errors='coerce', infer_datetime_format=True)

    # 时间 转 时间戳
    ts.map(lambda x: x.timestamp())
    """
    return time.strftime(format, time.localtime(timestamp))
