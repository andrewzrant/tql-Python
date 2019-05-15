#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'word_cloud'
__author__ = 'JieYuan'
__mtime__ = '19-3-1'
"""

from pyecharts import charts
import random


class WordCloud(object):

    def __init__(self, data_pair, shape=None):
        """
        wc.render()
        wc.render_notebook()
        """
        shapes = ['circle', 'cardioid', 'diamond', 'triangle-forward', 'triangle', 'pentagon', 'star']
        self.data_pair = data_pair
        self.shape = shape if shape else random.choice(shapes)

    @property
    def wc(self):
        wc = charts.WordCloud()
        wc.add("WordCloud", data_pair=self.data_pair, shape=self.shape)
        return wc


if __name__ == '__main__':
    pairs = [('中国', 33),
             ('苹果', 24),
             ('奚梦瑶', 20),
             ('美国', 16),
             ('特朗普', 16),
             ('何猷君', 15),
             ('戛纳', 13),
             ('红毯', 12),
             ('iPhone', 12),
             ('车队', 9),
             ('车祸', 9),
             ('优衣', 9),
             ('信息', 9),
             ('李亚鹏', 9),
             ('恋情', 9),
             ('任素', 9),
             ('男孩', 9),
             ('亚洲', 8),
             ('孩子', 8),
             ('大学生', 8)]

    WordCloud(pairs).wc.render_notebook()
    WordCloud(pairs).wc.render()
