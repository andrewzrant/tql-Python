#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'cprint'
__author__ = 'JieYuan'
__mtime__ = '19-1-10'
"""
from pprint import pprint


class Cprint(object):
    def __init__(self, p=True):
        colors = ['green', 'yellow', 'black', 'cyan', 'blue', 'red', 'white', 'purple']
        self.p = p
        self.fc = dict(zip(colors, range(40, 97)))
        self.bc = dict(zip(colors, range(90, 97)))

        print("Colors: \033[94m%s\033[0m\n" % colors)

    def cprint(self, s='Hello World !', bg='blue', fg='', mode=1):
        """
        :param s: string
        :param fg/bg: foreground/background
            'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'
        :param mode:
            0（默认值）
            1（高亮）
            22（非粗体）
            4（下划线）
            24（非下划线）
            5（闪烁）
            25（非闪烁）
            7（反显）
            27（非反显）
            https://www.cnblogs.com/hellojesson/p/5961570.html
        :return:
        """
        if fg:
            _ = '\033[%s;%s;%sm%s\033[0m' % (mode, self.fc[fg], self.bc[bg], s)
        else:
            _ = '\033[%s;%sm%s\033[0m' % (mode, self.bc[bg], s)
        if self.p:
            pprint(_)
        else:
            return _


if __name__ == '__main__':
    Cprint().cprint()
