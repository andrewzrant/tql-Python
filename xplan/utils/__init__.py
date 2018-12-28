#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '18-12-17'
"""
import os
from sklearn.utils import shuffle
from .download_file import DownloadFile

base_dir = os.path.dirname(os.path.realpath('__file__'))
get_module_path = lambda path, file: os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(file), path))
