#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-Python.
# @File         : setup
# @Time         : 2019-06-17 16:12
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


import os
import re
import time
from setuptools import find_packages, setup

# rename
package_name = 'tql'
project_name = 'tql-Python'
version = '0.0.%d' % time.time()
# with open('./%s/__init__.py' % package_name, encoding='utf-8') as f:
#     version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()


def get_requirements():
    _ = './requirements.txt'
    if os.path.isfile(_):
        with open(_, encoding='utf-8') as f:
            return f.read().split()


setup(
    name=package_name,
    version=version,
    url='https://github.com/Jie-Yuan/' + project_name,
    keywords=["tool wheel", "yuanjie", 'utils'],
    description=('description'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='JieYuan',
    author_email='313303303@qq.com',
    maintainer='JieYuan',
    maintainer_email='313303303@qq.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.*']},
    platforms=["all"],
    python_requires='>=3.5',
    classifiers=[
        'Topic :: ML',
        'Topic :: DL',
        'Topic :: CV',
        'Topic :: NLP',

        'Development Status :: Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
    ],

    install_requires=get_requirements()
)
