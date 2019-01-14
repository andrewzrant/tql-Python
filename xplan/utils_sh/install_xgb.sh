#!/usr/bin/env bash
git clone --recursive https://github.com/dmlc/xgboost; cd xgboost
mkdir build; cd build
cmake .. -DUSE_CUDA=ON
make -j4
cd python-package
python setup.py install