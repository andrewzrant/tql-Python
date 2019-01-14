#!/usr/bin/env bash
# git@v9.git.n.xiaomi.com:yuanjie/LightGBM.git
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
cd ../python-package
python setup.py install