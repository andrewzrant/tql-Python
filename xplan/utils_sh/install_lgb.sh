#!/usr/bin/env bash
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
cd ../python-package
python setup.py install