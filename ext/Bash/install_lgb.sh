#!/usr/bin/env bash
# git@v9.git.n.xiaomi.com:yuanjie/LightGBM.git
pip uninstall lightgbm -y
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
cd ../python-package
python setup.py install

# gpu
# apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev
