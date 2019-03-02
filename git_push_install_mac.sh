#!/usr/bin/env bash
git pull
git add *
git commit -m "mac update"
git push
pip uninstall x-plan -y \
&& pip install git+https://github.com/Jie-Yuan/X-plan.git -i https://pypi.tuna.tsinghua.edu.cn/simple