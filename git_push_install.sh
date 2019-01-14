#!/usr/bin/env bash
while getopts ":m:b:c:" opt
do
    case $opt in
        m)
        export m=$OPTARG;;
        b)
        echo "参数b的值$OPTARG";;
        c)
        echo "参数c的值$OPTARG";;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

git pull
git add *
git commit -m $m
git push
pip uninstall x-plan -y && pip install git+https://github.com/Jie-Yuan/X-plan.git -U --user