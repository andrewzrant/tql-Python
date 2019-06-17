#!/usr/bin/env bash
git pull
git add *
git commit -m 'update'
git push

#
cd ./xx
git init
git add * && git commit -m 'add xx'

git remote remove origin
git remote add origin git@github.com:Jie-Yuan/xx.git
git push origin master -f