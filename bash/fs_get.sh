#!/bin/bash
echo "hdfs路径：$1";
echo "本地路径: $2";
echo "hdfs --cluster zjyprc-hadoop dfs -get $1 $2"
echo $line
hdfs --cluster zjyprc-hadoop dfs -get $1 $2
