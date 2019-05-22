#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pyhive'
__author__ = 'JieYuan'
__mtime__ = '2019-05-21'
"""

from pyhive import hive


class Sql(object):

    def __init__(self, conn='hive'):
        """Hive Server List: http://infra.d.xiaomi.net/hive/services.html"""
        self.conn = self.__getattribute__('_conn_%s' % conn)

    @property
    def _conn_hive(self):
        conn = hive.connect(
            # host='c3prc-hadoop.hive.srv', port=10000,  # c3
            host="zjyprc-hadoop.hive.srv", port=10000,  # zjy
            auth="KERBEROS", kerberos_service_name="sql_prc",
            configuration={
                'mapreduce.map.memory.mb': '4096',
                'mapreduce.reduce.memory.mb': '4096',
                'mapreduce.map.java.opts': '-Xmx3072m',
                'mapreduce.reduce.java.opts': '-Xmx3072m',
                'hive.input.format': 'org.apache.hadoop.hive.ql.io.HiveInputFormat',
                'hive.limit.optimize.enable': 'false',
                'mapreduce.job.queuename': 'production.miui_group.o2o.o2o_zjy_online',  # zjy
            },
        )
        return conn

    @property
    def _conn_spark(self):
        # Spark SQL Thrift JDBC Server
        conn = hive.connect(
            host='c3prc-hadoop.spark-sql.hadoop.srv', port=10000,  # zjyprc-hadoop.spark-sql.hadoop.srv
            auth="KERBEROS", kerberos_service_name="sql_prc",
            configuration={
                'mapreduce.job.queuename': 'production.miui_group.o2o.o2o_zjy_online',
            },
        )
        return conn


if __name__ == '__main__':
    import pandas as pd

    s = Sql()
    sql = "SHOW tables"
    df = pd.read_sql(sql, s.conn)
    df.tail()
