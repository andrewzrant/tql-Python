#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'HotSpider'
__author__ = 'JieYuan'
__mtime__ = '2019-05-15'
"""
import requests
import pandas as pd
from lxml.etree import HTML
from fake_useragent import UserAgent


class HotSpider(object):

    def __init__(self, query):
        self.query = query
        self.ua = UserAgent()

    @property
    def df_sites_info(self):
        dfs = []
        for (site, url) in self.df_sites[['site', 'url']].values:
            print('爬取🕷' + site)
            df = pd.read_html(self._request(url).text)[0]
            df.columns = ['rank', 'title', 'hot', 'site']
            df.site = site
            dfs.append(df)
        return pd.merge(self.df_sites, pd.concat(dfs))

    @property
    def df_sites(self):
        url = 'https://tophub.today/search?q='
        r = self._request(url + self.query)
        dom_tree = HTML(r.text)
        sites = dom_tree.xpath('//div[@class="zb-Rb-Cb"]/text()')
        urls = dom_tree.xpath('//div/a[starts-with(@href,"/n")]/@href')
        subscriber_num = dom_tree.xpath('//span[@class="zb-Rb-subscriber-Wc"]/text()')

        df_sites = pd.DataFrame()
        df_sites['site'] = sites
        df_sites['url'] = ['https://tophub.today' + i for i in urls]
        df_sites['subscriber_num'] = [int(i) for i in subscriber_num]
        df_sites.sort_values('subscriber_num', 0, False, True)
        return df_sites

    @property
    def headers(self):
        return {'user-agent': self.ua.random}

    def _request(self, url):
        try:
            r = requests.get(url, timeout=500, headers=self.headers)
            r.raise_for_status()
            r.encoding = 'utf8'  # r.apparent_encoding
            return r

        except Exception as e:
            print('爬取失败：%s' % e)
