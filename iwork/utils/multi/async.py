#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
__title__ = 'async'
__author__ = 'JieYuan'
__mtime__ = '2019-05-22'
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from aiohttp import ClientSession

"""
TODO: 多进程+协程
"""


class Async(object):
    """https://www.cnblogs.com/shenh/p/9090586.html"""

    def __init__(self, concurrent_number=500):
        self.response = None
        self.loop = asyncio.get_event_loop()
        # self.loop.set_default_executor(ProcessPoolExecutor(10))
        self.semaphore = asyncio.Semaphore(concurrent_number)

    def gets(self, urls):
        tasks = map(lambda url: asyncio.ensure_future(self._get(url)), urls)
        return self.loop.run_until_complete(asyncio.gather(*tasks))  # 收集响应

    async def _get(self, url):
        async with self.semaphore:
            async with ClientSession() as session:
                async with session.get(url) as response:
                    print('Hello World:%s' % time.time())
                    return await response.text()


if __name__ == '__main__':
    a = Async()

    # urls = ['https://www.baidu.com/1', 'https://www.baidu.com/1'] * 1000
    urls = ["""http://dev.web.algo.browser.miui.srv/nlp/cut/jieba?text=不知道"""]*4000
    r = a.gets(urls)
    print(r)

