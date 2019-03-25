#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'demo'
__author__ = 'JieYuan'
__mtime__ = '19-3-22'
"""

from pprint import pprint

import requests
import itchat

url = 'http://www.tuling123.com/openapi/api?key=3ac26126997942458c0d93de30d52212&info='


# len(msg) == 34 # isGroupChat
# len(msg) == 31 # isFriendChat/isMpChat

@itchat.msg_register(['Text'], isGroupChat=True)
def text_reply(msg):
    username = msg['User']['NickName']
    question = msg['Text']
    if isinstance(question, str) and username == '南京小米资讯算法':
        username = msg['ActualNickName']
        print(username, '1')

        print(msg['User']['RemarkName'], '2')

        print(question, '3')

        answer = requests.get(url + question).json()['text']
        answer = f'@机器人\n{answer}'
        itchat.send(answer, msg['FromUserName'])


# @itchat.msg_register(['Text'], isFriendChat=True)
# def text_reply(msg):
#     username = msg['User']['NickName']
#     question = msg['Text']
#
#     if isinstance(question, str) and '饕餮' in username:
#         answer = requests.get(url + question).json()['text']
#         answer = f'@你的小可爱\n{answer}'
#         itchat.send(answer, msg['FromUserName'])


if __name__ == '__main__':
    itchat.auto_login()
    itchat.run()
