#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""从汉典网按 unicode 编码获取汉字，拼音，五笔"""

import io
import re
import sys
from time import sleep
import time

from bs4 import BeautifulSoup
import requests

from progressbar import ProgressBar

from pypinyin.pinyin_dict import pinyin_dict

class Message(object):
    def __init__(self, file_name):
        self.f = io.open(file_name, 'w', encoding='utf8')

    def write(self, msg):
        self.f.write(u'%s\n' % msg)

    def __del__(self):
        self.f.close()

# sys.stderr = Message('error.txt')
# sys.stdout = Message('info.txt')


def request(url, headers, cookies):
    r = requests.get(url, headers=headers, cookies=cookies)
    if r.ok:
        cookies.update(r.cookies.get_dict())
        return r.text
    else:
        print >> sys.stderr, url

def unicode2int(word):
    c=repr(word)
    if '\u' not in c:
        return 0
    else:
        d= c.translate(None,r"\u'")
        return int(d,16)


def parse_word_url(html):
    soup = BeautifulSoup(html)
    tag_a = soup.select('li a.usual')
    if not tag_a:
        return
    a = tag_a[0]
    unicode_num = a.select('span')[0].text
    url = u'http://www.zdic.net' + a.attrs.get('href')
    return unicode_num.strip(), url.strip()


def parse_wb(html):
    soup = BeautifulSoup(html)
    word_html = soup.find(id='ziip').text.encode(
        'raw_unicode_escape'
    ).decode('utf8')
    words = re.findall(ur'“([^”]+)”', word_html)
    word = words[0] if words else ''

    try:

        wb = soup.select('td.z_i_t4')[0].text
        wb = wb.encode('raw_unicode_escape').decode('utf8')

    except Exception as e:
        e.word = word
        raise

    return word, wb

def get_word(n, url_base, headers, cookies):
    url = url_base % '{0:x}'.format(n)
    #print hex(n)
    try:
        html = request(url, headers, cookies)
        unicode_num, url = parse_word_url(html)
        html = request(url, headers, cookies)
        word, wb = parse_wb(html)
        # print unicode_num, repr(word), pinyins
        return word, wb
    except Exception as e:
        print e
        return '{0:x}'.format(n).upper(), getattr(e, 'word', ''), []

def main():

    url_base = 'http://www.zdic.net/sousuo/ac/?q=%s&tp=tp2&lb=uno'
    headers = {
        'Host': 'www.zdic.net',
        'User-Agent': ('Mozilla/5.0 (Windows NT 6.2; rv:26.0) Gecko/20100101'
                       'Firefox/26.0'),
        'Accept': 'text/javascript, text/html, application/xml, text/xml, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': 1,
        'X-Requested-With': 'XMLHttpRequest',
        'X-Prototype-Version': '1.5.0',
        'Referer': 'http://www.zdic.net/',
        'Connection': 'keep-alive'
    }
    cookies = {}

    count = 0
    with open("./data/dict.txt", "rb") as dict:
        for word in dict:
            count +=1


    pbar = ProgressBar(maxval=count).start()


    with io.open("./data/dict_pw.txt", 'w', buffering=1, encoding='utf8') as f:
        with open("./data/dict.txt", "rb") as dict:
            for i, word in enumerate(dict):
                time.sleep(0.01)
                pbar.update(i + 1)
                word = word.strip().decode('utf-8')
                n = unicode2int(word)
                word, wb = get_word(n, url_base, headers, cookies)
                #print word+"\t"+wb
                f.write(word+"\t"+wb+"\n")

    pbar.finish()

if __name__ == '__main__':
    main()
