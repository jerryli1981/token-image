# -*- coding: utf-8 -*-

import os
import time
import jieba
from bs4 import BeautifulSoup as bs

from pypinyin import lazy_stroke, pinyin
import pypinyin

from progressbar import ProgressBar

import re

import codecs

import itertools


Dir = "./data/SogouCAS"

sport_url = set()
finance_url = set()
ent_url = set()
auto_url = set()
tech_url  = set()

Dic = {'Ａ':'A','Ｂ':'B','Ｃ':'C','Ｄ':'D','Ｅ':'E','Ｆ':'F','Ｇ':'G','Ｈ':'H','Ｉ':'I','Ｊ':'J','Ｋ':'K','Ｌ':'L','Ｍ':'M','Ｎ':'N','Ｏ':'O','Ｐ':'P','Ｑ':'Q',\
'Ｒ':'R','Ｓ':'S','Ｔ':'T','Ｕ':'U','Ｖ':'V','Ｗ':'W','Ｘ':'X','Ｙ':'Y','Ｚ':'Z','ａ':'a','ｂ':'b','ｃ':'c','ｄ':'d','ｅ':'e','ｆ':'f','ｇ':'g','ｈ':'h','ｉ':'i','ｊ':'j',\
'ｋ':'k','ｌ':'l','ｍ':'m','ｎ':'n','ｏ':'o','ｐ':'p','ｑ':'q','ｒ':'r','ｓ':'s','ｔ':'t','ｕ':'u','ｖ':'v','ｗ':'w','ｘ':'x','ｙ':'y','ｚ':'z','１':'1','２':'2','３':'3',\
'４':'4','５':'5','６':'6','７':'7','８':'8','９':'9','０':'0','．':'.'}

dot = {'。','（','）','！','「','」','，','、','；','：','”','“','～',\
        '＜','＞','．','é','︶','『','』','﹗','ī','ō','／',"〔", '〕','｜',\
        "？","＠","｛","｝","￥","《","》",'…','【','】','︿','＃','＄','％','＆','＊','＋','⊙','［','］',\
       "［","］","—","·","－"}

def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def seg(content):

    content = content.encode('utf-8')
    content = replace_all(content,Dic)

    Cut=jieba.cut(''.join(content.split())) #斷詞
    li = []
    for u in Cut:
        if u.encode('utf-8') not in dot:  #清洗全形符號
            li.append(u) #沒在dot裡的就寫進陣列

    content = (' '.join(li)).encode('utf-8') #將陣列用空白格開, 傳回字串

    return content, len(li)



if __name__ == "__main__":

    totfiles = len(os.listdir(Dir))
    pbar = ProgressBar(maxval=totfiles).start()

    sport_cnt = 0
    ent_cnt = 0
    auto_cnt = 0
    fin_cnt = 0
    it_cnt = 0

    num_train = 0
    num_test = 0

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--format",dest="format",type=str,default=None)
    args = parser.parse_args()

    if args.format == "stk":
        transfer = lazy_stroke
        error="ignore"
        style = None
    elif args.format == "py":
        transfer = pinyin
        style = pypinyin.TONE2
        error="ignore"
    else:
        raise("Wrong format")

    with open("./data/train_"+args.format+".csv", 'w') as tr, open("./data/test_"+args.format+".csv", 'w') as te:

        for t, name in enumerate(os.listdir(Dir)): 
            if name.startswith('.'):
                continue
            time.sleep(0.01)
            pbar.update(t + 1)
            From = Dir + '/' + name
            with codecs.open(From, encoding='gbk', errors='ignore') as f:

                soup = bs(f.read(),"html.parser")
                for i, j in zip(soup.select('url'), soup.select('content')): 

                    url_mention = i.text
                    content = j.text

                    if content == '' or url_mention == '':
                        continue    

                    if "http://sports." in url_mention and url_mention not in sport_url:
                        sport_url.add(url_mention)

                        content, seqlen = seg(content)

                        if seqlen > 30:
                            sport_cnt += 1
                            if sport_cnt <= 50000:

                                content = content.decode('utf-8')
                                toks = content.split(" ")

                                pys = transfer(toks, style=style, errors=error)

                                pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')

                                exp = "\"1\"" + str(',') + "\""+pys+"\""

                                if sport_cnt % 10 !=0:
                                    tr.write(exp+"\n")
                                    num_train += 1
                                else:
                                    te.write(exp+"\n")
                                    num_test += 1
                                    

                    elif "http://ent." in url_mention and url_mention not in ent_url:
                        ent_url.add(url_mention)

                        content, seqlen = seg(content)

                        if seqlen > 30:
                            ent_cnt += 1

                            if ent_cnt <= 50000:

                                content = content.decode('utf-8')
                                toks = content.split(" ")

                                pys = transfer(toks, style=style, errors=error)
                                pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')
                                exp = "\"2\"" + str(',') + "\""+pys+"\""

                                if ent_cnt % 10 !=0:
                                    tr.write(exp+"\n")
                                    num_train += 1
                                else:
                                    te.write(exp+"\n")
                                    num_test += 1


                    elif "http://auto." in url_mention and url_mention not in auto_url:
                        auto_url.add(url_mention)

                        
                        content, seqlen = seg(content)

                        if seqlen > 30:
                            auto_cnt += 1

                            if auto_cnt <= 50000:

                                content = content.decode('utf-8')

                                toks = content.split(" ")
                                pys = transfer(toks, style=style, errors=error)
                                pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')
                                exp = "\"3\"" + str(',') + "\""+pys+"\""

                                if auto_cnt % 10 !=0:
                                    tr.write(exp+"\n")
                                    num_train += 1
                                else:
                                    te.write(exp+"\n")
                                    num_test += 1
       

                    elif "http://finance." in url_mention and url_mention not in finance_url:
                        finance_url.add(url_mention)

                        
                        content, seqlen = seg(content)

                        if seqlen > 30:
                            fin_cnt += 1

                            if fin_cnt <= 50000:

                                content = content.decode('utf-8')

                                toks = content.split(" ")
                                pys = transfer(toks, style=style, errors=error)
                                pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')
                                exp = "\"4\"" + str(',') + "\""+pys+"\""


                                if fin_cnt % 10 !=0:
                                    tr.write(exp+"\n")
                                    num_train += 1
                                else:
                                    te.write(exp+"\n")
                                    num_test += 1


                    elif ("http://tech." in url_mention or "http://it." in url_mention) and url_mention not in tech_url:
                        tech_url.add(url_mention)
                        content, seqlen = seg(content)
                        if seqlen > 30:
                            it_cnt += 1
                            if it_cnt <= 50000:

                                content = content.decode('utf-8')

                                toks = content.split(" ")
                                pys = transfer(toks, style=style, errors=error)
                                pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')
                                exp = "\"5\"" + str(',') + "\""+pys+"\""

                                if it_cnt % 10 !=0:
                                    tr.write(exp+"\n")
                                    num_train += 1
                                else:
                                    te.write(exp+"\n")
                                    num_test += 1

                

                                
    pbar.finish()

    print '\n---------------'
    print 'Sport_cnt',"\t",sport_cnt
    print 'Fin_cnt',"\t",fin_cnt
    print 'Ent_cnt',"\t",ent_cnt
    print 'Auto_cnt',"\t",auto_cnt
    print 'Tech_cnt',"\t",it_cnt

    print 'Train_cnt',"\t",num_train
    print 'Test_cnt',"\t",num_test
    print '---------------\n'

