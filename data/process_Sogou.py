# -*- coding: utf-8 -*-

import sys,os
import time
import jieba
from bs4 import BeautifulSoup as bs

sys.path.insert(0, "../")

from pypinyin import stroke, pinyin, wubi
import pypinyin

from progressbar import ProgressBar

import re

import codecs

import itertools

import random

import math


unicode_ranges = (
        ('2E80', '2EFF'),     # CJK 部首扩展:[2E80-2EFF]
        ('2F00', '2FDF'),     # CJK 康熙部首:[2F00-2FDF]
        ('31C0', '31EF'),     # CJK 笔画:[31C0-31EF]
        ('3400', '4DBF'),     # CJK 扩展 A:[3400-4DBF]
        ('4E00', '9FFF'),     # CJK 基本:[4E00-9FFF]
        ('F900', 'FAFF'),     # CJK 兼容:[F900-FAFF]
        ('20000', '2A6DF'),   # CJK 扩展 B:[20000-2A6DF]
        ('2A700', '2B73F'),   # CJK 扩展 C:[2A700-2B73F]
        ('2B740', '2B81D'),   # CJK 扩展 D:[2B740-2B81D]
        ('2F800', '2FA1F'),   # CJK 兼容扩展:[2F800-2FA1F]
    )

uniset = set()
for unicode_range in unicode_ranges:
    for n in xrange(int(unicode_range[0], 16), int(unicode_range[1], 16) + 1):
        uniset.add(n)



Dic = {'Ａ':'A','Ｂ':'B','Ｃ':'C','Ｄ':'D','Ｅ':'E','Ｆ':'F','Ｇ':'G','Ｈ':'H','Ｉ':'I','Ｊ':'J','Ｋ':'K','Ｌ':'L','Ｍ':'M','Ｎ':'N','Ｏ':'O','Ｐ':'P','Ｑ':'Q',\
'Ｒ':'R','Ｓ':'S','Ｔ':'T','Ｕ':'U','Ｖ':'V','Ｗ':'W','Ｘ':'X','Ｙ':'Y','Ｚ':'Z','ａ':'a','ｂ':'b','ｃ':'c','ｄ':'d','ｅ':'e','ｆ':'f','ｇ':'g','ｈ':'h','ｉ':'i','ｊ':'j',\
'ｋ':'k','ｌ':'l','ｍ':'m','ｎ':'n','ｏ':'o','ｐ':'p','ｑ':'q','ｒ':'r','ｓ':'s','ｔ':'t','ｕ':'u','ｖ':'v','ｗ':'w','ｘ':'x','ｙ':'y','ｚ':'z','１':'1','２':'2','３':'3',\
'４':'4','５':'5','６':'6','７':'7','８':'8','９':'9','０':'0','．':'.','”':'"','“':'"','。':'.','！':'!',\
'，':',','、':',','；':';','：':':','～':'~','＜':'<','＞':'>','﹗':'!','／':'/',"〔":'[', '〕':']','｜':'|',\
        "？":'?',"＠":'@',"｛":'{',"｝":'}',"￥":'$',"《":'<',"》":'>','…':'...','【':'[','】':']','︿':'^','＃':'#','＄':'$','％':'%',\
         '＆':'&','＊':'*','＋':'+','⊙':'*','［':'[','］':']',"［":'[',"］":']',"—":'-',"·":'.',"－":'-'}


dot = {'é','︶','『','』','ī','ō','︿','⊙','［','］'}

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

def unicode2int(word):
    c=repr(word)
    if '\u' not in c:
        return 0
    else:
        d= c.translate(None,r"\u'")
        return int(d,16)

def transEscape(content):
    content =re.sub(r'\n', "\\\\n", content)
    content =re.sub(r'\"', "\"\"", content)
    return content

def transformation(content, label, wordDict):

    content, seqlen = seg(content)
    if seqlen < 5:
        return None

    content = content.decode('utf-8')
    toks = content.split(" ")

    for tok in toks:
        for word in tok:
            idx = unicode2int(word)
            if word not in vocab and idx in uniset:
                vocab.add(word)
                wordDict.write(word.encode("utf-8") + '\n')  

    pys = transfer(toks, style=style, errors=error)
    pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')
    pys = transEscape(pys)

    exp = "\""+label+"\"" + str(',') + "\""+pys+"\""
    return exp

"""
http://meirong. http://yule. http://sports. http://caifu. http://house.
http://tour. http://career. http://sina. http://travel. http://auto.
http://money. http://ent. http://mil. http://military. http://finance.
http://tech. http://learning. http://it. http://lady. http://women.
http://health. http://edu. http://cul. http://business. http://www.
http://eladies. http://2008. http://fun. http://news. http://war.
http://china. http://culture.
"""


if __name__ == "__main__":

    Dir = "./SogouCAS"

    sport_list = []
    ent_list = []
    auto_list = []
    fin_list = []
    it_list = []

    vocab = set()
    
    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--format",dest="format",type=str,default=None)
    args = parser.parse_args()

    if args.format == "stk":
        transfer = stroke
        error="ignore"
        style = None
    elif args.format == "py":
        transfer = pinyin
        style = pypinyin.TONE2
        error="default"
    elif args.format == "wb":
        transfer = wubi
        style = None
        error="ignore"
    else:
        raise("Wrong format")

    with open("./train_"+args.format+".csv", 'w') as tr, open("./test_"+args.format+".csv", 'w') as te, \
        open("./dict.txt", 'w')as wordDict:

        totfiles = len(os.listdir(Dir))
        pbar = ProgressBar(maxval=totfiles).start()

        for t, name in enumerate(os.listdir(Dir)): 
            if name.startswith('.'):
                continue

            time.sleep(0.01)
            pbar.update(t + 1)

            From = Dir + '/' + name
            with codecs.open(From, encoding='gbk', errors='ignore') as f:
                soup = bs(f.read(),"html.parser")

                for i, j, k, l in zip(soup.select('url'), soup.select('content'), 
                    soup.select('contenttitle'), soup.select('docno')): 

                    url = i.text
                    content = j.text
                    title = k.text
                    docno = l.text

                    if url == '' or content == '':
                        continue    

                    if "http://sports." in url:

                        if content != '':

                            exp = transformation(content, "1", wordDict)
                            if exp != None:
                                sport_list.append(exp)


                    elif "http://ent." in url or "http://yule." in url or "http://fun" in url:

                        if content != '':

                            exp = transformation(content, "2", wordDict)
                            if exp != None:
                                ent_list.append(exp)

                    elif "http://auto." in url or "auto" in url:

                        if content != '':

                            exp = transformation(content, "3", wordDict)
                            if exp != None:
                                auto_list.append(exp)

                    elif "http://finance." in url:

                        if content != '':

                            exp = transformation(content, "4", wordDict)
                            if exp != None:
                                fin_list.append(exp)

                    elif "http://tech." in url or "http://it." in url:

                        if content != '':

                            exp = transformation(content, "5", wordDict)
                            if exp != None:
                                it_list.append(exp)

                    
        pbar.finish()

        print '\n---------------'
        num_sports = len(set(sport_list))
        num_fin = len(set(fin_list))
        num_ent = len(set(ent_list))
        num_auto = len(set(auto_list))
        num_it = len(set(it_list))
        print 'Sport_cnt',"\t",num_sports
        print 'Fin_cnt',"\t",num_fin
        print 'Ent_cnt',"\t",num_ent
        print 'Auto_cnt',"\t",num_auto
        print 'Tech_cnt',"\t",num_it
        print 'Word Size', "\t", len(vocab)

        num_samples = min([num_sports, num_fin, num_ent, num_auto, num_it])
        sports_samples = random.sample(set(sport_list), num_samples)
        ent_samples = random.sample(set(ent_list), num_samples)
        auto_samples = random.sample(set(auto_list), num_samples)
        fin_samples = random.sample(set(fin_list), num_samples)
        it_samples = random.sample(set(it_list), num_samples)

        all_samples = sports_samples + ent_samples + auto_samples + fin_samples + it_samples

        random.shuffle(all_samples)

        train_size = int(math.floor(num_samples * 0.87 * 5))

        training, test = all_samples[:train_size], all_samples[train_size:]

        for data in training:
            tr.write(data+"\n")

        for data in test:
            te.write(data+"\n")        

        print 'Train_cnt',"\t",len(training)
        print 'Test_cnt',"\t",len(test)
        print '---------------\n'

        """
        seqlen < 2
        auto:59366
        tech:67645

        """

