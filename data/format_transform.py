# -*- coding: utf-8 -*-

import sys,os
import time

import re

import csv
import itertools

csv.field_size_limit(sys.maxsize)


sys.path.insert(0, "../")

from pypinyin import stroke, pinyin, wubi
import pypinyin

from progressbar import ProgressBar

def transEscape(content):
    content =re.sub(r'\n', "\\\\n", content)
    content =re.sub(r'\"', "\"\"", content)
    return content


def transformation(content, label):

    content = content.decode('utf-8')
    toks = content.split(" ")

    if transfer != None:
        pys = transfer(toks, style=style, errors=error)
        pys = ' '.join(list(itertools.chain(*pys))).encode('utf-8')
        pys = transEscape(pys)
    else:
        pys = ' '.join(toks).encode('utf-8')
        pys = transEscape(pys)

    exp = "\""+label+"\"" + str(',') + "\""+pys+"\""
    return exp

if __name__ == "__main__":
    
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
        open("./train_origin.csv", 'rb') as tr_origin, open("./test_origin.csv", 'rb') as te_origin:

        reader_tr_origin = csv.reader(tr_origin, delimiter=',', quotechar='"')
        reader_te_origin = csv.reader(te_origin, delimiter=',', quotechar='"')

        #Train_cnt   262513
        #Test_cnt    39227

        pbar_train = ProgressBar(maxval=262513).start()

        for i, row in enumerate(reader_tr_origin):
            time.sleep(0.01)
            pbar_train.update(i + 1)
            pys = transformation(row[1], row[0])
            tr.write(pys+"\n")


        pbar_train.finish()

        pbar_test = ProgressBar(maxval=39227).start()


        for i, row in enumerate(reader_te_origin):
            time.sleep(0.01)
            pbar_test.update(i + 1)
            pys = transformation(row[1], row[0])
            te.write(pys+"\n")

        pbar_test.finish()
            
