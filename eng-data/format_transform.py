# -*- coding: utf-8 -*-

import sys,os
import time

import re

import csv
import itertools

from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

import eng2ch_dict

ENG_WUBI_DICT = eng2ch_dict.eng2ch_dict.copy()

stemmer = LancasterStemmer()

csv.field_size_limit(sys.maxsize)

from progressbar import ProgressBar

def tokenize(content):
    return [stemmer.stem(t) for t in word_tokenize(content) if re.match(r'[a-z]+', t, re.M|re.I)]

def transformation(content, label):
    toks = tokenize(content)

    wbs = []
    for tok in toks:
        if tok in ENG_WUBI_DICT:
            wbs.append(ENG_WUBI_DICT[tok])

    wbs = ' '.join(wbs)

    exp = "\""+label+"\"" + str(',') + "\""+wbs+"\""
    return exp

if __name__ == "__main__":
        
    with open("./train_wb.csv", 'w') as tr, open("./test_wb.csv", 'w') as te, \
        open("./train.csv", 'rb') as tr_origin, open("./test.csv", 'rb') as te_origin:

        reader_tr_origin = csv.reader(tr_origin, delimiter=',', quotechar='"')
        reader_te_origin = csv.reader(te_origin, delimiter=',', quotechar='"')

        #Train_cnt   120000
        #Test_cnt    7600

        pbar_train = ProgressBar(maxval=120000).start()

        for i, row in enumerate(reader_tr_origin):
            time.sleep(0.01)
            pbar_train.update(i + 1)
            pys = transformation(row[2], row[0])
            tr.write(pys+"\n")


        pbar_train.finish()

        pbar_test = ProgressBar(maxval=7600).start()

        for i, row in enumerate(reader_te_origin):
            time.sleep(0.01)
            pbar_test.update(i + 1)
            pys = transformation(row[2], row[0])
            te.write(pys+"\n")

        pbar_test.finish()
            
