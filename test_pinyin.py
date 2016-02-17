#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from pypinyin import (
    pinyin, slug, lazy_pinyin, lazy_stroke, load_single_dict,
    load_phrases_dict, NORMAL, TONE, TONE2, INITIALS,
    FIRST_LETTER, FINALS, FINALS_TONE, FINALS_TONE2
)
from pypinyin.compat import SUPPORT_UCS4

if __name__ == "__main__":

    print lazy_stroke(['中国', '国人', '人'])

