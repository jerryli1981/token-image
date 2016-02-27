#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open("wb_dict.py", "w") as s:
	s.write("wb_dict = {\n")

	with open("dict_wb.txt") as f:
		for i, line in enumerate(f):
			word = line.split("\t")[0]
			wb = line.split("\t")[1].strip()
			wb = "\"" + wb + "\""
			s.write("\t0x" + word.decode("utf8").encode('unicode-escape').upper()[2:] +":"+wb+","+"\n")

	s.write("}")

