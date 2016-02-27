#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""从stroke.dict.yaml 按 unicode 编码获取汉字及 stroke"""

with open("stroke_dict.py", "w") as s:
	s.write("stroke_dict = {\n")

	with open("stroke.dict.yaml") as f:
		for i, line in enumerate(f):
			if "#" in line:
				continue
			if line  == '\n' :
				continue

			if i > 22:
				word = line.split("\t")[0]
				stroke = line.split("\t")[1].strip()
				stroke = "\"" + stroke + "\""
				s.write("\t0x" + word.decode("utf8").encode('unicode-escape').upper()[2:] +":"+stroke+","+"\n")

	s.write("}")

