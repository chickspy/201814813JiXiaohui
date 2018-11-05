# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:01:25 2018

@author: CS
"""

import json
import numpy
from copy import deepcopy

#########################################################################
##################程序运行起点
#读取统计的词频
print('读取所有文件词频统计结果')
with open(r".\word_dict.json",'r') as load_f:
    word_dict = json.load(load_f)
print(len(word_dict))


#按照给定的范围删除高频词和低频词
print('删除高、低频单词')

All_word_count = {}#记录所有单词频次
for file in word_dict:
    for w in word_dict[file]:
        if w in All_word_count:
            All_word_count[w] += 1
        else:
            All_word_count.setdefault(w,1)
print(len(All_word_count))

temp = []
for w in All_word_count:
    if not (5 < All_word_count[w] < 1000):
        temp.append(w)
        for file in word_dict:
            if w in word_dict[file]:
                word_dict[file].pop(w)
for w in temp:
    All_word_count.pop(w)
print(len(All_word_count))

#计算VSM
word_VSM = deepcopy(word_dict)
print('计算VSM')
word_IDF = {}
for w in All_word_count:
    w_in_file_cnt = 0
    for file in word_dict:
        if w in word_dict[file]:
            w_in_file_cnt += 1
    idf = numpy.log(len(word_dict)/w_in_file_cnt)
    word_IDF.setdefault(w,idf)

Alfa = 0.1
cnt = 0
for file in word_dict:
    if cnt%1000 == 0:
        print(cnt)
    cnt += 1
    max_word_cnt = 1###避免分母为0
    for w in word_dict[file]:
        if max_word_cnt <  word_dict[file][w]:
            max_word_cnt = word_dict[file][w]
    for w in word_dict[file]:
        tf = Alfa + (1-Alfa)*(word_dict[file][w]/max_word_cnt)
        word_VSM[file][w] = tf * word_IDF[w]


#保存所有文件的VSM结果
print('保存VSM结果')
jsobj = json.dumps(word_VSM)
fileobject = open('ALL_files_VSM.json','w')
fileobject.write(jsobj)
fileobject.close()





