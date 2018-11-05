# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:01:25 2018

@author: CS
"""

import json
import numpy
import random
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
    if not (4 < All_word_count[w] < 5000):
        temp.append(w)
        for file in word_dict:
            if w in word_dict[file]:
                word_dict[file].pop(w)
for w in temp:
    All_word_count.pop(w)
print(len(All_word_count))

#对词频字典按文件种类进行分类
type_dict = {}
panjue = {}
for path in word_dict:
    type_dict.setdefault(path.split('\\')[-2],[]).append(path)
    panjue.setdefault(path.split('\\')[-2],0)
print(len(type_dict))

#试验循环
for i in range(10):
    print('第',i,'次测试')
    #将数据分为80%的训练数据 和 20%的测试数据
    train_set = deepcopy(word_dict)  ##训练数据集
    test_set = {}                    ##待分类测试数据集
    for doc_type in type_dict:
        test = random.sample(type_dict[doc_type],int(len(type_dict[doc_type])*0.2))
        for path in test:
            test_set.setdefault(path,{})
            test_set[path] = word_dict[path].copy()
            train_set.pop(path)
    
    #计算训练数据VSM
    print('计算训练数据IDF')
     #计算IDF，只统计训练数据集
    word_IDF = {}
    for w in All_word_count:
        w_in_file_cnt = 0
        for doc in train_set:##train_set
            if w in train_set[doc]:##train_set
                w_in_file_cnt += 1
        if w_in_file_cnt != 0:
            idf = numpy.log(len(train_set)/w_in_file_cnt)##train_set
        else:
            idf = 0
        word_IDF.setdefault(w,idf)
    
    #计算训练数据集的VSM
    print('计算训练数据VSM')
    Alfa = 0.1
    for doc in train_set:
        max_word_cnt = 1###避免分母为0
        for w in train_set[doc]:
            if max_word_cnt <  train_set[doc][w]:
                max_word_cnt = train_set[doc][w]
        for w in train_set[doc]:
            tf = Alfa + (1-Alfa)*(train_set[doc][w]/max_word_cnt)
            train_set[doc][w] = tf * word_IDF[w]
            
     #计算测试数据集的VSM 
    print('计算测试数据VSM')         
    for doc in test_set:
        max_word_cnt = 1###避免分母为0
        for w in test_set[doc]:
            if max_word_cnt <  test_set[doc][w]:
                max_word_cnt = test_set[doc][w]
        for w in test_set[doc]:
            tf = Alfa + (1-Alfa)*(test_set[doc][w]/max_word_cnt)
            test_set[doc][w] = tf * word_IDF[w]
    
    #遍历测试文件计算与所有训练数据的cos值
    cnt = 0  ##测试文件计数
    same = 0 ##成功分类计数
    K = 7
    print('遍历测试数据进行KNN分类 K=',K)
    for doc_test in test_set:
        cnt += 1
        Cos_dict = {}##保留所有Cos
        for doc_train in train_set:
            ##计算并保留所有Cos
            Vij = 0.0
            Vii = 0.0
            Vjj = 0.0
            for word in test_set[doc_test].keys():
                vi = test_set[doc_test].get(word,0)
                vj = train_set[doc_train].get(word,0)
                Vij += vi * vj
                Vii += vi * vi
            for word in train_set[doc_train].keys():
                vj = train_set[doc_train].get(word,0)
                Vjj += vj * vj
            if (Vii*Vjj) == 0:
                value = 0
            else:
                value = Vij/((Vii**0.5)*(Vjj**0.5))
            Cos_dict.setdefault(doc_train,value)
        #对Cos结果进行排序
        sort_result = sorted(Cos_dict.items(),key=lambda item:item[1],reverse = True)
#        print(sort_result)

        for key in panjue:
            panjue[key] = 0
        
        for j in range(K):
            panjue[sort_result[j][0].split('\\')[-2]] += 1

        sort_result = sorted(panjue.items(),key=lambda item:item[1],reverse = True)
        
        typer = ''
        if sort_result[0][1] !=  sort_result[1][1]:
            typer = sort_result[0][0]

        #得到分类结果是否正确
        if typer == doc_test.split('\\')[-2]:
            same += 1

        if cnt%100 == 0:
            print(same,'/',cnt)
    print(same,'/',cnt)
    #保留运行结果
    file_out = open('KNN测试结果.txt','a+')
    file_out.write('第{:2d}次测试 : {:5d} / {:5d} = {:.4f}\n'.format(i,same,cnt,same/cnt))
    file_out.close()
