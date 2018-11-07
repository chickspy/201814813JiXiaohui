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

#对词频字典按文件种类进行分类
type_path_dict = {}
for path in word_dict:
    type_path_dict.setdefault(path.split('\\')[-2],[]).append(path)
print(len(type_path_dict))

#试验循环
for i in range(20):
    print('第',i,'次测试')
    #将数据分为80%的训练数据 和 20%的测试数据
    train_set = deepcopy(word_dict)  ##训练数据集
    test_set = {}                    ##待分类测试数据集
    cnt_typer = {}
    for doc_type in type_path_dict:
        test = random.sample(type_path_dict[doc_type],int(len(type_path_dict[doc_type])*0.2))
        for path in test:
            test_set.setdefault(path,{})
            test_set[path] = word_dict[path].copy()
            train_set.pop(path)
        cnt_typer.setdefault(doc_type,len(type_path_dict[doc_type])-len(test))
    
    print(len(test_set))
    print(len(train_set))
    
    #计算每一类概率 给分类训练数据初始化数据结构
    classification = {}
    class_word_cnt = {}
    for typer in cnt_typer:
        cnt_typer[typer] = cnt_typer[typer]*1.0 / len(train_set)
        classification.setdefault(typer,[]).append({})
        classification.setdefault(typer,[]).append({})
        class_word_cnt.setdefault(typer,[0,0])
    
    #统计训练数据
    print('统计训练数据')
    for doc in train_set:
        typer = doc.split('\\')[-2]
        for t in classification:
            if t == typer:
                index = 0
            else:
                index = 1
            for w in train_set[doc]:
                if w in classification[typer][index]:
                    classification[typer][index][w] += train_set[doc][w]
                else:
                    classification[typer][index].setdefault(w,train_set[doc][w])
                class_word_cnt[typer][index] += train_set[doc][w]
    #计算训练数据概率分布
    for typer in classification:
        for v in range(2):
            for w in classification[typer][v]:##numpy.log
                classification[typer][v][w] = numpy.log((classification[typer][v][w]+1)/(class_word_cnt[typer][v]+len(classification[typer][v])))
    
    #对测试集进行分类
    same = 0
    cnt = 0
    for test in test_set:
        cnt += 1
#        print(test)
        all_result = {}
        for typer in cnt_typer:
            all_result.setdefault(typer,[numpy.log(class_word_cnt[typer][0]/len(classification[typer][0])),numpy.log(class_word_cnt[typer][1]/len(classification[typer][1]))])
        for w in test_set[test]:
            for typer in classification:
                for v in range(2):
                    if w in classification[typer][v]:
                        all_result[typer][v] += classification[typer][v][w]*test_set[test][w]
                    else:
                        all_result[typer][v] += numpy.log(1.0*test_set[test][w]/(class_word_cnt[typer][v]+len(classification[typer][v])))
        
        sort_data = {}
        for typer in all_result:
#            sort_data.setdefault(typer,(all_result[typer][0]/all_result[typer][1]))
            sort_data.setdefault(typer,(all_result[typer][0]-all_result[typer][1]))
        
#        sort_result = sorted(sort_data.items(),key=lambda item:item[1],reverse = True)##False
        sort_result = sorted(sort_data.items(),key=lambda item:item[1],reverse = False)
        #排序取出最大类
        
        if test.split('\\')[-2] == sort_result[0][0]:
            same += 1
    print("第",i,"次测试：",same,"/",cnt,"=",same/cnt)
#    file_out = open('NBC测试结果_相除_取最大.txt','a+')
    file_out = open('NBC测试结果_相减_取最小.txt','a+')
    file_out.write('第{:3}次测试: {}/{}={}\n'.format(i,same,cnt,same/cnt))
    file_out.close()
