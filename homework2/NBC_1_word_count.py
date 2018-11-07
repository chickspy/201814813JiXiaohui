# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:33:10 2018

@author: CS
"""
import os

#list = os.listdir()

#print(list)

def list_files(curdir):
    #列出当前目录下所有文件
    files = []
    list = os.listdir(curdir) 
    for file in list:
        path = os.path.join(curdir,file)
        if os.path.isdir(path):
            files.extend(list_files(path))
        if os.path.isfile(path):
            files.append(path)
    return files


###############################################################
##########程序运行起点
files = list_files(r".\\20news-18828\\")

#所有单词的统计表
word_dict = {}

#遍历文件的计数，统计词频时作为字典值的列表项和文件的对应
file_cnt = 0

#遍历所有文件，做预处理和统计词频
for cur_file in files:
    #遍历所有文件，一次处理一个文件
    fin = open(cur_file,'rb')
    contentB = list(fin.read())
    fin.close()

    #文件计数
    file_cnt += 1
    
    #替换非打印字符为空格    
    for i in range(len(contentB)):
        if not chr(contentB[i]).isprintable():##isalpha()
            contentB[i] = 0x20
    wordlist = (''.join(chr(i) for i in contentB)).split()
#    print(wordlist)
    
    #分词 && 小写化
    mytext = (''.join(chr(i) for i in contentB)).lower()
    from nltk.tokenize import word_tokenize
    wordlist = word_tokenize(mytext)
#    print(wordlist)
    
    #词干提取
#    from nltk.stem import SnowballStemmer
#    english_stemmer = SnowballStemmer('english')
#    for i in range(len(wordlist)):
#        wordlist[i] = english_stemmer.stem(wordlist[i])
#    print(wordlist)
    
    #词性还原
    from nltk.stem import WordNetLemmatizer
    english_lemmatizer  = WordNetLemmatizer()
    for i in range(len(wordlist)):
        wordlist[i] = english_lemmatizer.lemmatize(wordlist[i])
#    print(wordlist)
    
    #删除非单词项
    from nltk.corpus import wordnet
    for w in wordlist[:]:
        if not wordnet.synsets(w):
            wordlist.remove(w)
#    print(wordlist)  
    
    #删除停用词
    from nltk.corpus import stopwords
    sr = stopwords.words('english')
    for w in wordlist[:]:
        if w in sr:
            wordlist.remove(w)
#    print(wordlist)
    
    #删除长度为1的单词 && 数字
    for w in wordlist[:]:
        if (len(w)==1) or (w[0].isdigit()):
            wordlist.remove(w)
        
    word_dict.setdefault(cur_file,{})
    #利用词典统计单词
    for w in wordlist:
        if w in word_dict[cur_file]:
            word_dict[cur_file][w] += 1
        else:
            word_dict[cur_file].setdefault(w,1)

    ##
    if file_cnt%1000 == 0:
        print(file_cnt)
    
#    print(r'\n\n')
#    print(word_dict)
    
#    if (file_cnt == 3):
#        break

print(file_cnt)

import json

#保存单词统计结果
jsobj = json.dumps(word_dict)
fileobject = open('word_dict.json','w')
fileobject.write(jsobj)
fileobject.close()
 



