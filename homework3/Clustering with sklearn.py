# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:28:19 2019

@author: CS
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 08:51:39 2019

@author: CS
"""

import json
from sklearn import mixture, preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


##读取数据
docs = []
labels = []
with open('Tweets.txt', 'r') as f:
    for line in f.readlines():
        temp = json.loads(line)
        docs.append(temp['text'])
        labels.append(temp['cluster'])

##数据预处理
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

vectorizer = CountVectorizer()
X_cv = vectorizer.fit_transform(docs)

K = len(set(labels))


########################测试分类##################################
# K-Means 
for i in range(10): #分类10次
    mcluster = KMeans(n_clusters=K)
    mcluster.fit(X)
    labelsResult = mcluster.labels_
    print('第',i,'次','Kmeans NML = ', normalized_mutual_info_score(labels, labelsResult))


# Affinity Propagation
mcluster = AffinityPropagation().fit(X)
cluster_centers_indices = mcluster.cluster_centers_indices_
labelsResult = mcluster.labels_
print('Affinity Propagation NML = ', normalized_mutual_info_score(labels, labelsResult))


# Mean-Shift
temp = preprocessing.scale(X_cv.toarray())
mcluster = MeanShift(bandwidth=9).fit(temp)
labelsResult = mcluster.labels_
print('MeanShift NML = ', normalized_mutual_info_score(labels, labelsResult))


# Spectral Clustering
mcluster = SpectralClustering(n_clusters=K)
mcluster.fit(X)
labelsResult = mcluster.labels_
print('Spectral Clustering NML = ', normalized_mutual_info_score(labels, labelsResult))


# Ward Hierarchical Clustering
mcluster = AgglomerativeClustering(n_clusters=K).fit(X.toarray())
labelsResult = mcluster.labels_
print('Ward Hierarchical Clustering NML = ', normalized_mutual_info_score(labels, labelsResult))


# Agglomerative Clustering
mcluster = AgglomerativeClustering(linkage='complete', n_clusters=K).fit(X.toarray())
labelsResult = mcluster.labels_
print('Agglomerative Clustering NML = ', normalized_mutual_info_score(labels, labelsResult))


# DBSCAN
mcluster = DBSCAN(eps=0.1, min_samples=1).fit(X_cv.todense())
labelsResult = mcluster.labels_
print('DBSCAN NML = ', normalized_mutual_info_score(labels, labelsResult))


# Gaussian Mixtures
for i in range(10): #分类10次
    mcluster = mixture.GaussianMixture(n_components=K, covariance_type='diag')
    mcluster.fit(X.toarray())
    labelsResult = mcluster.predict(X.toarray())
    print('第',i,'次','Gaussian Mixtures NML = ', normalized_mutual_info_score(labels, labelsResult))
