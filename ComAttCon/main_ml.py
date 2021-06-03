# coding=utf-8
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# x_train = x_train.tolist
npzfile= np.load("data_pre.npz")
x_train = npzfile['x_train']
x_train_qg = npzfile['x_train_qg']
x_train_zw = npzfile['x_train_zw']
x_test = npzfile['x_test']
x_test_qg = npzfile['x_test_qg']
x_test_zw = npzfile['x_test_zw']
y_train = npzfile['y_train']
y_test = npzfile['y_test']


vocab = np.load("vocab.npy",allow_pickle=True)
vocab = vocab.tolist()


print('Build model...')
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
    # 列表形式呈现文本生成的词典
print(vec.get_feature_names())
x_test = vec.transform(x_test)
###bayes###
mnb = MultinomialNB()
mnb.fit(x_train , y_train)
y_predict = mnb.predict(x_test)
score = mnb.score(x_test , y_test)
print(score)
print(classification_report(y_test , y_predict ,digits=6))
##bayes###

##SVM###
clf = SVC(kernel='linear')
clf.fit(x_train , y_train)
y_predict = clf.predict(x_test)
score1 = clf.score(x_test , y_test)
print(score1)
print(classification_report(y_test ,y_predict , digits=6))
##SVM###


import csv

import pandas as pd

max_features = 20000






