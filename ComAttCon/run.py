# coding=utf-8
import jieba
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from text_att_birnn import TextAttBiRNN
from text_cnn import TextCNN
from text_cnn_all import TextCNN_all
from atae_lstm import Atae_lstm

from word2vec import load_vec
import csv

import pandas as pd

max_features = 10000
#词表长度
maxlen = 250
embedding_dims = 100
max_sen = 97
max_zw = 96
# max_zw = 27
max_qg = 97


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

GLOVE_DIR ="D:/BaiduNetdiskDownload"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'sgns.weibo.word'),encoding='gb18030',errors= 'ignore')
for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[2:], dtype='float32')
        embeddings_index[word] = coefs
        ##embedding_index是个dict
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(vocab) + 1, embedding_dims ))
for word, i in vocab.items():
    embedding_word = embeddings_index.get(word)
    if embedding_word is not None:
     embedding_matrix[i] = embedding_word[:100]



i=0
for i in range(1):
    print('Build model...')
    # model = TextAttBiRNN(max_sen, max_zw, max_qg, max_features, embedding_dims, embedding_matrix).get_model()
    # model = TextAttBiRNN(max_sen,max_zw,max_features, embedding_dims).get_model()
    model = TextCNN_all(max_sen, max_zw,max_qg,max_features, embedding_dims,embedding_matrix).get_model()
    # model = Atae_lstm(maxlen, max_features, embedding_dims).get_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Train...')

    filepath = 'weights.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    model.load_weights('weights.best.hdf5')
    # compile 编译
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Created model and loaded weights from hdf5 file')

    # estimate
    loss, acc = model.evaluate(x={'main_input': x_test, 'aux_input1': x_test_zw, 'aux_input2': x_test_qg},
                               y=y_test, batch_size=32, verbose=1)

    print("{0}: {1:.2f}%".format(model.metrics_names[1], acc * 100))
    val_predictq = model.predict(x={'main_input': x_test, 'aux_input1': x_test_zw, 'aux_input2': x_test_qg})
    val_predict = np.round(val_predictq).reshape(-1)
    # val_predict = (np.asarray(model.predict(x_test))).round()
    val_targ = y_test

    po_val_precision = precision_score(val_targ, val_predict)
    po_val_recall = recall_score(val_targ, val_predict)
    po_val_f1 = f1_score(val_targ, val_predict)
    print("pre_pos: ")
    print('-val_f1: %.6f --val_precision: %.6f --val_recall: %.6f' % (po_val_f1, po_val_precision, po_val_recall))

    # pre_neg#
    ne_val_precision = precision_score(val_targ, val_predict, pos_label=0)
    ne_val_recall = recall_score(val_targ, val_predict, pos_label=0)
    ne_val_f1 = f1_score(val_targ, val_predict, pos_label=0)
    print("pre_neg: ")
    print('-val_f1: %.6f --val_precision: %.6f --val_recall: %.6f' % (ne_val_f1, ne_val_precision, ne_val_recall))

    rows = [(acc, po_val_precision, po_val_recall, po_val_f1, ne_val_precision, ne_val_recall, ne_val_f1)]
    with open('./jieguo/jieguo1.csv', 'a') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)

    i=i+1