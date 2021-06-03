# coding=utf-8

import numpy as np
import os

from time import *
from text_cnn_all import TextCNN_all
from keras.models import Model

max_features = 10000
# 词表长度
maxlen = 250
embedding_dims = 100
max_sen = 97
max_zw = 96
# max_zw = 27
max_qg = 97

# 需要修改路劲
npzfile= np.load("../TextAttBiRNN-zw/data_pre.npz")
x_train = npzfile['x_train']
x_train_qg = npzfile['x_train_qg']
x_train_zw = npzfile['x_train_zw']
x_test = npzfile['x_test']
x_test_qg = npzfile['x_test_qg']
x_test_zw = npzfile['x_test_zw']
y_train = npzfile['y_train']
y_test = npzfile['y_test']
x_train_qg1 = npzfile['x_train_qg1']
x_test_qg1 = npzfile['x_test_qg1']

# 需要修改路劲
vocab = np.load("../TextAttBiRNN-zw/vocab.npy", allow_pickle=True)
vocab = vocab.tolist()

# 需要修改路劲
GLOVE_DIR = "../TextAttBiRNN-zw/"
# GLOVE_DIR = 'D:/BaiduNetdiskDownload'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'sgns.weibo.word'), encoding='gb18030', errors='ignore')
# embedding_index是个dict
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[2:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(vocab) + 1, embedding_dims ))
for word, i in vocab.items():
    embedding_word = embeddings_index.get(word)
    if embedding_word is not None:
        embedding_matrix[i] = embedding_word[:100]

if __name__ == "__main__":
    '''
    需要某层的输出值，将model.layers[5].output中的
    数字修改中间的层数即可，如想看第五层的输出就将[]里
    面的数字修改为5
    '''
    model, a_c = TextCNN_all(max_sen, max_zw, max_qg, max_features, embedding_dims, embedding_matrix).get_model()
    # 需要修改路劲
    model.load_weights('../TextAttBiRNN-zw/weights.best.hdf5')
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[5].output)
    data = [x_train, x_train_zw, x_train_qg]
    print(intermediate_layer_model.predict(data))
