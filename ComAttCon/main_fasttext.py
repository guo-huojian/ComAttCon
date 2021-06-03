# coding=utf-8

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.preprocessing import sequence

from fast_text import FastText
import jieba
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
import csv

max_features = 10000
#词表长度
maxlen = 250
embedding_dims = 100
max_sen = 97
max_zw = 96
# max_zw = 27
max_qg = 97

def create_ngram_set(input_list, ngram_value=1):
    """
    Extract a set of n-grams from a list of integers.
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=3):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 15000
maxlen = 400
batch_size = 32
embedding_dims = 100
epochs = 20


npzfile= np.load("data_pre.npz")
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

# x_train_qg =x_train_qg1
# x_test_qg =x_test_qg1


vocab = np.load("vocab.npy",allow_pickle=True)
vocab = vocab.tolist()
print('Loading data...')

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
# model = FastText(maxlen, max_features, embedding_dims).get_model()
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#
# print('Train...')
# early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           callbacks=[early_stopping],
#           validation_data=(x_test, y_test))
#
# print('Test...')
# result = model.predict(x_test)

i=0
for i in range(5):
    print('Build model...')
    model = model = FastText(maxlen, max_features, embedding_dims).get_model()
    # model = TextAttBiRNN(max_sen,max_zw,max_features, embedding_dims).get_model()
    # model = TextCNN_all(max_sen, max_zw,max_qg,max_features, embedding_dims,embedding_matrix).get_model()
    # model = Atae_lstm(maxlen, max_features, embedding_dims).get_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Train...')

    filepath = 'weights.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    print(type(x_train))

    ##//**有两个输入的模型fit
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        validation_data=(x_test, y_test))

    model.load_weights('weights.best.hdf5')
    # compile 编译
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Created model and loaded weights from hdf5 file')

    # estimate
    loss, acc = model.evaluate(x_test,y_test, batch_size=32, verbose=1)

    print("{0}: {1:.2f}%".format(model.metrics_names[1], acc * 100))
    val_predictq = model.predict(x_test)
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
    with open('./jieguo/jieguo.csv', 'a') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)

    i=i+1