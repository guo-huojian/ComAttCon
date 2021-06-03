# coding=utf-8
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from text_att_birnn import TextAttBiRNN
from atae_lstm import Atae_lstm

max_features = 10000
#词表长度
maxlen = 250
embedding_dims = 128


jieba.load_userdict('D:/Python3.6/Lib/site-packages/jieba/dict.txt')

# def stop_word(path):
#
#     content = [line.split() for line in open(path, 'r' , encoding='utf-8').readlines()]
#     return content

def load_data_and_labels(positive_data_file, negative_data_file,positive_zw_file, negative_zw_file):

    negative = []
    positive = []
    negative_zw = []
    positive_zw = []

    def fenju(file):
        for line in open(file, "r", encoding='utf-8').readlines():
            # negative.append(de_sequence(line))
            file.append(" ".join(jieba.lcut(line)))
            file = np.array(file)
            return file
    fenju(positive_data_file)
    fenju(negative_data_file)
    fenju(positive_zw_file)
    fenju(negative_zw_file)
    for line in open(negative_data_file, "r" , encoding='utf-8').readlines():
        # negative.append(de_sequence(line))
        negative.append(" ".join(jieba.lcut(line)))

    for line in open(positive_zw_file, "r", encoding='utf-8').readlines():
            fenju = " ".join(jieba.lcut(line))
            positive_zw.append(fenju)

    for line in open(negative_zw_file, "r", encoding='utf-8').readlines():
            negative_zw.append(" ".join(jieba.lcut(line)))

    positive = np.array(positive)
    # positive1 = np.expand_dims(positive,axis=0)
    negative = np.array(negative)
    # negative1 = np.expand_dims(negative, axis=0)
    positive_zw = np.array(positive_zw)
    # positive_zw1 = np.expand_dims(positive_zw, axis=0)
    negative_zw = np.array(negative_zw)
    # negative_zw1 = np.expand_dims(negative_zw, axis=0)
    # positive_all = np.concatenate((positive1, positive_zw1), axis=0)
    # negative_all = np.concatenate((negative1, negative_zw1), axis=0)

    index = [i for i in range(len(positive))]
    np.random.shuffle(index)
    x_train_pos = positive[index[0:int(len(index)*0.9)]]
    x_train_pos_zw = positive_zw[index[0:int(len(index) * 0.9)]]
    x_test_pos = positive[index[int(len(index) * 0.9):]]
    x_test_pos_zw = positive_zw[index[int(len(index) * 0.9):]]

    index = [i for i in range(len(negative))]
    x_train_neg = negative[index[0:int(len(index)*0.9)]]
    x_train_neg_zw = negative_zw[index[0:int(len(index) * 0.9)]]
    x_test_neg = negative[index[int(len(index) * 0.9):]]
    x_test_neg_zw = negative_zw[index[int(len(index) * 0.9):]]
    y_train_pos = [1. for _ in x_train_pos]
    y_train_neg = [0. for _ in x_train_neg]
    y_test_pos = [1. for _ in x_test_pos]
    y_test_neg = [0. for _ in x_test_neg]


    x_text = np.concatenate((positive, negative), 0)
    x_text_zw = np.concatenate((positive_zw, negative_zw), 0)

    def create_vocab(x_s,x_zw):

        word_freqs = {}
        vocab = {}
        index = 1
        x_s = np.concatenate((x_s,x_zw),0)
        for feature in x_s:

            lin= jieba.cut(feature)
            for li in lin:
                # if len(li) == 1:
                #     continue
                words = li.split()

                for w in words:
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        word_freqs[w] = 1

        for word, _ in word_freqs.items():
            vocab[word] = index
            index += 1

        return vocab,word_freqs

    vocab, word_freqs = create_vocab(x_text,x_text_zw)


    # negative_labels = [[0 , 1] for _ in negetive]
    # x_train_pos , x_test_pos , y_train_pos , y_test_pos = train_test_split(positive_all , positive_labels ,
    #                                                                        test_size=0.1,shuffle=True)
    # x_train_neg , x_test_neg , y_train_neg , y_test_neg = train_test_split(negative_all , negative_labels ,
    #                                                                        test_size=0.1,shuffle=True)
# 正负的比例
#     p_pos = positive.shape[0] / np.shape(x_text)[0]
#     p_neg = negative.shape[0] / np.shape(x_text)[0]
    # time.sleep(50000)


    x_train = np.concatenate((x_train_pos , x_train_neg) , 0)
    x_test = np.concatenate((x_test_pos , x_test_neg) , 0)
    x_train_zw = np.concatenate((x_train_pos_zw , x_train_neg_zw) , 0)
    x_test_zw = np.concatenate((x_test_pos_zw , x_test_neg_zw) , 0)
    y_train = np.concatenate((y_train_pos , y_train_neg) , 0)
    y_test = np.concatenate((y_test_pos , y_test_neg) , 0)

    return [x_train, x_train_zw, x_test, x_test_zw, y_train, y_test, vocab ]
###########source data###########

def read_dataset(vocab,x_p):

    data_x = []

    for feature in x_p:
        indices = []
        words = jieba.lcut(feature)

        for w in words:
            # if len(w)==1:
            #     continue
            if w == ' ':
                continue
            if w == '\n':
                continue
            if w == '\xa0':
                continue
            else:
                indices.append(vocab[w])
        data_x.append(indices)
    return data_x

x_train, x_train_zw ,x_test,x_test_zw, y_train, y_test, vocab  = load_data_and_labels('./po2.txt' , './ne2.txt','./po2_zw.txt' , './ne2_zw.txt')
print('x_train_shape: ' , np.shape(x_train))
print('x_test_shape: ' , np.shape(x_test))
print(x_test_zw)
# length_train = []
# for sentence in x_train:
#     length_train.append(len(sentence))
# # middle_train = sorted(length_train)[-int(np.shape(x_train)[0]/2)]
# middle_train = sorted(length_train)[-1]
# length_zw = []
# for sentence in x_train_zw:
#     length_zw.append(len(sentence))
# # middle_test = sorted(length_test)[-int(np.shape(x_test)[0]/2)]
# middle_zw = sorted(length_zw)[-1]
# middle = middle_train if middle_train > middle_zw else middle_zw
# print("mid_len: " , middle)

vocab_len = len(vocab)
print("vocab_len: ", vocab_len)