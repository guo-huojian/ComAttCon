# coding=utf-8
import jieba
# from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from keras.preprocessing import sequence


max_features = 10000
#词表长度
maxlen = 250


jieba.load_userdict('D:/Python3.6/Lib/site-packages/jieba/dict.txt')
# def stop_word(path):
#     content = [line.split() for line in open(path, 'r' , encoding='utf-8').readlines()]
#     return content

def load_data_and_labels(positive_data_file, negative_data_file,positive_zw_file, negative_zw_file):

    negative = []
    positive = []
    negative_zw = []
    positive_zw = []

    for line in open(positive_data_file, "r" , encoding='utf-8').readlines():
        # positive.append(de_sequence(line))
        fenju = " ".join(jieba.lcut(line))
        positive.append(fenju)

    for line in open(negative_data_file, "r" , encoding='utf-8').readlines():
        negative.append(" ".join(jieba.lcut(line)))

    for line in open(positive_zw_file, "r", encoding='utf-8').readlines():
        fenju = " ".join(jieba.lcut(line))
        positive_zw.append(fenju)

    for line in open(negative_zw_file, "r", encoding='utf-8').readlines():
        negative_zw.append(" ".join(jieba.lcut(line)))

    positive = np.array(positive)
    negative = np.array(negative)
    positive_zw = np.array(positive_zw)
    negative_zw = np.array(negative_zw)

    index = [i for i in range(len(positive))]
    np.random.shuffle(index)
    x_train_pos = positive[index[0:int(len(index)*0.9)]]
    x_train_pos_zw = positive_zw[index[0:int(len(index) * 0.9)]]
    x_test_pos = positive[index[int(len(index) * 0.9):]]
    x_test_pos_zw = positive_zw[index[int(len(index) * 0.9):]]

    index = [i for i in range(len(negative))]
    np.random.shuffle(index)
    x_train_neg = negative[index[0:int(len(index)*0.9)]]
    x_train_neg_zw = negative_zw[index[0:int(len(index) * 0.9)]]
    x_test_neg = negative[index[int(len(index) * 0.9):]]
    x_test_neg_zw = negative_zw[index[int(len(index) * 0.9):]]
    y_train_pos = [1 for _ in x_train_pos]
    y_train_neg = [0 for _ in x_train_neg]
    y_test_pos = [1 for _ in x_test_pos]
    y_test_neg = [0 for _ in x_test_neg]

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

    # vocab_po, word_freqs_po = create_vocab(positive, positive)
    # vocab_ne, word_freqs_ne = create_vocab(negative, negative)

    vocab, word_freqs = create_vocab(x_text, x_text_zw)
    x_train = np.concatenate((x_train_pos , x_train_neg) , 0)
    x_test = np.concatenate((x_test_pos , x_test_neg) , 0)
    x_train_zw = np.concatenate((x_train_pos_zw , x_train_neg_zw) , 0)
    x_test_zw = np.concatenate((x_test_pos_zw , x_test_neg_zw) , 0)
    y_train = np.concatenate((y_train_pos , y_train_neg) , 0)
    y_test = np.concatenate((y_test_pos , y_test_neg) , 0)
    return [x_train, x_train_zw, x_test, x_test_zw, y_train, y_test, vocab]
    # return [x_train, x_train_zw, x_test, x_test_zw, y_train, y_test, vocab,vocab_po, vocab_ne]
###########source data###########
def read_dataset_qg(vocab,vocab_qg,x_p):
    data_x = []
    data_qg = []
    data_qg1 = []
    for sentence in x_p:
        sen = []
        qg = []
        qg1 = []
        words = jieba.lcut(sentence)
        for w in words:
            if w == ' ':
                continue
            if w == '\n':
                continue
            if w == '\xa0':
                continue
            else:
                sen.append(vocab[w])
                if w in vocab_qg:
                    qg.append(0)
                else:
                    qg.append(vocab[w])
                    qg1.append(1)
        if len(sen) != len(qg):
            print(sentence)

        data_x.append(sen)
        data_qg.append(qg)
        data_qg1.append(qg1)

    return data_x,data_qg,data_qg1

def read_dataset(vocab,x_p):
    data_x = []
    for sentence in x_p:
        indices = []
        words = jieba.lcut(sentence)
        for w in words:
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
def load_qg_voc(positive_data_file, negative_data_file,positive_zw_file, negative_zw_file):

    negative = []
    positive = []
    negative_zw = []
    positive_zw = []

    for line in open(positive_data_file, "r" , encoding='utf-8').readlines():
        positive.append(line.replace('\n',''))

    for line in open(negative_data_file, "r" , encoding='utf-8').readlines():
        negative.append(line.replace('\n',''))

    for line in open(positive_zw_file, "r", encoding='utf-8').readlines():
        positive_zw.append(line.replace('\n',''))

    for line in open(negative_zw_file, "r", encoding='utf-8').readlines():
        negative_zw.append(line.replace('\n',''))
    vocab_qg = positive+negative+positive_zw+negative_zw
    return vocab_qg
x_train, x_train_zw ,x_test,x_test_zw, y_train, y_test, vocab = load_data_and_labels('../data/po_1+2.txt' , '../data/ne_1+2.txt','../data/po_bt_1+2.txt' , '../data/ne_bt_1+2.txt')
#x_train, x_train_zw ,x_test,x_test_zw, y_train, y_test, vocab,vocab_po, vocab_ne  = load_data_and_labels('./gj_po_all_quchu.txt' , './gj_ne_all_quchu.txt','./gj_po_bt_all.txt' , './gj_ne_bt_all.txt')
print('x_train_shape: ', np.shape(x_train))
print('x_test_shape: ', np.shape(x_test))

# print(x_test_zw)
# vocab_qg = load_qg_voc('./sentiment/s1.txt', './sentiment/s2.txt','./sentiment/s3.txt', './sentiment/s4.txt')
vocab_qg = load_qg_voc('./sentiment/NTUSD_negative_simplified.txt', './sentiment/NTUSD_positive_simplified.txt','./sentiment/tsinghua.negative.gb.txt', './sentiment/tsinghua.positive.gb.txt')
print(len(vocab_qg))
## 考察情感词重合度
# zhengti = [ i for i in vocab if i in vocab_qg]
# guandian = [ i for i in vocab_po if i in vocab_qg]
# feiguan = [ i for i in vocab_ne if i in vocab_qg]
# print(guandian)
# print(len(guandian))
# print(feiguan)
# print(len(feiguan))
## 考察情感词重合度

vocab_len = len(vocab)
print("vocab_len: ", vocab_len)
x_train, x_train_qg ,x_train_qg1= read_dataset_qg(vocab,vocab_qg, x_train)
x_test, x_test_qg,x_test_qg1 = read_dataset_qg(vocab,vocab_qg, x_test)
x_train_zw = read_dataset(vocab, x_train_zw)
x_test_zw = read_dataset(vocab, x_test_zw)
x_qg = np.concatenate([x_train_qg,x_test_qg],axis=0)
x_qg1 = np.concatenate([x_train_qg1,x_test_qg1],axis=0)
sen_len = []
for sentence in x_train:
    sen_len.append(len(sentence))
# middle_train = sorted(length_train)[-int(np.shape(x_train)[0]/2)]
max_sen = sorted(sen_len)[-1]
zw_len = []
for sentence in x_train_zw:
    zw_len.append(len(sentence))
# # middle_test = sorted(length_test)[-int(np.shape(x_test)[0]/2)]
max_zw = sorted(zw_len)[-1]
# # middle = middle_train if middle_train > middle_zw else middle_zw
print("max_sen: " , max_sen)
print("max_zw: " , max_zw)
qg_len = []
for sentence in x_qg:
    qg_len.append(len(sentence))
max_qg = sorted(qg_len)[-1]
print("max_qg: ", max_qg)


x_train = sequence.pad_sequences(x_train, maxlen=max_sen , padding='post')
x_train_qg = sequence.pad_sequences(x_train_qg, maxlen=max_qg , padding='post')
x_train_qg1 = sequence.pad_sequences(x_train_qg1, maxlen=max_qg , padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_sen, padding='post')
x_test_qg = sequence.pad_sequences(x_test_qg, maxlen=max_qg, padding='post')
x_test_qg1 = sequence.pad_sequences(x_test_qg1, maxlen=max_qg, padding='post')
x_train_zw = sequence.pad_sequences(x_train_zw, maxlen=max_zw , padding='post')
x_test_zw = sequence.pad_sequences(x_test_zw, maxlen=max_zw , padding='post')
np.savez("data_pre.npz", x_train=x_train, x_train_qg=x_train_qg,x_train_qg1=x_train_qg1, x_train_zw=x_train_zw,
         x_test=x_test, x_test_qg=x_test_qg, x_test_qg1=x_test_qg1,x_test_zw=x_test_zw, y_train=y_train, y_test=y_test)

np.save("vocab.npy", vocab)
