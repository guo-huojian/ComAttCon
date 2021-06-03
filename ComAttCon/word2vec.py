import os
import numpy as np
import pandas as pd
EMBEDDING_DIM = 100

vocab = np.load("vocab.npy",allow_pickle=True)
vocab = vocab.tolist()
def load_vec(vocab):
    GLOVE_DIR ="D:/BaiduNetdiskDownload"
    embeddings_index = {}
    a=0
    f = open(os.path.join(GLOVE_DIR, 'sgns.weibo.word'),encoding='gb18030',errors= 'ignore')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[2:], dtype='float32')
        embeddings_index[word] = coefs
        ##embedding_index是个dict
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    for word, i in vocab.items():
        embedding_word = embeddings_index.get(word)
        if embedding_word is not None:
    # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_word[:100]
          a=a+1
    print(a)
    return embedding_matrix

# embedding_matrix = load_vec(vocab)

# from keras.layers import Embedding
# embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,
#                             weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH, trainable=False)

