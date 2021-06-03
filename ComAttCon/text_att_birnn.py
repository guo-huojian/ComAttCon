# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional,concatenate,Lambda,RepeatVector
import keras.backend as K
from attention import Attention
from attention_zw import InteractiveAttention
# from attention_qg import InteractiveEmb
# Attention_zw,
import numpy as np

class TextAttBiRNN(object):
    def __init__(self, max_sen, max_zw,max_qg, max_features, embedding_dims,embedding_matrix,
                 class_num=1,
                 last_activation='sigmoid'):
        self.max_sen = max_sen
        self.max_zw = max_zw
        self.max_qg = max_qg
        self.max_features = max_features
        self.embedding_matrix = embedding_matrix
        self.embedding_dims = 100
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_units = 128

    def get_model(self):
        # input1= Input((self.maxlen,))
        main_input = Input(shape=(self.max_sen,), dtype='int32', name='main_input')
        auxiliary_input1 = Input(shape=(self.max_zw,), dtype='int32',name='aux_input1')
        auxiliary_input2 = Input(shape=(self.max_qg,), dtype='int32', name='aux_input2')

        embedding = Embedding(input_dim=len(self.embedding_matrix), output_dim=len(self.embedding_matrix[0]),
                                    weights=[self.embedding_matrix],  # 表示直接使用预训练的词向量
                                    trainable=True)(main_input)
        embedding_zw = Embedding(input_dim=len(self.embedding_matrix), output_dim=len(self.embedding_matrix[0]),
                              weights=[self.embedding_matrix],  # 表示直接使用预训练的词向量
                              trainable=True)(auxiliary_input1)

        embedding_qg = Embedding(input_dim=len(self.embedding_matrix), output_dim=len(self.embedding_matrix[0]),
                              weights=[self.embedding_matrix],  # 表示直接使用预训练的词向量
                                 trainable=True)(auxiliary_input2)



        # embedding = Embedding(self.max_features,self.embedding_dims, input_length=self.max_sen)(main_input)
        # embedding_zw = Embedding(self.max_features, self.embedding_dims, input_length=self.max_zw)(auxiliary_input1)
        # embedding_qg = Embedding(self.max_features, self.embedding_dims, input_length=self.max_qg)(auxiliary_input2)

        # embedding_qg = Lambda(lambda y_m: K.mean(y_m, axis=1))(embedding_qg)
        # embedding_zw = RepeatVector(self.max_sen)(embedding_zw)
        # embedding_all = concatenate([embedding, embedding_qg], axis=-1)

        # x = Bidirectional(LSTM(64,dropout=0.5))(embedding)  # LSTM or GRU
        ##return_sequences：默认 False。在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。 False 返回单个， true 返回全部。
        # lstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.5))
        # x1 = lstm(embedding)
        # y = lstm(embedding_zw)
        # x = Attention(self.max_sen)(embedding_all)
        x = Bidirectional(LSTM(128,dropout=0.5))(embedding)
        # x1 = Bidirectional(LSTM(128,dropout=0.5))(embedding)
        # y = Bidirectional(LSTM(128,return_sequences=True,dropout=0.5))(embedding_zw)
        # y = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(embedding_zw)
        # x,y维度: ?,?,128
        # y = RepeatVector(self.maxlen)(y)
        # y = concatenate([x, y], axis=-1)
        # x = concatenate([x, y], axis=-1)
        # x = Bidirectional(LSTM(64, dropout=0.5))(embedding)  # LSTM or GRU
        # x = Attention(self.max_sen)(x)
        # x: shape=(?,128)

        # y = RepeatVector(self.maxlen)(y)
        # y = concatenate([x, y], axis=-1)
        # y = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(y)
        # sentence = Attention(self.maxlen)(x)
        # sentence = Attention(self.maxlen)(y)

        # inter = InteractiveAttention()([x,y])
        # x = Lambda(lambda y_m: K.mean(y_m, axis=1))(x)
        # x = Lambda(lambda y_m: y_m[:,self.max_sen,:])(x)
        # x = concatenate([x, inter], axis=-1)
        # x = Attention(self.max_sen)(x)
        # x = x[self.maxlen-1,]
        #
        # x = Attention_zw(self.maxlen)([x, y])
        # output = Dense(self.class_num, activation=self.last_activation)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        # dense_layer = Dense(self.dense_units, activation='relu')(sentence)
        # output = Dense(self.class_num, activation='softmax')(dense_layer)
        # output: shape=(?,1)
        model = Model(inputs=[main_input, auxiliary_input1,auxiliary_input2], outputs=output)
        return model
