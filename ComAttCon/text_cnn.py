# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout,concatenate,Lambda,RepeatVector
import keras.backend as K
from attentionCNN import AttentionCNN
from attentionCNNinter import InteractiveAttentionCNN
from attention import Attention
class TextCNN(object):
    def __init__(self, max_sen, max_zw,max_qg,max_features, embedding_dims,embedding_matrix,
                 class_num=1,
                 last_activation='sigmoid'):
        self.max_sen = max_sen
        self.max_zw = max_zw
        self.max_qg = max_qg
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.embedding_matrix = embedding_matrix
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_units = 128
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        main_input = Input(shape=(self.max_sen,), dtype='int32', name='main_input')
        auxiliary_input1 = Input(shape=(self.max_zw,), dtype='int32', name='aux_input1')
        auxiliary_input2 = Input(shape=(self.max_qg,), dtype='int32', name='aux_input2')
        embedding = Embedding(input_dim=len(self.embedding_matrix), output_dim=len(self.embedding_matrix[0]),
                                    weights=[self.embedding_matrix],  # 表示直接使用预训练的词向量
                                    trainable=True)(main_input)
        # # embedding_zw = Embedding(input_dim=len(self.embedding_matrix), output_dim=len(self.embedding_matrix[0]),
        #                       weights=[self.embedding_matrix],  # 表示直接使用预训练的词向量
        #                       trainable=True)(auxiliary_input1)
        #
        embedding_qg = Embedding(input_dim=len(self.embedding_matrix), output_dim=len(self.embedding_matrix[0]),
                              weights=[self.embedding_matrix],  # 表示直接使用预训练的词向量
                                 trainable=True)(auxiliary_input2)


        # embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_sen)(main_input)
        # embedding_zw = Embedding(self.max_features, self.embedding_dims, input_length=self.max_zw)(auxiliary_input)
        # embedding_zw = Lambda(lambda y_m: K.mean(y_m, axis=1))(embedding_zw)
        # embedding_zw = RepeatVector(self.max_sen)(embedding_zw)
        embedding_all = concatenate([embedding, embedding_qg], axis=-1)
        # embedding_all_at = Attention(self.max_sen)(embedding_all)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(50, kernel_size, padding='same',activation='relu')(embedding)

            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        print(x)

        # for kernel_size in [4]:
        # x = Conv1D(100, 4, padding='same',activation='relu')(embedding_all_at)
        # x1 = GlobalMaxPooling1D()(x)
        # embedding_inter = InteractiveAttention()([embedding,embedding_zw])
        # y = Conv1D(100, 4, padding='same', activation='relu')(embedding_zw)
        # y1 = GlobalMaxPooling1D()(y)
        # inter = InteractiveAttention()([x, y])
            # convs.append(c)
        # x = Concatenate()(convs)
        # x = Attention(self.max_sen)(c)

        # x = Lambda(lambda y_m: K.mean(y_m, axis=1))(x)
        # x = concatenate([x1, inter], axis=-1)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[main_input, auxiliary_input1, auxiliary_input2], outputs=output)
        return model
