# coding=utf-8
import numpy as np
import jieba
from gensim.models import word2vec

def loadvec(positive_data_file, negative_data_file,positive_zw_file, negative_zw_file):

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

vocab_qg = loadvec('./sentiment/s1.txt', './sentiment/s2.txt','./sentiment/s3.txt', './sentiment/s4.txt')
print(vocab_qg)










# w2vModel =  word2vec.load_word2vec_format('model/result.bin', binary=False)

# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('D:/BaiduNetdiskDownload/sgns.weibo.word',binary=True)
# print(model['我'])

# vocab={} # 词汇表为数据预处理后得到的词汇字典
# # 构建词向量索引字典
# ## 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量
# glove_dir="D:/BaiduNetdiskDownload/sgns.weibo.word"
# f=open(glove_dir,"r",encoding="utf-8")
# ## 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,注意，部分预训练好的词向量的第一行并不是该词向量的维度
# l,w=f.readline().split()
# ## 创建词向量索引字典
# embeddings_index={}
# for line in f:
#     ## 读取词向量文件中的每一行
#     values=line.split()
#     ## 获取当前行的词
#     word=values[0]
#     ## 获取当前词的词向量
#     coefs=np.asarray(values[1:],dtype="float32")
#     ## 将读入的这行词向量加入词向量索引字典
#     embeddings_index[word]=coefs
# f.close()
#
# # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
# ## 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
# embedding_matrix=np.zeros((len(vocab)+1,int(w))
# ## 遍历词汇表中的每一项
# for word,index in vocab.items():
#     ## 在词向量索引字典中查询单词word的词向量
#     embedding_vector=embeddings_index.get(word)
#     ## 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
#     if embedding_vector is not None:
#         embedding_matrix[index]=embedding_vector
#
# # 修改模型中嵌入层代码
# # embedder=Embedding(len(vocab)+1,w,input_length = 64, weights = [embedding_matrix], trainable = False))
