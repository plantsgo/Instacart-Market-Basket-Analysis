# -*- coding: utf-8 -*-
import gensim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#model = gensim.models.word2vec.Word2Vec.load_word2vec_format("input/product2vec.model.bin", binary=True)
model = gensim.models.Word2Vec.load("features/product2vec.model")
print len(model.wv.vocab)
embedding_matrix = np.zeros((49688, 100))
for i in range(1,49689):
    word=str(i)
    if word in model.wv.vocab:
        embedding_matrix[i-1] = model.wv[word]

#PCA降维
features_num=2
pca=PCA(features_num)
embedding_matrix=pca.fit_transform(embedding_matrix)
features=pd.DataFrame(embedding_matrix)
features.columns=["w2v_%s"%j for j in range(features_num)]
features["product_id"]=range(1,49689)
print features.head()

features.to_csv("features/product2vec.csv",index=None)


