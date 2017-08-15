"""
by Omar Essam
Word2Vec for products analysis + 0.01 LB
"""
import pandas as pd
import numpy as np
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_orders = pd.read_csv("input/order_products__train.csv")
prior_orders = pd.read_csv("input/order_products__prior.csv")
products = pd.read_csv("input/products.csv").set_index('product_id')

#Turn the product ID to a string
#This is necessary because Gensim's Word2Vec expects sentences, so we have to resort to this dirty workaround
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

#Create the final sentences
sentences = prior_products.append(train_products).values

#Train Word2Vec model
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

model.save("features/product2vec.model")
model.wv.save_word2vec_format("features/product2vec.model.bin", binary=True)
