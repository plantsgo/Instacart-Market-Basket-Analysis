#encoding=utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import chain
import gc
from com_util import *

path="input/"
priors=pd.read_csv(path+"order_products__prior.csv")
orders = pd.read_csv(path + 'orders.csv')
#重新构造这三个文件
train_user=orders[orders.eval_set=="train"][["user_id"]].copy()
train_user["mark"]=1
orders=orders[orders.eval_set=="prior"].copy()
orders=orders.merge(train_user,on="user_id",how="left").fillna(0)
#orders["order_number_max"]=orders.groupby("user_id").order_number.max()
orders=merge_max(orders,["user_id"],"order_number","order_number_max")
print orders.head(20)
def mark(a,b,c):
    if a!=b:
        return "prior"
    else:
        if c==1:
            return "train"
        else:
            return "test"
orders["eval_set"]=map(lambda a,b,c:mark(a,b,c),orders["order_number"],orders["order_number_max"],orders["mark"])
del orders["mark"]
del orders["order_number_max"]

print orders.head(20)

orders_train=orders[orders.eval_set=="train"][["order_id"]].copy()
orders_test=orders[orders.eval_set=="test"][["order_id"]].copy()
orders_prior=orders[orders.eval_set=="prior"][["order_id"]].copy()

train=priors.merge(orders_train,on="order_id",how="inner")
test=priors.merge(orders_test,on="order_id",how="inner")
priors=priors.merge(orders_prior,on="order_id",how="inner")

orders.to_csv("output/orders.csv",index=None)
train.to_csv("output/order_products__train.csv",index=None)
test.to_csv("output/order_products__test.csv",index=None)
priors.to_csv("output/order_products__prior.csv",index=None)
