#encoding=utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import chain
import gc
from com_util import *
from sklearn.metrics import log_loss
from scipy.stats import mode


def eval_fun(labels, preds):
    labels = labels.split(' ')
    preds = preds.split(' ')
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError
        return 0.0
    return f1

def create_data(data,treshold):
    d = dict()
    for row in data.itertuples():
        if row.label >= treshold:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in data.order_id:
        if order not in d:
            d[order] = 'None'
    data = pd.DataFrame.from_dict(d, orient='index')
    data.reset_index(inplace=True)
    data.columns = ['order_id', 'products']
    data = data.sort_values("order_id")
    return data

def normalize(f):
    return f/f.std()

##添加数据##
train_add=pd.read_csv("output/train_add.csv")
test_add=pd.read_csv("output/test_add.csv")
all_add=pd.concat([train_add,test_add])
#all_add["orders_people"]=normalize(all_add["orders_people"])
#all_add["orders"]=normalize(all_add["orders"])
print(all_add.shape)

#删去只有三单的
all_add=all_add[(all_add.nb_orders>=4)&(all_add.days_since_prior_order<30)]
#all_add=all_add[(all_add.nb_orders>=4)&(all_add.nb_orders<30)&(all_add.days_since_prior_order<30)&(all_add.total_items<30)]

##添加数据##
train_add_1=pd.read_csv("output_2/train_add.csv")
test_add_1=pd.read_csv("output_2/test_add.csv")
all_add_1=pd.concat([train_add_1,test_add_1])
#all_add["orders_people"]=normalize(all_add["orders_people"])
#all_add["orders"]=normalize(all_add["orders"])
print(all_add_1.shape)

#删去只有三单的
all_add_1=all_add_1[(all_add_1.nb_orders>=4)&(all_add_1.days_since_prior_order<30)]
#all_add_1=all_add_1[(all_add_1.nb_orders>=4)&(all_add_1.nb_orders<30)&(all_add_1.days_since_prior_order<30)&(all_add_1.total_items<30)]

print(all_add_1.shape)

del train_add
del test_add
del train_add_1
del test_add_1
gc.collect()

all_add=pd.concat([all_add,all_add_1])

del all_add_1
gc.collect()

all_add_y=all_add["label"]
del all_add["label"]

train_x=pd.read_csv("train_data.csv")
train_y=train_x["label"]
del train_x["label"]

#预测数据
#test_data=features(test_orders)
test_data=pd.read_csv("test_data.csv")
#test_data=train_x[:10000]      #测试用
print(test_data.shape)

gc.collect()

#分割数据
score_list=[0,0,0,0,0]
logloss_list=[]

train_num = train_x.shape[0]
from sklearn.cross_validation import KFold
kf=KFold(train_num,5,random_state=2017,shuffle=True)

st=0
preds=np.zeros((test_data.shape[0],5))
j=0
for tr_index,te_index in kf:
    # lgb算法
    #tr_x = lgb.Dataset(pd.concat([train_x.iloc[tr_index],all_add]).reset_index(drop=True), label=(train_y[tr_index].append(all_add_y).reset_index(drop=True)))

    test_all=train_x.iloc[te_index]
    y_test=train_y[te_index]

    train_all=pd.concat([train_x.iloc[tr_index],all_add])
    y_all=pd.concat([train_y[tr_index],all_add_y])

    tr_x = lgb.Dataset(train_all, label=y_all)
    te_x = lgb.Dataset(test_all, label=y_test)

    """
    # 线下评分数据
    true = train_x.iloc[te_index].copy()
    true["label"] = train_y[te_index]
    true = create_data(true, 1)
    #true.to_csv("true.csv", index=None)
    """

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 2 ** 5,
        'max_depth': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'seed': 2017,
        'nthread': 30,
    }
    num_round = 10000
    early_stopping_rounds = 100
    model = lgb.train(params, tr_x, num_round, valid_sets=te_x,
                      early_stopping_rounds=early_stopping_rounds,
                      )

    # 线下评分
    pre_score = model.predict(test_all, num_iteration=model.best_iteration)
    #pre_score = model.predict(test_all)
    score_data = test_all.copy()
    score_data['label'] = pre_score
    score_data = score_data[["order_id", "product_id", "label"]].copy()
    #保存训练的以后后期做stacking使用
    if st==0:
        stacking_data=score_data
    else:
        stacking_data=stacking_data.append(score_data)
    st+=1
    #test_all[:10000].to_csv("save_little_2_%s.csv"%st)
    #pd.DataFrame(y_test).to_csv("save_y_2_%s.csv"%st)
    logloss=log_loss(y_test,pre_score)
    logloss_list.append(logloss)
    print(logloss_list)

    pred = model.predict(test_data,num_iteration=model.best_iteration)
    preds[:,j]=pred
    j+=1

    del model
    gc.collect()

stacking_data.to_csv("stacking_data_v12_shuff_10000.csv", index=None)

with open("score_note.txt","a") as f:
    f.write(str(train_x.shape[1])+"\n"+str(score_list)+"=====>"+str(np.mean(score_list))+"\n"+str(logloss_list)+"=====>"+str(np.mean(logloss_list))+"\n")


preds=np.mean(preds,axis=1)
test_data['label'] = preds
test_data=test_data[["order_id","product_id","label"]].copy()
test_data.to_csv("sub_pre_cv5_v12_shuff_10000.csv",index=None)

