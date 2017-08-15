#encoding=utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import chain
from scipy.stats import mode
import gc
#"""
path="input/"
priors=pd.read_csv(path+"order_products__prior.csv")
train=pd.read_csv(path+"order_products__train.csv")
orders = pd.read_csv(path + 'orders.csv').fillna(0)
products = pd.read_csv(path + 'products.csv')
departments = pd.read_csv(path + 'departments.csv')
aisles = pd.read_csv(path + 'aisles.csv')


products.drop(['product_name'], axis=1, inplace=True)
priors = priors.merge(orders, on='order_id', how='left')
priors = priors.merge(products, on='product_id', how='left')

###############################################User X order##################################################
orders_count=pd.DataFrame(priors.groupby(["user_id","order_id"]).add_to_cart_order.count()).reset_index()
orders_count.columns=["user_id","order_id","order_id_count"]

user_order=pd.DataFrame()
user_order["order_count_std"]=orders_count.groupby(["user_id"]).order_id.std()

orders_re_count=pd.DataFrame(priors.groupby(["order_id"]).reordered.sum()).reset_index()
orders_re_count.columns=["order_id","orders_re_count"]

orders_re_mean=pd.DataFrame(priors.groupby(["order_id"]).reordered.mean()).reset_index()
orders_re_mean.columns=["order_id","orders_re_mean"]

orders_all_count=pd.DataFrame(priors.groupby(["order_id"]).reordered.count()).reset_index()
orders_all_count.columns=["order_id","orders_all_count"]


pri_orders=orders[(orders.eval_set=="prior")&(orders.order_number!=1)].copy()
pri_orders=pri_orders.merge(orders_re_count,on="order_id",how="left")
pri_orders=pri_orders.merge(orders_re_mean,on="order_id",how="left")

user_orders=pd.DataFrame()
user_orders['user_id_orders_re_count_list'] = pri_orders.groupby('user_id')['orders_re_count'].apply(list)
#user_orders['user_id_orders_re_count_list']=user_orders['user_id_orders_re_count_list'].apply(lambda x:x[-3:])
user_orders['user_id_orders_re_count_max'] = user_orders['user_id_orders_re_count_list'].apply(max)
user_orders['user_id_orders_re_count_min'] = user_orders['user_id_orders_re_count_list'].apply(min)
user_orders['user_id_orders_re_count_mean'] = user_orders['user_id_orders_re_count_list'].apply(lambda x:np.mean(x))
user_orders['user_id_orders_re_count_gap']=user_orders['user_id_orders_re_count_max']-user_orders['user_id_orders_re_count_min']

user_orders['user_id_orders_re_mean_list'] = pri_orders.groupby('user_id')['orders_re_mean'].apply(list)
#user_orders['user_id_orders_re_mean_list']=user_orders['user_id_orders_re_mean_list'].apply(lambda x:x[-3:])
user_orders['user_id_orders_re_mean_max'] = user_orders['user_id_orders_re_mean_list'].apply(max)
user_orders['user_id_orders_re_mean_min'] = user_orders['user_id_orders_re_mean_list'].apply(min)
user_orders['user_id_orders_re_mean_mean'] = user_orders['user_id_orders_re_mean_list'].apply(lambda x:np.mean(x))
user_orders['user_id_orders_re_mean_gap']=user_orders['user_id_orders_re_mean_max']-user_orders['user_id_orders_re_mean_min']

dow_hour_mean=pd.DataFrame(pri_orders.groupby(['order_dow','order_hour_of_day'])['orders_re_mean'].mean()).reset_index()
dow_hour_mean.columns=['order_dow','order_hour_of_day','dow_hour_mean']


user_orders['user_id_orders_re_0_times'] = user_orders['user_id_orders_re_mean_list'].apply(lambda x:x.count(0))    #为None的次数

del user_orders["user_id_orders_re_count_list"]
del user_orders["user_id_orders_re_mean_list"]
gc.collect()
print user_orders.head(10)


#增加每个订单的重购次数，重购率，购买商品数
orders_new=orders.merge(orders_re_count,on=["order_id"],how="left")
orders_new=orders_new.merge(orders_re_mean,on=["order_id"],how="left")
orders_new=orders_new.merge(orders_all_count,on=["order_id"],how="left")
###################################################################################################################

###############################################order_number features########################################################
order_number=pd.DataFrame()
order_number['order_number_reorder_rt_mean'] = priors.groupby('order_number')['reordered'].mean()
order_number['order_number_reorder_rt_sum'] = priors.groupby('order_number')['reordered'].sum()

###############################################User features############################################################
#通过priors统计user购买过哪些products
usr = pd.DataFrame()
usr['days_since_prior_order_mean'] = orders.groupby('user_id')['days_since_prior_order'].sum().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)    #用户订单个数，和order_number重复了，需要删掉一个
usr['days_since_prior_order_mean']=usr['days_since_prior_order_mean']/(usr['nb_orders']-1)        #用户平均每个订单间隔天数
usr['order_hour_of_day_mean'] = orders.groupby('user_id')["order_hour_of_day"].mean().astype(np.float32)  #用户每天什么时候购买
usr['order_dow_mean'] = orders.groupby('user_id')["order_dow"].mean().astype(np.float32)  #用户每周什么时候购买
usr['order_dow_mode'] = orders.groupby("user_id").order_dow.apply(lambda x:mode(x)[0][0])  #用户每周什么时候最多次购买

usr['days_since_prior_order_list'] = orders.groupby('user_id')['days_since_prior_order'].apply(list)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)    #用户总共买过多少商品
users['user_reordered'] = priors.groupby('user_id')["reordered"].sum().astype(np.int16)    #用户重购次数
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)    #用户购买过多少不同的商品
users = users.join(usr)
print(users.head(10))
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)   #平均每个订单购买多少个商品
users['user_reordered_rt'] = (users.user_reordered / (users.nb_orders-1)).astype(np.float32)   #平均每个订单重购多少个商品
users['average_distinct_basket'] = (users.total_distinct_items / users.nb_orders).astype(np.float32)   #平均每个订单购买多少不同的商品
users['distinct_basket_rt'] = (users.total_distinct_items / users.total_items).astype(np.float32)   #购买重复率高低
users['days_since_prior_order_list_30_count']=users['days_since_prior_order_list'].apply(lambda x:x.count(30))
users['days_since_prior_order_list_30_count_rt'] = (users.days_since_prior_order_list_30_count / users.nb_orders).astype(np.float32)
gc.collect()

###################################################################################################################
order_streaks = pd.read_csv('features/order_streaks_features.csv')
streaks=pd.DataFrame()
streaks["user_streaks_sum"]=order_streaks.groupby('user_id')['order_streak'].sum()
streaks["user_streaks_mean"]=order_streaks.groupby('user_id')['order_streak'].mean()


################################################user time features########################################################
user_dow=pd.DataFrame(priors.groupby(["user_id","order_dow"]).order_id.count()).reset_index()
user_dow.columns=["user_id","order_dow","order_dow_count"]

##########################################################################################################################
#判断是否为None单
stacking_data=pd.read_csv("stacking_data_v12.csv")
sub_pre_cv5=pd.read_csv("sub_pre_cv5_v12.csv")
none_deal=stacking_data.append(sub_pre_cv5)
none_deal["label_d"]=1-none_deal["label"]
none=pd.DataFrame()
none["label_d_list"]=none_deal.groupby("order_id").label_d.apply(list)
none["label_list"]=none_deal.groupby("order_id").label.apply(list)
from functools import reduce
def mul(l):
    return reduce(lambda x, y: x * y, l)
none["label_d_mul"]=none["label_d_list"].apply(mul)
none["label_max"]=none_deal.groupby("order_id").label.max()
none["label_sum"]=none_deal.groupby("order_id").label.sum()
none["label_mean"]=none_deal.groupby("order_id").label.mean()
none["label_median"]=none_deal.groupby("order_id").label.median()
none["label_min"]=none_deal.groupby("order_id").label.min()
########################################################################################################
######################################################统计order表的相关信息#########################################
orders_feature1=pd.DataFrame()
orders_feature1["none_number_list"]=orders_new[orders_new.orders_re_count==0].groupby("user_id").order_number.apply(list)   #用户none经历过哪些订单
orders_feature1["none_number_list"]=orders_feature1["none_number_list"].apply(lambda x:sorted(x))

print orders_feature1.head(10)

orders_feature2=pd.DataFrame()
orders_feature2["orders_all_count_list"]=orders_new.groupby("user_id").orders_all_count.apply(list)

orders_feature3=pd.DataFrame()
orders_feature3["order_number_count"]=orders_new[orders_new.orders_re_count==0].groupby("order_number").user_id.count()


######################
#根据优化方法
th_features=pd.read_csv("features/th_features.csv")
optimizer_features=pd.read_csv("features/optimizer_features.csv")

def count_over(a,b):
    count=0
    for i in a:
        if i>b:
            count+=1
    return count

none["label_over_mean"]=map(lambda a,b:count_over(a,b),none["label_list"],none["label_mean"])
none["label_over_0.2"]=map(lambda a:count_over(a,0.2),none["label_list"])

del none["label_d_list"]
del none["label_list"]



#预测重购的个数
label=pd.DataFrame(train.groupby("order_id").reordered.sum()).reset_index()
label.columns=["order_id","label"]
label["label"]=label["label"].apply(lambda x:1 if x==0 else 0)

def features(selected_orders, labels_given=False):
    df=selected_orders[["order_id","user_id"]].copy()
    df = df.join(users, on="user_id")
    df = df.join(user_orders, on="user_id")
    df = df.join(orders_feature1, on="user_id")
    df = df.join(orders_feature2, on="user_id")


    #df = df.join(streaks, on="user_id")
    df=df.merge(orders, on=["order_id","user_id"], how="left")
    #df=df.merge(user_dow, on=["user_id","order_dow"], how="left")
    df=df.merge(dow_hour_mean, on=["order_dow","order_hour_of_day"], how="left")
    df=df.join(user_order, on="user_id", how="left")
    df=df.join(order_number, on="order_number", how="left")
    df=df.join(none,on="order_id")
    df=df.merge(th_features,on="order_id",how="left")
    df=df.merge(optimizer_features,on="order_id",how="left")
    if labels_given:
        df = df.merge(label, on=['order_id'], how="left")

    df["user_id_orders_re_0_times_rt"]=df["user_id_orders_re_0_times"]/df["nb_orders"]

    def skip_days(a,b):
        up_skip_days_list=[]
        if len(b)==1:
            up_skip_days_list=[365]
        else:
            for t in range(len(b)-1):
                t1=b[t]
                t2=b[t+1]
                days_sum=sum(a[t1:t2])
                up_skip_days_list.append(days_sum)
        return up_skip_days_list

    df["up_skip_days_list"]=list(chain(map(lambda a,b:skip_days(a,b),df["days_since_prior_order_list"],df["none_number_list"])))    #间隔天数列表
    df["up_skip_days_max"]=df["up_skip_days_list"].apply(max)
    df["up_skip_days_min"]=df["up_skip_days_list"].apply(min)
    df["up_skip_days_mean"]=df["up_skip_days_list"].apply(lambda x:np.mean(x))
    del df["up_skip_days_list"]

    #'''
    def skip_products(a,b):
        up_skip_days_list=[]
        if len(b)==1:
            up_skip_days_list=[0]
        else:
            for t in range(len(b)-1):
                t1=b[t]
                t2=b[t+1]
                days_sum=sum(a[t1+1:t2])
                up_skip_days_list.append(days_sum)
        return up_skip_days_list

    df["up_skip_products_list"]=list(chain(map(lambda a,b:skip_products(a,b),df["orders_all_count_list"],df["none_number_list"])))    #间隔天数列表
    df["up_skip_products_max"]=df["up_skip_products_list"].apply(max)
    df["up_skip_products_min"]=df["up_skip_products_list"].apply(min)
    df["up_skip_products_mean"]=df["up_skip_products_list"].apply(lambda x:np.mean(x))
    del df["up_skip_products_list"]
    #'''

    def skip_orders(a,b):
        if len(a)==1:
            return [b-a[0]+1]
        up_time_gap_list=[]
        for t in range(1,len(a)):
            gap=a[t]-a[t-1]
            up_time_gap_list.append(gap)
        return up_time_gap_list

    df["up_time_gap_list"]=list(chain(map(lambda a,b:skip_orders(a,b),df["none_number_list"],df["nb_orders"])))    #间隔订单列表

    df["up_time_gap_max"]=df["up_time_gap_list"].apply(max)  #最大间隔了多少次订单重购的
    df["up_time_gap_min"]=df["up_time_gap_list"].apply(min)  #最小间隔了多少次订单重购的
    del df["up_time_gap_list"]

    df["none_last_order_num"]=df["none_number_list"].apply(lambda x:x[-1])
    df["since_last_none"]=df["order_number"]-df["none_last_order_num"]

    del df["none_number_list"]
    del df["days_since_prior_order_list"]
    del df["orders_all_count_list"]

    del df["eval_set"]
    del df["all_products"]
    del df["order_number"]
    return df

#构造训练和预测集
train_orders = orders[orders.eval_set == 'train']
test_orders = orders[orders.eval_set == 'test']

train_x=features(train_orders,labels_given=True)
train_x.to_csv("train_x_num.csv",index=None)
train_x.corr().to_csv("none_corr.csv")
#"""
#train_x=pd.read_csv("train_x_num.csv")

#lgb算法
test_data=features(test_orders)
train_num = train_x.shape[0]
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss,roc_auc_score
kf=KFold(train_num,5,random_state=2017,shuffle=True)

preds=np.zeros((test_data.shape[0],5))
stacking_data=np.zeros((train_x.shape[0],1))
st=0
j=0
score_list=[]
for tr_index,te_index in kf:
    # lgb算法
    tr_x = train_x.iloc[tr_index].copy()
    te_x = train_x.iloc[te_index].copy()

    tr_y=tr_x["label"]
    te_y=te_x["label"]

    del tr_x["label"]
    del te_x["label"]

    tr_x_d = lgb.Dataset(tr_x, label=tr_y)
    te_x_d = lgb.Dataset(te_x, label=te_y)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #'metric': 'binary_logloss',
        'metric': 'auc',
        'num_leaves': 2 ** 3,
        'max_depth': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.85,
        'bagging_freq': 10,
        'seed': 2017,
        'nthread': 30,
    }

    num_round = 3000
    early_stopping_rounds = 100
    model = lgb.train(params, tr_x_d, num_round, valid_sets=[tr_x_d,te_x_d],
                          early_stopping_rounds=early_stopping_rounds,
                          )

    pre=model.predict(te_x,num_iteration=model.best_iteration)
    #保存训练的以后后期做stacking使用
    stacking_data[te_index,0]=pre

    score=roc_auc_score(te_y,pre)
    score_list.append(score)

    pred = model.predict(test_data, num_iteration=model.best_iteration)
    preds[:, j] = pred
    j += 1

    del model
    gc.collect()

stacking_data=pd.DataFrame(stacking_data)
stacking_data.columns=["reorder_num"]
stacking_data["order_id"]=train_x["order_id"].values
stacking_data.to_csv("stacking_none_prob.csv", index=None)

print score_list
print np.mean(score_list)
with open("score_auc.txt","a") as f:
    f.write(str(score_list)+"  "+str(np.mean(score_list))+"\n")
preds=np.mean(preds,axis=1)
pre=pd.DataFrame(preds)
pre.columns=["reorder_num"]
pre["order_id"]=test_data["order_id"].values
pre.to_csv("none_prob.csv",index=None)