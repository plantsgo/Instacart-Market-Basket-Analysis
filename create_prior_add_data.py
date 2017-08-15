#encoding=utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import chain
import gc
from com_util import *
from sklearn.metrics import log_loss
from scipy.stats import mode

#"""
path="output/"
priors=pd.read_csv(path+"order_products__prior.csv")
train=pd.read_csv(path+"order_products__train.csv")
test=pd.read_csv(path+"order_products__test.csv")
all=train.append(test)

orders = pd.read_csv(path + 'orders.csv').fillna(0)
orders=merge_max(orders,["user_id"],"order_number","order_number_max")
orders["order_dow_days"]=orders["order_dow"].apply(lambda x:1 if x<=1 else 0)
orders["order_hour_of_day_split"]=orders["order_hour_of_day"].apply(lambda x:0 if 6<x<=12 else (1 if 12<x<=18 else 2))

path="input/"
products = pd.read_csv(path + 'products.csv')
departments = pd.read_csv(path + 'departments.csv')
aisles = pd.read_csv(path + 'aisles.csv')
product2vec = pd.read_csv('features/product2vec_begin_1.csv')
#用户最近三单
'''
orders_3=orders[orders.eval_set=="prior"]
orders_3=pd.DataFrame(orders_3.groupby("user_id").order_number.nlargest(3)).reset_index()
orders_3.columns=["user_id","level_1","order_number"]
del orders_3["level_1"]
orders_3=orders_3.merge(orders,on=["user_id","order_number"],how="left")

priors_3 = priors.merge(orders_3, on='order_id', how='inner')
priors_3 = priors_3.merge(products, on='product_id', how='left')
priors_3['user_product'] = priors_3.product_id + priors_3.user_id * 100000
priors_3['user_aisle'] = priors_3.aisle_id + priors_3.user_id * 1000
priors_3['user_department'] = priors_3.department_id + priors_3.user_id * 100
print priors_3.shape
'''


#优化内存
orders.order_dow = orders.order_dow.astype(np.int8)
orders.order_hour_of_day = orders.order_hour_of_day.astype(np.int8)
orders.order_number = orders.order_number.astype(np.int16)
orders.order_id = orders.order_id.astype(np.int32)
orders.user_id = orders.user_id.astype(np.int64)
orders.days_since_prior_order = orders.days_since_prior_order.astype(np.float32)

products.drop(['product_name'], axis=1, inplace=True)
products.aisle_id = products.aisle_id.astype(np.int64)
products.department_id = products.department_id.astype(np.int64)
products.product_id = products.product_id.astype(np.int64)

all.reordered = all.reordered.astype(np.int8)
all.add_to_cart_order = all.add_to_cart_order.astype(np.int16)

priors.order_id = priors.order_id.astype(np.int32)
priors.add_to_cart_order = priors.add_to_cart_order.astype(np.int16)
priors.reordered = priors.reordered.astype(np.int8)
priors.product_id = priors.product_id.astype(np.int64)
#构造特征
priors = priors.merge(orders, on='order_id', how='left')
priors = priors.merge(products, on='product_id', how='left')
priors=merge_count(priors,["order_id"],"reordered","order_id_products")
priors=merge_mean(priors,["order_id"],"reordered","order_id_products_reordered_rt")
priors['user_product'] = priors.product_id + priors.user_id * 100000
priors['user_aisle'] = priors.aisle_id + priors.user_id * 1000
priors['user_department'] = priors.department_id + priors.user_id * 100
priors["order_number_desc"]=priors.order_number_max-priors.order_number-1
priors["user_product_decay"]=0.8**priors["order_number_desc"]
priors["user_product_add_cart_decay"]=0.99**(priors["add_to_cart_order"]-1)*priors["user_product_decay"]

#priors["user_product_decay"]=1.0#/(priors["order_number_desc"]*0.4+1)
#priors["user_product_add_cart_decay"]=1.0/(priors["order_number_desc"]*0.15+1)*priors["user_product_decay"]

print(priors.head())

###############################################User X order##################################################
orders_count=pd.DataFrame(priors.groupby(["user_id","order_id"]).add_to_cart_order.count()).reset_index()
orders_count.columns=["user_id","order_id","order_id_count"]
user_order=pd.DataFrame(orders_count.groupby(["user_id"]).order_id.std()).reset_index()
user_order.columns=["user_id","order_count_std"]

###############################################################################################################
#基于最近的订单进行统计
#priors_near=merge_max(priors,["user_id"],"order_number","order_number_max")
#priors_near=priors_near[priors_near.order_number==priors_near.order_number_max].copy()

###############################################Products features#####################################################
#统计商品的总购买次数和重新购买次数
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).order_id.nunique().astype(np.int32)    #该商品被购买多少次
prods['orders_people'] = priors.groupby(priors.product_id).user_id.nunique().astype(np.int32)    #该商品被多少人购买
#prods['orders_people_count'] = priors.groupby(priors.product_id).user_id.count().astype(np.int32)    #该商品被多少次购买
prods['orders_people_rt']=prods['orders']/prods['orders_people']

prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
prods['add_to_cart_order_product_mean'] = priors['add_to_cart_order'].groupby(priors.product_id).mean().astype(np.float32)
#prods['order_id_products_mean'] = priors['order_id_products'].groupby(priors.product_id).mean().astype(np.float32)   #出现该products的订单大小
user_product_count=pd.DataFrame(priors.groupby(["user_id","product_id"]).order_id.nunique()).reset_index()
user_product_count.columns=["user_id","product_id","orders_user_product_count"]
user_product_count=user_product_count[user_product_count.orders_user_product_count==1].copy()

product_only_one_count=pd.DataFrame()
product_only_one_count["product_only_one_count"]=user_product_count.groupby('product_id').user_id.count().astype(np.int16)

products = products.join(prods, on='product_id')
products = products.join(product_only_one_count, on='product_id')
products["product_only_one_count_rt"]=products["product_only_one_count"]/products["orders_people"]

del prods
###################################################################################################################
###############################################Aisle features#####################################################
#统计Aisle 的总购买次数和重新购买次数
ais = pd.DataFrame()
ais['orders_aisle'] = priors.groupby(priors.aisle_id).order_id.count().astype(np.float32)
ais['reorders_aisle'] = priors['reordered'].groupby(priors.aisle_id).sum().astype(np.float32)
ais['reorder_rate_aisle'] = (ais.reorders_aisle / ais.orders_aisle).astype(np.float32)
ais['add_to_cart_order_product_mean_aisle'] = priors['add_to_cart_order'].groupby(priors.aisle_id).mean().astype(np.float32)

products = products.join(ais, on='aisle_id')
del ais
###################################################################################################################
###############################################Department  features#####################################################
#统计Department的总购买次数和重新购买次数
deps = pd.DataFrame()
deps['orders_department'] = priors.groupby(priors.department_id).order_id.count().astype(np.float32)
deps['reorders_department'] = priors['reordered'].groupby(priors.department_id).sum().astype(np.float32)
deps['reorder_rate_department'] = (deps.reorders_department / deps.orders_department).astype(np.float32)
deps['add_to_cart_order_product_mean_department'] = priors['add_to_cart_order'].groupby(priors.department_id).mean().astype(np.float32)

products = products.join(deps, on='department_id')
del deps
###################################################################################################################
###############################################User features########################################################
#通过priors统计user购买过哪些products
usr = pd.DataFrame()
usr['days_since_prior_order_mean'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').order_id.nunique().astype(np.int16)
usr['order_hour_of_day_mean'] = orders.groupby('user_id')["order_hour_of_day"].mean().astype(np.float32)
usr['days_since_prior_order_list'] = orders.groupby('user_id')['days_since_prior_order'].apply(list)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').product_id.count().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['reorder_products_count'] = priors.groupby('user_id')['reordered'].sum().astype(np.int16)
#users['order_id_products_max'] = priors.groupby('user_id')['order_id_products'].max().astype(np.int16)   #用户每单的购买最大值
#users['order_id_products_min'] = priors.groupby('user_id')['order_id_products'].min().astype(np.int16)   #用户每单的购买最小值
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
print(users.head(10))
del usr
users['average_basket'] = (users.total_items / (users.nb_orders-1)).astype(np.float32)   #平均每个订单购买多少个
#users['reorder_products_count_rt'] = (users.reorder_products_count / users.nb_orders).astype(np.float32)   #平均每个订单重复购买多少个
users['average_distinct_basket'] = (users.total_distinct_items / users.nb_orders).astype(np.float32)   #平均每个订单购买多少不同的
users['distinct_basket_rt'] = (users.total_distinct_items / users.total_items).astype(np.float32)   #购买重复率
users['days_since_prior_order_list_30_count']=users['days_since_prior_order_list'].apply(lambda x:x.count(30))
users['days_since_prior_order_list_30_count_rt'] = (users.days_since_prior_order_list_30_count / users.nb_orders).astype(np.float32)
#users['average_basket_everyday'] = (users.average_basket / users.order_hour_of_day_mean).astype(np.float32)   #平均每天购买多少个
gc.collect()


#user_only_one_count=pd.DataFrame()
#user_only_one_count["user_only_one_count"]=user_product_count.groupby('user_id').product_id.count().astype(np.int16)
#del user_product_count
#gc.collect()
###################################################################################################################

################################################User X Products features###########################################
#用户和商品的时间队列
up_time=pd.DataFrame()
up_time["up_time_list"]=priors.groupby("user_product").order_number.apply(list)   #用户商品经历过哪些订单
up_time["up_time_list"]=up_time["up_time_list"].apply(lambda x:sorted(x))
print(up_time.head(10))

#第一次和最后一次单号
userXproduct=priors[["user_product","order_number"]].copy()
userXproduct=userXproduct.sort_values("order_number")
userXproduct_last=userXproduct.drop_duplicates("user_product",keep="last")
userXproduct_last.columns=["user_product","last_order_number"]
userXproduct_first=userXproduct.drop_duplicates("user_product",keep="first")
userXproduct_first.columns=["user_product","first_order_number"]

#倒数第二次的单号
userXproduct_2=userXproduct.merge(userXproduct_last,on="user_product",how="left")
userXproduct_2=userXproduct_2[userXproduct_2.order_number!=userXproduct_2.last_order_number][["user_product","order_number"]].copy()
userXproduct_2=userXproduct_2.sort_values("order_number")
userXproduct_last_2=userXproduct_2.drop_duplicates("user_product",keep="last")
userXproduct_last_2.columns=["user_product","last_order_number_2"]

userXproduct=priors[["user_product","days_since_prior_order"]].copy().fillna(0)
userXproduct=userXproduct.sort_values("days_since_prior_order")
userXproduct_last_days=userXproduct.drop_duplicates("user_product",keep="last")
userXproduct_last_days.columns=["user_product","last_days_since_prior_order"]

#user_product=pd.DataFrame(priors.groupby(["user_product"]).order_id.nunique()).reset_index()
#user_product.columns=["user_product","orders_user_product"]

user_product=pd.DataFrame(priors.groupby(["user_product"]).user_product_decay.sum()).reset_index()
user_product.columns=["user_product","orders_user_product"]

#user_product=pd.DataFrame(priors.groupby(["user_product"]).user_product_add_cart_decay.sum()).reset_index()
#user_product.columns=["user_product","orders_user_product"]

#用户商品对和时间的关系
user_product_order_dow_days=pd.DataFrame(priors.groupby(["user_product","order_dow_days"]).reordered.mean()).reset_index()
user_product_order_dow_days.columns=["user_product","order_dow_days","user_product_order_dow_days_reordered_rt"]

user_product_order_hour_of_day_split=pd.DataFrame(priors.groupby(["user_product","order_hour_of_day_split"]).reordered.mean()).reset_index()
user_product_order_hour_of_day_split.columns=["user_product","order_hour_of_day_split","user_product_order_hour_of_day_split_reordered_rt"]

up=pd.DataFrame()
up['user_product_add_to_cart_order_mean'] = priors.groupby("user_product").add_to_cart_order.mean()              #每个用户对不同product的偏爱
up['order_id_products_reordered_rt_mean'] = priors.groupby("user_product").order_id_products_reordered_rt.mean()  #用户商品来自订单的转购率，反应是否是噪音订单
up['order_id_products_reordered_rt_sum'] = priors.groupby("user_product").order_id_products_reordered_rt.sum()  #来自不同订单的product应该拥有不同的权重
up['user_product_count'] = priors.groupby("user_product").product_id.count()  #user买过多少该product
up['user_product_dow_mode'] = priors.groupby("user_product").order_dow.apply(lambda x: np.bincount(x).argmax())  #用户商品对某天的偏爱
up['user_product_add_to_cart_order_mode'] = priors.groupby("user_product").add_to_cart_order.apply(lambda x: np.bincount(x).argmax())  #用户商品对某天的偏爱


#up['user_product_add_to_cart_order_std'] = priors.groupby("user_product").add_to_cart_order.std()

user_product=user_product.join(up,on="user_product",how="left")
user_product=user_product.join(up_time,on="user_product",how="left")
user_product=user_product.merge(userXproduct_first,on="user_product",how="left")
user_product=user_product.merge(userXproduct_last,on="user_product",how="left")
user_product=user_product.merge(userXproduct_last_2,on="user_product",how="left").fillna(1)
user_product["last_order_number_gap"]=user_product["last_order_number"]-user_product["last_order_number_2"]

print(user_product.head(10))

priors=priors.sort_values(["user_id","order_number","add_to_cart_order"])
print(priors.head(10))
priors["up_range"]=range(priors.shape[0])
up_features=pd.DataFrame()
up_features["up_num_list"]=priors.groupby(['user_product'])['up_range'].apply(list)
print(up_features.head(10))

###################################################################################################################

###################################################User X Aisles features########################################

userXaisle=priors[["user_aisle","order_number"]].copy()
userXaisle=userXaisle.sort_values("order_number")
userXaisle_last=userXaisle.drop_duplicates("user_aisle",keep="last")
userXaisle_last.columns=["user_aisle","last_order_number_aisle"]
userXaisle_first=userXaisle.drop_duplicates("user_aisle",keep="first")
userXaisle_first.columns=["user_aisle","first_order_number_aisle"]

#倒数第二次的单号
'''
userXaisle_2=userXaisle.merge(userXaisle_last,on="user_aisle",how="left")
userXaisle_2=userXaisle_2[userXaisle_2.order_number!=userXaisle_2.last_order_number_aisle][["user_aisle","order_number"]].copy()
userXaisle_2=userXaisle_2.sort_values("order_number")
userXaisle_last_2=userXaisle_2.drop_duplicates("user_aisle",keep="last")
userXaisle_last_2.columns=["user_aisle","last_order_number_aisle_2"]
'''

user_aisle=pd.DataFrame(priors.groupby(["user_aisle"]).order_id.nunique()).reset_index()
user_aisle.columns=["user_aisle","orders_user_aisle"]

ua=pd.DataFrame()
ua['user_aisle_product_nunique'] = priors['product_id'].groupby(priors.user_aisle).nunique().astype(np.int16)
#ua['user_aisle_product_count'] = priors['product_id'].groupby(priors.user_aisle).count().astype(np.int16)
#ua['user_aisle_product_add_mean'] = priors.groupby("user_aisle").add_to_cart_order.mean()              #每个用户对不同aisle的偏爱

user_aisle=user_aisle.join(ua,on="user_aisle",how="left")
user_aisle=user_aisle.merge(userXaisle_first,on="user_aisle",how="left")
user_aisle=user_aisle.merge(userXaisle_last,on="user_aisle",how="left")
#user_aisle=user_aisle.merge(userXaisle_last_2,on="user_aisle",how="left").fillna(1)
#user_aisle["last_order_number_aisle_gap"]=user_aisle["last_order_number_aisle"]-user_aisle["last_order_number_aisle_2"]
################################################################################################################

###################################################User X Department features########################################

userXdepartment=priors[["user_department","order_number"]].copy()
userXdepartment=userXdepartment.sort_values("order_number")
userXdepartment_last=userXdepartment.drop_duplicates("user_department",keep="last")
userXdepartment_last.columns=["user_department","last_order_number_department"]
userXdepartment_first=userXdepartment.drop_duplicates("user_department",keep="first")
userXdepartment_first.columns=["user_department","first_order_number_department"]

user_department=pd.DataFrame(priors.groupby(["user_department"]).order_id.nunique()).reset_index()
user_department.columns=["user_department","orders_user_department"]

de=pd.DataFrame()
de['user_department_product_nunique'] = priors['product_id'].groupby(priors.user_department).nunique().astype(np.int16)
#de['user_department_product_count'] = priors['product_id'].groupby(priors.user_department).count().astype(np.int16)
#de['user_department_product_add_mean'] = priors.groupby("user_department").add_to_cart_order.mean()              #每个用户对不同department的偏爱

user_department=user_department.join(de,on="user_department",how="left")
user_department=user_department.merge(userXdepartment_first,on="user_department",how="left")
user_department=user_department.merge(userXdepartment_last,on="user_department",how="left")
################################################################################################################
###################################################Products X time features########################################
products_times=pd.DataFrame(priors.groupby(["product_id"]).order_id.count()).reset_index()
products_times.columns=["product_id","product_count"]

products_hour=pd.DataFrame(priors.groupby(["product_id","order_hour_of_day"]).order_id.count()).reset_index()
products_hour.columns=["product_id","order_hour_of_day","product_hour_count"]
products_hour=products_hour.merge(products_times,on="product_id",how="left")
products_hour["product_hour_count_rt"]=products_hour["product_hour_count"]/products_hour["product_count"]
del products_hour["product_count"]
del products_hour["product_hour_count"]

'''
products_day=pd.DataFrame(priors.groupby(["product_id","order_dow"]).order_id.count()).reset_index()
products_day.columns=["product_id","order_dow","product_day_count"]
products_day=products_day.merge(products_times,on="product_id",how="left")
products_day["product_day_count_rt"]=products_day["product_day_count"]/products_day["product_count"]
del products_day["product_count"]
del products_day["product_day_count"]

products_day_hour=pd.DataFrame(priors.groupby(["product_id","order_dow","order_hour_of_day"]).order_id.count()).reset_index()
products_day_hour.columns=["product_id","order_dow","order_hour_of_day","product_day_hour_count"]
products_day_hour=products_day_hour.merge(products_times,on="product_id",how="left")
products_day_hour["product_day_hour_count_rt"]=products_day_hour["product_day_hour_count"]/products_day_hour["product_count"]
del products_day_hour["product_count"]
del products_day_hour["product_day_hour_count"]
'''
###############################################order_number features########################################################
order_number=pd.DataFrame()
order_number['order_number_reorder_rt_mean'] = priors.groupby('order_number')['reordered'].mean()
################################################################################################################
################################################################################################################
#构造训练和预测集
train_orders = orders[orders.eval_set == 'train']
test_orders = orders[orders.eval_set == 'test']

#标签
label=all[['order_id', 'product_id']].copy()
label["label"]=1

del orders["order_number_max"]
def features(selected_orders, labels_given=False):
    order_list = []
    user_list=[]
    product_list = []
    i = 0
    #根据有交互的user和product对构造训练集
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        user_list += [user_id] * len(user_products)
    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list,'user_id':user_list}, dtype=np.int64)

    if labels_given:
        df=df.merge(label,on=['order_id', 'product_id'],how="left").fillna(0)

    del order_list
    del product_list
    del user_list

    #merge特征
    df=df.merge(products,on="product_id",how="left")
    #df=df.merge(user_order,on="user_id",how="left")
    df=df.join(users,on="user_id")
    #df=df.join(user_only_one_count,on="user_id")

    df=df.merge(orders, on=["order_id","user_id"], how="left")
    df = df.merge(product2vec, on="product_id", how="left")
    #df = df.merge(aisle2vec, on="aisle_id", how="left")
    #df = df.merge(department2vec, on="department_id", how="left")

    df['user_product'] = df.product_id + df.user_id * 100000
    df['user_aisle'] = df.aisle_id + df.user_id * 1000
    df['user_department'] = df.department_id + df.user_id * 100
    df=df.merge(user_product,on="user_product",how="left")
    df=df.join(up_features,on="user_product",how="left")
    df=df.join(order_number,on="order_number",how="left")
    df=df.merge(user_aisle,on="user_aisle",how="left")
    df=df.merge(user_department,on="user_department",how="left")
    df=df.merge(products_hour,on=["product_id","order_hour_of_day"],how="left")

    df=df.merge(user_product_order_dow_days,on=["user_product","order_dow_days"],how="left")
    df=df.merge(user_product_order_hour_of_day_split,on=["user_product","order_hour_of_day_split"],how="left")
    #df=df.merge(products_day,on=["product_id","order_dow"],how="left")
    #df=df.merge(products_day_hour,on=["product_id","order_dow","order_hour_of_day"],how="left")
    #构造特征
    df["up_order_rate"]=df["orders_user_product"]/(df["nb_orders"]-1)
    df["up_items_rate"]=df["orders_user_product"]/df["total_items"]
    df["up_product_rate"]=df["orders_user_product"]/(df["orders_user_aisle"])

    #df["user_aisle_product_count_rt"]=df["user_aisle_product_count"]/(df["total_items"])
    #df["user_department_product_count_rt"]=df["user_department_product_count"]/(df["total_items"])

    #df["user_only_one_count_rt"]=df["user_only_one_count"]/(df["total_items"])
    #df["user_only_one_count_nunique_rt"]=df["user_only_one_count"]/(df["total_distinct_items"])

    df["since_last_order"]=df["nb_orders"]-df["last_order_number"]

    df["order_hour_of_day_dif"]=df["order_hour_of_day"]-df["order_hour_of_day_mean"]
    df["since_first_order"]=df["nb_orders"]-df["first_order_number"]+1
    df["up_order_rate_since_first_order"]=df["orders_user_product"]/(df["nb_orders"] - df["first_order_number"])
    df["days_since_last_order"]=list(chain(map(lambda a,b:sum(b[0-int(a):]),df["since_last_order"],df["days_since_prior_order_list"])))
    df["days_since_first_order"]=list(chain(map(lambda a,b:sum(b[0-int(a):]),df["since_first_order"],df["days_since_prior_order_list"])))

    #df["days_since_last_order_gap"]=list(chain(map(lambda a,b,c:sum(c[int(a)-1:int(b)]),df["last_order_number_2"],df["last_order_number"],df["days_since_prior_order_list"])))
    #df["days_since_last_order_dif"] = df["days_since_last_order"] - df["days_since_last_order_gap"]

    df=merge_mean(df,["product_id"],"since_last_order","products_since_last_order_mean")
    df=merge_mean(df,["product_id"],"days_since_last_order","products_days_since_last_order_mean")
    df=merge_mean(df,["product_id"],"last_order_number_gap","products_last_order_number_gap_mean")

    #df=merge_max(df,["user_id"],"up_order_rate","up_order_rate_max")
    #df=merge_mean(df,["aisle_id"],"last_order_number_aisle_gap","last_order_number_aisle_gap_mean")

    df["since_last_order_dif"]=df["products_last_order_number_gap_mean"]-df["since_last_order"]
    #df["since_last_order_aisle_dif"]=df["last_order_number_aisle_gap_mean"]-df["since_last_order"]
    df["average_basket_out"]=df["user_product_add_to_cart_order_mean"]-df["average_basket"]

    df["orders_user_product_order_skip_mean"] = (df["last_order_number"] - df["first_order_number"])/(df["orders_user_product"]-1)
    df["orders_user_product_days_skip_mean"] = (df["days_since_last_order"] - df["days_since_first_order"])/(df["orders_user_product"]-1)
    df["since_last_order_rt_self_mean"]=df["since_last_order"]/df["orders_user_product_order_skip_mean"]
    df["days_since_last_order_rt_self_mean"]=df["days_since_last_order"]/df["orders_user_product_days_skip_mean"]

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

    df["up_skip_days_list"]=list(chain(map(lambda a,b:skip_days(a,b),df["days_since_prior_order_list"],df["up_time_list"])))    #间隔天数列表
    df["up_skip_days_max"]=df["up_skip_days_list"].apply(max)
    df["up_skip_days_min"]=df["up_skip_days_list"].apply(min)
    df["up_skip_days_mean"]=df["up_skip_days_list"].apply(lambda x:np.mean(x))

    #df["up_skip_days_max_dif"]=df["days_since_last_order"]-df["up_skip_days_max"]
    #df["up_skip_days_min_dif"]=df["days_since_last_order"]-df["up_skip_days_min"]
    def skip_products(a,b):
        up_skip_days_list=[]
        if len(b)==1:
            up_skip_days_list=[200]
        else:
            for t in range(len(a)-1):
                t1=b[t]
                t2=b[t+1]
                days_sum=t2-t1
                up_skip_days_list.append(days_sum)
        return up_skip_days_list

    df["up_skip_products_list"]=list(chain(map(lambda a,b:skip_products(a,b),df["up_num_list"],df["up_time_list"])))    #间隔列表
    df["up_skip_products_max"]=df["up_skip_products_list"].apply(max)
    df["up_skip_products_min"]=df["up_skip_products_list"].apply(min)
    df["up_skip_products_mean"]=df["up_skip_products_list"].apply(lambda x:np.mean(x))
    del df["up_skip_products_list"]
    del df["up_num_list"]

    #'''
    def skip_orders(a,b):
        if len(a)==1:
            return [b-a[0]+1]
        up_time_gap_list=[]
        for t in range(1,len(a)):
            gap=a[t]-a[t-1]
            up_time_gap_list.append(gap)
        return up_time_gap_list

    df["up_time_gap_list"]=list(chain(map(lambda a,b:skip_orders(a,b),df["up_time_list"],df["nb_orders"])))    #间隔订单列表

    df["up_time_gap_mode"]=df["up_time_gap_list"].apply(lambda x:mode(x)[0][0])  #这个商品正常间隔多少单购买
    del df["up_time_gap_list"]
    #'''

    df=merge_mean(df,["product_id"],"up_skip_days_mean","product_skip_days_mean")

    df["user_product_rt"]=df["user_product_count"]/df["total_items"]
    df["user_product_like"]=df["orders_user_product"]*df["user_product_rt"]

    df["dow_equal"]=list(chain(map(lambda a,b:1 if a==b else 0,df["user_product_dow_mode"],df["order_dow"]))) #购买的当时是不是购买次数最多的那一次



    del df["eval_set"]
    del df["days_since_prior_order_list"]
    del df["up_time_list"]
    del df["up_skip_days_list"]
    del df["all_products"]

    del df["user_product"]
    del df["user_aisle"]
    del df["user_department"]
    del df["user_id"]

    del df["order_number"]

    del df["order_dow_days"]
    del df["order_hour_of_day_split"]

    return df

train_add=features(train_orders,labels_given=True)
train_add.to_csv("output/train_add.csv",index=None)
print(train_add.shape)
del train_add
gc.collect()

test_add=features(test_orders,labels_given=True)
test_add.to_csv("output/test_add.csv",index=None)
print(test_add.shape)

