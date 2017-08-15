#encoding=utf8
import pandas as pd
arbor=pd.read_csv("prediction_arboretum.csv")
print arbor.head()
arbor.columns=["order_id","prediction_arbor","product_id"]
lgbm=pd.read_csv("prediction_lgbm.csv")
lgbm.columns=["order_id","prediction_lgbm","product_id"]
print lgbm.head()
mine_1=pd.read_csv("sub_pre_cv5_v12.csv")
mine_1.columns=["order_id","product_id","label_1"]
mine_2=pd.read_csv("sub_pre_cv5_v12_shuff_10000.csv")
mine_2.columns=["order_id","product_id","label_2"]
print mine_1.head()

all=mine_1.merge(arbor,on=["order_id","product_id"],how="left")
all=all.merge(lgbm,on=["order_id","product_id"],how="left")
all=all.merge(mine_2,on=["order_id","product_id"],how="left")

all["label"]=0.28*all["prediction_arbor"]+0.12*all["prediction_lgbm"]+0.36*all["label_2"]+0.24*all["label_1"]
#all["label"]=0.18*all["prediction_arbor"]+0.36*all["prediction_lgbm"]+0.25*all["label_2"]+0.21*all["label_1"]
print all.head()

all[["order_id","product_id","label"]].to_csv("blending_28_12_36_24_shuff.csv",index=None)


