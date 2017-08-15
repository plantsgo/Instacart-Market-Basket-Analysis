# Instacart-Market-Basket-Analysis
https://www.kaggle.com/c/instacart-market-basket-analysis

MODEL:

1.user product pair model.

2.if none model.

FEATURES:

User X Order features:

Products features:

Aisle features:

Department features:

User features:

User X Products features:

User X Aisles features:

User X Department features:

Products X time features:

order_number features:

DATA:

I add the last prior as train data.Also,I add the second to the last prior,but not didn't improve.

ENSEMBLE:

all["label"]=0.28*all["prediction_arbor"]+0.12*all["prediction_lgbm"]+0.36*all["label_2"]+0.24*all["label_1"]

prediction_arbor AND prediction_lgbm is from sh1ng's baseline.

label_2 trained with shuffle, label_1 traind split by userid.

SUB:

Use faron's script with my "none model".
