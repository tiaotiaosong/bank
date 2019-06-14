import xgboost as xgb
import pandas as pd
from math import log
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"top20.csv",index_col=0)
data = data.fillna(1)
for i in data.columns:
    data[i]=data[i].map(lambda d: log(d,10))
# print(data)
params = {
    # 'booster':'gblinear',
    "objective": "reg:linear",
    "eta": 0.05,
    "max_depth": 7,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "silent": 1,
    'seed': 1954,
    'missing': 0,
    'gamma': 0.2,  # 0.2 is ok
    # 'max_depth': 9,
    'lambda': 1200,
    'min_child_weight': 3,
    # "alpha": 1  怎么调都是降低
}
y = data['12a'].values
X = data[['1a','2a','3a','4a','5a','6a','7a','8a','9a']].values
XX = data[['4a','5a','6a','7a','8a','9a','10a','11a','12a']].values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4)
xgbtrain = xgb.DMatrix(train_x, label=train_y)
xgbtest = xgb.DMatrix(test_x, label= test_y)
test = xgb.DMatrix(XX)
num_rounds =8000
watchlist = [(xgbtrain, 'train'), (xgbtest, 'test')]
model = xgb.train(params, xgbtrain, num_rounds, watchlist,
                      early_stopping_rounds=15)

y_submission =  model.predict(test,ntree_limit = model.best_ntree_limit)

res=pd.DataFrame(index=data.index)
res['yhat']=y_submission
res = res.sort_values(by='yhat', axis=0, ascending=False).head(20)
print(res)
x=res.index

def save_result(re_na, QRTA, ISIR, VOLUME, top20):
    with open(re_na+'.txt', 'w') as fr:
        fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
        fr.write('TOP20')
        for i in top20:
            fr.write('\t')
            fr.write(i)
save_result("result",1664,2078,270333720,x)