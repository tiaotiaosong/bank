import xgboost as xgb
import pandas as pd
from math import log
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data = pd.read_csv(r"xdata.csv",index_col=0,header=0)
data= data.set_index('xinbianhao')
# print(data)
for i in data.columns:
    data[i]=data[i].map(lambda d: log(float(d)+1,5))
dates=['2016-01-31', '2016-02-29', '2016-03-31','2016-04-30', '2016-05-31', '2016-06-30','2016-07-31',
       '2016-08-31', '2016-09-30', '2016-10-31','2016-11-30', '2016-12-27','2016-12-28','2016-12-29','2016-12-30','2016-12-31',]
dates1=data.columns[-90:]
l=[]
for i in dates1:
    temp=data.sort_values(by=i, axis=0, ascending=False).head(20).index
    l.extend(temp)
for i in dates:
    temp = data.sort_values(by=i, axis=0, ascending=False).head(20).index
    l.extend(temp)
user=set(l)


# print(data)
# y = data.iloc[:, 365].values
# X = data.iloc[:, 0:275].values
# XX = data.iloc[:, 91:].values
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4)
# xgbtrain = xgb.DMatrix(train_x, label=train_y)
# xgbtest = xgb.DMatrix(test_x, label=test_y)
# test = xgb.DMatrix(XX)
# num_rounds = 8000
# watchlist = [(xgbtrain, 'train'), (xgbtest, 'test')]
# parameters = {'min_child_weight': np.arange(1,5)}
# model=xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear',
#                      nthread=4,gamma=0, max_delta_step=0, subsample=1, colsample_bytree=1,
# colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,seed=1, missing=None)
# clf = GridSearchCV(model, parameters)
# clf.fit(X, y,  eval_set=watchlist, early_stopping_rounds=15, verbose=True)
# print(sorted(clf.cv_results_.keys()))

params = {
        # 'booster':'gblinear',
        "objective": "reg:linear",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "silent": 1,
        'seed': 1954,
        'missing': 0,
        'gamma': 0.2,  # 0.2 is ok
        # 'max_depth': 9,
        'lambda': 1200,
        'min_child_weight': 2,
        # "alpha": 1  怎么调都是降低
    }
y = data.iloc[:, 365].values
X = data.iloc[:, 0:275].values
XX = data.iloc[:, 91:].values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4)
xgbtrain = xgb.DMatrix(train_x, label=train_y)
xgbtest = xgb.DMatrix(test_x, label=test_y)
test = xgb.DMatrix(XX)
num_rounds = 8000
watchlist = [(xgbtrain, 'train'), (xgbtest, 'test')]
model = xgb.train(params, xgbtrain, num_rounds, watchlist,
                      early_stopping_rounds=15)

p=pd.Series(model.get_fscore()).sort_values(ascending=False)
y_submission =  model.predict(test,ntree_limit = model.best_ntree_limit)
res=pd.DataFrame(index=data.index)
res['yhat']=y_submission
res = res.sort_values(by='yhat', axis=0, ascending=False).head(200)
res.to_csv('top200.csv',header=True, encoding="utf-8")
print(res)
print(p)
# x=res.index
# print(x)
#
# def save_result(re_na, QRTA, ISIR, VOLUME, top20):
#     with open(re_na+'.txt', 'w') as fr:
#         fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
#         fr.write('TOP20')
#         for i in top20:
#             fr.write('\t')
#             fr.write(i)
# save_result("result",1614,2078,270333720,x)