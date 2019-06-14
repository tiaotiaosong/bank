import xgboost as xgb
import pandas as pd
from math import log
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing
import gc
def pianchaifo(df):
    answer = df.sort_values(by='y', axis=0, ascending=False).head(20).index
    listnum = np.arange(1,21)
    answer=dict(zip(answer,listnum))
    result = df.sort_values(by='yhat', axis=0, ascending=False).head(20).index
    # print(answer)
    # print(dict(zip(result,listnum)))
    x = .0
    for index, value in enumerate(result):
            if value in answer.keys():
                    x += np.square(answer[value] - index - 1)
            else:
                    x += np.square(20)
    return (np.sqrt(x / 20))

def chooseusers(data):
    dates = ['2016-01-31', '2016-02-29', '2016-03-31', '2016-04-30', '2016-05-31', '2016-06-30', '2016-07-31',
             '2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30', '2016-12-31']
    data = data.loc[:, dates]
    l = []
    for i in dates:
        temp = data.sort_values(by=i, axis=0, ascending=False).head(20).index
        l.extend(temp)
    users = list(set(l))
    return users
def make_month_data(data,i):
    res = pd.DataFrame(index=data.index)
    temp=[]
    for j in data.columns:
        yuefen = pd.to_datetime(j).month
        if (yuefen == i):
            temp.append(j)
    datayuefen = data.ix[:, temp]
    res['{}-mean'.format(i)] = np.mean(datayuefen, axis=1)
    res['{}-max'.format(i)] = np.max(datayuefen, axis=1)
    res['{}-min'.format(i)] = np.min(datayuefen, axis=1)
    res['{}-q0.25'.format(i)] = datayuefen.quantile(0.25, axis=1)
    res['{}-q0.5'.format(i)] = datayuefen.quantile(0.5, axis=1)
    res['{}-q0.75'.format(i)] = datayuefen.quantile(0.75, axis=1)
    res['{}-q1'.format(i)] = datayuefen.quantile(0.75, axis=1)
    return res
def make_y_data(data,i):
    res = pd.DataFrame(index=data.index)
    if(i in [4,6,9,11]):
        date = 30
    elif(i==2):
        date = 29
    else:
        date = 31
    if(i<10):
        res['y'] = data['2016-0{mon}-{date}'.format(mon=i,date=date)]
    else:
        res['y'] = data['2016-{mon}-{date}'.format(mon=i, date=date)]
    return res
def get_feature_imp(model,feature):
    p = pd.Series(model.get_fscore()).sort_values(ascending=False)
    p = dict(p)
    fea_dict = dict()
    for i in p.keys():
        x = int(i[1:])
        fea_dict[feature[x]] = p[i]
    feature_important = pd.DataFrame.from_dict(fea_dict, orient='index')
    feature_important.to_csv('feature_important.csv', index=True, encoding="utf-8")
data = pd.read_csv(r"xdata.csv",index_col=0,header=0)
data = data.set_index('xinbianhao')
data_1 = pd.read_csv(r"xgbdata.csv",index_col=0,header=0)
users = data_1.index
data=data[data.index.isin(users)]
train_data = pd.DataFrame()
verify_data = pd.DataFrame()
for mon in range(1,10):
    data_1_lie=[]
    for i in data_1.columns:
        if (i.split('_')[0]=='{}yue'.format(mon)):
            data_1_lie.append(i)
    data_1_lie = sorted(data_1_lie)
    df1 = data_1.loc[:,data_1_lie]
    df2 = make_month_data(data,mon)
    df1 = pd.merge(df1,df2,how='left',left_index=True,right_index=True)
    df3 = make_y_data(data,mon+3)
    df1 = pd.merge(df1, df3, how='left', left_index=True, right_index=True)
    del df2
    del df3
    gc.collect()
    if(mon==9):
        verify_data = df1
        break
    if (mon==1):
        train_data = df1
    else:
        df1.columns = train_data.columns
        train_data = pd.concat([train_data,df1],axis=0,ignore_index=True)
    del df1
    gc.collect()

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(train_data.iloc[:,0:-1])
y = min_max_scaler.fit_transform(train_data.iloc[:,-1].values.reshape(-1,1))

X_v = min_max_scaler.fit_transform(verify_data.iloc[:,0:-1])
y_v = min_max_scaler.fit_transform(verify_data.iloc[:,-1].values.reshape(-1,1))
# print(X.shape)
# print(train_data.columns)
# print(y.shape)
# 对每个记录进行排名
# length=data.shape[0]
# for i in data.columns:
#        data=data.sort_values(by=i,ascending=False)
#        data[i]=np.arange(1,length+1)
# print(data)
pppp=dict()
for i in np.arange(0,15,1):
    params = {
        # 'booster':'gblinear',
        "objective": "reg:linear",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "silent": 1,
        'seed': 1954,
        'missing': 0,
        'gamma': 0,  # 0.2 is ok
        # 'max_depth': i,
        'lambda': 0,
        'min_child_weight': 2,
        "alpha": 0 # 怎么调都是降低
    }
    # y = X_train_minmax.iloc[:,-1].values
    # X = X_train_minmax.iloc[:, 0:-1].values
    # XX = data.iloc[:, 91:].values
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4)
    xgbtrain = xgb.DMatrix(train_x, label=train_y)
    xgbtest = xgb.DMatrix(test_x, label=test_y)
    verify = xgb.DMatrix(X_v)
    num_rounds = 8000
    watchlist = [(xgbtrain, 'train'), (xgbtest, 'test')]
    model = xgb.train(params, xgbtrain, num_rounds, watchlist,
                      early_stopping_rounds=15)  # feval=pianchaifo,
    feature = train_data.columns
    get_feature_imp(model, feature)
    y_pre = model.predict(verify, ntree_limit=model.best_ntree_limit)
    res = verify_data.loc[:, ['y']]
    res['yhat'] = y_pre
    result = pianchaifo(res)
    pppp[i]=result
# print(res)
print(pppp)


# p=pd.Series(model.get_fscore()).sort_values(ascending=False)
# y_submission =  model.predict(test,ntree_limit = model.best_ntree_limit)
# res=pd.DataFrame(index=data.index)
# res['yhat']=y_submission
# res = res.sort_values(by='yhat', axis=0, ascending=False).head(200)
# res.to_csv('top200.csv',header=True, encoding="utf-8")
# print(res)
# print(p)
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