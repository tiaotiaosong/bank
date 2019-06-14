# f = open('train.txt',encoding='UTF-8')
# # f = open('train.txt')
# # open(‘1.txt’,encoding=’gb18030’，errors=‘ignore’)
# # for eachline in f:
# #     # p=eachline.split(',')
# #     print(eachline)
# print(f.read())
#     # if('trx' in p):
#     #     print(eachline)
import pandas as pd
import numpy as np
import xgboost as xgb
# df = pd.DataFrame(np.random.rand(4,5),index=['a','b','c','d'])
# print(df.shape[0])
# Record = pd.to_datetime('2016-01-05')
# print(Record)
# print(Record.month)
# print(int(('-35465161.00').split('.')[0])
# gbm = xgb.XGBRegressor(objective="rank:pairwise")
#
# X =  np.random.normal(0, 1, 1000).reshape(100, 10)
# y = np.random.randint(0, 5, 100)
#
# gbm.fit(X, y) ### --- no group id needed???
#
# print(gbm.predict(X))
#
# # should be in reverse order of relevance score
# print (y[gbm.predict_proba(X)[:, 1].argsort()][::-1])
# user = pd.read_csv('top20.csv',index_col=0)
# p = user.sort_values(by='1a', axis=0, ascending=False).tail(20)
# x=p['1a'].map(lambda d:str(d))
#
# def save_result(re_na, QRTA, ISIR, VOLUME, top20):
#     with open(re_na+'.txt', 'w') as fr:
#         fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
#         fr.write('TOP20')
#         for i in top20:
#             fr.write('\t')
#             fr.write(i)
# save_result("result",1664,2078,270333720,x)
print(float('-478850.84'))
