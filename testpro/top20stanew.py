import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from matplotlib import pyplot as plt
import gc
# df1=pd.read_csv('data1.csv',index_col=0,header=0)
# # print(df1.columns)
# df2=pd.read_csv('data2.csv',index_col=0,header=0)
# # print(df2.columns)
# df3=pd.read_csv('datalast.csv',index_col=0,header=0)
# # print(df3.shape[0])
# res=pd.concat([df1,df2,df3],axis=0,ignore_index=True)
# res.columns=[d.split()[0] for d in res.columns]
# res.to_csv('xdata.csv',index=True,header=True)
def generatedata(df,i):
    df4 = pd.DataFrame()
    df4['ds'] = df.columns
    df4['y'] = list(df.loc[i])
    return df4
df=pd.read_csv('xdata.csv',index_col=0,header=0)
df= df.set_index('xinbianhao')
# dates=['2016-01-31', '2016-02-29', '2016-03-31','2016-04-30', '2016-05-31', '2016-06-30','2016-07-31',
#        '2016-08-31', '2016-09-30', '2016-10-31','2016-11-30', '2016-12-27','2016-12-28','2016-12-29','2016-12-30','2016-12-31',]
# dates1=df.columns
# l=[]
# for i in dates1:
#     temp=df.sort_values(by=i, axis=0, ascending=False).head(30).index
#     l.extend(temp)
# user=set(l)
users =pd.read_csv('beixuandata.csv',index_col=0,header=0).index
length=len(users)
print(length)
# df=df[df.index.isin(users)]
# for i in df.columns:
#        df=df.sort_values(by=i,ascending=False)
#        df[i]=np.arange(1,length+1)
res=dict()

playoffs = pd.DataFrame({
  'holiday': 'yuandan',
  'ds': pd.to_datetime(['2016-01-01', '2016-01-02', '2016-01-03',
                        '2016-12-31','2017-01-01', '2017-01-02']),
  'lower_window': 0,
  'upper_window': 0,
})
superbowls = pd.DataFrame({
  'holiday': 'chunjie',
  'ds': pd.to_datetime(['2016-02-07', '2016-02-08', '2016-02-09','2016-02-10','2016-02-11','2016-02-12','2016-02-13',
                        '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02']),
  'lower_window': 0,
  'upper_window': 0,
})
holidays = pd.concat((playoffs, superbowls))
# future = m.make_future_dataframe(periods=1826)
# future['cap'] = 8.5
# fcst = m.predict(future)
# m.plot(fcst);
# i = 60
# user = users[i]
user = 'add06affbce0ba91833434932f6ed808'
# i='18d6fd26478c524b5ed08f09cff437e9'
df2 = generatedata(df,user)
print(df2)
# df2['cap'] = 1.1 * (np.max(df2['y']))
# df2['floor'] = 0.9*(np.min(df2['y']))
# print(df1)
plt.figure(1)
plt.plot(df2['ds'], df2['y'])
plt.grid(True)
# prophet = Prophet( n_changepoints=50,growth='logistic',changepoint_prior_scale=0.02,weekly_seasonality=False)#,yearly_seasonality=True holidays=holidays
prophet = Prophet(weekly_seasonality=False,changepoint_prior_scale=0.2,holidays=holidays)
prophet.fit(df2)
future = prophet.make_future_dataframe(periods=90)#, include_history=False
# future['cap'] = 1.1 * (np.max(df2['y']))
# future['floor'] = 0.9*(np.min(df2['y']))
forecast = prophet.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
prophet.plot_components(forecast)
plt.grid(True)
plt.figure(3)
plt.plot(forecast['ds'], forecast['yhat'])
plt.grid(True)
plt.show()
remain = forecast.ix[179, ['yhat']]
res[user] = remain
print(user,remain)
# df3 = pd.DataFrame.from_dict(res, orient='index')
# df3.columns=['a']
# p = df3.sort_values(by='a', axis=0,ascending=False)
# p.to_csv('beixuandata.csv', index=True, header=True)
# p = df3.sort_values(by='a', axis=0,ascending=False).head(20)
# x=p.index
#
# def save_result(re_na, QRTA, ISIR, VOLUME, top20):
#     with open(re_na+'.txt', 'w') as fr:
#         fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
#         fr.write('TOP20')
#         for i in top20:
#             fr.write('\t')
#             fr.write(i)
# save_result("result",1614,2078,255003720,x)


# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
# print(np.mean(forecast['yhat']))
# prophet.plot_components(forecast)
# plt.grid(True)
# plt.show()