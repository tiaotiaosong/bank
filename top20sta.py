import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt

def generatedata(bianhao):
    f = open('b6bce3abb838406daea9af48bf059c633.txt', encoding='UTF-8')
    next(f)
    user_remain = dict()
    ub = pd.date_range('2016-01-01', '2016-12-31')
    i=1
    f1 = open('testtest.txt', mode='w', encoding='UTF-8')
    for eachline in f:
        Record = eachline.split()
        Record[0] = pd.to_datetime(Record[0])
        if (Record[6] == bianhao and Record[3] == '32' and i==1):
            f1.write(eachline)
            if(Record[2]=='C'):
                remain=abs(int(Record[5].split('.')[0]))-abs(int(Record[4].split('.')[0]))
                for j in ub:
                    if(j<=Record[0]):
                        user_remain[j]=remain
            if(Record[2]=='D'):
                remain = abs(int(Record[5].split('.')[0])) + abs(int(Record[4].split('.')[0]))
                for j in ub:
                    if (j < Record[0]):
                        user_remain[j] = remain
            i=2
        elif(Record[6] == bianhao and Record[3] == '32'):
            f1.write(eachline)
            user_remain[Record[0]] = int(Record[5].split('.')[0])
    f.close()
    f1.close()
    for i in ub:
        if (i not in user_remain):
            user_remain[i]=user_remain[i-1]
        # else:
        #     user_remain[i]=user_remain[i]
    df1 = pd.DataFrame.from_dict(user_remain, orient='index')
    df1.columns = ['y']
    df1 = df1.reset_index()
    df1 = df1.reset_index(drop=True)
    df1.columns = ['ds', 'y']
    df1 = df1.sort_values(by='ds')
    return (df1)

user = pd.read_csv('top20.csv',index_col=0).index

res=dict()
# plt.figure(1)
# plt.plot(df2['ds'],df2['y'])
# plt.grid(True)
# print(user[0])
# print(df2)
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
for i in user:
    df2 = generatedata(i)
    df2['cap'] = 1.1 * (np.max(df2['y']))
    plt.figure(1)
    plt.plot(df2['ds'], df2['y'])
    plt.grid(True)
    prophet = Prophet(growth='logistic',holidays=holidays, changepoint_prior_scale=0.5)#,yearly_seasonality=True
    prophet.fit(df2)
    future = prophet.make_future_dataframe(periods=90)#, include_history=False
    future['cap'] = 1.1 * (np.max(df2['y']))
    forecast = prophet.predict(future)
    # print(forecast)
    prophet.plot_components(forecast)
    plt.grid(True)
    plt.figure(3)
    plt.plot(forecast['ds'], forecast['yhat'])
    plt.grid(True)
    plt.show()
    remain = forecast.ix[89, ['yhat']]
    res[i] = remain
df3 = pd.DataFrame.from_dict(res, orient='index')
df3.columns=['a']
p = df3.sort_values(by='a', axis=0, ascending=False).head(20)
x=p.index

def save_result(re_na, QRTA, ISIR, VOLUME, top20):
    with open(re_na+'.txt', 'w') as fr:
        fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
        fr.write('TOP20')
        for i in top20:
            fr.write('\t')
            fr.write(i)
save_result("result",1614,2078,270333720,x)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
# print(np.mean(forecast['yhat']))
# prophet.plot_components(forecast)
# plt.grid(True)
# plt.show()
