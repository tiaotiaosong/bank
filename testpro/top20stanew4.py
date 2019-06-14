import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from matplotlib import pyplot as plt
import gc
def generatedata(df,i):
    df4 = pd.DataFrame()
    df4['ds'] = df.columns
    df4['y'] = list(df.loc[i])
    return df4
df=pd.read_csv('xdata.csv',index_col=0,header=0)
df= df.set_index('xinbianhao')
dates=['2016-01-31', '2016-02-29', '2016-03-31','2016-04-30', '2016-05-31', '2016-06-30','2016-07-31',
       '2016-08-31', '2016-09-30', '2016-10-31','2016-11-30', '2016-12-31']
l=[]
for i in dates:
    temp=df.sort_values(by=i, axis=0, ascending=False).head(20).index
    l.extend(temp)
user=set(l)
res=dict()
print('用户 = ',len(user))
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
for i in user:
    df2 = generatedata(df,i)
    # plt.figure(1)
    # # p=plt.boxplot(df2['y'], sym="o", whis=1.5)
    # plt.plot(df2['ds'], df2['y'])
    UpperLimit = df2[['y']].quantile(0.75, axis=0) + (df2[['y']].quantile(0.75, axis=0) - df2[['y']].quantile(0.25, axis=0)) * 0.8
    LowerLimit = df2[['y']].quantile(0.25, axis=0) - (df2[['y']].quantile(0.75, axis=0) - df2[['y']].quantile(0.25, axis=0)) * 0.8
    middle = float(df2[['y']].quantile(0.5, axis=0))
    templist = []
    # df2['y'] = df2['y'].map(lambda d: if ((d <= float(UpperLimit)) and (d >= float(LowerLimit))) else np.NAN )
    for d in df2['y']:
        if ((d <= float(UpperLimit)) and (d >= float(LowerLimit))):
            templist.append(d)
        else:
            templist.append(np.NaN)
    df2['y'] = templist
    # df2 = df2[df2['y'] != np.NaN]
    # plt.show()
    df2['cap'] = 2 * (np.max(df2['y']))
    df2['floor'] = 0
    # print(df2)
    # plt.figure(2)
    # plt.plot(df2['ds'], df2['y'])
    # plt.grid(True)
    # prophet = Prophet( mcmc_samples=100,growth='logistic',weekly_seasonality=False)#,yearly_seasonality=True holidays=holidays
    prophet = Prophet(growth='logistic',weekly_seasonality=False,changepoint_prior_scale=0.2)
    prophet.fit(df2)
    future = prophet.make_future_dataframe(periods=90,include_history=False)#, include_history=False
    future['cap'] = 2 * (np.max(df2['y']))
    future['floor'] = 0
    # df_cv = cross_validation(
    #     prophet, '90 days', initial='200 days', period='90 days')
    # print(df_cv)
    # fig = plt.figure(facecolor='w', figsize=(10, 6))
    # ax = fig.add_subplot(111)
    # # plt.plot(df_cv['ds'], df_cv['y'])
    # # plt.plot(df_cv['ds'], df_cv['yhat'])
    # ax.plot(prophet.history['ds'].values, prophet.history['y'], 'k')
    # ax.plot(df_cv['ds'].values, df_cv['yhat'], ls='-', c='#0072B2')
    # ax.fill_between(df_cv['ds'].values, df_cv['yhat_lower'],
    #                 df_cv['yhat_upper'], color='#0072B2',
    #                 alpha=0.2)
    # plt.grid(True)
    # plt.show()
    forecast = prophet.predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    # # print(forecast)
    # # prophet.plot_components(forecast)
    # plt.grid(True)
    # plt.figure(3)
    # plt.plot(forecast['ds'], forecast['yhat'])
    # plt.grid(True)
    # plt.show()
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
save_result("result",1614,2078,255003720,x)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
# print(np.mean(forecast['yhat']))
# prophet.plot_components(forecast)
# plt.grid(True)
# plt.show()