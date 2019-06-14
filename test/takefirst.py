import pandas as pd
from math import log
import gc
def finduesrs():
    df = pd.read_csv('xdata.csv', index_col=0, header=0)
    df = df.set_index('xinbianhao')
    dates = df.columns
    l = []
    for i in dates:
        temp = df.sort_values(by=i, axis=0, ascending=False).head(50).index
        l.extend(temp)
    users = set(l)
    return users
def makedata():
    f = open('b6bce3abb838406daea9af48bf059c633.txt', mode='r', encoding='UTF-8')
    next(f)
    userdict = {'date': [],
                'zhaiyao': [],
                'drection': [],
                'bizhong': [],
                'trade': [],
                'remain': [],
                'bianhao': []}
    for eachline in f:
        l = eachline.split()
        userdict['date'].append(l[0])
        userdict['zhaiyao'].append(l[1])
        userdict['drection'].append(l[2])
        userdict['bizhong'].append(l[3])
        userdict['trade'].append(l[4])
        userdict['remain'].append(l[5])
        userdict['bianhao'].append(l[6])
    data = pd.DataFrame(userdict)
    return data
def buquanzhaiyao(df,zhaiyaoji):
    lieming=['bianhao']
    lieming.extend(['zhaiyao_{}'.format(i) for i in zhaiyaoji])
    dftemp = pd.get_dummies(df.loc[:, ['bianhao', 'zhaiyao']], columns=['zhaiyao'])
    zhaiyaoyu = set(zhaiyaoji) - set(df2['zhaiyao'])
    for yu in zhaiyaoyu:
        dftemp['zhaiyao_{}'.format(yu)] = 0
    dftemp=dftemp.loc[:, lieming]
    return dftemp
def buquanbizhong(df,bizhongji):
    lieming = ['bianhao']
    lieming.extend(['bizhong_{}'.format(i) for i in bizhongji])
    dftemp = pd.get_dummies(df.loc[:, ['bianhao', 'bizhong']], columns=['bizhong'])
    bizhongyu = set(bizhongji) - set(df['bizhong'])
    for yu in bizhongyu:
        dftemp['bizhong_{}'.format(yu)] = 0
    dftemp = dftemp.loc[:, lieming]
    return dftemp
users = finduesrs()
df =makedata()
df=df[df['bianhao'].isin(users)]
bizhongji=list(set(df['bizhong']))
zhaiyaoji=list(set(df['zhaiyao']))
# print(df)
# print(len(users))
df['month'] = df['date'].map(lambda d:pd.to_datetime(d).month)
del df['date']
df['drection'] = df['drection'].map(lambda d: 1 if d=='D' else 0)
df['tradetol']=df['trade'].map(lambda d:abs(float(d)))
df['trade']=df['trade'].map(lambda d:float(d))
df['trade_num_23']=1
res=pd.DataFrame(index=set(df['bianhao']))
print(len(set(res.index)))
print(res)
for mon in range(1,13):
    df1 = df[df['month']==mon]
    del df1['month']
    df2 = df1[df1['bizhong']=='32']
    del df2['bizhong']
    df2temp = buquanzhaiyao(df2,zhaiyaoji)
    df2temp = df2temp.groupby(['bianhao']).sum()
    df2temp['trade_mean'] = df2.groupby(['bianhao']).mean()['tradetol']
    df2temp['trade_min'] = df2.groupby(['bianhao']).min()['tradetol']
    df2temp['trade_max'] = df2.groupby(['bianhao']).max()['tradetol']
    df2temp['trade_num_23'] = df2.groupby(['bianhao']).sum()['trade_num_23']
    df2temp['tradetol'] = df2.groupby(['bianhao']).sum()['tradetol']
    df2temp['drection'] = df2.groupby(['bianhao']).sum()['drection']
    # print(df2temp)
    # print(df2temp.columns)

    del df2
    # del df2temp
    gc.collect()

    df3 = df1[df1['bizhong']!='32']
    df3temp=buquanbizhong(df3, bizhongji)
    df3temp = df3temp.groupby(['bianhao']).sum()
    df3temp['drection'] = df3.groupby(['bianhao']).sum()['drection']
    df2temp = pd.merge(df2temp,df3temp, how='outer', left_index=True, right_index=True)
    # print(df2temp)
    # print(len(set(df2temp.index)))

    del df3
    del df3temp
    gc.collect()

    df2temp.columns=['{x}yue_'.format(x=mon)+i for i in df2temp.columns]
    df2temp=df2temp.fillna(0)
    res = pd.merge(res, df2temp, how='left', left_index=True, right_index=True)
    # print(res)
    # print(res.columns)
    # break
    del df2temp
    gc.collect()
res=res.fillna(0)
res.to_csv('xgbdata.csv',index=True,header=True)
print(res)


# df['remain']=df['remain'].map(lambda d:round(float(d),2))
# # print(min(df['trade']))round(a,2)
# df['start']=df['remain']-df['trade']
# # df['date']=df['date'].map(lambda d:pd.to_datetime(d))
# df['start']=df['start'].map(lambda d:round(d,2))
# df=df[df['bizhong']=='32']
# print(len(set(df['bianhao'])))
# df=df[['date','bianhao','start','remain']]
# df.to_csv('data.csv',index=True,header=True)



