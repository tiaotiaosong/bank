import pandas as pd
import gc
df=pd.read_csv('data.csv',index_col=0,header=0)
# df['date']=df['date'].map(lambda d:pd.to_datetime(d))
ub=pd.date_range('2016-01-01','2016-12-31')
users=set(df['bianhao'])
print((len(users)))
l=[]
pp=20000
df5=pd.read_csv('data1.csv',index_col=0,header=0)
df6=pd.read_csv('data2.csv',index_col=0,header=0)
users=users-(set(df5['xinbianhao']) | set(df6['xinbianhao']))
del df5
del df6
gc.collect()
for user in users:
    temp=pd.Series()
    temp['xinbianhao']=user
    df1=df[df['bianhao']==user]
    df1 = df1.sort_values(by='date')
    # print(df1)
    startday=min(df1['date'])
    # print(startday)
    if(startday!='2016-01-01'):
        ub1 = pd.date_range('2016-01-01', startday,freq='D')
        df2 = df1[df1['date'] == startday]
        df2=df2.reset_index(drop=True)
        # print(df2)
        if (df2.shape[0] == 0):
            continue
        elif (df2.shape[0] == 1):
            for day in ub1[:-1]:
                temp[day] = round(df2['start'].values[0],2)
        else:
            tempdict = dict()
            for i in df2['start']:
                p = min(df2['remain'].map(lambda d: abs(d - i)))
                tempdict[i] = p
            truestart = max(tempdict, key=tempdict.get)
            for day in ub1[:-1]:
                temp[day] = truestart
        # print(temp)
    for day in ub:
        if day in temp:
            continue
        df2=df1[df1['date']==day]
        if(df2.shape[0]==0):
            temp[day] = temp[day - 1]
        elif(df2.shape[0]==1):
            temp[day]=round(df2['remain'].values[0],2)
        else:
            tempdict=dict()
            for i in df2['remain']:
                p=min(df2['start'].map(lambda d:abs(d-i)))
                tempdict[i]=p
            temp[day]=max(tempdict, key=tempdict.get)
    l.append(temp)
    pp+=1
    print(pp)
    if((pp%10000)==0):
        df1=pd.DataFrame(l)
        df1.to_csv('data{i}.csv'.format(i=(pp/10000)),index=True,header=True)
        l=[]
df1=pd.DataFrame(l)
df1.to_csv('datalast.csv',index=True,header=True)