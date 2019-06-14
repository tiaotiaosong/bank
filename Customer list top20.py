import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def chooseuser():
    f = open('b6bce3abb838406daea9af48bf059c633.txt', encoding='UTF-8')
    next(f)
    user = dict()
    l = []
    for i in range(1, 13):
        for eachline in f:
            Record = eachline.split()
            mon = pd.to_datetime(Record[0]).month
            if (Record[3] == '32' and mon == i):
                user[Record[6]] = int(Record[5].split('.')[0])
            if (mon == i + 1):
                break
        df1 = pd.DataFrame.from_dict(user, orient='index')
        df1.columns = ['a']
        p = df1.sort_values(by='a', axis=0, ascending=False).head(30).index
        l.append(p)
    temp = dict()
    for i in l:
        for j in i:
            if (j not in temp):
                temp[j] = 1
            else:
                temp[j] += 1
    d1 = pd.DataFrame.from_dict(temp, orient='index')
    f.close()
    return (d1)
def sta():
    f = open('b6bce3abb838406daea9af48bf059c633.txt', encoding='UTF-8')
    next(f)
    tradeDict = {
        'date': [],
        'abstract': [],
        'dre': [],
        'bizhong': [],
        'trade':[],
        'remain':[],
        'bianhao':[],
    }
    for eachline in f:
        i=eachline.split()
        tradeDict['date'].append(i[0])
        tradeDict['abstract'].append(i[1])
        tradeDict['dre'].append(i[2])
        tradeDict['bizhong'].append(i[3])
        tradeDict['trade'].append(i[4])
        tradeDict['remain'].append(i[5])
        tradeDict['bianhao'].append(i[6])
    df = pd.DataFrame(tradeDict)
    f.close()
    return (df)
df=sta()
# user=list(set(df['bianhao']))
# res = pd.DataFrame(index=user)
# dnf = pd.DataFrame(np.zeros((data.values.shape[0],len(wifis))),index=None,columns=wifis)
res=chooseuser()
user=res.index
print(len(user))
df['date']=df['date'].map(lambda d: pd.to_datetime(d))
df['remain']=df['remain'].map(lambda d: int(d.split('.')[0]))
df['month']=df['date'].map(lambda d: d.month)

temp = dict()
for i in range(1,13):
    df1=df[(df.month==i) & (df.bizhong=='32') ]
    for index, row in df1.iterrows():
        if(row['bianhao'] in user):
            if(row['remain']==0):
                row['remain']=1
            temp[row['bianhao']]=row['remain']
        else:
            pass
        # if(row['remain']==0):
        #     row['remain']=1
        # temp[row['bianhao']]=row['remain']
    df2 = pd.DataFrame.from_dict(temp, orient='index')
    df2.columns=['{i}a'.format(i=i)]
    res=res.join(df2,lsuffix='_caller',rsuffix='_other')
res = res.fillna(1)
res.to_csv("top20.csv", index=True, header=True, encoding="utf-8")


# p = df2.sort_values(by='a', axis=0, ascending=False).head(20)
# x=p.index

# def save_result(re_na, QRTA, ISIR, VOLUME, top20):
#     with open(re_na+'.txt', 'w') as fr:
#         fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
#         fr.write('TOP20')
#         for i in top20:
#             fr.write('\t')
#             fr.write(i)
# save_result("result",1664,2078,270333720,x)