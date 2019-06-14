import pandas as pd
from math import log
data = pd.read_csv('xdata.csv',index_col=0,header=0)
# users = pd.read_csv('../top20.csv',index_col=0).index
# data=data[data['xinbianhao'].isin(users)]
data = data.set_index('xinbianhao')
# print(data)
data['out']=data.iloc[:,90]-data.iloc[:,0]+data.iloc[:,-1]
x=data.sort_values(by='out',ascending=False).head(20).index

# start['mean']=data.iloc[:,0:7].mean(axis=1)
#
# print(start['mean'].values)

# start=pd.DataFrame(start)
# start.columns=['a']
# print(start['a'])
# mubiao = data.iloc[:,90:97]
# # print(mubiao)
# data = data.iloc[:,360:].mean(axis=1)
# res=pd.DataFrame(data*(mubiao/(start+1)))
# print(res)
# res=res.sort_value
# res.columns=['a']
# res = res.sort_values(by='a', axis=0, ascending=False).head(20)
# # print(res)
# x=res.index
def save_result(re_na, QRTA, ISIR, VOLUME, top20):
    with open(re_na+'.txt', 'w') as fr:
        fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
        fr.write('TOP20')
        for i in top20:
            fr.write('\t')
            fr.write(i)
save_result("result",1614,2078,270333720,x)