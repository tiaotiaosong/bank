import pandas as pd
# import matplotlib.pyplot as plt
f = open('b6bce3abb838406daea9af48bf059c633.txt',encoding='UTF-8')
QRTA_NUM=dict()
ISIR_NUM=dict()
next(f)
for eachline in f:
    Record=eachline.split()
    Record[0] = pd.to_datetime(Record[0])
    if(Record[1]=='QRTA'and Record[3]=='32'):
        if(Record[0] not in QRTA_NUM):
            QRTA_NUM[Record[0]] = 1
        else:
            QRTA_NUM[Record[0]] += 1
    elif(Record[1]=='ISIR'and Record[3]=='32'):
        if(Record[0] not in ISIR_NUM):
            ISIR_NUM[Record[0]] = 1
        else:
            ISIR_NUM[Record[0]] += 1
# df1=pd.DataFrame.from_dict(QRTA_NUM,orient='index')
# df1.columns=['y']
# df1=df1.reset_index()
# df1=df1.reset_index(drop=True)
# df1.columns=['ds', 'y']
# print(df1)
# df1.to_csv ("perdayQRTA.csv" , index=True,header=True,encoding = "utf-8")
ub=pd.date_range('2016-01-01','2016-12-31')
for i in ub:
    if (i not in ISIR_NUM.keys()):
        ISIR_NUM[i]=0
# print(ISIR_NUM)
df2=pd.DataFrame.from_dict(ISIR_NUM,orient='index')
df2.columns=['y']
df2=df2.reset_index()
df2=df2.reset_index(drop=True)
df2.columns=['ds', 'y']
df2=df2.sort_values(by='ds')
print(df2)
df2.to_csv ("perdayISIR.csv" , index=True,header=True,encoding = "utf-8")

# 每个月QRTA的次数统计
# p=dict()
# for i in ISIR_NUM:
#     g = i.month
#     if(g not in p):
#         p[g] = ISIR_NUM[i]
#     else:
#         p[g] +=ISIR_NUM[i]
# print(p)
# x=0
# for i in p:
#     x +=p[i]
# x=int(x/366)
# print("ISIR",x)
#
# p=dict()
# for i in QRTA_NUM:
#     g = i.month
#     if(g not in p):
#         p[g] = QRTA_NUM[i]
#     else:
#         p[g] +=QRTA_NUM[i]
# print(p)
# x=0
# for i in p:
#     x +=p[i]
# x=int(x/366)
# print("QRTA",x)



# {1: 51335, 2: 36588, 3: 52260, 4: 50485, 5: 53901, 6: 55250, 7: 52824, 8: 57589, 9: 55068, 10: 48519, 11: 59599, 12: 64846}
# 分周末，工作日统计，未包含节假日
# df2=pd.DataFrame.from_dict(ISIR_NUM,orient='index')
# print(df2)
# l=[]
# p=[]
# for i in range(df1.shape[0]):
#     j=i%7
#     if(j==1 or j==2):
#         l.append(df1.iloc[i,0])
#     else:
#         p.append(df1.iloc[i,0])
# plt.figure(1)
# plt.subplot(211)
# plt.plot(l)
# plt.grid(True)
# plt.subplot(212)
# plt.plot(p)
# plt.grid(True)
# plt.show()


