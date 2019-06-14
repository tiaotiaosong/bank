import pandas as pd
import matplotlib.pyplot as plt
f = open('b6bce3abb838406daea9af48bf059c633.txt',encoding='UTF-8')
QRTA_NUM=dict()
next(f)
for eachline in f:
    Record=eachline.split()
    Record[0] = pd.to_datetime(Record[0])
    if(Record[1]=='QRTA'and Record[3]=='32'):
        if(Record[0] not in QRTA_NUM):
            QRTA_NUM[Record[0]] = abs(int(Record[4].split('.')[0]))
        else:
            QRTA_NUM[Record[0]] += abs(int(Record[4].split('.')[0]))
df1=pd.DataFrame.from_dict(QRTA_NUM,orient='index')
df1.columns=['y']
df1=df1.reset_index()
df1=df1.reset_index(drop=True)
df1.columns=['ds', 'y']
print(df1)
df1.to_csv ("perdayvolume.csv" , index=True,header=True,encoding = "utf-8")
# 分月统计每月的
# p=dict()
# for i in QRTA_NUM:
#     g = i.month
#     if(g not in p):
#         p[g] = QRTA_NUM[i]
#     else:
#         p[g] +=QRTA_NUM[i]
# print(p)
# {1: 13034300972, 2: 11110729064, 3: 13205692968, 4: 11611624640, 5: 14108570411, 6: 13513613293, 7: 14062860649, 8: 9498872132, 9: 12068972523, 10: 9327888247, 11: 12977362516, 12: 16823941956}

# l=dict()
# for i in a:
#     if(i==2):
#         s=int(a[i]/28)
#         l[i]=s
#     elif(i==4 or i==6 or i==9 or i==11):
#         s = int(a[i] / 30)
#         l[i] = s
#     else:
#         s = int(a[i] / 31)
#         l[i] = s
# print (l)
# 各个月每天的均值{1: 420461321, 2: 396811752, 3: 425990095, 4: 387054154, 5: 455115174, 6: 450453776, 7: 453640666, 8: 306415230, 9: 402299084, 10: 300899620, 11: 432578750, 12: 542707805}

# a= {1: 13034300972, 2: 11110729064, 3: 13205692968, 4: 11611624640, 5: 14108570411, 6: 13513613293, 7: 14062860649, 8: 9498872132, 9: 12068972523, 10: 9327888247, 11: 12977362516, 12: 16823941956}
#
# month1=int((a[1]+a[2]+a[3])/(31+28+31))
# month2=int((a[4]+a[5]+a[6])/(30+30+31))
# month3=int((a[7]+a[8]+a[9])/(31+30+31))
# month4=int((a[10]+a[11]+a[12])/(31+30+31))
# 每个季度均值 415008033 431140751 387290275 425317312