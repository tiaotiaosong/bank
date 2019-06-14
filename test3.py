f = open('b6bce3abb838406daea9af48bf059c633.txt',encoding='UTF-8')
#共2349385条记录，共31047个用户，
# f = open('train.txt')
# open(‘1.txt’,encoding=’gb18030’，errors=‘ignore’)
# g=dict()
# for eachline in f:
#     p=eachline.split()
#     # print(eachline)
#     if(p[3] not in g):
#         g[p[3]] = 1
#     else:
#         g[p[3]] += 1
# {'币种代码': 1, '32': 2187430, '35': 86570, '21': 35712, '10': 10862, '29': 6914, '65': 11950, '39': 2704, '43': 5193, '87': 643, '69': 1406}
# g=dict()
# sum=0
# for eachline in f:
#     p=eachline.split()
#     if(p[6] not in g):
#         g[p[6]] = 1
#     else:
#         g[p[6]] += 1
#     sum +=1
# print(g)
# print(sum)
# print (len(g))
g=dict()
sum=0
for eachline in f:
    p=eachline.split()
    if(p[1] not in g):
        g[p[1]] = 1
    else:
        g[p[1]] += 1
    sum +=1
print(g)
print(sum)
print (len(g))
# {'摘要代码': 1, 'SSWA': 4268, 'QRTA': 638264, 'FEZZ': 4942, 'ISIR': 785199, 'FENC': 698, 'FEZ1': 21064, 'FEIS': 473222, 'FFM2':
#  35831, 'ISCP': 23207, 'GALO': 6881, '8001': 1965, 'FFT2': 838, 'DPBC': 2934, 'AGPY': 1352, 'SSW5': 2028, 'DPBF': 241, 'DPBD': 3060,
# 'ISOR': 1485, 'IINT': 112991, 'SKRD': 135, 'SSW8': 1579, 'EXME': 107, 'ISFT': 476, 'SSW6': 224, 'ONFC': 92, 'TRRA': 52, 'AGRD': 40,
# 'FEEE': 226077, 'ONTX': 42, '8002': 4, 'EXVF': 5, 'T1RO': 2, 'NNNN': 51, 'FFS2': 22, 'ACTF': 3, 'EXVN': 2, '8003': 1}
