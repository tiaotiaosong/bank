import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
def pianchaifo(df):
    answer = df.sort_values(by='y', axis=0, ascending=False).head(20).index
    listnum = np.arange(1,21)
    answer=dict(zip(answer,listnum))
    result = df.sort_values(by='yhat', axis=0, ascending=False).head(20).index
    print(answer)
    print(dict(zip(result,listnum)))
    x = .0
    for index, value in enumerate(result):
            if value in answer.keys():
                    x += np.square(answer[value] - index - 1)
            else:
                    x += np.square(20)
    return (np.sqrt(x / 20))
def generatedata(df,i):
    df4 = pd.DataFrame()
    df4['ds'] = df.columns
    df4['y'] = list(df.loc[i])
    return df4
def find_users(df):
    dates = df.columns
    l = []
    for i in dates:
        temp = df.sort_values(by=i, axis=0, ascending=False).head(20).index
        l.extend(temp)
    users = set(l)
    print('总用户数 =',len(users))
    return users
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)
def timeseries_to_supervised(data):
    data = pd.DataFrame(data)
    data.columns = ['y']
    data['y+90'] = data['y'].shift(-90)
    data = data.fillna(np.mean(data['y']))
    return data
    # return data
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
df=pd.read_csv('../sourcedata/xdata.csv',index_col=0,header=0)
df= df.set_index('xinbianhao')
users = find_users(df)
df=df[df.index.isin(users)]
resdict = dict()
p=1
for user in users:
    # temp = list(generatedata(df,i).values[:,1])
    # x_train = series_to_supervised(temp)
    temp = generatedata(df,user)
    raw_values = temp['y'].values
    # print(temp)
    diff_values = difference(temp['y'].values)
    supervised = timeseries_to_supervised(diff_values)
    # supervised = timeseries_to_supervised(temp['y'])
    supervised_values = supervised.values
    train, test = supervised_values[0:-90], supervised_values[-90:]
    scaler, train_scaled, test_scaled = scale(train, test)
    lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    # walk-forward validation on the test data
    predictions = list()
    expected = dict()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(yhat)
        # j = len(train) + i + 90
        # expected[j]=raw_values[j]
        # print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, raw_values[j]))
        print('Month=%d, Predicted=%f ,false_y=%f' % (i + 1, yhat,supervised.values[i-90,-1]))
        if(i==89):
            resdict[user] = yhat
            print('user-{}-'.format(p),yhat)
            p +=1
res = pd.DataFrame.from_dict(resdict, orient='index')
res.columns=['yhat']
res = res.sort_values(by='yhat', axis=0, ascending=False).head(20)
print(res)
x=res.index

def save_result(re_na, QRTA, ISIR, VOLUME, top20):
    with open(re_na+'.txt', 'w') as fr:
        fr.writelines(['QRTA'+'\t'+str(QRTA)+'\n', 'ISIR'+'\t'+str(ISIR)+'\n', 'VOLUME'+'\t'+str(VOLUME)+'\n'])
        fr.write('TOP20')
        for i in top20:
            fr.write('\t')
            fr.write(i)
save_result("result",1614,2078,250003720,x)
