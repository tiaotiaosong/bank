import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt
file_path = 'perdayvolume.csv'
df = pd.read_csv(file_path,index_col=0)
plt.plot(df['ds'],df['y'])
print(max(df['y']))
plt.show()
print(df)
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
prophet = Prophet(holidays=holidays)
prophet.fit(df)
future=prophet.make_future_dataframe(periods=90)
forecast = prophet.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
print(np.mean(forecast['yhat']))
prophet.plot_components(forecast)
plt.show()
