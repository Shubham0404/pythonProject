#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib','inline')

#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

data = pd.read_csv('/home/user/Downloads/AirPassengers.csv')
#print data.head()
print '\n Data Types: '
#print data.dtypes

dateparse = lambda dates: pd.datetime.strptime(dates,'%Y-%m')
data2 = pd.read_csv('/home/user/Downloads/AirPassengers.csv', parse_dates='Month', index_col='Month', date_parser=dateparse)
print data2.head()

#print data2.index

ts=data2['#Passengers'] 
#print ts.head(10)

#print ts['1949-01-01']

#print ts['1949-01-01':'1949-04-01']

#print plt.plot(ts)

#from statsmodels.tsa.stattools import adfuller
#def test_stationarity(timeseries):
    
    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    #orig = plt.plot(timeseries, color='blue',label='Original')
    #mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    #std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    #plt.legend(loc='best')
    #plt.title('Rolling Mean & Standard Deviation')
    #plt.show()
    #plt.show(block=False)

#test_stationarity(ts)

ts_log = np.log(ts)
print ts_log.head()
#moving_avg = pd.rolling_mean(ts_log,12)

#ts_log_moving_avg_diff = ts_log - moving_avg
#print ts_log_moving_avg_diff.head(12)
#ts_log_moving_avg_diff.dropna(inplace='true')

ts_log_diff = ts_log - ts_log.shift()
print ts_log_diff.head()
#print ts_log_diff.head()
#plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)
#plt.plot(ts_log)
#plt.plot(moving_avg,color='red')
#plt.show()

#test_stationarity(ts_log_moving_avg_diff)

#from statsmodels.tsa.stattools import acf, pacf
#lag_acf = acf(ts_log_diff,nlags=20)
#lag_pacf = pacf(ts_log_diff,nlags=20, method='ols')
#plt.subplot(121) 
#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')


#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')
#plt.tight_layout()

#plt.show()

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
#plt.show()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print "prediction of arima model"
print predictions_ARIMA_diff.head()

#adjusting differences
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print predictions_ARIMA_diff_cumsum.head()

#getting log values of original series
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print predictions_ARIMA_log.head()

#predicting values

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()

