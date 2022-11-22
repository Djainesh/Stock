# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
''''import sys 
  
  
print("the name of the program is ", sys.argv[0]) 
  
n = len(sys.argv[1]) 
a = sys.argv[1][1:n-1] 
a = a.split(', ') 
  
for i in a: 
    print(i)'''

import datetime
import sys
import os

tomorrow = datetime.date.today() + datetime.timedelta(days=1)
print("Tomorrow date is",str(tomorrow))
tm_stme = datetime.time(0, 0, 0)
tm_etime = datetime.time(23,59,59)
tm_stdate = datetime.datetime.combine(tomorrow, tm_stme)
tm_enddate = datetime.datetime.combine(tomorrow,tm_etime)

print("tomorrow start date:",tm_stdate)
print("tomorrow end date:",tm_enddate)




import yfinance as yf
from datetime import date
today1=date.today()
print(today1)
a=input("Enter the stock name")
# download dataframe
df = yf.download(a, start="2019-01-01", end=today1)
df = df.reset_index()
print(df.columns)
  

# reading the data


# looking at the first five rows of the data
print(df.tail())
print('\n Shape of the data:')
print(df.shape)

# setting the index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Low'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Low'][i] = data['Low'][i]

# NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

# splitting into train and validation
train = new_data[:int(len(new_data)*0.8)]
valid = new_data[int(len(new_data)*0.8):]

# shapes of training set
print('\n Shape of training set:')
print(train.shape)

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['Low'][len(train)-int(len(new_data)*0.2)+i:].sum() + sum(preds)
    b = a/int(len(new_data)*0.2)
    preds.append(b)

# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Low'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms)




#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Low'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Low'][i] = data['Low'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[:int(len(new_data)*0.8),:]
valid = dataset[int(len(new_data)*0.8):,:]
print(len(valid))

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(21,len(train)):
    x_train.append(scaled_data[i-21:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 21:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(21,inputs.shape[0]):
    X_test.append(inputs[i-21:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)#importing required libraries


rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


train=new_data[:int(len(new_data)*0.8)]
valid=new_data[int(len(new_data)*0.8):]
valid['Predictions'] = closing_price

plt.plot(train['Low'])
plt.plot(valid['Predictions'])
plt.plot(valid['Low'])
print(valid['Low'][0]-valid['Predictions'][0])

# %%
# ipynb-py-convert C:/Users/shahr/Low_price(TCS).ipynb C:/Users/shahr/Low_price(TCS).py

# %%
