import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment=None
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Python Module/Library 

def tickerData():
    tickerName = input('What is your Stock Ticker Name?').upper()
    startDate = input('Enter Start Date: YYYY-MM-DD')
    endDate = input('Enter End Date: YYYY-MM-DD')
    df = yf.download(tickerName, startDate, endDate)

# Dont Include
# plt.figure(figsize=(16,8))
# plt.title('Close Price History of ' +str(tickerName))
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize = 18)
# plt.ylabel('Close Price USD ($)', fontsize = 18)
# plt.show()

data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train datasets

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# function
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mean_absolute_error'])

model.fit(x_train, y_train, batch_size=1, epochs=5)

test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models preducted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

print(valid)