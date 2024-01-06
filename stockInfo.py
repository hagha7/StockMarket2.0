# This is where we define our classes when it comes to collecting data
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import math

'''
    The class below is used to collect our stock data with the yahoo finance API key. 
    We are compressing our data into a dict format as way to have a key: value method instead 
    of having all the data layed out in a table. 
'''
def fetch_stock_data(ticker, startDate, endDate) -> dict():
    try:
        stock_data = yf.download(ticker, start=startDate, end=endDate)
        if stock_data is not None and not stock_data.empty:
            company_name = yf.Ticker(ticker).info['longName']
            columns = stock_data.columns
            stock_data_result = {
                'company_name': company_name,
                'data': {col: stock_data[col].tolist() for col in columns}
            }
            return stock_data_result
        else:
            return {'company_name': 'N/A', 'data': {}}
    except Exception as e:
        print('Error:', str(e))
        return {'company_name': 'N/A', 'data': {}}
    
'''
    The class below is used for taking the collected data from the user 
    and essentially importing the data for training and testing. 
'''

class dataCollect():
    def __init__(self, ticker, startDate, endDate):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate

    def gather_data(self):
        stock_data = yf.download(self.ticker, self.startDate, self.endDate)
        return stock_data

    def compress_data_to_df(self):
        stock_data = self.gather_data()
        if stock_data is not None:
            df = pd.DataFrame(stock_data)
            return df
        else:
            return None
        
    def compress_data_to_dict(self):
        stock_data = self.gather_data()
        if stock_data is not None:
            data_dict = {'Date': stock_data.index, 'Close': stock_data['Close'].tolist()}
            return data_dict
        else:
            return None

def create_and_compile_model(stock_data_df):
    # Filter the 'Close' column from the DataFrame
    data = stock_data_df.filter(['Close'])
    dataset = data.values
        
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * 0.8)
        
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
        
    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]
        
    # Split the data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Create and compile the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mean_absolute_error'])

    model.fit(x_train, y_train, batch_size=1, epochs=7)

    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
        
    return model, rmse