from database.yfinanceapi import bulk_stock_data, add_daily_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import requests
import pywt
import math
from database.tables import Base, Account, User_Stocks, Stocks, Stock_Info
from sklearn.metrics import r2_score, mean_squared_error
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
# r squared
# mean squared error

def wavelet(data):
    wavelet = 'db4'
    coes = pywt.wavedec(data, wavelet, mode = 'reflect')
    threshold = .01
    coe_threshold = [pywt.threshold(c, threshold, mode='soft') for c in coes]
    smoothed = pywt.waverec(coe_threshold, wavelet)

    return smoothed

def attention_lstm(ticker, engine):


    stock_q= select(Stock_Info).where(Stock_Info.stock_id == 1)
    Session = sessionmaker(bind=engine)
    session = Session()
    data2 = session.connection().execute(stock_q).all()

    print(data2[0])
    s_open = []
    s_close = []
    s_high = []
    s_low = []
    s_volume = []
    for row in data2:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])


    device = 'cpu'
    data = {'Close': s_close, 'Open': s_open, 'High':s_high, 'Low':s_low, 'Volume':s_volume}
    data = pd.DataFrame(data)    
    #data = add_daily_data(ticker, 1300)
    #normalized volume
    data['Volume'] = (data['Volume'] - data['Volume'].min())/(data['Volume'].max() - data['Volume'].min())
    # normalized
    multiple = (data['Close'].max() - data['Close'].min())
    minimum =  data['Close'].min()
    data['Close'] = (data['Close'] - data['Close'].min())/ (data['Close'].max() - data['Close'].min()) 
    validation = data['Close']
    data['Close'] = wavelet(data['Close'])
    data['Low'] = wavelet(data['Low'])
    data['High'] = wavelet(data['High'])
    data['Open'] = wavelet(data['Open'])
    for i in range(1, len(data['High'])):
        data['Low'][i] = data['High'][i]-data['Low'][i]
        data['High'][i] = 1-(data['Close'][i]-data['Close'][i-1])
    
    data['High'][0] = 0
    data['Low'][0] = 0

    
    data['High'] = (data['High'] - data['High'].min())/ (data['High'].max() - data['High'].min())
    data['Low'] = (data['Low'] - data['Low'].min())/ (data['Low'].max() - data['Low'].min())
    data['Open'] = (data['Open'] - data['Open'].min())/ (data['Open'].max() - data['Open'].min()) 
    
    answer = data['Close']
    print(data)
    data = data.drop(columns=['Open'])
    
   
    print(data)
    
    def shif_data_frame(df, an, v, shift):
        df_np = df.to_numpy()
        an_np = an.to_numpy()
        v_np = v.to_numpy()
        x = []
        y = []
        z = []

        for i in range(len(df_np)-shift):
            row = [r for r in df_np[i:i+shift]]
            x.append(row)
            #We are only predicting precent price change
            label = an_np[i+shift]
            y.append(label)
            label = v_np[i+shift]
            z.append(label)
        return np.array(x), np.array(y), np.array(z)
    

    lookback = 10
    x_np, y_np, v_np = shif_data_frame(data, answer, validation, lookback)

    split_index = int(0.8 * len(x_np))

    split_index2 = int(0.9 * len(x_np))

    
    x_train = x_np[:split_index]
    x_test = x_np[split_index:split_index2]
    x_pred = x_np[split_index2:]
    #x_pred = x_np

    y_train = y_np[:split_index]
    y_test = y_np[split_index:split_index2]
    y_pred = y_np[split_index2:]
    v_pred = v_np[split_index2:]
    #y_pred = y_np
    
    x_train = x_train.reshape(-1, lookback, 4)
    x_test = x_test.reshape(-1, lookback, 4)
    x_pred = x_pred.reshape(-1, lookback, 4)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    v_pred = v_pred.reshape(-1, 1)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    x_pred = torch.tensor(x_pred).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    y_pred = torch.tensor(y_pred).float()
    v_pred = torch.tensor(v_pred).float()


    print(x_test.shape)
    class TimeSeriesData(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __getitem__(self, index):
            return self.x[index], self.y[index]

        def __len__(self):
            return len(self.x)


    train_dataset = TimeSeriesData(x_train, y_train)
    test_dataset = TimeSeriesData(x_test, y_test)
    pred_dataset = TimeSeriesData(x_pred, y_pred)
    batch_size = 10

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    pred_loader = DataLoader(dataset = pred_dataset, batch_size = 1, shuffle = False)

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    class Attention(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.d_in = d_in
            self.d_out = d_out
            self.Q = nn.Linear(d_in, d_out)
            self.K = nn.Linear(d_in, d_out)
            self.V = nn.Linear(d_in, d_out)

        def forward(self, x):
            queries = self.Q(x)
            keys = self.K(x)
            values = self.V(x)

            scores = torch.bmm(queries, keys.transpose(1,2))
            scores = scores / (self.d_out ** .5)

            attention = F.softmax(scores, dim=2)
            hidden_states = torch.bmm(attention, values)
            return hidden_states

    class LSTM_Attention_Model(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers):
            super(LSTM_Attention_Model, self).__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
            self.att = Attention(hidden_size, hidden_size)
            self.fc = nn.Linear(hidden_size, 1, )
        
        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            h1 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

            out, _ = self.lstm(x,  (h0,h1))
            out = self.att(out)
            out = self.fc(out[:,-1,:])
            return out


    model = LSTM_Attention_Model(4, 32, 2)


    model.to(device)


    learning_rate = 0.005
    num_epochs = 15
#
    loss_function = nn.MSELoss()
    
    #optimizer = torch.optim.SGD(model.named_parameters(), lr=learning_rate, momentum=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    def train_one_epoch():
        model.train()
        print(f'Epoch: {epoch+1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            
            output= model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 25 == 24:
                avg_loss_across_batches = running_loss / 25
                print('Batch {0}, Loss {1:.4f}'.format(batch_index+1,avg_loss_across_batches))
                running_loss = 0.0

    def validate_one_epoch():
        model.eval()
        running_loss = 0.0
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)


        print('AVG Loss: {0: .4f}'.format(avg_loss_across_batches))
        print()

    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()

    
    def predicting():
        model.eval()
        with torch.no_grad():
            predictions = []
            actual = []
            val = []
            count = 0
            for batch_index, batch in enumerate(pred_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                #print(x_batch)
                with torch.no_grad():
                    output = model(x_batch)
                    predictions.append((output[0][0].item()* multiple)+minimum)                
                    actual.append((y_batch[0][0].item()* multiple)+minimum)
                    val.append(output[0][0].item())
                    count+=1
            
            r2 = r2_score(v_pred, val)
            print('R2 Score: ', r2)

            mse = mean_squared_error(v_pred, val)

            np_val = np.array(val)
            np_v = np.array(v_pred)

            

            print("MSE: " + str(mse))
            print("RMSE: " + str(math.sqrt(mse)))
            print("MAPE: " + str(np.mean(np.abs((np_v - np_val) / np_v)) * 100))
            x = np.linspace(0, 250, 250)
            y = x
            #plt.plot(x, y, color = 'red')
            #plt.scatter(actual, predictions)
            plt.plot( actual)
            plt.plot(predictions, color = 'red')
            
            plt.show()
        
    predicting()

