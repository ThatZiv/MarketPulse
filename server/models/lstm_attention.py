from database.yfinanceapi import bulk_stock_data
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


def attention_lstm(ticker):
    device = 'cpu'
    data = bulk_stock_data(ticker)
    #normalized volume
    data['Volume'] = (data['Volume'] - data['Volume'].min())/ (data['Volume'].max() - data['Volume'].min())
    # normalized precent difference between open and close
    data['Open'] = ((data['Close']/data['Open'])-(data['Close']/data['Open']).min())/((data['Close']/data['Open']).max()-(data['Close']/data['Open']).min())
    # normalized
    for i in range(1, len(data['High'])):
        data['High'][i] = 1-(data['Close'][i]/data['Close'][i-1])
        data['Low'][i] = data['Close'][i]-data['Close'][i-1]
    data['High'][0] = 0
    data['Low'][0] = 0

    multiple = (data['Low'].max() - data['Low'].min())
    minimum =  data['Low'].min()
    close = data['Close']
    data['Close'] = (data['Close'] - data['Close'].min())/ (data['Close'].max() - data['Close'].min()) 
    data['High'] = (data['High'] - data['High'].min())/ (data['High'].max() - data['High'].min())
    data['Low'] = (data['Low'] - data['Low'].min())/ (data['Low'].max() - data['Low'].min())
    data.set_index('Date', inplace = True)
    
    answer = data['Low']
    data = data.drop(columns=['Low','Close', 'Dividends', 'Stock Splits'])
    
    print(data)
    
    def shif_data_frame(df, an,shift):
        df_np = df.to_numpy()
        an_np = an.to_numpy()
        x = []
        y = []

        for i in range(len(df_np)-shift):
            row = [r for r in df_np[i:i+shift]]
            x.append(row)
            #We are only predicting precent price change
            label = an_np[i+shift]
            y.append(label)
        return np.array(x), np.array(y)
    
    lookback = 20
    x_np, y_np = shif_data_frame(data, answer, lookback)

    split_index = int(0.90 * len(x_np))

    
    x_train = x_np[:split_index]
    x_test = x_np[split_index:-15]
    x_pred = x_np[-20:]
    
    y_train = y_np[:split_index]
    y_test = y_np[split_index:-15]
    y_pred = y_np[-20:]
    
    x_train = x_train.reshape(-1, lookback, 3)
    x_test = x_test.reshape(-1, lookback, 3)
    x_pred = x_pred.reshape(-1, lookback, 3)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    x_pred = torch.tensor(x_pred).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    y_pred = torch.tensor(y_pred).float()



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
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            h1 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

            out, _ = self.lstm(x, (h0, h1))
            out = self.att(out)
            #out = self.att(out)
            #out = self.att(out)
            out = self.fc(out[:,-1,:])
            return out


    model = LSTM_Attention_Model(3, 32, 8)

    model.to(device)

    learning_rate = 0.0001
    num_epochs = 5

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    def train_one_epoch():
        model.train()
        print(f'Epoch: {epoch+1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output= model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_index % 25 == 24:
                avg_loss_across_batches = running_loss / 25
                print('Batch {0}, Loss {1:.4f}'.format(batch_index+1,avg_loss_across_batches))
                running_loss = 0.0

    def validate_one_epoch():
        model.train(False)
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
            count = 0
            for batch_index, batch in enumerate(pred_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                #print(x_batch)
                with torch.no_grad():
                    output = model(x_batch)
                    predictions.append((output[0][0].item()*multiple)+minimum+close[len(close)-1-20+count])                
                    actual.append((y_batch[0][0].item()*multiple)+minimum+close[len(close)-1-20+count])
                    count+=1
            print(predictions)
            print(actual)
            plt.plot(actual, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.legend()
            plt.show()
        
    predicting()

