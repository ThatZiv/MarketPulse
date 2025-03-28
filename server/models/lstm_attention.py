# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=duplicate-code

import math
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pywt
#import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class AttentionLstm:

    def __init__(self, input_size = 3, hidden_size = 32, batch_size = 10, learning_rate = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.output_size = 1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = LSTMAttentionModel(self.input_size, self.hidden_size, self.num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ticker = "TSLA".upper()
        self.device = "cpu"
        self.model_dir = "checkpoints"
        self.use_spec_model = False
        self.file_name = f"attention_lstm_2{'-' + self.ticker if self.use_spec_model else '' }.pth"
        self.model_loc = f"{self.model_dir}/{self.file_name}"
        self.model_path = os.path.join(os.path.dirname(__file__), self.model_loc)
        self.loss_function = nn.MSELoss()

    def create_inout_sequences(self, input_data, shift, answer):
        input_data = input_data.to_numpy()
        answer = answer.to_numpy()
        x = []
        y = []
        for i in range(len(input_data)-shift):
            row = list(input_data[i:i + shift])
            x.append(row)
            label = answer[i+shift]
            y.append(label)
        #print("row")
        #print(np.array(x).shape)

        return np.array(x), np.array(y)

    # Return the last sequence to use for a prediction
    def create_prediction_sequence(self, input_data, shift):
        p = list(input_data[len(input_data) - shift + 1:len(input_data) + 1])
        return torch.tensor(np.array(p)).float()


    # Normalize smooth and format the data frame for predictions
    def format_data(self, data):
        data['Volume'] = (data['Volume'] - data['Volume'].min())/(data['Volume'].max() - data['Volume'].min())
        multiple = (data['Close'].max() - data['Close'].min())
        minimum =  data['Close'].min()
        data['Close'] = (data['Close'] - data['Close'].min())/ (data['Close'].max() - data['Close'].min())
        valid_answer = data
        data['Close'] = wavelet(data['Close'])[:len(data['Close'])]
        data['Low'] = wavelet(data['Low'])[:len(data['Close'])]
        data['High'] = wavelet(data['High'])[:len(data['Close'])]
        data['Open'] = wavelet(data['Open'])[:len(data['Close'])]
        for i in range(1, len(data['High'])):
            data.loc[i, 'Low'] = data.loc[i, 'High']-data.loc[i,'Low']
        #    data.loc[i, 'High'] = 1-(data.loc[i, 'Close']-data.loc[i-1,'Close'])
            valid_answer.loc[i, 'Low'] = valid_answer.loc[i, 'High']-valid_answer.loc[i,'Low']
        #    valid_answer.loc[i, 'High'] = 1-(valid_answer.loc[i, 'Close']-valid_answer.loc[i-1,'Close'])

        #data.loc[0, 'High'] = 0
        #data.loc[0, 'Low'] = 0

        #valid_answer.loc[0, 'High'] = 0
        #valid_answer.loc[0, 'Low'] = 0

        data['High'] = (data['High'] - data['High'].min())/ (data['High'].max() - data['High'].min())
        data['Low'] = (data['Low'] - data['Low'].min())/ (data['Low'].max() - data['Low'].min())
        data['Open'] = (data['Open'] - data['Open'].min())/ (data['Open'].max() - data['Open'].min())
        data = data.drop(columns=['Open', 'Volume', 'Sentiment_Data', 'News_Data'])


        valid_answer['Low'] = (valid_answer['Low'] - valid_answer['Low'].min())/ (valid_answer['Low'].max() - valid_answer['Low'].min())
        valid_answer['Open'] = (valid_answer['Open'] - valid_answer['Open'].min())/ (valid_answer['Open'].max() - valid_answer['Open'].min())
        sentiment = (valid_answer['Sentiment_Data'] - valid_answer['Sentiment_Data'].min())/ (valid_answer['Sentiment_Data'].max() - valid_answer['Sentiment_Data'].min())

        valid_answer = valid_answer.drop(columns=['Open', 'Volume', 'Sentiment_Data', 'News_Data'])
        answer = data
        #print(data)

        data2, answer = self.create_inout_sequences(data, 20, data)
        _, valid_answer = self.create_inout_sequences(data, 20, valid_answer)
        return data2, answer, valid_answer, multiple, minimum, sentiment

    def get_data(self, data, answer, train_answer, split):
        lookback  = 20
        split_index = int(len(data)*(1-split))
        split_index2 = int(len(data)*(1-split*2))

        x_train = data[:split_index2]
        x_test = data[split_index2:split_index]
        x_valid = data[split_index:]

        #print(x_train.shape)
        #print(x_test.shape)

        # training and testing use smoothed data validation uses raw data
        y_train = train_answer[:split_index2]
        y_test = train_answer[split_index2:split_index]
        y_valid = answer[split_index:]

        x_train = x_train.reshape(-1, lookback, 3)

        x_test = x_test.reshape(-1, lookback, 3)
        x_valid = x_valid.reshape(-1, lookback, 3)

        y_train = y_train.reshape(-1, 3)
        y_test = y_test.reshape(-1, 3)
        y_valid = y_valid.reshape(-1, 3)

        x_train = torch.tensor(x_train).float()
        x_test = torch.tensor(x_test).float()
        x_valid = torch.tensor(x_valid).float()

        y_train = torch.tensor(y_train).float()
        y_test = torch.tensor(y_test).float()
        y_valid = torch.tensor(y_valid).float()

        train_dataset = TimeSeriesData(x_train, y_train)
        test_dataset = TimeSeriesData(x_test, y_test)
        valid_dataset = TimeSeriesData(x_valid, y_valid)
        batch_size = 10

        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
        valid_loader = DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False)

        return train_loader, test_loader, valid_loader

    def model_training(self, train_loader, test_loader, epochs):

        def train(train_loader):
            self.model.train()
            print(f'Epoch: {epoch+1}')
            running_loss = 0.0
            for batch_index, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
                output= self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_index % 25 == 24:
                    avg_loss_across_batches = running_loss / 25
                    print(f'Batch {batch_index+1}, Loss {avg_loss_across_batches}')
                    running_loss = 0.0

        def test():
            self.model.eval()
            running_loss = 0.0
            for _, batch in enumerate(test_loader):
                x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

                with torch.no_grad():
                    output = self.model(x_batch)
                    loss = self.loss_function(output, y_batch)
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(test_loader)


            print(f'AVG Loss: {avg_loss_across_batches}')
            print()
        self.model.to(self.device)
        for epoch in range(epochs):
            train(train_loader)
            test()

    def evaluate(self, data_source):
        self.model.eval()

        val = []
        anws = []
        running_loss = 0.0
        for _, batch in enumerate(data_source):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()
                val.append(output[0][0].item())
                anws.append(y_batch[0][0].item())

        r2 = r2_score(anws, val)
        mse = mean_squared_error(anws, val)

        np_val = np.array(val)
        np_v = np.array(anws)

        print('R2 Score: ', r2)
        print("MSE: " + str(mse))
        print("RMSE: " + str(math.sqrt(mse)))
        print("MAPE: " + str(np.mean(np.abs((np_v - np_val) / np_v)) * 100))


    def forecast_seq(self, sequences, sentiment, period = 7):
        self.model.eval()

        average = sum(sentiment) / len(sentiment)
        adjustment = .05 * (sentiment[len(sentiment)-1]-average)
        p = []
        count=0
        for _ in range(period):
            count+=1
            output = self.model(sequences)
            p.append(output[-1][0].item()*(adjustment**count))
            temp = sequences[-1]
            temp = temp[1:]
            sequences[-1] = torch.cat((temp, output[-1].unsqueeze(0)), dim = 0 )
        print(p)
        return p


class Attention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)

    def forward(self, x):
        queries = self.q(x)
        keys = self.k(x)
        values = self.v(x)

        scores = torch.bmm(queries, keys.transpose(1,2))
        scores = scores / (self.d_out ** .5)

        attention = F.softmax(scores, dim=2)
        hidden_states = torch.bmm(attention, values)
        return hidden_states

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.att = Attention(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 3, )
        self.device =  "cpu"

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        h1 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x,  (h0,h1))
        out = self.att(out)
        out = self.fc(out[:,-1,:])
        return out

def wavelet(data):
    wavelet_graph = 'db4'
    coes = pywt.wavedec(data, wavelet_graph, mode = 'reflect')
    threshold = .001
    coe_threshold = [pywt.threshold(c, threshold, mode='soft') for c in coes]
    smoothed = pywt.waverec(coe_threshold, wavelet_graph)

    return smoothed

class TimeSeriesData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
