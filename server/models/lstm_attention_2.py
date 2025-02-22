import math
import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pywt
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error



class AttentionLstm:

    def __init__(self, input_size = 4, hidden_size = 32, num_layers = 2, output_size = 1, batch_size = 10, learning_rate = 0.005):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = LSTMAttentionModel(input_size, hidden_size, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = nn.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ticker = "TSLA".upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            row = [r for r in input_data[i:i+shift]]
            x.append(row)
            #We are only predicting precent price change
            label = answer[i+shift]
            y.append(label)
        return np.array(x), np.array(y)

    # Normalize smooth and format the data frame for predictions
    def format_data(self, data):
        data = pd.DataFrame(data)
        data['Volume'] = (data['Volume'] - data['Volume'].min())/(data['Volume'].max() - data['Volume'].min())
        multiple = (data['Close'].max() - data['Close'].min())
        minimum =  data['Close'].min()
        data['Close'] = (data['Close'] - data['Close'].min())/ (data['Close'].max() - data['Close'].min())
        valid_answer = data['Close']
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

        data = data.drop(columns=['Open'])
        answer = data['Close']

        return data, answer, valid_answer, multiple, minimum


    def get_data(self, data, answer, train_answer, split, lookback = 10):
        split_index = int(len(data)*(1-split))
        split_index2 = int(len(data)*(1-split*2))

        x_train = data[:split_index2]
        x_test = data[split_index2:split_index]
        x_valid = data[split_index:]

        # training and testing use smoothed data validation uses raw data
        y_train = train_answer[:split_index2]
        y_test = train_answer[split_index2:split_index]
        y_valid = answer[split_index:]

        x_train = x_train.reshape(-1, lookback, 4)
        x_test = x_test.reshape(-1, lookback, 4)
        x_valid = x_valid.reshape(-1, lookback, 4)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)

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
        for epoch in range(epochs):
            self.train(train_loader, epoch)
            self.test(test_loader, epoch)

    def train(self, train_data, epoch=0):
        self.model.train()
        print(f'Epoch: {epoch+1}')
        running_loss = 0.0
        for batch_index, batch in enumerate(train_data):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            output= self.model(x_batch)
            loss = self.loss_function(output, y_batch)
            running_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 25 == 24:
                avg_loss_across_batches = running_loss / 25
                print('Batch {0}, Loss {1:.4f}'.format(batch_index+1,avg_loss_across_batches))
                running_loss = 0.0

    def test(self, test_data, epoch):
        self.model.eval()
        running_loss = 0.0
        for batch_index, batch in enumerate(test_data):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_data)


        print('AVG Loss: {0: .4f}'.format(avg_loss_across_batches))
        print()

    def evaluate(self, eval_model, data_source):
        eval_model.eval()

        val = []
        anws = []
        running_loss = 0.0
        for batch_index, batch in enumerate(data_source):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = eval_model(x_batch)
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


    def forecast_seq(self, sequences):
        return sequences


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
        self.fc = nn.Linear(hidden_size, 1, )

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
    coes = pywt.wavedec(data, wavelet, mode = 'reflect')
    threshold = .01
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
