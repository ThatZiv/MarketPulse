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
import matplotlib.pyplot as plt
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
        for i in range(len(input_data)-shift-6):
            row = [r for r in input_data[i:i+shift]]
            x.append(row)
            #We are only predicting precent price change
            label = answer[i+shift:i+shift+7]
            #label = answer[i+shift]
            y.append(label)
        return np.array(x), np.array(y)

    # Normalize smooth and format the data frame for predictions
    def format_data(self, data):
        
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

        #data['High'] = (data['High'] - data['High'].min())/ (data['High'].max() - data['High'].min())
        data['Low'] = (data['Low'] - data['Low'].min())/ (data['Low'].max() - data['Low'].min())
        data['Open'] = (data['Open'] - data['Open'].min())/ (data['Open'].max() - data['Open'].min()) 
    
        data = data.drop(columns=['Open'])
        answer = data['Close']
        print(data)

        data2, answer = self.create_inout_sequences(data, 20, answer)
        _, valid_answer = self.create_inout_sequences(data, 20, valid_answer)
        
        return data2, answer, valid_answer, multiple, minimum
        

    def get_data(self, data, answer, train_answer, split, lookback = 20):
        split_index = int(len(data)*(1-split))
        split_index2 = int(len(data)*(1-split*2))

        x_train = data[:split_index2]
        x_test = data[split_index2:split_index]
        x_valid = data[split_index:]

        print(x_train.shape)
        print(x_test.shape)

        # training and testing use smoothed data validation uses raw data
        y_train = train_answer[:split_index2]
        y_test = train_answer[split_index2:split_index]
        y_valid = answer[split_index:]

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 4)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 4)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 4)

        y_train = y_train.reshape(-1, 7)
        y_test = y_test.reshape(-1, 7)
        y_valid = y_valid.reshape(-1, 7)

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
        for _, batch in enumerate(train_loader):
            print(batch)
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            print(x_batch.shape, y_batch.shape)
            break

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
                    print('Batch {0}, Loss {1:.4f}'.format(batch_index+1,avg_loss_across_batches))
                    running_loss = 0.0

        def test():
            self.model.eval()
            running_loss = 0.0
            for batch_index, batch in enumerate(test_loader):
                x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

                with torch.no_grad():
                    output = self.model(x_batch)
                    loss = self.loss_function(output, y_batch)
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(test_loader)


            print('AVG Loss: {0: .4f}'.format(avg_loss_across_batches))
            print()
        
        self.model.to(self.device)
        for epoch in range(epochs):
            train(train_loader)
            test()

    def evaluate(self, eval_model, data_source):
        self.model.eval()
        print(data_source.__len__())
        value = 0
       
        val =  [[value for _ in range(data_source.__len__())] for _ in range(7)]
        anws =  [[value for _ in range(data_source.__len__())] for _ in range(7)]
        running_loss = 0.0
        count = 0
        for batch_index, batch in enumerate(data_source):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()
                #print(output, y_batch)
        
                val[0][count] = output[0][0].item()
                anws[0][count] = y_batch[0][0].item()
                val[1][count] = output[0][1].item()
                anws[1][count] = y_batch[0][1].item()
                val[2][count] = output[0][2].item()
                anws[2][count] = y_batch[0][2].item()
                val[3][count] = output[0][3].item()
                anws[3][count] = y_batch[0][3].item()
                val[4][count] = output[0][4].item()
                anws[4][count] = y_batch[0][4].item()
                val[5][count] = output[0][5].item()
                anws[5][count] = y_batch[0][5].item()
                val[6][count] = output[0][6].item()
                anws[6][count] = y_batch[0][6].item()
                print(output[0][0].item(), output[0][1].item(), output[0][2].item(), output[0][3].item(), output[0][4].item(), output[0][5].item(), output[0][6].item())
                print(val[0][count],val[1][count],val[2][count],val[3][count],val[4][count],val[5][count],val[6][count])
                print(y_batch[0][0].item(), y_batch[0][1].item(), y_batch[0][2].item(), y_batch[0][3].item(), y_batch[0][4].item(), y_batch[0][5].item(), y_batch[0][6].item())
                print(anws[0][count],anws[1][count],anws[2][count],anws[3][count],anws[4][count],anws[5][count],anws[6][count])
                print()
                count += 1

        for i in range(7):
            print("Day: " + str(i))
            r2 = r2_score(anws[i], val[i])
            mse = mean_squared_error(anws[i], val[i])

            np_val = np.array(val[i])
            np_v = np.array(anws[i])
            print('R2 Score: ', r2)
            print("MSE: " + str(mse))
            print("RMSE: " + str(math.sqrt(mse)))
            print("MAPE: " + str(np.mean(np.abs((np_v - np_val) / np_v)) * 100))
            

        plt.plot(anws[0])
        plt.plot(val[0], color = 'red')
            
        plt.show()
        plt.plot(anws[1])
        plt.plot(val[1], color = 'red')
            
        plt.show()
        plt.plot(anws[2])
        plt.plot(val[2], color = 'red')
            
        plt.show()
        plt.plot(anws[3])
        plt.plot(val[3], color = 'red')
            
        plt.show()
        plt.plot(anws[4])
        plt.plot(val[4], color = 'red')
            
        plt.show()
        plt.plot(anws[5])
        plt.plot(val[5], color = 'red')
            
        plt.show()
        plt.plot(anws[6])
        plt.plot(val[6], color = 'red')
            
        plt.show()
        

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
        self.fc = nn.Linear(hidden_size, 7, )
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
