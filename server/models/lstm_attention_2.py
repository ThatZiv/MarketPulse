import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pywt
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error



class Attention_Lstm:

    def __init__(self, input_size = 4, hidden_size = 32, num_layers = 2, output_size = 1, batch_size = 10, learning_rate = 0.005):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = LSTM_Attention_Model(input_size, hidden_size, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = nn.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ticker = "TSLA".upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "checkpoints"
        self.use_spec_model = False
        self.file_name = f"attention_lstm_2{'-' + self.ticker if self.use_spec_model else '' }.pth"
        self.model_loc = f"{self.model_dir}/{self.file_name}"
        self.model_path = os.path.join(os.path.dirname(__file__), self.model_loc)
    
    def create_inout_sequences(self, input_data, tw):

    
    def training_seq(self, train_data, val_data, epochs=150):
    

    def get_data(self, data, split):
    
    def get_batch(self, source, i, batch_size=None):

    def train(self, train_data, epoch=0):
        self.model.train()
        print(f'Epoch: {epoch+1}')
        running_loss = 0.0
        for batch_index, batch in enumerate(train_data):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output= model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 25 == 24:
                avg_loss_across_batches = running_loss / 25
                print('Batch {0}, Loss {1:.4f}'.format(batch_index+1,avg_loss_across_batches))
                running_loss = 0.0

    def evaluate(self, eval_model, data_source):
        model.eval()
        
        val = []
        running_loss = 0.0
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()
                val.append(output[0][0].item())

        avg_loss_across_batches = running_loss / len(test_loader)
        

        print('AVG Loss: {0: .4f}'.format(avg_loss_across_batches))
        print()
        r2 = r2_score(v_pred, val)
        mse = mean_squared_error(v_pred, val)

        np_val = np.array(val)
        np_v = np.array(v_pred)
        
        print('R2 Score: ', r2)
        print("MSE: " + str(mse))
        print("RMSE: " + str(math.sqrt(mse)))
        print("MAPE: " + str(np.mean(np.abs((np_v - np_val) / np_v)) * 100))
            

    
    def forecast_seq(self, sequences):
    

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

def wavelet(data):
    wavelet = 'db4'
    coes = pywt.wavedec(data, wavelet, mode = 'reflect')
    threshold = .01
    coe_threshold = [pywt.threshold(c, threshold, mode='soft') for c in coes]
    smoothed = pywt.waverec(coe_threshold, wavelet)

    return smoothed

