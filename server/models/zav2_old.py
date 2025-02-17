import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import yfinance as yf
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

input_window = 10 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
batch_size = 250
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "models/checkpoints"
model_loc = f"{model_dir}/z-transformer.pth"
model_path = os.path.join(os.path.dirname(__file__), "checkpoints/z-transformer.pth")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_data(data, split):
    """Split ratio of training data"""

    series = data
    
    split = round(split*len(series))
    train_data = series[:split]
    test_data = series[split:]

    train_data = train_data.cumsum()
    train_data = 2*train_data # Training data augmentation, increase amplitude for the model to better generalize.(Scaling by 2 is aribitrary)
                              # Similar to image transformation to allow model to train on wider data sets

    test_data = test_data.cumsum()

    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

def train(train_data):
    model.train() # Turn on the evaluation mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if log_interval > 0 and batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'epoch {epoch:3d} | {batch:5d}/{len(train_data) // batch_size:5d} batches | '
                      f'lr {scheduler.get_lr()[0]:02.10f} | {elapsed * 1000 / log_interval:5.2f} ms | '
                      f'loss {cur_loss:5.7f}')
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            _data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(_data)
            total_loss += len(_data[0])* criterion(output, targets).cpu().item()
            all_preds.append(output.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_preds = all_preds.view(-1).numpy()
    all_targets = all_targets.view(-1).numpy()
    r2 = r2_score(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    mape = mean_absolute_percentage_error(all_targets, all_preds)
    print(f"R^2: {r2}")
    print(f"MSE: {mse}")
    print(f"MAPE: {mape}%")
    return total_loss / len(data_source)

def model_forecast(model, seqence):
    model.eval() 


    seq = np.pad(seqence, (0, 3), mode='constant', constant_values=(0, 0))
    seq = create_inout_sequences(seq, input_window)
    seq = seq[:-output_window].to(device)

    seq, _ = get_batch(seq, 0, 1)
    with torch.no_grad():
        for i in range(0, output_window):
            output = model(seq[-output_window:])
            seq = torch.cat((seq, output[-1:]))

    seq = seq.cpu().view(-1).numpy()

    return seq

def forecast_seq(model, sequences):
    """Sequences data has to been windowed and passed through device"""
    start_timer = time.time()
    model.eval()
    forecast_seq = torch.Tensor(0)
    actual = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(sequences) - 1):
            data, target = get_batch(sequences, i, 1)
            output = model(data)
            forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
            actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)
    timed = time.time()-start_timer
    print(f"{timed} sec")

    return forecast_seq, actual

def main():
    pass

ticker = "TM"
data = yf.download(ticker, start='2020-01-01', end='2024-12-27')
import json
import os
data = json.load(open(os.path.join(os.path.dirname(__file__), f'mockStocks/{ticker}.json'), 'r'))
close = pd.DataFrame(data["close"])

# close = np.array(pd.DataFrame(data['Close'].values))

# Calculate log returns
logreturn = np.diff(np.log(close), axis=0)
# normalize
csum_logreturn = logreturn.cumsum()
fig, axs = plt.subplots(2, 1)
axs[0].plot(close, color='red')
axs[0].set_title('Closing Price')
axs[0].set_ylabel('Close Price')
axs[0].set_xlabel('Time Steps')

axs[1].plot(csum_logreturn, color='green')
axs[1].set_title('Cumulative Sum of Log Returns')
axs[1].set_xlabel('Time Steps')

fig.tight_layout()
plt.show()



train_data, val_data = get_data(logreturn, 0.6) # 60% train, 40% test split
model = TransAm()
model.to(device)
criterion = nn.MSELoss() # Loss function
lr = 0.0005 # learning rate
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_loc))
    print("Model loaded from checkpoint.")
    test_eval = evaluate(model, val_data)
    print(f"Test loss: {test_eval}")
    test_result, truth = forecast_seq(model, val_data)
    plt.plot(truth, color='red', alpha=0.7)
    plt.plot(test_result, color='blue', linewidth=0.7)
    plt.title('Actual vs Forecast')
    plt.legend(['Actual', 'Forecast'])
    plt.xlabel('Time Steps')
    plt.show()
    exit(0)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs =  100

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    
    if(epoch % epochs == 0): # valid model after last training epoch
        val_loss = evaluate(model, val_data)
        print(f'epoch {epoch} | time: {time.time() - epoch_start_time}s | valid loss: {val_loss}')
    else:
        print(f'epoch {epoch} | time: {time.time() - epoch_start_time}s')

    scheduler.step() 

test_result, truth = forecast_seq(model, val_data)
plt.plot(truth, color='red', alpha=0.7)
plt.plot(test_result, color='blue', linewidth=0.7)
plt.title('Actual vs Forecast')
plt.legend(['Actual', 'Forecast'])
plt.xlabel('Time Steps')
plt.show()

torch.save(model.state_dict(), model_loc)



# if __name__ == "__main__":
#     main()