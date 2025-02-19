import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import yfinance as yf
import os
import json
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

# pylint: disable=line-too-long

class Transformer:
    """transformer model wrapper for timeseries forecasting"""
    def __init__(self, input_window=10, output_window=1, batch_size=250):
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "models/checkpoints"
        self.model_loc = f"{self.model_dir}/z-transformer2.pth"
        self.model_path = os.path.join(os.path.dirname(__file__), "checkpoints/z-transformer2.pth")
        self.ticker = "TM"
        # data = yf.download(self.ticker, start='2020-01-01', end='2024-12-27')
        data = json.load(open(os.path.join(os.path.dirname(__file__), f'mockStocks/{self.ticker}.json'), 'r', encoding='utf-8'))
        close = pd.DataFrame(data["close"])

        # close = np.array(pd.DataFrame(data['Close'].values))

        # log returns for normalization isntead of minmax
        logreturn = np.diff(np.log(close), axis=0)
        # normalize
        csum_logreturn = logreturn.cumsum()
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(close, color='red')
        axs[0].set_title('closing Prices')
        axs[0].set_ylabel('close Price')
        axs[0].set_xlabel('Time steps')

        axs[1].plot(csum_logreturn, color='green')
        axs[1].set_title('CSLR')
        axs[1].set_xlabel('Time Steps')

        fig.tight_layout()
        plt.show()

        train_data, val_data = self.get_data(logreturn, 0.6) # 60% train, 40% test split
        self.model = TransformerModel()
        self.model.to(self.device)
        self.criterion = nn.MSELoss() # Loss function
        lr = 0.0005 # learning rate
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_loc))
            print("Model loaded from checkpoint.")
            test_eval = self.evaluate(self.model, val_data)
            print(f"Test loss: {test_eval}")
            test_result, truth = self.forecast_seq(val_data)
            plt.plot(truth, color='red', alpha=0.7)
            plt.plot(test_result, color='blue', linewidth=0.7)
            plt.title('Actual vs Forecast')
            plt.legend(['Actual', 'Forecast'])
            plt.xlabel('Time Steps')
            plt.show()
            exit(0)


        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        epochs = 150

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train(train_data, epoch)

            if epoch % epochs == 0: # valid model after last training epoch
                val_loss = self.evaluate(self.model, val_data)
                print(f'epoch {epoch} | time: {time.time() - epoch_start_time}s | valid loss: {val_loss}')
            else:
                print(f'epoch {epoch} | time: {time.time() - epoch_start_time}s')

            self.scheduler.step()

        test_result, truth = self.forecast_seq(val_data)
        plt.plot(truth, color='red', alpha=0.7)
        plt.plot(test_result, color='blue', linewidth=0.7)
        plt.title('Actual vs Forecast')
        plt.legend(['Actual', 'Forecast'])
        plt.xlabel('Time Steps')
        plt.show()

        torch.save(self.model.state_dict(), self.model_loc)

    def create_inout_sequences(self, input_data, tw):
        """ create input and output sequences for transformer"""
        inout_seq = []
        input_len = len(input_data)
        for i in range(input_len-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+self.output_window:i+tw+self.output_window]
            inout_seq.append((train_seq ,train_label))
        return torch.FloatTensor(np.array(inout_seq))

    def get_data(self, data, split):
        """split data into train and test set"""

        series = data
        split = round(split*len(series))
        train_data = series[:split]
        test_data = series[split:]

        train_data = train_data.cumsum()
        train_data = 2*train_data # Training data augmentation, increase amplitude for the model to better generalize.(Scaling by 2 is aribitrary)
                                # Similar to image transformation to allow model to train on wider data sets

        test_data = test_data.cumsum()

        train_sequence = self.create_inout_sequences(train_data,self.input_window)
        train_sequence = train_sequence[:-self.output_window]

        test_data = self.create_inout_sequences(test_data,self.input_window)
        test_data = test_data[:-self.output_window]

        return train_sequence.to(self.device), test_data.to(self.device)

    def get_batch(self, source, i, batch_size=None):
        """get batch of data"""
        
        seq_len = min(batch_size or self.batch_size, len(source) - 1 - i)
        data = source[i:i+seq_len]
        x = torch.stack(torch.stack([item[0] for item in data]).chunk(self.input_window, 1))
        target = torch.stack(torch.stack([item[1] for item in data]).chunk(self.input_window, 1))
        return x, target

    def train(self, train_data, epoch=0):
        """train model on training set"""
        self.model.train() # Turn on the evaluation mode
        total_loss = 0.
        start_time = time.time()

        for batch, i in enumerate(range(0, len(train_data) - 1, self.batch_size)):
            data, targets = self.get_batch(train_data, i)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(train_data) / self.batch_size / 5)
            if log_interval > 0 and batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(f'epoch {epoch:3d} | {batch:5d}/{len(train_data) // self.batch_size:5d} batches | '
                      f'lr {self.scheduler.get_lr()[0]:02.10f} | {elapsed * 1000 / log_interval:5.2f} ms | '
                      f'loss {cur_loss:5.7f}')
                total_loss = 0
                start_time = time.time()

    def evaluate(self, eval_model, data_source):
        """evaluate given model on valuation data"""
        eval_model.eval()
        total_loss = 0.
        eval_batch_size = 1000
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                _data, targets = self.get_batch(data_source, i, eval_batch_size)
                output = eval_model(_data)
                total_loss += len(_data[0])* self.criterion(output, targets).cpu().item()
                all_preds.append(output.cpu())
                all_targets.append(targets.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_preds = all_preds.view(-1).numpy()
        all_targets = all_targets.view(-1).numpy()
        r2 = r2_score(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = root_mean_squared_error(all_targets, all_preds)
        # mape = mean_absolute_percentage_error(all_targets, all_preds)
        print(f"R^2: {r2}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        # TODO: figure out why mape is always not accurate
        # print(f"MAPE: {mape}")
        # print(f"% accurate based on MAPE: {mape*100:.2f}%")
        return total_loss / len(data_source)

    def forecast_seq(self, sequences):
        """Sequences data has to been windowed and passed through device"""
        start_timer = time.time()
        self.model.eval()
        forecast_seq = torch.Tensor(0)
        actual = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, len(sequences) - 1):
                data, target = self.get_batch(sequences, i, 1)
                output = self.model(data)
                forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
                actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)
        timed = time.time()-start_timer
        print(f"{timed} sec")

        return forecast_seq, actual

class PositionalEncoding(nn.Module):
    """positional encoding for variable sequences"""
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

class TransformerModel(nn.Module):
    """A transformer model for time series forecasting based on original transformer architecture"""
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the model"""
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        """Forward pass"""
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


def main():
    """ main func for testing """
    input_window = 10
    output_window = 1
    batch_size = 250
    Transformer(input_window, output_window, batch_size)

if __name__ == "__main__":
    main()

# pylint: enable=line-too-long