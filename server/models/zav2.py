"""
An implementation of a transformer model for stock price prediction
"""
import time
import os
# import json
import math
import torch
from torch import nn
import numpy as np
# import pandas as pd
# import yfinance as yf
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

# pylint: disable=line-too-long

class Transformer:
    """transformer model wrapper for timeseries forecasting"""
    def __init__(self, input_window=10, output_window=1, batch_size=250, lr=0.0005):
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.lr = lr
        self.ticker = "TSLA".upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "checkpoints"
        # use a ticker-specific model or a generalizable one/
        self.use_spec_model = False
        self.file_name = f"z-transformer2{'-' + self.ticker if self.use_spec_model else '' }.pth"
        self.model_loc = f"{self.model_dir}/{self.file_name}"
        self.model_path = os.path.join(os.path.dirname(__file__), self.model_loc)

        # pylint: disable=consider-using-with
        # data = json.load(open(os.path.join(os.path.dirname(__file__), f'mockStocks/{self.ticker}.json'), 'r', encoding='utf-8'))
        # pylint: enable=consider-using-with
        # close = pd.DataFrame(data["close"])

        # close = np.array(pd.DataFrame(data['Close'].values))

        # log returns for normalization isntead of minmax
        # logreturn = np.diff(np.log(close), axis=0)
        # # normalize
        # csum_logreturn = logreturn.cumsum()
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(close, color='red')
        # axs[0].set_title('closing Prices')
        # axs[0].set_ylabel('close Price')
        # axs[0].set_xlabel('Time steps')

        # axs[1].plot(csum_logreturn, color='green')
        # axs[1].set_title('CSLR')
        # axs[1].set_xlabel('Time Steps')

        # fig.tight_layout()
        # plt.show()

        self.model = TransformerModel()
        self.model.to(self.device)
        self.criterion = nn.MSELoss() # Loss function
        self.s_split = None

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        # # training
        # self.training_seq(train_data, val_data)

        # torch.save(self.model.state_dict(), self.model_path)
        # print(f"Model saved to {self.model_path}")

    def create_inout_sequences(self, input_data, tw):
        """ create input and output sequences for transformer"""
        inout_seq = []
        input_len = len(input_data)
        for i in range(input_len-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+self.output_window:i+tw+self.output_window]
            inout_seq.append((train_seq ,train_label))
        return torch.FloatTensor(np.array(inout_seq))

    # def load_and_run(self, val_data):
    #     """ load local model and run """
    #     test_eval = self.evaluate(self.model, val_data)
    #     print(f"Test loss: {test_eval}")
    #     test_result, truth = self.forecast_seq(val_data)
    #     plt.plot(truth, color='red', alpha=0.7)
    #     plt.plot(test_result, color='blue', linewidth=0.7)
    #     plt.title('Actual vs Forecast')
    #     plt.legend(['Actual', 'Forecast'])
    #     plt.xlabel('Time Steps')
    #     plt.show()

    def training_seq(self, train_data, val_data, epochs=200):
        """ train model on epoch """
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train(train_data, epoch)

            if epoch % epochs == 0:
                val_loss = self.evaluate(self.model, val_data)
                print(f'epoch {epoch} | time: {time.time() - epoch_start_time:.2f}s | valid loss: {val_loss}')
            else:
                print(f'epoch {epoch} | time: {time.time() - epoch_start_time:.2f}s')

            self.scheduler.step()


    def get_data(self, data, split):
        """split data into train and test set"""

        series = np.diff(np.log(data), axis=0)
        split_point = round(split * len(series))
        train_data = series[:split_point]
        test_data = series[split_point:]

        # Store initial price of test data (S_split)
        self.s_split = data[split_point]  # data is original price series

        train_data = train_data.cumsum()
        train_data = 2 * train_data  # Training data augmentation
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
        """ forecast sequence of data """
        self.model.eval()
        forecast_seq = torch.Tensor(0)
        actual = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, len(sequences) - 1):
                data, target = self.get_batch(sequences, i, 1)
                output = self.model(data)
                forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
                actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)

        # de-normalize the forecast to actual stock price values
        forecast_prices = self.s_split * np.exp(forecast_seq.numpy())
        actual_prices = self.s_split * np.exp(actual.numpy())

        return forecast_prices, actual_prices

    def predict_future(self, current_data, days_ahead, use_true_as_input=False):
        """
        predict stock prices for 'days_ahead' days into the future
        """
        self.model.eval()

        if len(current_data) < self.input_window:
            raise ValueError("not enough data to predict future")

        last_price = current_data[-1]

        log_returns = np.diff(np.log(current_data), axis=0)

        input_seq = log_returns[-self.input_window:].cumsum()

        input_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)

        predicted_returns = []

        with torch.no_grad():
            for i in range(days_ahead):
                x = input_seq.unsqueeze(2).transpose(0, 1)

                output = self.model(x)
                next_return = output[-1].item()
                predicted_returns.append(next_return)

                if i < days_ahead - 1:
                    if use_true_as_input and i + self.input_window < len(log_returns):
                        # use actual data (for backtesting only)
                        next_true_return = log_returns[self.input_window + i]
                        input_seq = torch.cat([input_seq[:, 1:],
                                            torch.FloatTensor([[next_true_return]]).to(self.device)], dim=1)
                    else:
                        input_seq = torch.cat([input_seq[:, 1:],
                                            torch.FloatTensor([[next_return]]).to(self.device)], dim=1)

        # Convert cumulative log returns to prices
        predicted_returns = np.array(predicted_returns)
        predicted_prices = last_price * np.exp(predicted_returns)

        return predicted_prices


    def predict_with_confidence(self, current_data, days_ahead, num_samples=100, confidence_level=0.95):
        """
        predict stock prices with confidence intervals - adding random noise

        """
        all_predictions = []

        for _ in range(num_samples):
            noise_level = 0.001  # sensitivity to noise
            noisy_data = current_data * (1 + np.random.normal(0, noise_level, size=current_data.shape))

            preds = self.predict_future(noisy_data, days_ahead)
            all_predictions.append(preds)

        all_predictions = np.array(all_predictions)

        mean_predictions = np.mean(all_predictions, axis=0)
        lower_bound = np.percentile(all_predictions, (1 - confidence_level) / 2 * 100, axis=0)
        upper_bound = np.percentile(all_predictions, (1 + confidence_level) / 2 * 100, axis=0)

        return mean_predictions, lower_bound, upper_bound
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
        """add positional encoding to the input"""
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
