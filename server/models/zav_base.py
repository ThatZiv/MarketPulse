
# timeseries transformer-based implementation of stock forecasting

import yfinance as yf
import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def create_sequence(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        # next value
        labels.append(data[i + window_size])

    return torch.stack(sequences), torch.stack(labels)

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class StockTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(StockTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads), num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # input transformation
        x = self.embedding(x)
        x = x.permute(1, 0, 2) # (seq_len, batch_size, hidden_size)
        x = self.transformer_encoder(x)
        # not sure what this means but it pools across sequence (interchangeable strategy)
        x = x.mean(dim=0) # (batch_size, hidden_size)
        return self.fc(x) # (batch_size, 1)

def predict_next_day(model, last_sequence, scaler):
    model.eval()
    with torch.no_grad():
        scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
        scaled_prediction = model(sequence_tensor)
        prediction = scaler.inverse_transform(scaled_prediction.cpu().numpy().reshape(-1, 1))
        return prediction[0][0]


def main():
    input_size = 1
    hidden_size = 64
    num_heads = 4
    num_layers = 2
    window_size = 12
    epochs = 10

    lr = 0.001
    data = yf.download('TSLA', start='2020-01-01', end='2024-12-27')
    # stock_data = yf.Ticker("TSLA")
    # data = stock_data.history(period="1y")
    data = data['Close'].values
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_sequences, train_labels = create_sequence(train_data, window_size)

    recent_data = data[-window_size:].copy()
    data = data.reshape(-1, 1)
    # normalize
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = torch.FloatTensor(data)
    data = data.view(-1, 1)

    sequences, labels = create_sequence(data, window_size)
    dataset = StockDataset(sequences, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = StockTransformer(input_size, hidden_size, num_heads, num_layers)

    # use mean squared err loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Trainingg
    model.train()
    for epoch in range(epochs):
        for batch_sequences, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f'epoch [{epoch + 1}/{epochs}], loss: {loss.item():.4f}')
        
 

    torch.save(model.state_dict(), 'models/checkpoints/transformer-old.pth')


    


if __name__ == "__main__":
    main()