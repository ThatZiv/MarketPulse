import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

modelLoc = 'models/checkpoints/transformer-old.pth'

def create_sequence(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(data[i + window_size])
    return torch.FloatTensor(sequences), torch.FloatTensor(labels)

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
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.fc(x)

def evaluate_model(model, dataloader, criterion, scaler):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.extend(outputs.numpy())
            all_labels.extend(labels.numpy())

    predictions = np.array(all_predictions).reshape(-1, 1)
    actuals = np.array(all_labels).reshape(-1, 1)

    # inversals
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    # metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # determine trend direction
    pred_direction = np.diff(predictions.flatten()) > 0
    actual_direction = np.diff(actuals.flatten()) > 0
    directional_accuracy = np.mean(pred_direction == actual_direction) * 100
    
    return {
        'loss': total_loss / len(dataloader),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'accuracy': directional_accuracy,
        'predictions': predictions,
        'actuals': actuals
    }

def plot_results(predictions, actuals, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='actual', color='blue')
    plt.plot(predictions, label='predicted', color='red')
    plt.title(title)
    plt.xlabel('date')
    plt.ylabel('price ($)')
    plt.legend()
    plt.show()

def main():
    # hyperparams
    input_size = 1
    hidden_size = 128
    num_heads = 4
    num_layers = 2
    window_size = 20
    epochs = 150
    lr = 0.0005

    # TODO: get from postgres
    # data = yf.download('F', start='2020-01-01', end='2024-12-27')
    import json
    import os
    data = json.load(open(os.path.join(os.path.dirname(__file__), 't.json')))
    # normalize
    # TODO: implement other features like volume, open, high, low
    data = np.array(data['Close']["values"]).reshape(-1, 1)
    # data = data['Close'].vales.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # train, validation, and test (60%, 20%, 20%)
    train_size = int(len(scaled_data) * 0.6)
    val_size = int(len(scaled_data) * 0.2)

    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size+val_size]
    test_data = scaled_data[train_size+val_size:]
    
    # transformer sequences
    train_sequences, train_labels = create_sequence(train_data, window_size)
    val_sequences, val_labels = create_sequence(val_data, window_size)
    test_sequences, test_labels = create_sequence(test_data, window_size)

    train_dataset = StockDataset(train_sequences, train_labels)
    val_dataset = StockDataset(val_sequences, val_labels)
    test_dataset = StockDataset(test_sequences, test_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = StockTransformer(input_size, hidden_size, num_heads, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Trainingg
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_sequences, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        val_metrics = evaluate_model(model, val_dataloader, criterion, scaler)

        # metrics
        print(f'\nEpoch [{epoch + 1}/{epochs}]')
        print(f'training Loss: {train_loss/len(train_dataloader):.4f}')
        print(f'validation Loss: {val_metrics["loss"]:.4f}')
        # print(f'validation RMSE: ${val_metrics["rmse"]:.2f}')
        print(f'validation R^2: {val_metrics["r2"]:.4f}')
        # print(f'validation Directional Accuracy: {val_metrics["accuracy"]:.2f}%')

        # early stopping when loss isnt improve
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), modelLoc)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nearly stopping...")
                break

    model.load_state_dict(torch.load(modelLoc))
    test_metrics = evaluate_model(model, test_dataloader, criterion, scaler)
    print('\nfinal metrics:')
    # print(f'MSE: {test_metrics["mse"]:.4f}')
    # print(f'RMSE: ${test_metrics["rmse"]:.2f}')
    print(f'R^2 Score: {test_metrics["r2"]:.4f}')
    # print(f'directional accuracy: {test_metrics["accuracy"]:.2f}%')
    plot_results(test_metrics["predictions"], test_metrics["actuals"],
                "test set predictions vs actual set")

if __name__ == "__main__":
    main()