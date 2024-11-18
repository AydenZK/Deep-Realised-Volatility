import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


DATA_DIR = Path(os.getcwd()).joinpath('data')


class LSTMModel(nn.Module):
    def __init__(self, num_features=16, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        out, _ = self.lstm(x)
        # Use the last hidden state
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()
    

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.to(torch.float32)
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def process_data():
    seq_features = pd.read_parquet(DATA_DIR.joinpath('train_seq_features.parquet'))
    train_targets = pd.read_csv(DATA_DIR.joinpath('train.csv'))
    target_array = pd.merge(
        seq_features.groupby(['stock_id', 'time_id']).count().reset_index(),
        train_targets,
        on=['time_id', 'stock_id']
    )['target'].to_numpy()

    features_target = torch.tensor(target_array)
    features_array = (
        seq_features
        .sort_values(['time_id', 'seconds_in_bucket'])
        .drop(columns=['time_id', 'stock_id'])
        .to_numpy()
    )

    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    features_tensor = torch.tensor(features_array.reshape((-1, 600, 16)))

    X_train, X_valid, y_train, y_valid = train_test_split(features_tensor, features_target, test_size=0.2)

    return StockDataset(X_train, y_train), StockDataset(X_valid, y_valid)
    
def rmspe(y_true, y_pred):
    epsilon = 1e-8
    return torch.sqrt(torch.mean(((y_true - y_pred) / (y_true + epsilon)) ** 2))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate")

    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    train_dataset, valid_dataset = process_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = LSTMModel(num_features=16)
    criterion = rmspe
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = torch.nan_to_num(inputs, nan=0.0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = torch.nan_to_num(inputs, nan=0.0)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        # Compute average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{args.num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


if __name__ == "__main__":
    train()