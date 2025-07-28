import numpy as np
import pandas as pd

import yfinance

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from arch import arch_model

import plotly.express as px
import plotly.graph_objects as go
start_date = '2005-01-01'
end_date = '2025-01-01'

vix = yfinance.download('^VIX', start=start_date, end=end_date, interval='1d', auto_adjust=False)
vix.columns = ['Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
vix.drop(columns='Volume', inplace=True)
vix['log_returns'] = np.log(vix['Close']/vix['Close'].shift(1))
vix.dropna(inplace=True)
print(f"VIX Number of NA's: {vix.isna().sum().sum()}")
print(vix.head())

sp500 = yfinance.download('^GSPC', start=start_date, end=end_date, interval='1d', auto_adjust=False)
sp500.columns = ['Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
sp500.drop(columns='Volume', inplace=True)
sp500['log_returns'] = np.log(sp500['Close']/sp500['Close'].shift(1))
sp500.dropna(inplace=True)
sp500['Volatility'] = sp500['High']/sp500['Low'] - 1

print(f"SP500 Number of NA's: {sp500.isna().sum().sum()}")
print(sp500.head())

garch_model = arch_model(y=sp500['log_returns'], x=sp500, mean='Zero', vol='GARCH', p=1, q=1)

garch_result = garch_model.fit()

print("Fitting done!\n")

print(garch_result.summary())
forecast = garch_result.forecast(params=garch_result.params, horizon=10)
forecast.variance
garch_std_resid = garch_result.std_resid.values.reshape(-1, 1)
vix_log_returns = vix['log_returns'].values.reshape(-1, 1)

print(garch_std_resid.shape)
print(vix_log_returns.shape)


# Some reason the length is off by 1 T^T (5032 vs 5033), idk why
min_len = min(len(garch_std_resid), len(vix_log_returns))
garch_std_resid = garch_std_resid[-min_len:]
vix_log_returns = vix_log_returns[-min_len:]

# GARCH standardized residuals are already standardized, so no need for StandardScaler
# Only scale the VIX log returns
vix_scaler = StandardScaler().fit(vix_log_returns)
vix_log_returns_norm = vix_scaler.transform(vix_log_returns)

features = np.concatenate([garch_std_resid, vix_log_returns_norm], axis=1)


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(features, SEQ_LEN)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Train/test split
split = int(0.6 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)



class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = StockDataset(X_train, y_train)
test_ds = StockDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(hidden_size // 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.fc1(out)
        # out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
EPOCHS = 50
train_losses = []
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()*xb.size(0)
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    if epoch%5 == 0: print(f"Epoch {epoch}/{EPOCHS}, Training Loss: {avg_loss:.6f}")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = model(X_test_tensor).cpu().numpy()

    preds_vix = preds.reshape(-1, 1)
    y_test_vix = y_test.reshape(-1, 1)
    preds_inv = vix_scaler.inverse_transform(preds_vix)
    y_test_inv = vix_scaler.inverse_transform(y_test_vix)

r2 = r2_score(y_test_inv, preds_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")