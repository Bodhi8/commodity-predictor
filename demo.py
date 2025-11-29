"""
Standalone Demo Script
Runs the full commodity prediction pipeline using synthetic data.
Works without network access.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


print("="*70)
print("COMMODITY PRICE PREDICTION - DEMO")
print("="*70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_commodity_data(n_days=2000, seed=42):
    """Generate synthetic commodity price data."""
    np.random.seed(seed)
    
    dates = pd.bdate_range(start='2017-01-01', periods=n_days)
    
    # Generate Crude Oil WTI with mean reversion
    prices = [70.0]
    for _ in range(n_days - 1):
        mean_price = 70
        theta = 0.02  # mean reversion speed
        sigma = 0.025  # volatility
        dW = np.random.normal(0, 1)
        new_price = prices[-1] + theta * (mean_price - prices[-1]) + sigma * prices[-1] * dW
        prices.append(max(new_price, 10))
    
    # Add seasonality
    seasonal = 0.03 * np.sin(2 * np.pi * np.arange(n_days) / 252)
    prices = np.array(prices) * (1 + seasonal)
    
    df = pd.DataFrame({'Crude Oil WTI': prices}, index=dates)
    
    # Add correlated commodities
    df['Gold'] = 1500 + np.cumsum(np.random.normal(0.5, 15, n_days))
    df['Natural Gas'] = 3 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 252) + np.cumsum(np.random.normal(0, 0.1, n_days))
    df['Natural Gas'] = df['Natural Gas'].clip(lower=1.5)
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df, target_col):
    """Create technical features."""
    result = df.copy()
    
    for col in df.columns:
        # Lag features
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            result[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Returns
        for period in [1, 5, 10, 21]:
            result[f'{col}_return_{period}d'] = df[col].pct_change(period)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            result[f'{col}_SMA_{window}'] = df[col].rolling(window=window).mean()
            result[f'{col}_price_to_SMA_{window}'] = df[col] / result[f'{col}_SMA_{window}']
        
        # Volatility
        returns = df[col].pct_change()
        for window in [5, 10, 21]:
            result[f'{col}_vol_{window}d'] = returns.rolling(window=window).std()
        
        # RSI
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result[f'{col}_RSI_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = df[col].rolling(window=20).mean()
        std = df[col].rolling(window=20).std()
        result[f'{col}_BB_position'] = (df[col] - (sma - 2*std)) / (4*std)
    
    return result


def prepare_sequences(df, target_col, seq_length=60, train_ratio=0.7, val_ratio=0.15):
    """Prepare sequences for LSTM."""
    # Drop NaN
    df = df.dropna()
    
    # Separate features and target
    target = df[target_col].shift(-1).dropna()  # Predict next day
    features = df.iloc[:-1]  # Align with target
    
    # Ensure alignment
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    
    # Scale
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i-seq_length:i])
        y.append(target_scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            target_scaler, features.index[seq_length + val_end:])


# ============================================================================
# MODELS
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 
                                                   dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001):
    """Train a model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size
    )
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return train_losses, val_losses


def predict(model, X):
    """Generate predictions."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        return model(X_tensor).cpu().numpy()


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Dir_Acc': dir_acc}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n[1/5] Generating synthetic commodity data...")
raw_data = generate_commodity_data(n_days=2500, seed=42)
print(f"Generated {len(raw_data)} days of data for {len(raw_data.columns)} commodities")

TARGET = 'Crude Oil WTI'
SEQ_LENGTH = 60

print(f"\n[2/5] Engineering features for {TARGET}...")
featured_data = create_features(raw_data, TARGET)
print(f"Created {len(featured_data.columns)} features")

print("\n[3/5] Preparing sequences...")
X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, test_dates = \
    prepare_sequences(featured_data, TARGET, seq_length=SEQ_LENGTH)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Input shape: {X_train.shape}")

input_size = X_train.shape[2]

print("\n[4/5] Training models...")
models_config = {
    'LSTM': LSTMModel(input_size, hidden_size=64, num_layers=2),
    'BiLSTM': BiLSTMModel(input_size, hidden_size=64, num_layers=2),
    'Transformer': TransformerModel(input_size, d_model=64, nhead=4, num_layers=2)
}

results = []
predictions = {}

for name, model in models_config.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")
    
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32
    )
    
    # Predict
    y_pred_scaled = predict(model, X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_actual = target_scaler.inverse_transform(y_test)
    
    predictions[name] = y_pred.flatten()
    
    # Metrics
    metrics = calculate_metrics(y_actual, y_pred)
    metrics['Model'] = name
    results.append(metrics)
    
    print(f"\n{name} Results:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  Directional Accuracy: {metrics['Dir_Acc']:.1f}%")

print("\n[5/5] Generating visualizations...")

# Create output directory
os.makedirs('/home/claude/commodity_predictor/results', exist_ok=True)

# Convert y_test back to actual values for plotting
y_actual = target_scaler.inverse_transform(y_test).flatten()

# Plot 1: All predictions vs actual
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(y_actual, label='Actual', color='black', linewidth=2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for (name, preds), color in zip(predictions.items(), colors):
    ax.plot(preds, label=name, alpha=0.7, color=color)
ax.set_title(f'{TARGET}: Model Predictions vs Actual')
ax.set_xlabel('Time Step')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/commodity_predictor/results/all_predictions.png', dpi=150)
plt.close()

# Plot 2: Model comparison bar chart
results_df = pd.DataFrame(results)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'Dir_Acc']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for ax, metric in zip(axes.flatten(), metrics_to_plot):
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.2f}', ha='center', va='bottom')

plt.suptitle('Model Performance Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('/home/claude/commodity_predictor/results/model_comparison.png', dpi=150)
plt.close()

# Plot 3: Best model predictions detail
best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
best_preds = predictions[best_model]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Time series
ax1 = axes[0]
ax1.plot(y_actual, label='Actual', color='blue', alpha=0.7)
ax1.plot(best_preds, label=f'{best_model} Predicted', color='red', alpha=0.7)
ax1.fill_between(range(len(y_actual)), y_actual, best_preds, alpha=0.3, color='gray')
ax1.set_title(f'{best_model}: Predictions vs Actual')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Price ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot
ax2 = axes[1]
ax2.scatter(y_actual, best_preds, alpha=0.5, s=20)
min_val, max_val = min(y_actual.min(), best_preds.min()), max(y_actual.max(), best_preds.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
ax2.set_xlabel('Actual Price ($)')
ax2.set_ylabel('Predicted Price ($)')
ax2.set_title('Predicted vs Actual (Scatter)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/commodity_predictor/results/best_model_detail.png', dpi=150)
plt.close()

# Save results
results_df.to_csv('/home/claude/commodity_predictor/results/results.csv', index=False)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(results_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'Dir_Acc']].to_string(index=False))
print(f"\nBest Model (by RMSE): {best_model}")
print(f"\nResults saved to: /home/claude/commodity_predictor/results/")
print("="*70)
