"""
Deep Learning Models for Commodity Price Prediction
Implements LSTM, BiLSTM, CNN-LSTM, and Transformer architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import math


class LSTMModel(nn.Module):
    """
    Standard LSTM model for time series prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for capturing both forward and backward patterns.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid: CNN extracts local patterns, LSTM captures temporal dependencies.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Back to (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.
    Uses self-attention to capture long-range dependencies.
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take the last time step
        x = x[:, -1, :]
        output = self.fc(x)
        return output


class LSTMTransformerHybrid(nn.Module):
    """
    Hybrid LSTM + Transformer model.
    LSTM captures sequential patterns, Transformer adds attention mechanism.
    """
    def __init__(self, input_size, hidden_size=64, d_model=64, nhead=4, 
                 num_lstm_layers=2, num_transformer_layers=2, dropout=0.2):
        super(LSTMTransformerHybrid, self).__init__()
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Transformer branch
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Combine both branches
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]
        
        # Transformer branch
        tf_x = self.input_projection(x)
        tf_x = self.pos_encoder(tf_x)
        tf_out = self.transformer(tf_x)
        tf_features = tf_out[:, -1, :]
        
        # Combine
        combined = torch.cat([lstm_features, tf_features], dim=1)
        output = self.fc(combined)
        return output


class ModelTrainer:
    """Handles training and evaluation of models."""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              early_stopping_patience=15, verbose=True):
        """
        Train the model with early stopping.
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}',
                    'best': f'{best_val_loss:.6f}'
                })
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.train_losses, self.val_losses
    
    def predict(self, X):
        """Generate predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()


def get_model(model_name, input_size, **kwargs):
    """Factory function to get model by name."""
    models = {
        'lstm': LSTMModel,
        'bilstm': BiLSTMModel,
        'cnn_lstm': CNNLSTMModel,
        'transformer': TransformerModel,
        'lstm_transformer': LSTMTransformerHybrid
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()](input_size, **kwargs)


if __name__ == "__main__":
    # Test model creation
    batch_size = 32
    seq_len = 60
    features = 50
    
    x = torch.randn(batch_size, seq_len, features)
    
    print("Testing models...")
    
    for name in ['lstm', 'bilstm', 'cnn_lstm', 'transformer', 'lstm_transformer']:
        model = get_model(name, features)
        output = model(x)
        print(f"{name:20s}: Input {x.shape} -> Output {output.shape}")
