"""
Feature Engineering for Commodity Price Prediction
Creates technical indicators, lag features, and transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Creates features for commodity price prediction.
    Based on research showing that feature engineering significantly
    improves LSTM and other deep learning model performance.
    """
    
    def __init__(self, target_column=None):
        self.target_column = target_column
        self.scalers = {}
        self.feature_names = []
        
    def create_lag_features(self, df, lags=[1, 2, 3, 5, 7, 14, 21, 30, 60, 90]):
        """Create lagged price features."""
        result = df.copy()
        
        for col in df.columns:
            for lag in lags:
                result[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return result
    
    def create_returns(self, df, periods=[1, 5, 10, 21, 63]):
        """Create return features over different periods."""
        result = df.copy()
        
        for col in df.columns:
            for period in periods:
                result[f'{col}_return_{period}d'] = df[col].pct_change(period)
        
        return result
    
    def create_moving_averages(self, df, windows=[5, 10, 20, 50, 100, 200]):
        """Create simple and exponential moving averages."""
        result = df.copy()
        
        for col in df.columns:
            for window in windows:
                # Simple MA
                result[f'{col}_SMA_{window}'] = df[col].rolling(window=window).mean()
                # Exponential MA
                result[f'{col}_EMA_{window}'] = df[col].ewm(span=window, adjust=False).mean()
                # Price relative to MA
                result[f'{col}_price_to_SMA_{window}'] = df[col] / result[f'{col}_SMA_{window}']
        
        return result
    
    def create_volatility_features(self, df, windows=[5, 10, 21, 63]):
        """Create volatility measures."""
        result = df.copy()
        returns = df.pct_change()
        
        for col in df.columns:
            for window in windows:
                # Rolling standard deviation of returns
                result[f'{col}_vol_{window}d'] = returns[col].rolling(window=window).std()
                # Rolling range
                result[f'{col}_range_{window}d'] = (
                    df[col].rolling(window=window).max() - 
                    df[col].rolling(window=window).min()
                ) / df[col].rolling(window=window).mean()
        
        return result
    
    def create_momentum_features(self, df, periods=[5, 10, 21, 63]):
        """Create momentum indicators."""
        result = df.copy()
        
        for col in df.columns:
            for period in periods:
                # Rate of Change
                result[f'{col}_ROC_{period}'] = (
                    (df[col] - df[col].shift(period)) / df[col].shift(period)
                )
                # Momentum
                result[f'{col}_momentum_{period}'] = df[col] - df[col].shift(period)
        
        return result
    
    def create_rsi(self, df, periods=[14, 21]):
        """Calculate Relative Strength Index."""
        result = df.copy()
        
        for col in df.columns:
            for period in periods:
                delta = df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                result[f'{col}_RSI_{period}'] = 100 - (100 / (1 + rs))
        
        return result
    
    def create_bollinger_bands(self, df, window=20, num_std=2):
        """Create Bollinger Band features."""
        result = df.copy()
        
        for col in df.columns:
            sma = df[col].rolling(window=window).mean()
            std = df[col].rolling(window=window).std()
            
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            
            # Position within bands (0 = lower band, 1 = upper band)
            result[f'{col}_BB_position'] = (df[col] - lower) / (upper - lower)
            # Bandwidth
            result[f'{col}_BB_width'] = (upper - lower) / sma
        
        return result
    
    def create_macd(self, df, fast=12, slow=26, signal=9):
        """Create MACD indicator."""
        result = df.copy()
        
        for col in df.columns:
            ema_fast = df[col].ewm(span=fast, adjust=False).mean()
            ema_slow = df[col].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            
            result[f'{col}_MACD'] = macd
            result[f'{col}_MACD_signal'] = macd_signal
            result[f'{col}_MACD_hist'] = macd - macd_signal
        
        return result
    
    def create_cross_commodity_features(self, df):
        """Create features based on relationships between commodities."""
        result = df.copy()
        
        # Correlation with other commodities (rolling)
        returns = df.pct_change()
        
        for col in df.columns:
            # Average correlation with other commodities
            other_cols = [c for c in df.columns if c != col]
            if other_cols:
                corrs = returns[col].rolling(window=21).corr(returns[other_cols].mean(axis=1))
                result[f'{col}_avg_corr_21d'] = corrs
        
        # Spread features for related commodities
        if 'Crude Oil WTI' in df.columns and 'Brent Crude' in df.columns:
            result['WTI_Brent_spread'] = df['Crude Oil WTI'] - df['Brent Crude']
            result['WTI_Brent_ratio'] = df['Crude Oil WTI'] / df['Brent Crude']
        
        if 'Gold' in df.columns and 'Silver' in df.columns:
            result['Gold_Silver_ratio'] = df['Gold'] / df['Silver']
        
        if 'Soybeans' in df.columns and 'Corn' in df.columns:
            result['Soybean_Corn_ratio'] = df['Soybeans'] / df['Corn']
        
        return result
    
    def create_calendar_features(self, df):
        """Create time-based features."""
        result = df.copy()
        
        result['day_of_week'] = df.index.dayofweek
        result['day_of_month'] = df.index.day
        result['month'] = df.index.month
        result['quarter'] = df.index.quarter
        result['year'] = df.index.year
        result['week_of_year'] = df.index.isocalendar().week.astype(int)
        
        # Cyclical encoding
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        return result
    
    def create_target(self, df, target_col, horizon=1, target_type='price'):
        """
        Create target variable for prediction.
        
        Args:
            target_col: Column to predict
            horizon: Days ahead to predict
            target_type: 'price', 'return', or 'direction'
        """
        if target_type == 'price':
            target = df[target_col].shift(-horizon)
        elif target_type == 'return':
            target = df[target_col].pct_change(horizon).shift(-horizon)
        elif target_type == 'direction':
            target = (df[target_col].shift(-horizon) > df[target_col]).astype(int)
        else:
            target = df[target_col].shift(-horizon)
        
        return target
    
    def build_features(self, df, 
                       include_lags=True,
                       include_returns=True,
                       include_ma=True,
                       include_volatility=True,
                       include_momentum=True,
                       include_rsi=True,
                       include_bb=True,
                       include_macd=True,
                       include_cross=True,
                       include_calendar=True,
                       verbose=True):
        """
        Build complete feature set.
        """
        result = df.copy()
        
        if verbose:
            print("Building features...")
        
        if include_lags:
            if verbose: print("  - Lag features")
            result = self.create_lag_features(result, lags=[1, 2, 3, 5, 7, 14, 21])
        
        if include_returns:
            if verbose: print("  - Return features")
            result = self.create_returns(df, periods=[1, 5, 10, 21])
        
        if include_ma:
            if verbose: print("  - Moving averages")
            result = self.create_moving_averages(df, windows=[5, 10, 20, 50])
        
        if include_volatility:
            if verbose: print("  - Volatility features")
            vol_features = self.create_volatility_features(df, windows=[5, 10, 21])
            for col in vol_features.columns:
                if col not in result.columns:
                    result[col] = vol_features[col]
        
        if include_momentum:
            if verbose: print("  - Momentum features")
            mom_features = self.create_momentum_features(df, periods=[5, 10, 21])
            for col in mom_features.columns:
                if col not in result.columns:
                    result[col] = mom_features[col]
        
        if include_rsi:
            if verbose: print("  - RSI")
            rsi_features = self.create_rsi(df, periods=[14])
            for col in rsi_features.columns:
                if col not in result.columns:
                    result[col] = rsi_features[col]
        
        if include_bb:
            if verbose: print("  - Bollinger Bands")
            bb_features = self.create_bollinger_bands(df)
            for col in bb_features.columns:
                if col not in result.columns:
                    result[col] = bb_features[col]
        
        if include_macd:
            if verbose: print("  - MACD")
            macd_features = self.create_macd(df)
            for col in macd_features.columns:
                if col not in result.columns:
                    result[col] = macd_features[col]
        
        if include_cross:
            if verbose: print("  - Cross-commodity features")
            cross_features = self.create_cross_commodity_features(df)
            for col in cross_features.columns:
                if col not in result.columns:
                    result[col] = cross_features[col]
        
        if include_calendar:
            if verbose: print("  - Calendar features")
            cal_features = self.create_calendar_features(df)
            for col in cal_features.columns:
                if col not in result.columns:
                    result[col] = cal_features[col]
        
        # Store feature names (excluding original price columns)
        self.feature_names = [c for c in result.columns if c not in df.columns]
        
        if verbose:
            print(f"\nTotal features created: {len(self.feature_names)}")
        
        return result
    
    def prepare_sequences(self, df, target_col, sequence_length=60, horizon=1,
                         train_ratio=0.7, val_ratio=0.15, scale=True):
        """
        Prepare data sequences for LSTM/Transformer models.
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Create target
        target = self.create_target(df, target_col, horizon=horizon, target_type='price')
        
        # Get features (drop original price columns for features, keep one for reference)
        feature_cols = [c for c in df.columns]
        features = df[feature_cols].copy()
        
        # Drop rows with NaN
        combined = pd.concat([features, target.rename('target')], axis=1)
        combined = combined.dropna()
        
        features = combined.drop('target', axis=1)
        target = combined['target']
        
        # Scale features
        if scale:
            self.scalers['features'] = MinMaxScaler()
            self.scalers['target'] = MinMaxScaler()
            
            features_scaled = self.scalers['features'].fit_transform(features)
            target_scaled = self.scalers['target'].fit_transform(target.values.reshape(-1, 1))
        else:
            features_scaled = features.values
            target_scaled = target.values.reshape(-1, 1)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/val/test
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Store dates for test set
        self.test_dates = combined.index[sequence_length + val_end:]
        
        print(f"\nSequence shape: {X.shape}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def inverse_transform_target(self, y_scaled):
        """Convert scaled predictions back to original scale."""
        if 'target' in self.scalers:
            return self.scalers['target'].inverse_transform(y_scaled.reshape(-1, 1))
        return y_scaled


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import CommodityDataLoader
    
    # Load data
    loader = CommodityDataLoader(start_date='2018-01-01')
    data = loader.load_all_data(source='yahoo')
    
    if data is not None:
        # Build features
        fe = FeatureEngineer()
        features = fe.build_features(data)
        
        print("\nFeature DataFrame shape:", features.shape)
        print("\nSample features:", features.columns[:20].tolist())
