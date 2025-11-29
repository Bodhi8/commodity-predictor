"""
Sample Data Generator for Testing
Creates realistic synthetic commodity price data when real data is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_gbm_prices(start_price, days, mu=0.0001, sigma=0.02, seed=None):
    """
    Generate prices using Geometric Brownian Motion.
    
    Args:
        start_price: Initial price
        days: Number of days to simulate
        mu: Drift (daily expected return)
        sigma: Volatility (daily standard deviation)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1  # daily
    prices = [start_price]
    
    for _ in range(days - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        dS = prices[-1] * (mu * dt + sigma * dW)
        prices.append(max(prices[-1] + dS, 0.01))  # Ensure positive price
    
    return np.array(prices)


def generate_mean_reverting_prices(start_price, days, mean_price, theta=0.1, 
                                   sigma=0.02, seed=None):
    """
    Generate prices using Ornstein-Uhlenbeck (mean-reverting) process.
    Good for commodities that tend to revert to a long-term average.
    """
    if seed is not None:
        np.random.seed(seed)
    
    prices = [start_price]
    
    for _ in range(days - 1):
        dW = np.random.normal(0, 1)
        new_price = prices[-1] + theta * (mean_price - prices[-1]) + sigma * prices[-1] * dW
        prices.append(max(new_price, 0.01))
    
    return np.array(prices)


def add_seasonality(prices, period=252, amplitude=0.05):
    """Add seasonal component to prices."""
    n = len(prices)
    seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / period)
    return prices * (1 + seasonal)


def add_trend(prices, daily_trend=0.0001):
    """Add linear trend to prices."""
    n = len(prices)
    trend = np.exp(daily_trend * np.arange(n))
    return prices * trend


def generate_correlated_prices(base_prices, correlation=0.7, sigma_ratio=1.0, seed=None):
    """
    Generate prices correlated with base prices.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(base_prices)
    base_returns = np.diff(np.log(base_prices))
    
    # Generate correlated returns
    noise = np.random.normal(0, 1, len(base_returns))
    correlated_returns = correlation * base_returns + np.sqrt(1 - correlation**2) * noise * sigma_ratio * np.std(base_returns)
    
    # Convert back to prices
    prices = [base_prices[0]]
    for r in correlated_returns:
        prices.append(prices[-1] * np.exp(r))
    
    return np.array(prices)


def generate_sample_commodity_data(start_date='2015-01-01', end_date=None, seed=42):
    """
    Generate a complete sample dataset mimicking real commodity prices.
    
    Returns:
        DataFrame with synthetic commodity prices
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create business day date range
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    
    print(f"Generating {n_days} days of synthetic commodity data...")
    
    np.random.seed(seed)
    
    # Generate base commodities with realistic starting prices and volatilities
    commodities = {}
    
    # Crude Oil WTI - mean reverting around $70
    oil = generate_mean_reverting_prices(70, n_days, mean_price=70, 
                                         theta=0.02, sigma=0.025, seed=seed)
    oil = add_seasonality(oil, period=252, amplitude=0.03)
    commodities['Crude Oil WTI'] = oil
    
    # Brent Crude - correlated with WTI
    commodities['Brent Crude'] = generate_correlated_prices(oil, correlation=0.95, 
                                                             sigma_ratio=0.9, seed=seed+1) * 1.05
    
    # Natural Gas - higher volatility, seasonal
    ng = generate_mean_reverting_prices(3.0, n_days, mean_price=3.5, 
                                        theta=0.03, sigma=0.04, seed=seed+2)
    ng = add_seasonality(ng, period=252, amplitude=0.15)  # Strong seasonality
    commodities['Natural Gas'] = ng
    
    # Gold - trending with lower volatility
    gold = generate_gbm_prices(1500, n_days, mu=0.0002, sigma=0.01, seed=seed+3)
    gold = add_trend(gold, daily_trend=0.00015)
    commodities['Gold'] = gold
    
    # Silver - correlated with gold, higher volatility
    commodities['Silver'] = generate_correlated_prices(gold, correlation=0.8, 
                                                        sigma_ratio=1.5, seed=seed+4) / 75
    
    # Copper - industrial metal
    copper = generate_mean_reverting_prices(3.5, n_days, mean_price=4.0, 
                                            theta=0.015, sigma=0.02, seed=seed+5)
    commodities['Copper'] = copper
    
    # Corn - agricultural, seasonal
    corn = generate_mean_reverting_prices(400, n_days, mean_price=450, 
                                          theta=0.02, sigma=0.025, seed=seed+6)
    corn = add_seasonality(corn, period=252, amplitude=0.08)
    commodities['Corn'] = corn
    
    # Wheat - correlated with corn
    commodities['Wheat'] = generate_correlated_prices(corn, correlation=0.7, 
                                                       sigma_ratio=1.1, seed=seed+7) * 1.4
    
    # Soybeans - agricultural
    soybeans = generate_mean_reverting_prices(1000, n_days, mean_price=1100, 
                                              theta=0.02, sigma=0.02, seed=seed+8)
    soybeans = add_seasonality(soybeans, period=252, amplitude=0.06)
    commodities['Soybeans'] = soybeans
    
    # Coffee - high volatility
    coffee = generate_mean_reverting_prices(150, n_days, mean_price=180, 
                                            theta=0.01, sigma=0.035, seed=seed+9)
    commodities['Coffee'] = coffee
    
    # Sugar
    sugar = generate_mean_reverting_prices(18, n_days, mean_price=20, 
                                           theta=0.02, sigma=0.025, seed=seed+10)
    commodities['Sugar'] = sugar
    
    # Cotton
    cotton = generate_mean_reverting_prices(80, n_days, mean_price=85, 
                                            theta=0.015, sigma=0.02, seed=seed+11)
    commodities['Cotton'] = cotton
    
    # Create DataFrame
    df = pd.DataFrame(commodities, index=dates)
    
    # Add some realistic noise and occasional jumps
    for col in df.columns:
        # Random small gaps (simulate missing data)
        mask = np.random.random(len(df)) > 0.005  # 0.5% missing
        df.loc[~mask, col] = np.nan
        
        # Forward fill
        df[col] = df[col].ffill()
    
    print(f"Generated data for {len(df.columns)} commodities")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def load_or_generate_data(start_date='2015-01-01', try_real=True, verbose=True):
    """
    Try to load real data, fall back to synthetic if unavailable.
    """
    if try_real:
        try:
            from data_loader import CommodityDataLoader
            loader = CommodityDataLoader(start_date=start_date)
            data = loader.load_all_data(source='yahoo', verbose=verbose)
            if data is not None and len(data) > 100:
                if verbose:
                    print("Successfully loaded real commodity data!")
                return data, 'real'
        except Exception as e:
            if verbose:
                print(f"Could not load real data: {e}")
    
    if verbose:
        print("Using synthetic data for demonstration...")
    
    return generate_sample_commodity_data(start_date=start_date), 'synthetic'


if __name__ == "__main__":
    # Generate sample data
    data = generate_sample_commodity_data(start_date='2018-01-01')
    
    print("\nData Summary:")
    print(data.describe())
    
    print("\nFirst few rows:")
    print(data.head())
    
    print("\nLast few rows:")
    print(data.tail())
    
    # Save to CSV for inspection
    data.to_csv('sample_commodity_data.csv')
    print("\nSaved to sample_commodity_data.csv")
