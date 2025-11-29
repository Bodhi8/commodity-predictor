"""
Data Loader for US Commodity Prices
Sources: Yahoo Finance (futures), FRED, World Bank Pink Sheet
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import io
import warnings
warnings.filterwarnings('ignore')


# US Commodity Futures Tickers (Yahoo Finance)
US_COMMODITY_TICKERS = {
    # Energy
    'CL=F': 'Crude Oil WTI',
    'BZ=F': 'Brent Crude',
    'NG=F': 'Natural Gas',
    'RB=F': 'Gasoline RBOB',
    'HO=F': 'Heating Oil',
    
    # Precious Metals
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'PL=F': 'Platinum',
    'PA=F': 'Palladium',
    
    # Base Metals
    'HG=F': 'Copper',
    
    # Agriculture
    'ZC=F': 'Corn',
    'ZW=F': 'Wheat',
    'ZS=F': 'Soybeans',
    'ZM=F': 'Soybean Meal',
    'ZL=F': 'Soybean Oil',
    'KC=F': 'Coffee',
    'SB=F': 'Sugar',
    'CC=F': 'Cocoa',
    'CT=F': 'Cotton',
    'OJ=F': 'Orange Juice',
    
    # Livestock
    'LE=F': 'Live Cattle',
    'HE=F': 'Lean Hogs',
    'GF=F': 'Feeder Cattle',
}

# FRED Series IDs for commodities
FRED_SERIES = {
    'DCOILWTICO': 'WTI Crude Oil',
    'DCOILBRENTEU': 'Brent Crude Oil',
    'DHHNGSP': 'Henry Hub Natural Gas',
    'GOLDAMGBD228NLBM': 'Gold London Fix',
    'SLVPRUSD': 'Silver Price',
    'PCOPPUSDM': 'Copper Price',
    'PMAIZMTUSDM': 'Corn Price',
    'PWHEAMTUSDM': 'Wheat Price',
    'PSOYBUSDM': 'Soybeans Price',
}


class CommodityDataLoader:
    """Load US commodity price data from multiple sources."""
    
    def __init__(self, start_date='2010-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        
    def load_yahoo_futures(self, tickers=None, verbose=True):
        """
        Load commodity futures data from Yahoo Finance.
        Returns daily close prices.
        """
        if tickers is None:
            tickers = list(US_COMMODITY_TICKERS.keys())
        
        if verbose:
            print(f"Loading {len(tickers)} commodities from Yahoo Finance...")
        
        all_data = {}
        
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                if len(data) > 0:
                    name = US_COMMODITY_TICKERS.get(ticker, ticker)
                    all_data[name] = data['Close']
                    if verbose:
                        print(f"  ✓ {name}: {len(data)} days")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {ticker}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        return None
    
    def load_world_bank_pink_sheet(self, verbose=True):
        """
        Load World Bank Pink Sheet monthly commodity data.
        Great for historical data going back to 1960.
        """
        url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
        
        if verbose:
            print("Loading World Bank Pink Sheet data...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read the Monthly Prices sheet
            df = pd.read_excel(
                io.BytesIO(response.content),
                sheet_name='Monthly Prices',
                header=4  # Data starts after header rows
            )
            
            # Clean up the dataframe
            df = df.rename(columns={df.columns[0]: 'Date'})
            df = df.dropna(subset=['Date'])
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.set_index('Date')
            
            # Filter to date range
            df = df.loc[self.start_date:self.end_date]
            
            # Select key US-relevant commodities
            us_commodities = [
                'CRUDE_WTI', 'CRUDE_BRENT', 'NGAS_US', 
                'GOLD', 'SILVER', 'COPPER',
                'MAIZE', 'WHEAT_US_HRW', 'SOYBEAN',
                'SUGAR_WLD', 'COFFEE_ARABIC', 'COTTON_A_INDX'
            ]
            
            available = [c for c in us_commodities if c in df.columns]
            df = df[available]
            
            if verbose:
                print(f"  ✓ Loaded {len(df.columns)} commodities, {len(df)} months")
            
            return df
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Error loading World Bank data: {e}")
            return None
    
    def create_composite_index(self, df, weights=None, method='equal'):
        """
        Create a composite commodity price index.
        
        Methods:
        - 'equal': Equal weighted
        - 'custom': Use provided weights dict
        - 'inverse_vol': Inverse volatility weighted
        """
        # Normalize prices to start at 100
        normalized = df / df.iloc[0] * 100
        
        if method == 'equal':
            index = normalized.mean(axis=1)
            
        elif method == 'inverse_vol':
            # Weight by inverse volatility (less volatile = higher weight)
            vol = df.pct_change().std()
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            index = (normalized * weights).sum(axis=1)
            
        elif method == 'custom' and weights is not None:
            # Normalize weights
            w = pd.Series(weights)
            w = w / w.sum()
            available_weights = w[w.index.isin(df.columns)]
            available_weights = available_weights / available_weights.sum()
            index = (normalized[available_weights.index] * available_weights).sum(axis=1)
            
        else:
            index = normalized.mean(axis=1)
        
        return index
    
    def load_all_data(self, source='yahoo', verbose=True):
        """
        Load data from specified source and create features.
        
        Args:
            source: 'yahoo', 'worldbank', or 'both'
        """
        if source in ['yahoo', 'both']:
            yahoo_data = self.load_yahoo_futures(verbose=verbose)
            
        if source in ['worldbank', 'both']:
            wb_data = self.load_world_bank_pink_sheet(verbose=verbose)
        
        if source == 'yahoo':
            self.data = yahoo_data
        elif source == 'worldbank':
            self.data = wb_data
        else:
            # Combine both sources (Yahoo for daily, WB for monthly backup)
            self.data = yahoo_data
        
        return self.data
    
    def get_data_summary(self):
        """Print summary statistics of loaded data."""
        if self.data is None:
            print("No data loaded. Call load_all_data() first.")
            return
        
        print("\n" + "="*60)
        print("COMMODITY DATA SUMMARY")
        print("="*60)
        print(f"Date Range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total Days: {len(self.data)}")
        print(f"Commodities: {len(self.data.columns)}")
        print("\nCoverage by Commodity:")
        print("-"*40)
        
        for col in self.data.columns:
            non_null = self.data[col].notna().sum()
            pct = non_null / len(self.data) * 100
            print(f"  {col:20s}: {non_null:5d} days ({pct:.1f}%)")
        
        print("\nPrice Statistics:")
        print("-"*40)
        print(self.data.describe().round(2))


def download_sample_data():
    """Quick function to download sample data for testing."""
    loader = CommodityDataLoader(start_date='2015-01-01')
    data = loader.load_all_data(source='yahoo')
    loader.get_data_summary()
    return loader, data


if __name__ == "__main__":
    loader, data = download_sample_data()
    
    # Create composite index
    if data is not None:
        index = loader.create_composite_index(data, method='equal')
        print("\n\nComposite Index (last 10 values):")
        print(index.tail(10))
