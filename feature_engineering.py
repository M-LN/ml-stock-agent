"""
Feature Engineering Module
Beregner tekniske indikatorer og lag features til ML modeller
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Beregner Relative Strength Index (RSI).
    RSI måler momentum og overbought/oversold conditions.
    
    Args:
        prices: Close prices
        period: Lookback period (default 14)
    
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Beregner MACD (Moving Average Convergence Divergence).
    MACD viser trend strength og retning.
    
    Args:
        prices: Close prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        DataFrame med MACD, Signal, og Histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Hist': histogram
    })


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Beregner Bollinger Bands.
    Bollinger Bands viser volatilitet og potentielle breakouts.
    
    Args:
        prices: Close prices
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
    
    Returns:
        DataFrame med Upper, Middle, Lower bands og Width
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    width = (upper - lower) / middle  # Normalized width
    
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': middle,
        'BB_Lower': lower,
        'BB_Width': width
    })


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Beregner Average True Range (ATR).
    ATR måler markedsvolatilitet.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)
    
    Returns:
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Beregner On-Balance Volume (OBV).
    OBV kombinerer pris og volume for at vise buying/selling pressure.
    
    Args:
        close: Close prices
        volume: Volume
    
    Returns:
        OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """
    Beregner Stochastic Oscillator.
    Stochastic viser momentum og overbought/oversold levels.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)
        smooth_k: %K smoothing (default 3)
        smooth_d: %D smoothing (default 3)
    
    Returns:
        DataFrame med %K og %D
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k = k.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    
    return pd.DataFrame({
        'Stoch_K': k,
        'Stoch_D': d
    })


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Beregner Average Directional Index (ADX).
    ADX måler trend strength (ikke retning).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)
    
    Returns:
        ADX values (0-100)
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx


def create_features(data: pd.DataFrame, include_volume: bool = True) -> pd.DataFrame:
    """
    Skaber et komplet sæt af features til ML modeller.
    
    Args:
        data: DataFrame med OHLCV data
        include_volume: Om volume-baserede features skal inkluderes
    
    Returns:
        DataFrame med alle features
    """
    df = data.copy()
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")
    
    # ==================== PRICE-BASED FEATURES ====================
    
    # Returns (different horizons)
    for period in [1, 5, 10, 20]:
        df[f'Return_{period}d'] = df['Close'].pct_change(period)
    
    # Moving Averages (only create if we have enough data)
    for period in [5, 10, 20, 50]:
        if len(df) >= period:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # 200-day MA only if we have enough data
    if len(df) >= 200:
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MA Crossovers (normalized) - only if we have the MAs
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        df['SMA_20_50_ratio'] = df['SMA_20'] / df['SMA_50']
    
    # Price position relative to MAs
    if 'SMA_20' in df.columns:
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
    if 'SMA_50' in df.columns:
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
    
    # ==================== MOMENTUM INDICATORS ====================
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['RSI_Fast'] = calculate_rsi(df['Close'], 7)  # Faster RSI
    
    # MACD
    macd_df = calculate_macd(df['Close'])
    df['MACD'] = macd_df['MACD']
    df['MACD_Signal'] = macd_df['MACD_Signal']
    df['MACD_Hist'] = macd_df['MACD_Hist']
    
    # Stochastic
    stoch_df = calculate_stochastic(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch_df['Stoch_K']
    df['Stoch_D'] = stoch_df['Stoch_D']
    
    # ==================== VOLATILITY INDICATORS ====================
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_df['BB_Upper']
    df['BB_Middle'] = bb_df['BB_Middle']
    df['BB_Lower'] = bb_df['BB_Lower']
    df['BB_Width'] = bb_df['BB_Width']
    
    # Price position in BB
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ATR
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['ATR_Pct'] = df['ATR'] / df['Close']  # Normalized ATR
    
    # Historical volatility
    df['Volatility_20'] = df['Return_1d'].rolling(window=20).std()
    df['Volatility_50'] = df['Return_1d'].rolling(window=50).std()
    
    # ==================== TREND INDICATORS ====================
    
    # ADX (trend strength)
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
    
    # Price channels
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    df['Channel_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    # ==================== VOLUME FEATURES ====================
    
    if include_volume and 'Volume' in df.columns:
        # Volume moving averages
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
        
        # Volume ratios
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Trend'] = df['Volume_SMA_20'] / df['Volume_SMA_50']
        
        # OBV
        df['OBV'] = calculate_obv(df['Close'], df['Volume'])
        df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
        
        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()
    
    # ==================== CANDLESTICK PATTERNS ====================
    
    # Body size (normalized)
    df['Body_Size'] = np.abs(df['Close'] - df['Open']) / df['Open']
    
    # Upper/Lower shadows
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
    
    # Daily range
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Low']
    
    # ==================== STATISTICAL FEATURES ====================
    
    # Rolling statistics
    for period in [5, 10, 20]:
        df[f'Rolling_Mean_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'Rolling_Std_{period}'] = df['Close'].rolling(window=period).std()
        df[f'Rolling_Max_{period}'] = df['Close'].rolling(window=period).max()
        df[f'Rolling_Min_{period}'] = df['Close'].rolling(window=period).min()
        
        # Z-score (distance from mean in std devs)
        df[f'Z_Score_{period}'] = (df['Close'] - df[f'Rolling_Mean_{period}']) / df[f'Rolling_Std_{period}']
    
    # ==================== TIME FEATURES ====================
    
    if df.index.dtype == 'datetime64[ns]' or isinstance(df.index, pd.DatetimeIndex):
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
    
    # ==================== CLEANUP ====================
    
    # Remove rows with NaN (from indicator calculation)
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_ohlcv: bool = True) -> list:
    """
    Returnerer liste af feature kolonner (ekskluderer OHLCV hvis ønsket).
    
    Args:
        df: DataFrame med features
        exclude_ohlcv: Om OHLCV kolonner skal ekskluderes
    
    Returns:
        Liste af feature kolonnenavne
    """
    if exclude_ohlcv:
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividends', 'Stock Splits']
        return [col for col in df.columns if col not in exclude_cols]
    else:
        return list(df.columns)


def normalize_features(df: pd.DataFrame, feature_cols: list, method: str = 'minmax') -> tuple:
    """
    Normaliserer features til [0, 1] range eller standard scale.
    
    Args:
        df: DataFrame med features
        feature_cols: Liste af kolonner at normalisere
        method: 'minmax' eller 'standard'
    
    Returns:
        (normalized_df, scaler_params) for at reverse senere
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    df_norm = df.copy()
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler params for later use
    scaler_params = {
        'method': method,
        'feature_cols': feature_cols,
        'scaler': scaler
    }
    
    return df_norm, scaler_params


if __name__ == "__main__":
    # Test feature engineering
    import yfinance as yf
    
    print("Testing Feature Engineering...")
    
    # Download test data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Create features
    features_df = create_features(data)
    
    print(f"\nFeatures data shape: {features_df.shape}")
    print(f"\nNumber of features created: {len(get_feature_columns(features_df))}")
    
    print("\nSample of features:")
    feature_cols = get_feature_columns(features_df)[:10]
    print(features_df[feature_cols].tail())
    
    print("\n✅ Feature engineering test successful!")
