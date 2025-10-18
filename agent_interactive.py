import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import json
import os
import pickle
import joblib
from datetime import datetime
from colorama import init, Fore, Back, Style
from math import sqrt
import xgboost as xgb
from prophet import Prophet

# Initialiser colorama til Windows support
init(autoreset=True)

# Model storage directory
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Training logs directory
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Manuel beregning af indikatorer (uden pandas-ta)
def calculate_rsi(data, periods=14):
    """Beregner RSI (Relative Strength Index)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, periods):
    """Beregner SMA (Simple Moving Average)"""
    return data['Close'].rolling(window=periods).mean()

def calculate_ema(data, periods):
    """Beregner EMA (Exponential Moving Average)"""
    return data['Close'].ewm(span=periods, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Beregner MACD (Moving Average Convergence Divergence)
    
    Returns:
        macd_line: MACD linje (EMA12 - EMA26)
        signal_line: Signal linje (EMA9 af MACD)
        histogram: MACD histogram (MACD - Signal)
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, periods=20, std_dev=2):
    """
    Beregner Bollinger Bands
    
    Returns:
        upper_band: Upper band (SMA + 2*std)
        middle_band: Middle band (SMA)
        lower_band: Lower band (SMA - 2*std)
    """
    middle_band = data['Close'].rolling(window=periods).mean()
    std = data['Close'].rolling(window=periods).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band

def calculate_atr(data, periods=14):
    """
    Beregner ATR (Average True Range) - mål for volatilitet
    
    Returns:
        atr: Average True Range
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=periods).mean()
    return atr

def calculate_stochastic(data, k_periods=14, d_periods=3):
    """
    Beregner Stochastic Oscillator (%K and %D)
    
    Returns:
        k: %K line (fast)
        d: %D line (slow, signal line)
    """
    low_min = data['Low'].rolling(window=k_periods).min()
    high_max = data['High'].rolling(window=k_periods).max()
    
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_periods).mean()
    
    return k, d

def calculate_cci(data, periods=20):
    """
    Beregner CCI (Commodity Channel Index)
    
    Returns:
        cci: Commodity Channel Index
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma = typical_price.rolling(window=periods).mean()
    mean_deviation = typical_price.rolling(window=periods).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_williams_r(data, periods=14):
    """
    Beregner Williams %R - momentum indikator
    
    Returns:
        williams_r: Williams %R values
    """
    highest_high = data['High'].rolling(window=periods).max()
    lowest_low = data['Low'].rolling(window=periods).min()
    
    williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
    return williams_r

def calculate_obv(data):
    """
    Beregner OBV (On-Balance Volume)
    
    Returns:
        obv: On-Balance Volume
    """
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_adx(data, periods=14):
    """
    Beregner ADX (Average Directional Index) - trend strength
    
    Returns:
        adx: Average Directional Index
    """
    # Calculate +DM and -DM
    high_diff = data['High'].diff()
    low_diff = -data['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate ATR
    atr = calculate_atr(data, periods)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=periods).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=periods).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=periods).mean()
    
    return adx, plus_di, minus_di

def calculate_vwap(data):
    """
    Beregner VWAP (Volume Weighted Average Price)
    
    Returns:
        vwap: Volume Weighted Average Price
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_fibonacci_levels(data):
    """
    Beregner Fibonacci retracement levels
    
    Returns:
        dict med fibonacci levels
    """
    # Find highest and lowest prices in recent period
    period_high = data['High'].tail(100).max()
    period_low = data['Low'].tail(100).min()
    diff = period_high - period_low
    
    levels = {
        '0.0%': period_high,
        '23.6%': period_high - 0.236 * diff,
        '38.2%': period_high - 0.382 * diff,
        '50.0%': period_high - 0.500 * diff,
        '61.8%': period_high - 0.618 * diff,
        '78.6%': period_high - 0.786 * diff,
        '100.0%': period_low
    }
    
    return levels

def calculate_pivot_points(data):
    """
    Beregner Pivot Points (support and resistance levels)
    
    Returns:
        dict med pivot levels
    """
    # Use yesterday's data
    high = data['High'].iloc[-2] if len(data) >= 2 else data['High'].iloc[-1]
    low = data['Low'].iloc[-2] if len(data) >= 2 else data['Low'].iloc[-1]
    close = data['Close'].iloc[-2] if len(data) >= 2 else data['Close'].iloc[-1]
    
    pivot = (high + low + close) / 3
    
    levels = {
        'R3': high + 2 * (pivot - low),
        'R2': pivot + (high - low),
        'R1': 2 * pivot - low,
        'Pivot': pivot,
        'S1': 2 * pivot - high,
        'S2': pivot - (high - low),
        'S3': low - 2 * (high - pivot)
    }
    
    return levels

def get_vix_data():
    """
    Henter VIX (CBOE Volatility Index) data - Fear Index
    
    Returns:
        dict: {
            'current': float - Nuværende VIX værdi,
            'level': str - 'Low', 'Normal', 'Elevated', 'High',
            'interpretation': str - Fortolkning af niveauet,
            'raw_data': DataFrame - Sidste 30 dages data
        }
    """
    try:
        vix = yf.download('^VIX', period='1mo', progress=False)
        if vix.empty:
            return None
        
        # Handle MultiIndex columns
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        current_vix = float(vix['Close'].iloc[-1])
        
        # VIX levels interpretation
        if current_vix < 15:
            level = "Low"
            interpretation = "📉 Lav volatilitet - Markedet er roligt og selvsikkert"
            sentiment = "bullish"
        elif current_vix < 20:
            level = "Normal"
            interpretation = "➡️ Normal volatilitet - Markedet er stabilt"
            sentiment = "neutral"
        elif current_vix < 30:
            level = "Elevated"
            interpretation = "📈 Forhøjet volatilitet - Øget markedsusikkerhed"
            sentiment = "cautious"
        else:
            level = "High"
            interpretation = "🚨 Høj volatilitet - Markedet er nervøst/panikagtigt"
            sentiment = "bearish"
        
        return {
            'current': current_vix,
            'level': level,
            'interpretation': interpretation,
            'sentiment': sentiment,
            'raw_data': vix
        }
    except Exception as e:
        print(f"{Fore.RED}⚠️ Kunne ikke hente VIX data: {str(e)}")
        return None

def get_fear_greed_index():
    """
    Henter Enhanced Fear & Greed Index baseret på 7 komponenter (aligned med CNN methodology)
    Kalibreret til at matche CNN Fear & Greed Index så tæt som muligt
    
    Komponenter:
    1. Market Momentum (S&P 500 vs 125-day average) - 20%
    2. Stock Price Strength (distance from 52-week high) - 20%
    3. Stock Price Breadth (advance/decline) - 10%
    4. Put/Call Ratio (approximated using VIX) - 10%
    5. Market Volatility (VIX levels) - 25% (MOST IMPORTANT)
    6. Safe Haven Demand (Gold vs S&P 500) - 10%
    7. Junk Bond Demand (Treasury yields) - 5%
    
    Returns:
        dict: {
            'value': int (0-100),
            'level': str - 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed',
            'interpretation': str,
            'sentiment': str,
            'components': dict - All 7 component scores
        }
    """
    try:
        # Fetch all required data
        sp500_data = yf.download('^GSPC', period='1y', progress=False)
        vix_data = yf.download('^VIX', period='5d', progress=False)
        gold_data = yf.download('GC=F', period='1mo', progress=False)
        treasury_data = yf.download('^TNX', period='1mo', progress=False)
        
        if sp500_data.empty or vix_data.empty:
            return None
        
        # Handle MultiIndex columns
        for df in [sp500_data, vix_data, gold_data, treasury_data]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        fear_greed_score = 50  # Neutral baseline
        components = []
        
        # 1. Market Momentum (S&P 500 vs 125-day average) - 20%
        if len(sp500_data) >= 125:
            sp500_125_avg = sp500_data['Close'].tail(125).mean()
            sp500_current = sp500_data['Close'].iloc[-1]
            momentum_pct = ((sp500_current - sp500_125_avg) / sp500_125_avg) * 100
            # More conservative scaling: ±1% = ±10 points
            momentum_score = 50 + min(max(momentum_pct * 1.0, -40), 40)
            components.append(('Momentum', momentum_score))
        
        # 2. Stock Price Strength (distance from 52-week high) - 20%
        if len(sp500_data) >= 252:
            high_52w = sp500_data['High'].tail(252).max()
            current_price = sp500_data['Close'].iloc[-1]
            distance_from_high = ((current_price - high_52w) / high_52w) * 100
            # -10% from high = score 30, at high = score 100
            strength_score = 100 + (distance_from_high * 7)
            strength_score = max(0, min(100, strength_score))
            components.append(('Price Strength', strength_score))
        
        # 3. Stock Price Breadth (advance/decline) - 10%
        if len(sp500_data) >= 10:
            recent_closes = sp500_data['Close'].tail(10)
            advances = sum(recent_closes.diff() > 0)
            breadth_score = 30 + (advances / 9) * 40
            components.append(('Breadth', breadth_score))
        
        # 4. Put/Call Ratio (approximated using VIX) - 10%
        if not vix_data.empty:
            vix_current = vix_data['Close'].iloc[-1]
            # VIX 15 = 50 (neutral), VIX 30 = 0 (fear)
            vix_score = max(0, min(100, 50 + (20 - vix_current) * 2.5))
            components.append(('Put/Call (VIX)', vix_score))
        
        # 5. Market Volatility (VIX levels) - 25% (MOST IMPORTANT)
        if not vix_data.empty:
            vix_avg = 18
            volatility_score = max(0, min(100, 70 - ((vix_current - vix_avg) * 4)))
            components.append(('Volatility', volatility_score))
        
        # 6. Safe Haven Demand (Gold vs S&P 500) - 10%
        if not gold_data.empty and len(gold_data) >= 20 and len(sp500_data) >= 20:
            gold_20d_change = ((gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-20]) / gold_data['Close'].iloc[-20] * 100)
            sp500_20d_change = ((sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[-20]) / sp500_data['Close'].iloc[-20] * 100)
            safe_haven_score = 50 + (sp500_20d_change - gold_20d_change) * 5
            safe_haven_score = max(0, min(100, safe_haven_score))
            components.append(('Safe Haven', safe_haven_score))
        
        # 7. Junk Bond Demand (Treasury yields) - 5%
        if not treasury_data.empty and len(treasury_data) >= 20:
            current_yield = treasury_data['Close'].iloc[-1]
            avg_yield = 4.0
            junk_score = max(0, min(100, 50 + (avg_yield - current_yield) * 10))
            components.append(('Junk Bond', junk_score))
        
        # Calculate weighted average with CNN-aligned weights
        weights = {
            'Momentum': 0.20,
            'Price Strength': 0.20,
            'Breadth': 0.10,
            'Put/Call (VIX)': 0.10,
            'Volatility': 0.25,
            'Safe Haven': 0.10,
            'Junk Bond': 0.05
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for name, score in components:
            weight = weights.get(name, 1.0 / len(components))
            weighted_sum += score * weight
            total_weight += weight
        
        fear_greed_score = weighted_sum / total_weight if total_weight > 0 else 50
        
        # Apply CNN-aligned calibration (compress scale to match CNN distribution)
        fear_greed_score = fear_greed_score * 0.62
        
        # Granular adjustments for different score ranges
        if fear_greed_score >= 60:
            fear_greed_score = 38 + (fear_greed_score - 60) * 0.5
        elif fear_greed_score >= 45:
            fear_greed_score = 32 + (fear_greed_score - 45) * 0.75
        elif fear_greed_score >= 30:
            fear_greed_score = fear_greed_score * 1.0
        else:
            fear_greed_score = fear_greed_score * 1.05
        
        fear_greed_score = max(0, min(100, fear_greed_score))
        fg_value = int(fear_greed_score)
        
        # Determine level and interpretation
        if fg_value <= 20:
            level = "Extreme Fear"
            interpretation = "😱 Ekstrem frygt - Potentiel købs-mulighed (contrarian)"
            sentiment = "extreme_fear"
        elif fg_value <= 40:
            level = "Fear"
            interpretation = "😰 Frygt - Markedet er negativt"
            sentiment = "fear"
        elif fg_value <= 60:
            level = "Neutral"
            interpretation = "😐 Neutral - Markedet er afventende"
            sentiment = "neutral"
        elif fg_value <= 80:
            level = "Greed"
            interpretation = "😊 Grådighed - Markedet er positivt"
            sentiment = "greed"
        else:
            level = "Extreme Greed"
            interpretation = "🤑 Ekstrem grådighed - Potentiel risiko for korrektion"
            sentiment = "extreme_greed"
        
        # Prepare components dict for output
        components_dict = {name: round(score, 1) for name, score in components}
        
        return {
            'value': fg_value,
            'level': level,
            'interpretation': interpretation,
            'sentiment': sentiment,
            'components': components_dict
        }
    except Exception as e:
        print(f"{Fore.RED}⚠️ Kunne ikke beregne Fear & Greed Index: {str(e)}")
        return None

def get_shiller_pe():
    """
    Henter Dynamic Shiller P/E (CAPE - Cyclically Adjusted Price-to-Earnings Ratio)
    Aligned med Market Overview beregning for konsistens
    
    CAPE beregning:
    - Bruger trailing P/E fra S&P 500
    - CAPE er typisk 1.4-1.8x højere end trailing P/E
    - I nuværende marked: trailing P/E ~20-22 → CAPE ~32-36
    
    Returns:
        dict: {
            'cape_ratio': float - Dynamisk beregnet CAPE ratio,
            'level': str - 'Undervalued', 'Fair', 'Overvalued', 'Highly Overvalued',
            'interpretation': str,
            'sentiment': str,
            'sp500_pe': float - Standard P/E ratio for reference,
            'historical_avg': float - Historical average CAPE (~16.8)
        }
    """
    try:
        # Fetch S&P 500 data
        sp500_ticker = yf.Ticker("^GSPC")
        sp500_data = sp500_ticker.history(period="5d")
        sp500_info = sp500_ticker.info
        
        if sp500_data.empty:
            return None
        
        # Handle MultiIndex
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500_data.columns = sp500_data.columns.get_level_values(0)
        
        # Try to get trailing P/E
        trailing_pe = sp500_info.get('trailingPE', None)
        
        if trailing_pe and trailing_pe > 0:
            # CAPE (Shiller P/E) is typically 1.4-1.8x higher than trailing P/E
            # This is because CAPE uses 10-year inflation-adjusted earnings
            # Current market: trailing P/E ~20-22, CAPE ~32-36
            estimated_cape = trailing_pe * 1.60  # Calibrated multiplier for accuracy
        else:
            # Fallback: calculate from price and estimate earnings
            sp500_current = sp500_data['Close'].iloc[-1]
            # Use historical average earnings yield (~4.5% in current high-P/E environment)
            estimated_earnings = sp500_current * 0.045
            # Calculate P/E then convert to CAPE estimate
            estimated_pe = sp500_current / estimated_earnings
            estimated_cape = estimated_pe * 1.60
            trailing_pe = estimated_pe  # For reporting
        
        # Determine valuation level
        # Historical CAPE average: ~16.8
        # Current market (2020s): ~30-35
        if estimated_cape < 20:
            level = "Undervalued"
            interpretation = "📉 Undervurderet - Historisk lav værdiansættelse"
            sentiment = "bullish"
        elif estimated_cape < 25:
            level = "Fair"
            interpretation = "➡️ Fair værdiansættelse - Omkring historisk gennemsnit"
            sentiment = "neutral"
        elif estimated_cape < 32:
            level = "Overvalued"
            interpretation = "📈 Overvurderet - Over historisk gennemsnit"
            sentiment = "cautious"
        else:
            level = "Highly Overvalued"
            interpretation = "🚨 Meget overvurderet - Høj risiko for korrektion"
            sentiment = "bearish"
        
        return {
            'cape_ratio': round(estimated_cape, 2),
            'level': level,
            'interpretation': interpretation,
            'sentiment': sentiment,
            'sp500_pe': round(trailing_pe, 2),
            'historical_avg': 16.8
        }
    except Exception as e:
        print(f"{Fore.RED}⚠️ Kunne ikke beregne Shiller P/E: {str(e)}")
        return None

def get_macro_indicators():
    """
    Henter alle makroøkonomiske indikatorer samlet
    
    Returns:
        dict: {
            'vix': dict,
            'fear_greed': dict,
            'shiller_pe': dict,
            'overall_sentiment': str - 'bullish', 'neutral', 'bearish'
        }
    """
    indicators = {
        'vix': get_vix_data(),
        'fear_greed': get_fear_greed_index(),
        'shiller_pe': get_shiller_pe()
    }
    
    # Beregn overall sentiment
    sentiments = []
    if indicators['vix']:
        sentiments.append(indicators['vix']['sentiment'])
    if indicators['fear_greed']:
        sentiments.append(indicators['fear_greed']['sentiment'])
    if indicators['shiller_pe']:
        sentiments.append(indicators['shiller_pe']['sentiment'])
    
    # Determine overall sentiment
    if not sentiments:
        overall = 'neutral'
    else:
        # Count bullish/bearish signals
        bullish_count = sum(1 for s in sentiments if 'bull' in s or s == 'greed')
        bearish_count = sum(1 for s in sentiments if 'bear' in s or 'fear' in s)
        
        if bullish_count > bearish_count:
            overall = 'bullish'
        elif bearish_count > bullish_count:
            overall = 'bearish'
        else:
            overall = 'neutral'
    
    indicators['overall_sentiment'] = overall
    
    return indicators

def generate_trading_signal(data, ml_forecasts=None, use_ml=True):
    """
    Genererer trading signal (BUY/SELL/HOLD) baseret på ML forecasts og tekniske indikatorer.
    
    Args:
        data: DataFrame med historisk data
        ml_forecasts: Dict med forecasts fra forskellige modeller
                     {'rf': {'forecast': [...], 'confidence': 0.85}, 
                      'xgboost': {...}, 'ensemble': {...}}
        use_ml: Om ML forecasts skal bruges (hvis False, kun tekniske indikatorer)
    
    Returns:
        Dict med:
        - signal: 'BUY', 'SELL', 'HOLD'
        - confidence: 0-100 (confidence score)
        - reasoning: List af grunde til signalet
        - risk_management: Dict med stop_loss, target_price, risk_reward_ratio
        - scores: Dict med individuelle scores (ml_score, technical_score, trend_score)
    """
    
    # Beregn tekniske indikatorer
    current_price = data['Close'].iloc[-1]
    rsi = calculate_rsi(data).iloc[-1]
    sma_20 = calculate_sma(data, 20).iloc[-1]
    sma_50 = calculate_sma(data, 50).iloc[-1]
    ema_12 = calculate_ema(data, 12).iloc[-1]
    macd_line, signal_line, histogram = calculate_macd(data)
    macd_current = histogram.iloc[-1]
    atr = calculate_atr(data).iloc[-1]
    
    # Initialize scoring
    total_score = 0
    max_score = 0
    reasoning = []
    
    # 1. ML FORECAST SCORE (40 points max hvis enabled)
    ml_score = 0
    if use_ml and ml_forecasts:
        ml_weight = 40
        max_score += ml_weight
        
        # Gennemsnit af alle forecasts
        forecast_changes = []
        for model_name, forecast_data in ml_forecasts.items():
            if forecast_data and 'forecast' in forecast_data:
                forecast = forecast_data['forecast']
                
                # Handle different forecast formats
                if isinstance(forecast, (list, pd.Series, np.ndarray)):
                    if len(forecast) > 0:
                        predicted_price = forecast[0] if isinstance(forecast, list) else forecast.iloc[0]
                        change_pct = ((predicted_price - current_price) / current_price) * 100
                        forecast_changes.append(change_pct)
                elif isinstance(forecast, (int, float)):
                    # Forecast is a single number
                    predicted_price = forecast
                    change_pct = ((predicted_price - current_price) / current_price) * 100
                    forecast_changes.append(change_pct)
        
        if forecast_changes:
            avg_change = np.mean(forecast_changes)
            
            if avg_change > 2:  # Stærk positiv forecast
                ml_score = ml_weight
                reasoning.append(f"📈 ML Forecast: +{avg_change:.1f}% (Stærk bullish)")
            elif avg_change > 0.5:
                ml_score = ml_weight * 0.7
                reasoning.append(f"📈 ML Forecast: +{avg_change:.1f}% (Moderat bullish)")
            elif avg_change < -2:
                ml_score = -ml_weight
                reasoning.append(f"📉 ML Forecast: {avg_change:.1f}% (Stærk bearish)")
            elif avg_change < -0.5:
                ml_score = -ml_weight * 0.7
                reasoning.append(f"📉 ML Forecast: {avg_change:.1f}% (Moderat bearish)")
            else:
                ml_score = 0
                reasoning.append(f"➡️ ML Forecast: {avg_change:.1f}% (Neutral)")
    
    total_score += ml_score
    
    # 2. RSI SCORE (20 points max)
    rsi_weight = 20
    max_score += rsi_weight
    rsi_score = 0
    
    if rsi < 30:  # Oversold
        rsi_score = rsi_weight
        reasoning.append(f"🟢 RSI: {rsi:.1f} (Oversold - køb mulighed)")
    elif rsi < 40:
        rsi_score = rsi_weight * 0.5
        reasoning.append(f"🟢 RSI: {rsi:.1f} (Lav - bullish)")
    elif rsi > 70:  # Overbought
        rsi_score = -rsi_weight
        reasoning.append(f"🔴 RSI: {rsi:.1f} (Overbought - sælg signal)")
    elif rsi > 60:
        rsi_score = -rsi_weight * 0.5
        reasoning.append(f"🔴 RSI: {rsi:.1f} (Høj - bearish)")
    else:
        rsi_score = 0
        reasoning.append(f"⚪ RSI: {rsi:.1f} (Neutral)")
    
    total_score += rsi_score
    
    # 3. MOVING AVERAGE SCORE (20 points max)
    ma_weight = 20
    max_score += ma_weight
    ma_score = 0
    
    if current_price > sma_20 and current_price > sma_50:
        if sma_20 > sma_50:  # Golden cross setup
            ma_score = ma_weight
            reasoning.append(f"🟢 MA: Pris over SMA20 & SMA50 (Stærk uptrend)")
        else:
            ma_score = ma_weight * 0.6
            reasoning.append(f"🟢 MA: Pris over begge MA (Bullish)")
    elif current_price < sma_20 and current_price < sma_50:
        if sma_20 < sma_50:  # Death cross setup
            ma_score = -ma_weight
            reasoning.append(f"🔴 MA: Pris under SMA20 & SMA50 (Stærk downtrend)")
        else:
            ma_score = -ma_weight * 0.6
            reasoning.append(f"🔴 MA: Pris under begge MA (Bearish)")
    else:
        ma_score = 0
        reasoning.append(f"⚪ MA: Pris mellem SMA20 & SMA50 (Neutral)")
    
    total_score += ma_score
    
    # 4. MACD SCORE (20 points max)
    macd_weight = 20
    max_score += macd_weight
    macd_score = 0
    
    if macd_current > 0:
        macd_score = macd_weight
        reasoning.append(f"🟢 MACD: Positiv histogram (Bullish momentum)")
    elif macd_current < 0:
        macd_score = -macd_weight
        reasoning.append(f"🔴 MACD: Negativ histogram (Bearish momentum)")
    else:
        macd_score = 0
        reasoning.append(f"⚪ MACD: Neutral")
    
    total_score += macd_score
    
    # 5. MACRO INDICATORS SCORE (30 points max - NEW!)
    macro_weight = 30
    max_score += macro_weight
    macro_score = 0
    macro_indicators = None
    
    try:
        macro_indicators = get_macro_indicators()
        
        if macro_indicators:
            # VIX Score (10 points)
            if macro_indicators['vix']:
                vix_data = macro_indicators['vix']
                if vix_data['sentiment'] == 'bullish':
                    macro_score += 10
                    reasoning.append(f"🟢 VIX: {vix_data['current']:.1f} ({vix_data['level']} - Lav volatilitet)")
                elif vix_data['sentiment'] == 'bearish':
                    macro_score -= 10
                    reasoning.append(f"🔴 VIX: {vix_data['current']:.1f} ({vix_data['level']} - Høj volatilitet)")
                else:
                    reasoning.append(f"⚪ VIX: {vix_data['current']:.1f} ({vix_data['level']})")
            
            # Fear & Greed Score (10 points)
            if macro_indicators['fear_greed']:
                fg_data = macro_indicators['fear_greed']
                if fg_data['sentiment'] in ['greed', 'extreme_greed']:
                    macro_score += 10
                    reasoning.append(f"🟢 Fear & Greed: {fg_data['value']}/100 ({fg_data['level']})")
                elif fg_data['sentiment'] in ['fear', 'extreme_fear']:
                    macro_score -= 10
                    reasoning.append(f"🔴 Fear & Greed: {fg_data['value']}/100 ({fg_data['level']})")
                else:
                    reasoning.append(f"⚪ Fear & Greed: {fg_data['value']}/100 ({fg_data['level']})")
            
            # Shiller P/E Score (10 points)
            if macro_indicators['shiller_pe']:
                pe_data = macro_indicators['shiller_pe']
                if pe_data['sentiment'] == 'bullish':
                    macro_score += 10
                    reasoning.append(f"🟢 Shiller P/E: {pe_data['cape_ratio']} ({pe_data['level']})")
                elif pe_data['sentiment'] == 'bearish':
                    macro_score -= 10
                    reasoning.append(f"🔴 Shiller P/E: {pe_data['cape_ratio']} ({pe_data['level']})")
                else:
                    reasoning.append(f"⚪ Shiller P/E: {pe_data['cape_ratio']} ({pe_data['level']})")
    
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Kunne ikke hente makro-indikatorer: {str(e)}")
        reasoning.append("⚠️ Makro-indikatorer ikke tilgængelige")
    
    total_score += macro_score
    
    # Beregn signal baseret på total score
    score_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    
    if score_percentage > 40:
        signal = "BUY"
        confidence = min(100, 50 + score_percentage * 0.5)
    elif score_percentage < -40:
        signal = "SELL"
        confidence = min(100, 50 + abs(score_percentage) * 0.5)
    else:
        signal = "HOLD"
        confidence = 50 + abs(score_percentage) * 0.3
    
    # Risk Management beregninger
    stop_loss = current_price - (2 * atr)  # 2x ATR under current price
    target_price = current_price + (3 * atr)  # 3x ATR over current price
    risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss)
    
    risk_management = {
        'current_price': current_price,
        'stop_loss': stop_loss,
        'target_price': target_price,
        'risk_reward_ratio': risk_reward_ratio,
        'atr': atr,
        'stop_loss_pct': ((stop_loss - current_price) / current_price) * 100,
        'target_price_pct': ((target_price - current_price) / current_price) * 100
    }
    
    # Tilføj risk management til reasoning
    reasoning.append(f"🎯 Stop Loss: ${stop_loss:.2f} ({risk_management['stop_loss_pct']:.1f}%)")
    reasoning.append(f"🎯 Target: ${target_price:.2f} (+{risk_management['target_price_pct']:.1f}%)")
    reasoning.append(f"📊 Risk/Reward: 1:{risk_reward_ratio:.2f}")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'reasoning': reasoning,
        'risk_management': risk_management,
        'macro_indicators': macro_indicators,
        'scores': {
            'total_score': total_score,
            'max_score': max_score,
            'score_percentage': score_percentage,
            'ml_score': ml_score,
            'rsi_score': rsi_score,
            'ma_score': ma_score,
            'macd_score': macd_score,
            'macro_score': macro_score
        }
    }

# ==================== MODEL MANAGEMENT FUNCTIONS ====================

def save_model(model, model_type, symbol, metadata=None):
    """
    Gem en trænet model til disk med metadata.
    
    Args:
        model: Den trænede model (sklearn, xgboost, prophet, etc.)
        model_type: Type af model ('rf', 'xgboost', 'prophet', 'lstm')
        symbol: Aktie symbol modellen er trænet på
        metadata: Dict med ekstra info (parameters, performance metrics, etc.)
    
    Returns:
        str: Filsti til den gemte model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{symbol}_{timestamp}.pkl"
    filepath = os.path.join(MODEL_DIR, filename)
    
    # Gem model og metadata sammen
    model_package = {
        'model': model,
        'model_type': model_type,
        'symbol': symbol,
        'timestamp': timestamp,
        'metadata': metadata or {},
        'deployed': False  # Default: ikke deployed
    }
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"{Fore.GREEN}✅ Model gemt: {filepath}")
        
        # Gem også metadata som JSON for nem læsning
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'model_type': model_type,
                'symbol': symbol,
                'timestamp': timestamp,
                'metadata': metadata or {},
                'deployed': False
            }, f, indent=2)
        
        return filepath
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved gem af model: {str(e)}")
        return None

def load_model(filepath):
    """
    Indlæs en gemt model fra disk.
    
    Args:
        filepath: Sti til model filen
    
    Returns:
        dict: Model package med model, type, symbol, timestamp, metadata
    """
    try:
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        print(f"{Fore.GREEN}✅ Model indlæst: {filepath}")
        return model_package
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved indlæsning af model: {str(e)}")
        return None

def save_training_log(log_data):
    """
    Gem træningslog til disk for senere analyse.
    
    Args:
        log_data: Dict med træningsinfo (metrics, epochs, params, etc.)
    
    Returns:
        str: Filsti til den gemte log
    """
    try:
        model_id = log_data.get('model_id', 'unknown')
        filename = f"{model_id}_training.json"
        filepath = os.path.join(LOGS_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"{Fore.CYAN}📝 Training log gemt: {filepath}")
        return filepath
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Kunne ikke gemme training log: {str(e)}")
        return None

def load_training_log(model_id):
    """
    Indlæs træningslog for en model.
    
    Args:
        model_id: Model ID (fx 'lstm_AAPL_20250112_143022')
    
    Returns:
        dict: Training log data eller None
    """
    try:
        filename = f"{model_id}_training.json"
        filepath = os.path.join(LOGS_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"{Fore.YELLOW}⚠️ Training log ikke fundet: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            log_data = json.load(f)
        
        print(f"{Fore.GREEN}✅ Training log indlæst: {filepath}")
        return log_data
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved indlæsning af training log: {str(e)}")
        return None

def list_saved_models(model_type=None, symbol=None):
    """
    Liste alle gemte modeller.
    
    Args:
        model_type: Filtrer på model type (valgfrit)
        symbol: Filtrer på symbol (valgfrit)
    
    Returns:
        list: Liste af dict med model info
    """
    models = []
    
    if not os.path.exists(MODEL_DIR):
        return models
    
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith('_metadata.json'):
            filepath = os.path.join(MODEL_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                
                # Filter
                if model_type and metadata['model_type'] != model_type:
                    continue
                if symbol and metadata['symbol'] != symbol:
                    continue
                
                metadata['filepath'] = filepath.replace('_metadata.json', '.pkl')
                metadata['filename'] = filename.replace('_metadata.json', '.pkl')
                models.append(metadata)
            except:
                continue
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return models

def delete_model(filepath):
    """
    Slet en gemt model.
    
    Args:
        filepath: Sti til model filen
    
    Returns:
        bool: True hvis slettet, False ellers
    """
    try:
        # Slet både model fil og metadata fil
        if os.path.exists(filepath):
            os.remove(filepath)
        
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        print(f"{Fore.GREEN}✅ Model slettet: {filepath}")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved sletning af model: {str(e)}")
        return False

def deploy_model(filepath):
    """
    Deploy en model (gør den tilgængelig i ML Forecast og Agent).
    
    Args:
        filepath: Sti til model filen
    
    Returns:
        bool: True hvis deployed, False ellers
    """
    try:
        # Load model package
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        # Set deployed flag
        model_package['deployed'] = True
        
        # Save back
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Update metadata JSON
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['deployed'] = True
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{Fore.GREEN}✅ Model deployed")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved deploy: {str(e)}")
        return False

def undeploy_model(filepath):
    """
    Undeploy en model (fjern fra ML Forecast og Agent).
    
    Args:
        filepath: Sti til model filen
    
    Returns:
        bool: True hvis undeployed, False ellers
    """
    try:
        # Load model package
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        # Set deployed flag
        model_package['deployed'] = False
        
        # Save back
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Update metadata JSON
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['deployed'] = False
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{Fore.GREEN}✅ Model undeployed")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved undeploy: {str(e)}")
        return False

def train_and_save_rf(data, symbol, n_estimators=100, max_depth=10, window=30, horizon=1):
    """
    Træn en Random Forest model med custom parametre og gem den.
    
    Args:
        data: DataFrame med historisk data
        symbol: Aktie symbol
        n_estimators: Antal træer
        max_depth: Maksimal dybde af træer
        window: Antal dage til features
        horizon: Forecast horizont
    
    Returns:
        dict: Model info og performance metrics
    """
    print(f"{Fore.CYAN}⚡ Træner Random Forest model...")
    print(f"   Parametre: n_estimators={n_estimators}, max_depth={max_depth}, window={window}")
    
    close_prices = data['Close'].values.flatten()  # Ensure 1D array
    
    # Create sliding windows
    X, y = [], []
    for i in range(len(close_prices) - window - horizon + 1):
        X.append(close_prices[i:i+window])
        y.append(close_prices[i+window+horizon-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Calculate performance
    train_predictions = rf_model.predict(X_train)
    val_predictions = rf_model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    # Get feature importance
    feature_importance = rf_model.feature_importances_.tolist()
    
    # Prepare timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"rf_{symbol}_{timestamp}"
    
    metadata = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'window': window,
        'horizon': horizon,
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'data_period': f"{data.index[0]} to {data.index[-1]}"
    }
    
    # Create training log
    training_log = {
        'model_id': model_id,
        'model_type': 'rf',
        'symbol': symbol,
        'timestamp': timestamp,
        'final_metrics': {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse)
        },
        'parameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'window': window,
            'horizon': horizon,
            'random_state': 42
        },
        'data_stats': {
            'total_samples': len(data),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'period': f"{data.index[0]} to {data.index[-1]}",
            'price_mean': float(data['Close'].mean()),
            'price_std': float(data['Close'].std()),
            'price_min': float(data['Close'].min()),
            'price_max': float(data['Close'].max())
        },
        'feature_importance': {
            'top_10': feature_importance[:10] if len(feature_importance) >= 10 else feature_importance,
            'mean': float(np.mean(feature_importance)),
            'std': float(np.std(feature_importance))
        }
    }
    
    # Save training log
    save_training_log(training_log)
    
    # Save model
    filepath = save_model(rf_model, 'rf', symbol, metadata)
    
    print(f"{Fore.GREEN}✅ Random Forest trænet!")
    print(f"   Training MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"   Training RMSE: ${train_rmse:.2f}, Val RMSE: ${val_rmse:.2f}")
    print(f"   Training samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    return {
        'model': rf_model,
        'filepath': filepath,
        'metadata': metadata
    }

def train_and_save_xgboost(data, symbol, n_estimators=100, max_depth=6, learning_rate=0.1, window=30, horizon=1):
    """
    Træn en XGBoost model med custom parametre og gem den.
    """
    print(f"{Fore.CYAN}⚡ Træner XGBoost model...")
    print(f"   Parametre: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    
    close_prices = data['Close'].values.flatten()  # Ensure 1D array
    
    # Create sliding windows
    X, y = [], []
    for i in range(len(close_prices) - window - horizon + 1):
        X.append(close_prices[i:i+window])
        y.append(close_prices[i+window+horizon-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    
    # Calculate performance
    train_predictions = xgb_model.predict(X_train)
    val_predictions = xgb_model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    # Get feature importance
    feature_importance = xgb_model.feature_importances_.tolist()
    
    # Prepare timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"xgboost_{symbol}_{timestamp}"
    
    metadata = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'window': window,
        'horizon': horizon,
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'data_period': f"{data.index[0]} to {data.index[-1]}"
    }
    
    # Create training log
    training_log = {
        'model_id': model_id,
        'model_type': 'xgboost',
        'symbol': symbol,
        'timestamp': timestamp,
        'final_metrics': {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse)
        },
        'parameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'window': window,
            'horizon': horizon,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'data_stats': {
            'total_samples': len(data),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'period': f"{data.index[0]} to {data.index[-1]}",
            'price_mean': float(data['Close'].mean()),
            'price_std': float(data['Close'].std()),
            'price_min': float(data['Close'].min()),
            'price_max': float(data['Close'].max())
        },
        'feature_importance': {
            'top_10': feature_importance[:10] if len(feature_importance) >= 10 else feature_importance,
            'mean': float(np.mean(feature_importance)),
            'std': float(np.std(feature_importance))
        }
    }
    
    # Save training log
    save_training_log(training_log)
    
    # Save model
    filepath = save_model(xgb_model, 'xgboost', symbol, metadata)
    
    print(f"{Fore.GREEN}✅ XGBoost trænet!")
    print(f"   Training MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"   Training RMSE: ${train_rmse:.2f}, Val RMSE: ${val_rmse:.2f}")
    
    return {
        'model': xgb_model,
        'filepath': filepath,
        'metadata': metadata
    }

def train_and_save_prophet(data, symbol, horizon=1, daily_seasonality=True, weekly_seasonality=True):
    """
    Træn en Prophet model med custom parametre og gem den.
    """
    print(f"{Fore.CYAN}📈 Træner Prophet model...")
    print(f"   Parametre: horizon={horizon}, daily={daily_seasonality}, weekly={weekly_seasonality}")
    
    # Prepare data for Prophet
    prophet_data = pd.DataFrame({
        'ds': data.index,
        'y': data['Close'].values.flatten()
    })
    
    # Suppress Prophet logging
    import logging
    logging.getLogger('prophet').setLevel(logging.ERROR)
    
    # Train model
    prophet_model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(prophet_data)
    
    # Make predictions on training data for metrics
    future = prophet_model.make_future_dataframe(periods=0)
    forecast = prophet_model.predict(future)
    
    # Split for validation
    split_idx = int(len(prophet_data) * 0.8)
    train_data = prophet_data[:split_idx]
    val_data = prophet_data[split_idx:]
    
    # Calculate training performance
    y_true = train_data['y'].values
    y_pred = forecast['yhat'].values[:len(train_data)]
    train_mae = mean_absolute_error(y_true, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Validation performance
    val_y_true = val_data['y'].values
    val_y_pred = forecast['yhat'].values[split_idx:]
    val_mae = mean_absolute_error(val_y_true, val_y_pred)
    val_rmse = np.sqrt(mean_squared_error(val_y_true, val_y_pred))
    
    # Prepare timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"prophet_{symbol}_{timestamp}"
    
    metadata = {
        'horizon': horizon,
        'daily_seasonality': daily_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'training_samples': len(train_data),
        'validation_samples': len(val_data),
        'data_period': f"{data.index[0]} to {data.index[-1]}"
    }
    
    # Create training log
    training_log = {
        'model_id': model_id,
        'model_type': 'prophet',
        'symbol': symbol,
        'timestamp': timestamp,
        'final_metrics': {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse)
        },
        'parameters': {
            'horizon': horizon,
            'daily_seasonality': daily_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'yearly_seasonality': False,
            'changepoint_prior_scale': 0.05
        },
        'data_stats': {
            'total_samples': len(prophet_data),
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'period': f"{data.index[0]} to {data.index[-1]}",
            'price_mean': float(data['Close'].mean()),
            'price_std': float(data['Close'].std()),
            'price_min': float(data['Close'].min()),
            'price_max': float(data['Close'].max())
        },
        'forecast_components': {
            'trend': forecast['trend'].describe().to_dict(),
            'has_daily': daily_seasonality,
            'has_weekly': weekly_seasonality
        }
    }
    
    # Save training log
    save_training_log(training_log)
    
    # Save model
    filepath = save_model(prophet_model, 'prophet', symbol, metadata)
    
    print(f"{Fore.GREEN}✅ Prophet trænet!")
    print(f"   Training MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"   Training RMSE: ${train_rmse:.2f}, Val RMSE: ${val_rmse:.2f}")
    
    return {
        'model': prophet_model,
        'filepath': filepath,
        'metadata': metadata
    }

def train_and_save_lstm(data, symbol, window=30, epochs=50, horizon=1):
    """
    Træn en LSTM model med custom parametre og gem den.
    """
    if not TENSORFLOW_AVAILABLE:
        print(f"{Fore.RED}❌ TensorFlow ikke tilgængelig!")
        return None
    
    print(f"{Fore.CYAN}🧠 Træner LSTM model...")
    print(f"   Parametre: window={window}, epochs={epochs}, horizon={horizon}")
    
    # Prepare data
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - window - horizon + 1):
        X.append(scaled_data[i:i+window])
        y.append(scaled_data[i+window+horizon-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Build LSTM model
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    
    lstm_model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(window, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train with history capture (keep output suppressed)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )
    
    sys.stdout = old_stdout
    
    # Calculate training performance
    train_predictions = lstm_model.predict(X_train, verbose=0)
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train_actual = scaler.inverse_transform(y_train)
    
    val_predictions = lstm_model.predict(X_val, verbose=0)
    val_predictions = scaler.inverse_transform(val_predictions)
    y_val_actual = scaler.inverse_transform(y_val)
    
    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
    val_mae = mean_absolute_error(y_val_actual, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val_actual, val_predictions))
    
    # Prepare timestamp for IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"lstm_{symbol}_{timestamp}"
    
    metadata = {
        'window': window,
        'epochs': epochs,
        'horizon': horizon,
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'data_period': f"{data.index[0]} to {data.index[-1]}",
        'scaler_min': float(scaler.data_min_[0]),
        'scaler_max': float(scaler.data_max_[0])
    }
    
    # Create training log
    training_log = {
        'model_id': model_id,
        'model_type': 'lstm',
        'symbol': symbol,
        'timestamp': timestamp,
        'epochs_data': [
            {
                'epoch': i + 1,
                'loss': float(history.history['loss'][i]),
                'val_loss': float(history.history['val_loss'][i])
            }
            for i in range(len(history.history['loss']))
        ],
        'final_metrics': {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse)
        },
        'parameters': {
            'window': window,
            'epochs': epochs,
            'horizon': horizon,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'mse'
        },
        'data_stats': {
            'total_samples': len(data),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'period': f"{data.index[0]} to {data.index[-1]}",
            'price_mean': float(data['Close'].mean()),
            'price_std': float(data['Close'].std()),
            'price_min': float(data['Close'].min()),
            'price_max': float(data['Close'].max())
        }
    }
    
    # Save training log
    save_training_log(training_log)
    
    # Save model and scaler together
    model_package_custom = {
        'lstm_model': lstm_model,
        'scaler': scaler
    }
    
    filepath = save_model(model_package_custom, 'lstm', symbol, metadata)
    
    print(f"{Fore.GREEN}✅ LSTM trænet!")
    print(f"   Training MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"   Training RMSE: ${train_rmse:.2f}, Val RMSE: ${val_rmse:.2f}")
    
    return {
        'model': model_package_custom,
        'filepath': filepath,
        'metadata': metadata
    }

def predict_with_saved_model(model_package, data, horizon=1):
    """
    Lav prediction med en indlæst model.
    
    Args:
        model_package: Model package fra load_model()
        data: DataFrame med ny data
        horizon: Forecast horizont (skal matche model's training horizon)
    
    Returns:
        dict: Forecast resultat
    """
    model = model_package['model']
    model_type = model_package['model_type']
    metadata = model_package['metadata']
    window = metadata.get('window', 30)
    
    close_prices = data['Close'].values
    
    if model_type in ['rf', 'xgboost']:
        # Use last window for prediction
        last_window = close_prices[-window:].reshape(1, -1)
        forecast = float(model.predict(last_window)[0])
        current_price = float(close_prices[-1])
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": ((forecast - current_price) / current_price) * 100,
            "model_type": model_type,
            "model_metadata": metadata
        }
    
    elif model_type == 'prophet':
        # Prophet has its own prediction method
        future = model.make_future_dataframe(periods=horizon)
        forecast_df = model.predict(future)
        forecast = float(forecast_df['yhat'].iloc[-1])
        current_price = float(close_prices[-1])
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": ((forecast - current_price) / current_price) * 100,
            "model_type": model_type,
            "model_metadata": metadata
        }
    
    elif model_type == 'lstm':
        # LSTM requires scaler and proper reshaping
        lstm_model = model['lstm_model']
        scaler = model['scaler']
        
        # Scale data
        scaled_data = scaler.transform(close_prices.reshape(-1, 1))
        
        # Get last window
        last_window = scaled_data[-window:].reshape(1, window, 1)
        
        # Predict
        prediction_scaled = lstm_model.predict(last_window, verbose=0)
        forecast = float(scaler.inverse_transform(prediction_scaled)[0][0])
        current_price = float(close_prices[-1])
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": ((forecast - current_price) / current_price) * 100,
            "model_type": model_type,
            "model_metadata": metadata
        }
    
    return None

# TensorFlow/Keras import - optional (kun nødvendig for ML mode)
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Sequential = None
    LSTM = None
    Dense = None

# ---------- Hjælpefunktioner ----------

def load_config():
    """Indlæser konfiguration fra config.json."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Fore.YELLOW}⚠️  config.json ikke fundet. Bruger default værdier.")
        return {
            "default_settings": {
                "data_period": "6mo",
                "data_interval": "1d",
                "ml_window": 30,
                "ml_epochs": 3
            },
            "watchlist": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
            "reflection_questions": {}
        }

def clear_screen():
    """Rydder terminal skærmen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Printer header for agenten."""
    clear_screen()
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("=" * 60)
    print("           📈 TRADING MENTOR AGENT 📈")
    print("=" * 60)
    print(f"{Style.RESET_ALL}")

def print_menu():
    """Viser hovedmenu."""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}╔════════════════════════════════════════╗")
    print(f"║         HVAD VIL DU LAVE I DAG?        ║")
    print(f"╚════════════════════════════════════════╝{Style.RESET_ALL}")
    print(f"{Fore.GREEN}1.{Fore.WHITE} 🔎 Teknisk analyse af en aktie")
    print(f"{Fore.GREEN}2.{Fore.WHITE} 📘 Læringsmodus - forstå indikatorer")
    print(f"{Fore.GREEN}3.{Fore.WHITE} 🤖 ML Forecast - forudsig kursbevægelse")
    print(f"{Fore.GREEN}4.{Fore.WHITE} � Interaktiv graf (Plotly)")
    print(f"{Fore.GREEN}5.{Fore.WHITE} 📄 Generer HTML rapport")
    print(f"{Fore.GREEN}6.{Fore.WHITE} �📋 Se din watchlist")
    print(f"{Fore.GREEN}7.{Fore.WHITE} ⚙️  Indstillinger")
    print(f"{Fore.RED}8.{Fore.WHITE} 🚪 Afslut")
    print()

def hent_data(symbol, period="6mo", interval="1d"):
    """Henter historiske aktidata fra Yahoo Finance."""
    try:
        print(f"{Fore.CYAN}📥 Henter data for {symbol}...")
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            print(f"{Fore.RED}❌ Ingen data fundet for {symbol}. Tjek om symbolet er korrekt.")
            return None
        
        # Fix for yfinance MultiIndex columns (ny adfærd)
        # Hvis columns er MultiIndex, flatten dem
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"{Fore.GREEN}✅ Data hentet: {len(data)} dage")
        return data
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved datahentning: {e}")
        return None

def teknisk_analyse(data, symbol, config=None, print_results=True):
    """Beregner tekniske indikatorer baseret på enabled_indicators i config."""
    try:
        enabled = config.get("enabled_indicators", {}) if config else {}
        settings = config.get("default_settings", {}) if config else {}
        close = data["Close"].iloc[-1].item()
        
        if print_results:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 TEKNISK ANALYSE: {symbol}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─' * 50}")
            print(f"{Fore.YELLOW}Aktuel kurs: {Fore.WHITE}${close:.2f}")
            print(f"{Fore.WHITE}{'─' * 50}")
        
        # RSI
        if enabled.get("rsi", True):
            data["RSI"] = calculate_rsi(data, periods=settings.get("rsi_period", 14))
            rsi = float(data["RSI"].iloc[-1])
            print(f"\n{Fore.CYAN}RSI (14):{Style.RESET_ALL}")
            print(f"  Værdi: {Fore.WHITE}{rsi:.1f}")
            if rsi > 70:
                print(f"  {Fore.RED}⚠️  OVERKØBT (>70) - potentiel korrektion")
            elif rsi < 30:
                print(f"  {Fore.GREEN}🔻 OVERSOLGT (<30) - potentiel recovery")
            else:
                print(f"  {Fore.GREEN}✅ Neutral zone (30-70)")
        
        # SMA
        if enabled.get("sma", True):
            data["SMA50"] = calculate_sma(data, periods=settings.get("sma_short", 50))
            data["SMA200"] = calculate_sma(data, periods=settings.get("sma_long", 200))
            sma50 = float(data["SMA50"].iloc[-1])
            sma200 = float(data["SMA200"].iloc[-1])
            print(f"\n{Fore.CYAN}Moving Averages:{Style.RESET_ALL}")
            print(f"  SMA50:  {Fore.WHITE}${sma50:.2f}")
            print(f"  SMA200: {Fore.WHITE}${sma200:.2f}")
            if sma50 > sma200:
                print(f"  {Fore.GREEN}📈 Golden Cross (bullish)")
            elif sma50 < sma200:
                print(f"  {Fore.RED}📉 Death Cross (bearish)")
        
        # MACD
        if enabled.get("macd", True):
            macd_line, signal_line, histogram = calculate_macd(data, 
                fast=settings.get("macd_fast", 12),
                slow=settings.get("macd_slow", 26),
                signal=settings.get("macd_signal", 9))
            data["MACD"] = macd_line
            data["MACD_Signal"] = signal_line
            data["MACD_Hist"] = histogram
            
            macd = macd_line.iloc[-1].item()
            signal = signal_line.iloc[-1].item()
            hist = histogram.iloc[-1].item()
            
            print(f"\n{Fore.CYAN}MACD:{Style.RESET_ALL}")
            print(f"  MACD Line:   {Fore.WHITE}{macd:.2f}")
            print(f"  Signal Line: {Fore.WHITE}{signal:.2f}")
            print(f"  Histogram:   {Fore.WHITE}{hist:.2f}")
            if macd > signal:
                print(f"  {Fore.GREEN}📈 Bullish signal (MACD > Signal)")
            else:
                print(f"  {Fore.RED}📉 Bearish signal (MACD < Signal)")
        
        # Bollinger Bands
        if enabled.get("bollinger_bands", True):
            upper, middle, lower = calculate_bollinger_bands(data,
                periods=settings.get("bb_period", 20),
                std_dev=settings.get("bb_std", 2))
            data["BB_Upper"] = upper
            data["BB_Middle"] = middle
            data["BB_Lower"] = lower
            
            bb_upper = upper.iloc[-1].item()
            bb_middle = middle.iloc[-1].item()
            bb_lower = lower.iloc[-1].item()
            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
            
            print(f"\n{Fore.CYAN}Bollinger Bands:{Style.RESET_ALL}")
            print(f"  Upper:  {Fore.WHITE}${bb_upper:.2f}")
            print(f"  Middle: {Fore.WHITE}${bb_middle:.2f}")
            print(f"  Lower:  {Fore.WHITE}${bb_lower:.2f}")
            print(f"  Width:  {Fore.WHITE}{bb_width:.2f}%")
            
            if close > bb_upper:
                print(f"  {Fore.RED}⚠️  Kurs over upper band (potentiel overkøb)")
            elif close < bb_lower:
                print(f"  {Fore.GREEN}� Kurs under lower band (potentiel oversalg)")
            else:
                print(f"  {Fore.GREEN}✅ Kurs inden for bands")
        
        # ATR (Volatilitet)
        if enabled.get("atr", True):
            data["ATR"] = calculate_atr(data, periods=settings.get("atr_period", 14))
            atr = float(data["ATR"].iloc[-1])
            atr_pct = (atr / close) * 100
            
            print(f"\n{Fore.CYAN}ATR (Volatilitet):{Style.RESET_ALL}")
            print(f"  ATR:     {Fore.WHITE}${atr:.2f}")
            print(f"  ATR %:   {Fore.WHITE}{atr_pct:.2f}%")
            if atr_pct > 3:
                print(f"  {Fore.YELLOW}⚠️  Høj volatilitet (>3%)")
            elif atr_pct < 1:
                print(f"  {Fore.CYAN}📉 Lav volatilitet (<1%)")
            else:
                print(f"  {Fore.GREEN}✅ Moderat volatilitet")
        
        print(f"{Fore.WHITE}{'─' * 50}")
        return data
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i teknisk analyse: {e}")
        import traceback
        traceback.print_exc()
        return None

def vis_graf(data, symbol):
    """Visualiserer aktidata med tekniske indikatorer i multi-panel layout."""
    try:
        # Opret 4 subplots: Pris+BB, RSI, MACD, ATR
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), 
                                                   gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Panel 1: Pris, SMA og Bollinger Bands
        ax1.plot(data.index, data["Close"], label="Close Price", linewidth=2, color='#2E86DE')
        ax1.plot(data.index, data["SMA50"], label="SMA50", linewidth=1.5, color='#10AC84', linestyle='--')
        ax1.plot(data.index, data["SMA200"], label="SMA200", linewidth=1.5, color='#EE5A6F', linestyle='--')
        
        # Tilføj Bollinger Bands hvis de findes
        if "BB_Upper" in data.columns and "BB_Lower" in data.columns:
            ax1.plot(data.index, data["BB_Upper"], label="BB Upper", linewidth=1, color='#FFA502', linestyle=':')
            ax1.plot(data.index, data["BB_Middle"], label="BB Middle", linewidth=1, color='#FFA502', linestyle='--')
            ax1.plot(data.index, data["BB_Lower"], label="BB Lower", linewidth=1, color='#FFA502', linestyle=':')
            ax1.fill_between(data.index, data["BB_Upper"], data["BB_Lower"], alpha=0.1, color='#FFA502')
        
        ax1.set_title(f"{symbol} - Teknisk Analyse", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Pris ($)", fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: RSI med overbought/oversold zoner
        if "RSI" in data.columns:
            ax2.plot(data.index, data["RSI"], label="RSI", linewidth=1.5, color='#9C27B0')
            ax2.axhline(y=70, color='#EE5A6F', linestyle='--', linewidth=1, label='Overbought (70)')
            ax2.axhline(y=30, color='#10AC84', linestyle='--', linewidth=1, label='Oversold (30)')
            ax2.fill_between(data.index, 70, 100, alpha=0.1, color='#EE5A6F')
            ax2.fill_between(data.index, 0, 30, alpha=0.1, color='#10AC84')
            ax2.set_ylabel("RSI", fontsize=11)
            ax2.set_ylim(0, 100)
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: MACD
        if "MACD_Line" in data.columns:
            ax3.plot(data.index, data["MACD_Line"], label="MACD Line", linewidth=1.5, color='#2E86DE')
            ax3.plot(data.index, data["MACD_Signal"], label="Signal Line", linewidth=1.5, color='#EE5A6F')
            ax3.bar(data.index, data["MACD_Hist"], label="Histogram", alpha=0.3, color='#546E7A')
            ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
            ax3.set_ylabel("MACD", fontsize=11)
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: ATR (Volatilitet)
        if "ATR" in data.columns:
            ax4.plot(data.index, data["ATR"], label="ATR (14)", linewidth=1.5, color='#FF6348')
            ax4.fill_between(data.index, data["ATR"], alpha=0.2, color='#FF6348')
            ax4.set_ylabel("ATR", fontsize=11)
            ax4.set_xlabel("Dato", fontsize=12)
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"{Fore.GREEN}✅ Graf vist med alle indikatorer!")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved visning af graf: {e}")

def vis_graf_plotly(data, symbol):
    """Visualiserer aktidata med interaktive plotly grafer."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Opret 4 subplots med plotly
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            subplot_titles=(
                f'{symbol} - Pris & Bollinger Bands', 
                'RSI', 
                'MACD', 
                'ATR (Volatilitet)'
            )
        )
        
        # Panel 1: Pris, SMA og Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"],
            name="Close Price",
            line=dict(color='#2E86DE', width=2),
            hovertemplate='Pris: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data["SMA50"],
            name="SMA50",
            line=dict(color='#10AC84', width=1.5, dash='dash'),
            hovertemplate='SMA50: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data["SMA200"],
            name="SMA200",
            line=dict(color='#EE5A6F', width=1.5, dash='dash'),
            hovertemplate='SMA200: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # Bollinger Bands hvis de findes
        if "BB_Upper" in data.columns:
            # Tilføj Lower band først
            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Lower"],
                name="BB Lower",
                line=dict(color='#FFA502', width=1, dash='dot'),
                hovertemplate='BB Lower: $%{y:.2f}<extra></extra>',
                showlegend=True
            ), row=1, col=1)
            
            # Tilføj Upper band med fill til Lower band
            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Upper"],
                name="BB Upper",
                line=dict(color='#FFA502', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(255, 165, 2, 0.1)',
                hovertemplate='BB Upper: $%{y:.2f}<extra></extra>',
                showlegend=True
            ), row=1, col=1)
            
            # Tilføj Middle band
            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Middle"],
                name="BB Middle (SMA20)",
                line=dict(color='#FFA502', width=1.5, dash='dash'),
                hovertemplate='BB Middle: $%{y:.2f}<extra></extra>',
                showlegend=True
            ), row=1, col=1)
        
        # Panel 2: RSI
        if "RSI" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["RSI"],
                name="RSI",
                line=dict(color='#9C27B0', width=2),
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ), row=2, col=1)
            
            # Overbought/Oversold linjer
            fig.add_hline(y=70, line_dash="dash", line_color="#EE5A6F", 
                         annotation_text="Overbought", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#10AC84",
                         annotation_text="Oversold", row=2, col=1)
            
            # Farvede zoner
            fig.add_hrect(y0=70, y1=100, fillcolor="#EE5A6F", opacity=0.1, row=2, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="#10AC84", opacity=0.1, row=2, col=1)
        
        # Panel 3: MACD
        if "MACD_Line" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["MACD_Line"],
                name="MACD Line",
                line=dict(color='#2E86DE', width=2),
                hovertemplate='MACD: %{y:.2f}<extra></extra>'
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data["MACD_Signal"],
                name="Signal Line",
                line=dict(color='#EE5A6F', width=2),
                hovertemplate='Signal: %{y:.2f}<extra></extra>'
            ), row=3, col=1)
            
            # Histogram med farver baseret på værdi
            colors = ['#10AC84' if val > 0 else '#EE5A6F' 
                     for val in data["MACD_Hist"]]
            fig.add_trace(go.Bar(
                x=data.index, y=data["MACD_Hist"],
                name="Histogram",
                marker_color=colors,
                opacity=0.5,
                hovertemplate='Histogram: %{y:.2f}<extra></extra>'
            ), row=3, col=1)
            
            fig.add_hline(y=0, line_color="gray", line_width=1, row=3, col=1)
        
        # Panel 4: ATR
        if "ATR" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ATR"],
                name="ATR",
                line=dict(color='#FF6348', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 99, 72, 0.2)',
                hovertemplate='ATR: $%{y:.2f}<extra></extra>'
            ), row=4, col=1)
        
        # Layout opdateringer
        fig.update_xaxes(title_text="Dato", row=4, col=1)
        fig.update_yaxes(title_text="Pris ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="ATR", row=4, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=True,
            hovermode='x unified',
            title_text=f"{symbol} - Interaktiv Teknisk Analyse",
            title_font_size=20
        )
        
        # Vis grafen i browser
        fig.show()
        print(f"{Fore.GREEN}✅ Interaktiv graf åbnet i browser!")
        
        return fig
        
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved visning af plotly graf: {e}")
        import traceback
        traceback.print_exc()
        return None

def generer_html_rapport(symbol, data, analyse_results=None, ml_results=None):
    """
    Genererer en komplet HTML rapport med teknisk analyse og interaktiv graf.
    
    Args:
        symbol: Aktiesymbol (fx AAPL)
        data: pandas DataFrame med aktiedata og indikatorer
        analyse_results: Dict med teknisk analyse resultater
        ml_results: Dict med ML forecast resultater
    
    Returns:
        Filnavn på den genererede rapport
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from datetime import datetime
        
        # Generer plotly figur
        fig = vis_graf_plotly(data, symbol)
        if fig is None:
            print(f"{Fore.RED}❌ Kunne ikke generere graf til rapport")
            return None
        
        # Konverter graf til HTML
        graph_html = fig.to_html(include_plotlyjs='cdn', div_id='interactive-chart')
        
        # Byg HTML rapport
        current_date = datetime.now().strftime("%d. %B %Y kl. %H:%M")
        current_price = data["Close"].iloc[-1].item()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="da">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} - Teknisk Analyse Rapport</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2E86DE;
            border-bottom: 3px solid #2E86DE;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #667eea;
            margin-top: 30px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .price {{
            font-size: 2.5em;
            font-weight: bold;
            color: #10AC84;
        }}
        .info-box {{
            background: #f8f9fa;
            border-left: 4px solid #2E86DE;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .indicator {{
            background: white;
            border: 1px solid #e0e0e0;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .indicator-name {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}
        .indicator-value {{
            font-size: 1.3em;
            margin: 5px 0;
        }}
        .bullish {{ color: #10AC84; }}
        .bearish {{ color: #EE5A6F; }}
        .neutral {{ color: #FFA502; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #888;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>📊 {symbol} - Teknisk Analyse Rapport</h1>
                <p style="color: #888;">Genereret: {current_date}</p>
            </div>
            <div class="price">${current_price:.2f}</div>
        </div>
        
        <div class="info-box">
            <strong>ℹ️ Om denne rapport:</strong><br>
            Denne rapport indeholder teknisk analyse af {symbol} med avancerede indikatorer 
            og machine learning forecasts. Grafen nedenfor er interaktiv - du kan zoome, 
            panorere og se detaljerede værdier ved at holde musen over grafen.
        </div>
"""
        
        # Tilføj teknisk analyse sektion hvis tilgængelig
        if analyse_results is not None and not (isinstance(analyse_results, pd.DataFrame) and analyse_results.empty):
            html_content += """
        <h2>📈 Teknisk Analyse</h2>
        <div class="indicator">
"""
            # Handle both dict and DataFrame
            if isinstance(analyse_results, dict):
                for key, value in analyse_results.items():
                    html_content += f"            <p><span class='indicator-name'>{key}:</span> {value}</p>\n"
            elif isinstance(analyse_results, pd.DataFrame):
                for key, value in analyse_results.to_dict().items():
                    html_content += f"            <p><span class='indicator-name'>{key}:</span> {value}</p>\n"
            html_content += "        </div>\n"
        
        # Tilføj ML forecast sektion hvis tilgængelig
        if ml_results is not None and ml_results:
            html_content += """
        <h2>🤖 Machine Learning Forecast</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Horisont</th>
                    <th>Forecast</th>
                    <th>Ændring</th>
                    <th>Signal</th>
                </tr>
            </thead>
            <tbody>
"""
            # Handle both list and dict format
            results_list = []
            if isinstance(ml_results, dict):
                # Convert dict format to list format
                current_price = float(data['Close'].iloc[-1])
                for key, value in ml_results.items():
                    if isinstance(value, dict):
                        results_list.append(value)
                    else:
                        # Value is a float (price forecast)
                        model_name = 'LSTM' if 'lstm' in key.lower() else 'Random Forest'
                        horizon_name = key.split('_')[-1]  # e.g., '1d', '5d', '22d'
                        forecast = float(value)
                        change_pct = ((forecast - current_price) / current_price) * 100
                        results_list.append({
                            'model': model_name,
                            'horizon_name': horizon_name,
                            'forecast': forecast,
                            'change_pct': change_pct
                        })
            else:
                results_list = ml_results
            
            for result in results_list:
                if isinstance(result, dict):
                    change_class = 'bullish' if result.get('change_pct', 0) > 0 else 'bearish'
                    signal = '📈 Bullish' if result.get('change_pct', 0) > 0 else '📉 Bearish'
                    html_content += f"""
                <tr>
                    <td>{result.get('model', 'LSTM')}</td>
                    <td>{result.get('horizon_name', 'N/A')}</td>
                    <td>${result.get('forecast', 0):.2f}</td>
                    <td class='{change_class}'>{result.get('change_pct', 0):+.2f}%</td>
                    <td class='{change_class}'>{signal}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
"""
        
        # Tilføj interaktiv graf
        html_content += f"""
        <h2>📊 Interaktiv Graf</h2>
        {graph_html}
        
        <div class="footer">
            <p>🤖 Genereret af ML Stock Agent</p>
            <p>⚠️ Denne rapport er kun til uddannelsesformål. Invester aldrig mere end du har råd til at tabe.</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Gem rapport
        filename = f"{symbol}_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"{Fore.GREEN}✅ HTML rapport gemt: {filename}")
        print(f"{Fore.CYAN}💡 Åbn filen i en browser for at se den interaktive rapport!")
        
        return filename
        
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl ved generering af HTML rapport: {e}")
        import traceback
        traceback.print_exc()
        return None

def ml_forecast(data, window=30, epochs=3, horizon=1):
    """
    Laver LSTM-baseret forecast.
    
    Args:
        data: pandas DataFrame med aktiedata
        window: Antal dage at bruge som input
        epochs: Antal træningsepocher
        horizon: Forecast horisont (1=1 dag, 5=5 dage, 21=1 måned)
    
    Returns:
        Dict med forecasts eller None ved fejl
    """
    if not TENSORFLOW_AVAILABLE:
        print(f"{Fore.RED}❌ TensorFlow er ikke installeret!")
        print(f"{Fore.YELLOW}   Installer med: pip install tensorflow")
        print(f"{Fore.CYAN}   Bemærk: TensorFlow kræver Python 3.8-3.11 på Windows.")
        return None
    
    try:
        horizon_name = {1: "1 dag", 5: "5 dage", 21: "1 måned"}.get(horizon, f"{horizon} dage")
        print(f"{Fore.CYAN}🤖 Træner LSTM-model for {horizon_name} forecast...")
        
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data[["Close"]].dropna())

        if len(scaled) < window + horizon + 10:
            print(f"{Fore.YELLOW}⚠️  For lidt data til ML (skal bruge minimum {window + horizon + 10} dage)")
            return None

        X, y = [], []
        for i in range(window, len(scaled) - horizon + 1):
            X.append(scaled[i-window:i, 0])
            y.append(scaled[i+horizon-1, 0])  # Target er 'horizon' dage frem
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Byg model tilpasset horizon
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
            LSTM(50),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        # Lav forecast
        pred = model.predict(X[-1].reshape(1, window, 1), verbose=0)
        forecast = float(scaler.inverse_transform(pred)[0][0])
        
        current_price = data["Close"].iloc[-1].item()
        change_pct = ((forecast - current_price) / current_price) * 100
        
        print(f"{Fore.GREEN}✅ Model trænet på {len(X)} samples")
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": change_pct,
            "horizon": horizon,
            "horizon_name": horizon_name
        }
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i ML forecast: {e}")
        import traceback
        traceback.print_exc()
        return None

def ml_forecast_rf(data, window=30, horizon=1):
    """
    Laver Random Forest baseret forecast.
    
    Args:
        data: pandas DataFrame med aktiedata
        window: Antal dage at bruge som input
        horizon: Forecast horisont (1=1 dag, 5=5 dage, 21=1 måned)
    
    Returns:
        Dict med forecasts eller None ved fejl
    """
    try:
        horizon_name = {1: "1 dag", 5: "5 dage", 21: "1 måned"}.get(horizon, f"{horizon} dage")
        print(f"{Fore.CYAN}🌲 Træner Random Forest model for {horizon_name} forecast...")
        
        # Prepare data - flatten if needed
        close_prices = data["Close"].dropna().values
        if close_prices.ndim > 1:
            close_prices = close_prices.flatten()
        
        if len(close_prices) < window + horizon + 10:
            print(f"{Fore.YELLOW}⚠️  For lidt data til ML")
            return None
        
        X, y = [], []
        for i in range(window, len(close_prices) - horizon + 1):
            X.append(close_prices[i-window:i])
            y.append(close_prices[i+horizon-1])
        
        X, y = np.array(X), np.array(y)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        
        # Make prediction
        last_window = close_prices[-window:].reshape(1, -1)
        forecast = float(rf_model.predict(last_window)[0])
        current_price = float(close_prices[-1])
        change_pct = ((forecast - current_price) / current_price) * 100
        
        print(f"{Fore.GREEN}✅ Random Forest trænet på {len(X)} samples")
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": change_pct,
            "horizon": horizon,
            "horizon_name": horizon_name
        }
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i Random Forest forecast: {e}")
        import traceback
        traceback.print_exc()
        return None

def ml_forecast_xgboost(data, window=30, horizon=1):
    """
    Laver XGBoost baseret forecast.
    
    Args:
        data: pandas DataFrame med aktiedata
        window: Antal dage at bruge som input
        horizon: Forecast horisont (1=1 dag, 5=5 dage, 22=1 måned)
    
    Returns:
        Dict med forecasts eller None ved fejl
    """
    try:
        horizon_name = {1: "1 dag", 5: "5 dage", 22: "1 måned"}.get(horizon, f"{horizon} dage")
        print(f"{Fore.CYAN}⚡ Træner XGBoost model for {horizon_name} forecast...")
        
        # Prepare data - flatten if needed
        close_prices = data["Close"].dropna().values
        if close_prices.ndim > 1:
            close_prices = close_prices.flatten()
        
        if len(close_prices) < window + horizon + 10:
            print(f"{Fore.YELLOW}⚠️  For lidt data til ML")
            return None
        
        X, y = [], []
        for i in range(window, len(close_prices) - horizon + 1):
            X.append(close_prices[i-window:i])
            y.append(close_prices[i+horizon-1])
        
        X, y = np.array(X), np.array(y)
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0  # Suppress output
        )
        xgb_model.fit(X, y)
        
        # Make prediction
        last_window = close_prices[-window:].reshape(1, -1)
        forecast = float(xgb_model.predict(last_window)[0])
        current_price = float(close_prices[-1])
        change_pct = ((forecast - current_price) / current_price) * 100
        
        print(f"{Fore.GREEN}✅ XGBoost trænet på {len(X)} samples")
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": change_pct,
            "horizon": horizon,
            "horizon_name": horizon_name
        }
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i XGBoost forecast: {e}")
        import traceback
        traceback.print_exc()
        return None

def ml_forecast_prophet(data, horizon=1):
    """
    Laver Prophet (Facebook) baseret forecast.
    Prophet er god til trends og seasonality.
    
    Args:
        data: pandas DataFrame med aktiedata
        horizon: Forecast horisont (1=1 dag, 5=5 dage, 22=1 måned)
    
    Returns:
        Dict med forecasts eller None ved fejl
    """
    try:
        horizon_name = {1: "1 dag", 5: "5 dage", 22: "1 måned"}.get(horizon, f"{horizon} dage")
        print(f"{Fore.CYAN}📈 Træner Prophet model for {horizon_name} forecast...")
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data['Close'].values.flatten() if data['Close'].values.ndim > 1 else data['Close'].values
        })
        
        if len(prophet_data) < 30:
            print(f"{Fore.YELLOW}⚠️  For lidt data til Prophet (behøver min. 30 dage)")
            return None
        
        # Initialize and fit model (suppress output)
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Not enough data usually
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=horizon)
        forecast_df = model.predict(future)
        
        # Get the forecast for the target horizon
        forecast = float(forecast_df['yhat'].iloc[-1])
        current_price = float(prophet_data['y'].iloc[-1])
        change_pct = ((forecast - current_price) / current_price) * 100
        
        print(f"{Fore.GREEN}✅ Prophet trænet på {len(prophet_data)} samples")
        
        return {
            "forecast": forecast,
            "current": current_price,
            "change_pct": change_pct,
            "horizon": horizon,
            "horizon_name": horizon_name,
            "trend": forecast_df['trend'].iloc[-1],
            "lower_bound": forecast_df['yhat_lower'].iloc[-1],
            "upper_bound": forecast_df['yhat_upper'].iloc[-1]
        }
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i Prophet forecast: {e}")
        import traceback
        traceback.print_exc()
        return None

def ml_forecast_ensemble(data, window=30, horizon=1, weights=None):
    """
    Ensemble forecast der kombinerer LSTM, Random Forest, XGBoost og Prophet.
    Bruger weighted average for bedre accuracy.
    
    Args:
        data: pandas DataFrame med aktiedata
        window: Antal dage at bruge som input (for LSTM, RF, XGBoost)
        horizon: Forecast horisont (1=1 dag, 5=5 dage, 22=1 måned)
        weights: Dict med vægte for hver model {'lstm': 0.25, 'rf': 0.25, 'xgb': 0.25, 'prophet': 0.25}
                 Hvis None, bruges lige vægte
    
    Returns:
        Dict med ensemble forecast og individuelle forecasts
    """
    try:
        horizon_name = {1: "1 dag", 5: "5 dage", 22: "1 måned"}.get(horizon, f"{horizon} dage")
        print(f"{Fore.MAGENTA}🎯 Træner Ensemble model for {horizon_name} forecast...")
        
        # Default weights hvis ikke specificeret
        if weights is None:
            weights = {'lstm': 0.2, 'rf': 0.3, 'xgb': 0.3, 'prophet': 0.2}
        
        forecasts = {}
        successful_models = []
        
        # Run all models (suppress output)
        import sys
        import io
        
        # Try LSTM
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            lstm_result = ml_forecast(data, window=window, epochs=20, horizon=horizon)
            sys.stdout = old_stdout
            if lstm_result:
                forecasts['lstm'] = lstm_result['forecast']
                successful_models.append('lstm')
        except:
            pass
        
        # Try Random Forest
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            rf_result = ml_forecast_rf(data, window=window, horizon=horizon)
            sys.stdout = old_stdout
            if rf_result:
                forecasts['rf'] = rf_result['forecast']
                successful_models.append('rf')
        except:
            pass
        
        # Try XGBoost
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            xgb_result = ml_forecast_xgboost(data, window=window, horizon=horizon)
            sys.stdout = old_stdout
            if xgb_result:
                forecasts['xgb'] = xgb_result['forecast']
                successful_models.append('xgb')
        except:
            pass
        
        # Try Prophet
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            prophet_result = ml_forecast_prophet(data, horizon=horizon)
            sys.stdout = old_stdout
            if prophet_result:
                forecasts['prophet'] = prophet_result['forecast']
                successful_models.append('prophet')
        except:
            pass
        
        if len(forecasts) == 0:
            print(f"{Fore.RED}❌ Ingen modeller kunne generere forecast")
            return None
        
        # Calculate weighted ensemble forecast
        # Normalize weights for successful models
        total_weight = sum([weights.get(model, 0) for model in successful_models])
        ensemble_forecast = sum([
            forecasts[model] * (weights.get(model, 0) / total_weight) 
            for model in successful_models
        ])
        
        current_price = float(data['Close'].iloc[-1])
        change_pct = ((ensemble_forecast - current_price) / current_price) * 100
        
        print(f"{Fore.GREEN}✅ Ensemble trænet med {len(forecasts)} modeller: {', '.join(successful_models)}")
        
        return {
            "forecast": ensemble_forecast,
            "current": current_price,
            "change_pct": change_pct,
            "horizon": horizon,
            "horizon_name": horizon_name,
            "models_used": successful_models,
            "individual_forecasts": forecasts,
            "weights": {model: weights.get(model, 0) / total_weight for model in successful_models}
        }
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i Ensemble forecast: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_model_metrics(actual, predicted):
    """
    Beregn performance metrics for ML model
    
    Args:
        actual: Array af faktiske priser
        predicted: Array af forudsagte priser
    
    Returns:
        Dict med MAE, RMSE, MAPE metrics
    """
    try:
        # Konverter til numpy arrays
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        # Fjern NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0 or len(predicted) == 0:
            return None
        
        # Mean Absolute Error
        mae = mean_absolute_error(actual, predicted)
        
        # Root Mean Square Error
        mse = mean_squared_error(actual, predicted)
        rmse = sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R² Score (coefficient of determination)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Win Rate (hvor mange gange var trend korrekt)
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(predicted[1:] - actual[:-1])
        correct_direction = np.sum(actual_direction == predicted_direction)
        win_rate = (correct_direction / len(actual_direction)) * 100 if len(actual_direction) > 0 else 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Win_Rate': win_rate,
            'Sample_Size': len(actual)
        }
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i metrics beregning: {e}")
        return None

def backtest_model(data, model_type='rf', window=30, horizon=1, test_periods=30, progress_callback=None):
    """
    Backtest en ML model på historiske data
    
    Args:
        data: DataFrame med historiske priser
        model_type: 'lstm' eller 'rf'
        window: Window size for model
        horizon: Forecast horizon (1, 5, eller 22 dage)
        test_periods: Antal perioder at teste
        progress_callback: Optional callback function(current, total, message)
    
    Returns:
        Dict med backtest results
    """
    try:
        print(f"{Fore.CYAN}🔄 Backtester {model_type.upper()} model...")
        print(f"   Window: {window}, Horizon: {horizon}, Test periods: {test_periods}")
        
        if progress_callback:
            progress_callback(0, test_periods, "Starter backtest...")
        
        # Tjek at vi har nok data
        min_data_needed = window + horizon + test_periods + 10
        if len(data) < min_data_needed:
            print(f"{Fore.YELLOW}⚠️  For lidt data til backtest (behøver {min_data_needed} dage)")
            return None
        
        actual_prices = []
        predicted_prices = []
        forecast_dates = []
        
        # Rolling window backtest
        for i in range(test_periods):
            # Progress indicator
            if progress_callback:
                progress_callback(i + 1, test_periods, f"Tester periode {i+1}/{test_periods}...")
            elif i % 5 == 0:
                print(f"{Fore.CYAN}   Progress: {i}/{test_periods} periods tested...")
            
            # Tag data frem til dette punkt
            end_idx = len(data) - test_periods + i
            if end_idx < window + horizon:
                continue
            
            train_data = data.iloc[:end_idx].copy()
            
            # Lav forecast (uden print statements)
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            if model_type == 'rf':
                result = ml_forecast_rf(train_data, window=window, horizon=horizon)
            elif model_type == 'lstm':
                result = ml_forecast(train_data, window=window, epochs=20, horizon=horizon)
            elif model_type == 'xgboost':
                result = ml_forecast_xgboost(train_data, window=window, horizon=horizon)
            elif model_type == 'prophet':
                result = ml_forecast_prophet(train_data, horizon=horizon)
            elif model_type == 'ensemble':
                result = ml_forecast_ensemble(train_data, window=window, horizon=horizon)
            else:
                sys.stdout = old_stdout
                continue
            
            sys.stdout = old_stdout
            
            if result and 'forecast' in result:
                # Sammenlign med faktisk pris
                actual_idx = end_idx + horizon - 1
                if actual_idx < len(data):
                    # Håndter MultiIndex
                    close_price = data['Close'].iloc[actual_idx]
                    if isinstance(close_price, pd.Series):
                        actual_price = float(close_price.iloc[0])
                    else:
                        actual_price = float(close_price)
                    predicted_price = result['forecast']
                    
                    actual_prices.append(actual_price)
                    predicted_prices.append(predicted_price)
                    forecast_dates.append(data.index[actual_idx])
        
        if len(actual_prices) < 3:
            print(f"{Fore.YELLOW}⚠️  Ikke nok forecasts til backtest")
            return None
        
        # Beregn metrics
        if progress_callback:
            progress_callback(test_periods, test_periods, "Beregner metrics...")
        
        metrics = calculate_model_metrics(actual_prices, predicted_prices)
        
        if metrics:
            print(f"{Fore.GREEN}✅ Backtest komplet!")
            print(f"   MAE: ${metrics['MAE']:.2f}")
            print(f"   RMSE: ${metrics['RMSE']:.2f}")
            print(f"   MAPE: {metrics['MAPE']:.2f}%")
            print(f"   Win Rate: {metrics['Win_Rate']:.1f}%")
            print(f"   R²: {metrics['R2']:.3f}")
        
        return {
            'metrics': metrics,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices,
            'forecast_dates': forecast_dates,
            'test_periods': len(actual_prices)
        }
        
    except Exception as e:
        print(f"{Fore.RED}❌ Fejl i backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

def log_analysis(symbol, analysis_type, data, results=None):
    """
    Logger en analyse til analyse_log.json
    
    Args:
        symbol: Aktiesymbol
        analysis_type: "technical" eller "ml_forecast"
        data: DataFrame med aktiedata og indikatorer
        results: Dict med ML resultater (optional)
    """
    try:
        import datetime
        
        log_file = "analyse_log.json"
        
        # Load existing log eller opret ny
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = {"analyses": []}
        
        # Opret log entry
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "symbol": symbol,
            "type": analysis_type,
            "current_price": data["Close"].iloc[-1].item()
        }
        
        # Tilføj teknisk analyse data
        if analysis_type == "technical":
            if "RSI" in data.columns:
                entry["rsi"] = float(data["RSI"].iloc[-1])
            if "SMA50" in data.columns:
                entry["sma50"] = float(data["SMA50"].iloc[-1])
            if "SMA200" in data.columns:
                entry["sma200"] = float(data["SMA200"].iloc[-1])
            if "MACD" in data.columns:
                entry["macd"] = float(data["MACD"].iloc[-1])
            if "ATR" in data.columns:
                entry["atr"] = float(data["ATR"].iloc[-1])
        
        # Tilføj ML forecast data
        if analysis_type == "ml_forecast" and results:
            entry["ml_results"] = results
        
        # Gem log
        log_data["analyses"].append(entry)
        
        # Behold kun de sidste 100 analyser
        if len(log_data["analyses"]) > 100:
            log_data["analyses"] = log_data["analyses"][-100:]
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        # Silent fail - logging skal ikke bryde programmet
        pass

def ask_reflection_question(questions):
    """Stiller refleksionsspørgsmål til brugeren."""
    if questions and len(questions) > 0:
        import random
        question = random.choice(questions)
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}🤔 Refleksion:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{question}")
        response = input(f"{Fore.CYAN}Din tanke: {Fore.WHITE}").strip()
        if response:
            print(f"{Fore.GREEN}💡 Godt tænkt! Husk altid at tænke kritisk.")

def mode_analyse(config):
    """Kører analyse mode."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}🔎 TEKNISK ANALYSE MODE{Style.RESET_ALL}\n")
    
    symbol = input(f"{Fore.YELLOW}Indtast aktiesymbol (fx AAPL, MSFT): {Fore.WHITE}").strip().upper()
    if not symbol:
        print(f"{Fore.RED}❌ Intet symbol indtastet.")
        return
    
    data = hent_data(symbol, period=config["default_settings"]["data_period"])
    if data is None:
        return
    
    data = teknisk_analyse(data, symbol, config)
    if data is None:
        return
    
    # Log analysen
    log_analysis(symbol, "technical", data)
    
    vis_graf_choice = input(f"\n{Fore.YELLOW}Vil du se grafen? (y/n): {Fore.WHITE}").strip().lower()
    if vis_graf_choice == 'y':
        vis_graf(data, symbol)
    
    # Refleksionsspørgsmål
    if "reflection_questions" in config and "analyse" in config["reflection_questions"]:
        ask_reflection_question(config["reflection_questions"]["analyse"])

def mode_laer(config):
    """Kører læringsmodus."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📘 LÆRINGSMODUS{Style.RESET_ALL}\n")
    
    print(f"{Fore.GREEN}Velkommen til læringsmodus! Her lærer du om tekniske indikatorer.\n")
    
    print(f"{Fore.YELLOW}═══ RSI (Relative Strength Index) ═══{Fore.WHITE}")
    print("RSI måler momentum i kursbevægelser på en skala fra 0-100.")
    print("• Over 70: Aktien kan være overkøbt (potentiel korrektion)")
    print("• Under 30: Aktien kan være oversolgt (potentiel recovery)")
    print("• 30-70: Neutral zone")
    
    print(f"\n{Fore.YELLOW}═══ SMA (Simple Moving Average) ═══{Fore.WHITE}")
    print("SMA udglatter kursdata og hjælper med at identificere trends.")
    print("• SMA50: Kort-mellem sigt trend (50 dages gennemsnit)")
    print("• SMA200: Lang sigt trend (200 dages gennemsnit)")
    print("• Golden Cross: SMA50 > SMA200 (bullish signal)")
    print("• Death Cross: SMA50 < SMA200 (bearish signal)")
    
    symbol = input(f"\n{Fore.YELLOW}Vil du se et eksempel? Indtast symbol (eller tryk Enter for at springe over): {Fore.WHITE}").strip().upper()
    
    if symbol:
        data = hent_data(symbol, period=config["default_settings"]["data_period"])
        if data:
            data = teknisk_analyse(data, symbol)
            if data:
                vis_graf(data, symbol)
    
    # Refleksionsspørgsmål
    if "reflection_questions" in config and "laer" in config["reflection_questions"]:
        ask_reflection_question(config["reflection_questions"]["laer"])

def mode_ml(config):
    """Kører ML forecast mode."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}🤖 ML FORECAST MODE{Style.RESET_ALL}\n")
    
    symbol = input(f"{Fore.YELLOW}Indtast aktiesymbol (fx AAPL, MSFT): {Fore.WHITE}").strip().upper()
    if not symbol:
        print(f"{Fore.RED}❌ Intet symbol indtastet.")
        return
    
    # Vælg model type
    print(f"\n{Fore.CYAN}Vælg ML model:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}1. {Fore.YELLOW}LSTM{Fore.WHITE} (Neural Network - god til sekvenser)")
    print(f"{Fore.WHITE}2. {Fore.YELLOW}Random Forest{Fore.WHITE} (Ensemble model - hurtigere)")
    print(f"{Fore.WHITE}3. {Fore.YELLOW}Begge{Fore.WHITE} (sammenlign resultater)")
    
    model_choice = input(f"{Fore.CYAN}Vælg (1-3, eller Enter for LSTM): {Fore.WHITE}").strip()
    
    # Vælg forecast horisont
    print(f"\n{Fore.CYAN}Vælg forecast horisont:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}1. {Fore.YELLOW}1 dag{Fore.WHITE} (kort sigt)")
    print(f"{Fore.WHITE}2. {Fore.YELLOW}5 dage{Fore.WHITE} (mellem sigt)")
    print(f"{Fore.WHITE}3. {Fore.YELLOW}1 måned (21 dage){Fore.WHITE} (lang sigt)")
    
    horizon_choice = input(f"{Fore.CYAN}Vælg (1-3, eller Enter for 1 dag): {Fore.WHITE}").strip()
    horizon_map = {"1": 1, "2": 5, "3": 21, "": 1}
    horizon = horizon_map.get(horizon_choice, 1)
    
    data = hent_data(symbol, period=config["default_settings"]["data_period"])
    if data is None:
        return
    
    # Run valgte modeller
    lstm_result = None
    rf_result = None
    
    if model_choice in ["1", "", "3"]:
        lstm_result = ml_forecast(data, 
                            window=config["default_settings"]["ml_window"],
                            epochs=config["default_settings"]["ml_epochs"],
                            horizon=horizon)
    
    if model_choice in ["2", "3"]:
        rf_result = ml_forecast_rf(data,
                            window=config["default_settings"]["ml_window"],
                            horizon=horizon)
    
    # Vis resultater
    if lstm_result or rf_result:
        horizon_name = (lstm_result or rf_result)["horizon_name"]
        current = (lstm_result or rf_result)["current"]
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}🔮 ML FORECAST RESULTAT ({horizon_name}){Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─' * 50}")
        print(f"{Fore.YELLOW}Nuværende kurs: {Fore.WHITE}${current:.2f}")
        print(f"{Fore.WHITE}{'─' * 50}")
        
        # LSTM resultat
        if lstm_result:
            forecast = lstm_result["forecast"]
            change = lstm_result["change_pct"]
            print(f"\n{Fore.MAGENTA}🧠 LSTM Neural Network:{Style.RESET_ALL}")
            print(f"  Forecast:  {Fore.WHITE}${forecast:.2f}")
            if change > 0:
                print(f"  Ændring:   {Fore.GREEN}+{change:.2f}% ↗")
            else:
                print(f"  Ændring:   {Fore.RED}{change:.2f}% ↘")
        
        # Random Forest resultat
        if rf_result:
            forecast = rf_result["forecast"]
            change = rf_result["change_pct"]
            print(f"\n{Fore.GREEN}🌲 Random Forest:{Style.RESET_ALL}")
            print(f"  Forecast:  {Fore.WHITE}${forecast:.2f}")
            if change > 0:
                print(f"  Ændring:   {Fore.GREEN}+{change:.2f}% ↗")
            else:
                print(f"  Ændring:   {Fore.RED}{change:.2f}% ↘")
        
        # Sammenligning
        if lstm_result and rf_result:
            diff = abs(lstm_result["forecast"] - rf_result["forecast"])
            diff_pct = (diff / current) * 100
            print(f"\n{Fore.CYAN}📊 Model Sammenligning:{Style.RESET_ALL}")
            print(f"  Forskel:   {Fore.WHITE}${diff:.2f} ({diff_pct:.2f}%)")
            if diff_pct < 1:
                print(f"  {Fore.GREEN}✅ Modellerne er enige (< 1% forskel)")
            elif diff_pct < 3:
                print(f"  {Fore.YELLOW}⚠️  Moderat uenighed (1-3% forskel)")
            else:
                print(f"  {Fore.RED}⚠️  Stor uenighed (> 3% forskel)")
        
        print(f"{Fore.WHITE}{'─' * 50}")
        print(f"{Fore.YELLOW}💡 Længere horisonter er mere usikre!")
        print(f"{Fore.YELLOW}💡 Sammenlign altid flere modeller for bedre indsigt!")
        
        # Log ML forecast
        ml_log_results = {}
        if lstm_result:
            ml_log_results["lstm"] = lstm_result
        if rf_result:
            ml_log_results["random_forest"] = rf_result
        log_analysis(symbol, "ml_forecast", data, ml_log_results)
    
    # Refleksionsspørgsmål
    if "reflection_questions" in config and "ml" in config["reflection_questions"]:
        ask_reflection_question(config["reflection_questions"]["ml"])

def show_watchlist(config):
    """Viser watchlist."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📋 DIN WATCHLIST{Style.RESET_ALL}\n")
    
    if "watchlist" in config and len(config["watchlist"]) > 0:
        for idx, symbol in enumerate(config["watchlist"], 1):
            print(f"{Fore.GREEN}{idx}.{Fore.WHITE} {symbol}")
        
        choice = input(f"\n{Fore.YELLOW}Vil du analysere en af disse? (indtast nummer eller 0 for at gå tilbage): {Fore.WHITE}").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(config["watchlist"]):
                symbol = config["watchlist"][idx]
                data = hent_data(symbol, period=config["default_settings"]["data_period"])
                if data:
                    data = teknisk_analyse(data, symbol)
                    if data:
                        vis_graf_choice = input(f"\n{Fore.YELLOW}Vil du se grafen? (y/n): {Fore.WHITE}").strip().lower()
                        if vis_graf_choice == 'y':
                            vis_graf(data, symbol)
    else:
        print(f"{Fore.YELLOW}Watchlist er tom. Tilføj aktier i config.json")

def show_settings(config):
    """Viser indstillinger."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}⚙️  INDSTILLINGER{Style.RESET_ALL}\n")
    
    settings = config.get("default_settings", {})
    print(f"{Fore.YELLOW}Data periode:     {Fore.WHITE}{settings.get('data_period', '6mo')}")
    print(f"{Fore.YELLOW}Data interval:    {Fore.WHITE}{settings.get('data_interval', '1d')}")
    print(f"{Fore.YELLOW}ML window:        {Fore.WHITE}{settings.get('ml_window', 30)} dage")
    print(f"{Fore.YELLOW}ML epochs:        {Fore.WHITE}{settings.get('ml_epochs', 3)}")
    
    enabled = config.get("enabled_indicators", {})
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Aktiverede Indikatorer:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}RSI:            {Fore.GREEN if enabled.get('rsi', True) else Fore.RED}{'✓' if enabled.get('rsi', True) else '✗'}")
    print(f"{Fore.YELLOW}SMA:            {Fore.GREEN if enabled.get('sma', True) else Fore.RED}{'✓' if enabled.get('sma', True) else '✗'}")
    print(f"{Fore.YELLOW}MACD:           {Fore.GREEN if enabled.get('macd', True) else Fore.RED}{'✓' if enabled.get('macd', True) else '✗'}")
    print(f"{Fore.YELLOW}Bollinger Bands:{Fore.GREEN if enabled.get('bollinger_bands', True) else Fore.RED}{'✓' if enabled.get('bollinger_bands', True) else '✗'}")
    print(f"{Fore.YELLOW}ATR:            {Fore.GREEN if enabled.get('atr', True) else Fore.RED}{'✓' if enabled.get('atr', True) else '✗'}")
    
    print(f"\n{Fore.CYAN}💡 Rediger config.json for at ændre indstillinger.")

def mode_graf_plotly(config):
    """Vis interaktiv Plotly graf."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📊 INTERAKTIV GRAF (PLOTLY){Style.RESET_ALL}\n")
    
    symbol = input(f"{Fore.YELLOW}Indtast aktiesymbol (fx AAPL, MSFT, TSLA): {Fore.WHITE}").strip().upper()
    
    if not symbol:
        print(f"{Fore.RED}❌ Intet symbol indtastet.")
        return
    
    # Hent data
    data = hent_data(symbol, period=config["default_settings"]["data_period"])
    if not data:
        return
    
    # Beregn indikatorer (uden at printe)
    # Vi tilføjer indikatorerne manuelt
    data["RSI"] = calculate_rsi(data)
    data["SMA50"] = calculate_sma(data, periods=50)
    data["SMA200"] = calculate_sma(data, periods=200)
    
    macd_line, signal_line, histogram = calculate_macd(data)
    data["MACD_Line"] = macd_line
    data["MACD_Signal"] = signal_line
    data["MACD_Hist"] = histogram
    
    upper, middle, lower = calculate_bollinger_bands(data)
    data["BB_Upper"] = upper
    data["BB_Middle"] = middle
    data["BB_Lower"] = lower
    
    data["ATR"] = calculate_atr(data)
    
    print(f"\n{Fore.CYAN}🎨 Genererer interaktiv graf...")
    print(f"{Fore.YELLOW}💡 Grafen åbnes i din browser med zoom, pan og hover funktionalitet!")
    
    # Vis plotly graf
    vis_graf_plotly(data, symbol)

def mode_html_rapport(config):
    """Generer HTML rapport."""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📄 GENERER HTML RAPPORT{Style.RESET_ALL}\n")
    
    symbol = input(f"{Fore.YELLOW}Indtast aktiesymbol (fx AAPL, MSFT, TSLA): {Fore.WHITE}").strip().upper()
    
    if not symbol:
        print(f"{Fore.RED}❌ Intet symbol indtastet.")
        return
    
    # Hent data
    data = hent_data(symbol, period=config["default_settings"]["data_period"])
    if not data:
        return
    
    # Beregn indikatorer
    print(f"\n{Fore.CYAN}📊 Beregner tekniske indikatorer...")
    data["RSI"] = calculate_rsi(data)
    data["SMA50"] = calculate_sma(data, periods=50)
    data["SMA200"] = calculate_sma(data, periods=200)
    
    macd_line, signal_line, histogram = calculate_macd(data)
    data["MACD_Line"] = macd_line
    data["MACD_Signal"] = signal_line
    data["MACD_Hist"] = histogram
    
    upper, middle, lower = calculate_bollinger_bands(data)
    data["BB_Upper"] = upper
    data["BB_Middle"] = middle
    data["BB_Lower"] = lower
    
    data["ATR"] = calculate_atr(data)
    print(f"{Fore.GREEN}✅ Indikatorer beregnet!")
    
    # Spørg om ML forecast skal inkluderes
    include_ml = input(f"\n{Fore.YELLOW}Vil du inkludere ML forecasts i rapporten? (y/n): {Fore.WHITE}").strip().lower()
    
    ml_results = None
    if include_ml == 'y':
        print(f"\n{Fore.CYAN}🤖 Genererer ML forecasts...")
        ml_results = []
        
        for horizon, name in [(1, "1 dag"), (5, "5 dage"), (22, "22 dage")]:
            print(f"\n{Fore.CYAN}Forecast for {name}...")
            
            # LSTM
            lstm_result = ml_forecast(data, 
                                     window=config["default_settings"]["ml_window"],
                                     epochs=config["default_settings"]["ml_epochs"],
                                     horizon=horizon)
            if lstm_result:
                lstm_result['model'] = 'LSTM'
                ml_results.append(lstm_result)
            
            # Random Forest
            rf_result = ml_forecast_rf(data, 
                                       window=config["default_settings"]["ml_window"],
                                       horizon=horizon)
            if rf_result:
                rf_result['model'] = 'Random Forest'
                ml_results.append(rf_result)
    
    # Generer HTML rapport
    print(f"\n{Fore.CYAN}📝 Genererer HTML rapport...")
    filename = generer_html_rapport(symbol, data, ml_results=ml_results)
    
    if filename:
        # Spørg om rapporten skal åbnes
        open_report = input(f"\n{Fore.YELLOW}Vil du åbne rapporten nu? (y/n): {Fore.WHITE}").strip().lower()
        if open_report == 'y':
            import webbrowser
            import os
            webbrowser.open('file://' + os.path.realpath(filename))
            print(f"{Fore.GREEN}✅ Rapporten åbnes i din browser!")

def interactive_mode():
    """Hovedloop for interaktiv mode."""
    config = load_config()
    
    print_header()
    print(f"{Fore.GREEN}Velkommen til Trading Mentor Agent! 🎉")
    print(f"{Fore.WHITE}Din personlige mentor til at lære om aktiemarkedet.\n")
    
    while True:
        print_menu()
        choice = input(f"{Fore.CYAN}Vælg en option (1-8): {Fore.WHITE}").strip()
        
        if choice == "1":
            mode_analyse(config)
        elif choice == "2":
            mode_laer(config)
        elif choice == "3":
            mode_ml(config)
        elif choice == "4":
            mode_graf_plotly(config)
        elif choice == "5":
            mode_html_rapport(config)
        elif choice == "6":
            show_watchlist(config)
        elif choice == "7":
            show_settings(config)
        elif choice == "8":
            print(f"\n{Fore.CYAN}👋 Tak for i dag! Held og lykke med din trading journey!")
            print(f"{Fore.YELLOW}💡 Husk: Invester kun hvad du har råd til at tabe.{Style.RESET_ALL}\n")
            break
        else:
            print(f"{Fore.RED}❌ Ugyldig valg. Prøv igen.")
        
        input(f"\n{Fore.CYAN}Tryk Enter for at fortsætte...{Style.RESET_ALL}")

# ========== V2 MODEL WRAPPERS ==========

def train_and_save_rf_v2(data, symbol, n_estimators=200, max_depth=15, window=30, horizon=1, use_features=True):
    """
    Wrapper for RF v2 with enhanced features.
    Imports and calls ml_training_enhanced.py
    """
    from ml_training_enhanced import train_and_save_rf_v2 as train_rf_v2_impl
    return train_rf_v2_impl(data, symbol, n_estimators, max_depth, window, horizon, use_features)


def train_and_save_xgboost_v2(data, symbol, n_estimators=300, max_depth=8, learning_rate=0.05, 
                               window=30, horizon=1, use_features=True):
    """
    Wrapper for XGBoost v2 with enhanced features.
    Imports and calls ml_training_enhanced.py
    """
    from ml_training_enhanced import train_and_save_xgboost_v2 as train_xgb_v2_impl
    return train_xgb_v2_impl(data, symbol, n_estimators, max_depth, learning_rate, 
                             window, horizon, use_features)


def train_and_save_lstm_v2(data, symbol, sequence_length=30, lstm_units=[32, 16], 
                            epochs=100, use_attention=True, n_features=20):
    """
    Wrapper for LSTM v2 Tuned with Attention.
    Imports and calls lstm_tuned.py
    """
    from lstm_tuned import train_and_save_lstm_tuned
    return train_and_save_lstm_tuned(data, symbol, sequence_length, lstm_units, 
                                    epochs, 32, use_attention, n_features)


def get_available_model_versions():
    """
    Returns list of available model versions for UI selection.
    """
    return {
        'rf': {
            'v1': {
                'name': 'Random Forest v1 (Legacy)',
                'description': 'Original RF with basic price features',
                'function': train_and_save_rf
            },
            'v2': {
                'name': 'Random Forest v2 (Enhanced)',
                'description': 'RF with 67 technical indicators',
                'function': train_and_save_rf_v2,
                'recommended': True
            }
        },
        'xgboost': {
            'v1': {
                'name': 'XGBoost v1 (Legacy)',
                'description': 'Original XGBoost with basic features',
                'function': train_and_save_xgboost
            },
            'v2': {
                'name': 'XGBoost v2 (Enhanced)',
                'description': 'XGBoost with 67 indicators + early stopping',
                'function': train_and_save_xgboost_v2,
                'recommended': True
            }
        },
        'lstm': {
            'v1': {
                'name': 'LSTM v1 (Simple)',
                'description': 'Basic 2-layer LSTM',
                'function': train_and_save_lstm
            },
            'v2': {
                'name': 'LSTM v2 (Bi-directional + Attention)',
                'description': 'Advanced Bi-LSTM with Attention mechanism',
                'function': train_and_save_lstm_v2,
                'recommended': True
            }
        },
        'prophet': {
            'v1': {
                'name': 'Prophet (Time Series)',
                'description': 'Facebook Prophet for trend forecasting',
                'function': train_and_save_prophet
            }
        }
    }


# ---------- Main Entry Point ----------

def main():
    """Main entry point - starter interaktiv mode."""
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️  Program afbrudt af bruger.")
        print(f"{Fore.CYAN}👋 Vi ses næste gang!{Style.RESET_ALL}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Uventet fejl: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()
