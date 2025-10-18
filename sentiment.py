"""
Sentiment Analysis Module
Henter psykologiske signaler fra markedet (Fear & Greed + News Sentiment)
"""

import requests
import yfinance as yf
import pandas as pd
from typing import Dict, Optional
import os


def get_fear_greed() -> Dict:
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
        Dict med value (0-100), classification, sentiment og components
    """
    try:
        # Fetch all required data
        sp500_data = yf.download('^GSPC', period='1y', progress=False)
        vix_data = yf.download('^VIX', period='5d', progress=False)
        gold_data = yf.download('GC=F', period='1mo', progress=False)
        treasury_data = yf.download('^TNX', period='1mo', progress=False)
        
        if sp500_data.empty or vix_data.empty:
            return {
                "value": 50,
                "classification": "Neutral",
                "source": "Fallback",
                "error": "Could not fetch market data"
            }
        
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
            momentum_score = 50 + min(max(momentum_pct * 1.0, -40), 40)
            components.append(('Momentum', momentum_score))
        
        # 2. Stock Price Strength (distance from 52-week high) - 20%
        if len(sp500_data) >= 252:
            high_52w = sp500_data['High'].tail(252).max()
            current_price = sp500_data['Close'].iloc[-1]
            distance_from_high = ((current_price - high_52w) / high_52w) * 100
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
        
        # Apply CNN-aligned calibration
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
            "value": fg_value,
            "classification": level,
            "interpretation": interpretation,
            "sentiment": sentiment,
            "components": components_dict,
            "source": "CNN-Aligned Multi-Factor",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        return {
            "value": 50,
            "classification": "Neutral",
            "source": "Error",
            "error": str(e)
        }


def get_fear_greed_interpretation(value: int) -> str:
    """
    Konverterer Fear & Greed value til tekst.
    
    Args:
        value: Fear & Greed værdi (0-100)
    
    Returns:
        Interpretation string
    """
    if value >= 75:
        return "Ekstrem Grådighed - Markedet kan være overophedet"
    elif value >= 55:
        return "Grådighed - Bullish stemning"
    elif value >= 45:
        return "Neutral - Afventende marked"
    elif value >= 25:
        return "Frygt - Bearish stemning"
    else:
        return "Ekstrem Frygt - Potentiel buying opportunity"


def get_news_sentiment(symbol: str, api_key: Optional[str] = None) -> Dict:
    """
    Henter nyheds-sentiment fra Finnhub API.
    
    Args:
        symbol: Ticker symbol
        api_key: Finnhub API key (eller sættes via FINNHUB_API_KEY env var)
    
    Returns:
        Dict med sentiment score (-1 til 1) og detaljer
    """
    # Tjek API key
    if api_key is None:
        api_key = os.getenv("FINNHUB_API_KEY")
    
    if not api_key:
        return {
            "symbol": symbol,
            "score": 0,
            "interpretation": "Neutral",
            "error": "No API key provided. Set FINNHUB_API_KEY or pass api_key parameter."
        }
    
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Finnhub returnerer sentiment score
        sentiment_data = data.get("sentiment", {})
        score = sentiment_data.get("score", 0)
        
        return {
            "symbol": symbol,
            "score": score,
            "interpretation": interpret_news_sentiment(score),
            "buzz_score": sentiment_data.get("buzz", 0),
            "source": "Finnhub"
        }
    
    except Exception as e:
        return {
            "symbol": symbol,
            "score": 0,
            "interpretation": "Neutral",
            "error": str(e)
        }


def interpret_news_sentiment(score: float) -> str:
    """
    Konverterer news sentiment score til tekst.
    
    Args:
        score: Sentiment score (-1 til 1)
    
    Returns:
        Interpretation string
    """
    if score > 0.3:
        return "Meget Positivt - Stærkt momentum i nyhedsflow"
    elif score > 0.1:
        return "Positivt - Optimistisk nyhedsflow"
    elif score > -0.1:
        return "Neutralt - Blandet nyhedsflow"
    elif score > -0.3:
        return "Negativt - Pessimistisk nyhedsflow"
    else:
        return "Meget Negativt - Stor risiko for fald"


def get_vix_fear_gauge() -> Dict:
    """
    Henter VIX (Volatility Index) som alternativ fear gauge.
    VIX > 20 indikerer frygt, VIX > 30 indikerer panik.
    
    Returns:
        Dict med VIX værdi og interpretation
    """
    try:
        import yfinance as yf
        
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period="1d")
        
        if not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            
            if current_vix > 30:
                interpretation = "Panik - Høj volatilitet forventet"
            elif current_vix > 20:
                interpretation = "Frygt - Øget markedsuro"
            elif current_vix > 15:
                interpretation = "Moderat - Normal volatilitet"
            else:
                interpretation = "Komplacent - Lav volatilitet"
            
            return {
                "value": round(current_vix, 2),
                "interpretation": interpretation,
                "source": "VIX (CBOE Volatility Index)"
            }
    
    except Exception as e:
        return {
            "value": None,
            "interpretation": "Data ikke tilgængelig",
            "error": str(e)
        }
    
    return {
        "value": None,
        "interpretation": "Data ikke tilgængelig"
    }


def get_combined_sentiment(symbol: str, api_key: Optional[str] = None) -> Dict:
    """
    Kombinerer alle sentiment-kilder til ét samlet billede.
    
    Args:
        symbol: Ticker symbol
        api_key: Finnhub API key (optional)
    
    Returns:
        Dict med alle sentiment metrics
    """
    fear_greed = get_fear_greed()
    news = get_news_sentiment(symbol, api_key)
    vix = get_vix_fear_gauge()
    
    # Beregn samlet sentiment score (0-100)
    # 40% Fear/Greed, 40% News, 20% VIX
    fg_component = fear_greed["value"] * 0.4
    
    # Konverter news score (-1 til 1) til 0-100
    news_component = ((news["score"] + 1) / 2 * 100) * 0.4
    
    # Konverter VIX til 0-100 (inverse - høj VIX = lav score)
    if vix["value"] is not None:
        vix_component = max(0, (50 - vix["value"]) / 50 * 100) * 0.2
    else:
        vix_component = 50 * 0.2  # Neutral hvis ingen data
    
    combined_score = int(fg_component + news_component + vix_component)
    
    return {
        "symbol": symbol,
        "combined_score": combined_score,
        "fear_greed": fear_greed,
        "news_sentiment": news,
        "vix": vix,
        "interpretation": get_combined_interpretation(combined_score)
    }


def get_combined_interpretation(score: int) -> str:
    """
    Interpreterer kombineret sentiment score.
    
    Args:
        score: Combined sentiment (0-100)
    
    Returns:
        Interpretation string
    """
    if score >= 75:
        return "Meget Bullish - Men pas på FOMO og hype"
    elif score >= 60:
        return "Bullish - Positivt momentum"
    elif score >= 40:
        return "Neutral - Afventende marked"
    elif score >= 25:
        return "Bearish - Negativ stemning"
    else:
        return "Meget Bearish - Men måske buying opportunity"


if __name__ == "__main__":
    # Test (uden API key - vil bruge mock data)
    print("\n=== Fear & Greed ===")
    fg = get_fear_greed()
    print(f"Value: {fg['value']}/100")
    print(f"Classification: {fg['classification']}")
    print(f"Interpretation: {get_fear_greed_interpretation(fg['value'])}")
    
    print("\n=== VIX Fear Gauge ===")
    vix = get_vix_fear_gauge()
    print(f"Value: {vix['value']}")
    print(f"Interpretation: {vix['interpretation']}")
    
    print("\n=== News Sentiment (AAPL) ===")
    news = get_news_sentiment("AAPL")
    print(f"Score: {news['score']}")
    print(f"Interpretation: {news['interpretation']}")
    if "error" in news:
        print(f"Note: {news['error']}")
    
    print("\n=== Combined Sentiment ===")
    combined = get_combined_sentiment("AAPL")
    print(f"Combined Score: {combined['combined_score']}/100")
    print(f"Interpretation: {combined['interpretation']}")
