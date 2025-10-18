"""
Sentiment Analysis Module
Henter psykologiske signaler fra markedet (Fear & Greed + News Sentiment)
"""

import requests
from typing import Dict, Optional
import os


def get_fear_greed() -> Dict:
    """
    Henter Fear & Greed Index fra Alternative.me (Crypto Fear & Greed).
    
    Note: Dette er crypto-baseret. For aktier kan man bruge CNN Fear & Greed
    eller bygge egen index baseret på VIX, put/call ratio, etc.
    
    Returns:
        Dict med value (0-100) og classification
    """
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        value = int(data["data"][0]["value"])
        classification = data["data"][0]["value_classification"]
        
        return {
            "value": value,
            "classification": classification,
            "source": "Alternative.me (Crypto)",
            "timestamp": data["data"][0]["timestamp"]
        }
    
    except Exception as e:
        return {
            "value": 50,  # Neutral default
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
