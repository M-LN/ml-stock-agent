"""
Agent Recommendations Main Module
Kombinerer fundamentals, sentiment og mentor til komplet analyse
"""

from fundamentals import analyze_fundamentals, analyze_multiple_tickers
from sentiment import get_combined_sentiment
from mentor import mentor_comment_simple, mentor_multi_stock_summary
from typing import Dict, List, Optional
import os


def analyze_ticker(
    symbol: str,
    industry_pe: float = 20.0,
    finnhub_api_key: Optional[str] = None
) -> Dict:
    """
    Komplet analyse af en enkelt ticker.
    
    Args:
        symbol: Ticker symbol (fx "AAPL")
        industry_pe: Branche P/E ratio (default 20)
        finnhub_api_key: Finnhub API key (optional)
    
    Returns:
        Dict med alle analyser + mentor kommentar
    """
    # Hent fundamentals
    fundamentals = analyze_fundamentals(symbol, industry_pe)
    
    # Hent sentiment
    sentiment = get_combined_sentiment(symbol, finnhub_api_key)
    
    # Generer mentor kommentar
    mentor_comment = mentor_comment_simple(fundamentals, sentiment)
    
    return {
        "symbol": symbol,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "mentor_comment": mentor_comment,
        "timestamp": sentiment.get("fear_greed", {}).get("timestamp")
    }


def analyze_multiple(
    ticker_list: List[str],
    industry_pe: float = 20.0,
    finnhub_api_key: Optional[str] = None
) -> Dict:
    """
    Analyserer flere tickers og laver komparativ analyse.
    
    Args:
        ticker_list: Liste af ticker symbols
        industry_pe: Branche P/E (default 20)
        finnhub_api_key: Finnhub API key (optional)
    
    Returns:
        Dict med alle analyser + komparativ summary
    """
    analyses = []
    
    for symbol in ticker_list:
        analysis = analyze_ticker(symbol, industry_pe, finnhub_api_key)
        analyses.append(analysis)
    
    # Generer komparativ summary
    summary = mentor_multi_stock_summary(analyses)
    
    return {
        "analyses": analyses,
        "comparative_summary": summary,
        "count": len(analyses)
    }


def get_quick_score(symbol: str, industry_pe: float = 20.0) -> Dict:
    """
    Quick score uden sentiment (hurtigere).
    
    Args:
        symbol: Ticker symbol
        industry_pe: Branche P/E
    
    Returns:
        Dict med fundamentals + quick recommendation
    """
    fundamentals = analyze_fundamentals(symbol, industry_pe)
    
    score = fundamentals["score"]
    
    if score >= 75:
        recommendation = "✅ Stærk købs-kandidat"
    elif score >= 60:
        recommendation = "🤔 Overvej - kræver timing"
    elif score >= 45:
        recommendation = "⚠️ Lidt overvurderet"
    else:
        recommendation = "❌ Undgå - overprissat"
    
    return {
        "symbol": symbol,
        "score": score,
        "recommendation": recommendation,
        **fundamentals
    }


def screen_stocks(
    ticker_list: List[str],
    min_score: int = 60,
    industry_pe: float = 20.0
) -> List[Dict]:
    """
    Screener der filtrerer aktier baseret på minimum score.
    
    Args:
        ticker_list: Liste af tickers
        min_score: Minimum fundamental score (0-100)
        industry_pe: Branche P/E
    
    Returns:
        Liste af tickers der opfylder kriterier
    """
    results = []
    
    for symbol in ticker_list:
        analysis = get_quick_score(symbol, industry_pe)
        
        if analysis["score"] >= min_score:
            results.append(analysis)
    
    # Sorter efter score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results


if __name__ == "__main__":
    # Test
    print("=== SINGLE TICKER ANALYSIS ===")
    result = analyze_ticker("AAPL")
    
    print(f"\n{result['symbol']}:")
    print(f"Fundamental Score: {result['fundamentals']['score']}/100")
    print(f"Sentiment Score: {result['sentiment']['combined_score']}/100")
    print(f"\nMentor Kommentar:\n{result['mentor_comment']}")
    
    print("\n\n=== MULTI TICKER ANALYSIS ===")
    multi = analyze_multiple(["AAPL", "MSFT", "GOOGL"])
    print(multi['comparative_summary'])
    
    print("\n\n=== STOCK SCREENER ===")
    screened = screen_stocks(["AAPL", "MSFT", "TSLA", "NVDA"], min_score=65)
    print(f"\nFandt {len(screened)} aktier med score >= 65:")
    for stock in screened:
        print(f"- {stock['symbol']}: {stock['score']}/100 - {stock['recommendation']}")
