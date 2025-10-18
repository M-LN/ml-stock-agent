"""
Fundamentals Analysis Module
Beregner fair value score baseret på klassiske værdiansættelses-modeller
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional, Tuple


def get_peg_ratio(ticker: yf.Ticker, info: dict) -> Tuple[Optional[float], str]:
    """
    Henter PEG ratio med fallback-kæde for bedst mulig datakvalitet.
    
    Fallback-kæde:
    1. Yahoo Finance pegRatio (forward PEG)
    2. Beregn fra forward P/E og forventet vækst
    3. Beregn fra trailing P/E og historisk vækst (med disclaimer)
    4. Return None hvis ingen data
    
    Returns:
        Tuple[peg_value, data_source]
    """
    # 1. Forsøg direkte pegRatio fra Yahoo
    peg = info.get("pegRatio")
    if peg and peg > 0:
        return peg, "Yahoo Finance"
    
    # 2. Beregn fra forward P/E og earnings growth
    forward_pe = info.get("forwardPE")
    earnings_growth = info.get("earningsGrowth")  # Forward årlig vækst
    
    if forward_pe and earnings_growth and earnings_growth > 0:
        peg_calculated = forward_pe / (earnings_growth * 100)
        return peg_calculated, "Beregnet (Forward)"
    
    # 3. Beregn fra trailing P/E og historisk vækst
    trailing_pe = info.get("trailingPE")
    earnings_quarterly_growth = info.get("earningsQuarterlyGrowth")
    
    if trailing_pe and earnings_quarterly_growth and earnings_quarterly_growth > 0:
        # Annualiseret vækst
        annual_growth = earnings_quarterly_growth * 100
        peg_calculated = trailing_pe / annual_growth
        return peg_calculated, "Beregnet (Historisk)"
    
    # 4. Forsøg at beregne vækst fra historiske earnings
    try:
        earnings = ticker.earnings
        if earnings is not None and not earnings.empty and len(earnings) >= 2:
            # Beregn CAGR fra earnings
            first_earnings = earnings.iloc[0]["Earnings"]
            last_earnings = earnings.iloc[-1]["Earnings"]
            years = len(earnings) - 1
            
            if first_earnings > 0 and last_earnings > 0 and years > 0:
                cagr = (pow(last_earnings / first_earnings, 1/years) - 1) * 100
                if cagr > 0 and trailing_pe:
                    peg_calculated = trailing_pe / cagr
                    return peg_calculated, "Beregnet (CAGR)"
    except Exception:
        pass
    
    return None, "Ikke Tilgængelig"


def get_forward_growth(ticker: yf.Ticker, info: dict) -> Optional[float]:
    """
    Henter fremtidig vækst estimat med fallback.
    
    Returns:
        Growth rate i % eller None
    """
    # 1. Direkte earnings growth (analyst estimates)
    earnings_growth = info.get("earningsGrowth")
    if earnings_growth and earnings_growth > 0:
        return earnings_growth * 100
    
    # 2. Revenue growth som proxy
    revenue_growth = info.get("revenueGrowth")
    if revenue_growth and revenue_growth > 0:
        return revenue_growth * 100
    
    # 3. Quarterly earnings growth annualiseret
    earnings_quarterly_growth = info.get("earningsQuarterlyGrowth")
    if earnings_quarterly_growth and earnings_quarterly_growth > 0:
        return earnings_quarterly_growth * 100
    
    return None


def analyze_fundamentals(symbol: str, industry_pe: float = 20.0) -> Dict:
    """
    Analyserer fundamentale nøgletal for en aktie.
    
    Args:
        symbol: Ticker symbol (fx "AAPL")
        industry_pe: Branche P/E ratio (default 20)
    
    Returns:
        Dict med score + nøgletal
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Hent basis nøgletal
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        
        # Forbedret PEG ratio acquisition
        peg, peg_source = get_peg_ratio(ticker, info)
        
        # Forbedret growth estimate
        growth = get_forward_growth(ticker, info)
        if growth is None:
            growth = info.get("earningsQuarterlyGrowth", 0) * 100 if info.get("earningsQuarterlyGrowth") else 0
        
        # Beregn fair value score
        score = calculate_fair_value_score(pe, industry_pe, peg, pb, growth)
        
        # Samlet resultat
        return {
            "symbol": symbol,
            "score": score,
            "pe": pe,
            "pb": pb,
            "peg": peg,
            "peg_source": peg_source,  # Ny: viser datakilde
            "growth": growth,
            "industry_pe": industry_pe,
            "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A")
        }
    
    except Exception as e:
        return {
            "symbol": symbol,
            "score": 0,
            "error": str(e),
            "pe": None,
            "pb": None,
            "peg": None,
            "peg_source": "Error",
            "growth": 0
        }


def calculate_fair_value_score(
    pe: Optional[float],
    industry_pe: float,
    peg: Optional[float],
    pb: Optional[float],
    growth: float
) -> int:
    """
    Beregner fair value score (0-100) baseret på fundamentale metrics.
    
    Scoring system:
    - P/E ratio: 25 pts max
    - PEG ratio: 25 pts max
    - P/B ratio: 20 pts max
    - Growth: 20 pts max
    - Buffer: 10 pts
    """
    score = 0
    
    # P/E ratio scoring (25 pts)
    if pe is not None:
        if pe <= industry_pe:
            score += 25  # Under eller lig branchen
        elif pe <= industry_pe * 1.2:
            score += 15  # Op til 20% over branchen
        elif pe <= industry_pe * 1.5:
            score += 10  # Op til 50% over branchen
        else:
            score += 5   # Meget dyr
    else:
        score += 5  # Ingen data
    
    # PEG ratio scoring (25 pts)
    if peg is not None and peg > 0:
        if peg <= 1.0:
            score += 25  # Undervurderet ift. vækst
        elif peg <= 1.5:
            score += 15  # Rimelig prissat
        elif peg <= 2.0:
            score += 10  # Lidt dyr
        else:
            score += 5   # Overprissat
    else:
        score += 5  # Ingen data
    
    # P/B ratio scoring (20 pts)
    if pb is not None:
        if 0.8 <= pb <= 1.5:
            score += 20  # Ideelt område
        elif pb < 0.8:
            score += 15  # Under bogført værdi (potentielt undervurderet)
        elif pb <= 2.5:
            score += 10  # Acceptabelt
        else:
            score += 5   # Høj P/B
    else:
        score += 10  # Ingen data
    
    # Growth scoring (20 pts)
    if growth >= 15:
        score += 20  # Høj vækst
    elif growth >= 10:
        score += 15  # God vækst
    elif growth >= 5:
        score += 10  # Moderat vækst
    elif growth >= 0:
        score += 5   # Lav vækst
    else:
        score += 0   # Negativ vækst
    
    # Buffer (10 pts) - bruges til fremtidig justering
    score += 10
    
    return min(score, 100)


def get_valuation_category(score: int) -> str:
    """
    Konverterer score til kategori.
    
    Args:
        score: Fair value score (0-100)
    
    Returns:
        Kategori string
    """
    if score >= 75:
        return "Fundamentalt Stærk"
    elif score >= 60:
        return "Rimeligt Prissat"
    elif score >= 45:
        return "Lidt Overvurderet"
    else:
        return "Overvurderet"


def analyze_multiple_tickers(ticker_list: list, industry_pe: float = 20.0) -> list:
    """
    Analyserer flere tickers på én gang.
    
    Args:
        ticker_list: Liste af ticker symbols
        industry_pe: Branche P/E (default 20)
    
    Returns:
        Liste af fundamentals dicts
    """
    results = []
    
    for symbol in ticker_list:
        result = analyze_fundamentals(symbol, industry_pe)
        result["category"] = get_valuation_category(result["score"])
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Test
    result = analyze_fundamentals("AAPL")
    print(f"\n{result['symbol']} Analysis:")
    print(f"Score: {result['score']}/100")
    print(f"P/E: {result['pe']}")
    print(f"PEG: {result['peg']}")
    print(f"P/B: {result['pb']}")
    print(f"Growth: {result['growth']:.2f}%")
    print(f"Category: {get_valuation_category(result['score'])}")
