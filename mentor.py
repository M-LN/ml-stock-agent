"""
Mentor Reflection Module
Kombinerer fundamentals + sentiment til en refleksiv mentor-kommentar
"""

from typing import Dict


def mentor_comment(
    fundamentals: Dict,
    fear_greed: int,
    news_sentiment: float,
    vix: float = None
) -> str:
    """
    Genererer mentor-kommentar baseret på fundamentals + sentiment.
    
    Args:
        fundamentals: Dict fra fundamentals.analyze_fundamentals()
        fear_greed: Fear & Greed værdi (0-100)
        news_sentiment: News sentiment score (-1 til 1)
        vix: VIX værdi (optional)
    
    Returns:
        Refleksiv mentor-kommentar
    """
    symbol = fundamentals.get("symbol", "?")
    score = fundamentals.get("score", 0)
    pe = fundamentals.get("pe")
    growth = fundamentals.get("growth", 0)
    
    # === DEL 1: FUNDAMENTALS BASE ===
    if score >= 75:
        base = f"**{symbol}** ser fundamentalt stærk ud (score: {score}/100)."
    elif score >= 60:
        base = f"**{symbol}** er rimeligt prissat (score: {score}/100)."
    elif score >= 45:
        base = f"**{symbol}** virker lidt overvurderet (score: {score}/100)."
    else:
        base = f"**{symbol}** ser dyr ud ift. fundamentals (score: {score}/100)."
    
    # === DEL 2: VALUATION DETALJER ===
    details = []
    
    if pe is not None:
        industry_pe = fundamentals.get("industry_pe", 20)
        if pe <= industry_pe:
            details.append(f"P/E på {pe:.1f} er under branchen ({industry_pe})")
        elif pe <= industry_pe * 1.5:
            details.append(f"P/E på {pe:.1f} er over branchen ({industry_pe})")
        else:
            details.append(f"P/E på {pe:.1f} er markant over branchen ({industry_pe})")
    
    if growth > 15:
        details.append(f"høj vækst på {growth:.1f}%")
    elif growth > 5:
        details.append(f"moderat vækst på {growth:.1f}%")
    elif growth > 0:
        details.append(f"lav vækst på {growth:.1f}%")
    else:
        details.append("negativ vækst")
    
    if details:
        base += f" {', '.join(details).capitalize()}."
    
    # === DEL 3: MARKET SENTIMENT ===
    sentiment_comment = ""
    
    # Fear & Greed
    if fear_greed >= 75:
        sentiment_comment = "\n\n⚠️ **Markedet er i grådigheds-mode.** "
        sentiment_comment += "Det kan være fristende at hoppe med på bølgen, men husk: "
        sentiment_comment += "når alle er optimistiske, er risikoen ofte størst. "
        sentiment_comment += "Overvej at vente på et pullback."
    
    elif fear_greed >= 60:
        sentiment_comment = "\n\n📈 **Markedet er bullish.** "
        sentiment_comment += "Positiv stemning kan give momentum, men vær opmærksom på "
        sentiment_comment += "om fundamentals understøtter priserne."
    
    elif fear_greed <= 25:
        sentiment_comment = "\n\n🛡️ **Markedet er præget af frygt.** "
        sentiment_comment += "Det kan være en god tid at kigge efter kvalitetsaktier til rabat, "
        sentiment_comment += "men sørg for at have en god risikostyring."
    
    elif fear_greed <= 40:
        sentiment_comment = "\n\n⚖️ **Markedet er forsigtigt.** "
        sentiment_comment += "Neutral stemning kan være godt for rationelle beslutninger."
    
    # === DEL 4: NEWS SENTIMENT ===
    news_comment = ""
    
    if news_sentiment > 0.3:
        news_comment = "\n\n📰 **Nyhedsflowet er meget positivt** – "
        news_comment += "det kan give kortsigtet momentum, men pas på hype. "
        news_comment += "Tjek om det er substans eller bare FOMO."
    
    elif news_sentiment > 0.1:
        news_comment = "\n\n📰 **Nyhedsflowet er positivt** – "
        news_comment += "kan understøtte optrend."
    
    elif news_sentiment < -0.3:
        news_comment = "\n\n📰 **Nyhedsflowet er meget negativt** – "
        news_comment += "vær opmærksom på risiko for yderligere fald. "
        news_comment += "Men hvis fundamentals er stærke, kan det være en overreaktion."
    
    elif news_sentiment < -0.1:
        news_comment = "\n\n📰 **Nyhedsflowet er negativt** – "
        news_comment += "hold øje med udviklingen."
    
    # === DEL 5: VIX WARNING ===
    vix_comment = ""
    
    if vix is not None:
        if vix > 30:
            vix_comment = f"\n\n🔥 **VIX er på {vix:.1f}** (panik-niveau). "
            vix_comment += "Markedet forventer høj volatilitet. Vær ekstra forsigtig med timing."
        elif vix > 25:
            vix_comment = f"\n\n⚡ **VIX er på {vix:.1f}** (forhøjet). "
            vix_comment += "Øget markedsuro – overvej mindre positioner."
    
    # === DEL 6: KOMBINERET ANBEFALING ===
    recommendation = generate_recommendation(score, fear_greed, news_sentiment)
    
    # === SAMLET KOMMENTAR ===
    full_comment = base + sentiment_comment + news_comment + vix_comment
    full_comment += f"\n\n**Mentor Anbefaling:** {recommendation}"
    
    return full_comment


def generate_recommendation(
    fundamental_score: int,
    fear_greed: int,
    news_sentiment: float
) -> str:
    """
    Genererer konkret anbefaling baseret på alle faktorer.
    
    Args:
        fundamental_score: Fair value score (0-100)
        fear_greed: Fear & Greed værdi (0-100)
        news_sentiment: News sentiment (-1 til 1)
    
    Returns:
        Anbefaling string
    """
    # Scenario 1: Stærke fundamentals + Grådigt marked
    if fundamental_score >= 70 and fear_greed >= 70:
        return "✋ VENT – Selvom fundamentals er stærke, er markedet meget bullish. Risiko for correction. Overvej at tage profit eller vent på pullback."
    
    # Scenario 2: Stærke fundamentals + Frygt i markedet
    if fundamental_score >= 70 and fear_greed <= 30:
        return "✅ KØB – Stærke fundamentals møder frygt = potentiel buying opportunity. Men brug dollar-cost averaging."
    
    # Scenario 3: Stærke fundamentals + Neutral marked
    if fundamental_score >= 70 and 40 <= fear_greed <= 60:
        return "✅ KØB – Gode fundamentals og rationelt marked. God balance mellem risiko og reward."
    
    # Scenario 4: Rimelige fundamentals + Grådigt marked
    if 55 <= fundamental_score < 70 and fear_greed >= 70:
        return "⚠️ PAS PÅ – Rimelige fundamentals, men markedet er for bullish. Kræver præcis timing. Overvej at vente."
    
    # Scenario 5: Rimelige fundamentals + Frygt
    if 55 <= fundamental_score < 70 and fear_greed <= 30:
        return "🤔 OVERVEJ – Fundamentals er OK, og markedet er bange. Kan være en chance, men ikke den bedste i klassen."
    
    # Scenario 6: Rimelige fundamentals + Neutral
    if 55 <= fundamental_score < 70 and 40 <= fear_greed <= 60:
        return "⚖️ NEUTRAL – Fundamentals er rimelige, marked er rationelt. Følg udviklingen, men ikke et klart køb."
    
    # Scenario 7: Svage fundamentals + Grådigt marked
    if fundamental_score < 55 and fear_greed >= 70:
        return "🚫 UNDGÅ – Svage fundamentals + grådigt marked = farlig kombination. Risiko for større fald."
    
    # Scenario 8: Svage fundamentals + Frygt
    if fundamental_score < 55 and fear_greed <= 30:
        return "❌ UNDGÅ – Svage fundamentals selv med frygt. Der er bedre muligheder."
    
    # Scenario 9: Svage fundamentals + Neutral
    if fundamental_score < 55:
        return "❌ UNDGÅ – Fundamentals understøtter ikke værdiansættelsen. Find bedre alternativer."
    
    # Juster for ekstrem news sentiment
    if news_sentiment < -0.4:
        return "⏳ VENT – Meget negativt nyhedsflow. Lad støvet lægge sig før du handler."
    
    if news_sentiment > 0.4 and fundamental_score < 70:
        return "⚠️ PAS PÅ HYPE – Meget positivt nyhedsflow kan drive prisen, men fundamentals følger ikke med."
    
    return "⚖️ NEUTRAL – Ingen klar retning. Følg udviklingen."


def mentor_comment_simple(fundamentals: Dict, sentiment_data: Dict) -> str:
    """
    Simplificeret version der tager hele sentiment dict.
    
    Args:
        fundamentals: Dict fra fundamentals.analyze_fundamentals()
        sentiment_data: Dict fra sentiment.get_combined_sentiment()
    
    Returns:
        Mentor-kommentar
    """
    fear_greed = sentiment_data.get("fear_greed", {}).get("value", 50)
    news = sentiment_data.get("news_sentiment", {}).get("score", 0)
    vix_data = sentiment_data.get("vix", {})
    vix = vix_data.get("value")
    
    return mentor_comment(fundamentals, fear_greed, news, vix)


def mentor_multi_stock_summary(analyses: list) -> str:
    """
    Genererer sammenligning af flere aktier.
    
    Args:
        analyses: Liste af dicts med fundamentals + sentiment
    
    Returns:
        Komparativ mentor-kommentar
    """
    if not analyses:
        return "Ingen aktier at analysere."
    
    # Sorter efter fundamental score
    sorted_stocks = sorted(analyses, key=lambda x: x['fundamentals']['score'], reverse=True)
    
    summary = "## 📊 Multi-Stock Analyse\n\n"
    summary += "### Top Picks (baseret på fundamentals + sentiment):\n\n"
    
    for i, stock in enumerate(sorted_stocks[:3], 1):
        symbol = stock['fundamentals']['symbol']
        score = stock['fundamentals']['score']
        sentiment = stock.get('sentiment', {})
        fg = sentiment.get('fear_greed', {}).get('value', 50)
        
        summary += f"**#{i}: {symbol}** (Score: {score}/100, Market Sentiment: {fg}/100)\n"
    
    summary += f"\n### Undgå:\n\n"
    
    for stock in sorted_stocks[-2:]:
        symbol = stock['fundamentals']['symbol']
        score = stock['fundamentals']['score']
        summary += f"- **{symbol}** (Score: {score}/100) – Svage fundamentals\n"
    
    return summary


if __name__ == "__main__":
    # Test med mock data
    mock_fundamentals = {
        "symbol": "AAPL",
        "score": 72,
        "pe": 28.5,
        "pb": 35.2,
        "peg": 2.1,
        "growth": 8.5,
        "industry_pe": 25
    }
    
    print("\n=== SCENARIO 1: Gode fundamentals + Grådigt marked ===")
    comment1 = mentor_comment(mock_fundamentals, fear_greed=78, news_sentiment=0.3)
    print(comment1)
    
    print("\n\n=== SCENARIO 2: Gode fundamentals + Frygt i markedet ===")
    comment2 = mentor_comment(mock_fundamentals, fear_greed=25, news_sentiment=-0.2)
    print(comment2)
    
    print("\n\n=== SCENARIO 3: Svage fundamentals + Grådigt marked ===")
    weak_fundamentals = {
        "symbol": "HYPE",
        "score": 42,
        "pe": 65.2,
        "pb": 12.5,
        "peg": 3.8,
        "growth": 2.1,
        "industry_pe": 25
    }
    comment3 = mentor_comment(weak_fundamentals, fear_greed=82, news_sentiment=0.5)
    print(comment3)
