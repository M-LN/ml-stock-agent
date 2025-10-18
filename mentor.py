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
        Refleksiv mentor-kommentar med handlingsanbefaling
    """
    symbol = fundamentals.get("symbol", "Aktien")
    score = fundamentals.get("score", 0)
    pe = fundamentals.get("pe")
    peg = fundamentals.get("peg")
    peg_source = fundamentals.get("peg_source", "Unknown")
    pb = fundamentals.get("pb")
    growth = fundamentals.get("growth", 0)
    industry_pe = fundamentals.get("industry_pe", 20)
    
    # === DEL 1: FUNDAMENTALS BASE ===
    if score >= 75:
        base = f"**{symbol}**: Fundamentalt ser aktien stærk ud – du får meget for pengene."
    elif score >= 55:
        base = f"**{symbol}**: Aktien virker rimeligt prissat – ikke et kup, men heller ikke farligt dyr."
    else:
        base = f"**{symbol}**: Tallene peger på overvurdering – her skal du være ekstra kritisk."
    
    # === DEL 2: VALUATION DETALJER ===
    details = []
    
    # P/E ratio
    if pe is not None and pe > 0:
        if pe <= industry_pe:
            details.append(f"✅ P/E på {pe:.1f} er attraktiv (under branchen: {industry_pe:.0f})")
        elif pe <= industry_pe * 1.5:
            details.append(f"⚠️ P/E på {pe:.1f} er over branchen ({industry_pe:.0f}) – let overpris")
        else:
            details.append(f"🔴 P/E på {pe:.1f} er markant over branchen ({industry_pe:.0f}) – dyr!")
    else:
        details.append("ℹ️ P/E data ikke tilgængelig")
    
    # PEG ratio (med kilde-info)
    if peg is not None and peg > 0:
        source_label = f" [{peg_source}]" if peg_source != "Yahoo Finance" else ""
        if peg <= 1.0:
            details.append(f"✅ PEG på {peg:.2f}{source_label} – god værdi ift. vækst")
        elif peg <= 2.0:
            details.append(f"⚠️ PEG på {peg:.2f}{source_label} – fair, men ikke billig")
        else:
            details.append(f"🔴 PEG på {peg:.2f}{source_label} – overprissat ift. vækst")
    else:
        if peg_source == "Ikke Tilgængelig":
            details.append("ℹ️ PEG ikke tilgængelig (mangler vækst-data)")
        else:
            details.append("ℹ️ PEG beregning fejlede – tjek vækst-estimater manuelt")
    
    # P/B ratio
    if pb is not None and pb > 0:
        if pb < 1.0:
            details.append(f"✅ P/B på {pb:.2f} – under bogført værdi")
        elif pb <= 3.0:
            details.append(f"⚖️ P/B på {pb:.2f} – acceptabelt niveau")
        else:
            details.append(f"⚠️ P/B på {pb:.2f} – højt (vækstaktie eller overvurderet)")
    
    # Growth
    if growth > 15:
        details.append(f"🚀 Høj vækst: {growth:.1f}% – men tjek om det kan fortsætte")
    elif growth > 5:
        details.append(f"📈 Moderat vækst: {growth:.1f}% – steady growth")
    elif growth > 0:
        details.append(f"📉 Lav vækst: {growth:.1f}% – mere defensiv")
    else:
        details.append(f"⚠️ Negativ vækst: {growth:.1f}% – rødt flag")
    
    # Tilføj detaljer
    if details:
        base += "\n\n**Nøgletal:**\n" + "\n".join([f"- {d}" for d in details])
    
    # === DEL 3: MARKET SENTIMENT ===
    sentiment_comment = "\n\n**Markedsstemning:**\n"
    
    # Fear & Greed
    if fear_greed >= 70:
        sentiment_comment += "🤑 Markedet er i **grådigheds-mode** – mange køber på hype, og risikoen for bobler er høj. "
        sentiment_comment += "Når alle er optimistiske, er det ofte tid til at være forsigtig."
    elif fear_greed >= 55:
        sentiment_comment += "� Markedet er **bullish** – positiv stemning, men vær kritisk og tjek fundamentals."
    elif fear_greed <= 30:
        sentiment_comment += "� Markedet er præget af **frygt** – det kan åbne for gode købsmuligheder, men kræver is i maven. "
        sentiment_comment += "Kvalitetsaktier kan være på udsalg."
    elif fear_greed <= 45:
        sentiment_comment += "😰 Markedet er **forsigtigt** – defensiv stemning, men også færre irrationelle beslutninger."
    else:
        sentiment_comment += "😐 Stemningen er **neutral** – her handler det om at fokusere på dine egne analyser."
    
    # === DEL 4: NEWS SENTIMENT ===
    news_comment = "\n\n**Nyhedsflow:**\n"
    
    if news_sentiment > 0.3:
        news_comment += "📰 **Meget positivt** – det kan give kortsigtet momentum, men pas på at det ikke bare er hype. "
        news_comment += "Substans > støj!"
    elif news_sentiment > 0.1:
        news_comment += "📰 **Positivt** – kan understøtte en optrend."
    elif news_sentiment < -0.3:
        news_comment += "📰 **Meget negativt** – pas på, at du ikke køber ind i en nedtur. "
        news_comment += "Men hvis fundamentals er stærke, kan det være en overreaktion."
    elif news_sentiment < -0.1:
        news_comment += "📰 **Negativt** – hold øje med udviklingen og lad støvet lægge sig."
    else:
        news_comment += "📰 **Blandet/Neutral** – ingen tydelig retning, så fokuser på det lange spil."
    
    # === DEL 5: VIX WARNING ===
    vix_comment = ""
    
    if vix is not None:
        vix_comment = "\n\n**Volatilitet (VIX):**\n"
        if vix > 30:
            vix_comment += f"🔥 VIX på **{vix:.1f}** (panik-niveau) – markedet forventer store udsving. Reducer position-størrelse!"
        elif vix > 25:
            vix_comment += f"⚡ VIX på **{vix:.1f}** (forhøjet) – øget uro, vær forsigtig med timing."
        elif vix > 20:
            vix_comment += f"📊 VIX på **{vix:.1f}** (moderat) – normal markedsvolatilitet."
        else:
            vix_comment += f"✅ VIX på **{vix:.1f}** (lav) – roligt marked, men pas på selvtilfredshed."
    
    # === DEL 6: HANDLINGSANBEFALING ===
    action = generate_action_recommendation(score, fear_greed, news_sentiment, vix)
    
    # === SAMLET KOMMENTAR ===
    full_comment = base + sentiment_comment + news_comment + vix_comment
    full_comment += f"\n\n{action}"
    
    return full_comment


def generate_action_recommendation(
    fundamental_score: int,
    fear_greed: int,
    news_sentiment: float,
    vix: float = None
) -> str:
    """
    Genererer konkret handlingsanbefaling.
    
    Args:
        fundamental_score: Fair value score (0-100)
        fear_greed: Fear & Greed værdi (0-100)
        news_sentiment: News sentiment (-1 til 1)
        vix: VIX værdi (optional)
    
    Returns:
        Handlingsanbefaling string
    """
    recommendation = "---\n\n## 🎯 Handlingsanbefaling\n\n"
    
    # Scenario 1: Stærke fundamentals + Grådigt marked
    if fundamental_score >= 70 and fear_greed >= 70:
        recommendation += "**✋ VENT**\n\n"
        recommendation += "Selvom fundamentals er stærke, er markedet meget bullish. "
        recommendation += "Risiko for correction når alle er grådige.\n\n"
        recommendation += "**Strategi:** Vent på pullback (5-10%) eller sæt buy-limit ordre under current price."
        return recommendation
    
    # Scenario 2: Stærke fundamentals + Frygt i markedet
    if fundamental_score >= 70 and fear_greed <= 30:
        recommendation += "**✅ KØB (med forsigtighed)**\n\n"
        recommendation += "Stærke fundamentals møder frygt = potentiel buying opportunity!\n\n"
        recommendation += "**Strategi:** Brug dollar-cost averaging (spred købet over 2-4 uger). "
        recommendation += "Sæt stop-loss 8-10% under entry."
        return recommendation
    
    # Scenario 3: Stærke fundamentals + Neutral marked
    if fundamental_score >= 70 and 40 < fear_greed < 60:
        recommendation += "**✅ KØB**\n\n"
        recommendation += "Gode fundamentals og rationelt marked = god balance mellem risiko og reward.\n\n"
        recommendation += "**Strategi:** Standard position size. Sæt stop-loss 5-8% under entry."
        return recommendation
    
    # Scenario 4: Rimelige fundamentals + Grådigt marked
    if 55 <= fundamental_score < 70 and fear_greed >= 70:
        recommendation += "**⚠️ PAS PÅ / HOLD**\n\n"
        recommendation += "Rimelige fundamentals, men markedet er for bullish. Kræver præcis timing.\n\n"
        recommendation += "**Strategi:** Hvis du allerede ejer den – overvej at tage profit. "
        recommendation += "Hvis ny position – vent på bedre entry."
        return recommendation
    
    # Scenario 5: Rimelige fundamentals + Frygt
    if 55 <= fundamental_score < 70 and fear_greed <= 30:
        recommendation += "**🤔 OVERVEJ (small position)**\n\n"
        recommendation += "Fundamentals er OK, og markedet er bange. Kan være en chance.\n\n"
        recommendation += "**Strategi:** Mindre position (50% af normal). Lad resten være cash til hvis det falder mere."
        return recommendation
    
    # Scenario 6: Rimelige fundamentals + Neutral
    if 55 <= fundamental_score < 70:
        recommendation += "**⚖️ NEUTRAL / HOLD**\n\n"
        recommendation += "Fundamentals er rimelige, men ikke overbevisende.\n\n"
        recommendation += "**Strategi:** Følg udviklingen. Hvis du ejer den – behold. "
        recommendation += "Hvis ny – find bedre alternativer."
        return recommendation
    
    # Scenario 7: Svage fundamentals + Grådigt marked
    if fundamental_score < 55 and fear_greed >= 70:
        recommendation += "**🚫 UNDGÅ / SÆLG**\n\n"
        recommendation += "Svage fundamentals + grådigt marked = farlig kombination!\n\n"
        recommendation += "**Strategi:** Hvis du ejer den – sælg ved næste rally. "
        recommendation += "Hvis ny – find bedre muligheder."
        return recommendation
    
    # Scenario 8: Svage fundamentals + Frygt
    if fundamental_score < 55 and fear_greed <= 30:
        recommendation += "**❌ UNDGÅ**\n\n"
        recommendation += "Svage fundamentals selv med frygt. Der er bedre muligheder derude.\n\n"
        recommendation += "**Strategi:** Brug energien på kvalitetsaktier til rabatpris."
        return recommendation
    
    # Scenario 9: Svage fundamentals generelt
    if fundamental_score < 55:
        recommendation += "**❌ UNDGÅ**\n\n"
        recommendation += "Fundamentals understøtter ikke værdiansættelsen.\n\n"
        recommendation += "**Strategi:** Find bedre alternativer med stærkere fundamentals."
        return recommendation
    
    # Juster for ekstrem news sentiment
    if news_sentiment < -0.4:
        recommendation += "**⏳ VENT**\n\n"
        recommendation += "Meget negativt nyhedsflow – lad støvet lægge sig før du handler.\n\n"
        recommendation += "**Strategi:** Vent 1-2 uger og se om det stabiliserer."
        return recommendation
    
    if news_sentiment > 0.4 and fundamental_score < 70:
        recommendation += "**⚠️ PAS PÅ HYPE**\n\n"
        recommendation += "Meget positivt nyhedsflow kan drive prisen, men fundamentals følger ikke med.\n\n"
        recommendation += "**Strategi:** Hvis du vil spille momentum – tight stop-loss (3-5%). Ellers vent."
        return recommendation
    
    # High VIX adjustment
    if vix is not None and vix > 30:
        recommendation += "**⏸️ PAUSE**\n\n"
        recommendation += "VIX over 30 = panik-niveau. Markedet er meget ustabilt.\n\n"
        recommendation += "**Strategi:** Reducer alle positions (max 50% normal size). "
        recommendation += "Hold ekstra cash til når det stabiliserer."
        return recommendation
    
    # Default
    recommendation += "**⚖️ NEUTRAL**\n\n"
    recommendation += "Ingen klar retning baseret på de nuværende metrics.\n\n"
    recommendation += "**Strategi:** Følg udviklingen og vent på tydeligere signaler."
    return recommendation


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
