# Portfolio Management - Feature Documentation

## Overview
Avanceret porteføljestyring med live data, performance tracking, skatteoptimering og risiko-analyse.

## Features

### 📊 Tab 1: Portfolio Overview
**Basis funktionalitet:**
- Tilføj/fjern positioner (aktie symbol, antal aktier, avg cost, sektor, asset class)
- Portfolio summary med total værdi, antal positioner, total gain/loss
- Live prisdata fra Yahoo Finance med **"🔄 Update Live Prices"** knap
- **"💾 Save Snapshot"** knap til at gemme portfolio state til historik

**Hvordan det virker:**
1. Tilføj positions manuelt ELLER brug "Update Live Prices" til at hente nuværende kurs
2. Priser caches i 5 minutter for performance
3. Save Snapshot gemmer current state til performance tracking

---

### ⚖️ Tab 2: Asset Allocation & Rebalancing
**Target Allocation:**
- Sæt ønsket fordeling mellem Stocks, Bonds, Commodities (skal være 100%)
- Visual sammenligning mellem current og target allocation
- Pie chart visualization

**Rebalancing:**
- Threshold-baseret: Trigger når allocation afviger med X%
- Frequency: Monthly, Quarterly, Semi-Annually, Annually
- **"🔄 Calculate Rebalancing Trades"**: Viser nøjagtigt hvad der skal købes/sælges
- Output: "BUY/SELL $X of Asset Class" med current → target %

---

### 🎯 Tab 3: Position Sizing
**3 Sizing Metoder:**

1. **Fixed Percent (5%)**: Simple, fast procentdel af portfolio
2. **Risk-Adjusted (Volatility-based)**: Justerer size baseret på volatilitet
   - Høj volatilitet = mindre position
   - Lav volatilitet = større position (max 15%)
3. **Kelly Criterion**: Matematisk optimal sizing
   - Input: Win rate, Win/Loss ratio
   - Output: Optimal position size (bruger half-Kelly for safety)

**Calculator:**
- Indtast portfolio value og stock price
- Får antal shares at købe + dollar amount
- Side-by-side sammenligning af alle 3 metoder

---

### 🤖 Tab 4: AI Portfolio Mentor
**Personlig Portfolio Feedback:**

Risk Profile indstilling:
- Conservative: Fokus på kapitalbevarelse
- Moderate: Balanced tilgang
- Aggressive: Fokus på vækst

**Mentor identificerer:**
1. ⚠️ **Overconcentration**: "Microsoft fylder 25% - reducer til 15%"
2. 📊 **Allocation Drift**: "Stocks er 75%, target er 60% - rebalancer"
3. 🏢 **Sector Concentration**: "Technology er 40% - diversificer"
4. 🛡️ **Risk Mismatch**: "Conservative profil men 70% i stocks"
5. 📈 **Scenario Analysis**: "Hvis renter stiger 1%, så..."
6. ✅ **Positive Feedback**: Når alt ser godt ud

**Output:**
- Expandable feedback cards med specifik guidance
- Sector diversification bar chart
- Concentration warnings ved >30% i én sektor

---

### 📈 Tab 5: Performance Tracking
**Historical Tracking:**
- Gemmer portfolio snapshots over tid (op til 365)
- Portfolio value chart over tid
- Cumulative returns chart

**Performance Metrics:**
- Total Return (%)
- Annualized Return (%)
- Volatility (Annual %)
- Sharpe Ratio (risk-adjusted return)
- Max Drawdown (worst peak-to-trough decline)
- Win Rate (% af positive dage)
- Start Value vs Current Value

**Hvordan det virker:**
1. Klik "Save Snapshot" i Portfolio Overview tab regelmæssigt
2. Byg historik over tid (daily, weekly, etc.)
3. Se performance charts og metrics i denne tab
4. View Historical Data table med sidste 30 snapshots

---

### 💰 Tab 6: Tax-Loss Harvesting
**Identificer Skattebesparelser:**

**Settings:**
- Minimum Loss Threshold slider (fx -5%)
- Viser kun positions med tab større end threshold

**Output for hver opportunity:**
1. **Position Details**:
   - Shares, Cost Basis, Market Value
2. **Tax Benefit**:
   - Realized Loss (dollar amount)
   - Est. Tax Savings (assumes 30% tax rate)
   - Loss % 
3. **Replacement Options**:
   - Sector ETF alternativer for at undgå wash sale rule
   - Fx hvis du sælger AAPL → Køb XLK (Technology ETF)

**Wash Sale Warning:**
- Kan ikke købe samme eller substantially identical security i 30 dage før/efter salg
- Foreslår sector ETFs som replacement

**Portfolio Summary:**
- Total Unrealized Losses
- Total Unrealized Gains
- Net Unrealized (gains - losses)

**Use Case:**
- End of year: Harvest losses til at offset gains
- Maintain exposure via sector ETFs
- Reduce tax bill

---

### 🔗 Tab 7: Correlation Analysis
**Diversification Check:**

**Correlation Matrix:**
- Heatmap af alle positions mod hinanden
- Farver: Rød (høj correlation) → Grøn (lav/negativ correlation)
- Values: -1.0 til +1.0

**High Correlation Risks:**
- Threshold slider (default 0.7)
- Identificerer pairs med høj correlation
- Risk level: High (>0.85) eller Moderate (0.7-0.85)

**Interpretation Guide:**
- +1.0: Perfect positive (move together)
- +0.7 til +1.0: Strong positive
- -0.3 til +0.3: Weak/no correlation
- -1.0: Perfect negative (move opposite)

**Diversification Tips:**
- Aim for correlations < 0.7
- Mix asset classes for lower correlation
- Negative correlations provide hedging

**Analysis Period:** 1mo, 3mo, 6mo, 1y, 2y

---

### 🎲 Tab 8: Monte Carlo Simulation
**Future Portfolio Outcomes:**

**Settings:**
- Number of Simulations: 100, 500, 1000, 2500, 5000
- Simulation Period: 21, 63, 126, 252, 504 trading days

**Key Metrics:**
1. **Current Value**: Starting point
2. **Mean Final Value**: Average outcome
3. **Median Final Value**: Middle outcome
4. **Probability of Profit**: % chance of positive return

**Outcome Percentiles:**
- **5th Percentile**: Worst 5% of outcomes (downside risk)
- **25th Percentile**: Lower quartile
- **75th Percentile**: Upper quartile
- **95th Percentile**: Best 5% of outcomes (upside potential)

**Risk Metrics:**
- **Value at Risk (95%)**: Max expected loss with 95% confidence
- **Standard Deviation**: Volatility of outcomes
- **Range**: Min to Max possible value

**Visualizations:**
1. **Simulation Paths Chart**: Shows 100 sample paths
   - Individual paths (light gray lines)
   - Median path (blue line)
   - 5th and 95th percentile bands (red/green dashed)
2. **Distribution Histogram**: Frequency of final values
   - Vertical lines for current, median, 5th/95th percentiles

**Interpretation:**
- Baseret på historical volatility og correlation
- 90% confidence interval: 5th til 95th percentile
- Use for risk management og scenario planning
- NOT a prediction of actual future performance

**Use Cases:**
- Assess downside risk before making changes
- Understand range of possible outcomes
- Compare strategies (run simulation before/after rebalancing)
- Set realistic expectations

---

## Integration Flow

### Recommended Workflow:
1. **Setup** (Tab 1): Tilføj positions → Update Live Prices
2. **Set Targets** (Tab 2): Define asset allocation goals
3. **Analyze Risk** (Tab 7, 8): Check correlations og run Monte Carlo
4. **Get Advice** (Tab 4): AI mentor feedback
5. **Optimize** (Tab 3, 6): Position sizing og tax-loss harvesting
6. **Monitor** (Tab 5): Save snapshots regelmæssigt, track performance
7. **Rebalance** (Tab 2): Calculate trades når threshold nås

### Data Sources:
- **Yahoo Finance**: Live prices, historical data (via yfinance)
- **Local Storage**: Portfolio snapshots, saved portfolios (JSON files)
- **Caching**: 5-minute cache for live prices (performance optimization)

### Performance Tips:
- Update prices during market hours for real-time data
- Save snapshots daily/weekly for consistent tracking
- Run correlation analysis monthly
- Run Monte Carlo before major portfolio changes
- Check tax-loss opportunities in Q4 (end of year)

---

## Technical Details

### Dependencies:
- `yfinance`: Live price data
- `scipy`: Correlation og statistical analysis
- `numpy`: Monte Carlo simulations
- `pandas`: Data manipulation
- `plotly`: Interactive charts

### Data Structures:

**Position:**
```python
{
    'symbol': 'AAPL',
    'shares': 10,
    'avg_cost': 150.00,
    'current_price': 175.00,
    'sector': 'Technology',
    'asset_class': 'stocks',
    'added_date': '2025-10-19',
    'price_change': 2.50,
    'price_change_pct': 1.45,
    'last_updated': '2025-10-19T14:30:00'
}
```

**Performance Snapshot:**
```python
{
    'timestamp': '2025-10-19T14:30:00',
    'total_value': 100000.00,
    'positions': 5,
    'asset_allocation': {'stocks': 60, 'bonds': 30, 'commodities': 10},
    'positions_detail': [...]
}
```

### Caching Strategy:
- Live prices cached per symbol per 5-minute window
- Cache key: `{symbol}_{YYYYMMDD_HHMM}`
- Reduces API calls og improves performance

### Storage:
- Portfolio files: `portfolios/portfolio_YYYYMMDD_HHMMSS.json`
- Includes: positions, settings, performance snapshots
- Load/Save via sidebar

---

## Future Enhancements

Potential additions:
- [ ] Auto-rebalancing execution (via broker API)
- [ ] Real-time WebSocket price feeds
- [ ] Options strategies integration
- [ ] Dividend tracking og reinvestment
- [ ] Currency hedging for international positions
- [ ] ESG scoring integration
- [ ] Backtesting strategy changes
- [ ] Multi-portfolio comparison
- [ ] Mobile app notifications
- [ ] Integration med broker accounts (Alpaca, Interactive Brokers)

---

## Known Limitations

1. **Yahoo Finance Data**: Free tier kan have delays eller rate limits
2. **Historical Data**: Limited til Yahoo Finance history (nogle assets mangler data)
3. **Tax Calculation**: Simplified 30% rate (ikke personalized til din bracket)
4. **Wash Sale**: Replacement suggestions er generiske (ikke personalized)
5. **Monte Carlo**: Assumes normal distribution (kan underestimate tail risks)
6. **Correlation**: Based on historical data (kan ændre sig over tid)

---

## Support & Troubleshooting

**Common Issues:**

1. **"No data available" for symbol**
   - Check symbol is correct (use Yahoo Finance format)
   - Some symbols not available via yfinance
   - Try different symbol or ETF

2. **Correlation matrix empty**
   - Need at least 2 positions
   - Need sufficient historical data (30+ days)
   - Try longer analysis period

3. **Monte Carlo takes long time**
   - Reduce number of simulations
   - Reduce simulation period
   - Fewer positions = faster simulation

4. **Performance tracking shows no data**
   - Need to save snapshots first
   - Click "Save Snapshot" in Overview tab
   - Build history over time

---

Built with ❤️ using Streamlit, YFinance, and AI-powered insights.
