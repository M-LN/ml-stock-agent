# 💼 Portfolio Management - Quick Start

## Nye Avancerede Features

Jeg har tilføjet 5 kraftfulde nye features til Portfolio Management siden:

### 1. 🔄 Live Priser fra Yahoo Finance
- **Auto-fetch** aktuelle priser for alle positions
- **Real-time updates** med én klik
- **Smart caching** (5 min) for performance
- Inkluderer: price, change %, volume, market cap, P/E ratio

**Sådan bruger du:**
```
Tab 1: Portfolio Overview → Klik "🔄 Update Live Prices"
```

---

### 2. 📈 Historical Performance Tracking
- **Track portfolio over tid** med snapshots
- **Performance metrics**: Total return, Sharpe ratio, Max drawdown, Win rate
- **Visual charts**: Value over time, cumulative returns
- **365 dages historik**

**Sådan bruger du:**
```
1. Tab 1: Klik "💾 Save Snapshot" regelmæssigt (daily/weekly)
2. Tab 5: Se performance charts og metrics
```

**Metrics du får:**
- Total Return %
- Annualized Return %
- Volatility (Annual %)
- Sharpe Ratio
- Max Drawdown %
- Win Rate %

---

### 3. 💰 Tax-Loss Harvesting
- **Identificer skattebesparelser** automatisk
- **Beregn tax savings** (assumes 30% rate)
- **Wash sale protection** med replacement forslag
- **Sector ETF alternatives** til at maintain exposure

**Sådan bruger du:**
```
Tab 6: Tax-Loss Harvesting → Set threshold → Klik "🔍 Identify Opportunities"
```

**Output:**
- Positions med tab > threshold
- Realized loss amount ($)
- Estimated tax savings ($)
- Replacement securities (sector ETFs)

**Use Case:**
```
End of year: AAPL har -$5,000 tab
→ Sell AAPL (harvest loss)
→ Save ~$1,500 i tax (30% rate)
→ Buy XLK (tech ETF) for exposure
→ Wait 31 days, buy AAPL back hvis du vil
```

---

### 4. 🔗 Correlation Analysis
- **Correlation matrix** mellem alle positions
- **Heatmap visualization** (rød = høj, grøn = lav)
- **Risk identification**: Flag pairs med high correlation
- **Diversification guidance**

**Sådan bruger du:**
```
Tab 7: Correlation Analysis → Select period → Klik "📊 Calculate Correlation"
```

**What to look for:**
- **< 0.7**: God diversification ✅
- **> 0.7**: High correlation ⚠️ (positions move together)
- **> 0.85**: Very high correlation 🔴 (reducer exposure)
- **Negative**: Hedging benefit 🎯

**Example:**
```
AAPL ↔ MSFT: 0.85 → High correlation
→ Hvis AAPL falder, MSFT falder sandsynligvis også
→ Overvej at reducere en af dem eller tilføj uncorrelated asset
```

---

### 5. 🎲 Monte Carlo Risk Simulation
- **1000+ simulations** af portfolio outcomes
- **Probability distributions** af future values
- **Value at Risk (VaR)** calculation
- **Percentile analysis** (5th, 25th, 75th, 95th)

**Sådan bruger du:**
```
Tab 8: Monte Carlo Simulation → Set parameters → Klik "🎲 Run Simulation"
```

**Parameters:**
- Number of simulations: 100 - 5000
- Period: 21 - 504 trading days (1 month - 2 years)

**Output:**
1. **Probability of Profit**: % chance af positive return
2. **Expected Range**: 5th til 95th percentile outcomes
3. **Value at Risk**: Max loss med 95% confidence
4. **Simulation Paths**: Visual af possible trajectories
5. **Distribution**: Histogram af final values

**Example Interpretation:**
```
Current: $100,000
Mean Final: $108,500
Median: $107,200
5th Percentile: $85,000 (worst 5%)
95th Percentile: $135,000 (best 5%)
Probability of Profit: 72%

→ 72% chance af profit
→ 90% confident outcome mellem $85k - $135k
→ Max expected loss (VaR): $15,000
```

---

## Integration Flow

### Recommended Daily/Weekly Routine:

```
1. UPDATE PRICES (Tab 1)
   ↓
2. SAVE SNAPSHOT (Tab 1)
   ↓
3. CHECK MENTOR FEEDBACK (Tab 4)
   ↓
4. [If rebalance needed] → CALCULATE TRADES (Tab 2)
```

### Monthly Review:

```
1. PERFORMANCE REVIEW (Tab 5)
   ↓
2. CORRELATION CHECK (Tab 7)
   ↓
3. MONTE CARLO ANALYSIS (Tab 8)
   ↓
4. [If needed] → ADJUST ALLOCATION (Tab 2)
```

### Quarterly/Year-End:

```
1. TAX-LOSS HARVESTING (Tab 6)
   ↓
2. REBALANCING (Tab 2)
   ↓
3. POSITION SIZING REVIEW (Tab 3)
```

---

## Real-World Example Workflow

### Scenario: Du har $100k portfolio

**Step 1: Setup**
```
Tab 1: Add positions
- AAPL: 50 shares @ $150 = $7,500
- MSFT: 30 shares @ $300 = $9,000
- GOOGL: 40 shares @ $140 = $5,600
- ... (continue)

→ Click "Update Live Prices"
→ Click "Save Snapshot"
```

**Step 2: Set Targets**
```
Tab 2: Asset Allocation
- Stocks: 60%
- Bonds: 30%
- Commodities: 10%

→ Current allocation shows 80% stocks, 15% bonds, 5% commodities
→ "Calculate Rebalancing Trades" shows:
  - SELL $20,000 of Stocks
  - BUY $15,000 of Bonds
  - BUY $5,000 of Commodities
```

**Step 3: Check Correlations**
```
Tab 7: Correlation Analysis
→ Discovers: AAPL ↔ MSFT = 0.88 (very high!)
→ Action: Consider reducing one or diversifying into different sectors
```

**Step 4: Risk Assessment**
```
Tab 8: Monte Carlo (1000 sims, 252 days)
→ Results:
  - Current: $100k
  - Mean Final: $108k (+8%)
  - 5th Percentile: $82k (-18%)
  - 95th Percentile: $138k (+38%)
  - Probability of Profit: 68%

→ Decision: Acceptable risk for moderate investor
```

**Step 5: AI Feedback**
```
Tab 4: AI Mentor
→ Warnings:
  1. "Technology sector represents 45% - diversify"
  2. "AAPL is 25% of portfolio - reduce to 15%"
  3. "High correlation between AAPL and MSFT"

→ Action items:
  - Reduce AAPL position from 25% → 15%
  - Add healthcare or consumer stocks
  - Consider sector ETFs for diversification
```

**Step 6: Tax Optimization (Year-End)**
```
Tab 6: Tax-Loss Harvesting
→ Found: XYZ stock has -$3,000 loss
→ Action:
  - Sell XYZ (harvest loss)
  - Save ~$900 in taxes
  - Replace with sector ETF
→ Net result: Same exposure, $900 tax savings
```

**Step 7: Regular Monitoring**
```
Daily: Update prices + Save snapshot
Weekly: Check mentor feedback
Monthly: Review performance metrics (Tab 5)
Quarterly: Run Monte Carlo + Correlation analysis
Annually: Tax-loss harvesting + Rebalancing
```

---

## Quick Tips

### 🎯 Best Practices:
1. **Diversification**: Keep positions < 15% each
2. **Correlation**: Aim for < 0.7 between positions
3. **Rebalancing**: 10% threshold is standard
4. **Snapshots**: Save daily for accurate tracking
5. **Monte Carlo**: Run before major changes

### ⚠️ Watch Out For:
- High sector concentration (>30%)
- Very high correlation (>0.85) between large positions
- Wash sale violations when tax-loss harvesting
- Over-reliance on Monte Carlo (based on historical data)
- Ignoring rebalancing signals

### 💡 Pro Tips:
1. Update prices during market hours for real-time data
2. Save snapshots at consistent times (e.g., every Friday 4pm)
3. Run correlation analysis monthly to catch drift
4. Use half-Kelly sizing for safety margin
5. Check tax-loss opportunities in Q4 (tax year end)

---

## Technical Requirements

### Dependencies Installed:
```
yfinance>=0.2.18    # Live price data
scipy>=1.10.0       # Correlation & statistics
numpy>=1.23.0       # Monte Carlo simulations
pandas>=1.5.0       # Data manipulation
plotly>=5.17.0      # Interactive charts
```

### Performance Notes:
- **Live prices**: 5-min cache reduces API calls
- **Correlation**: Requires 30+ days historical data
- **Monte Carlo**: 1000 sims ~5 seconds, 5000 sims ~20 seconds
- **Historical data**: Fetched from Yahoo Finance (free tier limits apply)

---

## Need Help?

📖 **Full Documentation**: `docs/PORTFOLIO_MANAGEMENT_GUIDE.md`

🔗 **Data Source**: [Yahoo Finance](https://finance.yahoo.com/)

💬 **Support**: Check documentation eller spørg!

---

**Deployed til Streamlit Cloud** ✅

Gå til din app og se den nye **💼 Portfolio Management** side i menuen! 🚀
