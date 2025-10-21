# 💱 Multi-Currency Support - Portfolio Management

## Overview
Portfolio Management nu understøtter **multi-currency portfolios** med automatisk valutakonvertering. Du kan holde aktier i forskellige valutaer (DKK, USD, EUR, GBP osv.) og se alt konverteret til din foretrukne portfolio-valuta.

## Supported Currencies

### Primary Currencies:
- 🇩🇰 **DKK** (Danish Krone) - kr.
- 🇺🇸 **USD** (US Dollar) - $
- 🇪🇺 **EUR** (Euro) - €
- 🇬🇧 **GBP** (British Pound) - £
- 🇸🇪 **SEK** (Swedish Krona) - kr
- 🇳🇴 **NOK** (Norwegian Krone) - kr
- 🇨🇭 **CHF** (Swiss Franc) - CHF
- 🇯🇵 **JPY** (Japanese Yen) - ¥
- 🇨🇦 **CAD** (Canadian Dollar) - C$
- 🇦🇺 **AUD** (Australian Dollar) - A$

## How It Works

### 1. Set Your Portfolio Currency
**Location:** Sidebar → Portfolio Settings → Base Currency

```
Default: DKK (Danish Krone)
Changeable: Dropdown med alle supported currencies
```

**Når du ændrer currency:**
- Alle portfolio values konverteres automatisk
- Exchange rates hentes fra Yahoo Finance
- Visning opdateres i real-time

---

### 2. Automatic Currency Detection

Når du bruger **"Update Live Prices"**, detekteres currency automatisk baseret på symbol:

#### Exchange Suffixes:
- `.CO` → DKK (Copenhagen, Denmark)
- `.ST` → SEK (Stockholm, Sweden)
- `.OL` → NOK (Oslo, Norway)
- `.L` → GBP (London, UK)
- `.PA` → EUR (Paris, France)
- `.DE` → EUR (Frankfurt, Germany)
- `.MI` → EUR (Milan, Italy)
- `.AS` → EUR (Amsterdam, Netherlands)
- No suffix → USD (US markets)

#### Examples:
```
NOVO-B.CO  → Detected as DKK (Novo Nordisk, Copenhagen)
ERIC-B.ST  → Detected as SEK (Ericsson, Stockholm)
AAPL       → Detected as USD (Apple, US)
AIR.PA     → Detected as EUR (Airbus, Paris)
```

---

### 3. Manual Position Entry

Ved manuel tilføjelse af position:

**Fields:**
1. Symbol (fx `NOVO-B.CO`)
2. Shares (antal aktier)
3. Avg Cost (købs pris)
4. Current Price (nuværende pris)
5. Sector
6. Asset Class
7. **Currency** ⭐ NY - Dropdown til at vælge

**Default Currency:**
- Hvis din portfolio currency er DKK → defaults til DKK
- Ellers defaults til USD

---

### 4. Exchange Rate Fetching

**Source:** Yahoo Finance currency pairs (fx `USDDKK=X`)

**Caching:**
- Exchange rates cached i 1 time
- Reduces API calls
- Format: `{from_currency}_{to_currency}_{date_hour}`

**Error Handling:**
- Hvis exchange rate ikke kan hentes → bruger 1.0 og viser warning
- Fortsætter uden at crashe

**View Current Rates:**
Sidebar → "📊 Current Exchange Rates" (expandable)
- Viser alle rates til din portfolio currency
- Updates automatisk ved currency change

---

## Display Features

### Portfolio Summary

**Shows:**
```
💱 Portfolio Currency: DKK
Total Value: 750,000.00 kr.
Positions: 5
Total Gain/Loss: +125,000.00 kr. (+20.00%)
```

**Conversion:**
- Alle positions konverteres til portfolio currency
- Cost basis også converted
- Gain/Loss beregnet i portfolio currency

---

### Position Table

**Columns:**
1. Symbol
2. **Currency** ⭐ - Shows original currency (USD, DKK, EUR)
3. Shares
4. Avg Cost - I original currency
5. Current Price - I original currency
6. **Market Value (DKK)** ⭐ - Converted til portfolio currency
7. Gain/Loss - I portfolio currency
8. Gain/Loss %
9. Allocation %
10. Sector
11. Asset Class

**Example:**
```
Symbol: AAPL
Currency: USD
Shares: 10
Avg Cost: $150.00
Current Price: $175.00
Market Value (DKK): 12,250.00 kr.  ← Converted fra $1,750
Gain/Loss: +1,750.00 kr.
```

---

### Currency Exposure Breakdown

**New Section:** "💱 Currency Exposure"

Vises kun hvis du har positions i flere currencies.

**Shows:**
```
Currency          Value              Percentage
DKK kr.          500,000.00 kr.     66.7%
USD $            200,000.00 kr.     26.7%
EUR €             50,000.00 kr.      6.7%
```

**Use Case:**
- Se hvor meget currency exposure du har
- Identify currency concentration risk
- Balance mellem currencies

---

## Use Cases

### Use Case 1: Danish Investor with US Stocks

**Setup:**
```
Portfolio Currency: DKK
Positions:
- Novo Nordisk (NOVO-B.CO): 100 shares @ 850 kr.
- Apple (AAPL): 50 shares @ $175
- Microsoft (MSFT): 30 shares @ $380
```

**Result:**
```
Total Value: 450,000 kr.
- Novo Nordisk: 85,000 kr. (18.9%)
- Apple: 61,250 kr. (13.6%) ← Converted fra $8,750
- Microsoft: 79,800 kr. (17.7%) ← Converted fra $11,400

Currency Exposure:
- DKK: 85,000 kr. (18.9%)
- USD: 365,000 kr. (81.1%)
```

---

### Use Case 2: Multi-Region Portfolio

**Setup:**
```
Portfolio Currency: DKK
Positions:
- Maersk (MAERSK-B.CO): DKK
- Novo Nordisk (NOVO-B.CO): DKK
- Ericsson (ERIC-B.ST): SEK
- Apple (AAPL): USD
- Airbus (AIR.PA): EUR
```

**Result:**
```
Currency Exposure:
- DKK: 40%
- USD: 30%
- EUR: 20%
- SEK: 10%

→ Diversified currency exposure
→ Reduces single-currency risk
```

---

### Use Case 3: Update Prices with Auto-Detection

**Workflow:**
```
1. Add positions manually (set currency)
2. Click "🔄 Update Live Prices"
3. System fetches:
   - Current price i original currency
   - Detects correct currency fra symbol
   - Updates currency field
4. Click "💾 Save Snapshot"
5. All values auto-converted til DKK
```

---

## Technical Details

### Exchange Rate Fetching

**Yahoo Finance Format:**
```python
pair = f"{from_currency}{to_currency}=X"
# Example: USDDKK=X for USD to DKK
```

**Supported Pairs:**
- USDDKK=X (USD → DKK)
- EURDKK=X (EUR → DKK)
- GBPDKK=X (GBP → DKK)
- SEKDKK=X (SEK → DKK)
- DKKUSD=X (DKK → USD)
- etc. (any major pair)

**Cache Key:**
```python
f"{from_currency}_{to_currency}_{YYYYMMDD_HH}"
# Example: USD_DKK_20251021_14
```

---

### Conversion Function

```python
def convert_to_portfolio_currency(
    amount: float, 
    from_currency: str, 
    portfolio_currency: str
) -> float:
    """Convert amount from one currency to portfolio currency"""
    
    if from_currency == portfolio_currency:
        return amount  # No conversion needed
    
    rate = fetch_exchange_rate(from_currency, portfolio_currency)
    return amount * rate
```

**Example:**
```
Amount: $1,000
From: USD
To: DKK
Rate: 7.0 (1 USD = 7.0 DKK)
Result: 7,000 kr.
```

---

### Format Currency Function

```python
def format_currency(amount: float, currency: str) -> str:
    """Format amount with currency symbol"""
    
    # DKK/SEK/NOK: "amount kr."
    if currency in ['DKK', 'SEK', 'NOK']:
        return f"{amount:,.2f} kr."
    
    # USD: "$amount"
    elif currency == 'USD':
        return f"${amount:,.2f}"
    
    # EUR: "€amount"
    elif currency == 'EUR':
        return f"€{amount:,.2f}"
```

---

## Persistence

### Save Portfolio

**Saved Fields:**
```json
{
  "portfolio": {
    "positions": [
      {
        "symbol": "AAPL",
        "shares": 10,
        "avg_cost": 150.0,
        "current_price": 175.0,
        "currency": "USD",  ← Saved
        "sector": "Technology",
        "asset_class": "stocks"
      }
    ]
  },
  "portfolio_currency": "DKK",  ← Saved
  "performance_snapshots": [...],
  "saved_at": "2025-10-21T14:30:00"
}
```

### Load Portfolio

**Restores:**
- All positions med currency
- Portfolio currency preference
- Performance snapshots
- Exchange rates re-fetched on load

---

## Best Practices

### 1. Currency Hedging
```
Problem: High USD exposure (80%) i DKK portfolio
Solution: 
- Buy DKK or EUR assets
- Use currency-hedged ETFs
- Balance exposure til ~30-40% per currency
```

### 2. Exchange Rate Risk
```
Scenario: Du har 500,000 kr. i USD assets
Risk: Hvis USD falder 10%, mister du 50,000 kr.
Mitigation:
- Diversify across currencies
- Use forward contracts (advanced)
- Balance with local assets
```

### 3. Cost Basis Tracking
```
Important: Track avg_cost i original currency
Why: 
- Accurate P&L calculation
- Tax reporting (meist i original currency)
- Historical accuracy
```

### 4. Regular Updates
```
Workflow:
1. Update prices weekly
2. Check exchange rates monthly
3. Rebalance if currency exposure > 50%
4. Save snapshots for tracking
```

---

## Limitations

### 1. Exchange Rate Data
- **Source:** Yahoo Finance (free tier)
- **Delay:** May have slight delays (few minutes)
- **Availability:** Major pairs only
- **Fallback:** Uses 1.0 if rate unavailable

### 2. No Real-Time Forex
- Rates cached for 1 hour
- Not tick-by-tick updates
- Sufficient for portfolio tracking
- Not for day trading

### 3. Crypto Not Included
- Currently only fiat currencies
- No BTC, ETH, etc.
- Can add crypto as "commodities" in USD equivalent

### 4. Tax Implications Not Calculated
- Shows converted values only
- Doesn't calculate forex gains/losses
- Consult tax professional for forex taxation

---

## Future Enhancements

Potential additions:
- [ ] Crypto currency support (BTC, ETH)
- [ ] Real-time forex rates (WebSocket)
- [ ] Currency hedging recommendations
- [ ] Forex gain/loss tracking (separate fra stock gains)
- [ ] Forward rate predictions
- [ ] Currency correlation analysis
- [ ] Currency pair trading suggestions
- [ ] Tax reporting med forex adjustments

---

## Example: Complete Workflow

### Step 1: Setup
```
1. Sidebar → Set Portfolio Currency til DKK
2. Verify exchange rates i "Current Exchange Rates"
```

### Step 2: Add Positions
```
Option A - Manual:
- Add NOVO-B.CO (100 shares @ 850 kr., DKK)
- Add AAPL (50 shares @ $175, USD)
- Add AIR.PA (20 shares @ €140, EUR)

Option B - Auto-detect:
- Add symbols only
- Click "Update Live Prices"
- Currency detected automatically
```

### Step 3: Review
```
Portfolio Summary:
- Total Value: ~450,000 kr. (all converted)
- Positions: 3

Currency Exposure:
- DKK: 85,000 kr. (18.9%)
- USD: 61,250 kr. (13.6%)
- EUR: 19,600 kr. (4.4%)
```

### Step 4: Monitor
```
1. Check exchange rates monthly
2. If USD/DKK rate changes 5%+:
   - Portfolio value changes accordingly
   - Consider rebalancing
3. Track performance i DKK terms
```

### Step 5: Rebalance
```
If currency exposure > 50% i én currency:
→ Buy assets i andre currencies
→ Maintain balance (~30-40% max per currency)
```

---

## Support

**Common Questions:**

**Q: Hvorfor vises priser stadig i USD når jeg har valgt DKK?**
A: Avg Cost og Current Price vises i original currency, men Market Value er converted til DKK.

**Q: Hvordan opdaterer jeg exchange rates?**
A: Rates auto-fetches ved "Update Live Prices" eller currency change. Cache i 1 time.

**Q: Kan jeg se historical exchange rates?**
A: Ikke direkte, men performance snapshots gemmer converted values over tid.

**Q: Hvad hvis Yahoo Finance ikke har en exchange rate?**
A: System viser warning og bruger 1.0 (no conversion). Check symbol format.

---

Built with ❤️ for multi-currency portfolios.
Yahoo Finance integration for real-time exchange rates.
