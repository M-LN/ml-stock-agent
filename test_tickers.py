"""Test ticker symbols and PEG data quality"""
from fundamentals import analyze_fundamentals

# Test forskellige aktier
stocks = {
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Banking": ["JPM", "BAC", "WFC", "C"],
    "Danske": ["NOVO-B.CO", "DSV.CO", "MAERSK-B.CO"]
}

print("=" * 80)
print(f"{'Ticker':<12} | {'P/E':>6} | {'PEG':>6} | {'Source':<22} | {'Growth':>7}")
print("=" * 80)

for group_name, tickers in stocks.items():
    print(f"\n{group_name}:")
    print("-" * 80)
    
    for symbol in tickers:
        try:
            result = analyze_fundamentals(symbol)
            
            pe = result.get('pe')
            peg = result.get('peg')
            peg_source = result.get('peg_source', 'N/A')
            growth = result.get('growth', 0)
            
            pe_str = f"{pe:.1f}" if pe else "N/A"
            peg_str = f"{peg:.2f}" if peg else "N/A"
            
            print(f"{symbol:<12} | {pe_str:>6} | {peg_str:>6} | {peg_source:<22} | {growth:>6.1f}%")
            
        except Exception as e:
            print(f"{symbol:<12} | ERROR: {str(e)}")

print("\n" + "=" * 80)
print("Legend: N/A = Data not available")
print("=" * 80)
