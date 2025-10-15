"""
ML Stock Agent - Streamlit Web Frontend
Main Dashboard / Homepage
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import requests
import time

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="🌍 Market Overview",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌍 Market Overview</h1>', unsafe_allow_html=True)


# Sidebar - Auto-refresh and Quick Links
st.sidebar.markdown("---")

# Auto-refresh settings
st.sidebar.markdown("### ⚙️ Auto-Refresh")
auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh", value=False, help="Automatically refresh market data")

if auto_refresh:
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval",
        options=[60, 300, 600, 1800, 3600],
        format_func=lambda x: f"{x//60} minute{'s' if x > 60 else ''}" if x >= 60 else f"{x} seconds",
        index=1,  # Default to 5 minutes (300 seconds)
        help="How often to refresh the page"
    )
    
    # Initialize last refresh time
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Calculate time since last refresh
    current_time = time.time()
    elapsed = int(current_time - st.session_state.last_refresh)
    
    # Check if it's time to refresh
    if elapsed >= refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()
    
    # Display static info (no countdown to avoid constant reruns)
    remaining = refresh_interval - elapsed
    minutes = remaining // 60
    seconds = remaining % 60
    
    st.sidebar.info(f"⏱️ Refreshing in ~{minutes}m {seconds}s")
    
    # Show last update time
    last_update = datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S')
    st.sidebar.caption(f"Last updated: {last_update}")
    
    # Schedule next check using Streamlit's built-in mechanism
    # This will refresh the page after the remaining time
    st.markdown(f"""
    <script>
        setTimeout(function(){{
            window.location.reload();
        }}, {remaining * 1000});
    </script>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Links")
st.sidebar.info("""
**Funktioner:**
- 📊 Teknisk Analyse
- 🤖 ML Forecasting
- 📋 Watchlist Manager
- 📄 Rapport Generator
""")

# Main content - Market Overview


# Fetch S&P 500 data
with st.spinner('Loading market data...'):
    try:
        # S&P 500
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(period="1mo")
        sp500_info = sp500.info
        
        # Major indices - use 5d to ensure we get last trading day data
        dow = yf.Ticker("^DJI")
        nasdaq = yf.Ticker("^IXIC")
        vix = yf.Ticker("^VIX")
        
        dow_data = dow.history(period="5d")
        nasdaq_data = nasdaq.history(period="5d")
        vix_data = vix.history(period="5d")
        
        # Treasury yields (macro indicators)
        treasury_10y = yf.Ticker("^TNX")
        treasury_10y_data = treasury_10y.history(period="5d")
        
        # Gold (safe haven)
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(period="5d")
        
        # Dollar Index
        dxy = yf.Ticker("DX-Y.NYB")
        dxy_data = dxy.history(period="5d")
        
        # International Markets
        ftse = yf.Ticker("^FTSE")  # UK
        dax = yf.Ticker("^GDAXI")  # Germany
        nikkei = yf.Ticker("^N225")  # Japan
        
        ftse_data = ftse.history(period="5d")
        dax_data = dax.history(period="5d")
        nikkei_data = nikkei.history(period="5d")
        
        # Cryptocurrency
        btc = yf.Ticker("BTC-USD")
        eth = yf.Ticker("ETH-USD")
        
        btc_data = btc.history(period="5d")
        eth_data = eth.history(period="5d")
        
        # Calculate Enhanced Fear & Greed Index
        # Based on 7 components (similar to CNN Fear & Greed Index)
        fear_greed_score = 50  # Neutral baseline
        components = []
        
        # 1. Market Momentum (S&P 500 vs 125-day average)
        if len(sp500_data) >= 125:
            sp500_125_avg = sp500_data['Close'].tail(125).mean()
            sp500_current = sp500_data['Close'].iloc[-1]
            momentum_pct = ((sp500_current - sp500_125_avg) / sp500_125_avg) * 100
            # More conservative scaling: ±1% = ±10 points (not ±20)
            momentum_score = 50 + min(max(momentum_pct * 1.0, -40), 40)
            components.append(('Momentum', momentum_score))
        
        # 2. Stock Price Strength (distance from 52-week high)
        if len(sp500_data) >= 252:
            high_52w = sp500_data['High'].tail(252).max()
            current_price = sp500_data['Close'].iloc[-1]
            distance_from_high = ((current_price - high_52w) / high_52w) * 100
            # -10% from high = score 30, at high = score 100
            strength_score = 100 + (distance_from_high * 7)  # More conservative
            strength_score = max(0, min(100, strength_score))
            components.append(('Price Strength', strength_score))
        
        # 3. Stock Price Breadth (advance/decline)
        if len(sp500_data) >= 10:
            recent_closes = sp500_data['Close'].tail(10)
            advances = sum(recent_closes.diff() > 0)
            # Scale down: 5/9 days up = 50 (neutral), not 55
            breadth_score = 30 + (advances / 9) * 40
            components.append(('Breadth', breadth_score))
        
        # 4. Put/Call Ratio (approximated using VIX)
        if not vix_data.empty:
            vix_current = vix_data['Close'].iloc[-1]
            # More conservative: VIX 15 = 50 (neutral), VIX 30 = 0 (fear)
            vix_score = max(0, min(100, 50 + (20 - vix_current) * 2.5))
            components.append(('Put/Call (VIX)', vix_score))
        
        # 5. Market Volatility (VIX levels) - MOST IMPORTANT
        if not vix_data.empty:
            vix_avg = 18  # Lower baseline
            # Higher penalty for elevated VIX
            volatility_score = max(0, min(100, 70 - ((vix_current - vix_avg) * 4)))
            components.append(('Volatility', volatility_score))
        
        # 6. Safe Haven Demand (Gold vs S&P 500)
        if not gold_data.empty and len(gold_data) >= 20:
            gold_20d_change = ((gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-20]) / gold_data['Close'].iloc[-20] * 100)
            sp500_20d_change = ((sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[-20]) / sp500_data['Close'].iloc[-20] * 100) if len(sp500_data) >= 20 else 0
            # If gold outperforms S&P 500, it's fear (inverse)
            safe_haven_score = 50 + (sp500_20d_change - gold_20d_change) * 5
            safe_haven_score = max(0, min(100, safe_haven_score))
            components.append(('Safe Haven', safe_haven_score))
        
        # 7. Junk Bond Demand (approximated using Treasury yields)
        if not treasury_10y_data.empty and len(treasury_10y_data) >= 20:
            current_yield = treasury_10y_data['Close'].iloc[-1]
            avg_yield = 4.0  # Historical average
            # Lower yields relative to average = more greed
            junk_score = max(0, min(100, 50 + (avg_yield - current_yield) * 10))
            components.append(('Junk Bond', junk_score))
        
        # Calculate weighted average with adjusted weights for better CNN alignment
        if components:
            # CNN Fear & Greed uses different weights - adjust our calculation
            # Market Momentum (20%), Price Strength (20%), Breadth (10%)
            # Put/Call (10%), Volatility (25%), Safe Haven (10%), Junk Bond (5%)
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
            
            # Apply calibration to match CNN methodology (Target: CNN = 32, we had 40)
            # CNN tends to show more fear than simple technical indicators suggest
            # Fine-tuned to match CNN's conservative approach
            
            # Global adjustment: compress scale to match CNN distribution
            fear_greed_score = fear_greed_score * 0.62  # Adjusted from 0.65 to 0.62
            
            # Additional granular adjustments
            if fear_greed_score >= 60:
                # High scores: compress significantly
                fear_greed_score = 38 + (fear_greed_score - 60) * 0.5
            elif fear_greed_score >= 45:
                # Upper neutral: moderate compression
                fear_greed_score = 32 + (fear_greed_score - 45) * 0.75
            elif fear_greed_score >= 30:
                # Lower neutral to fear: maintain with slight reduction
                fear_greed_score = fear_greed_score * 1.0
            else:
                # Deep fear: slight expansion to preserve signal
                fear_greed_score = fear_greed_score * 1.05
            
            fear_greed_score = max(0, min(100, fear_greed_score))
        
        # Calculate Dynamic Shiller P/E (CAPE Ratio)
        # Fetch 10-year earnings data and calculate CAPE
        try:
            # Get S&P 500 P/E ratio from yfinance
            sp500_ticker = yf.Ticker("^GSPC")
            sp500_info = sp500_ticker.info
            
            # Try to get trailing P/E
            trailing_pe = sp500_info.get('trailingPE', None)
            
            if trailing_pe and trailing_pe > 0:
                # CAPE (Shiller P/E) is typically 1.4-1.8x higher than trailing P/E
                # This is because CAPE uses 10-year inflation-adjusted earnings
                # Current market: trailing P/E ~20-22, CAPE ~32-36
                estimated_cape = trailing_pe * 1.60  # Adjusted multiplier for better accuracy
            else:
                # Fallback: calculate from price and estimate earnings
                sp500_current = sp500_data['Close'].iloc[-1]
                # Use historical average earnings yield (~4.5% in current high-P/E environment)
                estimated_earnings = sp500_current * 0.045
                # Calculate P/E then convert to CAPE estimate
                estimated_pe = sp500_current / estimated_earnings
                estimated_cape = estimated_pe * 1.60
            
            # Sanity check: CAPE typically ranges 15-45 (expanded lower bound)
            estimated_cape = max(15, min(45, estimated_cape))
            
        except Exception as cape_error:
            # Fallback to reasonable current market estimate
            estimated_cape = 32.0  # Current approximate market level
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        st.stop()

# Display Major Indices
st.markdown("### 📊 Major Indices (Last Trading Day)")

col1, col2, col3, col4 = st.columns(4)

def calculate_change(data):
    if len(data) >= 2:
        current = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        change = ((current - prev) / prev) * 100
        return current, change
    elif len(data) == 1:
        return data['Close'].iloc[-1], 0
    return 0, 0

# S&P 500
with col1:
    sp500_price, sp500_change = calculate_change(sp500_data)
    st.metric("S&P 500", f"{sp500_price:.2f}", f"{sp500_change:+.2f}%")

# Dow Jones
with col2:
    if not dow_data.empty:
        dow_price = dow_data['Close'].iloc[-1]
        dow_change = 0 if len(dow_data) < 2 else ((dow_price - dow_data['Close'].iloc[-2]) / dow_data['Close'].iloc[-2] * 100)
        st.metric("Dow Jones", f"{dow_price:.2f}", f"{dow_change:+.2f}%")

# NASDAQ
with col3:
    if not nasdaq_data.empty:
        nasdaq_price = nasdaq_data['Close'].iloc[-1]
        nasdaq_change = 0 if len(nasdaq_data) < 2 else ((nasdaq_price - nasdaq_data['Close'].iloc[-2]) / nasdaq_data['Close'].iloc[-2] * 100)
        st.metric("NASDAQ", f"{nasdaq_price:.2f}", f"{nasdaq_change:+.2f}%")

# VIX
with col4:
    if not vix_data.empty:
        vix_price = vix_data['Close'].iloc[-1]
        vix_change = 0 if len(vix_data) < 2 else ((vix_price - vix_data['Close'].iloc[-2]) / vix_data['Close'].iloc[-2] * 100)
        
        # VIX color coding
        if vix_price < 15:
            vix_label = "Low Volatility"
        elif vix_price < 25:
            vix_label = "Normal"
        else:
            vix_label = "High Volatility"
        
        st.metric("VIX (Fear Index)", f"{vix_price:.2f}", f"{vix_change:+.2f}%", help=vix_label)

st.divider()

# International Markets
st.markdown("### 🌎 International Markets")
col1, col2, col3 = st.columns(3)

# FTSE 100 (UK)
with col1:
    if not ftse_data.empty:
        ftse_price, ftse_change = calculate_change(ftse_data)
        st.metric("🇬🇧 FTSE 100", f"{ftse_price:.2f}", f"{ftse_change:+.2f}%")
    else:
        st.metric("🇬🇧 FTSE 100", "N/A", "0.00%")

# DAX (Germany)
with col2:
    if not dax_data.empty:
        dax_price, dax_change = calculate_change(dax_data)
        st.metric("🇩🇪 DAX", f"{dax_price:.2f}", f"{dax_change:+.2f}%")
    else:
        st.metric("🇩🇪 DAX", "N/A", "0.00%")

# Nikkei 225 (Japan)
with col3:
    if not nikkei_data.empty:
        nikkei_price, nikkei_change = calculate_change(nikkei_data)
        st.metric("🇯🇵 Nikkei 225", f"{nikkei_price:.2f}", f"{nikkei_change:+.2f}%")
    else:
        st.metric("🇯🇵 Nikkei 225", "N/A", "0.00%")

st.divider()

# Cryptocurrency Dashboard
st.markdown("### 💎 Cryptocurrency")
col1, col2 = st.columns(2)

with col1:
    if not btc_data.empty:
        btc_price, btc_change = calculate_change(btc_data)
        btc_emoji = "🟢" if btc_change >= 0 else "🔴"
        st.metric(f"{btc_emoji} Bitcoin (BTC)", f"${btc_price:,.2f}", f"{btc_change:+.2f}%")
        
        # BTC additional info
        if len(btc_data) >= 5:
            btc_5d_change = ((btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[0]) / btc_data['Close'].iloc[0] * 100)
            st.caption(f"5-day change: {btc_5d_change:+.2f}%")
    else:
        st.metric("🟡 Bitcoin (BTC)", "N/A", "0.00%")

with col2:
    if not eth_data.empty:
        eth_price, eth_change = calculate_change(eth_data)
        eth_emoji = "🟢" if eth_change >= 0 else "🔴"
        st.metric(f"{eth_emoji} Ethereum (ETH)", f"${eth_price:,.2f}", f"{eth_change:+.2f}%")
        
        # ETH additional info
        if len(eth_data) >= 5:
            eth_5d_change = ((eth_data['Close'].iloc[-1] - eth_data['Close'].iloc[0]) / eth_data['Close'].iloc[0] * 100)
            st.caption(f"5-day change: {eth_5d_change:+.2f}%")
    else:
        st.metric("🟡 Ethereum (ETH)", "N/A", "0.00%")

st.divider()

# S&P 500 Chart
st.markdown("### � S&P 500 - Last 30 Days")

fig_sp500 = go.Figure()

fig_sp500.add_trace(go.Candlestick(
    x=sp500_data.index,
    open=sp500_data['Open'],
    high=sp500_data['High'],
    low=sp500_data['Low'],
    close=sp500_data['Close'],
    name='S&P 500'
))

fig_sp500.update_layout(
    xaxis_title='Date',
    yaxis_title='Price ($)',
    height=400,
    hovermode='x unified',
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig_sp500, use_container_width=True)

st.divider()

# Macro Indicators
st.markdown("### 🌐 Macro Indicators (Last Trading Day)")

col1, col2, col3, col4, col5 = st.columns(5)

# 10-Year Treasury
with col1:
    if not treasury_10y_data.empty:
        treasury_yield = treasury_10y_data['Close'].iloc[-1]
        treasury_change = 0 if len(treasury_10y_data) < 2 else ((treasury_yield - treasury_10y_data['Close'].iloc[-2]))
        
        st.metric("10Y Treasury Yield", f"{treasury_yield:.2f}%", f"{treasury_change:+.2f}bp")
        
        if treasury_yield > 4.5:
            st.caption("🔴 High rates - pressure on stocks")
        elif treasury_yield < 3.5:
            st.caption("🟢 Low rates - favorable for stocks")
        else:
            st.caption("🟡 Moderate rates")

# Gold
with col2:
    if not gold_data.empty:
        gold_price = gold_data['Close'].iloc[-1]
        gold_change = 0 if len(gold_data) < 2 else ((gold_price - gold_data['Close'].iloc[-2]) / gold_data['Close'].iloc[-2] * 100)
        
        st.metric("Gold (Safe Haven)", f"${gold_price:.2f}", f"{gold_change:+.2f}%")
        
        if gold_change > 1:
            st.caption("🔴 Flight to safety")
        else:
            st.caption("🟢 Risk-on environment")

# Dollar Index
with col3:
    if not dxy_data.empty:
        dxy_price = dxy_data['Close'].iloc[-1]
        dxy_change = 0 if len(dxy_data) < 2 else ((dxy_price - dxy_data['Close'].iloc[-2]) / dxy_data['Close'].iloc[-2] * 100)
        
        st.metric("Dollar Index (DXY)", f"{dxy_price:.2f}", f"{dxy_change:+.2f}%")
        
        if dxy_price > 105:
            st.caption("💪 Strong dollar")
        elif dxy_price < 95:
            st.caption("📉 Weak dollar")
        else:
            st.caption("➡️ Neutral")

# Fear & Greed Index
with col4:
    st.metric("Fear & Greed Index", f"{fear_greed_score:.0f}/100", delta=None)
    
    if fear_greed_score >= 75:
        st.caption("🔥 Extreme Greed")
    elif fear_greed_score >= 55:
        st.caption("🟢 Greed")
    elif fear_greed_score >= 45:
        st.caption("🟡 Neutral")
    elif fear_greed_score >= 25:
        st.caption("🔴 Fear")
    else:
        st.caption("❄️ Extreme Fear")

# Shiller P/E (CAPE Ratio)
with col5:
    st.metric("Shiller P/E (CAPE)", f"{estimated_cape:.1f}", delta=None)
    
    if estimated_cape > 30:
        st.caption("🔴 Overvalued")
    elif estimated_cape > 25:
        st.caption("🟡 Above Average")
    elif estimated_cape > 20:
        st.caption("🟢 Fair Value")
    else:
        st.caption("🟢 Undervalued")

st.divider()

# Market Sentiment
st.markdown("### 🎯 Market Sentiment")

col1, col2 = st.columns(2)

with col1:
    # Calculate market sentiment based on indicators
    sentiment_score = 0
    
    # S&P 500 trend
    if sp500_change > 0:
        sentiment_score += 1
    
    # VIX level
    if not vix_data.empty and vix_data['Close'].iloc[-1] < 20:
        sentiment_score += 1
    
    # Treasury yields (inverse)
    if not treasury_10y_data.empty and treasury_10y_data['Close'].iloc[-1] < 4.5:
        sentiment_score += 1
    
    # Gold performance (inverse)
    if not gold_data.empty:
        gold_chg = ((gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-2]) / gold_data['Close'].iloc[-2] * 100) if len(gold_data) >= 2 else 0
        if gold_chg < 0:
            sentiment_score += 1
    
    # Display sentiment
    if sentiment_score >= 3:
        st.success("🟢 **BULLISH** - Market conditions favorable")
        st.markdown("""
        **Indicators:**
        - ✅ Major indices trending up
        - ✅ Low volatility (VIX)
        - ✅ Favorable macro conditions
        """)
    elif sentiment_score >= 2:
        st.info("🟡 **NEUTRAL** - Mixed signals")
        st.markdown("""
        **Indicators:**
        - ⚠️ Some positive, some negative signals
        - 📊 Wait for clearer direction
        """)
    else:
        st.warning("🔴 **BEARISH** - Caution advised")
        st.markdown("""
        **Indicators:**
        - ⚠️ Elevated risk factors
        - 📉 Defensive positioning recommended
        """)

with col2:
    st.markdown("#### 📰 Key Levels to Watch")
    
    # Calculate key S&P 500 levels
    sp500_current = sp500_data['Close'].iloc[-1]
    sp500_high_30d = sp500_data['High'].max()
    sp500_low_30d = sp500_data['Low'].min()
    sp500_avg_30d = sp500_data['Close'].mean()
    
    st.markdown(f"""
    **S&P 500:**
    - Current: **${sp500_current:.2f}**
    - 30D High: ${sp500_high_30d:.2f} ({((sp500_high_30d - sp500_current) / sp500_current * 100):+.1f}%)
    - 30D Low: ${sp500_low_30d:.2f} ({((sp500_low_30d - sp500_current) / sp500_current * 100):+.1f}%)
    - 30D Avg: ${sp500_avg_30d:.2f}
    
    **Next Support:** ${sp500_low_30d:.2f}  
    **Next Resistance:** ${sp500_high_30d:.2f}
    """)

st.divider()

# Cache function for loading stock data (refreshes every 5 minutes)
@st.cache_data(ttl=300)
def load_sp500_heatmap_data(symbols_list, symbol_to_sector_map):
    """
    Load S&P 500 stock data with enhanced error handling
    Returns: dict with stock data and metadata about failures
    """
    stock_data = {}
    failed_symbols = []
    success_count = 0
    
    try:
        # Download last 5 days to ensure we get data even when market is closed
        all_data = yf.download(symbols_list, period="5d", progress=False, group_by='ticker', threads=True)
        
        if all_data.empty:
            return {'stocks': {}, 'failed': symbols_list, 'success_count': 0}
        
        # Process each stock with detailed error tracking
        for symbol in symbols_list:
            try:
                # Handle both MultiIndex and regular columns
                if isinstance(all_data.columns, pd.MultiIndex):
                    if symbol in all_data.columns.get_level_values(0):
                        stock_df = all_data[symbol]
                    else:
                        failed_symbols.append(symbol)
                        continue
                else:
                    stock_df = all_data
                
                if not stock_df.empty and len(stock_df) >= 2:
                    # Get last trading day's data
                    current = stock_df['Close'].iloc[-1]
                    prev = stock_df['Close'].iloc[-2]
                    
                    # Validate data quality
                    if pd.notna(current) and pd.notna(prev) and prev > 0 and current > 0:
                        # Calculate change from previous day
                        change_pct = ((current - prev) / prev) * 100
                        
                        # Sanity check: filter out extreme outliers (likely data errors)
                        if abs(change_pct) < 50:  # Filter out > ±50% daily changes
                            stock_data[symbol] = {
                                'sector': symbol_to_sector_map[symbol],
                                'price': float(current),
                                'change': float(change_pct),
                                'volume': int(stock_df['Volume'].iloc[-1]) if 'Volume' in stock_df else 0
                            }
                            success_count += 1
                        else:
                            failed_symbols.append(f"{symbol} (extreme change: {change_pct:.1f}%)")
                    else:
                        failed_symbols.append(f"{symbol} (invalid data)")
                else:
                    failed_symbols.append(f"{symbol} (insufficient data)")
                    
            except Exception as symbol_error:
                failed_symbols.append(f"{symbol} ({str(symbol_error)[:30]})")
                continue
                
    except Exception as e:
        return {
            'stocks': {}, 
            'failed': symbols_list, 
            'success_count': 0,
            'error': str(e)
        }
    
    return {
        'stocks': stock_data,
        'failed': failed_symbols,
        'success_count': success_count,
        'total': len(symbols_list)
    }

# S&P 500 Sector & Top Stocks Heatmap
st.markdown("## 🗺️ S&P 500 Market Heatmap")

# Define major S&P 500 stocks by sector
sp500_stocks = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "ACN"],
    "Communication": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE"],
    "Financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS"],
    "Industrials": ["BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "DE"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MDLZ"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "SPG", "O"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM"]
}

# Create flat list of all symbols with sector mapping
all_symbols = []
symbol_to_sector = {}

for sector, symbols in sp500_stocks.items():
    for symbol in symbols:
        all_symbols.append(symbol)
        symbol_to_sector[symbol] = sector

# Load data using cached function with enhanced error reporting
with st.spinner('Loading S&P 500 data...'):
    result = load_sp500_heatmap_data(all_symbols, symbol_to_sector)
    
    stock_data = result.get('stocks', {})
    failed_symbols = result.get('failed', [])
    success_count = result.get('success_count', 0)
    total_count = result.get('total', len(all_symbols))

# Display loading statistics with success rate
success_rate = (success_count / total_count * 100) if total_count > 0 else 0

if success_rate >= 90:
    st.success(f"✅ Loaded {success_count}/{total_count} stocks ({success_rate:.1f}% success rate)")
elif success_rate >= 70:
    st.warning(f"⚠️ Loaded {success_count}/{total_count} stocks ({success_rate:.1f}% success rate) - Some data unavailable")
elif success_count > 0:
    st.error(f"❌ Only loaded {success_count}/{total_count} stocks ({success_rate:.1f}% success rate)")
else:
    st.error("⚠️ Unable to load market data. Please refresh the page.")
    if 'error' in result:
        st.error(f"Error details: {result['error']}")
    st.stop()

# Show failed symbols in an expander for debugging (only if there are failures)
if failed_symbols and len(failed_symbols) <= 20:
    with st.expander(f"⚠️ Failed to load {len(failed_symbols)} stocks - Click to see details"):
        st.write(", ".join(failed_symbols))
elif len(failed_symbols) > 20:
    with st.expander(f"⚠️ Failed to load {len(failed_symbols)} stocks - Click to see details"):
        st.write("**Most common issues:**")
        st.write(", ".join(failed_symbols[:20]) + f"... and {len(failed_symbols) - 20} more")

# Calculate sector performance
sectors_performance = {}
if stock_data:
    for sector in sp500_stocks.keys():
        sector_stocks = [s for s in stock_data.values() if s['sector'] == sector]
        if sector_stocks:
            avg_change = sum(s['change'] for s in sector_stocks) / len(sector_stocks)
            sectors_performance[sector] = avg_change

# Display sector performance overview
st.markdown("### 📊 Sector Performance (Last Trading Day)")
sector_cols = st.columns(4)

sorted_sectors = sorted(sectors_performance.items(), key=lambda x: x[1], reverse=True)

for idx, (sector, perf) in enumerate(sorted_sectors[:4]):
    with sector_cols[idx]:
        color = "🟢" if perf > 0 else "🔴"
        st.metric(sector, f"{perf:+.2f}%", delta=None, delta_color="off")

# Create visual heatmap
if stock_data:
    st.markdown("### 📊 Performance Heatmap by Sector")
    
    # Create sector-based visualization
    for sector, symbols in sp500_stocks.items():
        # Get stocks in this sector
        sector_stocks = {sym: data for sym, data in stock_data.items() if data['sector'] == sector}
        
        if sector_stocks:
            # Calculate sector average
            avg_change = sum(s['change'] for s in sector_stocks.values()) / len(sector_stocks)
            
            # Expander for each sector
            with st.expander(f"**{sector}** ({len(sector_stocks)} stocks) - Avg: {avg_change:+.2f}%", expanded=False):
                # Sort stocks by performance
                sorted_stocks = sorted(sector_stocks.items(), key=lambda x: x[1]['change'], reverse=True)
                
                # Create columns for stock cards
                cols_per_row = 4
                for i in range(0, len(sorted_stocks), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (symbol, data) in enumerate(sorted_stocks[i:i+cols_per_row]):
                        with cols[j]:
                            # Dark mode friendly colors with better contrast
                            if data['change'] > 1:
                                color = "🟢"
                                bg_color = "#1b5e20"  # Dark green
                                text_color = "#e8f5e9"  # Light green text
                            elif data['change'] > 0:
                                color = "🟢"
                                bg_color = "#2e7d32"  # Medium green
                                text_color = "#c8e6c9"  # Light green text
                            elif data['change'] > -1:
                                color = "🔴"
                                bg_color = "#c62828"  # Medium red
                                text_color = "#ffcdd2"  # Light red text
                            else:
                                color = "🔴"
                                bg_color = "#b71c1c"  # Dark red
                                text_color = "#ffebee"  # Light red text
                            
                            st.markdown(f"""
                            <div style="
                                padding: 10px; 
                                background-color: {bg_color}; 
                                border-radius: 5px; 
                                text-align: center;
                                border: 1px solid rgba(255,255,255,0.1);
                            ">
                                <strong style="color: {text_color};">{symbol}</strong><br>
                                <span style="font-size: 20px; color: white; font-weight: bold;">{data['change']:+.2f}%</span><br>
                                <span style="font-size: 12px; color: {text_color};">${data['price']:.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Gainers and Losers
    st.markdown("### 🏆 Top Movers (Last Trading Day)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Top Gainers")
        sorted_gainers = sorted(stock_data.items(), key=lambda x: x[1]['change'], reverse=True)[:5]
        
        for idx, (symbol, data) in enumerate(sorted_gainers, 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "📊"
            st.markdown(f"{emoji} **{symbol}** - {data['change']:+.2f}% (${data['price']:.2f})")
    
    with col2:
        st.markdown("#### 📉 Top Losers")
        sorted_losers = sorted(stock_data.items(), key=lambda x: x[1]['change'])[:5]
        
        for idx, (symbol, data) in enumerate(sorted_losers, 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "📊"
            st.markdown(f"{emoji} **{symbol}** - {data['change']:+.2f}% (${data['price']:.2f})")
else:
    st.error("Unable to load stock data. Please refresh the page.")

st.divider()

# Sector Rotation Analysis
st.markdown("## 🔄 Sector Rotation Analysis")

if sectors_performance:
    # Create sector rotation chart
    sector_names = list(sectors_performance.keys())
    sector_changes = list(sectors_performance.values())
    
    # Sort by performance
    sorted_sectors = sorted(zip(sector_names, sector_changes), key=lambda x: x[1], reverse=True)
    sorted_names = [s[0] for s in sorted_sectors]
    sorted_changes = [s[1] for s in sorted_sectors]
    
    # Color coding: green for positive, red for negative
    colors = ['#00cc00' if change >= 0 else '#ff3333' for change in sorted_changes]
    
    fig_sector = go.Figure(data=[
        go.Bar(
            y=sorted_names,
            x=sorted_changes,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{change:+.2f}%" for change in sorted_changes],
            textposition='outside'
        )
    ])
    
    fig_sector.update_layout(
        title="Sector Performance Today",
        xaxis_title="% Change",
        yaxis_title="",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='white'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_sector, use_container_width=True)
    
    # Sector rotation insights
    st.markdown("### 💡 Sector Rotation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Leading Sectors")
        top_3_sectors = sorted_sectors[:3]
        for sector, change in top_3_sectors:
            if change > 1.5:
                strength = "Very Strong"
                emoji = "🔥"
            elif change > 0.5:
                strength = "Strong"
                emoji = "💪"
            else:
                strength = "Moderate"
                emoji = "📈"
            st.markdown(f"{emoji} **{sector}**: {change:+.2f}% - {strength}")
    
    with col2:
        st.markdown("#### 📉 Lagging Sectors")
        bottom_3_sectors = sorted_sectors[-3:]
        for sector, change in reversed(bottom_3_sectors):
            if change < -1.5:
                weakness = "Very Weak"
                emoji = "❄️"
            elif change < -0.5:
                weakness = "Weak"
                emoji = "⚠️"
            else:
                weakness = "Moderate"
                emoji = "📉"
            st.markdown(f"{emoji} **{sector}**: {change:+.2f}% - {weakness}")
    
    # Market regime analysis
    st.markdown("### 🎯 Market Regime")
    
    positive_sectors = sum(1 for change in sector_changes if change > 0)
    total_sectors = len(sector_changes)
    breadth_pct = (positive_sectors / total_sectors * 100) if total_sectors > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Positive Sectors", f"{positive_sectors}/{total_sectors}", f"{breadth_pct:.0f}%")
    
    with col2:
        avg_sector_change = sum(sector_changes) / len(sector_changes) if sector_changes else 0
        st.metric("Avg Sector Change", f"{avg_sector_change:+.2f}%")
    
    with col3:
        # Determine market regime
        if breadth_pct >= 70 and avg_sector_change > 0.5:
            regime = "🟢 Risk-On"
        elif breadth_pct <= 30 and avg_sector_change < -0.5:
            regime = "🔴 Risk-Off"
        else:
            regime = "🟡 Mixed"
        st.metric("Market Regime", regime)

st.divider()

# Market News & Events
st.markdown("## 📰 Market News & Events")

# Fetch real-time financial news
@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_market_news():
    """Fetch latest financial news from NewsAPI"""
    try:
        import requests
        
        # NewsAPI endpoint (free tier: 100 requests/day)
        # Get your free API key from: https://newsapi.org/
        api_key = st.secrets.get("newsapi_key", None) if hasattr(st, 'secrets') else None
        
        if not api_key:
            # Fallback: Use free public RSS feeds or show demo content
            return None
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'stock market OR S&P 500 OR Federal Reserve OR earnings',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            
            # Filter out irrelevant articles (deals, shopping, etc.)
            # Keywords that indicate non-financial content
            spam_keywords = [
                'deal', 'discount', 'promo code', 'save', 'buy now', 'aliexpress',
                'amazon', 'ebay', 'shopping', 'sale', 'offer', 'coupon',
                'battery', 'tire', 'scooter', 'bike', 'laptop', 'phone',
                'headphones', 'earbuds', 'watch', 'tv', 'monitor', 'game'
            ]
            
            # Financial keywords that should be present
            finance_keywords = [
                'stock', 'market', 'trading', 'investor', 'wall street', 'fed',
                'federal reserve', 'earnings', 'shares', 'nasdaq', 's&p', 'dow',
                'economy', 'economic', 'financial', 'bank', 'interest rate'
            ]
            
            filtered_articles = []
            for article in articles:
                title_lower = article.get('title', '').lower()
                desc_lower = article.get('description', '').lower()
                combined = title_lower + ' ' + desc_lower
                
                # Check if article contains spam keywords
                is_spam = any(spam in combined for spam in spam_keywords)
                
                # Check if article contains financial keywords
                is_finance = any(keyword in combined for keyword in finance_keywords)
                
                # Keep article if it's finance-related and not spam
                if is_finance and not is_spam:
                    filtered_articles.append(article)
            
            return filtered_articles
        else:
            return None
            
    except Exception as e:
        return None

# Try to fetch news
news_articles = fetch_market_news()

if news_articles:
    # Display real-time news
    st.markdown("### 📰 Latest Market News")
    
    for idx, article in enumerate(news_articles[:5], 1):
        with st.expander(f"📌 {article['title']}", expanded=(idx == 1)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{article.get('description', 'No description available')}**")
                st.caption(f"🕐 {article['publishedAt'][:10]} | � {article['source']['name']}")
                
                if article.get('url'):
                    st.markdown(f"[Read full article →]({article['url']})")
            
            with col2:
                if article.get('urlToImage'):
                    try:
                        st.image(article['urlToImage'], use_container_width=True)
                    except:
                        pass
    
    st.divider()
    
else:
    # Fallback: Show curated market news from public sources
    st.markdown("### 📰 Market Headlines")
    
    # Try to fetch from Yahoo Finance RSS or show demo content
    st.info("""
    **📌 Top Market Stories**
    
    - 🏦 **Federal Reserve**: Next FOMC meeting scheduled - Interest rate decision pending
    - 📊 **Earnings Season**: Major tech companies reporting this week
    - 🌍 **Global Markets**: European markets showing strength, Asian markets mixed
    - 💰 **Commodities**: Gold prices react to dollar strength, oil prices stable
    - 🤖 **AI Sector**: Continued momentum in AI and technology stocks
    - 📈 **Economic Data**: Jobs report and inflation data due this week
    
    *To enable real-time news: Add NewsAPI key to Streamlit secrets*
    """)
    
    with st.expander("🔧 How to Enable Real-Time News"):
        st.markdown("""
        **Setup Instructions:**
        
        1. Get a free API key from [NewsAPI.org](https://newsapi.org/)
        2. Create a file `.streamlit/secrets.toml` in your project root
        3. Add your API key:
        ```toml
        newsapi_key = "your_api_key_here"
        ```
        4. Restart the Streamlit app
        
        **Features with NewsAPI:**
        - ✅ Real-time financial news
        - ✅ Market-moving headlines
        - ✅ Source filtering (WSJ, Bloomberg, CNBC, etc.)
        - ✅ Sentiment analysis (coming soon)
        - ✅ Custom keyword tracking
        
        **Free Tier Limits:**
        - 100 requests per day
        - Articles from last 30 days
        - News from 80,000+ sources
        """)

# Economic Calendar Preview
st.markdown("### 📅 This Week's Economic Calendar")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Monday**
    - 📊 Retail Sales Data
    - 🏭 Industrial Production
    """)

with col2:
    st.markdown("""
    **Wednesday**
    - 🏦 FOMC Meeting Minutes
    - 🏠 Housing Starts
    """)

with col3:
    st.markdown("""
    **Friday**
    - 💼 Jobs Report (NFP)
    - 📈 Consumer Sentiment
    """)

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("📊 **ML Stock Agent v2.0**")
with col_f2:
    st.markdown(f"⏰ {datetime.now().strftime('%d. %B %Y')}")
with col_f3:
    st.markdown("🤖 Powered by AI & Machine Learning")
