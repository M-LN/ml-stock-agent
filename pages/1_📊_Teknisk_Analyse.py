"""
ML Stock Agent - Teknisk Analyse Side
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_interactive import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_stochastic,
    calculate_cci,
    calculate_williams_r,
    calculate_obv,
    calculate_adx,
    calculate_vwap,
    calculate_fibonacci_levels,
    calculate_pivot_points,
    vis_graf_plotly
)

st.set_page_config(page_title="Teknisk Analyse", page_icon="📊", layout="wide")

st.title("📊 Teknisk Analyse")
st.markdown("Dybdegående teknisk analyse med alle indikatorer")

# Sidebar inputs
st.sidebar.title("⚙️ Indstillinger")

# Stock groups/watchlists
st.sidebar.markdown("### 📋 Aktie Grupper")

stock_groups = {
    "💼 Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "🏦 Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "⚡ Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "💊 Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK"],
    "🛒 Consumer": ["WMT", "COST", "TGT", "HD", "NKE"],
    "🏭 Industrial": ["BA", "CAT", "GE", "HON", "UPS"],
    "🏠 Real Estate": ["AMT", "PLD", "SPG", "EQIX", "PSA"],
    "💰 Crypto Stocks": ["COIN", "MSTR", "MARA", "RIOT", "HOOD"],
    "🌟 Populære Danske": ["NOVO-B.CO", "DSV.CO", "MAERSK-B.CO", "ORSTED.CO", "COLO-B.CO"],
    "📱 Custom": []  # User can add their own
}

# Group selector
selected_group = st.sidebar.selectbox(
    "Vælg Gruppe",
    options=list(stock_groups.keys()),
    help="Hurtig adgang til prædefinerede aktiegrupper"
)

# Show stocks in selected group
if selected_group and stock_groups[selected_group]:
    st.sidebar.markdown(f"**Aktier i {selected_group}:**")
    
    # Display as buttons for quick selection
    group_stocks = stock_groups[selected_group]
    
    # Create columns for buttons (3 per row)
    for i in range(0, len(group_stocks), 3):
        cols = st.sidebar.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(group_stocks):
                stock = group_stocks[i + j]
                if col.button(stock, key=f"btn_{stock}", use_container_width=True):
                    st.session_state['selected_symbol'] = stock

st.sidebar.markdown("---")

# Symbol input (with session state support)
default_symbol = st.session_state.get('selected_symbol', 'AAPL')
symbol = st.sidebar.text_input("Eller indtast symbol", value=default_symbol, placeholder="fx MSFT, TSLA").upper()

# Update session state
st.session_state['selected_symbol'] = symbol

period = st.sidebar.selectbox("Periode", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

st.sidebar.markdown("### 📊 Analyse Mode")
analysis_mode = st.sidebar.radio(
    "Vælg Mode",
    ["Single Stock", "Compare Group"],
    help="Enkelt aktie eller sammenlign flere"
)

if analysis_mode == "Compare Group":
    st.sidebar.info("📊 Sammenligner alle aktier i valgt gruppe")

st.sidebar.markdown("### 📌 Basis Indikatorer")
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_sma = st.sidebar.checkbox("SMA (50/200)", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
show_atr = st.sidebar.checkbox("ATR (Volatilitet)", value=True)

st.sidebar.markdown("### 🔧 Avancerede Indikatorer")
show_stochastic = st.sidebar.checkbox("Stochastic Oscillator", value=False)
show_cci = st.sidebar.checkbox("CCI (Commodity Channel)", value=False)
show_williams = st.sidebar.checkbox("Williams %R", value=False)
show_obv = st.sidebar.checkbox("OBV (On-Balance Volume)", value=False)
show_adx = st.sidebar.checkbox("ADX (Trend Strength)", value=False)
show_vwap = st.sidebar.checkbox("VWAP", value=False)
show_fibonacci = st.sidebar.checkbox("Fibonacci Levels", value=False)
show_pivots = st.sidebar.checkbox("Pivot Points", value=False)

# Check if auto-analyze flag is set (from Watchlist)
auto_start = st.session_state.get('auto_analyze', False)
if auto_start:
    st.info(f"🚀 Auto-analyserer {symbol} fra Watchlist...")
    st.session_state['auto_analyze'] = False  # Reset flag after use

# Main content
if st.button("🔍 Analysér", type="primary", use_container_width=True) or auto_start:
    
    # Group comparison mode
    if analysis_mode == "Compare Group" and selected_group and stock_groups[selected_group]:
        st.markdown(f"## 📊 Gruppe Sammenligning: {selected_group}")
        
        comparison_data = []
        stocks_to_compare = stock_groups[selected_group]
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, stock_symbol in enumerate(stocks_to_compare):
            try:
                status_text.text(f"Analyserer {stock_symbol}... ({idx + 1}/{len(stocks_to_compare)})")
                progress_bar.progress((idx + 1) / len(stocks_to_compare))
                
                stock_data = yf.download(stock_symbol, period=period, progress=False)
                
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)
                
                if not stock_data.empty and len(stock_data) >= 14:
                    current_price = stock_data['Close'].iloc[-1]
                    prev_price = stock_data['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    # Calculate indicators
                    rsi = calculate_rsi(stock_data).iloc[-1] if show_rsi else None
                    sma50 = calculate_sma(stock_data, 50).iloc[-1] if show_sma and len(stock_data) >= 50 else None
                    sma200 = calculate_sma(stock_data, 200).iloc[-1] if show_sma and len(stock_data) >= 200 else None
                    
                    comparison_data.append({
                        'Symbol': stock_symbol,
                        'Price': current_price,
                        'Change %': change_pct,
                        'RSI': rsi,
                        'SMA50': sma50,
                        'SMA200': sma200,
                        'Volume': stock_data['Volume'].iloc[-1]
                    })
            except Exception as e:
                st.warning(f"⚠️ Kunne ikke hente data for {stock_symbol}: {str(e)}")
                continue
        
        status_text.empty()
        progress_bar.empty()
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Display summary metrics
            st.markdown("### 📊 Gruppe Oversigt")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_change = df_comparison['Change %'].mean()
                st.metric("Gennemsnitlig Ændring", f"{avg_change:+.2f}%")
            
            with col2:
                positive = (df_comparison['Change %'] > 0).sum()
                st.metric("Positive Aktier", f"{positive}/{len(df_comparison)}")
            
            with col3:
                if 'RSI' in df_comparison.columns:
                    avg_rsi = df_comparison['RSI'].mean()
                    st.metric("Gennemsnitlig RSI", f"{avg_rsi:.1f}")
            
            with col4:
                best_performer = df_comparison.loc[df_comparison['Change %'].idxmax(), 'Symbol']
                best_change = df_comparison['Change %'].max()
                st.metric("Bedste", f"{best_performer} ({best_change:+.1f}%)")
            
            st.markdown("---")
            
            # Display comparison table
            st.markdown("### 📋 Detaljeret Sammenligning")
            
            # Format the dataframe
            styled_df = df_comparison.copy()
            
            def color_change(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}'
            
            # Display with formatting
            st.dataframe(
                styled_df.style.format({
                    'Price': '${:.2f}',
                    'Change %': '{:+.2f}%',
                    'RSI': '{:.1f}',
                    'SMA50': '${:.2f}',
                    'SMA200': '${:.2f}',
                    'Volume': '{:,.0f}'
                }).applymap(color_change, subset=['Change %']),
                use_container_width=True,
                height=400
            )
            
            # Performance chart
            st.markdown("### 📊 Performance Sammenligning")
            import plotly.express as px
            
            fig = px.bar(
                df_comparison.sort_values('Change %', ascending=True),
                x='Change %',
                y='Symbol',
                orientation='h',
                color='Change %',
                color_continuous_scale=['red', 'yellow', 'green'],
                title=f'{selected_group} - Daglig Performance'
            )
            fig.update_layout(height=max(400, len(df_comparison) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Analyse komplet for {len(comparison_data)} aktier!")
        else:
            st.error("❌ Ingen data kunne hentes for gruppen")
    
    # Single stock mode
    else:
        with st.spinner(f'Analyserer {symbol}...'):
            try:
                # Hent data
                data = yf.download(symbol, period=period, progress=False)
                
                # Fix MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if data.empty:
                    st.error(f"❌ Ingen data fundet for {symbol}")
                else:
                    current_price = data['Close'].iloc[-1]
                    
                    # Header metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                        change_pct = (change / data['Close'].iloc[-2]) * 100
                        st.metric("Aktuel Kurs", f"${current_price:.2f}", f"{change_pct:+.2f}%")
                    
                    with col2:
                        st.metric("Højeste", f"${data['High'].iloc[-1]:.2f}")
                    
                    with col3:
                        st.metric("Laveste", f"${data['Low'].iloc[-1]:.2f}")
                    
                    with col4:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]/1e6:.1f}M")
                    
                    st.markdown("---")
                    
                    # Beregn og vis indikatorer
                    st.markdown("## 📈 Tekniske Indikatorer")
                    
                    ind_col1, ind_col2 = st.columns(2)
                    
                    # RSI
                    if show_rsi:
                        with ind_col1:
                            data['RSI'] = calculate_rsi(data)
                            rsi = data['RSI'].iloc[-1]
                            st.markdown("### 📊 RSI (14)")
                            st.metric("RSI Værdi", f"{rsi:.1f}")
                            if rsi > 70:
                                st.error("⚠️ OVERKØBT (>70)")
                            elif rsi < 30:
                                st.success("🔻 OVERSOLGT (<30)")
                            else:
                                st.info("✅ Neutral zone (30-70)")
                    
                    # SMA
                    if show_sma:
                        with ind_col2:
                            data['SMA50'] = calculate_sma(data, 50)
                            data['SMA200'] = calculate_sma(data, 200)
                            sma50 = data['SMA50'].iloc[-1]
                            sma200 = data['SMA200'].iloc[-1]
                            
                            st.markdown("### 📈 Moving Averages")
                            st.metric("SMA50", f"${sma50:.2f}")
                            st.metric("SMA200", f"${sma200:.2f}")
                            
                            if sma50 > sma200:
                                st.success("📈 Golden Cross (bullish)")
                            else:
                                st.warning("📉 Death Cross (bearish)")
                    
                    # MACD
                    if show_macd:
                        with ind_col1:
                            macd_line, signal_line, histogram = calculate_macd(data)
                            data['MACD_Line'] = macd_line
                            data['MACD_Signal'] = signal_line
                            data['MACD_Hist'] = histogram
                            
                            macd = macd_line.iloc[-1]
                            signal = signal_line.iloc[-1]
                            hist = histogram.iloc[-1]
                            
                            st.markdown("### 📊 MACD")
                            st.metric("MACD Line", f"{macd:.2f}")
                            st.metric("Signal", f"{signal:.2f}")
                            st.metric("Histogram", f"{hist:.2f}")
                            
                            if macd > signal:
                                st.success("📈 Bullish signal")
                            else:
                                st.warning("📉 Bearish signal")
                    
                    # Bollinger Bands
                    if show_bb:
                        with ind_col2:
                            upper, middle, lower = calculate_bollinger_bands(data)
                            data['BB_Upper'] = upper
                            data['BB_Middle'] = middle
                            data['BB_Lower'] = lower
                            
                            bb_upper = upper.iloc[-1]
                            bb_middle = middle.iloc[-1]
                            bb_lower = lower.iloc[-1]
                            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
                            
                            st.markdown("### 📊 Bollinger Bands")
                            st.metric("Upper", f"${bb_upper:.2f}")
                            st.metric("Middle", f"${bb_middle:.2f}")
                            st.metric("Lower", f"${bb_lower:.2f}")
                            st.metric("Width", f"{bb_width:.2f}%")
                            
                            if current_price > bb_upper:
                                st.error("⚠️ Over upper band")
                            elif current_price < bb_lower:
                                st.success("🔻 Under lower band")
                            else:
                                st.info("✅ Inden for bands")
                    
                    # ATR
                    if show_atr:
                        with ind_col1:
                            data['ATR'] = calculate_atr(data)
                            atr = data['ATR'].iloc[-1]
                            atr_pct = (atr / current_price) * 100
                            
                            st.markdown("### 📊 ATR (Volatilitet)")
                            st.metric("ATR", f"${atr:.2f}")
                            st.metric("ATR %", f"{atr_pct:.2f}%")
                            
                            if atr_pct > 2:
                                st.warning("⚠️ Høj volatilitet")
                            elif atr_pct > 1:
                                st.info("📊 Moderat volatilitet")
                            else:
                                st.success("✅ Lav volatilitet")
                    
                    # Stochastic Oscillator
                    if show_stochastic:
                        with ind_col2:
                            stoch_k, stoch_d = calculate_stochastic(data)
                            data['Stochastic_K'] = stoch_k
                            data['Stochastic_D'] = stoch_d
                            
                            k_value = stoch_k.iloc[-1]
                            d_value = stoch_d.iloc[-1]
                            
                            st.markdown("### 📊 Stochastic Oscillator")
                            st.metric("%K", f"{k_value:.2f}")
                            st.metric("%D", f"{d_value:.2f}")
                            
                            if k_value > 80:
                                st.error("⚠️ Overkøbt (>80)")
                            elif k_value < 20:
                                st.success("✅ Oversolgt (<20)")
                            else:
                                st.info("📊 Neutral zone")
                            
                            if k_value > d_value:
                                st.success("📈 Bullish crossover")
                            else:
                                st.warning("📉 Bearish crossover")
                    
                    # CCI
                    if show_cci:
                        with ind_col1:
                            cci = calculate_cci(data)
                            data['CCI'] = cci
                            cci_value = cci.iloc[-1]
                            
                            st.markdown("### 📊 CCI (Commodity Channel)")
                            st.metric("CCI", f"{cci_value:.2f}")
                            
                            if cci_value > 100:
                                st.error("⚠️ Overkøbt (>100)")
                            elif cci_value < -100:
                                st.success("✅ Oversolgt (<-100)")
                            else:
                                st.info("📊 Normal range")
                    
                    # Williams %R
                    if show_williams:
                        with ind_col2:
                            williams = calculate_williams_r(data)
                            data['Williams_R'] = williams
                            williams_value = williams.iloc[-1]
                            
                            st.markdown("### 📊 Williams %R")
                            st.metric("Williams %R", f"{williams_value:.2f}")
                            
                            if williams_value > -20:
                                st.error("⚠️ Overkøbt (>-20)")
                            elif williams_value < -80:
                                st.success("✅ Oversolgt (<-80)")
                            else:
                                st.info("📊 Neutral zone")
                    
                    # OBV
                    if show_obv:
                        with ind_col1:
                            obv = calculate_obv(data)
                            data['OBV'] = obv
                            obv_current = obv.iloc[-1]
                            obv_20d_ago = obv.iloc[-20] if len(obv) >= 20 else obv.iloc[0]
                            obv_change = ((obv_current - obv_20d_ago) / abs(obv_20d_ago)) * 100 if obv_20d_ago != 0 else 0
                            
                            st.markdown("### 📊 OBV (On-Balance Volume)")
                            st.metric("OBV", f"{obv_current:,.0f}")
                            st.metric("20-dag ændring", f"{obv_change:+.2f}%")
                            
                            if obv_change > 10:
                                st.success("📈 Stærk akkumulering")
                            elif obv_change < -10:
                                st.error("📉 Stærk distribution")
                            else:
                                st.info("📊 Stabil trend")
                    
                    # ADX
                    if show_adx:
                        with ind_col2:
                            adx, plus_di, minus_di = calculate_adx(data)
                            data['ADX'] = adx
                            data['Plus_DI'] = plus_di
                            data['Minus_DI'] = minus_di
                            
                            adx_value = adx.iloc[-1]
                            plus_di_value = plus_di.iloc[-1]
                            minus_di_value = minus_di.iloc[-1]
                            
                            st.markdown("### 📊 ADX (Trend Strength)")
                            st.metric("ADX", f"{adx_value:.2f}")
                            st.metric("+DI", f"{plus_di_value:.2f}")
                            st.metric("-DI", f"{minus_di_value:.2f}")
                            
                            if adx_value > 25:
                                st.success("💪 Stærk trend")
                            elif adx_value > 20:
                                st.info("📊 Moderat trend")
                            else:
                                st.warning("⚠️ Svag/ingen trend")
                            
                            if plus_di_value > minus_di_value:
                                st.success("📈 Bullish retning")
                            else:
                                st.error("📉 Bearish retning")
                    
                    # VWAP
                    if show_vwap:
                        with ind_col1:
                            vwap = calculate_vwap(data)
                            data['VWAP'] = vwap
                            vwap_value = vwap.iloc[-1]
                            price_vs_vwap = ((current_price - vwap_value) / vwap_value) * 100
                            
                            st.markdown("### 📊 VWAP")
                            st.metric("VWAP", f"${vwap_value:.2f}")
                            st.metric("Pris vs VWAP", f"{price_vs_vwap:+.2f}%")
                            
                            if current_price > vwap_value:
                                st.success("📈 Over VWAP (Bullish)")
                            else:
                                st.error("📉 Under VWAP (Bearish)")
                    
                    # Fibonacci Levels
                    if show_fibonacci:
                        with ind_col2:
                            fib_levels = calculate_fibonacci_levels(data)
                            
                            st.markdown("### 📊 Fibonacci Retracement")
                            st.write(f"**Beregnet over sidste 100 dage**")
                            
                            for level_name, level_value in fib_levels.items():
                                if abs(current_price - level_value) / level_value < 0.02:  # Within 2%
                                    st.metric(f"🎯 {level_name}", f"${level_value:.2f}", delta="← AKTUEL")
                                else:
                                    st.metric(level_name, f"${level_value:.2f}")
                    
                    # Pivot Points
                    if show_pivots:
                        with ind_col1:
                            pivots = calculate_pivot_points(data)
                            
                            st.markdown("### 📊 Pivot Points")
                            st.write(f"**Baseret på gårsdagens data**")
                            
                            # Show resistance levels
                            st.write("**Modstand:**")
                            for level in ['R3', 'R2', 'R1']:
                                if abs(current_price - pivots[level]) / pivots[level] < 0.02:
                                    st.metric(f"🎯 {level}", f"${pivots[level]:.2f}", delta="← AKTUEL")
                                else:
                                    st.metric(level, f"${pivots[level]:.2f}")
                            
                            # Show pivot
                            st.write("**Pivot:**")
                            st.metric("Pivot", f"${pivots['Pivot']:.2f}")
                            
                            # Show support levels
                            st.write("**Støtte:**")
                            for level in ['S1', 'S2', 'S3']:
                                if abs(current_price - pivots[level]) / pivots[level] < 0.02:
                                    st.metric(f"🎯 {level}", f"${pivots[level]:.2f}", delta="← AKTUEL")
                                else:
                                    st.metric(level, f"${pivots[level]:.2f}")
                    
                    st.markdown("---")
                    
                    # Interactive chart
                    st.markdown("## 📊 Interaktiv Graf")
                    fig = vis_graf_plotly(data, symbol)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Analyse komplet for {symbol}!")
                    
            except Exception as e:
                st.error(f"❌ Fejl: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("👆 Klik på 'Analysér' knappen for at starte")
    st.markdown("### 📌 Sådan virker det:")
    st.markdown("""
    1. **Vælg indstillinger** i sidebaren
    2. **Klik Analysér** for at hente data
    3. **Se indikatorer** i oversigten
    4. **Udforsk grafen** med zoom og hover
    5. **Gem til watchlist** for senere
    """)
