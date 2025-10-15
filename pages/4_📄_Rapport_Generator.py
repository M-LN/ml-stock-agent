import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from agent_interactive import (
    hent_data, teknisk_analyse, ml_forecast, ml_forecast_rf, generer_html_rapport
)

st.set_page_config(page_title="Rapport Generator", page_icon="📄", layout="wide")

st.title("📄 Rapport Generator")
st.markdown("Generér professionelle HTML rapporter med teknisk analyse og ML forecasts")

# Sidebar inputs
st.sidebar.title("⚙️ Rapport Indstillinger")
symbol = st.sidebar.text_input("Aktiesymbol", value="AAPL", placeholder="fx MSFT, TSLA").upper()
period = st.sidebar.selectbox("Data Periode", ["6mo", "1y", "2y"], index=0)

st.sidebar.markdown("### 📊 Inkluder i Rapport")
include_technical = st.sidebar.checkbox("Teknisk Analyse", value=True)
include_ml = st.sidebar.checkbox("ML Forecasts", value=True)

if include_ml:
    st.sidebar.markdown("#### 🤖 ML Modeller")
    use_lstm = st.sidebar.checkbox("LSTM Neural Network", value=False)
    use_rf = st.sidebar.checkbox("Random Forest", value=True)
    
    if use_lstm:
        lstm_epochs = st.sidebar.slider("LSTM Epochs", 10, 100, 50)
    else:
        lstm_epochs = 50

st.sidebar.markdown("---")
st.sidebar.info("""
**Rapport indeholder:**
- Executive Summary
- Tekniske indikatorer
- Interaktive Plotly grafer
- ML forecasts
- Anbefalinger
""")

# Main content
st.markdown("### 📝 Preview")

col1, col2 = st.columns([2, 1])

with col1:
    st.info(f"""
    **Symbol:** {symbol}  
    **Periode:** {period}  
    **Teknisk Analyse:** {'✅' if include_technical else '❌'}  
    **ML Forecasts:** {'✅' if include_ml else '❌'}  
    """)

with col2:
    if st.button("📥 Generér Rapport", type="primary", use_container_width=True):
        st.session_state['generate_report'] = True

# Generate report
if st.session_state.get('generate_report', False):
    with st.spinner(f'Genererer rapport for {symbol}...'):
        try:
            # Hent data
            data = hent_data(symbol, period)
            
            if data.empty:
                st.error(f"❌ Ingen data fundet for {symbol}")
                st.session_state['generate_report'] = False
            else:
                current_price = data['Close'].iloc[-1]
                
                st.markdown("---")
                st.markdown("## 📊 Rapport Status")
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Teknisk analyse
                analyse_results = None
                if include_technical:
                    status_text.text("📊 Beregner tekniske indikatorer...")
                    progress_bar.progress(20)
                    # teknisk_analyse() returnerer data med beregnede indikatorer
                    data = teknisk_analyse(data, symbol, print_results=False)
                    
                    # Lav en dict med indikator værdier til rapporten
                    analyse_results = {}
                    if 'RSI' in data.columns:
                        rsi = data['RSI'].iloc[-1]
                        analyse_results['RSI'] = f"{rsi:.1f}"
                        if rsi > 70:
                            analyse_results['RSI Signal'] = "⚠️ OVERKØBT"
                        elif rsi < 30:
                            analyse_results['RSI Signal'] = "🔻 OVERSOLGT"
                        else:
                            analyse_results['RSI Signal'] = "✅ Neutral"
                    
                    if 'SMA50' in data.columns and 'SMA200' in data.columns:
                        sma50 = data['SMA50'].iloc[-1]
                        sma200 = data['SMA200'].iloc[-1]
                        analyse_results['SMA50'] = f"${sma50:.2f}"
                        analyse_results['SMA200'] = f"${sma200:.2f}"
                        if sma50 > sma200:
                            analyse_results['SMA Signal'] = "📈 Golden Cross (bullish)"
                        else:
                            analyse_results['SMA Signal'] = "📉 Death Cross (bearish)"
                    
                    if 'MACD_Line' in data.columns and 'MACD_Signal' in data.columns:
                        macd = data['MACD_Line'].iloc[-1]
                        signal = data['MACD_Signal'].iloc[-1]
                        analyse_results['MACD'] = f"{macd:.2f}"
                        analyse_results['MACD Signal Line'] = f"{signal:.2f}"
                        if macd > signal:
                            analyse_results['MACD Signal'] = "📈 Bullish"
                        else:
                            analyse_results['MACD Signal'] = "📉 Bearish"
                    
                    if 'BB_Upper' in data.columns:
                        bb_upper = data['BB_Upper'].iloc[-1]
                        bb_lower = data['BB_Lower'].iloc[-1]
                        bb_middle = data['BB_Middle'].iloc[-1]
                        analyse_results['BB Upper'] = f"${bb_upper:.2f}"
                        analyse_results['BB Middle'] = f"${bb_middle:.2f}"
                        analyse_results['BB Lower'] = f"${bb_lower:.2f}"
                    
                    if 'ATR' in data.columns:
                        atr = data['ATR'].iloc[-1]
                        atr_pct = (atr / current_price) * 100
                        analyse_results['ATR'] = f"${atr:.2f}"
                        analyse_results['ATR %'] = f"{atr_pct:.2f}%"
                
                # ML forecasts
                ml_results = None
                if include_ml:
                    ml_results = {}
                    
                    if use_lstm:
                        status_text.text("🧠 Træner LSTM model...")
                        progress_bar.progress(40)
                        
                        lstm_1d = ml_forecast(data, window=30, epochs=lstm_epochs, horizon=1)
                        lstm_5d = ml_forecast(data, window=30, epochs=lstm_epochs, horizon=5)
                        lstm_22d = ml_forecast(data, window=30, epochs=lstm_epochs, horizon=22)
                        
                        if lstm_1d:
                            ml_results['lstm_1d'] = lstm_1d['forecast']
                        if lstm_5d:
                            ml_results['lstm_5d'] = lstm_5d['forecast']
                        if lstm_22d:
                            ml_results['lstm_22d'] = lstm_22d['forecast']
                    
                    if use_rf:
                        status_text.text("🌲 Træner Random Forest model...")
                        progress_bar.progress(60)
                        
                        rf_1d = ml_forecast_rf(data, window=30, horizon=1)
                        rf_5d = ml_forecast_rf(data, window=30, horizon=5)
                        rf_22d = ml_forecast_rf(data, window=30, horizon=22)
                        
                        if rf_1d:
                            ml_results['rf_1d'] = rf_1d['forecast']
                        if rf_5d:
                            ml_results['rf_5d'] = rf_5d['forecast']
                        if rf_22d:
                            ml_results['rf_22d'] = rf_22d['forecast']
                
                # Generate HTML rapport
                status_text.text("📄 Genererer HTML rapport...")
                progress_bar.progress(80)
                
                filepath = generer_html_rapport(
                    symbol, 
                    data, 
                    analyse_results=analyse_results if include_technical else None,
                    ml_results=ml_results if include_ml and ml_results else None
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Rapport genereret!")
                
                st.markdown("---")
                st.success(f"✅ Rapport genereret succesfuldt!")
                
                # Display info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Symbol", symbol)
                
                with col2:
                    st.metric("Data Punkter", len(data))
                
                with col3:
                    st.metric("Fil", filepath.split('\\')[-1] if '\\' in filepath else filepath.split('/')[-1])
                
                # Download button
                st.markdown("### 📥 Download Rapport")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    st.download_button(
                        label="⬇️ Download HTML Rapport",
                        data=html_content,
                        file_name=f"{symbol}_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    st.info(f"""
                    **Rapport gemt lokalt:**  
                    `{filepath}`
                    
                    Du kan også åbne filen direkte fra din fil explorer.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Kunne ikke læse rapport fil: {e}")
                
                # Preview section
                with st.expander("👁️ Preview Rapport (første 50 linjer)"):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:50]
                            preview = ''.join(lines)
                        st.code(preview, language='html')
                    except:
                        st.warning("Kunne ikke vise preview")
                
                # Reset button
                if st.button("🔄 Generér Ny Rapport", use_container_width=True):
                    st.session_state['generate_report'] = False
                    st.rerun()
                
        except Exception as e:
            st.error(f"❌ Fejl under rapport generering: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state['generate_report'] = False

else:
    st.markdown("---")
    st.markdown("## 📋 Hvad Indeholder Rapporten?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Teknisk Analyse
        - **Header** med symbol og dato
        - **Pris oversigt** med change %
        - **Tekniske indikatorer:**
          - RSI med overbought/oversold zones
          - SMA50 & SMA200 med golden/death cross
          - MACD med signal line
          - Bollinger Bands
          - ATR (volatilitet)
        - **Interaktiv 4-panel Plotly graf**
        - **Fortolkning** af hvert indikator
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 ML Forecasts
        - **LSTM predictions:**
          - 1 dag forecast
          - 5 dage forecast
          - 22 dage forecast
        - **Random Forest predictions:**
          - 1 dag forecast
          - 5 dage forecast
          - 22 dage forecast
        - **Sammenligning** mellem modeller
        - **Visualisering** af forecasts
        - **Disclaimer** og advarsler
        """)
    
    st.markdown("---")
    st.markdown("### 🎨 Rapport Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📱 Responsive Design**
        - Fungerer på desktop
        - Fungerer på tablet
        - Fungerer på mobil
        """)
    
    with col2:
        st.success("""
        **🎨 Professional Styling**
        - Gradient backgrounds
        - Custom CSS styling
        - Clean typography
        """)
    
    with col3:
        st.warning("""
        **📊 Interaktive Grafer**
        - Zoom og pan
        - Hover for detaljer
        - Embedded Plotly charts
        """)
    
    st.markdown("---")
    st.markdown("### 💡 Brug Cases")
    
    st.markdown("""
    1. **📧 Email til klienter** - Del analyser professionelt
    2. **📱 Social media** - Del på Twitter, LinkedIn etc.
    3. **📚 Portfolio** - Gem til senere reference
    4. **🎓 Uddannelse** - Lær teknisk analyse
    5. **💼 Præsentationer** - Embed i slides
    """)
    
    st.markdown("---")
    st.info("👆 Konfigurér indstillingerne i sidebaren og klik 'Generér Rapport' for at starte!")
