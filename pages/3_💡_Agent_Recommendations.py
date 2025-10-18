import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from agent_interactive import (
    generate_trading_signal,
    ml_forecast_rf,
    ml_forecast,
    ml_forecast_xgboost,
    ml_forecast_prophet,
    ml_forecast_ensemble,
    list_saved_models,
    load_model,
    predict_with_saved_model
)
from fundamentals import analyze_fundamentals, get_valuation_category
from sentiment import get_combined_sentiment, get_fear_greed_interpretation
from mentor import mentor_comment_simple
import os

st.set_page_config(page_title="Agent Recommendations", page_icon="🎯", layout="wide")

st.title("🎯 Agent Trading Recommendations")
st.markdown("**AI-drevet trading agent der kombinerer ML forecasts med tekniske indikatorer**")

# Sidebar configuration
st.sidebar.header("⚙️ Agent Konfiguration")

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
                if col.button(stock, key=f"agent_btn_{stock}", use_container_width=True):
                    st.session_state['agent_selected_symbol'] = stock

st.sidebar.markdown("---")

# Stock selection (with session state support)
default_symbol = st.session_state.get('agent_selected_symbol', 'AAPL')
symbol = st.sidebar.text_input("Aktie Symbol", value=default_symbol, help="Indtast ticker symbol (f.eks. AAPL, TSLA)")
if symbol:
    st.session_state['agent_selected_symbol'] = symbol

period = st.sidebar.selectbox("Data Periode", ["3mo", "6mo", "1y", "2y"], index=2)

# Option to use saved model
st.sidebar.subheader("🔧 Model Valg")
use_saved_models_option = st.sidebar.checkbox("📂 Inkluder gemte modeller", value=False,
                                               help="Tilføj dine trænede modeller til forecasts")

# Model selection for forecasts
st.sidebar.subheader("🤖 Standard ML Modeller")
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_xgboost = st.sidebar.checkbox("XGBoost", value=True)
use_prophet = st.sidebar.checkbox("Prophet", value=False)
use_ensemble = st.sidebar.checkbox("Ensemble", value=True)
use_lstm = st.sidebar.checkbox("LSTM", value=False)

# Saved models (ONLY DEPLOYED)
saved_models_to_use = []
if use_saved_models_option:
    all_saved_models = list_saved_models(symbol=symbol)
    saved_models = [m for m in all_saved_models if m.get('deployed', False)]
    
    if saved_models:
        st.sidebar.markdown("### 📂 Deployed Modeller")
        for i, model_info in enumerate(saved_models[:5]):  # Max 5 deployed
            model_label = f"{model_info['model_type'].upper()} ({model_info['timestamp'][:8]})"
            if st.sidebar.checkbox(model_label, key=f"saved_{i}"):
                saved_models_to_use.append(model_info)
    else:
        st.sidebar.info(f"💡 Ingen deployed modeller for {symbol}. Deploy i Model Management.")

# Check if at least one model is selected
if not any([use_rf, use_xgboost, use_prophet, use_ensemble, use_lstm]) and not saved_models_to_use:
    st.sidebar.error("⚠️ Vælg mindst én ML model!")
    st.stop()

# Check if auto-start flag is set (from Watchlist)
auto_start_agent = st.session_state.get('auto_agent', False)
if auto_start_agent:
    st.info(f"🚀 Auto-starter Agent Recommendations for {symbol} fra Watchlist...")
    st.session_state['auto_agent'] = False  # Reset flag after use

# Analysis button
analyze_button = st.sidebar.button("🔍 Analyser", type="primary", use_container_width=True)

if analyze_button or auto_start_agent:
    with st.spinner(f"📊 Henter data for {symbol}..."):
        try:
            # Download data
            data = yf.download(symbol, period=period, progress=False)
            
            if data.empty:
                st.error(f"❌ Kunne ikke hente data for {symbol}")
                st.stop()
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            st.success(f"✅ Data hentet: {len(data)} dage")
            
            # Store in session state
            st.session_state.agent_data = data
            st.session_state.agent_symbol = symbol
            
        except Exception as e:
            st.error(f"❌ Fejl ved datahentning: {str(e)}")
            st.stop()

# Main analysis section
if 'agent_data' in st.session_state and 'agent_symbol' in st.session_state:
    data = st.session_state.agent_data
    symbol = st.session_state.agent_symbol
    
    # Generate ML forecasts
    st.subheader("🔮 ML Forecasts")
    
    ml_forecasts = {}
    forecast_horizon = 5
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if use_rf:
            with st.spinner("RF..."):
                try:
                    rf_result = ml_forecast_rf(data, horizon=forecast_horizon)
                    if rf_result:
                        ml_forecasts['rf'] = rf_result
                        forecast_val = rf_result['forecast']
                        st.metric("Random Forest", f"${forecast_val:.2f}", 
                                 delta=f"{((forecast_val - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
                except Exception as e:
                    st.error(f"RF fejl: {str(e)}")
    
    with col2:
        if use_xgboost:
            with st.spinner("XGBoost..."):
                try:
                    xgb_result = ml_forecast_xgboost(data, horizon=forecast_horizon)
                    if xgb_result:
                        ml_forecasts['xgboost'] = xgb_result
                        forecast_val = xgb_result['forecast']
                        st.metric("XGBoost", f"${forecast_val:.2f}",
                                 delta=f"{((forecast_val - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
                except Exception as e:
                    st.error(f"XGBoost fejl: {str(e)}")
    
    with col3:
        if use_prophet:
            with st.spinner("Prophet..."):
                try:
                    prophet_result = ml_forecast_prophet(data, horizon=forecast_horizon)
                    if prophet_result:
                        ml_forecasts['prophet'] = prophet_result
                        forecast_val = prophet_result['forecast']
                        st.metric("Prophet", f"${forecast_val:.2f}",
                                 delta=f"{((forecast_val - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
                except Exception as e:
                    st.error(f"Prophet fejl: {str(e)}")
    
    with col4:
        if use_ensemble:
            with st.spinner("Ensemble..."):
                try:
                    ensemble_result = ml_forecast_ensemble(data, horizon=forecast_horizon)
                    if ensemble_result:
                        ml_forecasts['ensemble'] = ensemble_result
                        forecast_val = ensemble_result['forecast']
                        st.metric("Ensemble", f"${forecast_val:.2f}",
                                 delta=f"{((forecast_val - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
                except Exception as e:
                    st.error(f"Ensemble fejl: {str(e)}")
    
    with col5:
        if use_lstm:
            with st.spinner("LSTM..."):
                try:
                    lstm_result = ml_forecast(data, horizon=forecast_horizon)
                    if lstm_result:
                        ml_forecasts['lstm'] = lstm_result
                        forecast_val = lstm_result['forecast']
                        st.metric("LSTM", f"${forecast_val:.2f}",
                                 delta=f"{((forecast_val - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
                except Exception as e:
                    st.error(f"LSTM fejl: {str(e)}")
    
    # Add saved model forecasts
    if saved_models_to_use:
        st.markdown("#### 📂 Gemte Modeller")
        saved_cols = st.columns(min(len(saved_models_to_use), 5))
        
        for idx, model_info in enumerate(saved_models_to_use):
            with saved_cols[idx]:
                with st.spinner(f"{model_info['model_type'].upper()}..."):
                    try:
                        # Load model
                        model_package = load_model(model_info['filepath'])
                        
                        # Make prediction
                        forecast_val = predict_with_saved_model(
                            model_package=model_package,
                            data=data,
                            horizon=forecast_horizon
                        )
                        
                        # Store in ml_forecasts with unique key
                        saved_key = f"saved_{model_info['model_type']}_{idx}"
                        ml_forecasts[saved_key] = {
                            'forecast': forecast_val,
                            'model_type': model_info['model_type'],
                            'is_saved': True
                        }
                        
                        # Display metric
                        current_price = data['Close'].iloc[-1]
                        change_pct = ((forecast_val - current_price) / current_price * 100)
                        st.metric(
                            f"{model_info['model_type'].upper()}\n(Gemt)",
                            f"${forecast_val:.2f}",
                            delta=f"{change_pct:.2f}%"
                        )
                    except Exception as e:
                        st.error(f"Fejl: {str(e)[:30]}")
    
    st.divider()
    
    # Generate trading signal
    st.subheader("🎯 Agent Recommendation")
    
    with st.spinner("🤖 Analyserer signal..."):
        signal_result = generate_trading_signal(data, ml_forecasts=ml_forecasts, use_ml=True)
    
    # Display signal with color coding
    signal = signal_result['signal']
    confidence = signal_result['confidence']
    
    if signal == "BUY":
        signal_color = "🟢"
        bg_color = "#1e4620"
        border_color = "#2e7d32"
    elif signal == "SELL":
        signal_color = "🔴"
        bg_color = "#4a1c1c"
        border_color = "#c62828"
    else:  # HOLD
        signal_color = "🟡"
        bg_color = "#3d3d1f"
        border_color = "#f9a825"
    
    # Main signal card
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 30px; border-radius: 10px; border: 3px solid {border_color}; text-align: center;">
        <h1 style="font-size: 60px; margin: 0;">{signal_color} {signal}</h1>
        <h3 style="margin: 10px 0;">Confidence: {confidence:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== FUNDAMENTALS + SENTIMENT ANALYSIS ==========
    st.subheader("📊 Fundamentals + Sentiment Analysis")
    st.markdown("**Værdiansættelse og markedspsykologi**")
    
    # Get API key from sidebar or environment
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    
    with st.spinner("Analyserer fundamentals + sentiment..."):
        try:
            # Get fundamentals
            fundamentals = analyze_fundamentals(symbol, industry_pe=20.0)
            
            # Get sentiment
            sentiment = get_combined_sentiment(symbol, finnhub_api_key)
            
            # Generate mentor comment
            mentor_comment = mentor_comment_simple(fundamentals, sentiment)
            
            # Display metrics
            fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
            
            with fund_col1:
                st.metric(
                    "📊 Fundamental Score",
                    f"{fundamentals['score']}/100",
                    delta=get_valuation_category(fundamentals['score'])
                )
            
            with fund_col2:
                st.metric(
                    "🎭 Sentiment Score",
                    f"{sentiment['combined_score']}/100",
                    delta=sentiment['interpretation']
                )
            
            with fund_col3:
                fg = sentiment.get('fear_greed', {})
                st.metric(
                    "😱 Fear & Greed",
                    f"{fg.get('value', 50)}/100",
                    delta=fg.get('classification', 'Neutral')
                )
            
            with fund_col4:
                vix = sentiment.get('vix', {})
                vix_val = vix.get('value')
                if vix_val:
                    st.metric(
                        "📉 VIX",
                        f"{vix_val:.1f}",
                        delta=vix.get('interpretation', 'N/A')
                    )
                else:
                    st.metric("📉 VIX", "N/A")
            
            # Expandable details
            with st.expander("📈 Se Detaljer"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("### 📊 Fundamentals")
                    
                    fund_data = {
                        "Metric": ["P/E Ratio", "PEG Ratio", "P/B Ratio", "Growth %"],
                        "Value": [
                            f"{fundamentals.get('pe', 'N/A'):.2f}" if fundamentals.get('pe') else "N/A",
                            f"{fundamentals.get('peg', 'N/A'):.2f}" if fundamentals.get('peg') else "N/A",
                            f"{fundamentals.get('pb', 'N/A'):.2f}" if fundamentals.get('pb') else "N/A",
                            f"{fundamentals.get('growth', 0):.2f}%"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(fund_data), use_container_width=True, hide_index=True)
                    
                    # Gauge for fundamentals
                    fig_fund = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fundamentals['score'],
                        title={'text': "Fundamental Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 45], 'color': "lightcoral"},
                                {'range': [45, 60], 'color': "lightyellow"},
                                {'range': [60, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ]
                        }
                    ))
                    fig_fund.update_layout(height=250)
                    st.plotly_chart(fig_fund, use_container_width=True)
                
                with detail_col2:
                    st.markdown("### 🎭 Sentiment")
                    
                    news = sentiment.get('news_sentiment', {})
                    
                    sent_data = {
                        "Source": ["Fear & Greed", "VIX", "News Sentiment"],
                        "Value": [
                            f"{fg.get('value', 'N/A')}/100",
                            f"{vix.get('value', 'N/A'):.1f}" if vix.get('value') else "N/A",
                            f"{news.get('score', 0):.2f}"
                        ],
                        "Status": [
                            fg.get('classification', 'N/A'),
                            vix.get('interpretation', 'N/A'),
                            news.get('interpretation', 'N/A')
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(sent_data), use_container_width=True, hide_index=True)
                    
                    # Gauge for sentiment
                    fig_sent = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment['combined_score'],
                        title={'text': "Sentiment Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 25], 'color': "red"},
                                {'range': [25, 40], 'color': "orange"},
                                {'range': [40, 60], 'color': "yellow"},
                                {'range': [60, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ]
                        }
                    ))
                    fig_sent.update_layout(height=250)
                    st.plotly_chart(fig_sent, use_container_width=True)
            
            # Mentor comment
            with st.expander("🧠 Mentor Refleksion", expanded=True):
                st.markdown(mentor_comment)
        
        except Exception as e:
            st.error(f"❌ Fejl ved fundamentals/sentiment analyse: {str(e)}")
    
    st.divider()
    
    # Risk Management Section
    st.subheader("📊 Risk Management")
    
    rm = signal_result['risk_management']
    current_price = rm['current_price']
    stop_loss = rm['stop_loss']
    target_price = rm['target_price']
    risk_reward = rm['risk_reward_ratio']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Stop Loss", f"${stop_loss:.2f}", 
                 delta=f"{rm['stop_loss_pct']:.2f}%",
                 delta_color="inverse")
    
    with col3:
        st.metric("Target Price", f"${target_price:.2f}",
                 delta=f"+{rm['target_price_pct']:.2f}%")
    
    with col4:
        st.metric("Risk/Reward", f"1:{risk_reward:.2f}",
                 help="Jo højere, jo bedre! Over 1.5 er godt.")
    
    # Visualize price levels
    fig = go.Figure()
    
    # Historical price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='lightblue', width=2)
    ))
    
    # Current price line
    fig.add_hline(y=current_price, line_dash="dash", line_color="white",
                  annotation_text="Current", annotation_position="right")
    
    # Stop loss line
    fig.add_hline(y=stop_loss, line_dash="dot", line_color="red",
                  annotation_text="Stop Loss", annotation_position="right")
    
    # Target price line
    fig.add_hline(y=target_price, line_dash="dot", line_color="green",
                  annotation_text="Target", annotation_position="right")
    
    fig.update_layout(
        title=f"{symbol} - Price Levels",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Macro Indicators Section (NEW!)
    if 'macro_indicators' in signal_result and signal_result['macro_indicators']:
        st.subheader("🌍 Makroøkonomiske Indikatorer")
        
        macro = signal_result['macro_indicators']
        
        col1, col2, col3 = st.columns(3)
        
        # VIX (Fear Index)
        with col1:
            if macro.get('vix'):
                vix = macro['vix']
                
                # Color based on level
                if vix['level'] == 'Low':
                    vix_color = "🟢"
                    delta_color = "normal"
                elif vix['level'] == 'High':
                    vix_color = "🔴"
                    delta_color = "inverse"
                else:
                    vix_color = "🟡"
                    delta_color = "off"
                
                st.metric(
                    f"{vix_color} VIX (Volatility Index)",
                    f"{vix['current']:.2f}",
                    delta=vix['level'],
                    delta_color=delta_color,
                    help="VIX måler markedsvolatilitet. Lav VIX = roligt marked, Høj VIX = nervøst marked"
                )
                
                with st.expander("ℹ️ VIX Detaljer"):
                    st.markdown(f"**Niveau**: {vix['level']}")
                    st.markdown(f"**Fortolkning**: {vix['interpretation']}")
                    st.markdown("**Hvad er VIX?**")
                    st.markdown("VIX (CBOE Volatility Index) kaldes også 'Fear Index'. Den måler forventet volatilitet i S&P 500 over de næste 30 dage.")
        
        # Fear & Greed Index
        with col2:
            if macro.get('fear_greed'):
                fg = macro['fear_greed']
                
                # Color based on level
                if fg['sentiment'] in ['greed', 'extreme_greed']:
                    fg_color = "🟢"
                    fg_emoji = "😊" if fg['sentiment'] == 'greed' else "🤑"
                elif fg['sentiment'] in ['fear', 'extreme_fear']:
                    fg_color = "🔴"
                    fg_emoji = "😰" if fg['sentiment'] == 'fear' else "😱"
                else:
                    fg_color = "🟡"
                    fg_emoji = "😐"
                
                st.metric(
                    f"{fg_color} Fear & Greed Index",
                    f"{fg['value']}/100 {fg_emoji}",
                    delta=fg['level'],
                    help="Måler markedets sentiment. 0 = Ekstrem frygt, 100 = Ekstrem grådighed"
                )
                
                with st.expander("ℹ️ Fear & Greed Detaljer"):
                    st.markdown(f"**Niveau**: {fg['level']}")
                    st.markdown(f"**Fortolkning**: {fg['interpretation']}")
                    st.markdown("**7 Komponenter (CNN Methodology):**")
                    
                    # Display all 7 components from the enhanced calculation
                    if fg.get('components'):
                        for component_name, component_score in fg['components'].items():
                            st.markdown(f"- {component_name}: {component_score}")
                    
                    st.markdown("\n**Hvad er Fear & Greed?**")
                    st.markdown("Denne indikator kombinerer 7 markedsindikatorer (aligned med CNN Fear & Greed Index) for at måle om investorer er bange (fear) eller grådige (greed). Ekstrem frygt kan være købs-mulighed, ekstrem grådighed kan signalere risiko for korrektion.")
                    st.markdown("\n**Komponenter inkluderer:** Market Momentum, Price Strength, Breadth, Put/Call Ratio, Volatility, Safe Haven Demand, og Junk Bond Demand.")
        
        # Shiller P/E Ratio
        with col3:
            if macro.get('shiller_pe'):
                pe = macro['shiller_pe']
                
                # Color based on level
                if pe['level'] == 'Undervalued':
                    pe_color = "🟢"
                elif pe['level'] in ['Overvalued', 'Highly Overvalued']:
                    pe_color = "🔴"
                else:
                    pe_color = "🟡"
                
                st.metric(
                    f"{pe_color} Shiller P/E (CAPE)",
                    f"{pe['cape_ratio']:.2f}",
                    delta=f"Avg: {pe['historical_avg']}",
                    help="Cyclically Adjusted P/E ratio. Måler om markedet er over/undervurderet"
                )
                
                with st.expander("ℹ️ Shiller P/E Detaljer"):
                    st.markdown(f"**Niveau**: {pe['level']}")
                    st.markdown(f"**Fortolkning**: {pe['interpretation']}")
                    st.markdown(f"**S&P 500 P/E**: {pe['sp500_pe']}")
                    st.markdown(f"**Historisk gennemsnit**: {pe['historical_avg']}")
                    st.markdown("\n**Hvad er Shiller P/E?**")
                    st.markdown("CAPE (Cyclically Adjusted Price-to-Earnings) ratio er udviklet af Robert Shiller. Den måler S&P 500's værdiansættelse ved at sammenligne prisen med 10-års gennemsnitlig indtjening justeret for inflation. Høj CAPE = dyrt marked, Lav CAPE = billigt marked.")
        
        # Overall sentiment
        overall_sentiment = macro.get('overall_sentiment', 'neutral')
        if overall_sentiment == 'bullish':
            sentiment_emoji = "🟢"
            sentiment_text = "Bullish"
            sentiment_desc = "Makroindikatorer peger generelt i positiv retning"
        elif overall_sentiment == 'bearish':
            sentiment_emoji = "🔴"
            sentiment_text = "Bearish"
            sentiment_desc = "Makroindikatorer peger generelt i negativ retning"
        else:
            sentiment_emoji = "🟡"
            sentiment_text = "Neutral"
            sentiment_desc = "Makroindikatorer er blandede"
        
        st.info(f"{sentiment_emoji} **Overall Makro-Sentiment: {sentiment_text}** - {sentiment_desc}")
        
        st.divider()
    
    # Reasoning Section
    st.subheader("🧠 Agent Reasoning")
    
    reasoning_col1, reasoning_col2 = st.columns([2, 1])
    
    with reasoning_col1:
        st.markdown("**Hvorfor dette signal?**")
        for reason in signal_result['reasoning']:
            st.markdown(f"- {reason}")
    
    with reasoning_col2:
        st.markdown("**Score Breakdown**")
        scores = signal_result['scores']
        
        # Score visualization
        score_fig = go.Figure(go.Bar(
            x=[scores['ml_score'], scores['rsi_score'], scores['ma_score'], scores['macd_score'], scores.get('macro_score', 0)],
            y=['ML Forecast', 'RSI', 'Moving Avg', 'MACD', 'Macro'],
            orientation='h',
            marker=dict(
                color=[scores['ml_score'], scores['rsi_score'], scores['ma_score'], scores['macd_score'], scores.get('macro_score', 0)],
                colorscale='RdYlGn',
                cmid=0
            )
        ))
        
        score_fig.update_layout(
            title="Score Components",
            xaxis_title="Score",
            template="plotly_dark",
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(score_fig, use_container_width=True)
    
    st.divider()
    
    # User feedback section
    st.subheader("📝 Feedback til Agent")
    st.markdown("*Hjælp agenten med at lære - giv feedback på anbefalingen*")
    
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    
    with feedback_col1:
        if st.button("👍 Enig - Godt signal!", use_container_width=True):
            st.success("✅ Tak for feedback! Agent lærer af dine inputs.")
    
    with feedback_col2:
        if st.button("👎 Uenig - Dårligt signal!", use_container_width=True):
            st.warning("📝 Tak for feedback! Agent vil justere vægtning.")
    
    with feedback_col3:
        if st.button("🤷 Usikker / Ved ikke", use_container_width=True):
            st.info("ℹ️ Tak for feedback! Hold øje med resultatet.")
    
    # Additional info
    with st.expander("ℹ️ Hvordan virker agenten?"):
        st.markdown("""
        **Agent Intelligence System:**
        
        **TRADING SIGNALS:**
        1. **ML Forecasts (40 points)**: Kombinerer forecasts fra flere modeller
        2. **RSI Indikator (20 points)**: Oversold/Overbought analyse
        3. **Moving Averages (20 points)**: Trend retning (SMA20, SMA50)
        4. **MACD (20 points)**: Momentum indikator
        5. **Makro-Indikatorer (30 points)**: VIX, Fear & Greed, Shiller P/E
        
        **FUNDAMENTALS + SENTIMENT:**
        - **Fundamental Score (0-100)**: P/E, PEG, P/B, Growth
        - **Sentiment Score (0-100)**: Fear & Greed + VIX + News
        - **Mentor Refleksion**: Kombinerer rationel analyse med markedspsykologi
        
        **Makro-Indikatorer Breakdown:**
        - **VIX (10 points)**: Markedsvolatilitet - Lav VIX = bullish, Høj VIX = bearish
        - **Fear & Greed (10 points)**: Markedsstemning - Grådighed = bullish, Frygt = bearish
        - **Shiller P/E (10 points)**: Værdiansættelse - Lav P/E = bullish, Høj P/E = bearish
        
        **Scoring System:**
        - **BUY Signal**: Total score > 40%
        - **SELL Signal**: Total score < -40%
        - **HOLD Signal**: Total score mellem -40% og 40%
        
        **Fundamental Scoring:**
        - **75-100**: Fundamentalt stærk - købs-kandidat
        - **60-74**: Rimeligt prissat - kræver timing
        - **45-59**: Lidt overvurderet - pas på
        - **0-44**: Overprissat - undgå
        
        **Risk Management:**
        - **Stop Loss**: 2x ATR under current price (begrænser tab)
        - **Target Price**: 3x ATR over current price (profit target)
        - **Risk/Reward**: Forholdet mellem potentiel profit og risiko
        
        **Confidence Score**: Baseret på hvor stærkt signalet er (50-100%)
        
        **Mentor Approach**: Kombinerer:
        - Rationel værdiansættelse (fundamentals)
        - Markedspsykologi (sentiment)
        - Bias-detektion (FOMO, panik)
        - Kontekstuel anbefaling
        """)


else:
    # Welcome screen
    st.info("👈 Vælg en aktie og klik 'Analyser' for at få agent recommendations")
    
    st.markdown("""
    ### 🎯 Hvad er Agent Recommendations?
    
    Dette er en intelligent trading agent der:
    
    **TRADING SIGNALS:**
    - 🤖 **Kombinerer ML forecasts** fra flere modeller (RF, XGBoost, Prophet, Ensemble, LSTM)
    - 📊 **Analyserer tekniske indikatorer** (RSI, Moving Averages, MACD)
    - 🌍 **Inkluderer makro-indikatorer** (VIX, Fear & Greed, Shiller P/E)
    - 🎯 **Genererer klare signaler**: BUY, SELL eller HOLD
    - 🛡️ **Beregner risk management**: Stop loss, target price, risk/reward ratio
    - 🧠 **Forklarer reasoning**: Hvorfor dette signal?
    
    **FUNDAMENTALS + SENTIMENT:**
    - 📊 **Fundamental Score**: P/E, PEG, P/B, Growth analyse
    - 🎭 **Sentiment Score**: Fear & Greed + VIX + News sentiment
    - 🧠 **Mentor Refleksion**: Kombinerer rationel analyse med markedspsykologi
    - ⚠️ **Bias Detection**: Identificerer FOMO, panik og hype
    
    ### 🚀 Sådan bruger du det:
    
    1. Vælg en aktie symbol (f.eks. AAPL, TSLA, MSFT)
    2. Vælg hvilke ML modeller der skal bruges
    3. Klik 'Analyser'
    4. Få klar BUY/SELL/HOLD anbefaling med confidence score
    5. Se fundamentals + sentiment analyse
    6. Læs mentor refleksion - værdiansat + markedspsykologi
    7. Se risk management levels (stop loss, target)
    
    **Note**: Dette er et analyseværktøj - ikke finansiel rådgivning!
    """)
