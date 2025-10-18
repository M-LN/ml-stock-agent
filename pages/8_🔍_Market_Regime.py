"""
Streamlit UI Component for Market Regime Detection
Viser current market regime og tillader regime-aware model training
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from market_regime_detector import MarketRegimeDetector, get_current_regime

# Try to import trainer, but handle gracefully if TensorFlow fails
try:
    from regime_aware_trainer import RegimeAwareTrainer, train_regime_ensemble
    TRAINER_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Regime training not available: {str(e)[:100]}...")
    RegimeAwareTrainer = None
    train_regime_ensemble = None
    TRAINER_AVAILABLE = False


def display_regime_dashboard(symbol: str, period: str = '6mo'):
    """
    Viser market regime dashboard for et givet symbol.
    """
    st.markdown(f"### 🔍 Market Regime Analysis for {symbol}")
    
    with st.spinner("Analyzing market regime..."):
        result = get_current_regime(symbol, period)
    
    # Regime badge
    regime_colors = {
        'bull': '🟢',
        'bear': '🔴',
        'sideways': '🟡',
        'high_volatility': '🟠',
        'unknown': '⚪'
    }
    
    regime = result['regime']
    color_badge = regime_colors.get(regime, '⚪')
    
    st.markdown(f"""
    ### {color_badge} Current Regime: **{regime.upper()}**
    **Confidence:** {result['confidence']:.0%}
    
    {result['description']}
    """)
    
    # Metrics
    if result['metrics']:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Trend Strength",
                f"{result['metrics']['trend_strength']:.3f}%/day",
                delta=None
            )
        
        with col2:
            st.metric(
                "Volatility",
                f"{result['metrics']['volatility']:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Consistency",
                f"{result['metrics']['consistency']:.0%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{result['metrics']['max_drawdown']:.1%}",
                delta=None
            )
    
    # Recommendations
    detector = MarketRegimeDetector()
    characteristics = detector.get_regime_characteristics(regime)
    
    if characteristics:
        st.markdown("---")
        st.markdown("### 💡 Recommendations for This Regime")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Best Models:**
            {', '.join([m.upper() for m in characteristics['best_models']])}
            
            **Recommended Window:**
            {characteristics['recommended_window']} days
            """)
        
        with col2:
            st.markdown(f"""
            **Risk Level:**
            {characteristics['risk_level'].upper()}
            
            **Strategy:**
            {characteristics['description']}
            """)
    
    return result


def regime_training_interface():
    """
    UI for træning af regime-specific models.
    """
    st.markdown("### 🎯 Regime-Aware Model Training")
    st.markdown("Train separate models for different market conditions for better accuracy.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", key="regime_symbol")
        period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1, key="regime_period")
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "XGBoost", "LSTM"],
            key="regime_model_type"
        )
    
    # Show current regime first
    if st.button("🔍 Analyze Current Regime", key="analyze_regime_btn"):
        display_regime_dashboard(symbol, period='6mo')
    
    st.markdown("---")
    
    # Training section
    st.markdown("#### Train Regime-Specific Models")
    st.info("""
    This will:
    1. Segment historical data by market regime (Bull, Bear, Sideways, High Volatility)
    2. Train separate models for each regime
    3. Save models that can auto-select based on current conditions
    """)
    
    if st.button("🚀 Train Regime Models", type="primary", key="train_regime_btn"):
        with st.spinner(f"Training regime-specific models for {symbol}..."):
            try:
                # Download data
                data = yf.download(symbol, period=period, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if len(data) < 200:
                    st.error(f"Insufficient data for {symbol}. Need at least 200 days.")
                    return
                
                # Train models
                trainer = RegimeAwareTrainer()
                
                model_type_map = {
                    'Random Forest': 'rf',
                    'XGBoost': 'xgboost',
                    'LSTM': 'lstm'
                }
                
                model_key = model_type_map[model_type]
                
                trained_models = trainer.train_regime_specific_models(
                    data, symbol,
                    model_type=model_key,
                    n_estimators=150,
                    max_depth=12,
                    window=30
                )
                
                if trained_models:
                    st.success(f"✅ Successfully trained {len(trained_models)} regime-specific models!")
                    
                    st.markdown("### 📊 Trained Models:")
                    for regime, model_id in trained_models.items():
                        st.markdown(f"- **{regime.upper()}**: `{model_id}`")
                    
                    st.info("💡 These models will automatically be used when the market is in their respective regime.")
                else:
                    st.warning("⚠️ No models were trained. May need more historical data.")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")


def regime_chart(symbol: str, period: str = '6mo'):
    """
    Creates a chart showing price with regime overlays.
    """
    import yfinance as yf
    
    data = yf.download(symbol, period=period, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Detect regimes over time
    detector = MarketRegimeDetector(lookback_period=20)
    
    regimes = []
    for i in range(20, len(data), 5):  # Every 5 days
        window = data.iloc[max(0, i-20):i]
        result = detector.detect_regime(window)
        regimes.append({
            'date': data.index[i],
            'regime': result['regime'],
            'confidence': result['confidence']
        })
    
    regimes_df = pd.DataFrame(regimes)
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add regime markers
    regime_colors_plot = {
        'bull': 'green',
        'bear': 'red',
        'sideways': 'gray',
        'high_volatility': 'orange'
    }
    
    for regime in ['bull', 'bear', 'sideways', 'high_volatility']:
        regime_data = regimes_df[regimes_df['regime'] == regime]
        if len(regime_data) > 0:
            fig.add_trace(go.Scatter(
                x=regime_data['date'],
                y=[data.loc[d, 'Close'] for d in regime_data['date']],
                mode='markers',
                name=regime.capitalize(),
                marker=dict(
                    size=10,
                    color=regime_colors_plot[regime],
                    symbol='circle'
                )
            ))
    
    fig.update_layout(
        title=f"{symbol} Price with Market Regimes",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    return fig


# Example usage in a Streamlit page
if __name__ == "__main__":
    st.set_page_config(page_title="Market Regime Analysis", page_icon="🔍", layout="wide")
    
    st.title("🔍 Market Regime Analysis")
    st.markdown("Identify market conditions and train regime-specific models")
    
    tab1, tab2, tab3 = st.tabs(["📊 Current Regime", "🎯 Train Models", "📈 Regime History"])
    
    with tab1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
        
        if st.button("Analyze Regime"):
            result = display_regime_dashboard(symbol)
            
            # Show chart
            st.markdown("---")
            fig = regime_chart(symbol)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        regime_training_interface()
    
    with tab3:
        st.markdown("### 📈 Regime Detection History")
        st.info("Shows how market regime has changed over time (Coming soon)")
