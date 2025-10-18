import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from agent_interactive import (
    ml_forecast, ml_forecast_rf, ml_forecast_xgboost, 
    ml_forecast_prophet, ml_forecast_ensemble,
    hent_data, calculate_model_metrics, backtest_model,
    list_saved_models, load_model, predict_with_saved_model
)
from prediction_tracker import save_prediction
from regime_prediction import RegimePredictionSystem
from market_regime_detector import get_current_regime

st.set_page_config(page_title="ML Forecast", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Forecast")
st.markdown("Forudsig fremtidige priser med LSTM og Random Forest modeller")

# Initialize session state
if 'ml_data' not in st.session_state:
    st.session_state.ml_data = None
if 'ml_symbol' not in st.session_state:
    st.session_state.ml_symbol = None

# Sidebar inputs
st.sidebar.title("⚙️ ML Indstillinger")

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
                if col.button(stock, key=f"ml_btn_{stock}", use_container_width=True):
                    st.session_state['ml_selected_symbol'] = stock

st.sidebar.markdown("---")

# Symbol input (with session state support)
default_symbol = st.session_state.get('ml_selected_symbol', 'AAPL')
symbol = st.sidebar.text_input("Aktiesymbol", value=default_symbol, placeholder="fx MSFT, TSLA").upper()
if symbol:
    st.session_state['ml_selected_symbol'] = symbol

st.sidebar.markdown("### �📅 Forecast Horizont")
horizons = st.sidebar.multiselect(
    "Vælg tidsperioder",
    options=["1 dag", "5 dage", "22 dage (1 måned)"],
    default=["1 dag", "5 dage"]
)

horizon_map = {
    "1 dag": 1,
    "5 dage": 5,
    "22 dage (1 måned)": 22
}

st.sidebar.markdown("---")

# 🆕 Regime Detection Section
st.sidebar.markdown("### 🔍 Market Regime Detection")
use_regime_aware = st.sidebar.checkbox(
    "🎯 Auto-select regime-specific model", 
    value=False,
    help="Automatically select best model based on current market regime"
)

if use_regime_aware:
    with st.sidebar:
        try:
            regime_result = get_current_regime(symbol, period='6mo')
            regime = regime_result['regime']
            confidence = regime_result['confidence']
            
            # Regime badge
            regime_colors = {
                'bull': '🟢',
                'bear': '🔴',
                'sideways': '🟡',
                'high_volatility': '🟠'
            }
            
            badge = regime_colors.get(regime, '⚪')
            st.info(f"{badge} **{regime.upper()}** ({confidence:.0%} confidence)")
            
            # Check if regime models exist
            system = RegimePredictionSystem()
            coverage = system.get_regime_model_coverage(symbol, 'rf')
            
            if coverage['has_regime_models']:
                if coverage['coverage'].get(regime, False):
                    st.success(f"✅ Regime-specific model available")
                else:
                    st.warning(f"⚠️ No model for {regime} regime. Will use standard model.")
            else:
                st.warning("⚠️ No regime models found. Train them in Market Regime page.")
        except Exception as e:
            st.error(f"Regime detection failed: {str(e)}")
            use_regime_aware = False

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔧 Brug Gemt Model?")
use_saved_model = st.sidebar.checkbox("📂 Brug gemt model i stedet", value=False,
                                     help="Brug en tidligere trænet model",
                                     disabled=use_regime_aware)  # Disable when regime-aware is active

if use_regime_aware:
    st.sidebar.info("💡 Regime-aware mode active. Model selection handled automatically.")

if use_saved_model and not use_regime_aware:
    # Get ONLY deployed models
    all_saved_models = list_saved_models(symbol=symbol)
    saved_models = [m for m in all_saved_models if m.get('deployed', False)]
    
    if not saved_models:
        st.sidebar.warning(f"⚠️ Ingen deployed modeller for {symbol}. Deploy en i Model Management.")
        use_saved_model = False
    else:
        model_options = [f"{m['model_type'].upper()} - {m['timestamp'][:8]} (MAE: ${m['metadata'].get('train_mae', 0):.2f})" 
                        for m in saved_models]
        selected_idx = st.sidebar.selectbox("Vælg model", range(len(model_options)), 
                                           format_func=lambda x: model_options[x])
        selected_saved_model = saved_models[selected_idx]
        st.sidebar.success(f"✅ Bruger: {selected_saved_model['model_type'].upper()}")

if not use_saved_model:
    st.sidebar.markdown("### 🧠 Modeller")
    use_lstm = st.sidebar.checkbox("LSTM Neural Network", value=False)
    use_rf = st.sidebar.checkbox("Random Forest", value=True)
    use_xgboost = st.sidebar.checkbox("⚡ XGBoost", value=True, help="Gradient Boosting - ofte bedre end RF")
    use_prophet = st.sidebar.checkbox("📈 Prophet (Facebook)", value=False, help="God til trends og seasonality")
    use_ensemble = st.sidebar.checkbox("🎯 Ensemble (Alle)", value=False, help="Kombinerer alle modeller")
    
    lstm_epochs = st.sidebar.slider("LSTM Epochs", 10, 100, 50) if use_lstm else 50
else:
    # Disable individual model selection when using saved model
    use_lstm = use_rf = use_xgboost = use_prophet = use_ensemble = False
    lstm_epochs = 50

# Check if auto-start flag is set (from Watchlist)
auto_start_ml = st.session_state.get('auto_ml', False)
if auto_start_ml:
    st.info(f"🚀 Auto-starter ML Forecast for {symbol} fra Watchlist...")
    st.session_state['auto_ml'] = False  # Reset flag after use

# Main content
if st.button("🚀 Start Forecast", type="primary", use_container_width=True) or auto_start_ml:
    if not horizons:
        st.error("❌ Vælg mindst én forecast horizont!")
    elif not any([use_lstm, use_rf, use_xgboost, use_prophet, use_ensemble]):
        st.error("❌ Vælg mindst én model!")
    else:
        with st.spinner(f'Træner modeller for {symbol}...'):
            try:
                # Hent data OG gem i session state
                data = hent_data(symbol, '2y')
                st.session_state.ml_data = data
                st.session_state.ml_symbol = symbol
                
                if data.empty:
                    st.error(f"❌ Ingen data fundet for {symbol}")
                else:
                    current_price = data['Close'].iloc[-1]
                    
                    # Header
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Aktuel Kurs", f"${current_price:.2f}")
                    with col2:
                        st.metric("Data Punkter", len(data))
                    with col3:
                        st.metric("Modeller", f"{int(use_lstm) + int(use_rf)}")
                    
                    st.markdown("---")
                    
                    # SAVED MODEL Predictions
                    if use_saved_model:
                        st.markdown(f"## 📂 Gemt Model: {selected_saved_model['model_type'].upper()}")
                        
                        saved_results = []
                        model_package = load_model(selected_saved_model['filepath'])
                        
                        if model_package:
                            for h_label in horizons:
                                h_days = horizon_map[h_label]
                                
                                with st.spinner(f'Prediction for {h_label}...'):
                                    result = predict_with_saved_model(model_package, data, horizon=h_days)
                                    
                                    if result:
                                        pred = result['forecast']
                                        change = pred - current_price
                                        change_pct = (change / current_price) * 100
                                        
                                        saved_results.append({
                                            'Horizont': h_label,
                                            'Forecast': f"${pred:.2f}",
                                            'Ændring': f"${change:+.2f}",
                                            'Ændring %': f"{change_pct:+.2f}%"
                                        })
                                        
                                        # Save prediction for tracking
                                        try:
                                            prediction_date = datetime.now().strftime("%Y-%m-%d")
                                            predictions_list = [pred]
                                            
                                            # Extract model ID from filepath
                                            model_id = selected_saved_model['filepath'].split('/')[-1].replace('.pkl', '').replace('.h5', '')
                                            
                                            save_prediction(
                                                model_id=model_id,
                                                symbol=symbol,
                                                model_type=selected_saved_model['model_type'],
                                                prediction_date=prediction_date,
                                                predictions=predictions_list,
                                                horizon=h_days,
                                                metadata={
                                                    'deployed': True,
                                                    'current_price': float(current_price),
                                                    'forecast_method': 'saved_model',
                                                    'model_metadata': model_package.get('metadata', {})
                                                }
                                            )
                                        except Exception as e:
                                            pass
                            
                            if saved_results:
                                df_saved = pd.DataFrame(saved_results)
                                st.dataframe(df_saved, use_container_width=True)
                                
                                # Show model info
                                with st.expander("ℹ️ Model Information"):
                                    st.json(model_package['metadata'])
                            else:
                                st.warning("⚠️ Model kunne ikke generere forudsigelser")
                    
                    # LSTM Predictions
                    lstm_results = []
                    if use_lstm:
                        st.markdown("## 🧠 LSTM Neural Network")
                        
                        for h_label in horizons:
                            h_days = horizon_map[h_label]
                            
                            with st.spinner(f'LSTM træning for {h_label}...'):
                                result = ml_forecast(data, window=30, epochs=lstm_epochs, horizon=h_days)
                                
                                if result:
                                    pred = result['forecast']
                                    change = pred - current_price
                                    change_pct = (change / current_price) * 100
                                    
                                    lstm_results.append({
                                        'Horizont': h_label,
                                        'Forecast': f"${pred:.2f}",
                                        'Ændring': f"${change:+.2f}",
                                        'Ændring %': f"{change_pct:+.2f}%"
                                    })
                        
                        if lstm_results:
                            df_lstm = pd.DataFrame(lstm_results)
                            st.dataframe(df_lstm, use_container_width=True)
                        else:
                            st.warning("⚠️ LSTM kunne ikke generere forudsigelser")
                    
                    # Random Forest Predictions
                    rf_results = []
                    if use_rf:
                        st.markdown("## 🌲 Random Forest")
                        
                        for h_label in horizons:
                            h_days = horizon_map[h_label]
                            
                            with st.spinner(f'Random Forest træning for {h_label}...'):
                                result = ml_forecast_rf(data, window=30, horizon=h_days)
                                
                                if result:
                                    pred = result['forecast']
                                    change = pred - current_price
                                    change_pct = (change / current_price) * 100
                                    
                                    rf_results.append({
                                        'Horizont': h_label,
                                        'Forecast': f"${pred:.2f}",
                                        'Ændring': f"${change:+.2f}",
                                        'Ændring %': f"{change_pct:+.2f}%"
                                    })
                                    
                                    # Save prediction for tracking
                                    try:
                                        prediction_date = datetime.now().strftime("%Y-%m-%d")
                                        # Generate predictions for each day up to horizon
                                        predictions_list = [pred]  # For now, single point prediction
                                        
                                        save_prediction(
                                            model_id=f"rf_{symbol}_forecast",
                                            symbol=symbol,
                                            model_type='rf',
                                            prediction_date=prediction_date,
                                            predictions=predictions_list,
                                            horizon=h_days,
                                            metadata={
                                                'window': 30,
                                                'current_price': float(current_price),
                                                'forecast_method': 'live_forecast'
                                            }
                                        )
                                    except Exception as e:
                                        pass  # Don't break forecast if tracking fails
                        
                        if rf_results:
                            df_rf = pd.DataFrame(rf_results)
                            st.dataframe(df_rf, use_container_width=True)
                        else:
                            st.warning("⚠️ Random Forest kunne ikke generere forudsigelser")
                    
                    # XGBoost Predictions
                    xgb_results = []
                    if use_xgboost:
                        st.markdown("## ⚡ XGBoost")
                        
                        for h_label in horizons:
                            h_days = horizon_map[h_label]
                            
                            with st.spinner(f'XGBoost træning for {h_label}...'):
                                result = ml_forecast_xgboost(data, window=30, horizon=h_days)
                                
                                if result:
                                    pred = result['forecast']
                                    change = pred - current_price
                                    change_pct = (change / current_price) * 100
                                    
                                    xgb_results.append({
                                        'Horizont': h_label,
                                        'Forecast': f"${pred:.2f}",
                                        'Ændring': f"${change:+.2f}",
                                        'Ændring %': f"{change_pct:+.2f}%"
                                    })
                                    
                                    # Save prediction for tracking
                                    try:
                                        prediction_date = datetime.now().strftime("%Y-%m-%d")
                                        predictions_list = [pred]
                                        
                                        save_prediction(
                                            model_id=f"xgboost_{symbol}_forecast",
                                            symbol=symbol,
                                            model_type='xgboost',
                                            prediction_date=prediction_date,
                                            predictions=predictions_list,
                                            horizon=h_days,
                                            metadata={
                                                'window': 30,
                                                'current_price': float(current_price),
                                                'forecast_method': 'live_forecast'
                                            }
                                        )
                                    except Exception as e:
                                        pass
                        
                        if xgb_results:
                            df_xgb = pd.DataFrame(xgb_results)
                            st.dataframe(df_xgb, use_container_width=True)
                        else:
                            st.warning("⚠️ XGBoost kunne ikke generere forudsigelser")
                    
                    # Prophet Predictions
                    prophet_results = []
                    if use_prophet:
                        st.markdown("## 📈 Prophet (Facebook)")
                        
                        for h_label in horizons:
                            h_days = horizon_map[h_label]
                            
                            with st.spinner(f'Prophet træning for {h_label}...'):
                                result = ml_forecast_prophet(data, horizon=h_days)
                                
                                if result:
                                    pred = result['forecast']
                                    change = pred - current_price
                                    change_pct = (change / current_price) * 100
                                    
                                    prophet_results.append({
                                        'Horizont': h_label,
                                        'Forecast': f"${pred:.2f}",
                                        'Ændring': f"${change:+.2f}",
                                        'Ændring %': f"{change_pct:+.2f}%"
                                    })
                        
                        if prophet_results:
                            df_prophet = pd.DataFrame(prophet_results)
                            st.dataframe(df_prophet, use_container_width=True)
                        else:
                            st.warning("⚠️ Prophet kunne ikke generere forudsigelser")
                    
                    # Ensemble Predictions
                    ensemble_results = []
                    if use_ensemble:
                        st.markdown("## 🎯 Ensemble (Kombineret)")
                        
                        for h_label in horizons:
                            h_days = horizon_map[h_label]
                            
                            with st.spinner(f'Ensemble træning for {h_label}...'):
                                result = ml_forecast_ensemble(data, window=30, horizon=h_days)
                                
                                if result:
                                    pred = result['forecast']
                                    change = pred - current_price
                                    change_pct = (change / current_price) * 100
                                    
                                    # Show which models were used
                                    models_used = ', '.join(result.get('models_used', []))
                                    
                                    ensemble_results.append({
                                        'Horizont': h_label,
                                        'Forecast': f"${pred:.2f}",
                                        'Ændring': f"${change:+.2f}",
                                        'Ændring %': f"{change_pct:+.2f}%",
                                        'Modeller': models_used
                                    })
                        
                        if ensemble_results:
                            df_ensemble = pd.DataFrame(ensemble_results)
                            st.dataframe(df_ensemble, use_container_width=True)
                            st.info("💡 Ensemble bruger weighted average af alle tilgængelige modeller")
                        else:
                            st.warning("⚠️ Ensemble kunne ikke generere forudsigelser")
                    
                    st.success("✅ Forecast komplet!")
                    
                    # Show tracking info
                    st.info("📊 **Predictions tracked!** Gå til Performance Dashboard → Live Tracking for at følge hvordan disse predictions performer over tid.")
                    
            except Exception as e:
                st.error(f"❌ Fejl: {e}")
                import traceback
                st.code(traceback.format_exc())

# Backtest sektion - vises kun hvis data findes
if st.session_state.ml_data is not None:
    data = st.session_state.ml_data
    symbol = st.session_state.ml_symbol
    current_price = data['Close'].iloc[-1]
    
    st.markdown("---")
    st.markdown("## 📊 Model Performance Metrics")
    st.markdown("Test modellernes nøjagtighed på historiske data")
    
    # Backtest settings
    col1, col2, col3 = st.columns(3)
    with col1:
        test_periods = st.number_input("Test Perioder", min_value=10, max_value=100, value=30, 
                                      help="Antal historiske perioder at teste på")
    with col2:
        test_horizon = st.selectbox("Test Horizont", [1, 5, 22], index=0,
                                    help="Forecast horizont for backtesting")
    with col3:
        test_window = st.number_input("Training Window", min_value=20, max_value=100, value=30,
                                     help="Antal dage brugt til træning")
    
    # Choose model to backtest
    test_model = st.radio("Vælg model at teste", 
                         ["Random Forest", "XGBoost", "Prophet", "Ensemble", "LSTM"],
                         horizontal=True)
    
    if st.button("🔬 Kør Backtest", type="secondary", use_container_width=True):
        model_map = {
            'Random Forest': 'rf',
            'XGBoost': 'xgboost',
            'Prophet': 'prophet',
            'Ensemble': 'ensemble',
            'LSTM': 'lstm'
        }
        model_type = model_map.get(test_model, 'rf')
        
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Progress callback function
        def update_progress(current, total, message):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"{message} ({current}/{total})")
        
        try:
            backtest_results = backtest_model(
                data=data,
                model_type=model_type,
                window=int(test_window),
                horizon=int(test_horizon),
                test_periods=int(test_periods),
                progress_callback=update_progress
            )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if backtest_results and 'metrics' in backtest_results:
                    metrics = backtest_results['metrics']
                    
                    st.markdown(f"### 🎯 {test_model} Backtest Resultater")
                    
                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        mae_value = metrics.get('MAE', 0)
                        mae_quality = "✅" if mae_value < current_price * 0.02 else "⚠️" if mae_value < current_price * 0.05 else "❌"
                        st.metric("MAE", f"${mae_value:.2f}", delta=mae_quality, delta_color="off")
                        st.caption("Mean Absolute Error")
                    
                    with col2:
                        rmse_value = metrics.get('RMSE', 0)
                        rmse_quality = "✅" if rmse_value < current_price * 0.03 else "⚠️" if rmse_value < current_price * 0.06 else "❌"
                        st.metric("RMSE", f"${rmse_value:.2f}", delta=rmse_quality, delta_color="off")
                        st.caption("Root Mean Square Error")
                    
                    with col3:
                        mape_value = metrics.get('MAPE', 0)
                        mape_quality = "✅" if mape_value < 5 else "⚠️" if mape_value < 10 else "❌"
                        st.metric("MAPE", f"{mape_value:.2f}%", delta=mape_quality, delta_color="off")
                        st.caption("Mean Abs. % Error")
                    
                    with col4:
                        r2_value = metrics.get('R2', 0)
                        r2_quality = "✅" if r2_value > 0.7 else "⚠️" if r2_value > 0.4 else "❌"
                        st.metric("R²", f"{r2_value:.3f}", delta=r2_quality, delta_color="off")
                        st.caption("Coefficient of Determination")
                    
                    with col5:
                        win_rate = metrics.get('Win_Rate', 0)
                        wr_quality = "✅" if win_rate > 60 else "⚠️" if win_rate > 50 else "❌"
                        st.metric("Win Rate", f"{win_rate:.1f}%", delta=wr_quality, delta_color="off")
                        st.caption("Direction Accuracy")
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Fortolkning")
                    
                    # Calculate overall confidence
                    confidence_score = 0
                    if mae_value < current_price * 0.02:
                        confidence_score += 20
                    elif mae_value < current_price * 0.05:
                        confidence_score += 10
                    
                    if rmse_value < current_price * 0.03:
                        confidence_score += 20
                    elif rmse_value < current_price * 0.06:
                        confidence_score += 10
                    
                    if mape_value < 5:
                        confidence_score += 20
                    elif mape_value < 10:
                        confidence_score += 10
                    
                    if r2_value > 0.7:
                        confidence_score += 20
                    elif r2_value > 0.4:
                        confidence_score += 10
                    
                    if win_rate > 60:
                        confidence_score += 20
                    elif win_rate > 50:
                        confidence_score += 10
                    
                    # Display confidence level
                    if confidence_score >= 80:
                        st.success(f"🎉 **Høj tillid** ({confidence_score}/100) - Modellen performer godt!")
                    elif confidence_score >= 50:
                        st.warning(f"⚠️ **Moderat tillid** ({confidence_score}/100) - Modellen er okay")
                    else:
                        st.error(f"❌ **Lav tillid** ({confidence_score}/100) - Modellen er ikke pålidelig")
                    
                    # Backtest visualization
                    if 'actual_prices' in backtest_results and 'predicted_prices' in backtest_results:
                        actual = backtest_results['actual_prices']
                        predicted = backtest_results['predicted_prices']
                        dates = backtest_results.get('forecast_dates', list(range(len(actual))))
                        
                        if len(actual) > 0 and len(predicted) > 0:
                            st.markdown("### 📈 Backtest Visualisering")
                            
                            fig = go.Figure()
                            
                            # Actual prices
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=actual,
                                mode='lines+markers',
                                name='Faktiske Priser',
                                line=dict(color='#3b82f6', width=2),
                                marker=dict(size=6)
                            ))
                            
                            # Predicted prices
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=predicted,
                                mode='lines+markers',
                                name='Forudsagte Priser',
                                line=dict(color='#f59e0b', width=2, dash='dot'),
                                marker=dict(size=6)
                            ))
                            
                            fig.update_layout(
                                title=f'{test_model} Backtest - Faktisk vs Forudsagt',
                                xaxis_title='Test Periode',
                                yaxis_title='Pris ($)',
                                template='plotly_dark',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("❌ Backtest fejlede - ingen resultater returneret")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Backtest fejl: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Metrics explanation
    with st.expander("ℹ️ Hvad betyder disse metrics?"):
        st.markdown("""
        **MAE (Mean Absolute Error)** - Gennemsnitlig absolut fejl
        - ✅ God: < 2% af prisen | ⚠️ OK: 2-5% | ❌ Dårlig: > 5%
        
        **RMSE (Root Mean Square Error)** - Straffer store fejl hårdere
        - ✅ God: < 3% af prisen | ⚠️ OK: 3-6% | ❌ Dårlig: > 6%
        
        **MAPE (Mean Absolute Percentage Error)** - Procentuel fejl
        - ✅ God: < 5% | ⚠️ OK: 5-10% | ❌ Dårlig: > 10%
        
        **R² (Coefficient of Determination)** - Hvor godt modellen forklarer variansen
        - ✅ God: > 0.7 | ⚠️ OK: 0.4-0.7 | ❌ Dårlig: < 0.4
        
        **Win Rate** - Hvor ofte retningen er korrekt (op/ned)
        - ✅ God: > 60% | ⚠️ OK: 50-60% | ❌ Dårlig: < 50%
        """)

else:
    st.info("👆 Klik på 'Start Forecast' knappen for at træne modeller")
    
    st.markdown("### 🤖 Om ML Modellerne:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🧠 LSTM (Long Short-Term Memory)
        - Neural network til tidsserier
        - God til at fange langsigtede mønstre
        - Trænes med konfigurerbare epochs
        - Bruger sliding window approach
        """)
    
    with col2:
        st.markdown("""
        #### 🌲 Random Forest
        - Ensemble af beslutningsträer
        - Robust overfor outliers
        - Multi-horizon output
        - Baseret på tekniske indikatorer
        """)
