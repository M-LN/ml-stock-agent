import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
from agent_interactive import (
    train_and_save_rf,
    train_and_save_xgboost,
    train_and_save_lstm,
    train_and_save_prophet,
    train_and_save_rf_v2,
    train_and_save_xgboost_v2,
    train_and_save_lstm_v2,
    get_available_model_versions,
    list_saved_models,
    load_model,
    delete_model,
    deploy_model,
    undeploy_model,
    predict_with_saved_model,
    calculate_model_metrics,
    MODEL_DIR
)
from storage_manager import StorageManager, get_stock_data_cached, clear_all_caches
from model_validator import ModelValidator, display_validation_report

# Phase 4: Import drift detection and A/B testing
try:
    from model_drift_detector import ModelDriftDetector
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False

try:
    from ab_testing import ABTestingFramework
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False

st.set_page_config(page_title="Model Management", page_icon="🔧", layout="wide")

st.title("🔧 ML Model Management")
st.markdown("**Træn, gem og administrer dine egne ML modeller med custom parametre**")

# Show deployment status
all_models = list_saved_models()
deployed_models = [m for m in all_models if m.get('deployed', False)]
if deployed_models:
    st.success(f"🚀 **{len(deployed_models)} deployed model(ler)** er aktive i ML Forecast og Agent Recommendations")
    with st.expander("📋 Se deployed modeller"):
        for m in deployed_models:
            timestamp = m.get('timestamp', 'N/A')[:8] if m.get('timestamp') else 'N/A'
            st.markdown(f"- **{m['model_type'].upper()}** ({m['symbol']}) - {timestamp}")
else:
    st.info("💡 Ingen deployed modeller. Træn og deploy modeller for at bruge dem i forecasts og agent.")

st.divider()

# ==================== STORAGE & CACHE SETTINGS ====================
with st.expander("⚙️ Storage & Cache Settings"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💾 Model Storage")
        storage_type = st.selectbox(
            "Storage Backend",
            ["local", "github"],
            help="Local: Ephemeral (modeller forsvinder ved redeploy)\nGitHub: Persistent via private Gists"
        )
        
        if storage_type == "github":
            if "GITHUB_TOKEN" in st.secrets:
                st.success("✅ GitHub token configured")
            else:
                st.warning("⚠️ Add GITHUB_TOKEN to secrets for persistent storage")
                st.markdown("""
                **Setup:**
                1. Create GitHub Personal Access Token with `gist` scope
                2. Add to Streamlit secrets: `GITHUB_TOKEN = "your_token"`
                """)
        else:
            st.info("📁 Using local storage (ephemeral)")
    
    with col2:
        st.markdown("### 🚀 Cache Status")
        st.markdown("Caching reduces API calls to Yahoo Finance and NewsAPI")
        
        if st.button("🗑️ Clear All Caches", help="Clear cached stock data and news"):
            clear_all_caches()
        
        st.markdown("""
        **Cache TTL:**
        - Stock data: 1 hour
        - Stock info: 30 minutes
        - News: 1 hour
        """)
    
    with col3:
        st.markdown("### 📊 Storage Info")
        model_count = len(all_models)
        st.metric("Saved Models", model_count)
        st.metric("Deployed Models", len(deployed_models))
        
        if storage_type == "local":
            st.caption("⚠️ Models will be lost on redeploy")
        else:
            st.caption("✅ Models persist across deploys")

# Initialize storage manager
storage_manager = StorageManager(storage_type=storage_type)

st.divider()

# Tabs for different functions
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏋️ Træn Nye Modeller", 
    "📂 Gemte Modeller", 
    "🔮 Brug Gemt Model", 
    "🧠 ML Mentor", 
    "🎛️ Grid Search",
    "🔍 Drift Detection",  # NEW
    "🆚 A/B Testing"        # NEW
])

# ==================== TAB 1: TRAIN NEW MODELS ====================
with tab1:
    st.subheader("🏋️ Træn en ny model")
    st.markdown("Vælg parametre og træn en model som du kan gemme og genbruge")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Data selection
        st.markdown("### 📊 Data")
        train_symbol = st.text_input("Aktie Symbol", value="AAPL", key="train_symbol")
        train_period = st.selectbox("Data Periode", ["6mo", "1y", "2y", "5y"], index=1, key="train_period")
        
        # Model type
        st.markdown("### 🤖 Model Type")
        model_type = st.radio("Vælg model", ["Random Forest", "XGBoost", "LSTM", "Prophet"], key="model_type")
        
        # Model version selection for v2 models
        st.markdown("### 📦 Model Version")
        if model_type in ["Random Forest", "XGBoost", "LSTM"]:
            model_versions = get_available_model_versions()
            
            # Map UI model names to version keys
            model_key_mapping = {
                "Random Forest": "rf",
                "XGBoost": "xgboost",
                "LSTM": "lstm"
            }
            model_key = model_key_mapping.get(model_type, model_type.lower().replace(" ", ""))
            
            version_options = []
            for version, info in model_versions.get(model_key, {}).items():
                label = f"{info['name']}"
                if info.get('recommended'):
                    label += " ⭐"
                version_options.append((version, label, info))
            
            # Only show selectbox if we have version options
            if version_options:
                selected_version = st.selectbox(
                    "Version",
                    options=version_options,
                    format_func=lambda x: x[1],
                    index=min(1, len(version_options) - 1),  # Default to v2 if available
                    help="v2 models use 67 technical indicators for better accuracy",
                    key="model_version"
                )
                
                version_code = selected_version[0]
                version_info = selected_version[2]
            else:
                # Fallback if no versions found
                version_code = "v1"
                version_info = None
                st.warning("⚠️ No versions found for this model type")
            
            # Show version description if available
            if version_info:
                st.info(f"ℹ️ {version_info['description']}")
        else:
            version_code = "v1"
            version_info = None
        
        # Common parameters
        st.markdown("### ⚙️ Parametre")
        if model_type != "Prophet":  # Prophet doesn't use window
            window = st.slider("Window Size", min_value=10, max_value=100, value=30, 
                              help="Antal dage brugt til features", key="window")
        else:
            window = 30  # Default for Prophet
        
        horizon = st.selectbox("Forecast Horizon", [1, 5, 22], index=0, 
                              help="Antal dage frem at forudsige", key="horizon")
        
        # Model-specific parameters
        if model_type == "Random Forest":
            st.markdown("### 🌳 Random Forest Parametre")
            n_estimators = st.slider("Antal Træer (n_estimators)", 
                                     min_value=10, max_value=500, value=100, step=10,
                                     help="Flere træer = bedre men langsommere")
            max_depth = st.slider("Max Dybde (max_depth)", 
                                 min_value=3, max_value=30, value=10,
                                 help="Dybde af hvert træ")
        
        elif model_type == "XGBoost":
            st.markdown("### ⚡ XGBoost Parametre")
            n_estimators = st.slider("Antal Boosts (n_estimators)", 
                                     min_value=10, max_value=500, value=100, step=10)
            max_depth = st.slider("Max Dybde (max_depth)", 
                                 min_value=3, max_value=15, value=6)
            learning_rate = st.slider("Learning Rate", 
                                     min_value=0.01, max_value=0.3, value=0.1, step=0.01,
                                     help="Lavere = mere præcis men langsommere")
        
        elif model_type == "LSTM":
            st.markdown("### 🧠 LSTM Parametre")
            epochs = st.slider("Epochs", 
                              min_value=10, max_value=200, value=50, step=10,
                              help="Antal træningsrunder")
            st.info("LSTM bruger 2 lag med 50 enheder hver")
        
        elif model_type == "Prophet":
            st.markdown("### 📈 Prophet Parametre")
            daily_seasonality = st.checkbox("Daily Seasonality", value=True,
                                           help="Fang daglige mønstre")
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True,
                                            help="Fang ugentlige mønstre")
            st.info("Prophet er god til trends og seasonality")
        
        # Initialize session state
        if 'training_state' not in st.session_state:
            st.session_state.training_state = 'idle'  # idle, validated, training, completed
        if 'validation_data' not in st.session_state:
            st.session_state.validation_data = None
        
        # Show current state for debugging
        st.caption(f"Current state: {st.session_state.training_state}")
        
        # Train button - only show if idle or completed
        if st.session_state.training_state in ['idle', 'completed']:
            train_button = st.button("🚀 Analyze & Prepare Training", type="primary", use_container_width=True)
        else:
            train_button = False
            st.info(f"State: {st.session_state.training_state}")
    
    with col2:
        # Handle training state machine
        if train_button and st.session_state.training_state in ['idle', 'completed']:
            st.info("🔄 Starting analysis...")
            
            # Reset validation data when starting fresh
            st.session_state.validation_data = None
            
            # Step 1: Fetch data and validate
            with st.spinner(f"📥 Henter data for {train_symbol}..."):
                try:
                    data = yf.download(train_symbol, period=train_period, progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    if data.empty:
                        st.error(f"❌ Kunne ikke hente data for {train_symbol}")
                        st.stop()
                    
                    st.success(f"✅ Data hentet: {len(data)} dage")
                    
                except Exception as e:
                    st.error(f"❌ Fejl: {str(e)}")
                    st.stop()
            
            # Validate data and parameters
            st.markdown("---")
            validator = ModelValidator(data, model_type, train_symbol)
            
            # Collect current parameters
            current_params = {'window': window, 'horizon': horizon}
            if model_type == "Random Forest":
                current_params.update({'n_estimators': n_estimators, 'max_depth': max_depth})
            elif model_type == "XGBoost":
                current_params.update({'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate})
            elif model_type == "LSTM":
                current_params.update({'sequence_length': window, 'lstm_units': 50, 'epochs': epochs, 'batch_size': 32})
            
            # Generate validation report
            validation_report = validator.generate_training_report(current_params)
            
            # Store in session state
            st.session_state.validation_data = {
                'data': data,
                'report': validation_report,
                'params': current_params,
                'model_type': model_type,
                'symbol': train_symbol,
                'period': train_period,
                'version_code': version_code
            }
            st.session_state.training_state = 'validated'
            st.rerun()
        
        elif train_button:
            # Button clicked but wrong state
            st.error(f"⚠️ Cannot analyze - current state is '{st.session_state.training_state}'. Expected 'idle' or 'completed'.")
            st.info("Click the '🔄 Reset' button or '🔄 Train Another Model' button to reset the workflow.")
        
        # Display validation report if in validated state
        if st.session_state.training_state == 'validated' and st.session_state.validation_data:
            val_data = st.session_state.validation_data
            
            st.success("✅ Data analyzed and validated!")
            st.markdown("---")
            st.markdown("### 📋 Validation Report")
            display_validation_report(val_data['report'], val_data['params'], val_data['model_type'])
            
            # Check if there are critical issues
            if len(val_data['report']['data_quality']['issues']) > 0:
                st.warning("⚠️ Critical issues detected. Training may not produce reliable results.")
                proceed = st.checkbox("I understand the risks and want to proceed anyway")
                if not proceed:
                    st.info("👆 Check the box above to continue, or adjust parameters and click 'Analyze & Prepare Training' again")
                    if st.button("🔄 Reset and Start Over"):
                        st.session_state.training_state = 'idle'
                        st.session_state.validation_data = None
                        st.rerun()
                    st.stop()
            
            st.markdown("---")
            
            # Confirmation button after seeing validation
            col_confirm1, col_confirm2 = st.columns([2, 1])
            with col_confirm1:
                confirm_button = st.button("✅ Confirm and Start Training", type="primary", use_container_width=True)
            with col_confirm2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.training_state = 'idle'
                    st.session_state.validation_data = None
                    st.rerun()
            
            if not confirm_button:
                st.info("👆 Review the validation report above, then click to start training")
                st.stop()
            
            # User confirmed - start training
            st.session_state.training_state = 'training'
            st.rerun()
        
        # Training state - actually train the model
        if st.session_state.training_state == 'training' and st.session_state.validation_data:
            val_data = st.session_state.validation_data
            data = val_data['data']
            model_type = val_data['model_type']
            train_symbol = val_data['symbol']
            
            st.markdown("---")
            st.markdown("### 🏋️ Training Model")
            
            # Train model
            with st.spinner(f"🏋️ Træner {model_type} model... (Est. {int(val_data['report']['estimated_time'])}s)"):
                try:
                    # Get version code from session state
                    version = val_data.get('version_code', 'v1')
                    
                    if model_type == "Random Forest":
                        if version == 'v2':
                            result = train_and_save_rf_v2(
                                data, train_symbol,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                window=window,
                                horizon=horizon,
                                use_features=True
                            )
                        else:
                            result = train_and_save_rf(
                                data, train_symbol,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                window=window,
                                horizon=horizon
                            )
                    elif model_type == "XGBoost":
                        if version == 'v2':
                            result = train_and_save_xgboost_v2(
                                data, train_symbol,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                window=window,
                                horizon=horizon,
                                use_features=True
                            )
                        else:
                            result = train_and_save_xgboost(
                                data, train_symbol,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                window=window,
                                horizon=horizon
                            )
                    elif model_type == "LSTM":
                        if version == 'v2':
                            result = train_and_save_lstm_v2(
                                data, train_symbol,
                                sequence_length=window,
                                lstm_units=[32, 16],
                                epochs=epochs,
                                use_attention=True,
                                n_features=20
                            )
                        else:
                            result = train_and_save_lstm(
                                data, train_symbol,
                                window=window,
                                epochs=epochs,
                                horizon=horizon
                            )
                    elif model_type == "Prophet":
                        from agent_interactive import train_and_save_prophet
                        result = train_and_save_prophet(
                            data, train_symbol,
                            horizon=horizon,
                            daily_seasonality=daily_seasonality,
                            weekly_seasonality=weekly_seasonality
                        )
                    
                    # Store result and move to completed state
                    st.session_state.validation_data['result'] = result
                    st.session_state.training_state = 'completed'
                    st.success("✅ Training completed! Showing results...")
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Training fejl: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.training_state = 'idle'
                    st.session_state.validation_data = None
        
        # Completed state - show results
        if st.session_state.training_state == 'completed' and st.session_state.validation_data:
            val_data = st.session_state.validation_data
            result = val_data.get('result')
            
            # Handle both string (model_id) and dict (full result) formats
            if result:
                st.success("� Model trænet og gemt!")
                
                # If result is a string (model_id from v2 models)
                if isinstance(result, str):
                    model_id = result
                    st.markdown("### ✅ Model Training Complete")
                    st.success(f"💾 **Model saved successfully!**")
                    st.code(f"models/{model_id}.pkl", language=None)
                    
                    # Try to load the model to get metadata
                    try:
                        from agent_interactive import load_model
                        model_path = f"models/{model_id}.pkl"
                        model_package = load_model(model_path)
                        
                        if model_package and 'metadata' in model_package:
                            metadata = model_package['metadata']
                            
                            st.markdown("### 📊 Training Resultater")
                            
                            # Main metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                train_mae = metadata.get('train_mae', 0)
                                st.metric("Training MAE", f"${train_mae:.2f}")
                            with col2:
                                val_mae = metadata.get('val_mae', 0)
                                st.metric("Validation MAE", f"${val_mae:.2f}", 
                                         delta=f"{((val_mae - train_mae) / train_mae * 100):.1f}%" if train_mae > 0 else None,
                                         delta_color="inverse")
                            with col3:
                                train_rmse = metadata.get('train_rmse', 0)
                                st.metric("Training RMSE", f"${train_rmse:.2f}")
                            with col4:
                                val_rmse = metadata.get('val_rmse', 0)
                                st.metric("Validation RMSE", f"${val_rmse:.2f}",
                                         delta=f"{((val_rmse - train_rmse) / train_rmse * 100):.1f}%" if train_rmse > 0 else None,
                                         delta_color="inverse")
                            
                            # Additional info
                            col5, col6 = st.columns(2)
                            with col5:
                                train_samples = metadata.get('training_samples', 0)
                                st.metric("Training Samples", train_samples)
                            with col6:
                                val_samples = metadata.get('validation_samples', 0)
                                st.metric("Validation Samples", val_samples)
                            
                            # Model version info
                            model_version = metadata.get('model_version', 'v1')
                            if 'v2' in str(model_version):
                                st.info("✨ This is a **v2 Enhanced Model** with 67 technical indicators")
                            
                            # Show detailed metadata
                            with st.expander("📋 Full Model Metadata"):
                                st.json(metadata)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load model metadata: {str(e)}")
                
                # If result is a dict (from v1 models)
                elif isinstance(result, dict):
                    st.markdown("### 📊 Training Resultater")
                    
                    # Safely get metadata
                    metadata = result.get('metadata', {})
                    
                    if metadata:
                        # Main metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            train_mae = metadata.get('train_mae', 0)
                            st.metric("Training MAE", f"${train_mae:.2f}")
                        with col2:
                            val_mae = metadata.get('val_mae', 0)
                            st.metric("Validation MAE", f"${val_mae:.2f}", 
                                     delta=f"{((val_mae - train_mae) / train_mae * 100):.1f}%" if train_mae > 0 else None,
                                     delta_color="inverse")
                        with col3:
                            train_rmse = metadata.get('train_rmse', 0)
                            st.metric("Training RMSE", f"${train_rmse:.2f}")
                        with col4:
                            val_rmse = metadata.get('val_rmse', 0)
                            st.metric("Validation RMSE", f"${val_rmse:.2f}",
                                     delta=f"{((val_rmse - train_rmse) / train_rmse * 100):.1f}%" if train_rmse > 0 else None,
                                     delta_color="inverse")
                        
                        # Additional info
                        col5, col6 = st.columns(2)
                        with col5:
                            train_samples = metadata.get('training_samples', 0)
                            st.metric("Training Samples", train_samples)
                        with col6:
                            val_samples = metadata.get('validation_samples', 0)
                            st.metric("Validation Samples", val_samples)
                        
                        # Model version info
                        model_version = metadata.get('model_version', 'v1')
                        if 'v2' in str(model_version):
                            st.info("✨ This is a **v2 Enhanced Model** with 67 technical indicators")
                        
                        # Show detailed metadata
                        with st.expander("📋 Full Model Metadata"):
                            st.json(metadata)
                    else:
                        st.warning("⚠️ Metadata not available")
                    
                    # Show filepath if available
                    filepath = result.get('filepath')
                    if filepath:
                        st.success(f"💾 **Model saved successfully!**")
                        st.code(filepath, language=None)
                
                else:
                    st.error("⚠️ Training completed but result is in unexpected format")
                    st.write(f"Result type: {type(result)}")
                    st.write(f"Result: {result}")
                
                # Action buttons (for all result types)
                st.markdown("---")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔄 Train Another Model", type="primary", use_container_width=True):
                        st.session_state.training_state = 'idle'
                        st.session_state.validation_data = None
                        st.rerun()
                with col_btn2:
                    if st.button("📊 View All Models", use_container_width=True):
                        st.session_state.training_state = 'idle'
                        st.session_state.validation_data = None
                        # User can manually switch to other tabs
                        st.info("👉 Switch to 'Gem Modeller' tab to see all models")
            else:
                # No result available
                st.warning("⚠️ Training completed but no result available")
                if st.button("🔄 Reset"):
                    st.session_state.training_state = 'idle'
                    st.session_state.validation_data = None
                    st.rerun()
        
        # Show instructions only when idle
        if st.session_state.training_state == 'idle':
            st.info("""
            ### 📚 Sådan træner du en model:
            
            1. **Vælg data**: Aktie symbol og periode
            2. **Vælg model type**: Random Forest eller XGBoost
            3. **Juster parametre**: 
               - **Window Size**: Hvor mange dage historik modellen bruger
               - **Horizon**: Hvor langt frem den forudsiger
               - **Model parametre**: Juster efter behov
            4. **Klik "Træn Model"**: Modellen trænes og gemmes automatisk
            
            ### 💡 Tips til parametre:
            
            **Random Forest:**
            - Flere træer = bedre performance men langsommere
            - Højere max_depth = mere kompleks model
            - Start med 100 træer, depth 10
            
            **XGBoost:**
            - Typisk hurtigere end Random Forest
            - Learning rate: 0.1 er et godt udgangspunkt
            - Lavere learning rate = mere præcis men kræver flere estimators
            
            **Window & Horizon:**
            - Window: 30 dage er standard
            - Horizon: 1 dag for daglig trading, 5-22 for længere
            """)

# ==================== TAB 2: SAVED MODELS ====================
with tab2:
    st.subheader("📂 Gemte Modeller")
    
    # ========== BULK CLEANUP SECTION ==========
    with st.expander("🗑️ **Bulk Cleanup** - Slet flere modeller på én gang"):
        st.markdown("**⚠️ ADVARSEL:** Denne funktion sletter modeller permanent!")
        
        cleanup_col1, cleanup_col2 = st.columns([2, 1])
        
        with cleanup_col1:
            # Get all models with metrics
            import json
            from collections import defaultdict
            
            all_models_cleanup = list_saved_models()
            models_with_metrics = []
            
            for model_info in all_models_cleanup:
                log_file = model_info['filepath'].replace('.pkl', '_training.json').replace('.h5', '_training.json')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                        model_info['val_mae'] = log_data.get('metrics', {}).get('val_mae', 0)
                        model_info['version'] = log_data.get('version', 1)
                        model_info['description'] = log_data.get('description', '')
                else:
                    model_info['val_mae'] = 0
                    model_info['version'] = 1
                    model_info['description'] = ''
                models_with_metrics.append(model_info)
            
            # Cleanup filters
            st.markdown("**📋 Vælg hvilke modeller der skal slettes:**")
            
            cleanup_criteria = st.radio(
                "Cleanup kriterie",
                ["🎯 Manuel valg", "📊 Performance filter", "🔢 Version filter"],
                horizontal=True
            )
            
            models_to_delete = []
            
            if cleanup_criteria == "🎯 Manuel valg":
                st.markdown("**Vælg modeller til sletning:**")
                
                # Group by symbol for better organization
                grouped_cleanup = defaultdict(list)
                for m in models_with_metrics:
                    grouped_cleanup[m['symbol']].append(m)
                
                for symbol in sorted(grouped_cleanup.keys()):
                    st.markdown(f"**{symbol}:**")
                    for model in grouped_cleanup[symbol]:
                        label = f"{model['model_type'].upper()} v{model['version']} - Val MAE: ${model['val_mae']:.2f}"
                        if model.get('description'):
                            label += f" - {model['description'][:50]}"
                        
                        if st.checkbox(label, key=f"cleanup_check_{model['filename']}"):
                            models_to_delete.append(model)
            
            elif cleanup_criteria == "📊 Performance filter":
                st.markdown("**Slet modeller med dårlig performance:**")
                
                mae_threshold = st.slider(
                    "Val MAE > threshold (højere = værre)",
                    min_value=0.0,
                    max_value=50.0,
                    value=20.0,
                    step=1.0,
                    help="Modeller med Val MAE højere end denne værdi vil blive slettet"
                )
                
                # Find models above threshold
                for model in models_with_metrics:
                    if model['val_mae'] > mae_threshold:
                        models_to_delete.append(model)
                
                if models_to_delete:
                    st.warning(f"⚠️ **{len(models_to_delete)} modeller** har Val MAE > ${mae_threshold:.2f}:")
                    for m in models_to_delete[:10]:  # Show first 10
                        st.markdown(f"- {m['model_type'].upper()} {m['symbol']} v{m['version']}: ${m['val_mae']:.2f}")
                    if len(models_to_delete) > 10:
                        st.markdown(f"... og {len(models_to_delete) - 10} flere")
                else:
                    st.info("✅ Ingen modeller over threshold")
            
            elif cleanup_criteria == "🔢 Version filter":
                st.markdown("**Slet gamle versioner:**")
                
                keep_latest = st.number_input(
                    "Behold seneste X versioner pr. symbol+type",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="F.eks. ved at vælge 3, beholdes kun v3, v4, v5 og ældre versioner slettes"
                )
                
                # Group by symbol + type
                grouped_versions = defaultdict(list)
                for model in models_with_metrics:
                    key = f"{model['symbol']}_{model['model_type']}"
                    grouped_versions[key].append(model)
                
                # Find models to delete
                for key, models in grouped_versions.items():
                    # Sort by version (newest first)
                    models_sorted = sorted(models, key=lambda m: m['version'], reverse=True)
                    
                    # Mark old versions for deletion
                    if len(models_sorted) > keep_latest:
                        models_to_delete.extend(models_sorted[keep_latest:])
                
                if models_to_delete:
                    st.warning(f"⚠️ **{len(models_to_delete)} gamle versioner** vil blive slettet:")
                    for m in models_to_delete[:10]:  # Show first 10
                        st.markdown(f"- {m['model_type'].upper()} {m['symbol']} v{m['version']}")
                    if len(models_to_delete) > 10:
                        st.markdown(f"... og {len(models_to_delete) - 10} flere")
                else:
                    st.info("✅ Alle modeller er inden for grænsen")
        
        with cleanup_col2:
            st.markdown("### 📊 Cleanup Stats")
            
            if models_to_delete:
                st.error(f"**{len(models_to_delete)}** modeller")
                st.markdown("vil blive slettet")
                
                # Show breakdown by type
                type_counts = defaultdict(int)
                for m in models_to_delete:
                    type_counts[m['model_type']] += 1
                
                for mtype, count in sorted(type_counts.items()):
                    st.markdown(f"- {mtype.upper()}: {count}")
                
                st.markdown("---")
                
                # Confirmation
                confirm = st.checkbox("✅ Ja, slet disse modeller", key="cleanup_confirm")
                
                if confirm:
                    if st.button("🗑️ **SLET NU**", type="primary", use_container_width=True):
                        deleted_count = 0
                        failed_count = 0
                        
                        with st.spinner("Sletter modeller..."):
                            for model in models_to_delete:
                                if delete_model(model['filepath']):
                                    deleted_count += 1
                                else:
                                    failed_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"✅ Slettede {deleted_count} modeller!")
                        if failed_count > 0:
                            st.error(f"❌ Kunne ikke slette {failed_count} modeller")
                        
                        st.rerun()
                else:
                    st.info("👆 Bekræft først")
            else:
                st.info("📭 Ingen modeller valgt")
    
    st.divider()
    
    # View toggle
    view_col1, view_col2, view_col3 = st.columns([1, 1, 2])
    with view_col1:
        view_mode = st.radio("Visning", ["🗂️ Grupperet", "📋 Liste"], horizontal=True, key="view_mode")
    with view_col2:
        if view_mode == "📋 Liste":
            filter_type = st.selectbox("Filtrer type", ["Alle", "rf", "xgboost", "lstm"], key="filter_type")
    with view_col3:
        st.write("")  # Spacer
    
    # Get all saved models
    all_saved_models = list_saved_models()
    
    if not all_saved_models:
        st.info("📭 Ingen gemte modeller fundet. Træn en model i 'Træn Nye Modeller' tab.")
    else:
        st.success(f"📦 Total: {len(all_saved_models)} model(ler)")
        
        # ========== GRUPPERET VISNING ==========
        if view_mode == "🗂️ Grupperet":
            # Group models by symbol and type
            from collections import defaultdict
            import json
            
            grouped = defaultdict(lambda: defaultdict(list))
            
            for model_info in all_saved_models:
                symbol = model_info['symbol']
                model_type = model_info['model_type']
                
                # Load training log to get metrics
                log_file = model_info['filepath'].replace('.pkl', '_training.json').replace('.h5', '_training.json')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                        model_info['metrics'] = log_data.get('metrics', {})
                        model_info['version'] = log_data.get('version', 1)
                        model_info['description'] = log_data.get('description', '')
                else:
                    model_info['metrics'] = {}
                    model_info['version'] = 1
                    model_info['description'] = ''
                
                grouped[symbol][model_type].append(model_info)
            
            # Display grouped models
            for symbol in sorted(grouped.keys()):
                st.markdown(f"### 📊 {symbol}")
                
                for model_type in sorted(grouped[symbol].keys()):
                    models = grouped[symbol][model_type]
                    model_count = len(models)
                    
                    # Find best model (lowest val_mae)
                    best_model = min(models, key=lambda m: m.get('metrics', {}).get('val_mae', float('inf')))
                    best_mae = best_model.get('metrics', {}).get('val_mae', 0)
                    
                    with st.expander(f"🤖 **{model_type.upper()}** ({model_count} versions) - Best: ${best_mae:.2f} ⭐"):
                        # Sort by version (newest first)
                        models_sorted = sorted(models, key=lambda m: m['version'], reverse=True)
                        
                        # Display each version
                        for model_info in models_sorted:
                            is_best = model_info == best_model
                            version = model_info['version']
                            description = model_info['description']
                            val_mae = model_info.get('metrics', {}).get('val_mae', 0)
                            
                            st.markdown("---")
                            
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                badge = "⭐ **BEST**" if is_best else ""
                                st.markdown(f"**v{version}** {badge}")
                                if description:
                                    st.caption(description)
                                st.markdown(f"Val MAE: **${val_mae:.2f}**")
                                timestamp = model_info.get('timestamp', 'N/A')
                                st.caption(f"📅 {timestamp}")
                            
                            with col_b:
                                # Deployment
                                is_deployed = model_info.get('deployed', False)
                                if is_deployed:
                                    st.success("✅ DEPLOYED")
                                    if st.button("⏸️ Undeploy", key=f"undeploy_grp_{model_info['filename']}", use_container_width=True):
                                        from agent_interactive import undeploy_model
                                        if undeploy_model(model_info['filepath']):
                                            st.success("✅ Undeployed!")
                                            st.rerun()
                                else:
                                    if st.button("🚀 Deploy", key=f"deploy_grp_{model_info['filename']}", use_container_width=True):
                                        from agent_interactive import deploy_model
                                        if deploy_model(model_info['filepath']):
                                            st.success("✅ Deployed!")
                                            st.rerun()
                                
                                # Delete button
                                if st.button("🗑️ Slet", key=f"delete_grp_{model_info['filename']}", use_container_width=True):
                                    if delete_model(model_info['filepath']):
                                        st.success("✅ Slettet!")
                                        st.rerun()
                
                st.markdown("")  # Spacer between symbols
        
        # ========== LISTE VISNING ==========
        else:
            # Filter models
            model_type_filter = None if filter_type == "Alle" else filter_type
            saved_models = list_saved_models(model_type=model_type_filter)
            
            if not saved_models:
                st.info("� Ingen modeller matcher filteret.")
            else:
                st.markdown(f"**{len(saved_models)} model(ler)** matcher filter")
                
                # Display models in cards
                for i, model_info in enumerate(saved_models):
                    timestamp = model_info.get('timestamp', 'N/A')
                    with st.expander(f"🤖 {model_info['model_type'].upper()} - {model_info['symbol']} ({timestamp})"):
                        col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Model Type:** {model_info['model_type'].upper()}")
                    st.markdown(f"**Symbol:** {model_info['symbol']}")
                    st.markdown(f"**Timestamp:** {timestamp}")
                    
                    if 'metadata' in model_info:
                        meta = model_info['metadata']
                        
                        # Show model version if available
                        model_version = meta.get('model_version', 'v1_legacy')
                        if 'v2' in model_version:
                            st.markdown(f"**Version:** 🆕 v2 (Enhanced with 67 features)")
                        else:
                            st.markdown(f"**Version:** v1 (Legacy)")
                        
                        st.markdown(f"**Window:** {meta.get('window', 'N/A')}")
                        st.markdown(f"**Horizon:** {meta.get('horizon', 'N/A')}")
                        st.markdown(f"**Training MAE:** ${meta.get('train_mae', 'N/A'):.2f}")
                        st.markdown(f"**Training RMSE:** ${meta.get('train_rmse', 'N/A'):.2f}")
                        
                        # Show MAPE and Directional Accuracy if available (v2 models)
                        if 'test_mape' in meta:
                            st.markdown(f"**Test MAPE:** {meta['test_mape']:.2f}%")
                        if 'test_direction_acc' in meta:
                            st.markdown(f"**Directional Accuracy:** {meta['test_direction_acc']:.1f}%")
                        
                        with st.expander("📋 Fuld Metadata"):
                            st.json(meta)
                
                with col_b:
                    # Deployment status
                    is_deployed = model_info.get('deployed', False)
                    if is_deployed:
                        st.success("✅ DEPLOYED")
                    else:
                        st.info("⏸️ Not Deployed")
                    
                    # Action buttons
                    if is_deployed:
                        if st.button(f"⏸️ Undeploy", key=f"undeploy_{i}", use_container_width=True):
                            from agent_interactive import undeploy_model
                            if undeploy_model(model_info['filepath']):
                                st.success("✅ Model undeployed!")
                                st.rerun()
                    else:
                        if st.button(f"🚀 Deploy", key=f"deploy_{i}", use_container_width=True):
                            from agent_interactive import deploy_model
                            if deploy_model(model_info['filepath']):
                                st.success("✅ Model deployed!")
                                st.rerun()
                    
                    if st.button(f"🗑️ Slet", key=f"delete_{i}", use_container_width=True):
                        if delete_model(model_info['filepath']):
                            st.success("✅ Model slettet!")
                            st.rerun()
                    
                    st.markdown(f"`{model_info['filename']}`")

# ==================== TAB 3: USE SAVED MODEL ====================
with tab3:
    st.subheader("🔮 Brug en gemt model til prediction")
    
    # Get all saved models
    all_models = list_saved_models()
    
    if not all_models:
        st.info("📭 Ingen gemte modeller. Træn først en model i 'Træn Nye Modeller' tab.")
    else:
        # Model selection
        model_options = [f"{m['model_type'].upper()} - {m['symbol']} ({m.get('timestamp', 'N/A')})" for m in all_models]
        selected_idx = st.selectbox("Vælg model", range(len(model_options)), 
                                    format_func=lambda x: model_options[x])
        
        selected_model_info = all_models[selected_idx]
        
        # Show model info
        with st.expander("ℹ️ Model Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", selected_model_info['model_type'].upper())
            with col2:
                st.metric("Symbol", selected_model_info['symbol'])
            with col3:
                metadata = selected_model_info.get('metadata', {})
                train_mae = metadata.get('train_mae', 0) if metadata else 0
                st.metric("Training MAE", f"${train_mae:.2f}")
        
        # Data for prediction
        st.markdown("### 📊 Prediction Data")
        use_same_symbol = st.checkbox("Brug samme symbol som træning", value=True)
        
        if use_same_symbol:
            pred_symbol = selected_model_info['symbol']
            st.info(f"Bruger symbol: {pred_symbol}")
        else:
            pred_symbol = st.text_input("Andet symbol", value="AAPL")
        
        pred_period = st.selectbox("Data periode", ["1mo", "3mo", "6mo", "1y"], index=2)
        
        # Predict button
        if st.button("🔮 Lav Prediction", type="primary", use_container_width=True):
            with st.spinner(f"📥 Henter data for {pred_symbol}..."):
                try:
                    data = yf.download(pred_symbol, period=pred_period, progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    if data.empty:
                        st.error(f"❌ Kunne ikke hente data")
                        st.stop()
                    
                except Exception as e:
                    st.error(f"❌ Fejl: {str(e)}")
                    st.stop()
            
            # Load and use model
            with st.spinner("🔮 Laver prediction..."):
                try:
                    model_package = load_model(selected_model_info['filepath'])
                    
                    if model_package:
                        metadata = selected_model_info.get('metadata', {})
                        horizon = metadata.get('horizon', 1) if metadata else 1
                        
                        result = predict_with_saved_model(
                            model_package, 
                            data, 
                            horizon=horizon
                        )
                        
                        if result:
                            st.success("✅ Prediction komplet!")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"${result['current']:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${result['forecast']:.2f}",
                                         delta=f"{result['change_pct']:.2f}%")
                            with col3:
                                direction = "📈 UP" if result['change_pct'] > 0 else "📉 DOWN"
                                st.metric("Direction", direction)
                            
                            # Chart
                            st.markdown("### 📈 Price Chart")
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='lightblue', width=2)
                            ))
                            
                            # Add prediction point
                            from datetime import timedelta
                            horizon_days = selected_model_info['metadata'].get('horizon', 1)
                            pred_date = data.index[-1] + timedelta(days=horizon_days)
                            
                            fig.add_trace(go.Scatter(
                                x=[pred_date],
                                y=[result['forecast']],
                                mode='markers',
                                name='Prediction',
                                marker=dict(color='red', size=15, symbol='star')
                            ))
                            
                            fig.update_layout(
                                title=f"{pred_symbol} - Prediction med gemt model",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                template="plotly_dark",
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Fejl ved prediction: {str(e)}")

# ==================== TAB 4: ML MENTOR ====================
with tab4:
    st.subheader("🧠 ML Mentor - Intelligent Model Analysis")
    st.markdown("**AI-drevet analyse af dine trænede modeller med actionable anbefalinger**")
    
    try:
        from ml_mentor_engine import analyze_saved_model, MLMentorEngine, calculate_health_score
        from ml_mentor_retrain import apply_recommendation_and_retrain
        
        if len(all_models) == 0:
            st.info("📭 Ingen gemte modeller. Træn en model først i 'Træn Nye Modeller' tab.")
        else:
            # Model selector
            model_options = [f"{m.get('timestamp', 'N/A')} - {m['model_type']} ({m['symbol']})" for m in all_models]
            selected_model_str = st.selectbox("📂 Vælg model til analyse", model_options)
            selected_model_id = selected_model_str.split(" - ")[0]
            
            # API Key config (optional for LLM analysis)
            with st.expander("⚙️ API Configuration (Optional - For LLM Analysis)"):
                # Try to get from secrets first
                default_api_key = ""
                if "OPENAI_API_KEY" in st.secrets:
                    default_api_key = st.secrets["OPENAI_API_KEY"]
                    st.success("✅ OpenAI API Key loaded from secrets")
                
                api_key = st.text_input(
                    "OpenAI API Key", 
                    value=default_api_key,
                    type="password", 
                    help="Only needed for LLM-powered analysis. Can be set in Streamlit secrets as OPENAI_API_KEY"
                )
                llm_model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4"])
            
            # Initialize session state for retrain results
            if 'retrain_results' not in st.session_state:
                st.session_state.retrain_results = {}
            
            # Analyze button
            if st.button("🔍 Analyze Model", type="primary"):
                with st.spinner("Analyserer model..."):
                    result = analyze_saved_model(selected_model_id, api_key if api_key else None, llm_model)
                    st.session_state.analysis_result = result  # Store in session state
                    
            # Display analysis result (from session state or fresh analysis)
            result = st.session_state.get('analysis_result')
            if result and result.get("success"):
                        # Health Score
                        col1, col2, col3 = st.columns([2,1,1])
                        with col1:
                            st.metric("🏥 Model Health Score", f"{result['health_score']:.1f}/100", 
                                     result.get("health_status", ""))
                        with col2:
                            st.metric("📊 Recommendations", result.get("recommendation_count", 0))
                        with col3:
                            st.metric("� High Priority", result.get("high_priority_count", 0))
                        
                        st.divider()
                        
                        # Metrics
                        st.markdown("### 📈 Performance Metrics")
                        metrics = result.get("metrics", {})
                        col1, col2, col3 = st.columns(3)
                        
                        # Handle None values in metrics
                        mae_val = metrics.get('mae')
                        rmse_val = metrics.get('rmse')
                        r2_val = metrics.get('r2')
                        
                        col1.metric("MAE", f"{mae_val:.2f}" if mae_val is not None else "N/A")
                        col2.metric("RMSE", f"{rmse_val:.2f}" if rmse_val is not None else "N/A")
                        col3.metric("R² Score", f"{r2_val:.3f}" if r2_val is not None else "N/A")
                        
                        st.divider()
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        recommendations = result.get("recommendations", [])
                        
                        if len(recommendations) == 0:
                            st.success("✅ No issues found! Model is performing well.")
                        else:
                            for i, rec in enumerate(recommendations):
                                priority = rec.get("priority", "MEDIUM")
                                emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
                                
                                with st.expander(f"{emoji} {rec.get('category', 'General')}: {rec.get('issue', 'N/A')}"):
                                    st.markdown(f"**Recommendation:** {rec.get('recommendation', 'N/A')}")
                                    st.markdown(f"**Expected Improvement:** {rec.get('expected_improvement', 'N/A')}")
                                    
                                    # Auto-retrain option
                                    retrain_key = f"retrain_{selected_model_id}_{i}"
                                    
                                    if st.button(f"🔄 Apply & Retrain", key=retrain_key):
                                        with st.spinner("Retraining model..."):
                                            # Find model filepath and load
                                            import os
                                            filepath = None
                                            if os.path.exists(MODEL_DIR):
                                                for filename in os.listdir(MODEL_DIR):
                                                    if filename.endswith('.pkl') and selected_model_id in filename:
                                                        filepath = os.path.join(MODEL_DIR, filename)
                                                        break
                                            
                                            if filepath:
                                                model_package = load_model(filepath)
                                                if model_package:
                                                    symbol = model_package.get("symbol")
                                                    model_type = model_package.get("model_type")
                                                    nested_metadata = model_package.get("metadata", {})
                                                    
                                                    retrain_result = apply_recommendation_and_retrain(
                                                        rec, selected_model_id, 
                                                        symbol, 
                                                        nested_metadata,
                                                        model_type
                                                    )
                                                    
                                                    # Store result in session state
                                                    st.session_state.retrain_results[retrain_key] = retrain_result
                                                else:
                                                    st.session_state.retrain_results[retrain_key] = {
                                                        "success": False,
                                                        "error": "Failed to load model"
                                                    }
                                            else:
                                                st.session_state.retrain_results[retrain_key] = {
                                                    "success": False,
                                                    "error": "Model file not found"
                                                }
                                    
                                    # Display retrain result if exists in session state
                                    if retrain_key in st.session_state.retrain_results:
                                        retrain_result = st.session_state.retrain_results[retrain_key]
                                        
                                        if retrain_result["success"]:
                                            st.success(f"✅ Model retrained! New model ID: {retrain_result['new_model_id']}")
                                            st.markdown("**Before vs After:**")
                                            
                                            old_metrics = retrain_result.get('old_metrics', {})
                                            new_metrics = retrain_result.get('new_metrics', {})
                                            improvement = retrain_result.get('improvement', {})
                                            
                                            col1, col2 = st.columns(2)
                                            col1.metric("Old MAE", f"{old_metrics.get('mae', 0):.2f}")
                                            col2.metric("New MAE", f"{new_metrics.get('mae', 0):.2f}", 
                                                       delta=f"{improvement.get('mae', 0):.1f}%")
                                        else:
                                            st.error(f"❌ {retrain_result.get('error', 'Unknown error')}")
            elif result:
                # Analysis was run but failed
                st.error(f"❌ {result.get('error', 'Unknown error')}")
    
    except ImportError as e:
        st.error(f"❌ ML Mentor dependencies not found: {e}")

# Footer info
st.divider()
st.markdown("""
### 💡 Om Model Management

Denne side lader dig:
- **Træne** modeller med custom parametre
- **Gemme** modeller til senere brug
- **Administrere** dine gemte modeller
- **Genbruge** modeller på ny data
- **Analysere** med ML Mentor ⭐ NEW!

**Fordele ved at gemme modeller:**
- ⏱️ Spar tid - træn én gang, brug mange gange
- 🎯 Sammenlign forskellige parametersæt
- 📊 Track performance over tid
- 🔄 Brug samme model på flere aktier
- 🧠 Få AI-drevne anbefalinger

**Modellerne gemmes i:** `{MODEL_DIR}/`
""")

# ==================== TAB 5: GRID SEARCH ====================
with tab5:
    st.subheader("🎛️ Grid Search - Automated Hyperparameter Tuning")
    st.markdown("**Automatisk find de bedste hyperparametre for dine modeller**")
    
    try:
        from grid_search_engine import GridSearchEngine, get_previous_grid_searches, SEARCH_SPACES
        
        # Previous searches
        with st.expander("📋 Previous Grid Searches"):
            previous = get_previous_grid_searches()
            if len(previous) > 0:
                st.dataframe(pd.DataFrame(previous), use_container_width=True)
            else:
                st.info("No previous searches found.")
        
        st.divider()
        
        # New search configuration
        st.markdown("### 🎯 Configure New Grid Search")
        
        col1, col2 = st.columns(2)
        with col1:
            gs_model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost", "LSTM"], key="gs_model")
        with col2:
            gs_symbol = st.text_input("Stock Symbol", value="AAPL", key="gs_symbol")
        
        # Search space selector
        search_size = st.select_slider("Search Space Size", 
                                       options=["Small (fast)", "Medium", "Large (slow)"],
                                       value="Small (fast)")
        
        # Convert display name to key
        search_size_map = {
            "Small (fast)": "small",
            "Medium": "medium",
            "Large (slow)": "large"
        }
        search_space_key = search_size_map[search_size]
        
        # Preview search space
        with st.expander("🔍 Preview Search Space"):
            if gs_model_type in SEARCH_SPACES:
                space = SEARCH_SPACES[gs_model_type].get(search_space_key, {})
                st.json(space)
                
                # Calculate combinations
                total_combos = 1
                for param_values in space.values():
                    total_combos *= len(param_values)
                st.info(f"Total parameter combinations: {total_combos}")
        
        st.divider()
        
        # Start search
        if st.button("🚀 Start Grid Search", type="primary"):
            if not gs_symbol:
                st.error("Please enter a stock symbol")
            else:
                with st.spinner(f"Running grid search for {gs_symbol}..."):
                    # Fetch data
                    import yfinance as yf
                    data = yf.download(gs_symbol, period="2y", progress=False)
                    
                    if data.empty:
                        st.error(f"No data found for {gs_symbol}")
                    else:
                        # Initialize engine with correct search_space_key
                        engine = GridSearchEngine(gs_model_type, gs_symbol, search_space_key)
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(current, total):
                            progress = current / total
                            progress_bar.progress(progress)
                            status_text.text(f"Testing combination {current}/{total}...")
                        
                        # Run search
                        try:
                            results = engine.run_search(data, progress_callback)
                            
                            if results["success"]:
                                st.success(f"✅ Grid search completed! Tested {results['total_trials']} combinations")
                                
                                # Best params and score side by side
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown("### 🏆 Best Parameters Found")
                                    st.json(results["best_params"])
                                
                                with col2:
                                    st.markdown("### 📊 Best Score")
                                    st.metric("Validation MAE", f"${results['best_score']:.2f}")
                                    if results.get('best_model_id'):
                                        st.info(f"Model saved")
                                
                                st.divider()
                                
                                # Results table
                                st.markdown("### 📊 All Results")
                                if results["all_results"] and len(results["all_results"]) > 0:
                                    results_df = pd.DataFrame(results["all_results"])
                                    
                                    # Format the dataframe for better display
                                    display_df = results_df.copy()
                                    
                                    # Convert params dict to string for display
                                    if 'params' in display_df.columns:
                                        display_df['params'] = display_df['params'].apply(lambda x: str(x))
                                    
                                    # Round numeric columns
                                    numeric_cols = ['val_mae', 'val_rmse', 'score']
                                    for col in numeric_cols:
                                        if col in display_df.columns:
                                            display_df[col] = display_df[col].round(2)
                                    
                                    # Sort by score (lower is better)
                                    display_df = display_df.sort_values("score")
                                    
                                    # Display
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Show summary stats
                                    st.markdown("#### 📈 Summary Statistics")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Best MAE", f"${results_df['val_mae'].min():.2f}")
                                    with col2:
                                        st.metric("Worst MAE", f"${results_df['val_mae'].max():.2f}")
                                    with col3:
                                        st.metric("Average MAE", f"${results_df['val_mae'].mean():.2f}")
                                else:
                                    st.warning("No successful trials completed.")
                                
                                # Train with best params button
                                if st.button("🎯 Train Model with Best Params"):
                                    st.info("Navigate to 'Træn Nye Modeller' tab and use these parameters:")
                                    st.json(results["best_params"])
                            else:
                                st.error(f"❌ {results.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"❌ Grid search failed: {str(e)}")
    
    except ImportError as e:
        st.error(f"❌ Grid Search dependencies not found: {e}")

# ==================== TAB 6: DRIFT DETECTION ====================
with tab6:
    st.subheader("🔍 Model Drift Detection")
    st.markdown("Detect when models need retraining due to data or performance drift")
    
    if not DRIFT_DETECTION_AVAILABLE:
        st.error("❌ Drift detection not available. Install dependencies: `pip install scipy`")
    else:
        # Model selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            drift_symbol = st.text_input("Stock Symbol", value="AAPL", key="drift_symbol")
            
            # Get models for this symbol
            symbol_models = [m for m in all_models if m['symbol'] == drift_symbol]
            
            if not symbol_models:
                st.warning(f"⚠️ No models found for {drift_symbol}")
            else:
                model_options = [
                    f"{m['model_type'].upper()} - {m.get('timestamp', 'N/A')[:8]}"
                    for m in symbol_models
                ]
                
                selected_idx = st.selectbox(
                    "Select Model to Check",
                    range(len(model_options)),
                    format_func=lambda x: model_options[x],
                    key="drift_model_select"
                )
                
                selected_model = symbol_models[selected_idx]
                model_id = selected_model['model_id']
        
        with col2:
            st.markdown("### ⚙️ Detection Settings")
            drift_threshold = st.slider(
                "PSI Drift Threshold",
                0.1, 0.5, 0.25,
                help="PSI > 0.25 indicates significant drift"
            )
            
            perf_threshold = st.slider(
                "Performance Degradation %",
                10, 50, 20,
                help="Alert if MAE increases by this percentage"
            )
        
        st.divider()
        
        # Run drift detection
        if st.button("🔍 Check for Drift", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing drift for {drift_symbol}..."):
                try:
                    detector = ModelDriftDetector(
                        drift_threshold=drift_threshold / 100,
                        performance_threshold=perf_threshold / 100
                    )
                    
                    results = detector.comprehensive_drift_check(
                        symbol=drift_symbol,
                        model_id=model_id
                    )
                    
                    # Display results
                    st.markdown("### 📊 Drift Detection Results")
                    
                    # Overall status
                    overall = results['overall']
                    if overall['drift_detected']:
                        st.error(f"🚨 **DRIFT DETECTED** - Priority: {overall['priority']}")
                        st.warning(f"**Recommendation:** {overall['recommendation']}")
                        
                        if overall['reasons']:
                            st.markdown("**Reasons:**")
                            for reason in overall['reasons']:
                                st.markdown(f"- {reason}")
                    else:
                        st.success("✅ **NO DRIFT DETECTED** - Model is stable")
                        st.info("Continue monitoring. No action needed.")
                    
                    st.divider()
                    
                    # Detailed results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔬 Statistical Tests")
                        
                        if 'concept_drift' in results['tests']:
                            cd = results['tests']['concept_drift']
                            
                            # KS Test
                            ks = cd['ks_test']
                            ks_status = "⚠️ DRIFT" if ks['drift_detected'] else "✅ OK"
                            st.metric(
                                "KS Test",
                                ks_status,
                                f"p-value: {ks['p_value']:.4f}",
                                delta_color="inverse"
                            )
                            
                            # PSI Test
                            psi = cd['psi_test']
                            psi_status = "⚠️ DRIFT" if psi['drift_detected'] else "✅ OK"
                            st.metric(
                                "PSI Test",
                                psi_status,
                                f"PSI: {psi['psi']:.4f}",
                                delta_color="inverse"
                            )
                    
                    with col2:
                        st.markdown("### 📈 Market Metrics")
                        
                        if 'concept_drift' in results['tests']:
                            vol = cd['volatility_change']
                            
                            st.metric(
                                "Early Volatility",
                                f"{vol['early_volatility']:.1%}"
                            )
                            
                            st.metric(
                                "Recent Volatility",
                                f"{vol['recent_volatility']:.1%}",
                                f"{vol['change_pct']:+.1f}%"
                            )
                            
                            if vol['significant_change']:
                                st.warning("⚠️ Significant volatility change detected")
                    
                    # Performance degradation (if available)
                    if 'performance' in results['tests']:
                        st.divider()
                        st.markdown("### 📊 Performance Analysis")
                        
                        perf = results['tests']['performance']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Training MAE",
                                f"${perf['train_mae']:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Recent MAE",
                                f"${perf['recent_mae']:.2f}",
                                f"{perf['degradation_pct']:+.1f}%"
                            )
                        
                        with col3:
                            if perf['degraded']:
                                st.error("⚠️ Degraded")
                            else:
                                st.success("✅ Stable")
                        
                        st.info(f"💡 {perf['interpretation']}")
                        st.markdown(f"**Recommendation:** {perf['recommendation']}")
                    
                    # Retraining recommendation
                    if detector.should_retrain(results):
                        st.divider()
                        st.error("### 🔄 RETRAINING RECOMMENDED")
                        st.markdown("""
                        The model shows signs of drift and should be retrained:
                        1. Go to **Træn Nye Modeller** tab
                        2. Train a new model with recent data
                        3. Compare with this model using A/B Testing
                        4. Deploy the better performing model
                        """)
                        
                        if st.button("🚀 Go to Training", key="goto_training"):
                            st.info("Navigate to 'Træn Nye Modeller' tab above")
                
                except Exception as e:
                    st.error(f"❌ Drift detection failed: {str(e)}")
                    st.exception(e)

# ==================== TAB 7: A/B TESTING ====================
with tab7:
    st.subheader("🆚 A/B Testing Framework")
    st.markdown("Compare multiple models scientifically and select the best performer")
    
    if not AB_TESTING_AVAILABLE:
        st.error("❌ A/B testing not available.")
    else:
        # Sub-tabs for create, view, analyze
        ab_tab1, ab_tab2, ab_tab3 = st.tabs(["➕ Create Test", "📊 Active Tests", "📈 Analyze Results"])
        
        # CREATE TEST
        with ab_tab1:
            st.markdown("### Create New A/B Test")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                test_name = st.text_input(
                    "Test Name",
                    value="model_comparison",
                    help="Unique name for this test"
                )
                
                test_symbol = st.text_input("Stock Symbol", value="AAPL", key="ab_symbol")
                
                test_duration = st.slider(
                    "Test Duration (days)",
                    7, 90, 30,
                    help="How long to run the test"
                )
            
            with col2:
                st.markdown("### Select Models to Test")
                
                # Get models for this symbol
                symbol_models = [m for m in all_models if m['symbol'] == test_symbol]
                
                if len(symbol_models) < 2:
                    st.warning(f"⚠️ Need at least 2 models for {test_symbol} to run A/B test")
                else:
                    # Model A
                    model_a_options = [
                        f"{m['model_type'].upper()} - {m.get('timestamp', 'N/A')[:8]}"
                        for m in symbol_models
                    ]
                    
                    model_a_idx = st.selectbox(
                        "Model A",
                        range(len(model_a_options)),
                        format_func=lambda x: model_a_options[x],
                        key="ab_model_a"
                    )
                    
                    # Model B
                    model_b_idx = st.selectbox(
                        "Model B",
                        range(len(model_a_options)),
                        format_func=lambda x: model_a_options[x],
                        index=1 if len(model_a_options) > 1 else 0,
                        key="ab_model_b"
                    )
                    
                    # Traffic split
                    st.markdown("### Traffic Split")
                    traffic_a = st.slider(
                        "Model A Traffic %",
                        0, 100, 50,
                        help="Percentage of predictions for Model A"
                    )
                    traffic_b = 100 - traffic_a
                    
                    st.info(f"Split: {traffic_a}% Model A, {traffic_b}% Model B")
            
            st.divider()
            
            if st.button("🚀 Create A/B Test", type="primary", use_container_width=True):
                if model_a_idx == model_b_idx:
                    st.error("❌ Please select different models for A and B")
                else:
                    try:
                        framework = ABTestingFramework()
                        
                        model_a = symbol_models[model_a_idx]
                        model_b = symbol_models[model_b_idx]
                        
                        test_id = framework.create_test(
                            test_name=test_name,
                            symbol=test_symbol,
                            models=[
                                {
                                    'model_id': model_a['model_id'],
                                    'model_type': model_a['model_type'],
                                    'version': 'v1'
                                },
                                {
                                    'model_id': model_b['model_id'],
                                    'model_type': model_b['model_type'],
                                    'version': 'v1'
                                }
                            ],
                            duration_days=test_duration,
                            traffic_split=[traffic_a / 100, traffic_b / 100]
                        )
                        
                        st.success(f"✅ A/B test created: `{test_id}`")
                        st.info("""
                        **Next steps:**
                        1. Test will automatically collect predictions
                        2. Check 'Active Tests' tab to monitor progress
                        3. Analyze results when sufficient data is collected
                        """)
                    
                    except Exception as e:
                        st.error(f"❌ Failed to create test: {str(e)}")
        
        # ACTIVE TESTS
        with ab_tab2:
            st.markdown("### 📊 Active A/B Tests")
            
            try:
                framework = ABTestingFramework()
                active_tests = framework.get_active_tests()
                
                if not active_tests:
                    st.info("No active A/B tests. Create one in the 'Create Test' tab.")
                else:
                    for test in active_tests:
                        with st.expander(f"🧪 {test['test_name']} ({test['symbol']})"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"**Test ID:** `{test['test_id']}`")
                                st.markdown(f"**Symbol:** {test['symbol']}")
                            
                            with col2:
                                st.markdown(f"**Created:** {test['created_at'][:10]}")
                                st.markdown(f"**Ends:** {test['end_date'][:10]}")
                            
                            with col3:
                                if st.button(f"📈 Analyze", key=f"analyze_{test['test_id']}"):
                                    st.session_state['analyze_test_id'] = test['test_id']
                                    st.info("Go to 'Analyze Results' tab")
                                
                                if st.button(f"🛑 Stop", key=f"stop_{test['test_id']}"):
                                    framework.stop_test(test['test_id'])
                                    st.success("Test stopped")
                                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error loading tests: {str(e)}")
        
        # ANALYZE RESULTS
        with ab_tab3:
            st.markdown("### 📈 Analyze A/B Test Results")
            
            try:
                framework = ABTestingFramework()
                active_tests = framework.get_active_tests()
                
                if not active_tests:
                    st.info("No active tests to analyze.")
                else:
                    # Test selection
                    test_names = [f"{t['test_name']} ({t['symbol']})" for t in active_tests]
                    
                    # Use session state if set
                    default_idx = 0
                    if 'analyze_test_id' in st.session_state:
                        for i, t in enumerate(active_tests):
                            if t['test_id'] == st.session_state['analyze_test_id']:
                                default_idx = i
                                break
                    
                    selected_test_idx = st.selectbox(
                        "Select Test",
                        range(len(test_names)),
                        format_func=lambda x: test_names[x],
                        index=default_idx
                    )
                    
                    selected_test = active_tests[selected_test_idx]
                    test_id = selected_test['test_id']
                    
                    if st.button("📊 Analyze Test", type="primary"):
                        with st.spinner("Analyzing test results..."):
                            analysis = framework.analyze_test(test_id)
                            
                            st.markdown(f"### Results: {analysis['test_name']}")
                            st.markdown(f"**Symbol:** {analysis['symbol']}")
                            
                            # Model comparison table
                            st.markdown("### 📊 Model Performance")
                            
                            comparison_data = []
                            for variant, data in analysis['models'].items():
                                metrics = data['metrics']
                                comparison_data.append({
                                    'Model': f"{variant}: {data['model_id'][:20]}...",
                                    'Predictions': metrics['n_predictions'],
                                    'With Actuals': metrics['n_with_actuals'],
                                    'MAE': f"${metrics['mae']:.2f}" if metrics['mae'] else 'N/A',
                                    'RMSE': f"${metrics['rmse']:.2f}" if metrics['rmse'] else 'N/A',
                                    'MAPE': f"{metrics['mape']:.2f}%" if metrics['mape'] else 'N/A',
                                    'Sufficient Data': '✅' if data['sufficient_data'] else '❌'
                                })
                            
                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                            
                            # Statistical comparison
                            if analysis.get('statistical_comparison'):
                                st.divider()
                                st.markdown("### 🔬 Statistical Analysis")
                                
                                comp = analysis['statistical_comparison']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("P-Value", f"{comp['p_value']:.4f}")
                                
                                with col2:
                                    st.metric(
                                        "Significant?",
                                        "Yes ✅" if comp['significant'] else "No ❌"
                                    )
                                
                                with col3:
                                    st.metric("Winner", f"Model {comp['winner']}")
                                
                                with col4:
                                    st.metric("Improvement", f"{comp['improvement_pct']:.1f}%")
                                
                                # Recommendation
                                st.divider()
                                if comp['winner'] != 'tie':
                                    st.success(f"### 🏆 {analysis['recommendation']}")
                                    st.markdown(f"""
                                    Model {comp['winner']} is statistically significantly better:
                                    - **{comp['improvement_pct']:.1f}% improvement** in MAE
                                    - **P-value: {comp['p_value']:.4f}** (< 0.05 = significant)
                                    - **Confidence: {comp['confidence_level']*100:.0f}%**
                                    
                                    ✅ Safe to deploy Model {comp['winner']}
                                    """)
                                else:
                                    st.info(f"### {analysis['recommendation']}")
                                    st.markdown("""
                                    No statistically significant difference found.
                                    - Continue testing or collect more data
                                    - Consider other factors (speed, complexity)
                                    """)
                            else:
                                st.warning("⚠️ " + analysis['recommendation'])
            
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

