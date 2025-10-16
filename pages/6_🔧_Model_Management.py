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
            st.markdown(f"- **{m['model_type'].upper()}** ({m['symbol']}) - {m['timestamp'][:8]}")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏋️ Træn Nye Modeller", "📂 Gemte Modeller", "🔮 Brug Gemt Model", "🧠 ML Mentor", "🎛️ Grid Search"])

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
        
        # Train button
        train_button = st.button("🚀 Træn Model", type="primary", use_container_width=True)
    
    with col2:
        if train_button:
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
            
            # Train model
            with st.spinner(f"🏋️ Træner {model_type} model..."):
                try:
                    if model_type == "Random Forest":
                        result = train_and_save_rf(
                            data, train_symbol,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            window=window,
                            horizon=horizon
                        )
                    elif model_type == "XGBoost":
                        result = train_and_save_xgboost(
                            data, train_symbol,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            window=window,
                            horizon=horizon
                        )
                    elif model_type == "LSTM":
                        from agent_interactive import train_and_save_lstm
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
                    
                    if result:
                        st.success("🎉 Model trænet og gemt!")
                        
                        # Display results
                        st.markdown("### 📊 Training Resultater")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Training MAE", f"${result['metadata']['train_mae']:.2f}")
                        with col_b:
                            st.metric("Training RMSE", f"${result['metadata']['train_rmse']:.2f}")
                        with col_c:
                            st.metric("Training Samples", result['metadata']['training_samples'])
                        
                        # Show metadata
                        with st.expander("📋 Model Metadata"):
                            st.json(result['metadata'])
                        
                        st.info(f"💾 Model gemt: `{result['filepath']}`")
                        
                except Exception as e:
                    st.error(f"❌ Fejl ved træning: {str(e)}")
        else:
            # Show instructions
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
                                st.caption(f"📅 {model_info['timestamp']}")
                            
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
                    with st.expander(f"🤖 {model_info['model_type'].upper()} - {model_info['symbol']} ({model_info['timestamp']})"):
                        col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Model Type:** {model_info['model_type'].upper()}")
                    st.markdown(f"**Symbol:** {model_info['symbol']}")
                    st.markdown(f"**Timestamp:** {model_info['timestamp']}")
                    
                    if 'metadata' in model_info:
                        meta = model_info['metadata']
                        st.markdown(f"**Window:** {meta.get('window', 'N/A')}")
                        st.markdown(f"**Horizon:** {meta.get('horizon', 'N/A')}")
                        st.markdown(f"**Training MAE:** ${meta.get('train_mae', 'N/A'):.2f}")
                        st.markdown(f"**Training RMSE:** ${meta.get('train_rmse', 'N/A'):.2f}")
                        
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
        model_options = [f"{m['model_type'].upper()} - {m['symbol']} ({m['timestamp']})" for m in all_models]
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
                st.metric("Training MAE", f"${selected_model_info['metadata'].get('train_mae', 0):.2f}")
        
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
                        result = predict_with_saved_model(
                            model_package, 
                            data, 
                            horizon=selected_model_info['metadata'].get('horizon', 1)
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
            model_options = [f"{m['timestamp']} - {m['model_type']} ({m['symbol']})" for m in all_models]
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
                                                    nested_metadata = model_package.get("metadata", {})
                                                    
                                                    retrain_result = apply_recommendation_and_retrain(
                                                        rec, selected_model_id, 
                                                        symbol, 
                                                        nested_metadata
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
        
        # Preview search space
        with st.expander("🔍 Preview Search Space"):
            if gs_model_type in SEARCH_SPACES:
                space = SEARCH_SPACES[gs_model_type]
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
                        # Initialize engine
                        search_space = SEARCH_SPACES.get(gs_model_type, {})
                        engine = GridSearchEngine(gs_model_type, gs_symbol, search_space)
                        
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
                                st.success(f"✅ Grid search completed! Tested {results['trials_completed']} combinations")
                                
                                # Best params
                                st.markdown("### 🏆 Best Parameters Found")
                                st.json(results["best_params"])
                                
                                # Best score
                                st.metric("Best MAE", f"{results['best_score']:.2f}")
                                
                                # Results table
                                st.markdown("### 📊 All Results")
                                results_df = pd.DataFrame(results["all_results"])
                                st.dataframe(results_df.sort_values("score"), use_container_width=True)
                                
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
