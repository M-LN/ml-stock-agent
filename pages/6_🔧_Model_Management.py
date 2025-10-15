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
    
    # Import ML Mentor Engine
    try:
        from ml_mentor_engine import MLMentorEngine, analyze_saved_model
        import json
        
        # Check for API key and model selection
        api_key = os.getenv('OPENAI_API_KEY')
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if api_key:
                st.success("✅ OpenAI API key fundet")
            else:
                st.warning("⚠️ Ingen OpenAI API key - bruger rule-based anbefalinger")
        
        with col2:
            if api_key:
                llm_model = st.selectbox(
                    "LLM Model",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                    help="gpt-4o-mini: Billig og hurtig | gpt-4: Bedre kvalitet men dyrere"
                )
            else:
                llm_model = "gpt-4o-mini"
        
        if not api_key:
            with st.expander("ℹ️ Hvordan sætter jeg API key op?"):
                st.markdown("""
                For at bruge LLM-baserede anbefalinger:
                
                1. Få en API key fra [OpenAI](https://platform.openai.com/api-keys)
                2. Sæt environment variable:
                   ```powershell
                   $env:OPENAI_API_KEY = "din-api-key-her"
                   ```
                3. Genstart Streamlit
                
                **Model sammenligning:**
                - **gpt-4o-mini**: Billigst (~$0.001/analyse), hurtig, god til basic analyse
                - **gpt-4o**: Balance mellem pris og kvalitet (~$0.005/analyse)
                - **gpt-4**: Bedst kvalitet (~$0.02/analyse), dybdegående analyser
                
                **Note**: Rule-based anbefalinger virker stadig godt uden API key!
                """)
        
        st.divider()
        
        # Add refresh button
        col_refresh1, col_refresh2 = st.columns([3, 1])
        with col_refresh2:
            if st.button("🔄 Opdater Liste", use_container_width=True):
                # Clear any cached state
                if 'mentor_result' in st.session_state:
                    del st.session_state.mentor_result
                st.rerun()
        
        # Get all models with training logs
        logs_dir = "logs"
        available_models = []
        
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('_training.json')]
            for log_file in log_files:
                model_id = log_file.replace('_training.json', '')
                
                # Check if corresponding model file exists
                model_file_pkl = os.path.join(MODEL_DIR, f"{model_id}.pkl")
                model_file_h5 = os.path.join(MODEL_DIR, f"{model_id}.h5")
                
                if os.path.exists(model_file_pkl) or os.path.exists(model_file_h5):
                    # Load log to get info
                    with open(os.path.join(logs_dir, log_file), 'r') as f:
                        log_data = json.load(f)
                    
                    available_models.append({
                        'model_id': model_id,
                        'model_type': log_data.get('model_type', 'unknown'),
                        'symbol': log_data.get('symbol', 'N/A'),
                        'timestamp': log_data.get('timestamp', 'N/A'),
                        'version': log_data.get('version', 1),
                        'description': log_data.get('description', '')
                    })
        
        if not available_models:
            st.info("📭 Ingen modeller med training logs fundet. Træn en model først i 'Træn Nye Modeller' tab.")
        else:
            with col_refresh1:
                st.markdown(f"**{len(available_models)} model(ler) klar til analyse**")
            
            # Group models by symbol and type
            from collections import defaultdict
            grouped_models = defaultdict(lambda: defaultdict(list))
            
            for i, model in enumerate(available_models):
                grouped_models[model['symbol']][model['model_type']].append((i, model))
            
            # Model selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Build grouped options for selectbox
                grouped_options = []
                index_mapping = []
                
                for symbol in sorted(grouped_models.keys()):
                    grouped_options.append(("header", f"📊 {symbol}"))
                    index_mapping.append(None)  # Headers don't have models
                    
                    for model_type in sorted(grouped_models[symbol].keys()):
                        models_list = grouped_models[symbol][model_type]
                        
                        # Find best model in this group
                        best_idx = 0
                        best_mae = float('inf')
                        for idx, (_, m) in enumerate(models_list):
                            log_file = os.path.join(logs_dir, f"{m['model_id']}_training.json")
                            if os.path.exists(log_file):
                                with open(log_file, 'r') as f:
                                    log_data = json.load(f)
                                    val_mae = log_data.get('metrics', {}).get('val_mae', float('inf'))
                                    if val_mae < best_mae:
                                        best_mae = val_mae
                                        best_idx = idx
                        
                        grouped_options.append(("subheader", f"  🤖 {model_type.upper()} ({len(models_list)} versions)"))
                        index_mapping.append(None)
                        
                        # Sort versions by version number (newest first)
                        models_sorted = sorted(models_list, key=lambda x: x[1]['version'], reverse=True)
                        
                        for model_idx, (orig_idx, model) in enumerate(models_sorted):
                            is_best = model_idx == best_idx
                            badge = " ⭐" if is_best else ""
                            
                            if model.get('description'):
                                label = f"    v{model['version']}: {model['description'][:35]}{badge}"
                            else:
                                label = f"    v{model['version']} ({model['timestamp'][:10]}){badge}"
                            
                            grouped_options.append(("model", label))
                            index_mapping.append(orig_idx)
                
                # Custom selectbox with grouped display
                def format_option(i):
                    option_type, label = grouped_options[i]
                    if option_type == "header":
                        return label
                    elif option_type == "subheader":
                        return label
                    else:
                        return label
                
                # Filter out headers/subheaders for selection
                selectable_indices = [i for i, (opt_type, _) in enumerate(grouped_options) if opt_type == "model"]
                
                if selectable_indices:
                    selected_display_idx = st.selectbox(
                        "Vælg model at analysere:",
                        selectable_indices,
                        format_func=format_option
                    )
                    
                    selected_idx = index_mapping[selected_display_idx]
                    selected_model = available_models[selected_idx]
                else:
                    st.error("Ingen valid modeller fundet")
                    st.stop()
            
            with col2:
                analyze_button = st.button("🔍 Analyser Model", type="primary", use_container_width=True)
            
            st.divider()
            
            # Run analysis
            if analyze_button or 'mentor_result' in st.session_state:
                if analyze_button:
                    mode_text = f"med {llm_model}" if api_key else "med rule-based logic"
                    with st.spinner(f"🤖 Analyserer model {mode_text}..."):
                        try:
                            # Run ML Mentor analysis
                            result = analyze_saved_model(
                                selected_model['model_id'], 
                                api_key=api_key,
                                model=llm_model
                            )
                            st.session_state.mentor_result = result
                            st.session_state.used_model = llm_model if api_key else "rule-based"
                        except Exception as e:
                            st.error(f"❌ Analyse fejl: {str(e)}")
                            st.stop()
                
                result = st.session_state.mentor_result
                
                # Show which model was used
                used_model = st.session_state.get('used_model', 'unknown')
                if used_model != 'rule-based':
                    st.info(f"🤖 Analyseret med: **{used_model}**")
                else:
                    st.info(f"📝 Analyseret med: **Rule-based logic** (ingen API key)")
                
                # Display Health Score
                health_score = result['health_score']
                
                # Determine color
                if health_score >= 80:
                    health_color = "🟢"
                    health_bg = "#1e4620"
                elif health_score >= 60:
                    health_color = "🟡"
                    health_bg = "#3d3d1f"
                else:
                    health_color = "🔴"
                    health_bg = "#4a1c1c"
                
                st.markdown(f"""
                <div style="background-color: {health_bg}; padding: 30px; border-radius: 10px; text-align: center;">
                    <h1 style="font-size: 50px; margin: 0;">{health_color} MODEL HEALTH SCORE</h1>
                    <h1 style="font-size: 60px; margin: 10px 0;">{health_score}/100</h1>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Display metrics
                st.markdown("### 📊 Performance Metrics")
                
                metrics = result['analysis']['metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Train MAE", f"${metrics['train_mae']:.2f}")
                
                with col2:
                    st.metric("Val MAE", f"${metrics['val_mae']:.2f}")
                
                with col3:
                    st.metric("Generalization Gap", f"{metrics['generalization_gap']:.1f}%",
                             delta=f"{'High' if metrics['generalization_gap'] > 50 else 'OK'}")
                
                with col4:
                    data_info = result['analysis']['data_info']
                    st.metric("Training Samples", data_info['training_samples'])
                
                # ========== VISUALIZATION SECTION ==========
                st.divider()
                st.markdown("### 📊 Training Visualizations")
                
                # Load training log for visualization data
                # Use model_id from result to ensure we're looking at the analyzed model
                model_id = result.get('model_id', selected_model['model_id'])
                log_file = os.path.join(logs_dir, f"{model_id}_training.json")
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                    
                    # Debug: Check if data was loaded
                    if not log_data:
                        st.error("⚠️ Training log is empty")
                        log_data = {}
                    
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Training Curves", "🎯 Model Comparison", "📊 Feature Analysis"])
                    
                    # TAB 1: Training Curves
                    with viz_tab1:
                        model_type = log_data.get('model_type', 'unknown')
                        
                        if model_type == 'lstm':
                            # LSTM Training History
                            history = log_data.get('training_history', {})
                            
                            if history and 'loss' in history:
                                import plotly.graph_objects as go
                                
                                epochs = list(range(1, len(history['loss']) + 1))
                                
                                # Loss Curve
                                fig_loss = go.Figure()
                                
                                fig_loss.add_trace(go.Scatter(
                                    x=epochs,
                                    y=history['loss'],
                                    mode='lines+markers',
                                    name='Training Loss',
                                    line=dict(color='#636EFA', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                if 'val_loss' in history:
                                    fig_loss.add_trace(go.Scatter(
                                        x=epochs,
                                        y=history['val_loss'],
                                        mode='lines+markers',
                                        name='Validation Loss',
                                        line=dict(color='#EF553B', width=2),
                                        marker=dict(size=6)
                                    ))
                                
                                fig_loss.update_layout(
                                    title='📉 Loss over Epochs',
                                    xaxis_title='Epoch',
                                    yaxis_title='Loss (MAE)',
                                    hovermode='x unified',
                                    height=400,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_loss, use_container_width=True)
                                
                                # Analysis of loss curves
                                train_loss = history['loss']
                                val_loss = history.get('val_loss', [])
                                
                                if val_loss:
                                    # Check for overfitting
                                    final_gap = val_loss[-1] - train_loss[-1]
                                    avg_gap = sum([v - t for v, t in zip(val_loss, train_loss)]) / len(val_loss)
                                    
                                    # Check for plateau
                                    last_5_train = train_loss[-5:] if len(train_loss) >= 5 else train_loss
                                    train_improvement = (last_5_train[0] - last_5_train[-1]) / last_5_train[0] * 100
                                    
                                    st.markdown("#### 🔍 Training Analysis")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if final_gap > avg_gap * 1.5:
                                            st.warning("⚠️ **Overfitting detected**\nValidation loss increasing")
                                        else:
                                            st.success("✅ **Good convergence**\nNo overfitting signs")
                                    
                                    with col2:
                                        if train_improvement < 1:
                                            st.info("📊 **Training plateau**\nLoss stopped improving")
                                        else:
                                            st.success("📈 **Active learning**\nModel still improving")
                                    
                                    with col3:
                                        best_epoch = val_loss.index(min(val_loss)) + 1
                                        st.metric("Best Epoch", f"{best_epoch}/{len(epochs)}")
                            else:
                                st.info("📭 No training history available for LSTM model")
                        
                        elif model_type in ['rf', 'xgboost']:
                            # For tree-based models, show training progression
                            st.markdown("#### 🌲 Tree-Based Model Training")
                            
                            params = log_data.get('parameters', {})
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Model Configuration:**")
                                if model_type == 'rf':
                                    st.markdown(f"- Estimators: **{params.get('n_estimators', 'N/A')}**")
                                    st.markdown(f"- Max Depth: **{params.get('max_depth', 'N/A')}**")
                                    st.markdown(f"- Min Samples Split: **{params.get('min_samples_split', 'N/A')}**")
                                else:  # xgboost
                                    st.markdown(f"- Estimators: **{params.get('n_estimators', 'N/A')}**")
                                    st.markdown(f"- Learning Rate: **{params.get('learning_rate', 'N/A')}**")
                                    st.markdown(f"- Max Depth: **{params.get('max_depth', 'N/A')}**")
                            
                            with col2:
                                st.markdown("**Performance:**")
                                metrics = log_data.get('metrics', {})
                                st.markdown(f"- Train MAE: **${metrics.get('train_mae', 0):.2f}**")
                                st.markdown(f"- Val MAE: **${metrics.get('val_mae', 0):.2f}**")
                                st.markdown(f"- Generalization Gap: **{metrics.get('generalization_gap', 0):.1f}%**")
                            
                            # Show training time info
                            train_time = log_data.get('training_time', 0)
                            st.info(f"⏱️ Training completed in **{train_time:.2f} seconds**")
                        
                        else:
                            st.info(f"📭 Visualization not available for {model_type} models")
                    
                    # TAB 2: Model Comparison (across versions)
                    with viz_tab2:
                        st.markdown("#### 📊 Version Comparison")
                        
                        # Get all versions of this symbol + type
                        symbol = log_data.get('symbol', 'N/A')
                        model_type = log_data.get('model_type', 'unknown')
                        
                        # Find all related models
                        version_data = []
                        for log_file_name in os.listdir(logs_dir):
                            if log_file_name.endswith('_training.json'):
                                with open(os.path.join(logs_dir, log_file_name), 'r') as f:
                                    other_log = json.load(f)
                                    
                                    if (other_log.get('symbol') == symbol and 
                                        other_log.get('model_type') == model_type):
                                        # Handle both old (metrics) and new (final_metrics) format
                                        metrics_data = other_log.get('final_metrics', other_log.get('metrics', {}))
                                        data_info = other_log.get('data_stats', other_log.get('data_info', {}))
                                        
                                        version_data.append({
                                            'version': other_log.get('version', 1),
                                            'val_mae': metrics_data.get('val_mae', 0),
                                            'train_mae': metrics_data.get('train_mae', 0),
                                            'samples': data_info.get('training_samples', 0),
                                            'description': other_log.get('description', ''),
                                            'timestamp': other_log.get('timestamp', '')
                                        })
                        
                        if len(version_data) > 1:
                            # Sort by version
                            version_data = sorted(version_data, key=lambda x: x['version'])
                            
                            # Create comparison chart
                            fig_comparison = go.Figure()
                            
                            versions = [f"v{v['version']}" for v in version_data]
                            val_maes = [v['val_mae'] for v in version_data]
                            train_maes = [v['train_mae'] for v in version_data]
                            
                            fig_comparison.add_trace(go.Bar(
                                x=versions,
                                y=val_maes,
                                name='Validation MAE',
                                marker_color='#EF553B',
                                text=[f"${v:.2f}" for v in val_maes],
                                textposition='outside'
                            ))
                            
                            fig_comparison.add_trace(go.Bar(
                                x=versions,
                                y=train_maes,
                                name='Training MAE',
                                marker_color='#636EFA',
                                text=[f"${v:.2f}" for v in train_maes],
                                textposition='outside'
                            ))
                            
                            fig_comparison.update_layout(
                                title=f'📊 {symbol} - {model_type.upper()} Version Performance',
                                xaxis_title='Version',
                                yaxis_title='MAE ($)',
                                barmode='group',
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # Show improvement table
                            st.markdown("**📈 Version History:**")
                            
                            for i, v in enumerate(version_data):
                                if i > 0:
                                    prev_mae = version_data[i-1]['val_mae']
                                    current_mae = v['val_mae']
                                    
                                    # Protect against division by zero
                                    if prev_mae > 0:
                                        improvement = ((prev_mae - current_mae) / prev_mae) * 100
                                    else:
                                        # If previous MAE was 0, just show absolute difference
                                        improvement = -(current_mae * 100) if current_mae > 0 else 0
                                    
                                    if improvement > 5:
                                        badge = "🎉 MAJOR"
                                    elif improvement > 0:
                                        badge = "✅ Better"
                                    elif improvement > -5:
                                        badge = "⚠️ Slightly worse"
                                    else:
                                        badge = "❌ WORSE"
                                    
                                    st.markdown(f"- **v{v['version']}** {badge} ({improvement:+.1f}%): {v['description'][:60]}")
                                else:
                                    st.markdown(f"- **v{v['version']}** (Baseline): {v['description'][:60]}")
                        
                        else:
                            st.info("📭 Only one version available. Train more versions to see comparison!")
                    
                    # TAB 3: Feature Analysis
                    with viz_tab3:
                        st.markdown("#### 🎯 Model Insights")
                        
                        # Load data from log_data (handles both old and new format)
                        data_stats = log_data.get('data_stats', log_data.get('data_info', {}))
                        final_metrics = log_data.get('final_metrics', log_data.get('metrics', {}))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Data Statistics:**")
                            st.markdown(f"- Training samples: **{data_stats.get('training_samples', 0)}**")
                            st.markdown(f"- Validation samples: **{data_stats.get('validation_samples', 0)}**")
                            st.markdown(f"- Total samples: **{data_stats.get('total_samples', 0)}**")
                            
                            # Calculate split ratio
                            train_samples = data_stats.get('training_samples', 0)
                            total_samples = data_stats.get('total_samples', 0)
                            if total_samples > 0:
                                train_pct = int((train_samples / total_samples) * 100)
                                val_pct = 100 - train_pct
                                st.markdown(f"- Split: **{train_pct}/{val_pct}**")
                            else:
                                st.markdown(f"- Split: **80/20**")
                        
                        with col2:
                            st.markdown("**🎯 Performance Metrics:**")
                            
                            train_mae = final_metrics.get('train_mae', 0)
                            val_mae = final_metrics.get('val_mae', 0)
                            
                            # Calculate generalization gap
                            if train_mae > 0:
                                gen_gap = ((val_mae - train_mae) / train_mae) * 100
                            else:
                                gen_gap = 0
                            
                            st.markdown(f"- Train MAE: **${train_mae:.2f}**")
                            st.markdown(f"- Val MAE: **${val_mae:.2f}**")
                            st.markdown(f"- Gen Gap: **{gen_gap:.1f}%**")
                            
                            if gen_gap > 50:
                                st.warning("⚠️ High overfitting")
                            elif gen_gap > 30:
                                st.info("📊 Moderate generalization gap")
                            else:
                                st.success("✅ Good generalization")
                        
                        # Show model parameters
                        st.markdown("---")
                        st.markdown("**⚙️ Model Configuration:**")
                        
                        params = log_data.get('parameters', {})
                        
                        # Display in expandable section
                        with st.expander("See all parameters"):
                            for key, value in params.items():
                                st.markdown(f"- **{key}**: {value}")
                
                else:
                    st.warning("⚠️ Training log not found - cannot display visualizations")
                
                # Display History if available
                if 'history' in result and result['history'].get('has_parent'):
                    st.divider()
                    history = result['history']
                    
                    st.markdown("### 📜 Model History")
                    
                    outcome = history.get('outcome')
                    if outcome:
                        # Show verdict banner
                        improvement = outcome['improvement_percent']
                        verdict = outcome['verdict']
                        
                        if verdict in ['major_improvement', 'improved']:
                            st.success(f"✅ **Previous attempt SUCCEEDED** ({improvement:+.1f}% improvement)")
                        elif verdict == 'minor_improvement':
                            st.info(f"📈 **Previous attempt had minor success** ({improvement:+.1f}%)")
                        elif verdict == 'minimal_degradation':
                            st.warning(f"⚠️ **Previous attempt had minimal impact** ({improvement:+.1f}%)")
                        elif verdict == 'degraded':
                            st.error(f"❌ **Previous attempt FAILED** ({improvement:+.1f}% worse)")
                        else:
                            st.error(f"🚨 **Previous attempt SEVERELY FAILED** ({improvement:+.1f}% worse)")
                        
                        # Show what was tried
                        applied_rec = history.get('applied_recommendation', {})
                        if applied_rec:
                            with st.expander("📋 See what was tried", expanded=False):
                                st.markdown(f"**Recommendation:** {applied_rec.get('title', 'N/A')}")
                                st.markdown(f"**Action:** {applied_rec.get('action', 'N/A')}")
                                st.markdown(f"**Priority:** {applied_rec.get('priority', 'N/A').upper()}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Before (Val MAE)", f"${outcome['parent_val_mae']:.2f}")
                                with col2:
                                    st.metric("After (Val MAE)", f"${outcome['current_val_mae']:.2f}", 
                                             delta=f"{improvement:+.1f}%")
                        
                        # Show learning message
                        if outcome['success']:
                            st.info("💡 **ML Mentor vil bruge denne success til at anbefale lignende forbedringer**")
                        else:
                            st.warning("💡 **ML Mentor ved nu at denne tilgang IKKE virker og vil anbefale alternative løsninger**")
                    
                    # Show version history
                    related = history.get('related_models', [])
                    if len(related) > 1:
                        with st.expander(f"📊 Version History ({len(related)} versions)", expanded=False):
                            for rel in related:
                                version = rel['version']
                                desc = rel.get('description', 'Initial version')
                                val_mae = rel.get('val_mae', 0)
                                
                                # Highlight current version
                                if version == history['version']:
                                    st.markdown(f"**→ v{version} (CURRENT): {desc}** - Val MAE: ${val_mae:.2f}")
                                else:
                                    st.markdown(f"v{version}: {desc} - Val MAE: ${val_mae:.2f}")
                
                st.divider()
                
                # Display Strengths and Issues
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Strengths")
                    strengths = result['analysis']['strengths']
                    if strengths:
                        for strength in strengths:
                            st.success(f"**{strength['type'].title()}**: {strength['description']}")
                    else:
                        st.info("Ingen specifikke styrker identificeret")
                
                with col2:
                    st.markdown("### ⚠️ Issues Found")
                    issues = result['analysis']['issues']
                    if issues:
                        for issue in issues:
                            severity_color = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "⚪"
                            st.warning(f"{severity_color} **[{issue['severity'].upper()}] {issue['description']}**\n\n{issue['impact']}")
                    else:
                        st.success("✅ Ingen alvorlige problemer fundet!")
                
                st.divider()
                
                # Display Recommendations
                st.markdown("### 💡 Recommendations")
                st.markdown("**Actionable anbefalinger til at forbedre din model:**")
                
                recommendations = result['recommendations']
                
                for i, rec in enumerate(recommendations, 1):
                    priority = rec['priority']
                    
                    # Priority colors
                    if priority == 'high':
                        priority_badge = "🔴 HIGH"
                        border_color = "#c62828"
                    elif priority == 'medium':
                        priority_badge = "🟡 MEDIUM"
                        border_color = "#f9a825"
                    else:
                        priority_badge = "🟢 LOW"
                        border_color = "#2e7d32"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="border-left: 4px solid {border_color}; padding-left: 15px; margin-bottom: 20px;">
                            <h4>{i}. {priority_badge} - {rec['title']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"**💭 Why:** {rec['why']}")
                            st.markdown(f"**🎯 Expected:** {rec['expected']}")
                        
                        with col2:
                            st.markdown(f"**✅ Action:**")
                            st.info(rec['action'])
                        
                        # Apply button with actual retraining
                        apply_col1, apply_col2 = st.columns([3, 1])
                        with apply_col2:
                            if st.button(f"🚀 Apply", key=f"apply_{i}", type="primary"):
                                st.session_state[f'applying_rec_{i}'] = True
                        
                        # Show retrain dialog if button clicked
                        if st.session_state.get(f'applying_rec_{i}', False):
                            with st.expander("🔄 Retrain Progress", expanded=True):
                                try:
                                    from ml_mentor_retrain import apply_recommendation_and_retrain
                                    
                                    # Get symbol from model_id
                                    symbol = selected_model['model_id'].split('_')[1]
                                    
                                    # Load full training log to get parameters
                                    log_file = os.path.join('logs', f"{selected_model['model_id']}_training.json")
                                    with open(log_file, 'r') as f:
                                        training_log = json.load(f)
                                    
                                    current_params = training_log.get('parameters', {})
                                    
                                    with st.spinner(f"🔄 Retrainer model med anbefalede parametre..."):
                                        retrain_result = apply_recommendation_and_retrain(
                                            recommendation=rec,
                                            model_id=selected_model['model_id'],
                                            symbol=symbol,
                                            current_params=current_params
                                        )
                                    
                                    if retrain_result['success']:
                                        st.success(f"✅ Model retrained successfully!")
                                        st.markdown(f"**🆕 New Model:** `{retrain_result['new_model_id']}`")
                                        if 'description' in retrain_result:
                                            st.info(f"📝 **{retrain_result['description']}**")
                                        
                                        # Show before/after comparison
                                        st.markdown("#### 📊 Before vs After")
                                        
                                        # Get metrics
                                        old_metrics = result['analysis']['metrics']
                                        old_val_mae = old_metrics['val_mae']
                                        old_train_mae = old_metrics.get('train_mae', 0)
                                        old_samples = result['analysis']['data_info'].get('training_samples', 0)
                                        old_gen_gap = old_val_mae - old_train_mae if old_train_mae > 0 else 0
                                        
                                        new_val_mae = retrain_result['new_metrics']['val_mae']
                                        new_train_mae = retrain_result['new_metrics'].get('train_mae', 0)
                                        new_samples = retrain_result.get('training_samples', retrain_result.get('data_samples', 0))
                                        new_gen_gap = retrain_result.get('new_gen_gap', 0)
                                        
                                        improvement = ((old_val_mae - new_val_mae) / old_val_mae) * 100
                                        
                                        # Add to result for delete button
                                        retrain_result['old_val_mae'] = old_val_mae
                                        retrain_result['old_samples'] = old_samples
                                        retrain_result['old_gen_gap'] = old_gen_gap
                                        
                                        # Show verdict banner based on improvement
                                        if improvement > 10:
                                            st.success("### 🎉 STOR FORBEDRING! Modellen er meget bedre!")
                                            st.markdown("✅ **Anbefaling:** Behold den nye model og brug den til trading!")
                                        elif improvement > 5:
                                            st.success("### ✅ FORBEDRET! Modellen performer bedre")
                                            st.markdown("✅ **Anbefaling:** Behold den nye model.")
                                        elif improvement > 0:
                                            st.info("### 📈 Lille forbedring - modellen er lidt bedre")
                                            st.markdown("💡 **Anbefaling:** Behold den nye model.")
                                        elif improvement > -5:
                                            st.warning("### ⚠️ MINIMAL FORVÆRRING - næsten samme performance")
                                            st.markdown("⚠️ **Anbefaling:** Du kan beholde den nye model, men overvej andre justeringer.")
                                        elif improvement > -20:
                                            st.error("### ❌ FORVÆRRET! Modellen performer dårligere")
                                            st.markdown("🚨 **Anbefaling:** Brug den GAMLE MODEL i stedet!")
                                            st.markdown("**Mulige årsager:**")
                                            st.markdown("- Anbefalingen passede ikke til denne model/data")
                                            st.markdown("- Mere data tilføjede noise i stedet for signal")
                                            st.markdown("- Andre parametre skal også justeres samtidig")
                                        else:
                                            st.error("### 🚨 MEGET VÆRRE! Brug IKKE denne model!")
                                            st.markdown("❌ **Anbefaling:** SLET denne model og brug den GAMLE MODEL!")
                                        
                                        # Show 4 metrics
                                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                                        
                                        with comp_col1:
                                            st.metric("Old Val MAE", f"${old_val_mae:.2f}")
                                        
                                        with comp_col2:
                                            st.metric("New Val MAE", f"${new_val_mae:.2f}", 
                                                     delta=f"{-improvement:.1f}%" if improvement > 0 else f"+{abs(improvement):.1f}%",
                                                     delta_color="inverse")
                                        
                                        with comp_col3:
                                            # Show training samples with delta
                                            if old_samples > 0:
                                                sample_delta = new_samples - old_samples
                                                st.metric("Training Samples", new_samples, f"{sample_delta:+d}")
                                            else:
                                                st.metric("Training Samples", new_samples)
                                        
                                        with comp_col4:
                                            # Show generalization gap comparison
                                            if old_gen_gap > 0 and new_gen_gap > 0:
                                                gap_change = ((new_gen_gap - old_gen_gap) / old_gen_gap) * 100
                                                st.metric("Gen. Gap", f"${new_gen_gap:.2f}", f"{gap_change:+.1f}%")
                                            else:
                                                st.metric("Gen. Gap", f"${new_gen_gap:.2f}" if new_gen_gap > 0 else "N/A")
                                        
                                        st.info("💡 Den nye model er blevet gemt og kan findes i Model Management listen!")
                                        
                                        # Show delete button if model is significantly worse
                                        if improvement < -5:
                                            st.markdown("---")
                                            delete_col1, delete_col2 = st.columns([3, 1])
                                            with delete_col1:
                                                st.warning("⚠️ Vil du slette den dårlige model og prøve en anden anbefaling?")
                                            with delete_col2:
                                                if st.button("🗑️ Slet Model", key=f"delete_bad_{i}", type="secondary"):
                                                    try:
                                                        # Get file paths
                                                        import os
                                                        model_id = retrain_result['new_model_id']
                                                        log_file = f"logs/{model_id}_training.json"
                                                        model_file = f"saved_models/{model_id}.pkl"
                                                        
                                                        # Delete files
                                                        if os.path.exists(log_file):
                                                            os.remove(log_file)
                                                        if os.path.exists(model_file):
                                                            os.remove(model_file)
                                                        
                                                        st.success(f"✅ Model {model_id} slettet!")
                                                        st.info("🔄 Tryk 'Refresh' for at opdatere listen")
                                                    except Exception as e:
                                                        st.error(f"❌ Kunne ikke slette model: {str(e)}")
                                
                                except Exception as e:
                                    st.error(f"❌ Fejl ved retrain: {str(e)}")
                        
                        st.markdown("---")
                
                st.divider()
                
                # Additional info
                with st.expander("ℹ️ Om ML Mentor"):
                    st.markdown("""
                    **Hvordan virker ML Mentor?**
                    
                    1. **Metrics Analysis**: Analyserer train/validation performance
                    2. **Issue Detection**: Finder overfitting, høje fejlrater, data-problemer
                    3. **LLM Recommendations**: GPT-4 genererer intelligente, actionable anbefalinger
                    4. **Health Score**: 0-100 score baseret på overall model kvalitet
                    
                    **Med LLM (GPT-4):**
                    - Dybere analyse af træningskurver
                    - Kontekst-aware anbefalinger
                    - Forventede forbedringer estimeret
                    
                    **Without LLM (Rule-based):**
                    - Stadig effektive anbefalinger
                    - Baseret på best practices
                    - Hurtigere og ingen API cost
                    
                    **Kommende features:**
                    - ✅ Auto-retrain med anbefalinger
                    - ✅ Before/after comparison
                    - ✅ Training curve visualisering
                    """)
        
    except ImportError:
        st.error("❌ ml_mentor_engine.py ikke fundet. Sørg for at filen er i samme directory.")

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
    
    from grid_search_engine import GridSearchEngine, get_previous_grid_searches
    import yfinance as yf
    import numpy as np
    
    st.markdown("""
    Grid Search tester systematisk forskellige kombinationer af hyperparametre 
    og finder den bedste konfiguration baseret på validation performance.
    
    **Hvordan det virker:**
    1. Vælg model type og data
    2. Vælg search space (small/medium/large)
    3. Start search - systemet tester alle kombinationer
    4. Se resultater og træn bedste model
    """)
    
    st.divider()
    
    # Previous searches
    with st.expander("📜 Se tidligere Grid Searches"):
        previous_searches = get_previous_grid_searches()
        
        if previous_searches:
            st.markdown(f"**{len(previous_searches)} tidligere searches fundet:**")
            
            for search in previous_searches[:10]:
                st.markdown(f"- **{search['model_type'].upper()}** - {search['symbol']} - {search['timestamp'][:8]}")
        else:
            st.info("Ingen tidligere grid searches fundet")
    
    st.divider()
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Model selection
        gs_model_type = st.selectbox(
            "Model Type",
            ["rf", "xgboost", "lstm"],
            format_func=lambda x: {"rf": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}[x],
            key="gs_model_type"
        )
        
        # Symbol
        gs_symbol = st.text_input("Symbol", "AAPL", key="gs_symbol").upper()
        
        # Date range
        gs_period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1, key="gs_period")
    
    with col2:
        # Search space
        search_space = st.selectbox(
            "Search Space",
            ["small", "medium", "large"],
            help="Small: ~10-20 combinations, Medium: ~50-100, Large: ~200-500"
        )
        
        # Max trials
        use_max_trials = st.checkbox("Limit number of trials", value=False)
        
        if use_max_trials:
            max_trials = st.number_input("Max trials", min_value=5, max_value=100, value=20)
        else:
            max_trials = None
        
        # Window and horizon (for data prep)
        gs_window = st.number_input("Window", min_value=5, max_value=120, value=30, key="gs_window")
        gs_horizon = st.number_input("Horizon", min_value=1, max_value=30, value=5, key="gs_horizon")
    
    # Show estimated combinations
    from grid_search_engine import GridSearchEngine as GSE_temp
    temp_engine = GSE_temp(gs_model_type, gs_symbol, None, None, None, None)
    param_grid = temp_engine.get_param_grid(search_space)
    
    from sklearn.model_selection import ParameterGrid
    total_combinations = len(list(ParameterGrid(param_grid)))
    
    if max_trials and max_trials < total_combinations:
        st.info(f"🔢 Will test **{max_trials}** random combinations out of {total_combinations} total")
    else:
        st.info(f"🔢 Will test all **{total_combinations}** combinations")
    
    with st.expander("📋 See parameter ranges"):
        st.json(param_grid)
    
    st.divider()
    
    # Run button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        run_grid_search = st.button("🚀 Start Grid Search", type="primary", use_container_width=True)
    
    # Run grid search
    if run_grid_search:
        st.markdown("### 🔄 Running Grid Search...")
        
        # Download data
        with st.spinner(f"📊 Downloading {gs_symbol} data..."):
            try:
                ticker = yf.Ticker(gs_symbol)
                data = ticker.history(period=gs_period)
                
                if data.empty:
                    st.error(f"❌ No data found for {gs_symbol}")
                    st.stop()
                
                st.success(f"✅ Downloaded {len(data)} days of data")
            except Exception as e:
                st.error(f"❌ Error downloading data: {str(e)}")
                st.stop()
        
        # Prepare features
        with st.spinner("🔧 Preparing features..."):
            try:
                from agent_interactive import prepare_features_for_training
                
                X, y = prepare_features_for_training(
                    data,
                    window=gs_window,
                    horizon=gs_horizon
                )
                
                # Train/val split
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                st.success(f"✅ Prepared {len(X_train)} training samples, {len(X_val)} validation samples")
            except Exception as e:
                st.error(f"❌ Error preparing features: {str(e)}")
                st.stop()
        
        # Run grid search
        st.markdown("### 📊 Grid Search Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_area = st.empty()
        
        # Track best results (use session state to avoid nonlocal issues)
        if 'gs_current_best_mae' not in st.session_state:
            st.session_state.gs_current_best_mae = float('inf')
            st.session_state.gs_current_best_params = None
        
        def progress_callback(trial_num, total_trials, result):
            # Update progress
            progress = trial_num / total_trials
            progress_bar.progress(progress)
            
            # Update status
            if result['success']:
                status_text.markdown(f"**Trial {trial_num}/{total_trials}** - Val MAE: ${result['val_mae']:.2f} (Train: ${result['train_mae']:.2f}, Gen Gap: {result['gen_gap']:.1f}%)")
                
                # Track best
                if result['val_mae'] < st.session_state.gs_current_best_mae:
                    st.session_state.gs_current_best_mae = result['val_mae']
                    st.session_state.gs_current_best_params = result['params']
                    
                    results_area.success(f"🎉 **New best!** Val MAE: ${st.session_state.gs_current_best_mae:.2f}")
            else:
                status_text.error(f"**Trial {trial_num}/{total_trials}** - Failed: {result['error']}")
        
        # Initialize engine
        engine = GridSearchEngine(gs_model_type, gs_symbol, X_train, y_train, X_val, y_val)
        
        # Run search
        try:
            search_result = engine.run_grid_search(
                search_space=search_space,
                max_trials=max_trials,
                progress_callback=progress_callback
            )
            
            # Save results
            result_file = engine.save_results(search_result)
            
            progress_bar.progress(1.0)
            status_text.success(f"✅ Grid Search completed! Results saved to: {result_file}")
            
            st.divider()
            
            # Display results
            if search_result['success']:
                st.markdown("## 🎉 Grid Search Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Val MAE", f"${search_result['best_val_mae']:.2f}")
                
                with col2:
                    st.metric("Best Train MAE", f"${search_result['best_train_mae']:.2f}")
                
                with col3:
                    st.metric("Gen Gap", f"{search_result['best_gen_gap']:.1f}%")
                
                with col4:
                    st.metric("Successful Trials", f"{search_result['successful_trials']}/{search_result['total_trials']}")
                
                if search_result['improvement_over_worst'] > 0:
                    st.success(f"📈 **{search_result['improvement_over_worst']:.1f}%** improvement over worst configuration!")
                
                st.markdown("### 🏆 Best Parameters")
                st.json(search_result['best_params'])
                
                # Show top 5 results
                st.markdown("### 📊 Top 5 Configurations")
                
                successful_results = [r for r in search_result['all_results'] if r['success']]
                top_5 = sorted(successful_results, key=lambda x: x['val_mae'])[:5]
                
                for i, result in enumerate(top_5, 1):
                    rank_emoji = ["🥇", "🥈", "🥉"][min(i-1, 2)] if i <= 3 else f"#{i}"
                    
                    with st.expander(f"{rank_emoji} Val MAE: ${result['val_mae']:.2f} (Train: ${result['train_mae']:.2f})"):
                        st.json(result['params'])
                        st.markdown(f"**Gen Gap:** {result['gen_gap']:.1f}%")
                        st.markdown(f"**Training Time:** {result['training_time']:.1f}s")
                
                # Visualization
                st.markdown("### 📈 Performance Distribution")
                
                import plotly.graph_objects as go
                
                val_maes = [r['val_mae'] for r in successful_results]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=val_maes,
                    nbinsx=20,
                    name='Val MAE Distribution'
                ))
                
                fig.update_layout(
                    title='Distribution of Validation MAE across all configurations',
                    xaxis_title='Val MAE ($)',
                    yaxis_title='Count',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Train button
                st.markdown("### 🚀 Train Best Model")
                
                if st.button("✅ Train model with best parameters", type="primary"):
                    st.info("💡 Go to 'Træn Nye Modeller' tab and use these parameters:")
                    st.json(search_result['best_params'])
            
            else:
                st.error(f"❌ Grid Search failed: {search_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Grid Search error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
