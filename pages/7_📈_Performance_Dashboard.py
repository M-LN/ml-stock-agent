import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime
from collections import defaultdict

st.set_page_config(page_title="Performance Dashboard", page_icon="📈", layout="wide")

st.title("📈 Performance Dashboard")
st.markdown("**Overview af alle modeller med leaderboards og performance trends**")

# Load all training logs
logs_dir = "logs"
models_dir = "models"

all_models = []

if os.path.exists(logs_dir):
    for log_file in os.listdir(logs_dir):
        if log_file.endswith('_training.json'):
            try:
                with open(os.path.join(logs_dir, log_file), 'r') as f:
                    log_data = json.load(f)
                
                # Check if model file exists
                model_id = log_data.get('model_id', log_file.replace('_training.json', ''))
                model_file_pkl = os.path.join(models_dir, f"{model_id}.pkl")
                model_file_h5 = os.path.join(models_dir, f"{model_id}.h5")
                
                if os.path.exists(model_file_pkl) or os.path.exists(model_file_h5):
                    # Get metrics
                    metrics = log_data.get('final_metrics', log_data.get('metrics', {}))
                    data_stats = log_data.get('data_stats', log_data.get('data_info', {}))
                    
                    train_mae = metrics.get('train_mae', 0)
                    val_mae = metrics.get('val_mae', 0)
                    gen_gap = ((val_mae - train_mae) / train_mae * 100) if train_mae > 0 else 0
                    
                    all_models.append({
                        'model_id': model_id,
                        'model_type': log_data.get('model_type', 'unknown').upper(),
                        'symbol': log_data.get('symbol', 'N/A'),
                        'version': log_data.get('version', 1),
                        'description': log_data.get('description', ''),
                        'train_mae': train_mae,
                        'val_mae': val_mae,
                        'gen_gap': gen_gap,
                        'samples': data_stats.get('training_samples', 0),
                        'timestamp': log_data.get('timestamp', ''),
                        'deployed': log_data.get('deployed', False),
                        'training_time': log_data.get('training_time', 0)
                    })
            except Exception as e:
                continue

if not all_models:
    st.info("📭 Ingen modeller fundet. Træn nogle modeller først!")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(all_models)

# ========== SUMMARY STATS ==========
st.markdown("## 📊 Summary Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Models", len(df))

with col2:
    deployed_count = df['deployed'].sum()
    st.metric("Deployed", deployed_count)

with col3:
    avg_val_mae = df['val_mae'].mean()
    st.metric("Avg Val MAE", f"${avg_val_mae:.2f}")

with col4:
    best_val_mae = df['val_mae'].min()
    st.metric("Best Val MAE", f"${best_val_mae:.2f}")

with col5:
    symbols_count = df['symbol'].nunique()
    st.metric("Symbols", symbols_count)

st.divider()

# ========== LEADERBOARD ==========
st.markdown("## 🏆 Leaderboard - Top Performing Models")

# Filters for leaderboard
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    symbol_filter = st.selectbox("Filter Symbol", ["All"] + sorted(df['symbol'].unique().tolist()))

with col2:
    type_filter = st.selectbox("Filter Type", ["All"] + sorted(df['model_type'].unique().tolist()))

with col3:
    sort_by = st.selectbox("Sort By", ["Val MAE (Best)", "Gen Gap (Lowest)", "Training Samples (Most)"])

# Apply filters
filtered_df = df.copy()
if symbol_filter != "All":
    filtered_df = filtered_df[filtered_df['symbol'] == symbol_filter]
if type_filter != "All":
    filtered_df = filtered_df[filtered_df['model_type'] == type_filter]

# Sort
if sort_by == "Val MAE (Best)":
    filtered_df = filtered_df.sort_values('val_mae')
elif sort_by == "Gen Gap (Lowest)":
    filtered_df = filtered_df.sort_values('gen_gap')
else:
    filtered_df = filtered_df.sort_values('samples', ascending=False)

# Display top 10
st.markdown("### 🥇 Top 10 Models")

for i, row in filtered_df.head(10).iterrows():
    rank_emoji = ["🥇", "🥈", "🥉"][min(i, 2)] if i < 3 else f"#{i+1}"
    deployed_badge = "🚀 **DEPLOYED**" if row['deployed'] else ""
    
    with st.expander(f"{rank_emoji} **{row['model_type']} - {row['symbol']}** v{row['version']} - Val MAE: ${row['val_mae']:.2f} {deployed_badge}"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("**Performance:**")
            st.markdown(f"- Train MAE: ${row['train_mae']:.2f}")
            st.markdown(f"- Val MAE: ${row['val_mae']:.2f}")
            st.markdown(f"- Gen Gap: {row['gen_gap']:.1f}%")
        
        with col_b:
            st.markdown("**Configuration:**")
            st.markdown(f"- Training Samples: {row['samples']}")
            st.markdown(f"- Version: v{row['version']}")
            st.markdown(f"- Training Time: {row['training_time']:.1f}s")
        
        with col_c:
            st.markdown("**Details:**")
            if row['description']:
                st.markdown(f"- {row['description'][:60]}")
            st.markdown(f"- Timestamp: {row['timestamp'][:8]}")
            st.caption(f"Model ID: {row['model_id']}")

st.divider()

# ========== PERFORMANCE COMPARISON ==========
st.markdown("## 📊 Performance Comparison")

tab1, tab2, tab3 = st.tabs(["📉 By Symbol", "🤖 By Model Type", "📈 Trends Over Time"])

# TAB 1: By Symbol
with tab1:
    st.markdown("### Val MAE by Symbol and Model Type")
    
    # Group by symbol and model type
    symbol_comparison = df.groupby(['symbol', 'model_type'])['val_mae'].min().reset_index()
    
    fig_symbol = px.bar(
        symbol_comparison,
        x='symbol',
        y='val_mae',
        color='model_type',
        barmode='group',
        title='Best Val MAE per Symbol and Model Type',
        labels={'val_mae': 'Val MAE ($)', 'symbol': 'Symbol', 'model_type': 'Model Type'},
        text='val_mae'
    )
    
    fig_symbol.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig_symbol.update_layout(height=500)
    
    st.plotly_chart(fig_symbol, use_container_width=True)
    
    # Show best model per symbol
    st.markdown("### 🎯 Best Model per Symbol")
    
    best_per_symbol = df.loc[df.groupby('symbol')['val_mae'].idxmin()]
    
    for _, row in best_per_symbol.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            st.markdown(f"**{row['symbol']}**")
        
        with col2:
            st.markdown(f"{row['model_type']} v{row['version']}")
            if row['description']:
                st.caption(row['description'][:40])
        
        with col3:
            st.metric("Val MAE", f"${row['val_mae']:.2f}")
        
        with col4:
            if row['deployed']:
                st.success("🚀 Deployed")
            else:
                st.info("Not deployed")

# TAB 2: By Model Type
with tab2:
    st.markdown("### Performance by Model Type")
    
    # Box plot of Val MAE by model type
    fig_type = go.Figure()
    
    for model_type in df['model_type'].unique():
        type_data = df[df['model_type'] == model_type]
        
        fig_type.add_trace(go.Box(
            y=type_data['val_mae'],
            name=model_type,
            boxmean='sd'
        ))
    
    fig_type.update_layout(
        title='Val MAE Distribution by Model Type',
        yaxis_title='Val MAE ($)',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_type, use_container_width=True)
    
    # Stats table
    st.markdown("### 📊 Model Type Statistics")
    
    type_stats = df.groupby('model_type').agg({
        'val_mae': ['mean', 'min', 'max', 'std'],
        'gen_gap': 'mean',
        'model_id': 'count'
    }).round(2)
    
    type_stats.columns = ['Avg MAE', 'Best MAE', 'Worst MAE', 'Std Dev', 'Avg Gen Gap', 'Count']
    
    st.dataframe(type_stats, use_container_width=True)

# TAB 3: Trends Over Time
with tab3:
    st.markdown("### Performance Trends Over Time")
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
    df_sorted = df.sort_values('datetime')
    
    # Create timeline plot
    fig_timeline = go.Figure()
    
    for model_type in df['model_type'].unique():
        type_data = df_sorted[df_sorted['model_type'] == model_type]
        
        fig_timeline.add_trace(go.Scatter(
            x=type_data['datetime'],
            y=type_data['val_mae'],
            mode='lines+markers',
            name=model_type,
            text=type_data.apply(lambda r: f"{r['symbol']} v{r['version']}<br>${r['val_mae']:.2f}", axis=1),
            hovertemplate='%{text}<extra></extra>',
            marker=dict(size=8)
        ))
    
    fig_timeline.update_layout(
        title='Val MAE Over Time by Model Type',
        xaxis_title='Training Date',
        yaxis_title='Val MAE ($)',
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Show improvement over versions
    st.markdown("### 📈 Version Improvements")
    
    # Group by symbol + type and show version progression
    for symbol in df['symbol'].unique():
        st.markdown(f"#### {symbol}")
        
        for model_type in df[df['symbol'] == symbol]['model_type'].unique():
            versions = df[(df['symbol'] == symbol) & (df['model_type'] == model_type)].sort_values('version')
            
            if len(versions) > 1:
                st.markdown(f"**{model_type}:**")
                
                improvements = []
                for i in range(1, len(versions)):
                    prev_mae = versions.iloc[i-1]['val_mae']
                    curr_mae = versions.iloc[i]['val_mae']
                    improvement = ((prev_mae - curr_mae) / prev_mae * 100) if prev_mae > 0 else 0
                    improvements.append(improvement)
                
                avg_improvement = sum(improvements) / len(improvements)
                
                if avg_improvement > 0:
                    st.success(f"✅ Average improvement: {avg_improvement:.1f}% per version ({len(versions)} versions)")
                else:
                    st.warning(f"⚠️ Average change: {avg_improvement:.1f}% per version ({len(versions)} versions)")

st.divider()

# ========== QUICK ACTIONS ==========
st.markdown("## ⚡ Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🗑️ Cleanup Suggestions")
    
    # Find worst performers
    worst_models = df.nsmallest(5, 'val_mae', keep='last').tail(5)
    
    if len(worst_models) > 0:
        st.warning(f"**{len(worst_models)} models** could be cleaned up:")
        for _, row in worst_models.iterrows():
            st.markdown(f"- {row['model_type']} {row['symbol']} v{row['version']}: ${row['val_mae']:.2f}")
        
        st.info("💡 Go to Model Management → Bulk Cleanup")

with col2:
    st.markdown("### 🚀 Deployment Suggestions")
    
    # Find best models not deployed
    not_deployed_best = df[df['deployed'] == False].nsmallest(5, 'val_mae')
    
    if len(not_deployed_best) > 0:
        st.success(f"**{len(not_deployed_best)} top models** not deployed:")
        for _, row in not_deployed_best.head(3).iterrows():
            st.markdown(f"- {row['model_type']} {row['symbol']} v{row['version']}: ${row['val_mae']:.2f}")
        
        st.info("💡 Go to Model Management → Deploy")

with col3:
    st.markdown("### 🔄 Retraining Suggestions")
    
    # Find models with high gen gap
    high_gen_gap = df[df['gen_gap'] > 50].nsmallest(5, 'val_mae')
    
    if len(high_gen_gap) > 0:
        st.warning(f"**{len(high_gen_gap)} models** with high overfitting:")
        for _, row in high_gen_gap.head(3).iterrows():
            st.markdown(f"- {row['model_type']} {row['symbol']} v{row['version']}: {row['gen_gap']:.0f}% gap")
        
        st.info("💡 Go to ML Mentor for recommendations")
