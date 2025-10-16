"""
ML Mentor Retrain Module
Applies ML Mentor recommendations and retrains models
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from agent_interactive import (
    train_and_save_rf,
    train_and_save_xgboost,
    train_and_save_lstm,
    train_and_save_prophet,
    calculate_model_metrics
)

def parse_recommendation(recommendation_text):
    """
    Parse a recommendation string into actionable parameters
    
    Args:
        recommendation_text: String with recommendation (e.g., "Increase n_estimators to 200")
    
    Returns:
        dict: Parsed parameters {'param_name': new_value}
    """
    params = {}
    
    # Common patterns to parse
    patterns = {
        'n_estimators': r'n_estimators.*?(\d+)',
        'max_depth': r'max_depth.*?(\d+)',
        'learning_rate': r'learning_rate.*?(0\.\d+)',
        'sequence_length': r'sequence.*?length.*?(\d+)',
        'lstm_units': r'lstm.*?units.*?(\d+)',
        'epochs': r'epochs.*?(\d+)',
        'batch_size': r'batch.*?size.*?(\d+)',
    }
    
    import re
    for param_name, pattern in patterns.items():
        match = re.search(pattern, recommendation_text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Convert to appropriate type
            if '.' in value:
                params[param_name] = float(value)
            else:
                params[param_name] = int(value)
    
    return params

def apply_recommendation_and_retrain(recommendation, model_id, symbol, current_params, model_type=None):
    """
    Apply a recommendation and retrain the model
    
    Args:
        recommendation: Dict with recommendation details
        model_id: Current model ID
        symbol: Stock symbol
        current_params: Dict with current model parameters
        model_type: Model type string (rf, xgboost, lstm, prophet) - optional, will be inferred if not provided
    
    Returns:
        dict: {
            'success': bool,
            'new_model_id': str,
            'old_metrics': dict,
            'new_metrics': dict,
            'improvement': dict
        }
    """
    
    try:
        # Parse recommendation
        recommendation_text = recommendation.get('recommendation', '')
        suggested_params = parse_recommendation(recommendation_text)
        
        # Merge with current params
        new_params = current_params.copy()
        new_params.update(suggested_params)
        
        # Determine model type if not provided
        if not model_type:
            # Try to extract from model_id or filename
            if 'rf' in model_id.lower():
                model_type = 'rf'
            elif 'xgb' in model_id.lower():
                model_type = 'xgboost'
            elif 'lstm' in model_id.lower():
                model_type = 'lstm'
            elif 'prophet' in model_id.lower():
                model_type = 'prophet'
        
        # Convert model_type to display format
        if model_type:
            if model_type.lower() == 'rf':
                model_type_display = 'Random Forest'
            elif model_type.lower() == 'xgboost':
                model_type_display = 'XGBoost'
            elif model_type.lower() == 'lstm':
                model_type_display = 'LSTM'
            elif model_type.lower() == 'prophet':
                model_type_display = 'Prophet'
            else:
                model_type_display = model_type
        else:
            return {
                'success': False,
                'error': f'Could not determine model type from model_id: {model_id}'
            }
        
        # Get fresh data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="2y")
        
        if data.empty:
            return {
                'success': False,
                'error': f'No data available for {symbol}'
            }
        
        # Train new model based on type
        with st.spinner(f"Training new {model_type_display} model with recommended parameters..."):
            
            if model_type_display == 'Random Forest':
                n_estimators = new_params.get('n_estimators', 100)
                max_depth = new_params.get('max_depth', 10)
                new_model_id = train_and_save_rf(
                    data, 
                    symbol, 
                    n_estimators=n_estimators,
                    max_depth=max_depth
                )
            
            elif model_type_display == 'XGBoost':
                n_estimators = new_params.get('n_estimators', 100)
                learning_rate = new_params.get('learning_rate', 0.1)
                max_depth = new_params.get('max_depth', 5)
                new_model_id = train_and_save_xgboost(
                    data,
                    symbol,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )
            
            elif model_type_display == 'LSTM':
                sequence_length = new_params.get('sequence_length', 60)
                lstm_units = new_params.get('lstm_units', 50)
                epochs = new_params.get('epochs', 50)
                batch_size = new_params.get('batch_size', 32)
                new_model_id = train_and_save_lstm(
                    data,
                    symbol,
                    sequence_length=sequence_length,
                    lstm_units=lstm_units,
                    epochs=epochs,
                    batch_size=batch_size
                )
            
            elif model_type_display == 'Prophet':
                new_model_id = train_and_save_prophet(data, symbol)
            
            else:
                return {
                    'success': False,
                    'error': f'Unsupported model type: {model_type_display}'
                }
        
        # Calculate old metrics (from current_params which is the model's metadata)
        old_metrics = {
            'mae': current_params.get('val_mae') or current_params.get('mae', 0),
            'rmse': current_params.get('val_rmse') or current_params.get('rmse', 0),
            'r2': current_params.get('r2_score', 0)
        }
        
        # Calculate new metrics - train_and_save_* returns a dict
        if isinstance(new_model_id, dict):
            # Extract metadata from return value
            new_metadata = new_model_id.get('metadata', {})
            new_metrics = {
                'mae': new_metadata.get('val_mae', 0),
                'rmse': new_metadata.get('val_rmse', 0),
                'r2': new_metadata.get('r2_score', 0)
            }
            # Extract actual model ID from filepath
            filepath = new_model_id.get('filepath', '')
            import os
            new_model_id = os.path.basename(filepath).replace('.pkl', '') if filepath else 'unknown'
        else:
            new_metrics = old_metrics
        
        # Calculate improvement
        improvement = {
            'mae': ((old_metrics['mae'] - new_metrics['mae']) / old_metrics['mae'] * 100) if old_metrics['mae'] > 0 else 0,
            'rmse': ((old_metrics['rmse'] - new_metrics['rmse']) / old_metrics['rmse'] * 100) if old_metrics['rmse'] > 0 else 0,
            'r2': ((new_metrics['r2'] - old_metrics['r2']) / abs(old_metrics['r2']) * 100) if old_metrics['r2'] != 0 else 0
        }
        
        return {
            'success': True,
            'new_model_id': new_model_id,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvement': improvement,
            'params_used': new_params
        }
    
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def batch_apply_recommendations(recommendations, model_id, symbol, current_params):
    """
    Apply multiple recommendations in sequence
    
    Args:
        recommendations: List of recommendation dicts
        model_id: Starting model ID
        symbol: Stock symbol
        current_params: Starting parameters
    
    Returns:
        list: Results for each recommendation
    """
    results = []
    current_id = model_id
    current_p = current_params.copy()
    
    for i, rec in enumerate(recommendations):
        st.write(f"Applying recommendation {i+1}/{len(recommendations)}...")
        
        result = apply_recommendation_and_retrain(rec, current_id, symbol, current_p)
        results.append(result)
        
        if result['success']:
            # Update for next iteration
            current_id = result['new_model_id']
            current_p = result['params_used']
        else:
            st.warning(f"Recommendation {i+1} failed: {result.get('error')}")
            break
    
    return results
