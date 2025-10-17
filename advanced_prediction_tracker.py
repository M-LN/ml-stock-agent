"""
Advanced Prediction Tracker with Auto-Update, Alerts, and Confidence Intervals
"""

import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from prediction_tracker import PREDICTIONS_DIR, get_all_predictions
import schedule
import time
import threading

# Alerts configuration
ALERTS_DIR = "alerts"
if not os.path.exists(ALERTS_DIR):
    os.makedirs(ALERTS_DIR)

# Performance thresholds for alerts
ALERT_THRESHOLDS = {
    'poor_mae': 50.0,  # MAE above $50
    'poor_mape': 10.0,  # MAPE above 10%
    'direction_accuracy': 40.0,  # Direction accuracy below 40%
    'consecutive_failures': 3  # 3 bad predictions in a row
}


def generate_multi_day_predictions(model, data, window=30, horizon=5):
    """
    Generate predictions for each day in the horizon
    
    Args:
        model: Trained model
        data: Historical price data
        window: Window size
        horizon: Number of days to predict
    
    Returns:
        list: Predictions for each day [day1, day2, ..., dayN]
    """
    close_prices = data['Close'].values.flatten()
    predictions = []
    
    # Recursive prediction - use previous predictions as input
    current_window = close_prices[-window:].copy()
    
    for i in range(horizon):
        # Predict next day
        X = current_window.reshape(1, -1)
        pred = float(model.predict(X)[0])
        predictions.append(pred)
        
        # Update window for next prediction
        current_window = np.append(current_window[1:], pred)
    
    return predictions


def calculate_confidence_intervals(model, X_val, y_val, confidence=0.95):
    """
    Calculate confidence intervals for predictions
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        confidence: Confidence level (default 0.95 for 95%)
    
    Returns:
        dict: Upper and lower bounds, std
    """
    try:
        # Get predictions on validation set
        predictions = model.predict(X_val)
        
        # Calculate residuals
        residuals = y_val - predictions
        
        # Calculate standard deviation of residuals
        std_residual = np.std(residuals)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Margin of error
        margin = z_score * std_residual
        
        return {
            'std': float(std_residual),
            'margin': float(margin),
            'lower_multiplier': -z_score,
            'upper_multiplier': z_score
        }
    except Exception as e:
        return {
            'std': 0,
            'margin': 0,
            'lower_multiplier': -1.96,
            'upper_multiplier': 1.96
        }


def save_prediction_with_confidence(model_id, symbol, model_type, prediction_date, 
                                   predictions, horizon, confidence_data=None, metadata=None):
    """
    Save prediction with confidence intervals
    
    Args:
        model_id: Model identifier
        symbol: Stock symbol
        model_type: Type of model
        prediction_date: Date of prediction
        predictions: List of predictions for each day
        horizon: Number of days
        confidence_data: Dict with confidence interval data
        metadata: Additional metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_id = f"{model_id}_{timestamp}"
    
    # Calculate target dates
    target_dates = []
    current_date = pd.Timestamp(prediction_date)
    for i in range(1, horizon + 1):
        next_date = current_date + timedelta(days=i)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        target_dates.append(next_date.strftime("%Y-%m-%d"))
    
    # Calculate confidence bounds if available
    confidence_bounds = None
    if confidence_data and 'std' in confidence_data:
        std = confidence_data['std']
        upper_mult = confidence_data.get('upper_multiplier', 1.96)
        lower_mult = confidence_data.get('lower_multiplier', -1.96)
        
        confidence_bounds = {
            'upper': [float(p + upper_mult * std) for p in predictions],
            'lower': [float(p + lower_mult * std) for p in predictions],
            'std': float(std)
        }
    
    prediction_data = {
        'prediction_id': prediction_id,
        'model_id': model_id,
        'symbol': symbol,
        'model_type': model_type,
        'prediction_date': prediction_date,
        'timestamp': timestamp,
        'horizon': horizon,
        'predictions': [float(p) for p in predictions],
        'target_dates': target_dates,
        'actual_prices': [None] * len(predictions),
        'confidence_bounds': confidence_bounds,
        'metadata': metadata or {},
        'status': 'pending',
        'alerts_triggered': []
    }
    
    # Save to file
    filepath = os.path.join(PREDICTIONS_DIR, f"{prediction_id}.json")
    with open(filepath, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    return prediction_id


def check_prediction_performance(pred_data):
    """
    Check if prediction performance triggers any alerts
    
    Returns:
        list: List of alert messages
    """
    alerts = []
    
    # Check if we have actual prices
    actual_prices = pred_data.get('actual_prices', [])
    if not any(a is not None for a in actual_prices):
        return alerts
    
    predictions = pred_data['predictions']
    
    # Calculate metrics
    valid_pairs = [(p, a) for p, a in zip(predictions, actual_prices) if a is not None]
    if not valid_pairs:
        return alerts
    
    preds, actuals = zip(*valid_pairs)
    
    # MAE check
    mae = np.mean([abs(p - a) for p, a in valid_pairs])
    if mae > ALERT_THRESHOLDS['poor_mae']:
        alerts.append(f"⚠️ High MAE: ${mae:.2f} (threshold: ${ALERT_THRESHOLDS['poor_mae']})")
    
    # MAPE check
    mape = np.mean([abs(p - a) / a * 100 for p, a in valid_pairs if a != 0])
    if mape > ALERT_THRESHOLDS['poor_mape']:
        alerts.append(f"⚠️ High MAPE: {mape:.1f}% (threshold: {ALERT_THRESHOLDS['poor_mape']}%)")
    
    # Direction accuracy (for multi-day predictions)
    if len(valid_pairs) > 1:
        correct_directions = 0
        for i in range(len(valid_pairs) - 1):
            pred_dir = 1 if preds[i+1] > preds[i] else -1
            actual_dir = 1 if actuals[i+1] > actuals[i] else -1
            if pred_dir == actual_dir:
                correct_directions += 1
        
        dir_accuracy = (correct_directions / (len(valid_pairs) - 1)) * 100
        if dir_accuracy < ALERT_THRESHOLDS['direction_accuracy']:
            alerts.append(f"⚠️ Poor direction accuracy: {dir_accuracy:.1f}% (threshold: {ALERT_THRESHOLDS['direction_accuracy']}%)")
    
    return alerts


def update_predictions_with_alerts():
    """
    Update predictions and trigger alerts for poor performance
    
    Returns:
        dict: Update summary with alerts
    """
    from prediction_tracker import update_predictions_with_actuals
    
    # Update all predictions
    updated_count = update_predictions_with_actuals()
    
    # Check for alerts
    all_alerts = []
    model_alerts = {}
    
    predictions = get_all_predictions()
    
    for pred in predictions:
        if pred.get('status') == 'complete':
            alerts = check_prediction_performance(pred)
            
            if alerts:
                model_id = pred['model_id']
                symbol = pred['symbol']
                
                alert_entry = {
                    'model_id': model_id,
                    'symbol': symbol,
                    'prediction_id': pred['prediction_id'],
                    'prediction_date': pred['prediction_date'],
                    'alerts': alerts,
                    'mae': pred.get('mae', 0),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                all_alerts.append(alert_entry)
                
                # Group by model
                if model_id not in model_alerts:
                    model_alerts[model_id] = []
                model_alerts[model_id].append(alert_entry)
                
                # Update prediction with alerts
                pred['alerts_triggered'] = alerts
                filepath = os.path.join(PREDICTIONS_DIR, f"{pred['prediction_id']}.json")
                with open(filepath, 'w') as f:
                    json.dump(pred, f, indent=2)
    
    # Save alerts summary
    if all_alerts:
        alerts_file = os.path.join(ALERTS_DIR, f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(alerts_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'alerts_count': len(all_alerts),
                'alerts': all_alerts
            }, f, indent=2)
    
    return {
        'updated_count': updated_count,
        'alerts_count': len(all_alerts),
        'alerts': all_alerts,
        'model_alerts': model_alerts
    }


def suggest_model_retirement():
    """
    Analyze model performance and suggest retirements
    
    Returns:
        list: Models recommended for retirement
    """
    predictions = get_all_predictions()
    
    # Group by model
    model_performance = {}
    
    for pred in predictions:
        if pred.get('status') != 'complete':
            continue
        
        model_id = pred['model_id']
        mae = pred.get('mae')
        
        if mae is None:
            continue
        
        if model_id not in model_performance:
            model_performance[model_id] = {
                'model_id': model_id,
                'symbol': pred['symbol'],
                'model_type': pred['model_type'],
                'predictions': [],
                'mae_values': [],
                'alerts_count': 0
            }
        
        model_performance[model_id]['predictions'].append(pred)
        model_performance[model_id]['mae_values'].append(mae)
        model_performance[model_id]['alerts_count'] += len(pred.get('alerts_triggered', []))
    
    # Analyze and suggest retirements
    retirement_suggestions = []
    
    for model_id, perf in model_performance.items():
        if len(perf['mae_values']) < 3:
            continue  # Need at least 3 predictions to judge
        
        avg_mae = np.mean(perf['mae_values'])
        recent_mae = np.mean(perf['mae_values'][-3:])  # Last 3 predictions
        trend = recent_mae - avg_mae
        
        # Criteria for retirement
        reasons = []
        retirement_score = 0
        
        # High average MAE
        if avg_mae > ALERT_THRESHOLDS['poor_mae']:
            reasons.append(f"High avg MAE: ${avg_mae:.2f}")
            retirement_score += 3
        
        # Worsening trend
        if trend > 10:
            reasons.append(f"Performance degrading (trend: +${trend:.2f})")
            retirement_score += 2
        
        # Multiple alerts
        if perf['alerts_count'] >= ALERT_THRESHOLDS['consecutive_failures']:
            reasons.append(f"{perf['alerts_count']} alerts triggered")
            retirement_score += 2
        
        # Recent poor performance
        if recent_mae > ALERT_THRESHOLDS['poor_mae'] * 1.5:
            reasons.append(f"Recent MAE very high: ${recent_mae:.2f}")
            retirement_score += 3
        
        if retirement_score >= 5:  # Threshold for retirement suggestion
            retirement_suggestions.append({
                'model_id': model_id,
                'symbol': perf['symbol'],
                'model_type': perf['model_type'],
                'retirement_score': retirement_score,
                'reasons': reasons,
                'avg_mae': float(avg_mae),
                'recent_mae': float(recent_mae),
                'prediction_count': len(perf['predictions']),
                'alerts_count': perf['alerts_count']
            })
    
    # Sort by retirement score
    retirement_suggestions.sort(key=lambda x: x['retirement_score'], reverse=True)
    
    return retirement_suggestions


def schedule_daily_update():
    """
    Schedule daily automatic update
    This should run as a background task
    """
    def job():
        try:
            print(f"[{datetime.now()}] Running scheduled prediction update...")
            result = update_predictions_with_alerts()
            print(f"[{datetime.now()}] Update complete: {result['updated_count']} predictions updated, {result['alerts_count']} alerts")
        except Exception as e:
            print(f"[{datetime.now()}] Scheduled update failed: {e}")
    
    # Schedule daily at 18:00 (after market close)
    schedule.every().day.at("18:00").do(job)
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def start_scheduler_background():
    """
    Start the scheduler in a background thread
    """
    scheduler_thread = threading.Thread(target=schedule_daily_update, daemon=True)
    scheduler_thread.start()
    return scheduler_thread
