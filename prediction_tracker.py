"""
Prediction Tracker - Track model predictions over time and compare with actual results
"""

import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

PREDICTIONS_DIR = "predictions"
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)


def save_prediction(model_id, symbol, model_type, prediction_date, predictions, horizon, metadata=None):
    """
    Save model predictions for future tracking
    
    Args:
        model_id: Unique model identifier
        symbol: Stock symbol
        model_type: Type of model (rf, xgboost, lstm)
        prediction_date: Date when prediction was made
        predictions: List of predicted prices [day1, day2, ..., dayN]
        horizon: Number of days predicted
        metadata: Additional model info
    
    Returns:
        str: Prediction ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_id = f"{model_id}_{timestamp}"
    
    # Calculate target dates
    target_dates = []
    current_date = pd.Timestamp(prediction_date)
    for i in range(1, horizon + 1):
        # Skip weekends
        next_date = current_date + timedelta(days=i)
        while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_date += timedelta(days=1)
        target_dates.append(next_date.strftime("%Y-%m-%d"))
    
    prediction_data = {
        'prediction_id': prediction_id,
        'model_id': model_id,
        'symbol': symbol,
        'model_type': model_type,
        'prediction_date': prediction_date,
        'timestamp': timestamp,
        'horizon': horizon,
        'predictions': predictions if isinstance(predictions, list) else [float(p) for p in predictions],
        'target_dates': target_dates,
        'actual_prices': [None] * len(predictions),  # Will be filled later
        'metadata': metadata or {},
        'status': 'pending'  # pending, partial, complete
    }
    
    # Save to file
    filepath = os.path.join(PREDICTIONS_DIR, f"{prediction_id}.json")
    with open(filepath, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    return prediction_id


def update_predictions_with_actuals():
    """
    Update all pending predictions with actual prices from yfinance
    Should be run periodically (daily)
    """
    updated_count = 0
    
    if not os.path.exists(PREDICTIONS_DIR):
        return updated_count
    
    for filename in os.listdir(PREDICTIONS_DIR):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(PREDICTIONS_DIR, filename)
        
        try:
            with open(filepath, 'r') as f:
                pred_data = json.load(f)
            
            # Skip if already complete
            if pred_data.get('status') == 'complete':
                continue
            
            symbol = pred_data['symbol']
            target_dates = pred_data['target_dates']
            
            # Get last target date
            last_date = pd.Timestamp(target_dates[-1])
            today = pd.Timestamp.now()
            
            # Check if we can get actual prices yet
            if today < last_date:
                # Still waiting - update what we have
                status = 'pending'
            else:
                status = 'complete'
            
            # Fetch actual prices
            try:
                # Get data from prediction date to now
                start_date = pred_data['prediction_date']
                data = yf.download(symbol, start=start_date, end=today, progress=False)
                
                if not data.empty:
                    actual_prices = []
                    for target_date in target_dates:
                        try:
                            # Try exact date first
                            if target_date in data.index:
                                actual_prices.append(float(data.loc[target_date]['Close']))
                            else:
                                # Find nearest date
                                target_ts = pd.Timestamp(target_date)
                                nearest_idx = data.index.get_indexer([target_ts], method='nearest')[0]
                                if nearest_idx >= 0:
                                    actual_prices.append(float(data.iloc[nearest_idx]['Close']))
                                else:
                                    actual_prices.append(None)
                        except:
                            actual_prices.append(None)
                    
                    # Update prediction data
                    pred_data['actual_prices'] = actual_prices
                    pred_data['status'] = status
                    pred_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Calculate errors if we have actuals
                    predictions = pred_data['predictions']
                    errors = []
                    for pred, actual in zip(predictions, actual_prices):
                        if actual is not None:
                            errors.append(abs(pred - actual))
                        else:
                            errors.append(None)
                    
                    pred_data['errors'] = errors
                    
                    if any(e is not None for e in errors):
                        valid_errors = [e for e in errors if e is not None]
                        pred_data['mae'] = float(np.mean(valid_errors))
                        pred_data['rmse'] = float(np.sqrt(np.mean([e**2 for e in valid_errors])))
                    
                    # Save updated data
                    with open(filepath, 'w') as f:
                        json.dump(pred_data, f, indent=2)
                    
                    updated_count += 1
            
            except Exception as e:
                print(f"Error updating {filename}: {e}")
                continue
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    
    return updated_count


def get_all_predictions(symbol=None, model_id=None, status=None):
    """
    Get all predictions, optionally filtered
    
    Args:
        symbol: Filter by symbol
        model_id: Filter by model ID
        status: Filter by status (pending, partial, complete)
    
    Returns:
        list: List of prediction dicts
    """
    predictions = []
    
    if not os.path.exists(PREDICTIONS_DIR):
        return predictions
    
    for filename in os.listdir(PREDICTIONS_DIR):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(PREDICTIONS_DIR, filename)
        
        try:
            with open(filepath, 'r') as f:
                pred_data = json.load(f)
            
            # Apply filters
            if symbol and pred_data.get('symbol') != symbol:
                continue
            if model_id and pred_data.get('model_id') != model_id:
                continue
            if status and pred_data.get('status') != status:
                continue
            
            predictions.append(pred_data)
        
        except Exception as e:
            continue
    
    return predictions


def get_model_live_performance(model_id, days=30):
    """
    Get live performance metrics for a model over recent predictions
    
    Args:
        model_id: Model ID
        days: Number of days to look back
    
    Returns:
        dict: Performance metrics
    """
    predictions = get_all_predictions(model_id=model_id)
    
    # Filter to recent predictions
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_preds = []
    
    for pred in predictions:
        pred_date = datetime.strptime(pred['timestamp'], "%Y%m%d_%H%M%S")
        if pred_date >= cutoff_date:
            recent_preds.append(pred)
    
    if not recent_preds:
        return None
    
    # Calculate aggregate metrics
    all_errors = []
    all_predictions = []
    all_actuals = []
    
    for pred in recent_preds:
        predictions_list = pred.get('predictions', [])
        actuals_list = pred.get('actual_prices', [])
        
        for p, a in zip(predictions_list, actuals_list):
            if a is not None:
                all_predictions.append(p)
                all_actuals.append(a)
                all_errors.append(abs(p - a))
    
    if not all_errors:
        return {
            'model_id': model_id,
            'prediction_count': len(recent_preds),
            'status': 'pending',
            'message': 'No actual prices available yet'
        }
    
    # Calculate metrics
    mae = np.mean(all_errors)
    rmse = np.sqrt(np.mean([e**2 for e in all_errors]))
    mape = np.mean([abs(p - a) / a * 100 for p, a in zip(all_predictions, all_actuals) if a != 0])
    
    # Direction accuracy (did we predict up/down correctly?)
    correct_directions = 0
    total_directions = 0
    
    for pred in recent_preds:
        preds = pred.get('predictions', [])
        actuals = pred.get('actual_prices', [])
        
        if len(preds) > 1 and len(actuals) > 1:
            # Compare first and last
            if actuals[0] is not None and actuals[-1] is not None:
                pred_direction = 1 if preds[-1] > preds[0] else -1
                actual_direction = 1 if actuals[-1] > actuals[0] else -1
                
                if pred_direction == actual_direction:
                    correct_directions += 1
                total_directions += 1
    
    direction_accuracy = (correct_directions / total_directions * 100) if total_directions > 0 else 0
    
    return {
        'model_id': model_id,
        'prediction_count': len(recent_preds),
        'data_points': len(all_errors),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'direction_accuracy': float(direction_accuracy),
        'avg_prediction': float(np.mean(all_predictions)),
        'avg_actual': float(np.mean(all_actuals)),
        'status': 'active'
    }


def get_symbol_live_performance(symbol, days=30):
    """
    Get live performance for all models tracking a symbol
    
    Args:
        symbol: Stock symbol
        days: Number of days to look back
    
    Returns:
        dict: Performance by model
    """
    predictions = get_all_predictions(symbol=symbol)
    
    # Group by model_id
    models = {}
    for pred in predictions:
        model_id = pred['model_id']
        if model_id not in models:
            models[model_id] = []
        models[model_id].append(pred)
    
    # Get performance for each model
    performance = {}
    for model_id in models:
        perf = get_model_live_performance(model_id, days=days)
        if perf:
            performance[model_id] = perf
    
    return performance
