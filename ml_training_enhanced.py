"""
Enhanced ML Training Functions with Feature Engineering
Forbedrede træningsfunktioner der bruger tekniske indikatorer
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import os
import pickle

from feature_engineering import create_features, get_feature_columns, normalize_features


def prepare_ml_data_with_features(data, window=30, horizon=1, use_features=True):
    """
    Forbereder data til ML træning med features.
    
    Args:
        data: DataFrame med OHLCV data
        window: Lookback window
        horizon: Forecast horizon
        use_features: Om tekniske indikatorer skal bruges
    
    Returns:
        X_train, X_val, y_train, y_val, feature_names, scaler_params
    """
    if use_features:
        # Create technical indicators
        data_with_features = create_features(data, include_volume=True)
        feature_cols = get_feature_columns(data_with_features, exclude_ohlcv=True)
        
        # Normalize features
        data_normalized, scaler_params = normalize_features(
            data_with_features, 
            feature_cols, 
            method='standard'  # StandardScaler for tree-based models
        )
        
        # Get all feature columns for training
        all_features = feature_cols + ['Close']  # Include target
        data_for_ml = data_normalized[all_features].values
        
    else:
        # Use only close prices (legacy mode)
        data_for_ml = data['Close'].values.reshape(-1, 1)
        feature_cols = ['Close']
        scaler_params = None
    
    # Create sequences
    X, y = [], []
    
    if use_features:
        # For features: use all features except Close as input
        n_features = data_for_ml.shape[1] - 1  # Exclude Close from features
        
        for i in range(window, len(data_for_ml) - horizon + 1):
            # Get window of features (excluding Close from input)
            window_features = data_for_ml[i-window:i, :-1]  # All except last column (Close)
            
            # Flatten to 1D
            X.append(window_features.flatten())
            
            # Target is the Close price at horizon
            y.append(data_with_features['Close'].iloc[i+horizon-1])
        
        feature_names = [f"{feat}_t-{j}" for j in range(window, 0, -1) for feat in feature_cols[:-1]]
        
    else:
        # Legacy: only close prices
        for i in range(window, len(data_for_ml) - horizon + 1):
            X.append(data_for_ml[i-window:i].flatten())
            y.append(data['Close'].iloc[i+horizon-1])
        
        feature_names = [f"Close_t-{j}" for j in range(window, 0, -1)]
    
    X = np.array(X)
    y = np.array(y)
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val, feature_names, scaler_params


def train_and_save_rf_v2(data, symbol, n_estimators=100, max_depth=10, 
                         window=30, horizon=1, use_features=True):
    """
    Træn Random Forest model med feature engineering (Version 2).
    
    Args:
        data: Historical stock data
        symbol: Stock symbol
        n_estimators: Number of trees
        max_depth: Max tree depth
        window: Lookback window
        horizon: Forecast horizon
        use_features: Whether to use technical indicators (recommended: True)
    
    Returns:
        model_id if successful, None otherwise
    """
    try:
        print(f"🌲 Træner Random Forest {'med features' if use_features else 'legacy'} for {symbol}...")
        
        # Prepare data with features
        X_train, X_val, y_train, y_val, feature_names, scaler_params = prepare_ml_data_with_features(
            data, window, horizon, use_features
        )
        
        # Check if we have valid data
        if len(X_train) == 0 or (hasattr(X_train, 'shape') and len(X_train.shape) < 2):
            print(f"   ❌ Insufficient data after preparation. Need more historical data.")
            return None
        
        print(f"   Training samples: {len(X_train)}, Features: {X_train.shape[1] if len(X_train.shape) > 1 else 0}")
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            min_samples_split=5,  # Prevent overfitting
            min_samples_leaf=2
        )
        rf_model.fit(X_train, y_train)
        
        # Calculate performance
        train_predictions = rf_model.predict(X_train)
        val_predictions = rf_model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
        val_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100
        
        # Calculate Directional Accuracy
        if len(y_train) > 1 and len(train_predictions) > 1:
            train_direction_acc = np.mean(
                np.sign(y_train[1:] - y_train[:-1]) == np.sign(train_predictions[1:] - train_predictions[:-1])
            ) * 100
            val_direction_acc = np.mean(
                np.sign(y_val[1:] - y_val[:-1]) == np.sign(val_predictions[1:] - val_predictions[:-1])
            ) * 100
        else:
            train_direction_acc = 0.0
            val_direction_acc = 0.0
        
        print(f"   ✅ Val MAE: ${val_mae:.2f}, MAPE: {val_mape:.2f}%, Dir Acc: {val_direction_acc:.1f}%")
        
        # Get top features
        if use_features and len(feature_names) == len(rf_model.feature_importances_):
            feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_features = []
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"rf_v2_{symbol}_{timestamp}"
        
        MODEL_DIR = "saved_models"
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        
        model_package = {
            'model': rf_model,
            'model_type': 'rf',
            'model_version': 'v2_with_features' if use_features else 'v1_legacy',
            'symbol': symbol,
            'window': window,
            'horizon': horizon,
            'use_features': use_features,
            'feature_names': feature_names,
            'scaler_params': scaler_params,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_mape': float(train_mape),
            'train_direction_acc': float(train_direction_acc),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_mape': float(val_mape),
            'val_direction_acc': float(val_direction_acc),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'top_features': top_features[:10] if top_features else [],
            'trained_at': datetime.now().isoformat(),
            'data_period': f"{data.index[0]} to {data.index[-1]}"
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save metadata
        metadata_path = os.path.join(MODEL_DIR, f"{model_id}_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            # Convert to JSON-serializable format
            metadata = {k: v for k, v in model_package.items() if k != 'model' and k != 'scaler_params'}
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Model saved: {model_id}")
        
        return model_id
        
    except Exception as e:
        print(f"   ❌ Error training RF: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_and_save_xgboost_v2(data, symbol, n_estimators=100, max_depth=6, 
                               learning_rate=0.1, window=30, horizon=1, use_features=True):
    """
    Træn XGBoost model med feature engineering (Version 2).
    
    Args:
        data: Historical stock data
        symbol: Stock symbol
        n_estimators: Number of boosting rounds
        max_depth: Max tree depth
        learning_rate: Learning rate
        window: Lookback window
        horizon: Forecast horizon
        use_features: Whether to use technical indicators (recommended: True)
    
    Returns:
        model_id if successful, None otherwise
    """
    try:
        print(f"⚡ Træner XGBoost {'med features' if use_features else 'legacy'} for {symbol}...")
        
        # Prepare data with features
        X_train, X_val, y_train, y_val, feature_names, scaler_params = prepare_ml_data_with_features(
            data, window, horizon, use_features
        )
        
        # Check if we have valid data
        if len(X_train) == 0 or (hasattr(X_train, 'shape') and len(X_train.shape) < 2):
            print(f"   ❌ Insufficient data after preparation. Need more historical data.")
            return None
        
        print(f"   Training samples: {len(X_train)}, Features: {X_train.shape[1] if len(X_train.shape) > 1 else 0}")
        
        # Train model with early stopping
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20  # NEW: Early stopping
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate performance
        train_predictions = xgb_model.predict(X_train)
        val_predictions = xgb_model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        # Calculate MAPE
        train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
        val_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100
        
        # Calculate Directional Accuracy
        if len(y_train) > 1 and len(train_predictions) > 1:
            train_direction_acc = np.mean(
                np.sign(y_train[1:] - y_train[:-1]) == np.sign(train_predictions[1:] - train_predictions[:-1])
            ) * 100
            val_direction_acc = np.mean(
                np.sign(y_val[1:] - y_val[:-1]) == np.sign(val_predictions[1:] - val_predictions[:-1])
            ) * 100
        else:
            train_direction_acc = 0.0
            val_direction_acc = 0.0
        
        print(f"   ✅ Val MAE: ${val_mae:.2f}, MAPE: {val_mape:.2f}%, Dir Acc: {val_direction_acc:.1f}%")
        
        # Get top features
        if use_features and len(feature_names) == len(xgb_model.feature_importances_):
            feature_importance = dict(zip(feature_names, xgb_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_features = []
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"xgb_v2_{symbol}_{timestamp}"
        
        MODEL_DIR = "saved_models"
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        
        model_package = {
            'model': xgb_model,
            'model_type': 'xgboost',
            'model_version': 'v2_with_features' if use_features else 'v1_legacy',
            'symbol': symbol,
            'window': window,
            'horizon': horizon,
            'use_features': use_features,
            'feature_names': feature_names,
            'scaler_params': scaler_params,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_mape': float(train_mape),
            'train_direction_acc': float(train_direction_acc),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_mape': float(val_mape),
            'val_direction_acc': float(val_direction_acc),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'top_features': top_features[:10] if top_features else [],
            'trained_at': datetime.now().isoformat(),
            'data_period': f"{data.index[0]} to {data.index[-1]}"
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save metadata
        metadata_path = os.path.join(MODEL_DIR, f"{model_id}_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            metadata = {k: v for k, v in model_package.items() if k != 'model' and k != 'scaler_params'}
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Model saved: {model_id}")
        
        return model_id
        
    except Exception as e:
        print(f"   ❌ Error training XGBoost: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test enhanced training
    import yfinance as yf
    
    print("Testing Enhanced ML Training...")
    
    # Download test data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    
    print(f"\nData shape: {data.shape}")
    
    # Test Random Forest with features
    print("\n" + "="*60)
    print("Testing Random Forest V2 with Features")
    print("="*60)
    model_id = train_and_save_rf_v2(
        data, 
        symbol="AAPL",
        n_estimators=50,  # Mindre for hurtig test
        max_depth=10,
        window=30,
        horizon=1,
        use_features=True
    )
    
    if model_id:
        print(f"\n✅ RF model trained successfully: {model_id}")
    
    # Test XGBoost with features
    print("\n" + "="*60)
    print("Testing XGBoost V2 with Features")
    print("="*60)
    model_id = train_and_save_xgboost_v2(
        data,
        symbol="AAPL",
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        window=30,
        horizon=1,
        use_features=True
    )
    
    if model_id:
        print(f"\n✅ XGBoost model trained successfully: {model_id}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
