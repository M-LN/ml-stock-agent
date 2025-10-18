"""
Enhanced LSTM Training with Bi-directional Layers, Attention, and Dropout
Phase 2: Advanced Neural Network Architecture
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import pickle
import json

from feature_engineering import create_features, get_feature_columns


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for LSTM
    Lärer modellen at fokusere på vigtige tidspunkter i sekvensen
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Compute attention scores
        e = keras.backend.tanh(keras.backend.dot(inputs, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        
        # Apply attention weights
        output = inputs * a
        
        return keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


def build_bidirectional_lstm(sequence_length, n_features, 
                             lstm_units=[64, 32], 
                             dropout_rate=0.2,
                             use_attention=True,
                             learning_rate=0.001):
    """
    Bygger Bi-directional LSTM model med Attention og Dropout.
    
    Args:
        sequence_length: Længde af input sekvens
        n_features: Antal features per tidspunkt
        lstm_units: Liste af LSTM units per lag (fx [64, 32])
        dropout_rate: Dropout rate for regularization
        use_attention: Om attention mechanism skal bruges
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(sequence_length, n_features))
    x = inputs
    
    # Første Bi-directional LSTM lag
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True, 
                   kernel_regularizer=keras.regularizers.l2(0.01))
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Andet Bi-directional LSTM lag (hvis specificeret)
    if len(lstm_units) > 1:
        return_seq = True if use_attention else False
        x = layers.Bidirectional(
            layers.LSTM(lstm_units[1], return_sequences=return_seq,
                       kernel_regularizer=keras.regularizers.l2(0.01))
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Attention layer
    if use_attention and len(lstm_units) > 1:
        x = AttentionLayer()(x)
    
    # Dense layers
    x = layers.Dense(16, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile med Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def prepare_lstm_data_with_features(data, sequence_length=60, horizon=1, 
                                     use_features=True, train_ratio=0.7, val_ratio=0.15):
    """
    Forbereder data til LSTM træning med features.
    
    Args:
        data: DataFrame med OHLCV data
        sequence_length: Længde af input sekvens
        horizon: Forecast horizon
        use_features: Om tekniske indikatorer skal bruges
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data (rest er test)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols
    """
    if use_features:
        # Create features
        data_with_features = create_features(data, include_volume=True)
        feature_cols = get_feature_columns(data_with_features, exclude_ohlcv=True)
        feature_cols.append('Close')  # Add target
        
        # Use selected features
        data_array = data_with_features[feature_cols].values
        
    else:
        # Legacy: only close price
        data_array = data['Close'].values.reshape(-1, 1)
        feature_cols = ['Close']
    
    # Normalize data (MinMaxScaler for LSTM)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_array)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(data_scaled) - horizon + 1):
        X.append(data_scaled[i-sequence_length:i])
        # Target is Close price (last column if features, only column if not)
        target_idx = -1 if use_features else 0
        y.append(data_array[i+horizon-1, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/val/test
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    print(f"   Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


def train_and_save_lstm_v2(data, symbol, 
                           sequence_length=60,
                           lstm_units=[64, 32],
                           dropout_rate=0.2,
                           use_attention=True,
                           epochs=100,
                           batch_size=32,
                           learning_rate=0.001,
                           horizon=1,
                           use_features=True,
                           patience=15):
    """
    Træn Bi-directional LSTM med Attention (Version 2).
    
    Args:
        data: Historical stock data
        symbol: Stock symbol
        sequence_length: Input sequence length
        lstm_units: List of LSTM units per layer
        dropout_rate: Dropout rate
        use_attention: Whether to use attention mechanism
        epochs: Max epochs
        batch_size: Batch size
        learning_rate: Learning rate
        horizon: Forecast horizon
        use_features: Whether to use technical indicators
        patience: Early stopping patience
    
    Returns:
        model_id if successful, None otherwise
    """
    try:
        print(f"🧠 Træner Bi-LSTM V2 {'med features' if use_features else 'legacy'} for {symbol}...")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = prepare_lstm_data_with_features(
            data, sequence_length, horizon, use_features
        )
        
        n_features = X_train.shape[2]
        print(f"   Sequence length: {sequence_length}, Features: {n_features}")
        
        # Build model
        model = build_bidirectional_lstm(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            learning_rate=learning_rate
        )
        
        print(f"   Model parameters: {model.count_params():,}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        print(f"   Training with early stopping (patience={patience})...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        actual_epochs = len(history.history['loss'])
        print(f"   Trained for {actual_epochs} epochs (early stopped)")
        
        # Evaluate on all sets
        train_predictions = model.predict(X_train, verbose=0).flatten()
        val_predictions = model.predict(X_val, verbose=0).flatten()
        test_predictions = model.predict(X_test, verbose=0).flatten()
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        # MAPE
        train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
        val_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100
        test_mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
        
        # Directional Accuracy
        if len(y_val) > 1:
            val_direction_acc = np.mean(
                np.sign(y_val[1:] - y_val[:-1]) == np.sign(val_predictions[1:] - val_predictions[:-1])
            ) * 100
            test_direction_acc = np.mean(
                np.sign(y_test[1:] - y_test[:-1]) == np.sign(test_predictions[1:] - test_predictions[:-1])
            ) * 100
        else:
            val_direction_acc = 0.0
            test_direction_acc = 0.0
        
        print(f"   ✅ Val MAE: ${val_mae:.2f}, Test MAE: ${test_mae:.2f}")
        print(f"   ✅ Val MAPE: {val_mape:.2f}%, Test MAPE: {test_mape:.2f}%")
        print(f"   ✅ Val Dir Acc: {val_direction_acc:.1f}%, Test Dir Acc: {test_direction_acc:.1f}%")
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"lstm_v2_{symbol}_{timestamp}"
        
        MODEL_DIR = "saved_models"
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # Save Keras model
        model_path = os.path.join(MODEL_DIR, f"{model_id}.h5")
        model.save(model_path)
        
        # Save metadata and scaler
        metadata_path = os.path.join(MODEL_DIR, f"{model_id}_metadata.json")
        scaler_path = os.path.join(MODEL_DIR, f"{model_id}_scaler.pkl")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        metadata = {
            'model_type': 'lstm',
            'model_version': 'v2_bidirectional_attention',
            'symbol': symbol,
            'sequence_length': sequence_length,
            'horizon': horizon,
            'use_features': use_features,
            'use_attention': use_attention,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs_trained': actual_epochs,
            'n_features': n_features,
            'feature_cols': feature_cols,
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_mape': float(train_mape),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_mape': float(val_mape),
            'val_direction_acc': float(val_direction_acc),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_mape': float(test_mape),
            'test_direction_acc': float(test_direction_acc),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'trained_at': datetime.now().isoformat(),
            'data_period': f"{data.index[0]} to {data.index[-1]}"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Model saved: {model_id}")
        
        return model_id
        
    except Exception as e:
        print(f"   ❌ Error training LSTM: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test enhanced LSTM
    import yfinance as yf
    
    print("Testing Enhanced Bi-directional LSTM with Attention...")
    
    # Download test data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    
    print(f"\nData shape: {data.shape}")
    
    # Test LSTM V2 with features
    print("\n" + "="*60)
    print("Testing Bi-LSTM V2 with Features & Attention")
    print("="*60)
    
    model_id = train_and_save_lstm_v2(
        data,
        symbol="AAPL",
        sequence_length=60,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        use_attention=True,
        epochs=50,  # Will early stop
        batch_size=32,
        learning_rate=0.001,
        horizon=1,
        use_features=True,
        patience=10
    )
    
    if model_id:
        print(f"\n✅ Bi-LSTM model trained successfully: {model_id}")
    
    print("\n" + "="*60)
    print("✅ LSTM V2 test completed!")
    print("="*60)
