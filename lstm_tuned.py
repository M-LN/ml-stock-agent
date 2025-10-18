"""
LSTM v2 - TUNED VERSION
Improved Bi-directional LSTM with Attention and better hyperparameters

Key improvements over lstm_enhanced.py:
1. Reduced sequence length (30 instead of 60)
2. Feature selection (top 20 most important features)
3. Better scaling strategy
4. Smaller LSTM units to prevent overfitting
5. More aggressive dropout
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import pickle
import json
import os
from datetime import datetime
from feature_engineering import create_features, normalize_features

# Custom Attention Layer
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                shape=(input_shape[-1], 1),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


def select_top_features(data, n_features=20):
    """
    Select top N most important features based on correlation with target
    and variance.
    
    Returns: List of selected feature names
    """
    if 'Close' not in data.columns:
        return []
    
    # Calculate target (next day return)
    target = data['Close'].pct_change().shift(-1)
    
    # Calculate correlations
    correlations = {}
    variances = {}
    
    for col in data.columns:
        if col != 'Close' and data[col].dtype in ['float64', 'float32', 'int64']:
            # Absolute correlation with target
            corr = abs(data[col].corr(target))
            # Variance (normalized)
            var = data[col].var()
            
            if not np.isnan(corr) and not np.isnan(var):
                correlations[col] = corr
                variances[col] = var
    
    # Normalize variances to 0-1 range
    if variances:
        max_var = max(variances.values())
        variances = {k: v/max_var for k, v in variances.items()}
    
    # Combined score: 70% correlation + 30% variance
    scores = {col: 0.7 * correlations.get(col, 0) + 0.3 * variances.get(col, 0) 
              for col in correlations.keys()}
    
    # Sort by score and take top N
    top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
    
    print(f"\n📊 Top {n_features} Selected Features:")
    for i, (feat, score) in enumerate(top_features, 1):
        print(f"  {i}. {feat}: {score:.4f}")
    
    return [feat for feat, _ in top_features]


def prepare_lstm_data_tuned(data, sequence_length=30, n_features=20, train_split=0.7, val_split=0.15):
    """
    Prepare data for LSTM with feature selection and better splits.
    
    Args:
        data: DataFrame with OHLCV data
        sequence_length: Number of time steps (reduced to 30)
        n_features: Number of top features to select (20)
        train_split: Training data ratio (0.7)
        val_split: Validation data ratio (0.15)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, selected_features
    """
    print(f"\n🔧 Preparing LSTM Data (Tuned)...")
    print(f"   Sequence Length: {sequence_length}")
    print(f"   Feature Selection: Top {n_features}")
    
    # Create features
    data_with_features = create_features(data)
    
    # Select top features
    selected_features = select_top_features(data_with_features, n_features)
    
    # Keep only selected features + Close price
    columns_to_use = selected_features + ['Close']
    data_filtered = data_with_features[columns_to_use].copy()
    
    # Drop NaN values
    data_filtered = data_filtered.dropna()
    
    print(f"   Data shape after filtering: {data_filtered.shape}")
    
    if len(data_filtered) < sequence_length + 50:
        raise ValueError(f"Not enough data: {len(data_filtered)} rows (need >{sequence_length + 50})")
    
    # Separate features and target
    feature_columns = [col for col in data_filtered.columns if col != 'Close']
    X_data = data_filtered[feature_columns].values
    y_data = data_filtered['Close'].values
    
    # Scale features to [0, 1]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    
    # Scale target (Close price) separately
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(sequence_length, len(X_scaled)):
        X_sequences.append(X_scaled[i-sequence_length:i])
        y_sequences.append(y_scaled[i])  # Use scaled target
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"   Sequences created: {X_sequences.shape}")
    
    # Split into train/val/test
    n_samples = len(X_sequences)
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    X_train = X_sequences[:train_end]
    X_val = X_sequences[train_end:val_end]
    X_test = X_sequences[val_end:]
    
    y_train = y_sequences[:train_end]
    y_val = y_sequences[train_end:val_end]
    y_test = y_sequences[val_end:]
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Return both scalers
    scalers = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, selected_features


def build_lstm_tuned(sequence_length, n_features, lstm_units=[32, 16], use_attention=True):
    """
    Build tuned Bi-directional LSTM with Attention.
    
    Improvements:
    - Smaller LSTM units (32, 16 instead of 64, 32)
    - More aggressive dropout (0.3 instead of 0.2)
    - Fewer features input (20 instead of 68)
    - Shorter sequences (30 instead of 60)
    """
    print(f"\n🏗️ Building Tuned LSTM Model...")
    print(f"   LSTM Units: {lstm_units}")
    print(f"   Attention: {use_attention}")
    print(f"   Dropout: 0.3")
    
    input_layer = Input(shape=(sequence_length, n_features))
    
    # First Bi-LSTM layer
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True,
                          kernel_regularizer=regularizers.l2(0.001)))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Increased from 0.2
    
    # Second Bi-LSTM layer
    if len(lstm_units) > 1:
        x = Bidirectional(LSTM(lstm_units[1], return_sequences=use_attention,
                              kernel_regularizer=regularizers.l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)  # Increased from 0.2
    
    # Attention layer
    if use_attention:
        x = AttentionLayer()(x)
    
    # Dense layers
    x = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)  # Reduced from 16
    x = Dropout(0.2)(x)
    
    # Output layer
    output = Dense(1)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"   ✅ Model built: {model.count_params():,} parameters")
    
    return model


def train_and_save_lstm_tuned(data, symbol, sequence_length=30, lstm_units=[32, 16],
                               epochs=100, batch_size=32, use_attention=True, 
                               n_features=20, model_dir='saved_models'):
    """
    Train and save tuned LSTM v2 model.
    
    Args:
        data: DataFrame with OHLCV data
        symbol: Stock ticker
        sequence_length: Time steps (default 30, reduced from 60)
        lstm_units: LSTM layer sizes (default [32, 16], reduced from [64, 32])
        epochs: Max training epochs (100)
        batch_size: Batch size (32)
        use_attention: Use attention mechanism (True)
        n_features: Number of top features to use (20)
        model_dir: Directory to save models
    
    Returns:
        model_id: Unique model identifier
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training LSTM v2 (TUNED) for {symbol}")
    print(f"{'='*60}")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, selected_features = \
        prepare_lstm_data_tuned(data, sequence_length, n_features)
    
    # Build model
    model = build_lstm_tuned(sequence_length, len(selected_features), lstm_units, use_attention)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    print(f"\n🏋️ Training for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    y_pred_test_scaled = model.predict(X_test).flatten()
    
    # Inverse transform predictions back to original scale
    y_pred_test = scalers['scaler_y'].inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
    y_test_original = scalers['scaler_y'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    test_mae = mean_absolute_error(y_test_original, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test))
    test_mape = np.mean(np.abs((y_test_original - y_pred_test) / y_test_original)) * 100
    
    # Directional accuracy
    y_test_direction = np.sign(y_test_original[1:] - y_test_original[:-1])
    y_pred_direction = np.sign(y_pred_test[1:] - y_pred_test[:-1])
    direction_acc = np.mean(y_test_direction == y_pred_direction) * 100
    
    print(f"\n✅ Test Results:")
    print(f"   MAE:  ${test_mae:.2f}")
    print(f"   RMSE: ${test_rmse:.2f}")
    print(f"   MAPE: {test_mape:.2f}%")
    print(f"   Directional Accuracy: {direction_acc:.1f}%")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"lstm_v2_tuned_{symbol}_{timestamp}"
    
    model_path = os.path.join(model_dir, f"{model_id}.h5")
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_id': model_id,
        'symbol': symbol,
        'model_type': 'lstm_v2_tuned',
        'timestamp': timestamp,
        'sequence_length': sequence_length,
        'lstm_units': lstm_units,
        'use_attention': use_attention,
        'n_features': n_features,
        'selected_features': selected_features,
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'test_direction_acc': float(direction_acc),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'model_version': 'v2_tuned_bidirectional_attention'
    }
    
    metadata_path = os.path.join(model_dir, f"{model_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {metadata_path}")
    
    # Save scalers
    scaler_path = os.path.join(model_dir, f"{model_id}_scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"   Scalers saved: {scaler_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ LSTM v2 Tuned Training Complete!")
    print(f"   Model ID: {model_id}")
    print(f"   Test MAE: ${test_mae:.2f}")
    print(f"   Test MAPE: {test_mape:.2f}%")
    print(f"{'='*60}\n")
    
    return model_id


# Test script
if __name__ == "__main__":
    print("🧪 Testing LSTM v2 Tuned on AAPL...")
    
    # Download data
    data = yf.download('AAPL', period='2y', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Train tuned model
    model_id = train_and_save_lstm_tuned(
        data, 
        symbol='AAPL',
        sequence_length=30,  # Reduced from 60
        lstm_units=[32, 16],  # Reduced from [64, 32]
        epochs=100,
        use_attention=True,
        n_features=20  # Top 20 features only
    )
    
    print(f"\n✅ Test complete! Model saved as: {model_id}")
    print("\n📊 Comparison with original lstm_enhanced.py:")
    print("   Original: ~60 sequence length, 68 features, [64, 32] units")
    print("   Tuned:    30 sequence length, 20 features, [32, 16] units")
    print("   Expected: Much lower MAE (should be <$50 instead of $200+)")
