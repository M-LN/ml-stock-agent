"""
Regime-Aware Model Training
Træner separate modeller for forskellige market regimes

Dette giver bedre accuracy fordi modeller specialiserer sig i specifikke markedsforhold.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

from market_regime_detector import MarketRegimeDetector
from ml_training_enhanced import train_and_save_rf_v2, train_and_save_xgboost_v2
from lstm_tuned import train_and_save_lstm_tuned


class RegimeAwareTrainer:
    """
    Trainer der automatisk opdeler data i regimes og træner separate modeller.
    """
    
    def __init__(self, model_dir: str = 'saved_models'):
        self.model_dir = model_dir
        self.detector = MarketRegimeDetector(lookback_period=50)
        os.makedirs(model_dir, exist_ok=True)
    
    def segment_data_by_regime(self, data: pd.DataFrame, 
                               window_size: int = 50) -> Dict[str, List[pd.DataFrame]]:
        """
        Opdeler data i segments baseret på market regime.
        
        Args:
            data: Full dataset
            window_size: Size of rolling window for regime detection
        
        Returns:
            Dict med regime -> liste af data segments
        """
        segments = {
            'bull': [],
            'bear': [],
            'sideways': [],
            'high_volatility': []
        }
        
        print(f"\n🔍 Segmenting {len(data)} days of data by regime...")
        
        # Rolling window detection
        for i in range(window_size, len(data), window_size // 2):  # 50% overlap
            window_data = data.iloc[max(0, i-window_size):i]
            
            if len(window_data) < window_size:
                continue
            
            # Detect regime for this window
            result = self.detector.detect_regime(window_data)
            regime = result['regime']
            
            if regime != 'unknown' and result['confidence'] > 0.5:
                segments[regime].append(window_data)
        
        # Print summary
        print(f"\n📊 Regime Segments:")
        for regime, segs in segments.items():
            if segs:
                total_days = sum(len(s) for s in segs)
                print(f"   {regime.upper()}: {len(segs)} segments, {total_days} days")
        
        return segments
    
    def train_regime_specific_models(self, data: pd.DataFrame, symbol: str,
                                     model_type: str = 'rf',
                                     **kwargs) -> Dict[str, str]:
        """
        Træner separate modeller for hver regime.
        
        Args:
            data: Full dataset
            symbol: Stock ticker
            model_type: 'rf', 'xgboost', eller 'lstm'
            **kwargs: Model-specific parameters
        
        Returns:
            Dict med regime -> model_id
        """
        print(f"\n🎯 Training Regime-Specific {model_type.upper()} Models for {symbol}")
        print("="*60)
        
        # Segment data by regime
        segments = self.segment_data_by_regime(data)
        
        trained_models = {}
        
        # Train model for each regime that has enough data
        for regime, data_segments in segments.items():
            if not data_segments:
                print(f"\n⏭️  Skipping {regime} - no data segments")
                continue
            
            # Combine all segments for this regime
            regime_data = pd.concat(data_segments, axis=0).drop_duplicates()
            
            if len(regime_data) < 100:
                print(f"\n⏭️  Skipping {regime} - insufficient data ({len(regime_data)} days)")
                continue
            
            print(f"\n🏋️  Training {model_type.upper()} for {regime.upper()} regime...")
            print(f"   Data: {len(regime_data)} days")
            
            try:
                # Train model based on type
                if model_type == 'rf':
                    model_id = train_and_save_rf_v2(
                        regime_data, 
                        f"{symbol}_{regime}",
                        n_estimators=kwargs.get('n_estimators', 200),
                        max_depth=kwargs.get('max_depth', 15),
                        window=kwargs.get('window', 30),
                        horizon=kwargs.get('horizon', 1),
                        use_features=True
                    )
                
                elif model_type == 'xgboost':
                    model_id = train_and_save_xgboost_v2(
                        regime_data,
                        f"{symbol}_{regime}",
                        n_estimators=kwargs.get('n_estimators', 300),
                        max_depth=kwargs.get('max_depth', 8),
                        learning_rate=kwargs.get('learning_rate', 0.05),
                        window=kwargs.get('window', 30),
                        horizon=kwargs.get('horizon', 1),
                        use_features=True
                    )
                
                elif model_type == 'lstm':
                    model_id = train_and_save_lstm_tuned(
                        regime_data,
                        f"{symbol}_{regime}",
                        sequence_length=kwargs.get('sequence_length', 30),
                        lstm_units=kwargs.get('lstm_units', [32, 16]),
                        epochs=kwargs.get('epochs', 100),
                        use_attention=True,
                        n_features=kwargs.get('n_features', 20)
                    )
                
                else:
                    print(f"   ❌ Unknown model type: {model_type}")
                    continue
                
                if model_id:
                    trained_models[regime] = model_id
                    print(f"   ✅ {regime.upper()} model trained: {model_id}")
                
            except Exception as e:
                print(f"   ❌ Error training {regime} model: {e}")
                continue
        
        # Save regime model mapping
        if trained_models:
            mapping_file = os.path.join(
                self.model_dir, 
                f"regime_models_{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(mapping_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'model_type': model_type,
                    'created': datetime.now().isoformat(),
                    'regime_models': trained_models
                }, f, indent=2)
            
            print(f"\n💾 Regime model mapping saved: {mapping_file}")
        
        return trained_models
    
    def select_best_model_for_regime(self, symbol: str, 
                                     current_regime: str,
                                     model_type: str = 'rf') -> Optional[str]:
        """
        Vælger den bedste model for current regime.
        
        Args:
            symbol: Stock ticker
            current_regime: Current market regime
            model_type: Model type to select
        
        Returns:
            Model ID eller None
        """
        # Find regime model mapping files
        mapping_files = [
            f for f in os.listdir(self.model_dir)
            if f.startswith(f"regime_models_{symbol}_{model_type}")
            and f.endswith('.json')
        ]
        
        if not mapping_files:
            print(f"⚠️  No regime models found for {symbol} {model_type}")
            return None
        
        # Get most recent mapping
        latest_mapping = sorted(mapping_files)[-1]
        mapping_path = os.path.join(self.model_dir, latest_mapping)
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        regime_models = mapping.get('regime_models', {})
        
        # Try to find model for current regime
        if current_regime in regime_models:
            return regime_models[current_regime]
        
        # Fallback: return any available model
        if regime_models:
            fallback = list(regime_models.values())[0]
            print(f"⚠️  No model for {current_regime}, using fallback: {fallback}")
            return fallback
        
        return None


def train_regime_ensemble(symbol: str, period: str = '2y') -> Dict:
    """
    Helper function til at træne ensemble af regime-specific modeller.
    
    Args:
        symbol: Stock ticker
        period: Data period
    
    Returns:
        Dict med trained models per regime
    """
    import yfinance as yf
    
    print(f"📥 Downloading {period} of data for {symbol}...")
    data = yf.download(symbol, period=period, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if len(data) < 200:
        print(f"❌ Insufficient data for {symbol}")
        return {}
    
    trainer = RegimeAwareTrainer()
    
    # Train RF models for each regime
    rf_models = trainer.train_regime_specific_models(
        data, symbol,
        model_type='rf',
        n_estimators=150,
        max_depth=12,
        window=30
    )
    
    return {
        'symbol': symbol,
        'rf_regime_models': rf_models,
        'total_models': len(rf_models)
    }


# Test script
if __name__ == "__main__":
    print("🧪 Testing Regime-Aware Training...")
    print("="*60)
    
    # Test with AAPL
    result = train_regime_ensemble('AAPL', period='2y')
    
    print("\n" + "="*60)
    print("✅ Regime-Aware Training Test Complete!")
    print(f"\n📊 Results:")
    print(f"   Symbol: {result.get('symbol')}")
    print(f"   Models Trained: {result.get('total_models')}")
    print(f"   Regime Models: {list(result.get('rf_regime_models', {}).keys())}")
