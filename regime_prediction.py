"""
Regime-Aware Prediction System
Automatically selects best model based on current market regime
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from market_regime_detector import get_current_regime

# Try to import trainer, but handle gracefully if dependencies fail
try:
    from regime_aware_trainer import RegimeAwareTrainer
    TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: RegimeAwareTrainer not available: {e}")
    RegimeAwareTrainer = None
    TRAINER_AVAILABLE = False


class RegimePredictionSystem:
    """
    System til at lave predictions ved automatisk at vælge bedste model
    baseret på current market regime.
    """
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.trainer = RegimeAwareTrainer() if TRAINER_AVAILABLE else None
    
    def get_latest_regime_mapping(self, symbol: str, model_type: str) -> Optional[Dict]:
        """
        Henter latest regime-to-model mapping for et givet symbol og model type.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            model_type: 'rf', 'xgboost', eller 'lstm'
        
        Returns:
            Dict med regime mapping eller None hvis ikke fundet
        """
        pattern = os.path.join(
            self.models_dir,
            f"regime_models_{symbol}_{model_type}_*.json"
        )
        
        mapping_files = glob.glob(pattern)
        
        if not mapping_files:
            return None
        
        # Find latest fil baseret på timestamp
        latest_file = max(mapping_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def predict_with_regime_awareness(
        self,
        symbol: str,
        model_type: str = 'rf',
        period: str = '6mo',
        fallback_to_standard: bool = True
    ) -> Dict:
        """
        Laver prediction ved automatisk at vælge model baseret på current regime.
        
        Args:
            symbol: Stock symbol
            model_type: 'rf', 'xgboost', eller 'lstm'
            period: Data period for regime detection
            fallback_to_standard: Hvis True, fald tilbage til standard model hvis regime model ikke findes
        
        Returns:
            Dict med prediction results og metadata
        """
        # 1. Detect current regime
        regime_result = get_current_regime(symbol, period)
        regime = regime_result['regime']
        confidence = regime_result['confidence']
        
        print(f"\n🔍 Detected regime: {regime.upper()} (confidence: {confidence:.0%})")
        
        # 2. Hent regime mapping
        mapping = self.get_latest_regime_mapping(symbol, model_type)
        
        if mapping is None:
            if fallback_to_standard:
                print(f"⚠️ No regime models found. Falling back to standard model.")
                return {
                    'regime': regime,
                    'confidence': confidence,
                    'model_used': f'{model_type}_standard',
                    'regime_specific': False,
                    'fallback': True
                }
            else:
                raise ValueError(f"No regime models found for {symbol} ({model_type})")
        
        # 3. Vælg model for current regime
        if regime in mapping['regime_models']:
            model_id = mapping['regime_models'][regime]
            print(f"✅ Using regime-specific model: {model_id}")
            
            return {
                'regime': regime,
                'confidence': confidence,
                'model_id': model_id,
                'model_used': f'{model_type}_{regime}',
                'regime_specific': True,
                'fallback': False,
                'metrics': regime_result.get('metrics', {}),
                'description': regime_result.get('description', '')
            }
        else:
            # Regime model ikke tilgængelig
            if fallback_to_standard:
                print(f"⚠️ No model for {regime} regime. Falling back to standard model.")
                return {
                    'regime': regime,
                    'confidence': confidence,
                    'model_used': f'{model_type}_standard',
                    'regime_specific': False,
                    'fallback': True,
                    'reason': f'No model trained for {regime} regime'
                }
            else:
                raise ValueError(f"No model trained for {regime} regime")
    
    def get_regime_model_coverage(self, symbol: str, model_type: str) -> Dict:
        """
        Checker hvilke regimes der har trained models.
        
        Returns:
            Dict med coverage information
        """
        mapping = self.get_latest_regime_mapping(symbol, model_type)
        
        if mapping is None:
            return {
                'has_regime_models': False,
                'coverage': {},
                'total_regimes': 0,
                'covered_regimes': 0
            }
        
        all_regimes = ['bull', 'bear', 'sideways', 'high_volatility']
        coverage = {regime: regime in mapping['regime_models'] for regime in all_regimes}
        
        return {
            'has_regime_models': True,
            'coverage': coverage,
            'total_regimes': len(all_regimes),
            'covered_regimes': sum(coverage.values()),
            'coverage_percentage': sum(coverage.values()) / len(all_regimes) * 100,
            'mapping_file': mapping.get('metadata', {}).get('created_at', 'Unknown')
        }
    
    def compare_regime_models(
        self,
        symbol: str,
        model_type: str = 'rf',
        test_period: str = '3mo'
    ) -> pd.DataFrame:
        """
        Sammenligner performance af regime-specific models vs standard model.
        
        Returns:
            DataFrame med comparison results
        """
        results = []
        
        # Download test data
        data = yf.download(symbol, period=test_period, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Detect regimes over test period
        from market_regime_detector import MarketRegimeDetector
        detector = MarketRegimeDetector()
        
        all_regimes = ['bull', 'bear', 'sideways', 'high_volatility']
        
        for regime in all_regimes:
            # Simulate predictions for this regime
            result = {
                'Regime': regime.capitalize(),
                'Has Model': False,
                'Model Performance': 'N/A'
            }
            
            mapping = self.get_latest_regime_mapping(symbol, model_type)
            if mapping and regime in mapping['regime_models']:
                result['Has Model'] = True
                result['Model ID'] = mapping['regime_models'][regime]
            
            results.append(result)
        
        return pd.DataFrame(results)


def main():
    """Test regime-aware prediction system"""
    print("=" * 70)
    print("REGIME-AWARE PREDICTION SYSTEM TEST")
    print("=" * 70)
    
    system = RegimePredictionSystem()
    
    # Test symbols
    symbols = ['AAPL', 'TSLA', 'SPY']
    
    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"Testing {symbol}")
        print('=' * 70)
        
        # 1. Check coverage
        print(f"\n📊 Regime Model Coverage:")
        coverage = system.get_regime_model_coverage(symbol, 'rf')
        
        if coverage['has_regime_models']:
            print(f"✅ Has regime models: {coverage['covered_regimes']}/{coverage['total_regimes']} regimes covered")
            print(f"Coverage: {coverage['coverage_percentage']:.0f}%")
            for regime, has_model in coverage['coverage'].items():
                status = "✅" if has_model else "❌"
                print(f"  {status} {regime.capitalize()}")
        else:
            print("❌ No regime models found")
        
        # 2. Make prediction
        print(f"\n🎯 Making Regime-Aware Prediction:")
        try:
            prediction = system.predict_with_regime_awareness(symbol, model_type='rf')
            
            print(f"\nRegime: {prediction['regime'].upper()}")
            print(f"Confidence: {prediction['confidence']:.0%}")
            print(f"Model Used: {prediction['model_used']}")
            print(f"Regime-Specific: {prediction['regime_specific']}")
            
            if prediction.get('metrics'):
                metrics = prediction['metrics']
                print(f"\nMarket Metrics:")
                print(f"  Trend: {metrics['trend_strength']:.3f}%/day")
                print(f"  Volatility: {metrics['volatility']:.1%}")
                print(f"  Consistency: {metrics['consistency']:.0%}")
        
        except Exception as e:
            print(f"⚠️ Prediction failed: {str(e)}")


if __name__ == "__main__":
    main()
