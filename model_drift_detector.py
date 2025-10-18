"""
Model Drift Detection System
Detects when model performance degrades and triggers retraining

Implements:
- Statistical drift tests (KS test, PSI)
- Performance degradation monitoring
- Automatic alerts and retraining triggers
- Drift visualization
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
import yfinance as yf

warnings.filterwarnings('ignore')


class ModelDriftDetector:
    """
    Detects model drift through multiple statistical tests and performance monitoring.
    """
    
    def __init__(
        self,
        models_dir: str = 'saved_models',
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.20,
        lookback_days: int = 30
    ):
        """
        Args:
            models_dir: Directory with model metadata
            drift_threshold: P-value threshold for drift detection (default 0.05)
            performance_threshold: Max acceptable performance degradation (default 20%)
            lookback_days: Days to look back for performance comparison
        """
        self.models_dir = models_dir
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.lookback_days = lookback_days
    
    def kolmogorov_smirnov_test(
        self,
        training_data: np.ndarray,
        recent_data: np.ndarray
    ) -> Dict:
        """
        Performs KS test to detect distribution shift.
        
        Returns:
            Dict with statistic, p_value, and drift_detected
        """
        statistic, p_value = ks_2samp(training_data, recent_data)
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': p_value < self.drift_threshold,
            'threshold': self.drift_threshold,
            'interpretation': 'Distributions differ significantly' if p_value < self.drift_threshold else 'No significant difference'
        }
    
    def population_stability_index(
        self,
        training_data: np.ndarray,
        recent_data: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Calculates PSI (Population Stability Index) for drift detection.
        
        PSI Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change (drift detected)
        
        Returns:
            Dict with PSI value and drift status
        """
        # Create bins based on training data
        bins = np.percentile(training_data, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates
        
        if len(bins) <= 1:
            return {
                'test': 'Population Stability Index (PSI)',
                'psi': 0.0,
                'drift_detected': False,
                'severity': 'none',
                'interpretation': 'Insufficient unique values for binning'
            }
        
        # Calculate distributions
        train_counts, _ = np.histogram(training_data, bins=bins)
        recent_counts, _ = np.histogram(recent_data, bins=bins)
        
        # Avoid division by zero
        train_dist = (train_counts + 1) / (len(training_data) + n_bins)
        recent_dist = (recent_counts + 1) / (len(recent_data) + n_bins)
        
        # Calculate PSI
        psi = np.sum((recent_dist - train_dist) * np.log(recent_dist / train_dist))
        
        # Determine severity
        if psi < 0.1:
            severity = 'none'
            drift_detected = False
        elif psi < 0.25:
            severity = 'moderate'
            drift_detected = True
        else:
            severity = 'significant'
            drift_detected = True
        
        return {
            'test': 'Population Stability Index (PSI)',
            'psi': float(psi),
            'drift_detected': drift_detected,
            'severity': severity,
            'threshold': 0.25,
            'interpretation': f'{severity.capitalize()} drift detected' if drift_detected else 'No significant drift'
        }
    
    def check_feature_drift(
        self,
        symbol: str,
        model_id: str,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Checks for drift in input features.
        
        Args:
            symbol: Stock symbol
            model_id: Model identifier
            current_data: Recent market data
        
        Returns:
            Dict with drift results for each feature
        """
        # Load model metadata to get training data statistics
        metadata_file = os.path.join(self.models_dir, f"{model_id}_metadata.json")
        
        if not os.path.exists(metadata_file):
            return {'error': f'Model metadata not found: {metadata_file}'}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        training_stats = metadata.get('training_statistics', {})
        
        if not training_stats:
            return {'error': 'No training statistics available in metadata'}
        
        # Check drift for each feature
        drift_results = {}
        
        for feature in current_data.columns:
            if feature in training_stats:
                train_mean = training_stats[feature].get('mean', 0)
                train_std = training_stats[feature].get('std', 1)
                
                current_values = current_data[feature].dropna().values
                
                if len(current_values) == 0:
                    continue
                
                current_mean = np.mean(current_values)
                current_std = np.std(current_values)
                
                # Z-score for mean shift
                mean_shift = abs(current_mean - train_mean) / (train_std + 1e-8)
                
                # Variance ratio
                variance_ratio = current_std / (train_std + 1e-8)
                
                drift_results[feature] = {
                    'train_mean': float(train_mean),
                    'current_mean': float(current_mean),
                    'mean_shift_zscore': float(mean_shift),
                    'train_std': float(train_std),
                    'current_std': float(current_std),
                    'variance_ratio': float(variance_ratio),
                    'drift_detected': mean_shift > 3 or variance_ratio > 2 or variance_ratio < 0.5
                }
        
        return drift_results
    
    def check_performance_degradation(
        self,
        symbol: str,
        model_id: str,
        recent_predictions: List[Dict]
    ) -> Dict:
        """
        Checks if model performance has degraded compared to training.
        
        Args:
            symbol: Stock symbol
            model_id: Model identifier
            recent_predictions: List of recent prediction results
        
        Returns:
            Dict with performance comparison and degradation status
        """
        # Load model metadata
        metadata_file = os.path.join(self.models_dir, f"{model_id}_metadata.json")
        
        if not os.path.exists(metadata_file):
            return {'error': f'Model metadata not found: {metadata_file}'}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        train_mae = metadata.get('train_mae', None)
        train_mape = metadata.get('train_mape', None)
        
        if train_mae is None:
            return {'error': 'No training MAE in metadata'}
        
        # Calculate recent performance
        if not recent_predictions:
            return {'error': 'No recent predictions to compare'}
        
        recent_errors = [abs(p['predicted'] - p['actual']) for p in recent_predictions if 'actual' in p]
        
        if not recent_errors:
            return {'error': 'No actual values available for comparison'}
        
        recent_mae = np.mean(recent_errors)
        recent_std = np.std(recent_errors)
        
        # Calculate degradation
        mae_degradation = (recent_mae - train_mae) / train_mae
        degraded = mae_degradation > self.performance_threshold
        
        return {
            'train_mae': float(train_mae),
            'recent_mae': float(recent_mae),
            'recent_std': float(recent_std),
            'mae_degradation': float(mae_degradation),
            'degradation_pct': float(mae_degradation * 100),
            'threshold_pct': float(self.performance_threshold * 100),
            'degraded': degraded,
            'interpretation': f'Performance degraded by {mae_degradation*100:.1f}%' if degraded else 'Performance stable',
            'recommendation': 'Retrain model' if degraded else 'Continue monitoring'
        }
    
    def detect_concept_drift(
        self,
        symbol: str,
        window_size: int = 30
    ) -> Dict:
        """
        Detects concept drift by comparing recent price patterns to historical.
        
        Concept drift = relationship between features and target changes over time
        
        Args:
            symbol: Stock symbol
            window_size: Size of sliding window for comparison
        
        Returns:
            Dict with drift detection results
        """
        # Download recent data
        data = yf.download(symbol, period='6mo', progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) < window_size * 2:
            return {'error': 'Insufficient data for concept drift detection'}
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # Split into two periods
        mid_point = len(returns) // 2
        early_period = returns.iloc[:mid_point].values
        recent_period = returns.iloc[mid_point:].values
        
        # KS test on returns distribution
        ks_result = self.kolmogorov_smirnov_test(early_period, recent_period)
        
        # PSI on returns
        psi_result = self.population_stability_index(early_period, recent_period)
        
        # Volatility change
        early_vol = np.std(early_period) * np.sqrt(252)
        recent_vol = np.std(recent_period) * np.sqrt(252)
        vol_change = (recent_vol - early_vol) / early_vol
        
        return {
            'symbol': symbol,
            'test_period': '6 months',
            'ks_test': ks_result,
            'psi_test': psi_result,
            'volatility_change': {
                'early_volatility': float(early_vol),
                'recent_volatility': float(recent_vol),
                'change_pct': float(vol_change * 100),
                'significant_change': abs(vol_change) > 0.5
            },
            'overall_drift_detected': ks_result['drift_detected'] or psi_result['drift_detected'],
            'recommendation': 'Retrain model - concept drift detected' if (ks_result['drift_detected'] or psi_result['drift_detected']) else 'Continue monitoring'
        }
    
    def comprehensive_drift_check(
        self,
        symbol: str,
        model_id: str,
        recent_predictions: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Performs comprehensive drift detection using all available tests.
        
        Returns:
            Dict with all drift test results and overall recommendation
        """
        results = {
            'symbol': symbol,
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # 1. Concept drift detection
        print(f"🔍 Checking concept drift for {symbol}...")
        concept_drift = self.detect_concept_drift(symbol)
        results['tests']['concept_drift'] = concept_drift
        
        # 2. Performance degradation (if predictions available)
        if recent_predictions:
            print(f"📊 Checking performance degradation...")
            perf_check = self.check_performance_degradation(symbol, model_id, recent_predictions)
            results['tests']['performance'] = perf_check
        
        # Determine overall status
        drift_detected = False
        reasons = []
        
        if concept_drift.get('overall_drift_detected'):
            drift_detected = True
            reasons.append('Concept drift detected')
        
        if recent_predictions:
            perf_check = results['tests'].get('performance', {})
            if perf_check.get('degraded'):
                drift_detected = True
                reasons.append(f"Performance degraded by {perf_check.get('degradation_pct', 0):.1f}%")
        
        results['overall'] = {
            'drift_detected': drift_detected,
            'reasons': reasons,
            'recommendation': 'RETRAIN MODEL' if drift_detected else 'Continue monitoring',
            'priority': 'HIGH' if drift_detected else 'LOW'
        }
        
        return results
    
    def should_retrain(self, drift_results: Dict) -> bool:
        """
        Determines if model should be retrained based on drift results.
        
        Returns:
            True if retraining is recommended
        """
        return drift_results.get('overall', {}).get('drift_detected', False)


def main():
    """Test drift detection system"""
    print("=" * 70)
    print("MODEL DRIFT DETECTION SYSTEM TEST")
    print("=" * 70)
    
    detector = ModelDriftDetector()
    
    # Test on multiple symbols
    test_symbols = [
        ('AAPL', 'rf_v2_AAPL_20251018_140647'),
        ('TSLA', 'rf_v2_TSLA_20251018_140647'),
    ]
    
    for symbol, model_id in test_symbols:
        print(f"\n{'='*70}")
        print(f"Testing Drift Detection: {symbol}")
        print('='*70)
        
        # Check if model exists
        metadata_file = os.path.join('saved_models', f"{model_id}_metadata.json")
        if not os.path.exists(metadata_file):
            print(f"⚠️ Model not found, using first available for {symbol}")
            # Find any model for this symbol
            import glob
            models = glob.glob(f'saved_models/*{symbol}*_metadata.json')
            if models:
                model_id = os.path.basename(models[0]).replace('_metadata.json', '')
                print(f"✅ Using model: {model_id}")
            else:
                print(f"❌ No models found for {symbol}")
                continue
        
        # Run comprehensive drift check
        results = detector.comprehensive_drift_check(symbol, model_id)
        
        # Display results
        print(f"\n📊 Drift Detection Results:")
        print(f"Symbol: {results['symbol']}")
        print(f"Model: {results['model_id']}")
        
        # Concept drift
        if 'concept_drift' in results['tests']:
            cd = results['tests']['concept_drift']
            print(f"\n🔍 Concept Drift:")
            print(f"  KS Test: {'⚠️ DRIFT' if cd['ks_test']['drift_detected'] else '✅ OK'} (p={cd['ks_test']['p_value']:.4f})")
            print(f"  PSI Test: {'⚠️ DRIFT' if cd['psi_test']['drift_detected'] else '✅ OK'} (PSI={cd['psi_test']['psi']:.4f})")
            print(f"  Volatility Change: {cd['volatility_change']['change_pct']:.1f}%")
        
        # Overall recommendation
        overall = results['overall']
        print(f"\n{'🚨' if overall['drift_detected'] else '✅'} Overall Status:")
        print(f"  Drift Detected: {overall['drift_detected']}")
        print(f"  Recommendation: {overall['recommendation']}")
        print(f"  Priority: {overall['priority']}")
        
        if overall['reasons']:
            print(f"  Reasons:")
            for reason in overall['reasons']:
                print(f"    - {reason}")


if __name__ == "__main__":
    main()
