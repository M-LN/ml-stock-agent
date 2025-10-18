"""
A/B Testing Framework for ML Models
Compare multiple models in production and select winners

Features:
- Run multiple models simultaneously
- Track performance metrics
- Statistical significance testing
- Automatic winner selection
- Performance visualization
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ABTestingFramework:
    """
    Framework for A/B testing ML models in production.
    """
    
    def __init__(
        self,
        test_dir: str = 'ab_tests',
        min_samples: int = 30,
        confidence_level: float = 0.95
    ):
        """
        Args:
            test_dir: Directory to store A/B test results
            min_samples: Minimum predictions needed for significance test
            confidence_level: Confidence level for statistical tests (default 95%)
        """
        self.test_dir = test_dir
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        os.makedirs(test_dir, exist_ok=True)
    
    def create_test(
        self,
        test_name: str,
        symbol: str,
        models: List[Dict],
        duration_days: int = 30,
        traffic_split: Optional[List[float]] = None
    ) -> str:
        """
        Creates a new A/B test.
        
        Args:
            test_name: Unique name for the test
            symbol: Stock symbol to test on
            models: List of model dicts with {model_id, model_type, version}
            duration_days: Test duration in days
            traffic_split: Optional list of traffic percentages (must sum to 1.0)
        
        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default equal split if not provided
        if traffic_split is None:
            traffic_split = [1.0 / len(models)] * len(models)
        
        if abs(sum(traffic_split) - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {sum(traffic_split)}")
        
        if len(traffic_split) != len(models):
            raise ValueError(f"Traffic split length ({len(traffic_split)}) must match models ({len(models)})")
        
        # Create test configuration
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'start_date': datetime.now().isoformat(),
            'end_date': (datetime.now() + timedelta(days=duration_days)).isoformat(),
            'duration_days': duration_days,
            'status': 'active',
            'models': [
                {
                    **model,
                    'traffic_percentage': traffic_split[i],
                    'variant': chr(65 + i)  # A, B, C, ...
                }
                for i, model in enumerate(models)
            ],
            'results': {model['model_id']: [] for model in models}
        }
        
        # Save configuration
        test_file = os.path.join(self.test_dir, f"{test_id}.json")
        with open(test_file, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        print(f"✅ Created A/B test: {test_id}")
        print(f"   Duration: {duration_days} days")
        print(f"   Models:")
        for model in test_config['models']:
            print(f"     {model['variant']}: {model['model_id']} ({model['traffic_percentage']*100:.0f}% traffic)")
        
        return test_id
    
    def record_prediction(
        self,
        test_id: str,
        model_id: str,
        prediction: float,
        actual: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Records a prediction result for a model in an A/B test.
        
        Args:
            test_id: Test identifier
            model_id: Model that made the prediction
            prediction: Predicted value
            actual: Actual value (if available)
            timestamp: Prediction timestamp
        """
        test_file = os.path.join(self.test_dir, f"{test_id}.json")
        
        if not os.path.exists(test_file):
            raise ValueError(f"Test not found: {test_id}")
        
        with open(test_file, 'r') as f:
            test_config = json.load(f)
        
        # Record result
        result = {
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'prediction': float(prediction),
            'actual': float(actual) if actual is not None else None
        }
        
        if model_id not in test_config['results']:
            test_config['results'][model_id] = []
        
        test_config['results'][model_id].append(result)
        
        # Save updated config
        with open(test_file, 'w') as f:
            json.dump(test_config, f, indent=2)
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculates performance metrics from prediction results.
        
        Returns:
            Dict with MAE, RMSE, MAPE, directional accuracy
        """
        # Filter results with actual values
        valid_results = [r for r in results if r.get('actual') is not None]
        
        if not valid_results:
            return {
                'n_predictions': len(results),
                'n_with_actuals': 0,
                'mae': None,
                'rmse': None,
                'mape': None,
                'directional_accuracy': None
            }
        
        predictions = np.array([r['prediction'] for r in valid_results])
        actuals = np.array([r['actual'] for r in valid_results])
        
        # Calculate metrics
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(np.abs(errors / actuals)) * 100 if np.all(actuals != 0) else None
        
        # Directional accuracy (did we predict direction correctly?)
        if len(valid_results) > 1:
            actual_directions = np.diff(actuals) > 0
            pred_directions = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        else:
            directional_accuracy = None
        
        return {
            'n_predictions': len(results),
            'n_with_actuals': len(valid_results),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape) if mape is not None else None,
            'directional_accuracy': float(directional_accuracy) if directional_accuracy is not None else None
        }
    
    def t_test_comparison(
        self,
        errors_a: np.ndarray,
        errors_b: np.ndarray
    ) -> Dict:
        """
        Performs t-test to compare two models' errors.
        
        Returns:
            Dict with t-statistic, p-value, and significance
        """
        # Use absolute errors for comparison
        abs_errors_a = np.abs(errors_a)
        abs_errors_b = np.abs(errors_b)
        
        # Paired t-test (if same length) or independent t-test
        if len(abs_errors_a) == len(abs_errors_b):
            statistic, p_value = stats.ttest_rel(abs_errors_a, abs_errors_b)
            test_type = 'paired'
        else:
            statistic, p_value = stats.ttest_ind(abs_errors_a, abs_errors_b)
            test_type = 'independent'
        
        significant = p_value < self.alpha
        
        # Determine winner
        mean_a = np.mean(abs_errors_a)
        mean_b = np.mean(abs_errors_b)
        
        if significant:
            winner = 'A' if mean_a < mean_b else 'B'
            improvement = abs((mean_b - mean_a) / mean_b * 100)
        else:
            winner = 'tie'
            improvement = 0
        
        return {
            'test_type': test_type,
            't_statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'confidence_level': self.confidence_level,
            'winner': winner,
            'improvement_pct': float(improvement),
            'mean_error_a': float(mean_a),
            'mean_error_b': float(mean_b)
        }
    
    def analyze_test(self, test_id: str) -> Dict:
        """
        Analyzes A/B test results and determines winner.
        
        Returns:
            Dict with analysis results and recommendation
        """
        test_file = os.path.join(self.test_dir, f"{test_id}.json")
        
        if not os.path.exists(test_file):
            raise ValueError(f"Test not found: {test_id}")
        
        with open(test_file, 'r') as f:
            test_config = json.load(f)
        
        analysis = {
            'test_id': test_id,
            'test_name': test_config['test_name'],
            'symbol': test_config['symbol'],
            'analyzed_at': datetime.now().isoformat(),
            'models': {}
        }
        
        # Calculate metrics for each model
        all_errors = {}
        
        for model in test_config['models']:
            model_id = model['model_id']
            variant = model['variant']
            results = test_config['results'].get(model_id, [])
            
            metrics = self.calculate_metrics(results)
            
            # Extract errors for statistical test
            valid_results = [r for r in results if r.get('actual') is not None]
            if valid_results:
                predictions = np.array([r['prediction'] for r in valid_results])
                actuals = np.array([r['actual'] for r in valid_results])
                all_errors[variant] = predictions - actuals
            
            analysis['models'][variant] = {
                'model_id': model_id,
                'traffic_percentage': model['traffic_percentage'],
                'metrics': metrics,
                'sufficient_data': metrics['n_with_actuals'] >= self.min_samples
            }
        
        # Statistical comparison (if 2 models)
        if len(analysis['models']) == 2 and all(
            analysis['models'][v]['sufficient_data'] for v in ['A', 'B']
        ):
            comparison = self.t_test_comparison(all_errors['A'], all_errors['B'])
            analysis['statistical_comparison'] = comparison
            analysis['recommendation'] = f"Deploy Model {comparison['winner']}" if comparison['winner'] != 'tie' else "Continue testing"
        else:
            analysis['statistical_comparison'] = None
            analysis['recommendation'] = "Insufficient data for statistical comparison"
        
        # Determine overall winner (by MAE)
        valid_models = {
            v: data for v, data in analysis['models'].items()
            if data['metrics']['mae'] is not None
        }
        
        if valid_models:
            best_variant = min(
                valid_models.keys(),
                key=lambda v: valid_models[v]['metrics']['mae']
            )
            analysis['best_model_by_mae'] = best_variant
        
        return analysis
    
    def get_active_tests(self) -> List[Dict]:
        """
        Returns list of active A/B tests.
        """
        active_tests = []
        
        for filename in os.listdir(self.test_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.test_dir, filename), 'r') as f:
                    test_config = json.load(f)
                
                if test_config.get('status') == 'active':
                    active_tests.append({
                        'test_id': test_config['test_id'],
                        'test_name': test_config['test_name'],
                        'symbol': test_config['symbol'],
                        'created_at': test_config['created_at'],
                        'end_date': test_config['end_date']
                    })
        
        return active_tests
    
    def stop_test(self, test_id: str, winner: Optional[str] = None):
        """
        Stops an A/B test and optionally declares a winner.
        
        Args:
            test_id: Test identifier
            winner: Optional variant name (A, B, C, ...) to declare as winner
        """
        test_file = os.path.join(self.test_dir, f"{test_id}.json")
        
        if not os.path.exists(test_file):
            raise ValueError(f"Test not found: {test_id}")
        
        with open(test_file, 'r') as f:
            test_config = json.load(f)
        
        test_config['status'] = 'completed'
        test_config['ended_at'] = datetime.now().isoformat()
        
        if winner:
            test_config['winner'] = winner
        
        with open(test_file, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        print(f"✅ Test stopped: {test_id}")
        if winner:
            print(f"   Winner: Model {winner}")


def main():
    """Demo A/B testing framework"""
    print("=" * 70)
    print("A/B TESTING FRAMEWORK DEMO")
    print("=" * 70)
    
    framework = ABTestingFramework()
    
    # Create a test
    print("\n📊 Creating A/B Test...")
    test_id = framework.create_test(
        test_name="rf_vs_xgboost_aapl",
        symbol="AAPL",
        models=[
            {'model_id': 'rf_v2_AAPL_20251018', 'model_type': 'rf', 'version': 'v2'},
            {'model_id': 'xgb_v2_AAPL_20251018', 'model_type': 'xgboost', 'version': 'v2'}
        ],
        duration_days=30,
        traffic_split=[0.5, 0.5]  # 50/50 split
    )
    
    # Simulate some predictions
    print("\n🎲 Simulating predictions...")
    np.random.seed(42)
    
    for i in range(40):
        # Model A predictions (slightly better)
        pred_a = 150 + np.random.normal(0, 2)
        actual_a = 150 + np.random.normal(0, 1.5)
        framework.record_prediction(test_id, 'rf_v2_AAPL_20251018', pred_a, actual_a)
        
        # Model B predictions (slightly worse)
        pred_b = 150 + np.random.normal(0, 3)
        actual_b = 150 + np.random.normal(0, 1.5)
        framework.record_prediction(test_id, 'xgb_v2_AAPL_20251018', pred_b, actual_b)
    
    # Analyze results
    print("\n📈 Analyzing Test Results...")
    analysis = framework.analyze_test(test_id)
    
    print(f"\nTest: {analysis['test_name']}")
    print(f"Symbol: {analysis['symbol']}")
    print(f"\nResults:")
    
    for variant, data in analysis['models'].items():
        print(f"\n  Model {variant}: {data['model_id']}")
        metrics = data['metrics']
        print(f"    Predictions: {metrics['n_predictions']}")
        print(f"    With Actuals: {metrics['n_with_actuals']}")
        if metrics['mae']:
            print(f"    MAE: ${metrics['mae']:.2f}")
            print(f"    RMSE: ${metrics['rmse']:.2f}")
            if metrics['mape']:
                print(f"    MAPE: {metrics['mape']:.2f}%")
    
    if analysis.get('statistical_comparison'):
        comp = analysis['statistical_comparison']
        print(f"\n📊 Statistical Comparison:")
        print(f"  Test: {comp['test_type']} t-test")
        print(f"  P-value: {comp['p_value']:.4f}")
        print(f"  Significant: {comp['significant']}")
        print(f"  Winner: Model {comp['winner']}")
        print(f"  Improvement: {comp['improvement_pct']:.1f}%")
    
    print(f"\n💡 Recommendation: {analysis['recommendation']}")
    
    # List active tests
    print(f"\n📋 Active Tests:")
    active = framework.get_active_tests()
    for test in active:
        print(f"  - {test['test_name']} ({test['symbol']})")


if __name__ == "__main__":
    main()
