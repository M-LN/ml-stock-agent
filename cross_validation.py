"""
Cross-Validation Framework for Time Series Models
Phase 2: Robust Model Evaluation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable
import json


class TimeSeriesCrossValidator:
    """
    Walk-Forward Cross-Validation for Time Series.
    Respekterer tidsmæssig rækkefølge (ingen data leakage).
    """
    
    def __init__(self, n_splits=5, test_size=30, gap=0):
        """
        Args:
            n_splits: Antal CV splits
            test_size: Størrelse af test set (i samples)
            gap: Gap mellem train og test for at undgå leakage
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, data):
        """
        Generer train/test indices for hver split.
        
        Yields:
            (train_indices, test_indices) for hver split
        """
        n_samples = len(data)
        
        # Beregn split points
        test_starts = np.linspace(
            n_samples - (self.n_splits * self.test_size),
            n_samples - self.test_size,
            self.n_splits,
            dtype=int
        )
        
        for test_start in test_starts:
            test_end = test_start + self.test_size
            train_end = test_start - self.gap
            
            if train_end < self.test_size:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self):
        return self.n_splits


def cross_validate_model(train_func: Callable, 
                         data: pd.DataFrame,
                         cv: TimeSeriesCrossValidator,
                         symbol: str,
                         model_params: Dict,
                         verbose: bool = True) -> Dict:
    """
    Udfører cross-validation på en model.
    
    Args:
        train_func: Training function (skal returnere predictions)
        data: Time series data
        cv: CrossValidator instance
        symbol: Stock symbol
        model_params: Parameters til model
        verbose: Print progress
    
    Returns:
        Dict med CV resultater
    """
    cv_results = {
        'splits': [],
        'mae_scores': [],
        'rmse_scores': [],
        'mape_scores': [],
        'direction_acc_scores': []
    }
    
    if verbose:
        print(f"\n🔄 Running {cv.get_n_splits()}-fold Time Series Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(data), 1):
        if verbose:
            print(f"\n   Fold {fold}/{cv.get_n_splits()}")
            print(f"   Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Train model and get predictions
        try:
            predictions, actuals = train_func(train_data, test_data, model_params)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            # Directional accuracy
            if len(actuals) > 1:
                direction_acc = np.mean(
                    np.sign(actuals[1:] - actuals[:-1]) == np.sign(predictions[1:] - predictions[:-1])
                ) * 100
            else:
                direction_acc = 0.0
            
            cv_results['mae_scores'].append(mae)
            cv_results['rmse_scores'].append(rmse)
            cv_results['mape_scores'].append(mape)
            cv_results['direction_acc_scores'].append(direction_acc)
            
            cv_results['splits'].append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'direction_acc': float(direction_acc)
            })
            
            if verbose:
                print(f"   ✅ MAE: ${mae:.2f}, MAPE: {mape:.2f}%, Dir Acc: {direction_acc:.1f}%")
        
        except Exception as e:
            if verbose:
                print(f"   ❌ Fold {fold} failed: {e}")
            continue
    
    # Calculate summary statistics
    if cv_results['mae_scores']:
        cv_results['summary'] = {
            'mean_mae': float(np.mean(cv_results['mae_scores'])),
            'std_mae': float(np.std(cv_results['mae_scores'])),
            'mean_rmse': float(np.mean(cv_results['rmse_scores'])),
            'std_rmse': float(np.std(cv_results['rmse_scores'])),
            'mean_mape': float(np.mean(cv_results['mape_scores'])),
            'std_mape': float(np.std(cv_results['mape_scores'])),
            'mean_direction_acc': float(np.mean(cv_results['direction_acc_scores'])),
            'std_direction_acc': float(np.std(cv_results['direction_acc_scores']))
        }
        
        if verbose:
            print(f"\n📊 CV Summary:")
            print(f"   MAE: ${cv_results['summary']['mean_mae']:.2f} ± ${cv_results['summary']['std_mae']:.2f}")
            print(f"   MAPE: {cv_results['summary']['mean_mape']:.2f}% ± {cv_results['summary']['std_mape']:.2f}%")
            print(f"   Direction Acc: {cv_results['summary']['mean_direction_acc']:.1f}% ± {cv_results['summary']['std_direction_acc']:.1f}%")
    
    return cv_results


def calculate_confidence_intervals(predictions: np.ndarray, 
                                   actuals: np.ndarray,
                                   confidence_level: float = 0.95) -> Dict:
    """
    Beregner confidence intervals for predictions.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        confidence_level: Confidence level (default 0.95)
    
    Returns:
        Dict med confidence intervals og statistikker
    """
    errors = actuals - predictions
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_errors = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(errors, size=len(errors), replace=True)
        bootstrap_errors.append(np.std(sample))
    
    error_std = np.mean(bootstrap_errors)
    
    # Calculate confidence intervals
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_score.get(confidence_level, 1.96)
    
    margin = z * error_std
    
    lower_bound = predictions - margin
    upper_bound = predictions + margin
    
    # Calculate coverage (% of actuals within CI)
    coverage = np.mean((actuals >= lower_bound) & (actuals <= upper_bound)) * 100
    
    return {
        'confidence_level': confidence_level,
        'margin_of_error': float(margin),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'coverage': float(coverage),
        'mean_interval_width': float(np.mean(upper_bound - lower_bound))
    }


def predict_with_confidence(model_func: Callable,
                            data: pd.DataFrame,
                            n_predictions: int = 5,
                            confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Lav predictions med confidence intervals.
    
    Args:
        model_func: Function der laver predictions
        data: Historical data
        n_predictions: Antal fremtidige predictions
        confidence_level: Confidence level
    
    Returns:
        DataFrame med predictions og confidence intervals
    """
    predictions = []
    
    for i in range(n_predictions):
        pred = model_func(data)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Monte Carlo for uncertainty
    pred_std = np.std(predictions)
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_score.get(confidence_level, 1.96)
    
    mean_pred = np.mean(predictions)
    lower = mean_pred - z * pred_std
    upper = mean_pred + z * pred_std
    
    return pd.DataFrame({
        'prediction': [mean_pred],
        'lower_bound': [lower],
        'upper_bound': [upper],
        'confidence_level': [confidence_level],
        'std_dev': [pred_std]
    })


def quantile_regression_intervals(model, X, quantiles=[0.025, 0.975]):
    """
    Beregner prediction intervals ved quantile regression.
    
    Args:
        model: Trained model med quantile prediction support
        X: Input features
        quantiles: Quantiles at beregne (default 2.5% og 97.5% for 95% CI)
    
    Returns:
        Lower og upper bounds
    """
    # Placeholder - requires quantile regression model
    # For now, return simple intervals
    predictions = model.predict(X)
    std = np.std(predictions) * 1.5
    
    lower = predictions - 1.96 * std
    upper = predictions + 1.96 * std
    
    return lower, upper


if __name__ == "__main__":
    print("Testing Cross-Validation Framework...")
    
    # Create dummy time series data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    prices = 100 + np.cumsum(np.random.randn(500) * 2)
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    data.set_index('Date', inplace=True)
    
    print(f"Data shape: {data.shape}")
    
    # Test CV
    cv = TimeSeriesCrossValidator(n_splits=5, test_size=30, gap=5)
    
    print(f"\nCV Configuration:")
    print(f"   Splits: {cv.get_n_splits()}")
    print(f"   Test size: {cv.test_size}")
    print(f"   Gap: {cv.gap}")
    
    # Dummy train function for testing
    def dummy_train(train_data, test_data, params):
        # Simple moving average prediction
        window = params.get('window', 20)
        ma = train_data['Close'].rolling(window=window).mean().iloc[-1]
        
        predictions = np.full(len(test_data), ma)
        actuals = test_data['Close'].values
        
        return predictions, actuals
    
    # Run CV
    cv_results = cross_validate_model(
        train_func=dummy_train,
        data=data,
        cv=cv,
        symbol="TEST",
        model_params={'window': 20},
        verbose=True
    )
    
    # Test confidence intervals
    print("\n" + "="*60)
    print("Testing Confidence Intervals")
    print("="*60)
    
    predictions = np.random.randn(100) * 10 + 100
    actuals = predictions + np.random.randn(100) * 5
    
    ci_results = calculate_confidence_intervals(predictions, actuals, confidence_level=0.95)
    
    print(f"\nConfidence Interval (95%):")
    print(f"   Margin of Error: ±${ci_results['margin_of_error']:.2f}")
    print(f"   Coverage: {ci_results['coverage']:.1f}%")
    print(f"   Mean Interval Width: ${ci_results['mean_interval_width']:.2f}")
    
    print("\n✅ All CV tests completed!")
