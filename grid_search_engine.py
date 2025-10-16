"""
Grid Search Engine for Hyperparameter Optimization
Systematically searches through parameter space to find optimal model configuration
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import itertools
from agent_interactive import (
    train_and_save_rf,
    train_and_save_xgboost,
    train_and_save_lstm,
    calculate_model_metrics,
    MODEL_DIR
)

# Grid search results directory
GRID_SEARCH_DIR = "grid_searches"
if not os.path.exists(GRID_SEARCH_DIR):
    os.makedirs(GRID_SEARCH_DIR)

# Predefined search spaces
SEARCH_SPACES = {
    'Random Forest': {
        'small': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        },
        'medium': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        },
        'large': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'small': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01]
        },
        'medium': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7]
        },
        'large': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.1, 0.05, 0.01, 0.001],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'LSTM': {
        'small': {
            'lstm_units': [32, 64],
            'sequence_length': [30, 60]
        },
        'medium': {
            'lstm_units': [32, 64, 128],
            'sequence_length': [30, 60, 90],
            'epochs': [30, 50]
        },
        'large': {
            'lstm_units': [32, 64, 128, 256],
            'sequence_length': [30, 60, 90, 120],
            'epochs': [30, 50, 100],
            'batch_size': [16, 32, 64]
        }
    }
}

class GridSearchEngine:
    """
    Engine for performing grid search over hyperparameter space
    """
    
    def __init__(self, model_type, symbol, search_space='medium'):
        """
        Initialize grid search engine
        
        Args:
            model_type: Type of model ('Random Forest', 'XGBoost', 'LSTM')
            symbol: Stock symbol
            search_space: Size of search space ('small', 'medium', 'large')
        """
        self.model_type = model_type
        self.symbol = symbol
        self.search_space_size = search_space
        self.search_space = SEARCH_SPACES.get(model_type, {}).get(search_space, {})
        self.results = []
        self.best_params = None
        self.best_score = float('inf')  # Lower is better for error metrics
    
    def get_param_combinations(self):
        """
        Generate all parameter combinations from search space
        
        Returns:
            list: List of parameter dicts
        """
        if not self.search_space:
            return []
        
        # Get all parameter names and values
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dicts
        param_dicts = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            param_dicts.append(param_dict)
        
        return param_dicts
    
    def run_search(self, data, progress_callback=None):
        """
        Run grid search over all parameter combinations
        
        Args:
            data: Historical stock data (pandas DataFrame)
            progress_callback: Optional callback function for progress updates
        
        Returns:
            dict: Search results with best parameters and all trials
        """
        param_combinations = self.get_param_combinations()
        total_combinations = len(param_combinations)
        
        if total_combinations == 0:
            return {
                'success': False,
                'error': 'No search space defined for this model type'
            }
        
        st.info(f"🔍 Starting grid search with {total_combinations} parameter combinations...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, params in enumerate(param_combinations):
            # Update progress
            progress = (i + 1) / total_combinations
            progress_bar.progress(progress)
            status_text.text(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            # Train model with these parameters
            try:
                if self.model_type == 'Random Forest':
                    model_id = train_and_save_rf(
                        data, 
                        self.symbol,
                        **params
                    )
                elif self.model_type == 'XGBoost':
                    model_id = train_and_save_xgboost(
                        data,
                        self.symbol,
                        **params
                    )
                elif self.model_type == 'LSTM':
                    model_id = train_and_save_lstm(
                        data,
                        self.symbol,
                        **params
                    )
                else:
                    continue
                
                # Load model and get metrics
                from agent_interactive import load_model
                model, metadata = load_model(model_id)
                
                if metadata:
                    mae = metadata.get('mae', float('inf'))
                    rmse = metadata.get('rmse', float('inf'))
                    r2 = metadata.get('r2_score', 0)
                    
                    # Store result
                    result = {
                        'params': params,
                        'model_id': model_id,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'score': mae  # Use MAE as primary metric
                    }
                    
                    self.results.append(result)
                    
                    # Update best if better
                    if mae < self.best_score:
                        self.best_score = mae
                        self.best_params = params
                        self.best_model_id = model_id
            
            except Exception as e:
                st.warning(f"Failed for params {params}: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("✅ Grid search complete!")
        
        # Save results
        search_id = self._save_results()
        
        return {
            'success': True,
            'search_id': search_id,
            'best_params': self.best_params,
            'best_model_id': self.best_model_id,
            'best_score': self.best_score,
            'total_trials': len(self.results),
            'all_results': self.results
        }
    
    def _save_results(self):
        """Save grid search results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_id = f"{self.model_type}_{self.symbol}_{timestamp}"
        
        results_data = {
            'search_id': search_id,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'search_space_size': self.search_space_size,
            'timestamp': timestamp,
            'best_params': self.best_params,
            'best_model_id': self.best_model_id,
            'best_score': self.best_score,
            'total_trials': len(self.results),
            'all_results': self.results
        }
        
        filename = os.path.join(GRID_SEARCH_DIR, f"{search_id}.json")
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return search_id

def get_previous_grid_searches():
    """
    Get list of previous grid searches
    
    Returns:
        list: List of previous search metadata
    """
    searches = []
    
    if not os.path.exists(GRID_SEARCH_DIR):
        return searches
    
    for filename in os.listdir(GRID_SEARCH_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(GRID_SEARCH_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                searches.append(data)
    
    # Sort by timestamp (newest first)
    searches.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return searches

def load_grid_search(search_id):
    """
    Load a specific grid search by ID
    
    Args:
        search_id: Search identifier
    
    Returns:
        dict: Search data or None if not found
    """
    filename = os.path.join(GRID_SEARCH_DIR, f"{search_id}.json")
    
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        return json.load(f)

def compare_grid_searches(search_ids):
    """
    Compare multiple grid searches
    
    Args:
        search_ids: List of search IDs to compare
    
    Returns:
        pandas.DataFrame: Comparison table
    """
    comparison_data = []
    
    for search_id in search_ids:
        data = load_grid_search(search_id)
        if data:
            comparison_data.append({
                'Search ID': search_id,
                'Model Type': data.get('model_type'),
                'Symbol': data.get('symbol'),
                'Space Size': data.get('search_space_size'),
                'Best MAE': data.get('best_score'),
                'Total Trials': data.get('total_trials'),
                'Timestamp': data.get('timestamp')
            })
    
    return pd.DataFrame(comparison_data)
