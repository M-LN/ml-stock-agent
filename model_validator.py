"""
Model Validator - Advanced Testing & Validation Framework
Provides comprehensive model validation, parameter recommendations, and diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

class ModelValidator:
    """
    Comprehensive model validation and testing framework
    """
    
    def __init__(self, data, model_type, symbol):
        self.data = data
        self.model_type = model_type
        self.symbol = symbol
        
    def validate_data_quality(self):
        """
        Check data quality and provide warnings/recommendations
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Check data length
        if len(self.data) < 100:
            issues.append("⚠️ **Very limited data** (< 100 points) - model may not be reliable")
            recommendations.append("Try increasing period to at least 6 months")
        elif len(self.data) < 252:
            warnings.append("⚡ Limited data (< 1 year) - consider longer period for better accuracy")
            
        # Check for missing values
        missing = self.data.isnull().sum().sum()
        if missing > 0:
            issues.append(f"❌ **Missing values detected**: {missing} data points")
            recommendations.append("Data will be forward-filled automatically")
            
        # Check volatility
        returns = self.data['Close'].pct_change()
        volatility = returns.std()
        if volatility > 0.05:
            warnings.append(f"📊 High volatility detected ({volatility:.2%}) - model predictions may be less accurate")
            recommendations.append("Consider using ensemble methods or shorter prediction horizons")
            
        # Check for outliers (3 sigma)
        z_scores = np.abs((self.data['Close'] - self.data['Close'].mean()) / self.data['Close'].std())
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            warnings.append(f"⚡ {outliers} outliers detected (> 3 sigma)")
            recommendations.append("Outliers will be included but may affect model performance")
            
        return {
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'data_quality_score': max(0, 100 - len(issues)*30 - len(warnings)*10)
        }
    
    def recommend_parameters(self):
        """
        Recommend optimal parameters based on data characteristics
        """
        data_length = len(self.data)
        volatility = self.data['Close'].pct_change().std()
        
        recommendations = {}
        
        if self.model_type == "Random Forest":
            # More data = more trees and deeper trees
            if data_length < 252:
                recommendations['n_estimators'] = 50
                recommendations['max_depth'] = 5
                recommendations['rationale'] = "Limited data - using conservative parameters"
            elif data_length < 504:
                recommendations['n_estimators'] = 100
                recommendations['max_depth'] = 10
                recommendations['rationale'] = "Moderate data - balanced parameters"
            else:
                recommendations['n_estimators'] = 200
                recommendations['max_depth'] = 15
                recommendations['rationale'] = "Sufficient data - using more complex model"
                
            # Adjust window based on volatility
            if volatility > 0.03:
                recommendations['window'] = 14  # Shorter window for volatile stocks
            else:
                recommendations['window'] = 30  # Longer window for stable stocks
                
        elif self.model_type == "XGBoost":
            if data_length < 252:
                recommendations['n_estimators'] = 50
                recommendations['learning_rate'] = 0.1
                recommendations['max_depth'] = 3
                recommendations['rationale'] = "Limited data - simpler model to avoid overfitting"
            elif data_length < 504:
                recommendations['n_estimators'] = 100
                recommendations['learning_rate'] = 0.05
                recommendations['max_depth'] = 5
                recommendations['rationale'] = "Moderate data - balanced complexity"
            else:
                recommendations['n_estimators'] = 200
                recommendations['learning_rate'] = 0.01
                recommendations['max_depth'] = 6
                recommendations['rationale'] = "Sufficient data - can use lower learning rate"
                
            recommendations['window'] = 14 if volatility > 0.03 else 30
            
        elif self.model_type == "LSTM":
            if data_length < 252:
                recommendations['sequence_length'] = 30
                recommendations['lstm_units'] = 25
                recommendations['epochs'] = 30
                recommendations['rationale'] = "Limited data - shorter sequences"
            elif data_length < 504:
                recommendations['sequence_length'] = 60
                recommendations['lstm_units'] = 50
                recommendations['epochs'] = 50
                recommendations['rationale'] = "Moderate data - standard LSTM setup"
            else:
                recommendations['sequence_length'] = 90
                recommendations['lstm_units'] = 100
                recommendations['epochs'] = 100
                recommendations['rationale'] = "Large dataset - deeper LSTM network"
                
            recommendations['batch_size'] = 32
            
        return recommendations
    
    def estimate_training_time(self, params):
        """
        Estimate training time based on data size and parameters
        """
        data_length = len(self.data)
        
        if self.model_type == "Random Forest":
            # RF is fast
            base_time = data_length * params.get('n_estimators', 100) * 0.001
            return max(5, min(60, base_time))  # 5-60 seconds
            
        elif self.model_type == "XGBoost":
            # XGBoost is also relatively fast
            base_time = data_length * params.get('n_estimators', 100) * 0.002
            return max(10, min(90, base_time))  # 10-90 seconds
            
        elif self.model_type == "LSTM":
            # LSTM is slower
            epochs = params.get('epochs', 50)
            sequence_length = params.get('sequence_length', 60)
            base_time = (data_length / sequence_length) * epochs * 0.1
            return max(30, min(300, base_time))  # 30-300 seconds
            
        elif self.model_type == "Prophet":
            # Prophet is moderate
            return max(15, min(60, data_length * 0.05))
            
        return 30  # default
    
    def calculate_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15):
        """
        Calculate indices for train/val/test split
        """
        n = len(self.data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train_size': train_end,
            'val_size': val_end - train_end,
            'test_size': n - val_end,
            'train_end_idx': train_end,
            'val_end_idx': val_end,
            'train_dates': (self.data.index[0], self.data.index[train_end-1]),
            'val_dates': (self.data.index[train_end], self.data.index[val_end-1]) if val_end > train_end else None,
            'test_dates': (self.data.index[val_end], self.data.index[-1]) if val_end < n else None
        }
    
    def generate_training_report(self, params):
        """
        Generate comprehensive pre-training report
        """
        data_quality = self.validate_data_quality()
        param_recommendations = self.recommend_parameters()
        split_info = self.calculate_train_val_test_split()
        est_time = self.estimate_training_time(params)
        
        return {
            'data_quality': data_quality,
            'param_recommendations': param_recommendations,
            'split_info': split_info,
            'estimated_time': est_time,
            'data_stats': {
                'total_samples': len(self.data),
                'date_range': f"{self.data.index[0]} to {self.data.index[-1]}",
                'price_range': f"${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}",
                'avg_price': f"${self.data['Close'].mean():.2f}",
                'volatility': f"{self.data['Close'].pct_change().std():.2%}"
            }
        }

def display_validation_report(report, params, model_type):
    """
    Display validation report in Streamlit UI
    """
    st.markdown("### 📊 Pre-Training Validation Report")
    
    # Data Quality Score
    quality_score = report['data_quality']['data_quality_score']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
        st.metric("Data Quality Score", f"{color} {quality_score}/100")
    
    with col2:
        st.metric("Total Samples", report['data_stats']['total_samples'])
    
    with col3:
        st.metric("Est. Training Time", f"~{int(report['estimated_time'])}s")
    
    # Issues and Warnings
    if report['data_quality']['issues']:
        st.error("**Critical Issues:**")
        for issue in report['data_quality']['issues']:
            st.markdown(f"- {issue}")
            
    if report['data_quality']['warnings']:
        st.warning("**Warnings:**")
        for warning in report['data_quality']['warnings']:
            st.markdown(f"- {warning}")
            
    # Recommendations
    if report['data_quality']['recommendations']:
        with st.expander("💡 Recommendations", expanded=True):
            for rec in report['data_quality']['recommendations']:
                st.markdown(f"- {rec}")
    
    # Parameter Comparison
    st.markdown("### 🎯 Parameter Analysis")
    
    rec_params = report['param_recommendations']
    if rec_params:
        st.info(f"**Rationale:** {rec_params.get('rationale', 'N/A')}")
        
        # Compare current vs recommended
        param_comparison = []
        for param_name, recommended_value in rec_params.items():
            if param_name != 'rationale':
                current_value = params.get(param_name, 'N/A')
                status = "✅" if current_value == recommended_value else "⚠️"
                param_comparison.append({
                    'Parameter': param_name,
                    'Your Value': current_value,
                    'Recommended': recommended_value,
                    'Status': status
                })
        
        if param_comparison:
            df = pd.DataFrame(param_comparison)
            st.dataframe(df, use_container_width=True)
    
    # Split Information
    st.markdown("### 📈 Data Split")
    split_info = report['split_info']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training", f"{split_info['train_size']} samples")
    col2.metric("Validation", f"{split_info['val_size']} samples")
    col3.metric("Test", f"{split_info['test_size']} samples")
    
    # Data Stats
    with st.expander("📊 Data Statistics"):
        stats = report['data_stats']
        st.markdown(f"""
        - **Date Range:** {stats['date_range']}
        - **Price Range:** {stats['price_range']}
        - **Average Price:** {stats['avg_price']}
        - **Volatility:** {stats['volatility']}
        """)
