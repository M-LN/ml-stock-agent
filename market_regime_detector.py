"""
Market Regime Detection Module
Identificerer markedets tilstand: Bull, Bear, Sideways, eller High Volatility

Dette bruges til at vælge den bedste model for nuværende markedsforhold.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta


class MarketRegimeDetector:
    """
    Detector til at identificere markedets regime baseret på tekniske indikatorer.
    """
    
    def __init__(self, lookback_period: int = 50):
        """
        Args:
            lookback_period: Antal dage at analysere for regime detection
        """
        self.lookback_period = lookback_period
        self.regime_history = []
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Beregner trend styrke baseret på lineær regression.
        
        Returns:
            Positive værdi = uptrend, negative = downtrend
            Størrelse indikerer styrke
        """
        closes = data['Close'].tail(self.lookback_period).values
        x = np.arange(len(closes))
        
        # Linear regression
        slope, intercept = np.polyfit(x, closes, 1)
        
        # Normalize by current price (percentage per day)
        normalized_slope = (slope / closes[-1]) * 100
        
        return normalized_slope
    
    def calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """
        Beregner historisk volatilitet (annualiseret).
        
        Returns:
            Volatility som årlig standard deviation
        """
        returns = data['Close'].pct_change().tail(period)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return volatility
    
    def calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """
        Måler hvor konsistent trenden er (0-1).
        
        Returns:
            1.0 = meget konsistent trend
            0.0 = ingen konsistent trend
        """
        closes = data['Close'].tail(self.lookback_period).values
        
        # Calculate moving averages
        sma_short = pd.Series(closes).rolling(window=10).mean()
        sma_long = pd.Series(closes).rolling(window=30).mean()
        
        # Count how many days short MA is above/below long MA
        if len(sma_short) < 30 or len(sma_long) < 30:
            return 0.5
        
        crossovers = (sma_short > sma_long).astype(int).diff().abs().sum()
        
        # Fewer crossovers = more consistent
        consistency = 1.0 - min(crossovers / 10, 1.0)
        
        return consistency
    
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Hovedfunktion til at detektere market regime.
        
        Args:
            data: DataFrame med OHLCV data
        
        Returns:
            Dict med regime info:
            {
                'regime': 'bull'|'bear'|'sideways'|'high_volatility',
                'confidence': 0.0-1.0,
                'metrics': {...}
            }
        """
        # Ensure we have enough data
        if len(data) < self.lookback_period:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'metrics': {},
                'description': 'Insufficient data for regime detection'
            }
        
        # Calculate metrics
        trend_strength = self.calculate_trend_strength(data)
        volatility = self.calculate_volatility(data)
        consistency = self.calculate_trend_consistency(data)
        
        # Calculate additional metrics
        returns_period = data['Close'].pct_change().tail(self.lookback_period)
        avg_return = returns_period.mean() * 252  # Annualized
        max_drawdown = self._calculate_max_drawdown(data)
        
        metrics = {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'consistency': consistency,
            'avg_return': avg_return,
            'max_drawdown': max_drawdown
        }
        
        # Regime detection logic
        regime, confidence = self._classify_regime(metrics)
        
        result = {
            'regime': regime,
            'confidence': confidence,
            'metrics': metrics,
            'description': self._get_regime_description(regime),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.regime_history.append(result)
        
        return result
    
    def _classify_regime(self, metrics: Dict) -> Tuple[str, float]:
        """
        Klassificer regime baseret på metrics.
        
        Returns:
            (regime_name, confidence)
        """
        trend = metrics['trend_strength']
        vol = metrics['volatility']
        consistency = metrics['consistency']
        
        # Thresholds
        HIGH_VOL_THRESHOLD = 0.35  # 35% annualized volatility
        STRONG_TREND_THRESHOLD = 0.15  # 0.15% daily trend
        WEAK_TREND_THRESHOLD = 0.05
        HIGH_CONSISTENCY_THRESHOLD = 0.7
        
        # High Volatility overrides everything
        if vol > HIGH_VOL_THRESHOLD:
            confidence = min((vol - HIGH_VOL_THRESHOLD) / HIGH_VOL_THRESHOLD, 1.0)
            return 'high_volatility', confidence
        
        # Strong uptrend with consistency = Bull
        if trend > STRONG_TREND_THRESHOLD and consistency > HIGH_CONSISTENCY_THRESHOLD:
            confidence = min(trend / STRONG_TREND_THRESHOLD, 1.0) * consistency
            return 'bull', confidence
        
        # Strong downtrend with consistency = Bear
        if trend < -STRONG_TREND_THRESHOLD and consistency > HIGH_CONSISTENCY_THRESHOLD:
            confidence = min(abs(trend) / STRONG_TREND_THRESHOLD, 1.0) * consistency
            return 'bear', confidence
        
        # Weak trend or low consistency = Sideways
        if abs(trend) < WEAK_TREND_THRESHOLD:
            confidence = 1.0 - abs(trend) / WEAK_TREND_THRESHOLD
            return 'sideways', confidence
        
        # Medium trend but low consistency = Sideways
        if consistency < HIGH_CONSISTENCY_THRESHOLD:
            confidence = 1.0 - consistency
            return 'sideways', confidence
        
        # Default: Sideways with medium confidence
        return 'sideways', 0.6
    
    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """
        Beregner maximum drawdown i perioden.
        
        Returns:
            Max drawdown som negativ procent
        """
        closes = data['Close'].tail(self.lookback_period)
        cummax = closes.cummax()
        drawdown = (closes - cummax) / cummax
        return drawdown.min()
    
    def _get_regime_description(self, regime: str) -> str:
        """Returns human-readable description of regime."""
        descriptions = {
            'bull': '🐂 Bull Market - Stigende trend med god momentum',
            'bear': '🐻 Bear Market - Faldende trend med nedadgående pres',
            'sideways': '📊 Sideways Market - Konsoliderende, ingen klar retning',
            'high_volatility': '⚡ High Volatility - Turbulent marked med store udsving',
            'unknown': '❓ Unknown - Utilstrækkelig data til at bestemme regime'
        }
        return descriptions.get(regime, 'Unknown regime')
    
    def get_regime_characteristics(self, regime: str) -> Dict:
        """
        Returnerer anbefalede model karakteristika for et givet regime.
        """
        characteristics = {
            'bull': {
                'best_models': ['lstm', 'xgboost'],
                'recommended_window': 20,  # Shorter for fast-moving bull
                'recommended_features': ['momentum', 'trend', 'volume'],
                'risk_level': 'medium',
                'description': 'Momentum og trend-following modeller performer bedst'
            },
            'bear': {
                'best_models': ['random_forest', 'lstm'],
                'recommended_window': 30,
                'recommended_features': ['volatility', 'support_resistance', 'volume'],
                'risk_level': 'high',
                'description': 'Defensive modeller med fokus på downside protection'
            },
            'sideways': {
                'best_models': ['random_forest', 'xgboost'],
                'recommended_window': 40,  # Longer for range-bound
                'recommended_features': ['mean_reversion', 'support_resistance', 'bollinger'],
                'risk_level': 'low',
                'description': 'Mean-reversion strategier og range trading'
            },
            'high_volatility': {
                'best_models': ['ensemble', 'random_forest'],
                'recommended_window': 15,  # Very short for rapid changes
                'recommended_features': ['volatility', 'volume', 'momentum'],
                'risk_level': 'very_high',
                'description': 'Ensemble modeller for at håndtere usikkerhed'
            }
        }
        return characteristics.get(regime, {})
    
    def get_regime_history(self, n_periods: int = 10) -> pd.DataFrame:
        """
        Returnerer historik af regime detections.
        
        Args:
            n_periods: Antal perioder at returnere
        
        Returns:
            DataFrame med regime historik
        """
        if not self.regime_history:
            return pd.DataFrame()
        
        history = self.regime_history[-n_periods:]
        
        df = pd.DataFrame([
            {
                'timestamp': h['timestamp'],
                'regime': h['regime'],
                'confidence': h['confidence'],
                'trend_strength': h['metrics'].get('trend_strength', 0),
                'volatility': h['metrics'].get('volatility', 0)
            }
            for h in history
        ])
        
        return df


def get_current_regime(symbol: str, period: str = '6mo') -> Dict:
    """
    Helper function til at få current regime for et symbol.
    
    Args:
        symbol: Stock ticker
        period: Data period for analysis
    
    Returns:
        Regime detection result
    """
    import yfinance as yf
    
    # Download data
    data = yf.download(symbol, period=period, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if data.empty or len(data) < 50:
        return {
            'regime': 'unknown',
            'confidence': 0.0,
            'metrics': {},
            'description': 'Insufficient data'
        }
    
    # Detect regime
    detector = MarketRegimeDetector(lookback_period=50)
    result = detector.detect_regime(data)
    
    return result


# Test script
if __name__ == "__main__":
    print("🔍 Testing Market Regime Detector...")
    print("="*60)
    
    # Test on multiple stocks
    test_symbols = ['AAPL', 'TSLA', 'SPY', 'NVDA']
    
    for symbol in test_symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        result = get_current_regime(symbol, period='6mo')
        
        print(f"   Regime: {result['regime'].upper()}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   {result['description']}")
        
        if result['metrics']:
            print(f"   Trend Strength: {result['metrics']['trend_strength']:.3f}%/day")
            print(f"   Volatility: {result['metrics']['volatility']:.1%}")
            print(f"   Consistency: {result['metrics']['consistency']:.1%}")
            print(f"   Max Drawdown: {result['metrics']['max_drawdown']:.1%}")
        
        # Get recommendations
        detector = MarketRegimeDetector()
        characteristics = detector.get_regime_characteristics(result['regime'])
        
        if characteristics:
            print(f"   💡 Best Models: {', '.join(characteristics['best_models'])}")
            print(f"   📏 Recommended Window: {characteristics['recommended_window']} days")
            print(f"   ⚠️  Risk Level: {characteristics['risk_level']}")
    
    print("\n" + "="*60)
    print("✅ Market Regime Detection Test Complete!")
