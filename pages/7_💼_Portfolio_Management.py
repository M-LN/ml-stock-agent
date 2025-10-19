"""
Portfolio Management Page
Handles portfolio allocation, position sizing, rebalancing, and AI mentor feedback
Enhanced with: Live prices, performance tracking, tax-loss harvesting, correlation analysis, Monte Carlo
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import json
import os
import yfinance as yf
from scipy import stats
from scipy.stats import norm

# Page configuration
st.set_page_config(
    page_title="Portfolio Management",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Portfolio Management")
st.markdown("---")

# Initialize session state for portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'positions': [],  # List of {symbol, shares, avg_cost, current_price, sector, asset_class}
        'target_allocation': {
            'stocks': 60,
            'bonds': 30,
            'commodities': 10
        },
        'risk_profile': 'moderate',  # conservative, moderate, aggressive
        'rebalance_threshold': 10,  # Percentage
        'max_position_size': 15,  # Max % per position
        'last_rebalance': None
    }

if 'portfolio_history' not in st.session_state:
    st.session_state.portfolio_history = []

if 'price_cache' not in st.session_state:
    st.session_state.price_cache = {}  # Cache for live prices

if 'performance_snapshots' not in st.session_state:
    st.session_state.performance_snapshots = []  # Historical portfolio snapshots


# ===== LIVE PRICE FUNCTIONS =====
def fetch_live_price(symbol: str) -> Dict:
    """Fetch live price from Yahoo Finance with caching"""
    # Check cache (valid for 5 minutes)
    cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if cache_key in st.session_state.price_cache:
        return st.session_state.price_cache[cache_key]
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return {'success': False, 'error': 'No data available'}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        result = {
            'success': True,
            'symbol': symbol,
            'price': float(current_price),
            'change': float(change),
            'change_pct': float(change_pct),
            'volume': int(hist['Volume'].iloc[-1]),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'sector': info.get('sector', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the result
        st.session_state.price_cache[cache_key] = result
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def update_portfolio_prices(positions: List[Dict]) -> List[Dict]:
    """Update all portfolio positions with live prices"""
    updated_positions = []
    
    for pos in positions:
        symbol = pos['symbol']
        price_data = fetch_live_price(symbol)
        
        if price_data['success']:
            pos_copy = pos.copy()
            pos_copy['current_price'] = price_data['price']
            pos_copy['price_change'] = price_data['change']
            pos_copy['price_change_pct'] = price_data['change_pct']
            pos_copy['last_updated'] = price_data['timestamp']
            
            # Update sector if not set or different
            if 'sector' not in pos or pos['sector'] == 'Unknown':
                pos_copy['sector'] = price_data['sector']
            
            updated_positions.append(pos_copy)
        else:
            updated_positions.append(pos)
    
    return updated_positions


def fetch_historical_prices(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical price data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


# ===== PERFORMANCE TRACKING FUNCTIONS =====
def save_portfolio_snapshot():
    """Save current portfolio state to history"""
    total_value = calculate_portfolio_value(st.session_state.portfolio['positions'])
    
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'total_value': total_value,
        'positions': len(st.session_state.portfolio['positions']),
        'asset_allocation': calculate_asset_allocation(st.session_state.portfolio['positions']),
        'positions_detail': st.session_state.portfolio['positions'].copy()
    }
    
    st.session_state.performance_snapshots.append(snapshot)
    
    # Keep only last 365 snapshots
    if len(st.session_state.performance_snapshots) > 365:
        st.session_state.performance_snapshots = st.session_state.performance_snapshots[-365:]


def calculate_portfolio_returns(snapshots: List[Dict]) -> pd.DataFrame:
    """Calculate portfolio returns over time"""
    if len(snapshots) < 2:
        return pd.DataFrame()
    
    data = []
    for snap in snapshots:
        data.append({
            'date': pd.to_datetime(snap['timestamp']),
            'value': snap['total_value']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    df['daily_return'] = df['value'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    return df


def calculate_portfolio_metrics(returns: pd.Series) -> Dict:
    """Calculate portfolio performance metrics"""
    if len(returns) < 2:
        return {}
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Annualized metrics (assuming daily returns)
    trading_days = 252
    
    metrics = {
        'total_return': float((1 + returns).prod() - 1),
        'annualized_return': float(returns.mean() * trading_days),
        'volatility': float(returns.std() * np.sqrt(trading_days)),
        'sharpe_ratio': float((returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))) if returns.std() != 0 else 0,
        'max_drawdown': float((returns.cumsum() - returns.cumsum().cummax()).min()),
        'win_rate': float((returns > 0).sum() / len(returns))
    }
    
    return metrics


# ===== TAX-LOSS HARVESTING FUNCTIONS =====
def identify_tax_loss_opportunities(positions: List[Dict], threshold: float = -5.0) -> List[Dict]:
    """Identify positions with losses for tax-loss harvesting"""
    opportunities = []
    
    for pos in positions:
        market_value = pos['shares'] * pos['current_price']
        cost_basis = pos['shares'] * pos['avg_cost']
        gain_loss = market_value - cost_basis
        gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
        
        if gain_loss_pct < threshold:
            opportunities.append({
                'symbol': pos['symbol'],
                'shares': pos['shares'],
                'cost_basis': cost_basis,
                'market_value': market_value,
                'loss': gain_loss,
                'loss_pct': gain_loss_pct,
                'sector': pos.get('sector', 'Unknown'),
                'recommendation': f"Consider selling to harvest ${abs(gain_loss):,.2f} loss for tax purposes"
            })
    
    return sorted(opportunities, key=lambda x: x['loss'])


def suggest_replacement_securities(symbol: str, sector: str) -> List[str]:
    """Suggest similar securities to avoid wash sale rule (simplified)"""
    # Sector ETF alternatives to maintain exposure
    sector_etfs = {
        'Technology': ['XLK', 'VGT', 'QQQ'],
        'Healthcare': ['XLV', 'VHT', 'IHI'],
        'Financial': ['XLF', 'VFH', 'KBE'],
        'Consumer': ['XLY', 'XLP', 'VCR'],
        'Energy': ['XLE', 'VDE', 'IYE'],
        'Industrial': ['XLI', 'VIS', 'IYJ'],
        'Materials': ['XLB', 'VAW', 'IYM'],
        'Utilities': ['XLU', 'VPU', 'IDU'],
        'Real Estate': ['XLRE', 'VNQ', 'IYR']
    }
    
    return sector_etfs.get(sector, ['SPY', 'VOO', 'VTI'])  # Default to broad market ETFs


# ===== CORRELATION ANALYSIS FUNCTIONS =====
def calculate_correlation_matrix(positions: List[Dict], period: str = "1y") -> pd.DataFrame:
    """Calculate correlation matrix between portfolio positions"""
    symbols = [pos['symbol'] for pos in positions]
    
    if len(symbols) < 2:
        return pd.DataFrame()
    
    # Fetch historical data for all symbols
    price_data = {}
    for symbol in symbols:
        hist = fetch_historical_prices(symbol, period)
        if not hist.empty:
            price_data[symbol] = hist['Close']
    
    if len(price_data) < 2:
        return pd.DataFrame()
    
    # Create DataFrame with all prices
    df = pd.DataFrame(price_data)
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Calculate correlation
    correlation = returns.corr()
    
    return correlation


def identify_correlation_risks(correlation_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    """Identify highly correlated positions"""
    if correlation_matrix.empty:
        return []
    
    risks = []
    
    # Find pairs with high correlation
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            
            if abs(corr_value) > threshold:
                risks.append({
                    'symbol1': correlation_matrix.index[i],
                    'symbol2': correlation_matrix.columns[j],
                    'correlation': float(corr_value),
                    'risk_level': 'High' if abs(corr_value) > 0.85 else 'Moderate',
                    'message': f"High correlation ({corr_value:.2f}) suggests these positions may move together, reducing diversification benefits"
                })
    
    return sorted(risks, key=lambda x: abs(x['correlation']), reverse=True)


# ===== MONTE CARLO SIMULATION FUNCTIONS =====
def run_monte_carlo_simulation(
    positions: List[Dict],
    simulations: int = 1000,
    days: int = 252,
    confidence_level: float = 0.95
) -> Dict:
    """Run Monte Carlo simulation for portfolio"""
    
    # Calculate current portfolio value
    current_value = calculate_portfolio_value(positions)
    
    if current_value == 0:
        return {}
    
    # Fetch historical data and calculate returns
    symbols = [pos['symbol'] for pos in positions]
    weights = [pos['shares'] * pos['current_price'] / current_value for pos in positions]
    
    # Fetch historical prices
    returns_data = {}
    for symbol in symbols:
        hist = fetch_historical_prices(symbol, "1y")
        if not hist.empty:
            returns_data[symbol] = hist['Close'].pct_change().dropna()
    
    if not returns_data:
        return {}
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 30:  # Need at least 30 days of data
        return {}
    
    # Calculate mean returns and covariance
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    # Run simulations
    portfolio_results = np.zeros((simulations, days))
    
    for i in range(simulations):
        # Generate random returns using multivariate normal distribution
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, days)
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(simulated_returns, weights)
        
        # Calculate cumulative portfolio value
        portfolio_results[i] = current_value * (1 + portfolio_returns).cumprod()
    
    # Calculate statistics
    final_values = portfolio_results[:, -1]
    
    results = {
        'current_value': float(current_value),
        'simulations': simulations,
        'days': days,
        'mean_final_value': float(np.mean(final_values)),
        'median_final_value': float(np.median(final_values)),
        'std_final_value': float(np.std(final_values)),
        'min_final_value': float(np.min(final_values)),
        'max_final_value': float(np.max(final_values)),
        'percentile_5': float(np.percentile(final_values, 5)),
        'percentile_25': float(np.percentile(final_values, 25)),
        'percentile_75': float(np.percentile(final_values, 75)),
        'percentile_95': float(np.percentile(final_values, 95)),
        'probability_profit': float((final_values > current_value).sum() / simulations),
        'var_95': float(current_value - np.percentile(final_values, 5)),  # Value at Risk
        'all_simulations': portfolio_results
    }
    
    return results


# Helper functions
def calculate_portfolio_value(positions: List[Dict]) -> float:
    """Calculate total portfolio value"""
    return sum(pos['shares'] * pos['current_price'] for pos in positions)


def calculate_position_percentage(position: Dict, total_value: float) -> float:
    """Calculate position as percentage of total portfolio"""
    if total_value == 0:
        return 0
    return (position['shares'] * position['current_price']) / total_value * 100


def calculate_asset_allocation(positions: List[Dict]) -> Dict[str, float]:
    """Calculate current asset class allocation"""
    total_value = calculate_portfolio_value(positions)
    if total_value == 0:
        return {}
    
    allocation = {}
    for pos in positions:
        asset_class = pos.get('asset_class', 'stocks')
        value = pos['shares'] * pos['current_price']
        allocation[asset_class] = allocation.get(asset_class, 0) + value
    
    # Convert to percentages
    return {k: (v / total_value * 100) for k, v in allocation.items()}


def calculate_sector_allocation(positions: List[Dict]) -> Dict[str, float]:
    """Calculate sector allocation"""
    total_value = calculate_portfolio_value(positions)
    if total_value == 0:
        return {}
    
    allocation = {}
    for pos in positions:
        sector = pos.get('sector', 'Unknown')
        value = pos['shares'] * pos['current_price']
        allocation[sector] = allocation.get(sector, 0) + value
    
    # Convert to percentages
    return {k: (v / total_value * 100) for k, v in allocation.items()}


def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
    """Calculate Kelly Criterion for position sizing"""
    if win_loss_ratio <= 0:
        return 0
    return (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio


def calculate_position_size(
    method: str,
    portfolio_value: float,
    volatility: float = None,
    win_rate: float = None,
    win_loss_ratio: float = None
) -> float:
    """Calculate position size based on method"""
    if method == "fixed_percent":
        return portfolio_value * 0.05  # 5% fixed
    
    elif method == "risk_adjusted":
        if volatility is None:
            volatility = 0.02  # Default 2% volatility
        base_size = portfolio_value * 0.05
        # Adjust for volatility (lower size for higher volatility)
        adjusted_size = base_size * (0.02 / max(volatility, 0.01))
        return min(adjusted_size, portfolio_value * 0.15)  # Max 15%
    
    elif method == "kelly":
        if win_rate is None or win_loss_ratio is None:
            return portfolio_value * 0.05
        kelly_pct = kelly_criterion(win_rate, win_loss_ratio)
        # Use half-Kelly for safety
        return portfolio_value * min(kelly_pct / 2, 0.15)
    
    return portfolio_value * 0.05


def generate_mentor_feedback(
    positions: List[Dict],
    target_allocation: Dict[str, float],
    risk_profile: str,
    max_position_size: float
) -> List[Dict[str, str]]:
    """Generate AI mentor feedback on portfolio"""
    feedback = []
    total_value = calculate_portfolio_value(positions)
    
    if total_value == 0:
        return [{
            'type': 'info',
            'title': 'Start Building Your Portfolio',
            'message': 'Your portfolio is empty. Consider starting with a diversified mix of stocks across different sectors.'
        }]
    
    # Check for overconcentration in single positions
    for pos in positions:
        pct = calculate_position_percentage(pos, total_value)
        if pct > max_position_size:
            feedback.append({
                'type': 'warning',
                'title': f'⚠️ Overconcentration in {pos["symbol"]}',
                'message': f'{pos["symbol"]} now represents {pct:.1f}% of your portfolio, exceeding your {max_position_size}% limit. This increases company-specific risk. Consider reducing to {max_position_size}% and diversifying into other sectors like healthcare or consumer staples.'
            })
    
    # Check asset allocation vs target
    current_allocation = calculate_asset_allocation(positions)
    for asset_class, target_pct in target_allocation.items():
        current_pct = current_allocation.get(asset_class, 0)
        diff = abs(current_pct - target_pct)
        
        if diff > 10:  # More than 10% deviation
            if current_pct > target_pct:
                feedback.append({
                    'type': 'warning',
                    'title': f'📊 {asset_class.title()} Overweight',
                    'message': f'Your {asset_class} allocation is {current_pct:.1f}%, above your {target_pct}% target. Consider rebalancing by selling some {asset_class} and buying underweight asset classes.'
                })
            else:
                feedback.append({
                    'type': 'info',
                    'title': f'📊 {asset_class.title()} Underweight',
                    'message': f'Your {asset_class} allocation is {current_pct:.1f}%, below your {target_pct}% target. Consider adding more exposure to {asset_class} to match your target allocation.'
                })
    
    # Check sector concentration
    sector_allocation = calculate_sector_allocation(positions)
    max_sector = max(sector_allocation.items(), key=lambda x: x[1]) if sector_allocation else None
    
    if max_sector and max_sector[1] > 30:
        feedback.append({
            'type': 'warning',
            'title': f'🏢 Sector Concentration: {max_sector[0]}',
            'message': f'The {max_sector[0]} sector represents {max_sector[1]:.1f}% of your portfolio. High sector concentration increases risk from sector-specific events. Consider diversifying into technology, healthcare, or financial sectors.'
        })
    
    # Risk profile advice
    if risk_profile == 'conservative':
        stocks_pct = current_allocation.get('stocks', 0)
        if stocks_pct > 50:
            feedback.append({
                'type': 'warning',
                'title': '🛡️ Risk Profile Mismatch',
                'message': f'Your portfolio has {stocks_pct:.1f}% in stocks, which may be too aggressive for a conservative risk profile. Consider increasing bonds allocation to reduce volatility.'
            })
    
    elif risk_profile == 'aggressive':
        bonds_pct = current_allocation.get('bonds', 0)
        if bonds_pct > 30:
            feedback.append({
                'type': 'info',
                'title': '🚀 Growth Opportunity',
                'message': f'With an aggressive risk profile and {bonds_pct:.1f}% in bonds, you might be missing growth opportunities. Consider shifting more into growth stocks or commodities if you can handle the volatility.'
            })
    
    # Scenario analysis
    feedback.append({
        'type': 'scenario',
        'title': '📈 Interest Rate Scenario',
        'message': 'If interest rates rise by 1%, bond prices typically fall 5-10%, but financial sector stocks (banks, insurance) often benefit from higher margins. Consider balancing your exposure.'
    })
    
    if not feedback:
        feedback.append({
            'type': 'success',
            'title': '✅ Portfolio Looks Balanced',
            'message': 'Your portfolio allocation aligns well with your targets and risk profile. Continue monitoring for rebalancing opportunities.'
        })
    
    return feedback


def calculate_rebalancing_trades(
    positions: List[Dict],
    target_allocation: Dict[str, float],
    total_value: float
) -> List[Dict]:
    """Calculate trades needed to rebalance portfolio"""
    trades = []
    current_allocation = calculate_asset_allocation(positions)
    
    # Group positions by asset class
    positions_by_class = {}
    for pos in positions:
        asset_class = pos.get('asset_class', 'stocks')
        if asset_class not in positions_by_class:
            positions_by_class[asset_class] = []
        positions_by_class[asset_class].append(pos)
    
    # Calculate trades for each asset class
    for asset_class, target_pct in target_allocation.items():
        current_pct = current_allocation.get(asset_class, 0)
        target_value = total_value * target_pct / 100
        current_value = total_value * current_pct / 100
        diff_value = target_value - current_value
        
        if abs(diff_value) > total_value * 0.01:  # Only if > 1% of portfolio
            action = "BUY" if diff_value > 0 else "SELL"
            trades.append({
                'asset_class': asset_class,
                'action': action,
                'amount': abs(diff_value),
                'current_pct': current_pct,
                'target_pct': target_pct
            })
    
    return trades


# Main UI
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Portfolio Overview",
    "⚖️ Asset Allocation",
    "🎯 Position Sizing",
    "🤖 AI Mentor",
    "📈 Performance Tracking",
    "💰 Tax-Loss Harvesting",
    "🔗 Correlation Analysis",
    "🎲 Monte Carlo Simulation"
])

with tab1:
    st.header("Portfolio Overview")
    
    # Live price update button
    if st.session_state.portfolio['positions']:
        col_update1, col_update2, col_update3 = st.columns([2, 1, 1])
        with col_update2:
            if st.button("🔄 Update Live Prices", use_container_width=True):
                with st.spinner("Fetching live prices from Yahoo Finance..."):
                    st.session_state.portfolio['positions'] = update_portfolio_prices(
                        st.session_state.portfolio['positions']
                    )
                    save_portfolio_snapshot()  # Save snapshot after update
                    st.success("✅ Prices updated!")
                    st.rerun()
        
        with col_update3:
            if st.button("💾 Save Snapshot", use_container_width=True):
                save_portfolio_snapshot()
                st.success(f"✅ Snapshot saved! Total: {len(st.session_state.performance_snapshots)}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add New Position")
        
        add_col1, add_col2, add_col3, add_col4 = st.columns(4)
        
        with add_col1:
            new_symbol = st.text_input("Symbol", placeholder="AAPL", key="new_symbol")
        
        with add_col2:
            new_shares = st.number_input("Shares", min_value=0.0, step=1.0, key="new_shares")
        
        with add_col3:
            new_cost = st.number_input("Avg Cost ($)", min_value=0.0, step=0.01, key="new_cost")
        
        with add_col4:
            new_price = st.number_input("Current Price ($)", min_value=0.0, step=0.01, key="new_price")
        
        add_col5, add_col6, add_col7 = st.columns(3)
        
        with add_col5:
            new_sector = st.selectbox(
                "Sector",
                ["Technology", "Healthcare", "Financial", "Consumer", "Energy", "Industrial", "Materials", "Utilities", "Real Estate"],
                key="new_sector"
            )
        
        with add_col6:
            new_asset_class = st.selectbox(
                "Asset Class",
                ["stocks", "bonds", "commodities"],
                key="new_asset_class"
            )
        
        with add_col7:
            st.write("")  # Spacing
            st.write("")
            if st.button("➕ Add Position", use_container_width=True):
                if new_symbol and new_shares > 0 and new_cost > 0 and new_price > 0:
                    st.session_state.portfolio['positions'].append({
                        'symbol': new_symbol.upper(),
                        'shares': new_shares,
                        'avg_cost': new_cost,
                        'current_price': new_price,
                        'sector': new_sector,
                        'asset_class': new_asset_class,
                        'added_date': datetime.now().strftime("%Y-%m-%d")
                    })
                    st.success(f"✅ Added {new_shares} shares of {new_symbol.upper()}")
                    st.rerun()
                else:
                    st.error("Please fill all fields with valid values")
    
    with col2:
        st.subheader("Portfolio Summary")
        total_value = calculate_portfolio_value(st.session_state.portfolio['positions'])
        st.metric("Total Value", f"${total_value:,.2f}")
        st.metric("Positions", len(st.session_state.portfolio['positions']))
        
        if total_value > 0:
            total_cost = sum(pos['shares'] * pos['avg_cost'] for pos in st.session_state.portfolio['positions'])
            total_gain = total_value - total_cost
            gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
            st.metric("Total Gain/Loss", f"${total_gain:,.2f}", f"{gain_pct:+.2f}%")
    
    # Display positions
    if st.session_state.portfolio['positions']:
        st.subheader("Current Positions")
        
        positions_data = []
        total_value = calculate_portfolio_value(st.session_state.portfolio['positions'])
        
        for i, pos in enumerate(st.session_state.portfolio['positions']):
            market_value = pos['shares'] * pos['current_price']
            cost_basis = pos['shares'] * pos['avg_cost']
            gain_loss = market_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
            allocation_pct = calculate_position_percentage(pos, total_value)
            
            positions_data.append({
                'Symbol': pos['symbol'],
                'Shares': f"{pos['shares']:.2f}",
                'Avg Cost': f"${pos['avg_cost']:.2f}",
                'Current Price': f"${pos['current_price']:.2f}",
                'Market Value': f"${market_value:,.2f}",
                'Gain/Loss': f"${gain_loss:,.2f}",
                'Gain/Loss %': f"{gain_loss_pct:+.2f}%",
                'Allocation %': f"{allocation_pct:.1f}%",
                'Sector': pos['sector'],
                'Asset Class': pos['asset_class'],
                'Index': i
            })
        
        df_positions = pd.DataFrame(positions_data)
        
        # Display without index column
        display_df = df_positions.drop('Index', axis=1)
        st.dataframe(display_df, use_container_width=True)
        
        # Remove position button
        st.subheader("Remove Position")
        remove_col1, remove_col2 = st.columns([3, 1])
        with remove_col1:
            remove_idx = st.selectbox(
                "Select position to remove",
                range(len(st.session_state.portfolio['positions'])),
                format_func=lambda x: f"{st.session_state.portfolio['positions'][x]['symbol']} ({st.session_state.portfolio['positions'][x]['shares']} shares)"
            )
        with remove_col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Remove", use_container_width=True):
                removed = st.session_state.portfolio['positions'].pop(remove_idx)
                st.success(f"✅ Removed {removed['symbol']}")
                st.rerun()
    else:
        st.info("📝 No positions in portfolio. Add your first position above!")

with tab2:
    st.header("Asset Allocation & Rebalancing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Allocation")
        
        target_stocks = st.slider(
            "Stocks %",
            0, 100,
            st.session_state.portfolio['target_allocation']['stocks'],
            key="target_stocks"
        )
        
        target_bonds = st.slider(
            "Bonds %",
            0, 100,
            st.session_state.portfolio['target_allocation']['bonds'],
            key="target_bonds"
        )
        
        target_commodities = st.slider(
            "Commodities %",
            0, 100,
            st.session_state.portfolio['target_allocation']['commodities'],
            key="target_commodities"
        )
        
        total_target = target_stocks + target_bonds + target_commodities
        
        if total_target != 100:
            st.warning(f"⚠️ Total allocation: {total_target}% (should be 100%)")
        else:
            st.success("✅ Target allocation sums to 100%")
        
        if st.button("💾 Save Target Allocation"):
            st.session_state.portfolio['target_allocation'] = {
                'stocks': target_stocks,
                'bonds': target_bonds,
                'commodities': target_commodities
            }
            st.success("✅ Target allocation saved!")
    
    with col2:
        st.subheader("Current Allocation")
        
        if st.session_state.portfolio['positions']:
            current_allocation = calculate_asset_allocation(st.session_state.portfolio['positions'])
            
            # Display current vs target
            comparison_data = []
            for asset_class in ['stocks', 'bonds', 'commodities']:
                current = current_allocation.get(asset_class, 0)
                target = st.session_state.portfolio['target_allocation'][asset_class]
                diff = current - target
                
                comparison_data.append({
                    'Asset Class': asset_class.title(),
                    'Current %': f"{current:.1f}%",
                    'Target %': f"{target:.1f}%",
                    'Difference': f"{diff:+.1f}%"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(current_allocation.keys()),
                values=list(current_allocation.values()),
                hole=0.3
            )])
            fig.update_layout(
                title="Current Asset Allocation",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Add positions to see current allocation")
    
    # Rebalancing section
    st.subheader("⚖️ Rebalancing")
    
    rebal_col1, rebal_col2, rebal_col3 = st.columns(3)
    
    with rebal_col1:
        st.session_state.portfolio['rebalance_threshold'] = st.number_input(
            "Rebalance Threshold (%)",
            min_value=1,
            max_value=50,
            value=st.session_state.portfolio['rebalance_threshold'],
            help="Rebalance when allocation deviates by this percentage"
        )
    
    with rebal_col2:
        rebalance_frequency = st.selectbox(
            "Rebalance Frequency",
            ["Monthly", "Quarterly", "Semi-Annually", "Annually"],
            index=2
        )
    
    with rebal_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Calculate Rebalancing Trades", use_container_width=True):
            if st.session_state.portfolio['positions']:
                total_value = calculate_portfolio_value(st.session_state.portfolio['positions'])
                trades = calculate_rebalancing_trades(
                    st.session_state.portfolio['positions'],
                    st.session_state.portfolio['target_allocation'],
                    total_value
                )
                
                if trades:
                    st.subheader("Recommended Trades")
                    for trade in trades:
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            emoji = "🔴" if trade['action'] == "SELL" else "🟢"
                            st.write(f"{emoji} **{trade['action']}** {trade['asset_class'].title()}")
                        with col_b:
                            st.write(f"${trade['amount']:,.2f}")
                        with col_c:
                            st.write(f"{trade['current_pct']:.1f}% → {trade['target_pct']:.1f}%")
                else:
                    st.success("✅ Portfolio is balanced! No trades needed.")
            else:
                st.warning("Add positions first")

with tab3:
    st.header("Position Sizing Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Position Sizing Settings")
        
        sizing_method = st.selectbox(
            "Position Sizing Method",
            ["fixed_percent", "risk_adjusted", "kelly"],
            format_func=lambda x: {
                "fixed_percent": "Fixed Percentage (5%)",
                "risk_adjusted": "Risk-Adjusted (Volatility-based)",
                "kelly": "Kelly Criterion"
            }[x]
        )
        
        st.session_state.portfolio['max_position_size'] = st.slider(
            "Max Position Size (%)",
            5, 30,
            st.session_state.portfolio['max_position_size'],
            help="Maximum percentage of portfolio for any single position"
        )
        
        # Method-specific inputs
        if sizing_method == "risk_adjusted":
            volatility = st.slider(
                "Expected Volatility",
                0.01, 0.10,
                0.02,
                0.01,
                format="%.2f",
                help="Historical or expected volatility of the asset"
            )
        elif sizing_method == "kelly":
            win_rate = st.slider(
                "Win Rate",
                0.0, 1.0,
                0.55,
                0.05,
                help="Probability of winning trade"
            )
            win_loss_ratio = st.slider(
                "Win/Loss Ratio",
                0.5, 5.0,
                2.0,
                0.1,
                help="Average win divided by average loss"
            )
    
    with col2:
        st.subheader("Position Size Calculator")
        
        calc_portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=1000.0,
            value=calculate_portfolio_value(st.session_state.portfolio['positions']) or 100000.0,
            step=1000.0
        )
        
        # Calculate position size
        if sizing_method == "fixed_percent":
            position_size = calculate_position_size("fixed_percent", calc_portfolio_value)
        elif sizing_method == "risk_adjusted":
            position_size = calculate_position_size("risk_adjusted", calc_portfolio_value, volatility=volatility)
        else:  # kelly
            position_size = calculate_position_size("kelly", calc_portfolio_value, win_rate=win_rate, win_loss_ratio=win_loss_ratio)
        
        position_pct = (position_size / calc_portfolio_value * 100)
        
        st.metric("Recommended Position Size", f"${position_size:,.2f}")
        st.metric("As % of Portfolio", f"{position_pct:.2f}%")
        
        # Example calculation
        example_price = st.number_input("Stock Price ($)", min_value=0.01, value=100.0, step=0.01)
        shares = int(position_size / example_price)
        
        st.info(f"💡 **Example:** At ${example_price:.2f}/share, buy **{shares} shares** (${shares * example_price:,.2f})")
    
    # Position sizing comparison
    st.subheader("📊 Method Comparison")
    
    methods_comparison = []
    portfolio_val = calculate_portfolio_value(st.session_state.portfolio['positions']) or 100000
    
    for method in ["fixed_percent", "risk_adjusted", "kelly"]:
        if method == "risk_adjusted":
            size = calculate_position_size(method, portfolio_val, volatility=0.02)
        elif method == "kelly":
            size = calculate_position_size(method, portfolio_val, win_rate=0.55, win_loss_ratio=2.0)
        else:
            size = calculate_position_size(method, portfolio_val)
        
        methods_comparison.append({
            'Method': method.replace('_', ' ').title(),
            'Position Size': f"${size:,.2f}",
            '% of Portfolio': f"{size/portfolio_val*100:.2f}%"
        })
    
    df_methods = pd.DataFrame(methods_comparison)
    st.dataframe(df_methods, use_container_width=True)

with tab4:
    st.header("🤖 AI Portfolio Mentor")
    
    st.markdown("""
    Get personalized feedback on your portfolio allocation, risk exposure, and potential improvements.
    The AI mentor analyzes your positions and provides human-like guidance.
    """)
    
    # Risk profile selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.session_state.portfolio['risk_profile'] = st.selectbox(
            "Risk Profile",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(st.session_state.portfolio['risk_profile'])
        )
    
    with col2:
        risk_descriptions = {
            "conservative": "🛡️ Conservative: Focus on capital preservation, lower volatility, higher bond allocation",
            "moderate": "⚖️ Moderate: Balanced approach between growth and stability",
            "aggressive": "🚀 Aggressive: Focus on growth, higher risk tolerance, higher stock allocation"
        }
        st.info(risk_descriptions[st.session_state.portfolio['risk_profile']])
    
    # Generate and display feedback
    if st.button("🔍 Get Portfolio Analysis", use_container_width=True):
        with st.spinner("Analyzing your portfolio..."):
            feedback_list = generate_mentor_feedback(
                st.session_state.portfolio['positions'],
                st.session_state.portfolio['target_allocation'],
                st.session_state.portfolio['risk_profile'],
                st.session_state.portfolio['max_position_size']
            )
            
            st.session_state.last_feedback = feedback_list
    
    # Display feedback
    if hasattr(st.session_state, 'last_feedback'):
        st.subheader("💬 Mentor Feedback")
        
        for feedback in st.session_state.last_feedback:
            if feedback['type'] == 'warning':
                with st.expander(feedback['title'], expanded=True):
                    st.warning(feedback['message'])
            elif feedback['type'] == 'success':
                with st.expander(feedback['title'], expanded=True):
                    st.success(feedback['message'])
            elif feedback['type'] == 'scenario':
                with st.expander(feedback['title'], expanded=False):
                    st.info(feedback['message'])
            else:
                with st.expander(feedback['title'], expanded=True):
                    st.info(feedback['message'])
    
    # Sector allocation visualization
    if st.session_state.portfolio['positions']:
        st.subheader("📊 Sector Diversification")
        
        sector_allocation = calculate_sector_allocation(st.session_state.portfolio['positions'])
        
        if sector_allocation:
            fig = px.bar(
                x=list(sector_allocation.keys()),
                y=list(sector_allocation.values()),
                labels={'x': 'Sector', 'y': 'Allocation (%)'},
                title='Sector Allocation',
                color=list(sector_allocation.values()),
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Concentration warning
            max_sector_pct = max(sector_allocation.values())
            if max_sector_pct > 30:
                st.warning(f"⚠️ High concentration detected: {max_sector_pct:.1f}% in one sector. Consider diversifying.")

with tab5:
    st.header("📈 Historical Performance Tracking")
    
    if st.session_state.performance_snapshots:
        st.info(f"📊 Tracking {len(st.session_state.performance_snapshots)} portfolio snapshots")
        
        # Calculate returns
        returns_df = calculate_portfolio_returns(st.session_state.performance_snapshots)
        
        if not returns_df.empty:
            # Performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=returns_df['date'],
                y=returns_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            if len(returns_df) > 1:
                metrics = calculate_portfolio_metrics(returns_df['daily_return'])
                
                if metrics:
                    st.subheader("📊 Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Return",
                            f"{metrics['total_return']*100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Annualized Return",
                            f"{metrics['annualized_return']*100:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Volatility (Annual)",
                            f"{metrics['volatility']*100:.2f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Sharpe Ratio",
                            f"{metrics['sharpe_ratio']:.2f}"
                        )
                    
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.metric(
                            "Max Drawdown",
                            f"{metrics['max_drawdown']*100:.2f}%"
                        )
                    
                    with col6:
                        st.metric(
                            "Win Rate",
                            f"{metrics['win_rate']*100:.1f}%"
                        )
                    
                    with col7:
                        current_val = returns_df['value'].iloc[-1]
                        initial_val = returns_df['value'].iloc[0]
                        st.metric(
                            "Start Value",
                            f"${initial_val:,.2f}"
                        )
                    
                    with col8:
                        st.metric(
                            "Current Value",
                            f"${current_val:,.2f}",
                            f"{((current_val - initial_val) / initial_val * 100):+.2f}%"
                        )
            
            # Cumulative returns chart
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=returns_df['date'],
                y=returns_df['cumulative_return'] * 100,
                mode='lines',
                name='Cumulative Return',
                fill='tozeroy',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig2.update_layout(
                title="Cumulative Returns (%)",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=300
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Historical data table
            with st.expander("📋 View Historical Data"):
                history_data = []
                for snap in st.session_state.performance_snapshots[-30:]:  # Last 30 snapshots
                    history_data.append({
                        'Date': snap['timestamp'][:10],
                        'Value': f"${snap['total_value']:,.2f}",
                        'Positions': snap['positions']
                    })
                
                st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear Performance History"):
            st.session_state.performance_snapshots = []
            st.success("✅ History cleared")
            st.rerun()
    
    else:
        st.info("📊 No performance history yet. Click 'Save Snapshot' in the Portfolio Overview tab to start tracking!")
        st.markdown("""
        **How to use Performance Tracking:**
        1. Add positions to your portfolio
        2. Click "Update Live Prices" to get current values
        3. Click "Save Snapshot" to record current state
        4. Repeat regularly (daily, weekly, etc.) to build history
        5. Return to this tab to see performance charts and metrics
        """)

with tab6:
    st.header("💰 Tax-Loss Harvesting Opportunities")
    
    st.markdown("""
    Identify positions with losses that can be sold for tax benefits. 
    The IRS allows you to offset capital gains with losses, reducing your tax bill.
    """)
    
    if st.session_state.portfolio['positions']:
        # Settings
        col1, col2 = st.columns(2)
        
        with col1:
            loss_threshold = st.slider(
                "Minimum Loss Threshold (%)",
                -50, 0,
                -5,
                1,
                help="Only show positions with losses greater than this threshold"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Identify Opportunities", use_container_width=True):
                opportunities = identify_tax_loss_opportunities(
                    st.session_state.portfolio['positions'],
                    loss_threshold
                )
                
                st.session_state.tax_loss_opportunities = opportunities
        
        # Display opportunities
        if hasattr(st.session_state, 'tax_loss_opportunities'):
            opportunities = st.session_state.tax_loss_opportunities
            
            if opportunities:
                st.subheader(f"📉 Found {len(opportunities)} Tax-Loss Harvesting Opportunities")
                
                for opp in opportunities:
                    with st.expander(f"💼 {opp['symbol']} - Loss: ${abs(opp['loss']):,.2f} ({opp['loss_pct']:.2f}%)", expanded=True):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.write("**Position Details:**")
                            st.write(f"- Shares: {opp['shares']:.2f}")
                            st.write(f"- Cost Basis: ${opp['cost_basis']:,.2f}")
                            st.write(f"- Market Value: ${opp['market_value']:,.2f}")
                        
                        with col_b:
                            st.write("**Tax Benefit:**")
                            # Estimate tax savings (assuming 30% tax rate)
                            tax_savings = abs(opp['loss']) * 0.30
                            st.write(f"- Realized Loss: ${abs(opp['loss']):,.2f}")
                            st.write(f"- Est. Tax Savings: ${tax_savings:,.2f}")
                            st.write(f"- Loss %: {opp['loss_pct']:.2f}%")
                        
                        with col_c:
                            st.write("**Replacement Options:**")
                            st.write("To avoid wash sale, wait 30 days or buy:")
                            replacements = suggest_replacement_securities(opp['symbol'], opp['sector'])
                            for rep in replacements:
                                st.write(f"- {rep} (Sector ETF)")
                        
                        st.info(f"💡 {opp['recommendation']}")
                        
                        st.warning("⚠️ **Wash Sale Rule:** If you buy the same or substantially identical security within 30 days before or after the sale, the loss will be disallowed for tax purposes.")
            else:
                st.success("✅ No tax-loss harvesting opportunities found based on your threshold. Your portfolio is performing well!")
        
        # Summary statistics
        st.subheader("📊 Portfolio Loss Summary")
        
        total_losses = 0
        total_gains = 0
        
        for pos in st.session_state.portfolio['positions']:
            market_value = pos['shares'] * pos['current_price']
            cost_basis = pos['shares'] * pos['avg_cost']
            gain_loss = market_value - cost_basis
            
            if gain_loss < 0:
                total_losses += abs(gain_loss)
            else:
                total_gains += gain_loss
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Unrealized Losses", f"${total_losses:,.2f}")
        
        with col2:
            st.metric("Total Unrealized Gains", f"${total_gains:,.2f}")
        
        with col3:
            net = total_gains - total_losses
            st.metric("Net Unrealized", f"${net:,.2f}", f"{'+' if net >= 0 else ''}{net:,.2f}")
    
    else:
        st.info("📝 Add positions to identify tax-loss harvesting opportunities")

with tab7:
    st.header("🔗 Portfolio Correlation Analysis")
    
    st.markdown("""
    Analyze how your positions move together. High correlation means positions tend to move in the same direction,
    reducing diversification benefits. Low or negative correlation provides better risk management.
    """)
    
    if len(st.session_state.portfolio['positions']) >= 2:
        # Calculate correlation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            analysis_period = st.selectbox(
                "Analysis Period",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=3,
                format_func=lambda x: {
                    "1mo": "1 Month",
                    "3mo": "3 Months",
                    "6mo": "6 Months",
                    "1y": "1 Year",
                    "2y": "2 Years"
                }[x]
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("📊 Calculate Correlation", use_container_width=True):
                with st.spinner("Calculating correlation matrix..."):
                    correlation_matrix = calculate_correlation_matrix(
                        st.session_state.portfolio['positions'],
                        analysis_period
                    )
                    
                    st.session_state.correlation_matrix = correlation_matrix
        
        # Display correlation matrix
        if hasattr(st.session_state, 'correlation_matrix') and not st.session_state.correlation_matrix.empty:
            corr_matrix = st.session_state.correlation_matrix
            
            st.subheader("📊 Correlation Matrix")
            
            # Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdYlGn_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Portfolio Correlation Heatmap",
                xaxis_title="Symbol",
                yaxis_title="Symbol",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Identify high correlations
            st.subheader("⚠️ Correlation Risks")
            
            correlation_threshold = st.slider(
                "High Correlation Threshold",
                0.5, 0.95,
                0.7,
                0.05,
                help="Correlations above this level will be flagged"
            )
            
            risks = identify_correlation_risks(corr_matrix, correlation_threshold)
            
            if risks:
                st.warning(f"Found {len(risks)} pairs with high correlation")
                
                for risk in risks:
                    emoji = "🔴" if risk['risk_level'] == 'High' else "🟡"
                    with st.expander(f"{emoji} {risk['symbol1']} ↔ {risk['symbol2']}: {risk['correlation']:.2f}"):
                        st.write(f"**Risk Level:** {risk['risk_level']}")
                        st.write(f"**Correlation:** {risk['correlation']:.2f}")
                        st.info(risk['message'])
                        
                        if abs(risk['correlation']) > 0.85:
                            st.error(f"⚠️ **Very High Correlation!** Consider reducing exposure to one of these positions to improve diversification.")
            else:
                st.success("✅ No high correlation pairs found. Your portfolio has good diversification!")
            
            # Correlation interpretation guide
            with st.expander("📖 How to Interpret Correlation"):
                st.markdown("""
                **Correlation Coefficient Ranges:**
                - **+1.0**: Perfect positive correlation (move exactly together)
                - **+0.7 to +1.0**: Strong positive correlation
                - **+0.3 to +0.7**: Moderate positive correlation
                - **-0.3 to +0.3**: Weak or no correlation
                - **-0.7 to -0.3**: Moderate negative correlation
                - **-1.0 to -0.7**: Strong negative correlation
                - **-1.0**: Perfect negative correlation (move in opposite directions)
                
                **Diversification Tips:**
                - Aim for correlations < 0.7 between positions
                - Negative correlations provide hedging benefits
                - Mix asset classes (stocks, bonds, commodities) for lower correlation
                - Diversify across sectors and geographies
                """)
        
        else:
            st.info("👆 Click 'Calculate Correlation' to analyze your portfolio")
    
    else:
        st.info("📝 Add at least 2 positions to analyze correlations")

with tab8:
    st.header("🎲 Monte Carlo Risk Simulation")
    
    st.markdown("""
    Run Monte Carlo simulations to understand potential future portfolio outcomes based on historical volatility.
    This helps visualize the range of possible returns and assess downside risk.
    """)
    
    if st.session_state.portfolio['positions']:
        # Simulation settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_simulations = st.select_slider(
                "Number of Simulations",
                options=[100, 500, 1000, 2500, 5000],
                value=1000
            )
        
        with col2:
            simulation_days = st.select_slider(
                "Simulation Period (Trading Days)",
                options=[21, 63, 126, 252, 504],
                value=252,
                format_func=lambda x: f"{x} days (~{x//21} months)"
            )
        
        with col3:
            st.write("")
            st.write("")
            if st.button("🎲 Run Simulation", use_container_width=True):
                with st.spinner(f"Running {num_simulations} simulations..."):
                    results = run_monte_carlo_simulation(
                        st.session_state.portfolio['positions'],
                        simulations=num_simulations,
                        days=simulation_days
                    )
                    
                    st.session_state.monte_carlo_results = results
        
        # Display results
        if hasattr(st.session_state, 'monte_carlo_results') and st.session_state.monte_carlo_results:
            results = st.session_state.monte_carlo_results
            
            st.success(f"✅ Completed {results['simulations']} simulations over {results['days']} days")
            
            # Key metrics
            st.subheader("📊 Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Value",
                    f"${results['current_value']:,.2f}"
                )
            
            with col2:
                mean_change = results['mean_final_value'] - results['current_value']
                mean_change_pct = (mean_change / results['current_value'] * 100)
                st.metric(
                    "Mean Final Value",
                    f"${results['mean_final_value']:,.2f}",
                    f"{mean_change_pct:+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Median Final Value",
                    f"${results['median_final_value']:,.2f}"
                )
            
            with col4:
                st.metric(
                    "Probability of Profit",
                    f"{results['probability_profit']*100:.1f}%"
                )
            
            # Percentile metrics
            st.subheader("📈 Outcome Percentiles")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                loss_5 = results['current_value'] - results['percentile_5']
                st.metric(
                    "5th Percentile (Worst 5%)",
                    f"${results['percentile_5']:,.2f}",
                    f"-${loss_5:,.2f}"
                )
            
            with col2:
                st.metric(
                    "25th Percentile",
                    f"${results['percentile_25']:,.2f}"
                )
            
            with col3:
                st.metric(
                    "75th Percentile",
                    f"${results['percentile_75']:,.2f}"
                )
            
            with col4:
                gain_95 = results['percentile_95'] - results['current_value']
                st.metric(
                    "95th Percentile (Best 5%)",
                    f"${results['percentile_95']:,.2f}",
                    f"+${gain_95:,.2f}"
                )
            
            # Value at Risk
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📉 Value at Risk (95%)",
                    f"${results['var_95']:,.2f}",
                    help="Maximum expected loss with 95% confidence"
                )
            
            with col2:
                st.metric(
                    "📊 Standard Deviation",
                    f"${results['std_final_value']:,.2f}"
                )
            
            with col3:
                range_val = results['max_final_value'] - results['min_final_value']
                st.metric(
                    "📏 Range (Min-Max)",
                    f"${range_val:,.2f}"
                )
            
            # Simulation paths chart
            st.subheader("📈 Simulation Paths")
            
            fig = go.Figure()
            
            # Plot a sample of simulation paths
            sample_size = min(100, results['simulations'])
            sample_indices = np.random.choice(results['simulations'], sample_size, replace=False)
            
            for i in sample_indices:
                fig.add_trace(go.Scatter(
                    y=results['all_simulations'][i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(100, 100, 100, 0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add median path
            median_path = np.median(results['all_simulations'], axis=0)
            fig.add_trace(go.Scatter(
                y=median_path,
                mode='lines',
                name='Median Path',
                line=dict(width=3, color='#1f77b4')
            ))
            
            # Add percentile bands
            percentile_5 = np.percentile(results['all_simulations'], 5, axis=0)
            percentile_95 = np.percentile(results['all_simulations'], 95, axis=0)
            
            fig.add_trace(go.Scatter(
                y=percentile_95,
                mode='lines',
                name='95th Percentile',
                line=dict(width=2, color='#2ca02c', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                y=percentile_5,
                mode='lines',
                name='5th Percentile',
                line=dict(width=2, color='#d62728', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation: {sample_size} Sample Paths (showing {sample_size}/{results['simulations']} simulations)",
                xaxis_title="Trading Days",
                yaxis_title="Portfolio Value ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution of final values
            st.subheader("📊 Distribution of Final Values")
            
            final_values = results['all_simulations'][:, -1]
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                name='Final Values',
                marker_color='#1f77b4'
            ))
            
            # Add vertical lines for percentiles
            fig2.add_vline(x=results['current_value'], line_dash="dash", line_color="black", 
                          annotation_text="Current Value")
            fig2.add_vline(x=results['median_final_value'], line_dash="dash", line_color="blue", 
                          annotation_text="Median")
            fig2.add_vline(x=results['percentile_5'], line_dash="dash", line_color="red", 
                          annotation_text="5th %ile")
            fig2.add_vline(x=results['percentile_95'], line_dash="dash", line_color="green", 
                          annotation_text="95th %ile")
            
            fig2.update_layout(
                title="Distribution of Portfolio Values",
                xaxis_title="Portfolio Value ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Interpretation
            with st.expander("📖 How to Interpret These Results"):
                st.markdown(f"""
                **What This Means:**
                
                Based on {results['simulations']} simulations over {results['days']} trading days:
                
                - **Probability of Profit:** {results['probability_profit']*100:.1f}% chance your portfolio will be worth more than ${results['current_value']:,.2f}
                
                - **Expected Range:** Your portfolio will likely be worth between ${results['percentile_5']:,.2f} and ${results['percentile_95']:,.2f} (with 90% confidence)
                
                - **Value at Risk:** In the worst 5% of scenarios, you could lose ${results['var_95']:,.2f} or more
                
                - **Best Case:** In the best 5% of scenarios, your portfolio could reach ${results['percentile_95']:,.2f}
                
                **Important Notes:**
                - These simulations are based on historical volatility and may not predict actual future performance
                - Markets can experience unprecedented events not captured in historical data
                - Use these results as one tool among many for risk management
                - Consider rerunning simulations periodically as market conditions change
                """)
        
        else:
            st.info("👆 Configure settings and click 'Run Simulation' to see results")
    
    else:
        st.info("📝 Add positions to run Monte Carlo simulation")

# Sidebar - Portfolio Settings
with st.sidebar:
    st.header("Portfolio Settings")
    
    # Risk profile display
    st.metric("Risk Profile", st.session_state.portfolio['risk_profile'].title())
    st.metric("Max Position Size", f"{st.session_state.portfolio['max_position_size']}%")
    
    st.markdown("---")
    
    # Save/Load portfolio
    st.subheader("💾 Save/Load")
    
    if st.button("Save Portfolio", use_container_width=True):
        # Save to file
        portfolio_data = {
            'portfolio': st.session_state.portfolio,
            'performance_snapshots': st.session_state.performance_snapshots,
            'saved_at': datetime.now().isoformat()
        }
        
        save_dir = "portfolios"
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        st.success(f"✅ Saved to {filename}")
    
    # Load portfolio
    if os.path.exists("portfolios"):
        portfolio_files = [f for f in os.listdir("portfolios") if f.endswith('.json')]
        
        if portfolio_files:
            selected_file = st.selectbox("Load Portfolio", [""] + portfolio_files)
            
            if selected_file and st.button("Load", use_container_width=True):
                filepath = os.path.join("portfolios", selected_file)
                with open(filepath, 'r') as f:
                    portfolio_data = json.load(f)
                
                st.session_state.portfolio = portfolio_data['portfolio']
                
                # Load performance snapshots if available
                if 'performance_snapshots' in portfolio_data:
                    st.session_state.performance_snapshots = portfolio_data['performance_snapshots']
                
                st.success(f"✅ Loaded {selected_file}")
                st.rerun()
    
    st.markdown("---")
    
    # Clear portfolio
    if st.button("🗑️ Clear Portfolio", use_container_width=True):
        if st.session_state.portfolio['positions']:
            st.session_state.portfolio['positions'] = []
            st.success("✅ Portfolio cleared")
            st.rerun()

# Footer
st.markdown("---")
st.caption("💼 Portfolio Management | Built with AI-powered insights")
