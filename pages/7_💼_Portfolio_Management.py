"""
Portfolio Management Page
Handles portfolio allocation, position sizing, rebalancing, and AI mentor feedback
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Overview",
    "⚖️ Asset Allocation",
    "🎯 Position Sizing",
    "🤖 AI Mentor"
])

with tab1:
    st.header("Portfolio Overview")
    
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
