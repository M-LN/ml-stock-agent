import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="Watchlist", page_icon="📋", layout="wide")

st.title("📋 Min Watchlist")
st.markdown("Overvåg dine favorit aktier og få hurtige indsigter")

# Watchlist fil
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    """Indlæs watchlist fra fil"""
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return []

def save_watchlist(watchlist):
    """Gem watchlist til fil"""
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=2)

def get_stock_info(symbol):
    """Hent aktie info fra yfinance"""
    try:
        data = yf.download(symbol, period='5d', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if not data.empty:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2] if len(data) > 1 else current
            change = current - prev
            change_pct = (change / prev) * 100
            
            return {
                'symbol': symbol,
                'price': current,
                'change': change,
                'change_pct': change_pct,
                'volume': data['Volume'].iloc[-1],
                'status': 'success'
            }
    except:
        pass
    
    return {'symbol': symbol, 'status': 'error'}

# Sidebar - Stock groups
st.sidebar.title("➕ Tilføj Aktier")

# Predefined stock groups
stock_groups = {
    "💼 Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "🏦 Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "⚡ Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "💊 Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK"],
    "🛒 Consumer": ["WMT", "COST", "TGT", "HD", "NKE"],
    "🏭 Industrial": ["BA", "CAT", "GE", "HON", "UPS"],
    "🏠 Real Estate": ["AMT", "PLD", "SPG", "EQIX", "PSA"],
    "💰 Crypto Stocks": ["COIN", "MSTR", "MARA", "RIOT", "HOOD"],
    "🌟 Populære Danske": ["NOVO-B.CO", "DSV.CO", "MAERSK-B.CO", "ORSTED.CO", "COLO-B.CO"],
}

st.sidebar.markdown("### 📋 Quick Add Gruppe")

# Group selector
selected_group = st.sidebar.selectbox(
    "Vælg Gruppe at Tilføje",
    options=["-- Vælg --"] + list(stock_groups.keys()),
    help="Tilføj hele grupper til watchlist med ét klik"
)

if selected_group != "-- Vælg --":
    group_stocks = stock_groups[selected_group]
    watchlist = load_watchlist()
    new_stocks = [s for s in group_stocks if s not in watchlist]
    already_added = [s for s in group_stocks if s in watchlist]
    
    if new_stocks:
        st.sidebar.info(f"**Tilføjer {len(new_stocks)} nye aktier:**\n" + ", ".join(new_stocks))
        if already_added:
            st.sidebar.caption(f"Allerede på listen: {', '.join(already_added)}")
        
        if st.sidebar.button(f"➕ Tilføj {selected_group}", use_container_width=True, type="primary"):
            watchlist.extend(new_stocks)
            save_watchlist(watchlist)
            st.sidebar.success(f"✅ {len(new_stocks)} aktier tilføjet!")
            st.rerun()
    else:
        st.sidebar.success(f"✅ Alle aktier fra {selected_group} er allerede på listen!")

st.sidebar.markdown("---")

# Manual add single stock
st.sidebar.markdown("### ✍️ Tilføj Enkelt Aktie")
new_symbol = st.sidebar.text_input("Aktiesymbol", placeholder="fx MSFT", key="manual_add").upper()

if st.sidebar.button("Tilføj", use_container_width=True):
    if new_symbol:
        watchlist = load_watchlist()
        if new_symbol not in watchlist:
            watchlist.append(new_symbol)
            save_watchlist(watchlist)
            st.sidebar.success(f"✅ {new_symbol} tilføjet!")
            st.rerun()
        else:
            st.sidebar.warning(f"⚠️ {new_symbol} er allerede på listen")
    else:
        st.sidebar.error("❌ Indtast et symbol")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Info")
st.sidebar.info("""
Watchlist opdateres når du genindlæser siden.
Klik på ❌ for at fjerne aktier.
""")

# Main content
watchlist = load_watchlist()

if not watchlist:
    st.info("📭 Din watchlist er tom. Vælg en gruppe i sidebaren eller tilføj enkelte aktier!")
    
    st.markdown("### 💡 Quick Start - Tilføj Hele Grupper:")
    
    # Display groups as cards
    for group_name, group_stocks in stock_groups.items():
        with st.expander(f"{group_name} ({len(group_stocks)} aktier)", expanded=False):
            st.markdown(f"**Aktier:** {', '.join(group_stocks)}")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("➕ Tilføj Gruppe", key=f"quickadd_{group_name}", use_container_width=True):
                    watchlist = load_watchlist()
                    new_stocks = [s for s in group_stocks if s not in watchlist]
                    watchlist.extend(new_stocks)
                    save_watchlist(watchlist)
                    st.success(f"✅ {len(new_stocks)} aktier tilføjet!")
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### ✍️ Eller tilføj enkelte aktier:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    popular_singles = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    for i, sym in enumerate(popular_singles):
        with [col1, col2, col3, col4][i % 4]:
            if st.button(f"➕ {sym}", key=f"add_{sym}", use_container_width=True):
                watchlist = load_watchlist()
                if sym not in watchlist:
                    watchlist.append(sym)
                    save_watchlist(watchlist)
                    st.rerun()

else:
    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Opdater", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Fetch all stock data
    with st.spinner('Henter data...'):
        stock_data = []
        for symbol in watchlist:
            info = get_stock_info(symbol)
            if info['status'] == 'success':
                stock_data.append(info)
    
    if not stock_data:
        st.warning("⚠️ Kunne ikke hente data for aktier på listen")
    else:
        # Display stocks in cards
        cols_per_row = 3
        
        for i in range(0, len(stock_data), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(stock_data):
                    stock = stock_data[idx]
                    
                    with col:
                        # Card styling
                        color = "#10b981" if stock['change'] >= 0 else "#ef4444"
                        arrow = "📈" if stock['change'] >= 0 else "📉"
                        
                        st.markdown(f"""
                        <div style="
                            padding: 20px;
                            border-radius: 10px;
                            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                            border-left: 4px solid {color};
                            margin-bottom: 10px;
                        ">
                            <h3 style="margin: 0; color: white;">{arrow} {stock['symbol']}</h3>
                            <h2 style="margin: 10px 0; color: white;">${stock['price']:.2f}</h2>
                            <p style="margin: 0; color: {color}; font-size: 18px; font-weight: bold;">
                                {stock['change']:+.2f} ({stock['change_pct']:+.2f}%)
                            </p>
                            <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 14px;">
                                Vol: {stock['volume']/1e6:.1f}M
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons - Row 1
                        btn_col1, btn_col2 = st.columns(2)
                        
                        with btn_col1:
                            if st.button("📊 Teknisk Analyse", key=f"analyze_{stock['symbol']}", use_container_width=True):
                                # Set the symbol in session state so Technical Analysis picks it up
                                st.session_state['selected_symbol'] = stock['symbol']
                                st.session_state['auto_analyze'] = True
                                st.switch_page("pages/1_📊_Teknisk_Analyse.py")
                        
                        with btn_col2:
                            if st.button("🤖 ML Forecast", key=f"ml_{stock['symbol']}", use_container_width=True):
                                # Set symbol for ML Forecast
                                st.session_state['ml_selected_symbol'] = stock['symbol']
                                st.session_state['auto_ml'] = True
                                st.switch_page("pages/2_🤖_ML_Forecast.py")
                        
                        # Action buttons - Row 2
                        btn_col3, btn_col4 = st.columns(2)
                        
                        with btn_col3:
                            if st.button("💡 Agent Rec.", key=f"agent_{stock['symbol']}", use_container_width=True):
                                # Set symbol for Agent Recommendations
                                st.session_state['agent_selected_symbol'] = stock['symbol']
                                st.session_state['auto_agent'] = True
                                st.switch_page("pages/3_💡_Agent_Recommendations.py")
                        
                        with btn_col4:
                            if st.button("❌ Fjern", key=f"remove_{stock['symbol']}", use_container_width=True):
                                watchlist = load_watchlist()
                                watchlist.remove(stock['symbol'])
                                save_watchlist(watchlist)
                                st.rerun()
        
        st.markdown("---")
        
        # Summary table
        st.markdown("## 📊 Oversigt")
        
        df = pd.DataFrame(stock_data)
        df = df[['symbol', 'price', 'change', 'change_pct', 'volume']]
        df.columns = ['Symbol', 'Pris', 'Ændring', 'Ændring %', 'Volume']
        
        # Format columns
        df['Pris'] = df['Pris'].apply(lambda x: f"${x:.2f}")
        df['Ændring'] = df['Ændring'].apply(lambda x: f"${x:+.2f}")
        df['Ændring %'] = df['Ændring %'].apply(lambda x: f"{x:+.2f}%")
        df['Volume'] = df['Volume'].apply(lambda x: f"{x/1e6:.1f}M")
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Statistics
        st.markdown("### 📈 Statistik")
        
        col1, col2, col3, col4 = st.columns(4)
        
        gainers = sum(1 for s in stock_data if s['change'] > 0)
        losers = sum(1 for s in stock_data if s['change'] < 0)
        avg_change = sum(s['change_pct'] for s in stock_data) / len(stock_data)
        
        with col1:
            st.metric("Total Aktier", len(stock_data))
        
        with col2:
            st.metric("📈 Gainers", gainers)
        
        with col3:
            st.metric("📉 Losers", losers)
        
        with col4:
            st.metric("Gns. Ændring", f"{avg_change:+.2f}%")
        
        # Management buttons
        st.markdown("---")
        st.markdown("### 🔧 Watchlist Administration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Ryd Watchlist", use_container_width=True, type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    save_watchlist([])
                    st.session_state['confirm_clear'] = False
                    st.success("✅ Watchlist ryddet!")
                    st.rerun()
                else:
                    st.session_state['confirm_clear'] = True
                    st.warning("⚠️ Klik igen for at bekræfte")
        
        with col2:
            # Export to text
            export_text = ", ".join(watchlist)
            st.download_button(
                label="📥 Eksporter Symbols",
                data=export_text,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # Sort alphabetically
            if st.button("🔤 Sorter A-Z", use_container_width=True):
                watchlist_sorted = sorted(watchlist)
                save_watchlist(watchlist_sorted)
                st.success("✅ Sorteret!")
                st.rerun()
