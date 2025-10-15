'''
ML Stock Agent - Welcome Page
'''
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="ML Stock Agent", page_icon="📊", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 ML Stock Agent</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>AI-Powered Stock Analysis & Forecasting</h3>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('''
    ### 👋 Velkommen til ML Stock Agent
    
    En komplet platform til aktieanalyse med:
    - 🌍 Market Overview - Realtids markedsoversigt
    - 📊 Teknisk Analyse - Avancerede indikatorer
    - 🤖 ML Forecast - Machine Learning forecasting
    - 💡 AI Anbefalinger - Intelligente signaler
    - 📋 Watchlist - Følg dine favoritter
    - 📄 Rapport Generator - Professionelle rapporter
    - 🔧 Model Management - Træn og deploy modeller
    - 📈 Performance Dashboard - Track præstation
    
    **Vælg en funktion fra sidebaren for at komme i gang →**
    ''')

st.sidebar.success("👈 Vælg en side fra menuen")
st.sidebar.markdown("---")
st.sidebar.info('''
### 💡 Tips
- Start med **Market Overview** for markedsindsigt
- Brug **Teknisk Analyse** for aktieanalyse  
- Prøv **ML Forecast** for prisforudsigelser
''')
