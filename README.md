# 📈 ML Stock Agent - AI Trading Assistant# 📈 Trading Mentor Agent



> **Professional stock market analysis platform med 14 tekniske indikatorer, ML forecasting, og real-time market insights**En interaktiv Python-baseret trading mentor, der kombinerer teknisk analyse med machine learning for at hjælpe dig med at lære og forstå aktiemarkedet.



[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)## 🎯 Formål

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)Denne agent er designet som en mentor, der ikke kun giver dig analyser, men også stiller refleksionsspørgsmål for at hjælpe dig med at lære og udvikle din trading-forståelse.



---## 🌟 Status: Fase 2 Færdig!



## 🚀 Features- ✅ **Fase 1:** Grundstruktur med teknisk analyse & ML

- ✅ **Fase 2:** Interaktiv mode med farver & mentor-tone

### 🌍 **Market Overview**- 🔄 **Fase 3:** Flere indikatorer & avancerede features (næste)

- Real-time S&P 500 sector heatmap

- Fear & Greed Index (7 komponenter, CNN-aligned)**Nyt i Fase 2:**

- Shiller P/E ratio med dynamic calculation- 🎨 Farvet terminal interface med menu

- Filtrerede finansnyheder (NewsAPI integration)- 🔄 Kontinuerlig session (ingen restart nødvendig)

- Macro indikatorer (VIX, Treasury yields, Dollar index)- 🤔 Refleksionsspørgsmål efter hver analyse

- 📋 Watchlist support

### 📊 **Teknisk Analyse** (14 Indikatorer)- ⚙️ Konfigurationssystem

**Classic:**

- RSI (Relative Strength Index)**→ Se [SUMMARY.md](SUMMARY.md) for komplet oversigt**

- SMA 50/200 (Golden Cross/Death Cross detection)

- EMA (Exponential Moving Average)## 🚀 Quick Start

- MACD (Moving Average Convergence Divergence)

- Bollinger Bands### 1. Kør setup scriptet

- ATR (Average True Range)```powershell

.\setup.ps1

**Advanced:**```

- Stochastic Oscillator

- CCI (Commodity Channel Index)Dette opretter virtual environment og installerer alle dependencies.

- Williams %R

- OBV (On-Balance Volume)### 2A. Interaktiv Mode (ANBEFALET for daglig brug)

- ADX (Average Directional Index)```powershell

- VWAP (Volume Weighted Average Price)python agent_interactive.py

- Fibonacci Retracement (7 levels)```

- Pivot Points (7 support/resistance levels)

Vælg fra menu:

### 🤖 **ML Forecast**- 1️⃣ Teknisk analyse

- LSTM Neural Network (TensorFlow/Keras)- 2️⃣ Læringsmodus  

- Random Forest Regressor- 3️⃣ ML Forecast

- Multi-horizon predictions (1 dag, 5 dage, 22 dage)- 4️⃣ Se watchlist

- Confidence scores og model sammenligning- 5️⃣ Indstillinger

- 6️⃣ Afslut

### 💡 **Agent Recommendations**

- AI-powered trading signaler**Se:** [QUICKSTART_INTERACTIVE.md](QUICKSTART_INTERACTIVE.md) for detaljer

- Macro context integration

- Risk assessment### 2B. CLI Mode (for scripting)

- Position sizing forslag```powershell

# Teknisk analyse

### 📋 **Watchlist**python agent.py --symbol AAPL --mode analyse

- Quick-add stock groups (9 predefined groups)

- Auto-start analysis fra watchlist# Læringsmodus

- Management tools (clear, export, sort)python agent.py --symbol MSFT --mode laer

- Multi-stock tracking

# ML Forecast

---python agent.py --symbol TSLA --mode ml

```

## 📦 Installation

### ⚠️ TensorFlow Problem?

### Lokal UdviklingHvis ML mode ikke virker, se [TENSORFLOW_GUIDE.md](TENSORFLOW_GUIDE.md)



```powershellDen interaktive agent virker stadig uden TensorFlow (4 ud af 5 features)!

# Clone repository

git clone https://github.com/YOUR_USERNAME/ml-stock-agent.git## 📚 Modes

cd ml-stock-agent

- **analyse**: Teknisk analyse med RSI, SMA50, SMA200 + visualisering

# Opret virtual environment- **laer**: Læringsmodus med forklaringer af indikatorer

python -m venv venv311- **ml**: Machine learning forecast med LSTM-model

.\venv311\Scripts\Activate.ps1

## 🛠️ Teknologier

# Install dependencies

pip install -r requirements.txt- **yfinance**: Henter historiske aktidata

- **pandas-ta**: Tekniske indikatorer (RSI, SMA, etc.)

# Opret .streamlit/secrets.toml- **matplotlib**: Visualisering af data

# newsapi_key = "YOUR_API_KEY"- **TensorFlow/Keras**: LSTM neural network til forecasting

- **scikit-learn**: Data preprocessing

# Kør app

streamlit run app.py## 📋 Dokumentation

```

- **[QUICKSTART_INTERACTIVE.md](QUICKSTART_INTERACTIVE.md)** - Guide til interaktiv mode

---- **[PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)** - Fase 2 features

- **[TENSORFLOW_GUIDE.md](TENSORFLOW_GUIDE.md)** - Løsning til TensorFlow problemer

## 🚀 Deployment- **[EXAMPLES.md](EXAMPLES.md)** - Use cases og eksempler

- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Næste trin i udviklingen

Se **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for detaljeret deployment guide.- **[roadmap.md](roadmap.md)** - Fuld udviklingsplan (Fase 1-6)



### Quick Deploy til Streamlit Cloud## ⚠️ Disclaimer



1. Push til GitHubDette værktøj er kun til læring og uddannelse. Det er ikke finansiel rådgivning. Invester aldrig mere end du har råd til at tabe.

2. Gå til [share.streamlit.io](https://share.streamlit.io)
3. Login med GitHub
4. Vælg dit repo
5. Tilføj secrets: `newsapi_key = "YOUR_KEY"`
6. Deploy! 🎉

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.28+
- **Data**: yfinance, pandas
- **ML**: TensorFlow 2.12+, scikit-learn
- **Visualization**: Plotly, matplotlib
- **API**: NewsAPI (real-time financial news)

---

## 📂 Projekt Struktur

```
ML_Stock_agent/
├── app.py                          # Hovedside
├── agent_interactive.py            # Core funktioner
├── pages/
│   ├── 0_🌍_Market_Overview.py    # Market dashboard
│   ├── 1_📊_Teknisk_Analyse.py    # 14 indikatorer
│   ├── 2_🤖_ML_Forecast.py         # ML predictions
│   ├── 3_💡_Agent_Recommendations.py # AI signals
│   └── 3_📋_Watchlist.py          # Stock tracking
├── .streamlit/
│   ├── config.toml                 # Dark theme
│   └── secrets.toml                # API keys (lokal)
├── requirements.txt                # Dependencies
└── config.json                     # App config
```

---

## 🔑 API Keys

Appen kræver en gratis NewsAPI key:
1. Gå til [newsapi.org](https://newsapi.org)
2. Opret gratis konto (100 requests/dag)
3. Tilføj key til `.streamlit/secrets.toml`

---

## 🎯 Roadmap

- [x] Phase 1-5: Core functionality
- [x] Dark theme med optimerede farver
- [x] News filtering (fjern spam/deals)
- [ ] Phase 6: Portfolio tracking
- [ ] Options analysis
- [ ] Advanced ML models

Se **[roadmap.md](roadmap.md)** for fuld roadmap.

---

**Lavet med ❤️ og Python**
