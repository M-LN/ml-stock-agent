# 🔑 Streamlit Cloud Secrets Setup Guide

## For at få NewsAPI til at virke på Streamlit Cloud:

### Step 1: Gå til Streamlit Cloud
👉 https://share.streamlit.io

### Step 2: Vælg din app
- Find `ml-stock-agent` i listen
- Klik på app'en

### Step 3: Åbn Settings
- Klik på **"⚙️ Settings"** eller **"Manage app"** (nederst til højre)
- Klik på **"Secrets"** i venstre menu

### Step 4: Tilføj denne secret (copy-paste):

```toml
NEWS_API_KEY = "6616306a22ee4d509fd3cb0d485ed0f4"
```

### Step 5: Save
- Klik **"Save"**
- App'en genstarter automatisk (~30 sekunder)

---

## ✅ Efter dette virker:
- 📰 Real-time news på Market Overview page
- 🔍 News filtering på alle stock pages
- 📊 Sentiment analysis

---

## 🔒 Sikkerhed:
- ✅ API key er i .gitignore (kommer IKKE på GitHub)
- ✅ Streamlit Cloud secrets er krypterede
- ✅ Kun din app kan tilgå secrets

---

## 🆓 NewsAPI Free Tier:
- 100 requests/dag
- Adgang til 80,000+ kilder
- Historiske artikler (30 dage)

Hvis du løber tør for requests, kan du opgradere på: https://newsapi.org/pricing
