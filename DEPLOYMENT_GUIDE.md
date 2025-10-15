# 🚀 ML Stock Agent - Deployment Guide

## 📦 Forudsætninger
- Git installeret
- GitHub konto
- NewsAPI key: `6616306a22ee4d509fd3cb0d485ed0f4`

---

## 🌟 OPTION 1: Streamlit Community Cloud (ANBEFALET)

### ✅ **Fordele:**
- 100% GRATIS
- Ingen kreditkort påkrævet
- Auto-deploy fra GitHub
- Built-in secrets management
- HTTPS inkluderet

### 📋 **Step-by-Step:**

#### 1️⃣ Opret GitHub Repository
```powershell
# I din projekt mappe
cd "c:\Users\mlund\OneDrive\Skrivebord\Scripts\ML_Stock_agent"

# Initialize git (hvis ikke allerede gjort)
git init

# Tilføj .gitignore
echo "venv311/
__pycache__/
*.pyc
.env
*.log
saved_models/*.h5
.streamlit/secrets.toml" > .gitignore

# Commit alle filer
git add .
git commit -m "Initial commit - ML Stock Agent ready for deployment"

# Opret nyt repo på GitHub (gå til github.com/new)
# Derefter link dit lokale repo:
git remote add origin https://github.com/DIT_BRUGERNAVN/ml-stock-agent.git
git branch -M main
git push -u origin main
```

#### 2️⃣ Deploy til Streamlit Cloud
1. Gå til **https://share.streamlit.io**
2. Klik **"Sign in with GitHub"**
3. Klik **"New app"**
4. Vælg:
   - **Repository**: `DIT_BRUGERNAVN/ml-stock-agent`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **"Advanced settings"**
6. Under **"Secrets"**, tilføj:
   ```toml
   newsapi_key = "6616306a22ee4d509fd3cb0d485ed0f4"
   ```
7. Klik **"Deploy!"** 🎉

#### 3️⃣ Færdig! 🎊
Din app er nu live på:
```
https://DIT_BRUGERNAVN-ml-stock-agent-app-xxxxx.streamlit.app
```

**Build tid**: ~3-5 minutter

---

## 💼 OPTION 2: Heroku (Professional Hosting)

### ✅ **Fordele:**
- Professional-grade hosting
- Custom domains
- Bedre performance ($7/måned Basic plan)
- Skalerbar

### 📋 **Step-by-Step:**

#### 1️⃣ Install Heroku CLI
```powershell
# Download fra: https://devcenter.heroku.com/articles/heroku-cli
# Eller via Chocolatey:
choco install heroku-cli
```

#### 2️⃣ Opret Heroku App
```powershell
# Login
heroku login

# Opret app
heroku create ml-stock-agent-app

# Tilføj buildpack
heroku buildpacks:add --index 1 heroku/python

# Set environment variables
heroku config:set NEWSAPI_KEY="6616306a22ee4d509fd3cb0d485ed0f4"
```

#### 3️⃣ Deploy
```powershell
# Commit hvis ikke allerede gjort
git add .
git commit -m "Ready for Heroku deployment"

# Push til Heroku
git push heroku main
```

#### 4️⃣ Åbn App
```powershell
heroku open
```

**URL**: `https://ml-stock-agent-app.herokuapp.com`

---

## 🐳 OPTION 3: Docker + Cloud Provider

### ✅ **Fordele:**
- Fuld kontrol
- Portabel (kør overalt)
- Professional setup

### 📋 **Dockerfile** (allerede inkluderet i projektet)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Create .streamlit directory
RUN mkdir -p ~/.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Deploy til Cloud Run (Google Cloud)
```powershell
# Build image
docker build -t ml-stock-agent .

# Tag for GCR
docker tag ml-stock-agent gcr.io/YOUR_PROJECT_ID/ml-stock-agent

# Push
docker push gcr.io/YOUR_PROJECT_ID/ml-stock-agent

# Deploy
gcloud run deploy ml-stock-agent \
  --image gcr.io/YOUR_PROJECT_ID/ml-stock-agent \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars NEWSAPI_KEY="6616306a22ee4d509fd3cb0d485ed0f4"
```

---

## 🔐 Secrets Management

### For Streamlit Cloud:
- UI-based i Advanced Settings
- Automatisk synkroniseret

### For Heroku:
```powershell
heroku config:set NEWSAPI_KEY="din_api_key"
```

### For Docker:
```powershell
docker run -e NEWSAPI_KEY="din_api_key" -p 8501:8501 ml-stock-agent
```

---

## 📊 Performance Tips

### 1. Cache Data
Din app bruger allerede `@st.cache_data` - perfekt! ✅

### 2. Optimize Imports
```python
# app.py har allerede optimeret imports
import streamlit as st
import yfinance as yf
# etc...
```

### 3. Resource Limits
- **Streamlit Cloud**: 1GB RAM (rigeligt)
- **Heroku Basic**: 512MB RAM (ok)
- **Heroku Standard**: 2.5GB RAM (perfekt for ML)

---

## 🧪 Test Før Deployment

```powershell
# Lokalt
cd "c:\Users\mlund\OneDrive\Skrivebord\Scripts\ML_Stock_agent"
.\venv311\Scripts\Activate.ps1
streamlit run app.py

# Test alle sider:
# ✅ Market Overview
# ✅ Teknisk Analyse
# ✅ ML Forecast
# ✅ Agent Recommendations
# ✅ Watchlist
```

---

## 🎯 MIN ANBEFALING

### **Start med Streamlit Community Cloud:**
1. ✅ 100% gratis
2. ✅ Super nem setup (5-10 min)
3. ✅ Perfekt til personlig brug
4. ✅ Automatisk updates ved git push

### **Upgrade til Heroku hvis:**
- Du vil have custom domain
- Du har >100 daglige brugere
- Du vil have bedre performance
- Du skal bruge mere RAM til ML modeller

---

## 🚨 Vigtige Filer for Deployment

Alle er allerede inkluderet i dit projekt! ✅

1. **`requirements.txt`** - Python dependencies
2. **`Procfile`** - Heroku process definition
3. **`setup.sh`** - Streamlit config setup
4. **`runtime.txt`** - Python version
5. **`.streamlit/config.toml`** - Streamlit theme
6. **`.gitignore`** - Git ignore patterns

---

## 📞 Support

Hvis du får problemer:
1. Streamlit: https://discuss.streamlit.io
2. Heroku: https://help.heroku.com
3. Check logs:
   ```powershell
   # Streamlit Cloud: Se logs i UI
   # Heroku: heroku logs --tail
   ```

---

## ✅ Quick Checklist

- [ ] Git repo oprettet
- [ ] Pushed til GitHub
- [ ] Streamlit Cloud account
- [ ] App deployed
- [ ] Secrets tilføjet
- [ ] Alle sider testet
- [ ] URL delt! 🎉

**Held og lykke! 🚀**
