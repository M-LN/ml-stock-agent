# 🚀 STEP-BY-STEP: Deploy til Streamlit Cloud

## ✅ Status: Klar til deployment!
- Git repository initialiseret ✅
- Filer committed ✅
- .gitignore beskytter secrets ✅

---

## 📋 TRIN 1: Opret GitHub Repository

1. **Gå til GitHub**: https://github.com/new
2. **Repository navn**: `ml-stock-agent`
3. **Description**: `AI-powered stock market analysis with 14 technical indicators and ML forecasting`
4. **Vælg**: Public (så Streamlit Cloud kan tilgå det gratis)
5. **IKKE tilføj**: README, .gitignore, eller license (vi har dem allerede)
6. **Klik**: "Create repository"

---

## 📋 TRIN 2: Push til GitHub

**Kopier DISSE kommandoer efter du har oprettet repo'et:**

```powershell
# Gå til projekt mappe
cd "c:\Users\mlund\OneDrive\Skrivebord\Scripts\ML_Stock_agent"

# Link til dit nye GitHub repo
# ERSTAT 'YOUR_USERNAME' med dit faktiske GitHub username!
git remote add origin https://github.com/YOUR_USERNAME/ml-stock-agent.git

# Omdøb branch til main
git branch -M main

# Push til GitHub
git push -u origin main
```

**Du bliver bedt om at logge ind med GitHub credentials.**

---

## 📋 TRIN 3: Deploy til Streamlit Cloud

### 3.1 Opret Streamlit Cloud Account
1. Gå til: **https://share.streamlit.io**
2. Klik **"Continue with GitHub"**
3. Godkend Streamlit adgang til dine repositories

### 3.2 Deploy Ny App
1. Klik **"New app"** (oppe i højre hjørne)
2. Udfyld:
   - **Repository**: `YOUR_USERNAME/ml-stock-agent`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `ml-stock-agent` (eller vælg dit eget)

### 3.3 Tilføj Secrets (VIGTIGT!)
1. Klik **"Advanced settings..."** (nederst)
2. I "Secrets" sektionen, tilføj:
   ```toml
   newsapi_key = "6616306a22ee4d509fd3cb0d485ed0f4"
   ```
3. Klik **"Save"**

### 3.4 Deploy!
1. Klik **"Deploy!"** knappen
2. Vent 3-5 minutter mens appen bygges
3. Se build log i real-time

---

## 🎉 TRIN 4: Færdig!

Din app vil være live på:
```
https://YOUR_USERNAME-ml-stock-agent.streamlit.app
```

eller

```
https://share.streamlit.io/YOUR_USERNAME/ml-stock-agent/main/app.py
```

---

## 🔄 Fremtidige Updates

Når du vil opdatere appen:

```powershell
# Gå til projekt mappe
cd "c:\Users\mlund\OneDrive\Skrivebord\Scripts\ML_Stock_agent"

# Stage ændringer
git add .

# Commit
git commit -m "Beskrivelse af ændringer"

# Push til GitHub
git push

# Streamlit Cloud auto-deployer automatisk! 🎉
```

---

## 🐛 Troubleshooting

### Fejl: "ModuleNotFoundError"
- Tjek at alle dependencies er i `requirements.txt`
- Genstart deployment

### Fejl: "NewsAPI key not found"
- Tjek at secrets er tilføjet korrekt i Streamlit Cloud
- Format: `newsapi_key = "KEY_HER"`

### App er langsom
- Normal ved første load (cold start)
- Data caches efter første brug

### Bygger ikke
- Tjek build logs i Streamlit Cloud UI
- Se efter Python version issues (vi bruger 3.11)

---

## 📊 Streamlit Cloud Limits

**GRATIS tier inkluderer:**
- ✅ 1GB RAM
- ✅ 1 CPU core
- ✅ Unlimited public apps
- ✅ HTTPS gratis
- ✅ Auto-rebuild på git push

**Begrænsninger:**
- ⚠️ Apps sleep efter 7 dage uden brug (vågner automatisk)
- ⚠️ Max 1GB RAM (rigeligt til din app)
- ⚠️ Public URL (alle kan tilgå)

---

## 🎯 Næste Steps Efter Deployment

1. **Test alle sider**:
   - Market Overview
   - Teknisk Analyse (test flere stocks)
   - ML Forecast
   - Agent Recommendations
   - Watchlist

2. **Del URL**:
   - LinkedIn
   - Twitter
   - Portfolio
   - Friends

3. **Monitor**:
   - Streamlit Cloud dashboard
   - App metrics
   - Error logs

---

## 📧 Support

**Streamlit Community:**
- Forum: https://discuss.streamlit.io
- Docs: https://docs.streamlit.io
- GitHub: https://github.com/streamlit/streamlit

**Din App:**
- Logs: Se i Streamlit Cloud UI
- Metrics: Built-in i dashboard
- Updates: Auto-deploy via git push

---

## ✅ Quick Checklist

- [ ] GitHub repo oprettet
- [ ] Git remote tilføjet
- [ ] Pushed til GitHub (`git push`)
- [ ] Streamlit Cloud account oprettet
- [ ] App deployed
- [ ] Secrets tilføjet (newsapi_key)
- [ ] App URL testet
- [ ] Alle sider virker
- [ ] URL delt! 🎊

---

**Held og lykke med deployment! 🚀**

Hvis du støder på problemer, åbn build logs i Streamlit Cloud UI eller spørg mig!
