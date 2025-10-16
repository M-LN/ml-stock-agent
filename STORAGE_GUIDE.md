# 💾 Persistent Storage & Caching Guide

## Problem
Streamlit Cloud bruger **ephemeral storage** - alle gemte filer forsvinder ved hver redeploy. Dette betyder at dine trænede ML modeller går tabt.

## Løsning

Vi har implementeret 2 løsninger:

### 1. 🔄 Intelligent Caching
Reducerer API calls og forbedrer performance.

**Implementeret caching:**
- ✅ **Stock data** (yfinance): Cache i 1 time
- ✅ **Stock info**: Cache i 30 minutter
- ✅ **News articles**: Cache i 1 time

**Fordele:**
- 🚀 Hurtigere page loads
- 💰 Færre API calls (sparer NewsAPI quota)
- 📉 Mindre Yahoo Finance rate limiting

**Brug:**
```python
from storage_manager import get_stock_data_cached, get_stock_info_cached

# Instead of: yf.Ticker(symbol).history(period="1y")
data = get_stock_data_cached(symbol, period="1y")

# Instead of: yf.Ticker(symbol).info
info = get_stock_info_cached(symbol)
```

### 2. 💾 Persistent Model Storage

Vi supporterer 2 storage backends:

#### A) Local Storage (Default)
- 📁 Gemmer modeller på Streamlit Cloud's filesystem
- ⚠️ **Ephemeral** - modeller forsvinder ved redeploy
- ✅ Gratis, ingen setup krævet
- 🎯 Godt til development og testing

#### B) GitHub Gist Storage (Recommended for Production)
- ☁️ Gemmer modeller som private GitHub Gists
- ✅ **Persistent** - modeller overlever redeployments
- 🔒 Private og secure
- 📊 100MB size limit per model
- 🆓 Gratis med GitHub account

---

## 🔧 Setup: GitHub Persistent Storage

### Step 1: Create GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Give it a name: `streamlit-ml-storage`
4. Select scopes:
   - ✅ `gist` (Create and access gists)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

### Step 2: Add to Streamlit Secrets

1. Go to your app on Streamlit Cloud
2. Click **"⚙️ Settings"** → **"Secrets"**
3. Add this line:
```toml
GITHUB_TOKEN = "ghp_your_token_here"
```
4. Click **"Save"**

### Step 3: Enable GitHub Storage

1. Go to **"Model Management"** page
2. Open **"⚙️ Storage & Cache Settings"**
3. Select **"github"** from Storage Backend dropdown
4. You should see ✅ **"GitHub token configured"**

**Done!** Your models will now persist across redeployments! 🎉

---

## 📊 Usage in Model Management

### Storage Settings Panel

In the Model Management page, you'll find:

```
⚙️ Storage & Cache Settings
├── 💾 Model Storage
│   ├── Storage Backend: [local/github]
│   └── GitHub status
├── 🚀 Cache Status
│   ├── Clear All Caches button
│   └── Cache TTL info
└── 📊 Storage Info
    ├── Saved Models count
    └── Deployed Models count
```

### How It Works

**When you train a model:**
1. Model is trained locally
2. Model is saved using StorageManager
3. If GitHub storage enabled:
   - Model is serialized to bytes
   - Uploaded as private Gist
   - Gist ID is stored locally for reference

**When you load a model:**
1. Check if model exists locally (cache)
2. If not found and GitHub enabled:
   - Download from Gist
   - Deserialize and cache locally
3. Use model for predictions

---

## 🎯 Best Practices

### For Development:
- Use **local storage** (fast, no setup)
- Clear caches frequently to test with fresh data

### For Production:
- Use **GitHub storage** (persistent)
- Keep important models backed up
- Monitor Gist storage usage (100MB per model limit)

### Caching Tips:
- **Clear cache** when:
  - Stock data seems stale
  - News not updating
  - After major market events
- Cache is **automatic** - no code changes needed
- Cache works **across all pages** that import storage_manager

---

## 🔒 Security

✅ **GitHub tokens** are:
- Stored encrypted in Streamlit secrets
- Never exposed to users
- Only used server-side

✅ **Private Gists** are:
- Not searchable on GitHub
- Only accessible with token
- Can be deleted manually if needed

✅ **API keys** (NewsAPI):
- Also in Streamlit secrets
- Never cached or stored in models
- Separate from model storage

---

## 📈 Storage Limits

### Local Storage (Streamlit Cloud):
- ⚠️ No persistence
- 🚀 Fast access
- 💾 Limited to available RAM/disk

### GitHub Gist Storage:
- ✅ Persistent forever (until you delete)
- 📊 100MB per Gist (per model)
- 🔢 Unlimited number of Gists
- 🌐 Global CDN (fast downloads)

### Recommendations:
- Small models (< 10MB): GitHub Gist ✅
- Large models (> 50MB): Consider external storage (S3, etc.)
- Very large models (> 100MB): Must use external storage

---

## 🆘 Troubleshooting

**Problem:** "GITHUB_TOKEN not found in secrets"
- **Solution:** Add token to Streamlit secrets (see Step 2 above)

**Problem:** Models still disappear after redeploy
- **Solution:** Make sure "github" is selected as storage backend

**Problem:** "Failed to save to GitHub: 401"
- **Solution:** Token is invalid or expired - create new token

**Problem:** Cache not clearing
- **Solution:** Click "🗑️ Clear All Caches" button in Storage Settings

**Problem:** Model too large for Gist
- **Solution:** Use local storage or implement S3/cloud storage

---

## 🚀 Future Enhancements

Planned features:
- [ ] AWS S3 storage backend
- [ ] Azure Blob Storage backend
- [ ] Model compression before upload
- [ ] Automatic cleanup of old models
- [ ] Model versioning and rollback
- [ ] Shared model repository (public models)

---

## 💡 Code Examples

### Using Cached Data Fetching

**Before (no caching):**
```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y")  # API call every time
info = ticker.info  # API call every time
```

**After (with caching):**
```python
from storage_manager import get_stock_data_cached, get_stock_info_cached

data = get_stock_data_cached("AAPL", period="1y")  # Cached for 1 hour
info = get_stock_info_cached("AAPL")  # Cached for 30 min
```

### Using Storage Manager Directly

```python
from storage_manager import StorageManager

# Initialize (use GitHub if token available)
storage = StorageManager(storage_type="github")

# Save model
storage.save_model(
    model=trained_model,
    model_id="AAPL_RF_20231016",
    metadata={
        "symbol": "AAPL",
        "model_type": "RandomForest",
        "accuracy": 0.85,
        "trained_at": "2023-10-16"
    }
)

# Load model
model, metadata = storage.load_model("AAPL_RF_20231016")

# List all models
all_models = storage.list_models()

# Delete model
storage.delete_model("AAPL_RF_20231016")
```

---

**Questions?** Check the Model Management page for real-time storage status and settings!
