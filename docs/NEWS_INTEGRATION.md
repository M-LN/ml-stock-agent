# 📰 News Integration Setup Guide

## Overview
The Market Overview page now supports real-time financial news through NewsAPI integration.

## Quick Setup (5 minutes)

### Step 1: Get NewsAPI Key (FREE)
1. Go to [NewsAPI.org](https://newsapi.org/)
2. Click "Get API Key" (top right)
3. Sign up with your email
4. Copy your API key

### Step 2: Configure Streamlit Secrets

#### Option A: Local Development
1. Create folder `.streamlit` in your project root (if it doesn't exist)
2. Create file `.streamlit/secrets.toml`
3. Add your API key:
```toml
newsapi_key = "your_actual_api_key_here"
```

#### Option B: Streamlit Cloud Deployment
1. Go to your app settings on Streamlit Cloud
2. Navigate to "Secrets" section
3. Add:
```toml
newsapi_key = "your_actual_api_key_here"
```

### Step 3: Restart App
```bash
# Stop current streamlit session (Ctrl+C)
# Restart
streamlit run app.py
```

## Features Enabled

### ✅ With NewsAPI Key:
- **Real-time financial news** from 80,000+ sources
- **Market-moving headlines** updated every 10 minutes
- **Source filtering**: WSJ, Bloomberg, CNBC, Reuters, etc.
- **Image thumbnails** for major stories
- **Direct links** to full articles
- **Timestamp** for each article

### 📋 Without API Key (Fallback):
- Curated market headlines (manual updates)
- Setup instructions for enabling real-time news
- Economic calendar preview

## API Limits (Free Tier)

| Limit | Value |
|-------|-------|
| Requests/day | 100 |
| Articles per request | Up to 100 |
| Data retention | Last 30 days |
| Commercial use | ❌ No |

**Note**: With 10-minute caching, typical usage is ~50 requests/day

## News Query

Current search query covers:
- "stock market"
- "S&P 500"
- "Federal Reserve"
- "earnings"

### Customize News Topics

Edit `pages/0_🌍_Market_Overview.py`, line ~904:

```python
params = {
    'q': 'YOUR KEYWORDS HERE',  # Change this
    'language': 'en',
    'sortBy': 'publishedAt',
    'pageSize': 10,
    'apiKey': api_key
}
```

**Example queries:**
- `'stock market OR cryptocurrency OR forex'`
- `'AAPL OR MSFT OR GOOGL OR TSLA'`
- `'Federal Reserve OR ECB OR inflation'`

## Advanced Configuration

### Filter by Source
```python
params = {
    'sources': 'bloomberg,cnbc,financial-times',
    'language': 'en',
    'sortBy': 'publishedAt',
    'apiKey': api_key
}
```

### Filter by Date Range
```python
from datetime import datetime, timedelta

yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

params = {
    'q': 'stock market',
    'from': yesterday,
    'language': 'en',
    'sortBy': 'publishedAt',
    'apiKey': api_key
}
```

## Troubleshooting

### "News articles not loading"
✅ Check API key is correct in `secrets.toml`
✅ Verify file is named exactly `secrets.toml` (not `.txt`)
✅ Restart Streamlit after adding secrets
✅ Check you haven't exceeded 100 requests/day

### "401 Unauthorized"
❌ Invalid API key - verify copy/paste was correct

### "429 Too Many Requests"
❌ Daily limit reached - wait until tomorrow or upgrade plan

### Cache Issues
Clear cache in Streamlit:
- Press `C` key while app is running
- Or add `?clear_cache=true` to URL

## Upgrade Options

Need more than 100 requests/day?

| Plan | Requests | Price |
|------|----------|-------|
| Free | 100/day | $0 |
| Developer | 250/day | $9/mo |
| Business | 100,000/day | $449/mo |

## Alternative News Sources

If NewsAPI doesn't fit your needs:

1. **Alpha Vantage** (Free)
   - News sentiment API
   - 25 requests/day free tier

2. **Finnhub** (Free)
   - Market news API
   - 60 calls/minute free tier

3. **Yahoo Finance RSS** (Free)
   - No API key needed
   - Limited to RSS feeds

## Security Notes

⚠️ **Important:**
- Never commit `secrets.toml` to git
- Add to `.gitignore`: `.streamlit/secrets.toml`
- Don't share API keys publicly
- Rotate keys if exposed

## Support

- NewsAPI Docs: https://newsapi.org/docs
- Streamlit Secrets: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management

---

**Last Updated**: October 2025
**Version**: 1.0
