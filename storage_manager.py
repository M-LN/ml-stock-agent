"""
Storage Manager for ML Stock Agent
Handles persistent storage of ML models using various backends
"""

import os
import pickle
import json
from datetime import datetime
from pathlib import Path
import streamlit as st

class StorageManager:
    """Manages persistent storage for ML models and metadata"""
    
    def __init__(self, storage_type="local"):
        """
        Initialize storage manager
        
        Args:
            storage_type: "local", "github", or "s3"
        """
        self.storage_type = storage_type
        self.local_dir = "saved_models"
        
        # Create local directory if it doesn't exist
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
    
    def save_model(self, model, model_id, metadata=None):
        """
        Save a model with metadata
        
        Args:
            model: The trained model object
            model_id: Unique identifier for the model
            metadata: Dict with model info (params, metrics, etc.)
        
        Returns:
            bool: Success status
        """
        try:
            if self.storage_type == "local":
                return self._save_local(model, model_id, metadata)
            elif self.storage_type == "github":
                return self._save_github(model, model_id, metadata)
            else:
                st.warning(f"Storage type '{self.storage_type}' not implemented yet")
                return self._save_local(model, model_id, metadata)
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_id):
        """
        Load a model by ID
        
        Args:
            model_id: Model identifier
        
        Returns:
            tuple: (model, metadata) or (None, None) if not found
        """
        try:
            if self.storage_type == "local":
                return self._load_local(model_id)
            elif self.storage_type == "github":
                return self._load_github(model_id)
            else:
                return self._load_local(model_id)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    def list_models(self):
        """
        List all saved models
        
        Returns:
            list: List of dicts with model info
        """
        try:
            if self.storage_type == "local":
                return self._list_local()
            elif self.storage_type == "github":
                return self._list_github()
            else:
                return self._list_local()
        except Exception as e:
            st.error(f"Error listing models: {str(e)}")
            return []
    
    def delete_model(self, model_id):
        """Delete a model by ID"""
        try:
            if self.storage_type == "local":
                return self._delete_local(model_id)
            else:
                return self._delete_local(model_id)
        except Exception as e:
            st.error(f"Error deleting model: {str(e)}")
            return False
    
    # ==================== LOCAL STORAGE ====================
    
    def _save_local(self, model, model_id, metadata):
        """Save model to local filesystem"""
        model_path = os.path.join(self.local_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(self.local_dir, f"{model_id}_metadata.json")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metadata:
            metadata['saved_at'] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return True
    
    def _load_local(self, model_id):
        """Load model from local filesystem"""
        model_path = os.path.join(self.local_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(self.local_dir, f"{model_id}_metadata.json")
        
        if not os.path.exists(model_path):
            return None, None
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def _list_local(self):
        """List all local models"""
        models = []
        
        for file in os.listdir(self.local_dir):
            if file.endswith("_metadata.json"):
                metadata_path = os.path.join(self.local_dir, file)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)
        
        return models
    
    def _delete_local(self, model_id):
        """Delete local model"""
        model_path = os.path.join(self.local_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(self.local_dir, f"{model_id}_metadata.json")
        
        deleted = False
        if os.path.exists(model_path):
            os.remove(model_path)
            deleted = True
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            deleted = True
        
        return deleted
    
    # ==================== GITHUB STORAGE (via Gist) ====================
    
    def _save_github(self, model, model_id, metadata):
        """
        Save model to GitHub Gist (for small models)
        Note: Gist has 100MB file size limit
        """
        import requests
        import base64
        
        # Get GitHub token from secrets
        github_token = st.secrets.get("GITHUB_TOKEN")
        if not github_token:
            st.warning("GITHUB_TOKEN not found in secrets. Falling back to local storage.")
            return self._save_local(model, model_id, metadata)
        
        # Serialize model to bytes
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Create Gist
        gist_data = {
            "description": f"ML Model: {model_id}",
            "public": False,
            "files": {
                f"{model_id}.pkl.b64": {
                    "content": model_b64
                },
                f"{model_id}_metadata.json": {
                    "content": json.dumps(metadata, indent=2)
                }
            }
        }
        
        response = requests.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            },
            json=gist_data
        )
        
        if response.status_code == 201:
            gist_id = response.json()['id']
            # Save gist ID mapping locally
            self._save_gist_mapping(model_id, gist_id)
            return True
        else:
            st.error(f"Failed to save to GitHub: {response.status_code}")
            return False
    
    def _load_github(self, model_id):
        """Load model from GitHub Gist"""
        import requests
        import base64
        
        # Get gist ID
        gist_id = self._get_gist_id(model_id)
        if not gist_id:
            st.warning("Model not found on GitHub. Trying local storage.")
            return self._load_local(model_id)
        
        github_token = st.secrets.get("GITHUB_TOKEN")
        headers = {}
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        
        response = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            gist = response.json()
            
            # Load model
            model_content = gist['files'][f"{model_id}.pkl.b64"]['content']
            model_bytes = base64.b64decode(model_content)
            model = pickle.loads(model_bytes)
            
            # Load metadata
            metadata_content = gist['files'][f"{model_id}_metadata.json"]['content']
            metadata = json.loads(metadata_content)
            
            return model, metadata
        else:
            st.error(f"Failed to load from GitHub: {response.status_code}")
            return None, None
    
    def _list_github(self):
        """List all GitHub Gists with models"""
        import requests
        
        github_token = st.secrets.get("GITHUB_TOKEN")
        if not github_token:
            return self._list_local()
        
        response = requests.get(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        
        if response.status_code == 200:
            gists = response.json()
            models = []
            
            for gist in gists:
                if gist['description'].startswith("ML Model:"):
                    # Extract metadata from gist files
                    for filename, file_info in gist['files'].items():
                        if filename.endswith('_metadata.json'):
                            metadata = json.loads(file_info['content'])
                            models.append(metadata)
            
            return models
        else:
            return self._list_local()
    
    def _save_gist_mapping(self, model_id, gist_id):
        """Save model_id to gist_id mapping"""
        mapping_file = os.path.join(self.local_dir, "gist_mapping.json")
        
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
        else:
            mapping = {}
        
        mapping[model_id] = gist_id
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
    
    def _get_gist_id(self, model_id):
        """Get gist_id for a model_id"""
        mapping_file = os.path.join(self.local_dir, "gist_mapping.json")
        
        if not os.path.exists(mapping_file):
            return None
        
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        return mapping.get(model_id)


# ==================== CACHING UTILITIES ====================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data_cached(symbol, period="1y"):
    """
    Cached version of yfinance data fetch
    
    Args:
        symbol: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        pandas.DataFrame: Stock data
    """
    import yfinance as yf
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    return data

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_info_cached(symbol):
    """
    Cached version of stock info fetch
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        dict: Stock info
    """
    import yfinance as yf
    
    ticker = yf.Ticker(symbol)
    return ticker.info

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_news_cached(symbol, api_key):
    """
    Cached version of news fetch
    
    Args:
        symbol: Stock ticker symbol
        api_key: NewsAPI key
    
    Returns:
        list: News articles
    """
    import requests
    
    url = f"https://newsapi.org/v2/everything"
    params = {
        'q': symbol,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 10
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        return []

def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.success("✅ All caches cleared!")
