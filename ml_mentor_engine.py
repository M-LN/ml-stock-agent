# ML Mentor Engine - Full Version
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from agent_interactive import load_model, MODEL_DIR

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class MLMentorEngine:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

def calculate_health_score(metadata):
    score = 100.0
    mae = metadata.get("mae", float("inf"))
    if mae > 10: score -= 30
    elif mae > 5: score -= 15
    elif mae > 2: score -= 5
    
    r2 = metadata.get("r2_score", 0)
    if r2 < 0: score -= 30
    elif r2 < 0.5: score -= 20
    elif r2 < 0.7: score -= 10
    
    if metadata.get("deployed"): score += 5
    return max(0, min(100, score))

def generate_rule_based_recommendations(metadata):
    recommendations = []
    mae = metadata.get("mae", 0)
    model_type = metadata.get("model_type", "")
    
    if mae > 5:
        if "RF" in model_type:
            recommendations.append({
                "priority": "HIGH",
                "category": "Hyperparameters",
                "issue": f"High MAE ({mae:.2f})",
                "recommendation": "Increase n_estimators to 200-300",
                "expected_improvement": "10-20%% MAE reduction"
            })
        elif "XGB" in model_type:
            recommendations.append({
                "priority": "HIGH",
                "category": "Learning",
                "issue": f"High MAE ({mae:.2f})",
                "recommendation": "Lower learning_rate to 0.01",
                "expected_improvement": "15-25%% improvement"
            })
    return recommendations

def analyze_saved_model(model_id, api_key=None, llm_model="gpt-4o-mini"):
    model, metadata = load_model(model_id)
    if not metadata:
        return {"success": False, "error": "Model not found"}
    
    health_score = calculate_health_score(metadata)
    recommendations = generate_rule_based_recommendations(metadata)
    
    return {
        "success": True,
        "health_score": health_score,
        "health_status": " Excellent" if health_score >= 85 else " Good",
        "model_id": model_id,
        "model_type": metadata.get("model_type"),
        "symbol": metadata.get("symbol"),
        "metrics": {
            "mae": metadata.get("mae", 0),
            "rmse": metadata.get("rmse", 0),
            "r2": metadata.get("r2_score", 0)
        },
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "high_priority_count": len([r for r in recommendations if r.get("priority") == "HIGH"]),
        "analyzed_at": datetime.now().isoformat(),
        "llm_used": False
    }

def compare_model_versions(model_ids):
    data = []
    for mid in model_ids:
        model, metadata = load_model(mid)
        if metadata:
            data.append({
                "Model ID": mid,
                "Type": metadata.get("model_type"),
                "MAE": metadata.get("mae", 0),
                "Health": calculate_health_score(metadata)
            })
    return pd.DataFrame(data)
