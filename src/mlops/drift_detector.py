import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.core.config import BASELINE_STATS_PATH


def save_baseline_statistics(df: pd.DataFrame, file_path=BASELINE_STATS_PATH):
    """Computes and saves baseline distributions for numerical and categorical features."""
    stats = {
        "numerical": {},
        "categorical": {},
        "row_count": len(df),
    }
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != "Attrition":
            stats["numerical"][col] = {
                "values": df[col].dropna().values.tolist(),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
            }
            
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in ["Attrition", "EmployeeFeedback"]:
            stats["categorical"][col] = (df[col].value_counts(normalize=True).to_dict())
            
    joblib.dump(stats, file_path)
    return stats


def detect_data_drift(new_data: list, baseline_path=BASELINE_STATS_PATH) -> dict:
    """
    Performs statistical drift detection comparing incoming records against training baseline.
    Uses two-sample Kolmogorov-Smirnov test for numerical features and
    Total Variation Distance (TVD) for categorical distributions.
    """
    if not baseline_path.exists():
        return {"drift_detected": False, "drift_score": 0.0, "message": "No baseline stats saved."}
        
    baseline = joblib.load(baseline_path)
    new_df = pd.DataFrame(new_data)
    
    if len(new_df) < 5:
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "message": "Sample size too small for statistical significance (<5 records)."
        }
        
    feature_drift_results = []
    num_drifted = 0
    total_features = 0
    
    # 1. Numerical drift via KS test
    for col, base_info in baseline.get("numerical", {}).items():
        if col in new_df.columns:
            total_features += 1
            new_vals = new_df[col].dropna().values
            if len(new_vals) >= 5:
                stat, p_val = ks_2samp(base_info["values"], new_vals)
                is_drift = bool(p_val < 0.05 and stat > 0.25)
                if is_drift:
                    num_drifted += 1
                feature_drift_results.append({
                    "feature": col,
                    "type": "numerical",
                    "metric": "KS_statistic",
                    "score": round(float(stat), 4),
                    "p_value": round(float(p_val), 4),
                    "drift_detected": is_drift,
                })
                
    # 2. Categorical drift via TVD
    for col, base_dist in baseline.get("categorical", {}).items():
        if col in new_df.columns:
            total_features += 1
            new_dist = new_df[col].value_counts(normalize=True).to_dict()
            all_keys = set(base_dist.keys()).union(set(new_dist.keys()))
            tvd = 0.5 * sum(abs(base_dist.get(k, 0.0) - new_dist.get(k, 0.0)) for k in all_keys)
            is_drift = bool(tvd > 0.25)
            if is_drift:
                num_drifted += 1
            feature_drift_results.append({
                "feature": col,
                "type": "categorical",
                "metric": "Total_Variation_Distance",
                "score": round(float(tvd), 4),
                "p_value": None,
                "drift_detected": is_drift,
            })
            
    composite_drift_ratio = round(num_drifted / max(total_features, 1), 3)
    drift_alert = bool(composite_drift_ratio >= 0.25)
    
    return {
        "drift_detected": drift_alert,
        "composite_drift_score": composite_drift_ratio,
        "drifted_features_count": num_drifted,
        "total_features_monitored": total_features,
        "feature_details": feature_drift_results,
    }


class DriftDetector:
    """Class wrapper for data drift detection operations."""

    def __init__(self, baseline_path=BASELINE_STATS_PATH):
        self.baseline_path = baseline_path

    def check_drift(self, incoming_data: list = None) -> dict:
        if incoming_data is None:
            # Check against recent monitoring logs if available
            try:
                from src.mlops.monitoring_db import get_recent_predictions
                recent = get_recent_predictions(limit=50)
                incoming_data = [json.loads(r["data_json"]) for r in recent if r.get("data_json")]
            except Exception:
                incoming_data = []

        if not incoming_data:
            return {
                "drift_detected": False,
                "composite_drift_score": 0.0,
                "drifted_features_count": 0,
                "total_features_monitored": 0,
                "summary": "No incoming streaming data to evaluate."
            }

        res = detect_data_drift(incoming_data, baseline_path=self.baseline_path)
        res["summary"] = "Drift detected across features." if res["drift_detected"] else "Features within baseline tolerance."
        return res

