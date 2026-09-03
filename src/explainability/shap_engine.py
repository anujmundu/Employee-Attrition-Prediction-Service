import numpy as np
import pandas as pd


class FeatureAttributionEngine:
    """
    Computes directional feature attributions (SHAP-style) for individual predictions,
    identifying the top positive (risk-increasing) and negative (risk-reducing) drivers.
    """
    
    def __init__(self, pipeline, feature_names: list, baseline_means: dict = None):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.baseline_means = baseline_means or {}

    def explain_instance(self, input_df: pd.DataFrame, top_k: int = 5) -> dict:
        """
        Computes perturbation-based feature sensitivity attributions for a single instance.
        Returns top risk-increasing and protective drivers with human-readable descriptions.
        """
        base_prob = float(self.pipeline.predict_proba(input_df)[0][1])
        attributions = []
        
        # Test marginal impact of each key feature
        for col in input_df.columns:
            val = input_df[col].iloc[0]
            if hasattr(val, "item"):
                val = val.item()
            perturbed_df = input_df.copy()
            
            # Substitute with neutral/baseline value
            if col in ["OverTime"]:
                perturbed_df[col] = "No" if val == "Yes" else "Yes"
            elif col in ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]:
                perturbed_df[col] = 4 if val <= 2 else 2
            elif col in ["MonthlyIncome"]:
                perturbed_df[col] = val * 1.25 if val < 7000 else val * 0.8
            elif col in ["YearsSinceLastPromotion"]:
                perturbed_df[col] = 0 if val >= 3 else 4
            elif col in ["DistanceFromHome"]:
                perturbed_df[col] = 2 if val >= 15 else 25
            else:
                continue
                
            try:
                new_prob = float(self.pipeline.predict_proba(perturbed_df)[0][1])
                diff = base_prob - new_prob  # positive means current value increases risk
                
                if abs(diff) > 0.005:
                    attributions.append({
                        "feature": col,
                        "current_value": val,
                        "marginal_impact": round(diff, 4),
                        "direction": "INCREASES_RISK" if diff > 0 else "REDUCES_RISK",
                        "description": self._format_description(col, val, diff),
                    })
            except Exception:
                continue
                
        # Sort by absolute impact
        attributions.sort(key=lambda x: abs(x["marginal_impact"]), reverse=True)
        
        risk_drivers = [a for a in attributions if a["direction"] == "INCREASES_RISK"][:top_k]
        protective_factors = [a for a in attributions if a["direction"] == "REDUCES_RISK"][:top_k]
        
        return {
            "base_probability": round(base_prob, 4),
            "top_risk_drivers": risk_drivers,
            "top_protective_factors": protective_factors,
            "all_attributions": attributions[:top_k * 2],
        }

    def _format_description(self, col: str, val, diff: float) -> str:
        pct = abs(round(diff * 100, 1))
        if col == "OverTime":
            return f"Overtime status '{val}' {'adds' if diff > 0 else 'removes'} ~{pct}% attrition risk."
        if col in ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]:
            qual = "Low" if val <= 2 else "High"
            return f"{qual} rating ({val}/4) in {col} {'increases risk by' if diff > 0 else 'mitigates risk by'} ~{pct}%."
        if col == "MonthlyIncome":
            return f"Compensation level (${int(val)}/mo) {'increases pressure by' if diff > 0 else 'provides protection of'} ~{pct}%."
        if col == "YearsSinceLastPromotion":
            return f"{int(val)} years without promotion adds ~{pct}% turnover risk."
        if col == "DistanceFromHome":
            return f"{int(val)} km commute adds ~{pct}% attrition risk."
        return f"{col}={val} shifts probability by {pct}%."
