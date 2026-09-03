import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import torch
import numpy as np
import pandas as pd

from src.core.config import (
    SUPERVISED_PIPELINE_PATH,
    KMEANS_MODEL_PATH,
    GMM_MODEL_PATH,
    PCA_MODEL_PATH,
    ISOLATION_FOREST_PATH,
    LOF_MODEL_PATH,
    AUTOENCODER_PATH,
    VAE_PATH,
    RISK_THRESHOLDS,
)
from src.models.deep_learning import AutoencoderNet, VAENet
from src.models.nlp_engine import NLPEngine
from src.explainability.shap_engine import FeatureAttributionEngine
from src.explainability.playbooks import generate_retention_playbook
from src.financial.turnover_cost import calculate_turnover_financials
from src.models.unsupervised import CLUSTER_PERSONAS


def to_native_types(obj):
    """Recursively converts all NumPy types to standard Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native_types(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [to_native_types(x) for x in obj.tolist()]
    return obj


class PredictionEngine:
    """
    Unified Production Inference Engine combining:
    1. Supervised Attrition Probability & Risk Tiering
    2. Unsupervised Persona Clustering & Latent PCA Projections
    3. Anomaly & Data Trust Shield (Isolation Forest, LOF, Deep Autoencoder, VAE)
    4. NLP Aspect Sentiment Extraction
    5. Feature Attribution (SHAP-style sensitivity)
    6. Prescriptive Retention Playbooks
    7. Turnover Financial Cost & ROI Modeling
    """

    def __init__(self):
        self._loaded = False
        self.pipeline = None
        self.preprocessor = None
        self.kmeans = None
        self.gmm = None
        self.pca = None
        self.iso_forest = None
        self.lof = None
        self.autoencoder = None
        self.vae = None
        self.nlp = None
        self.explainer = None

    def load_models(self):
        """Loads all serialized models into memory with lazy initialization."""
        if self._loaded:
            return

        if not SUPERVISED_PIPELINE_PATH.exists():
            try:
                from src.train_all_models import main as train_models
                train_models()
            except Exception as e:
                raise FileNotFoundError(
                    f"Trained models not found in {SUPERVISED_PIPELINE_PATH.parent}. "
                    f"Auto-training failed: {e}. Please run `python src/train_all_models.py`."
                )

        self.pipeline = joblib.load(SUPERVISED_PIPELINE_PATH)
        self.preprocessor = self.pipeline.named_steps["preprocessing"]


        if KMEANS_MODEL_PATH.exists():
            self.kmeans = joblib.load(KMEANS_MODEL_PATH)
        if GMM_MODEL_PATH.exists():
            self.gmm = joblib.load(GMM_MODEL_PATH)
        if PCA_MODEL_PATH.exists():
            self.pca = joblib.load(PCA_MODEL_PATH)
        if ISOLATION_FOREST_PATH.exists():
            self.iso_forest = joblib.load(ISOLATION_FOREST_PATH)
        if LOF_MODEL_PATH.exists():
            self.lof = joblib.load(LOF_MODEL_PATH)

        # Deep Learning Models
        sample_dim = len(self.preprocessor.get_feature_names_out())
        if AUTOENCODER_PATH.exists():
            ae = AutoencoderNet(sample_dim)
            ae.load_state_dict(torch.load(AUTOENCODER_PATH, map_location="cpu"))
            ae.eval()
            self.autoencoder = ae

        if VAE_PATH.exists():
            vae = VAENet(sample_dim)
            vae.load_state_dict(torch.load(VAE_PATH, map_location="cpu"))
            vae.eval()
            self.vae = vae

        self.nlp = NLPEngine()
        self.explainer = FeatureAttributionEngine(
            self.pipeline,
            feature_names=list(self.preprocessor.feature_names_in_),
        )
        self._loaded = True

    def _prepare_dataframe(self, input_data: dict) -> pd.DataFrame:
        """Sanitizes input dict and infers missing defaults based on standard medians."""
        clean = dict(input_data)
        defaults = {
            "Age": 35,
            "DistanceFromHome": 8,
            "Education": 3,
            "EnvironmentSatisfaction": 3,
            "JobSatisfaction": 3,
            "MonthlyIncome": 5000,
            "NumCompaniesWorked": 2,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 4,
            "TotalWorkingYears": 8,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
            "PerformanceRating": 3,
            "PercentSalaryHike": 14,
            "TrainingTimesLastYear": 3,
            "Department": "Research & Development",
            "JobRole": "Research Scientist",
            "EducationField": "Life Sciences",
            "MaritalStatus": "Married",
            "OverTime": "No",
            "BusinessTravel": "Travel_Rarely",
        }
        for k, v in defaults.items():
            if k not in clean or clean[k] is None or clean[k] == "":
                clean[k] = v

        return pd.DataFrame([clean])

    def predict_single(self, input_data: dict) -> dict:
        """Full multi-layer prediction for a single employee."""
        self.load_models()
        df = self._prepare_dataframe(input_data)

        # 1. Supervised Attrition Probability
        prob = float(self.pipeline.predict_proba(df)[0][1])
        pred = int(prob >= 0.40)

        risk_tier = "MINIMAL"
        for tier, thresh in RISK_THRESHOLDS.items():
            if prob <= thresh:
                risk_tier = tier
                break

        # 2. Preprocessed Matrix for Unsupervised & Deep Learning
        X_processed = self.preprocessor.transform(df)

        # 3. Clustering & Personas
        cluster_id = int(self.kmeans.predict(X_processed)[0]) if self.kmeans else 0
        persona = CLUSTER_PERSONAS.get(cluster_id, {"name": f"Segment {cluster_id}", "description": ""})
        pca_coords = self.pca.transform(X_processed)[0].tolist() if self.pca else [0.0, 0.0]

        # 4. Anomaly Detection & Trust Shield
        iso_anomaly = int(self.iso_forest.predict(X_processed)[0] == -1) if self.iso_forest else 0
        lof_anomaly = int(self.lof.predict(X_processed)[0] == -1) if self.lof else 0

        # Autoencoder Reconstruction Error
        recon_error = 0.0
        deep_anomaly = 0
        if self.autoencoder:
            with torch.no_grad():
                xt = torch.tensor(X_processed, dtype=torch.float32)
                recon = self.autoencoder(xt)
                recon_error = float(torch.mean((recon - xt) ** 2).item())
                deep_anomaly = int(recon_error > 1.2)

        # Calculate Data Trust Index (0-100%)
        trust_score = 98.0
        if iso_anomaly:
            trust_score -= 15.0
        if lof_anomaly:
            trust_score -= 15.0
        if deep_anomaly:
            trust_score -= min(35.0, recon_error * 20.0)
        trust_score = max(15.0, round(trust_score, 1))

        trust_status = "HIGH_CONFIDENCE"
        trust_warning = None
        if trust_score < 70.0:
            trust_status = "OUT_OF_DISTRIBUTION_WARNING"
            trust_warning = (
                "Employee profile is outside the standard training distribution. "
                "The supervised model prediction may be less reliable."
            )

        # 5. NLP Aspect-Based Sentiment Analysis
        feedback_text = input_data.get("EmployeeFeedback", "")
        nlp_analysis = self.nlp.analyze_feedback(feedback_text)

        # 6. Feature Attributions (SHAP-style)
        explanations = self.explainer.explain_instance(df, top_k=4)

        # 7. Prescriptive Retention Playbook
        playbook = generate_retention_playbook(input_data, explanations.get("top_risk_drivers", []))

        # 8. Financial Turnover Cost & ROI
        financials = calculate_turnover_financials(
            monthly_income=float(df["MonthlyIncome"].iloc[0]),
            job_role=str(df["JobRole"].iloc[0]),
            years_at_company=int(df["YearsAtCompany"].iloc[0]),
            attrition_probability=prob,
        )

        result = {
            "attrition_probability": round(prob, 4),
            "attrition_prediction": pred,
            "risk_tier": risk_tier,
            "data_trust_score": trust_score,
            "data_trust_status": trust_status,
            "trust_warning": trust_warning,
            "cluster_id": cluster_id,
            "persona_name": persona["name"],
            "persona_description": persona["description"],
            "pca_coordinates": [round(c, 3) for c in pca_coords],
            "is_isolation_forest_anomaly": iso_anomaly,
            "is_lof_anomaly": lof_anomaly,
            "is_deep_anomaly": deep_anomaly,
            "reconstruction_error": round(recon_error, 4),
            "nlp_analysis": nlp_analysis,
            "explanations": explanations,
            "retention_playbook": playbook,
            "financials": financials,
        }
        return to_native_types(result)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring for multiple employees (e.g. from uploaded CSV)."""
        self.load_models()
        records = df.to_dict(orient="records")
        scored = []
        for r in records:
            res = self.predict_single(r)
            fin = res["financials"]
            row_dict = {
                **r,
                "Attrition_Probability": res["attrition_probability"],
                "Risk_Tier": res["risk_tier"],
                "Data_Trust_Score": res["data_trust_score"],
                "Persona_Segment": res["persona_name"],
                "Anomaly_Flag": int(res["is_isolation_forest_anomaly"] or res["is_deep_anomaly"]),
                "Replacement_Cost": fin["replacement_cost"],
                "Expected_Loss_At_Risk": fin["expected_loss_at_risk"],
            }
            scored.append(row_dict)
        return pd.DataFrame(scored)

    def simulate_scenario(self, base_data: dict, modifications: dict) -> dict:
        """What-If scenario analysis comparing baseline vs modified employee profile."""
        base_result = self.predict_single(base_data)
        modified_data = {**base_data, **modifications}
        sim_result = self.predict_single(modified_data)

        prob_delta = round(sim_result["attrition_probability"] - base_result["attrition_probability"], 4)
        pct_change = round((prob_delta / max(base_result["attrition_probability"], 0.001)) * 100, 1)

        base_loss = base_result["financials"]["expected_loss_at_risk"]
        sim_loss = sim_result["financials"]["expected_loss_at_risk"]
        loss_saved = round(base_loss - sim_loss, 2)

        sim_output = {
            "baseline_probability": base_result["attrition_probability"],
            "simulated_probability": sim_result["attrition_probability"],
            "probability_delta": prob_delta,
            "percentage_risk_reduction": abs(pct_change) if pct_change < 0 else 0.0,
            "baseline_loss_at_risk": base_loss,
            "simulated_loss_at_risk": sim_loss,
            "projected_cost_savings": max(0.0, loss_saved),
            "baseline_risk_tier": base_result["risk_tier"],
            "simulated_risk_tier": sim_result["risk_tier"],
            "baseline_result": base_result,
            "simulated_result": sim_result,
        }
        return to_native_types(sim_output)


# Global singleton engine
_engine = None

def get_engine() -> PredictionEngine:
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine


def predict(input_data: dict) -> dict:
    """Backward-compatible entrypoint."""
    return get_engine().predict_single(input_data)
