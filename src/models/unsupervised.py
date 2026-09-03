import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.core.config import (
    KMEANS_MODEL_PATH,
    GMM_MODEL_PATH,
    PCA_MODEL_PATH,
    ISOLATION_FOREST_PATH,
    LOF_MODEL_PATH,
)

CLUSTER_PERSONAS = {
    0: {
        "name": "Core Veteran Specialists",
        "description": "High tenure, deep institutional knowledge, moderate-to-high compensation.",
    },
    1: {
        "name": "High-Output Commercial Drivers",
        "description": "High business travel, intense overtime, high variable commission.",
    },
    2: {
        "name": "At-Risk Junior Contributors",
        "description": "Early career stage (<3 years), long commute distance, low initial satisfaction.",
    },
    3: {
        "name": "Balanced Technical Professionals",
        "description": "Healthy work-life balance, steady promotion velocity, stable team environment.",
    },
}


class UnsupervisedModelSuite:
    """
    Orchestrates unsupervised learning models:
    - KMeans: Archetype persona segmentation
    - GMM: Probabilistic soft cluster assignments
    - PCA: 2D latent coordinates for interactive portal scatter plots
    - Isolation Forest: Out-of-distribution anomaly detection
    - Local Outlier Factor (LOF): Density-based novelty scoring
    """

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        self.pca = PCA(n_components=2, random_state=42)
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)

    def fit(self, X_processed: np.ndarray):
        """Fits all unsupervised models on preprocessed tabular feature matrix."""
        print("Training Unsupervised Model Suite...")
        self.kmeans.fit(X_processed)
        self.gmm.fit(X_processed)
        self.pca.fit(X_processed)
        self.iso_forest.fit(X_processed)
        self.lof.fit(X_processed)
        
        joblib.dump(self.kmeans, KMEANS_MODEL_PATH)
        joblib.dump(self.gmm, GMM_MODEL_PATH)
        joblib.dump(self.pca, PCA_MODEL_PATH)
        joblib.dump(self.iso_forest, ISOLATION_FOREST_PATH)
        joblib.dump(self.lof, LOF_MODEL_PATH)
        print("Unsupervised Model Suite trained and serialized successfully.")

    def transform_and_score(self, X_processed: np.ndarray) -> dict:
        """Evaluates an employee vector across all unsupervised models."""
        cluster_id = int(self.kmeans.predict(X_processed)[0])
        persona = CLUSTER_PERSONAS.get(cluster_id, {"name": f"Segment {cluster_id}", "description": ""})
        
        # GMM soft probabilities
        gmm_probs = self.gmm.predict_proba(X_processed)[0].tolist()
        
        # PCA 2D coordinates
        pca_coords = self.pca.transform(X_processed)[0].tolist()
        
        # Isolation Forest
        iso_pred = int(self.iso_forest.predict(X_processed)[0] == -1)
        iso_score = float(self.iso_forest.score_samples(X_processed)[0])
        
        # LOF Novelty
        lof_pred = int(self.lof.predict(X_processed)[0] == -1)
        lof_score = float(self.lof.score_samples(X_processed)[0])
        
        return {
            "cluster_id": cluster_id,
            "persona_name": persona["name"],
            "persona_description": persona["description"],
            "gmm_cluster_probabilities": [round(p, 3) for p in gmm_probs],
            "pca_coordinates": [round(c, 3) for c in pca_coords],
            "is_isolation_forest_anomaly": iso_pred,
            "isolation_forest_score": round(iso_score, 4),
            "is_lof_anomaly": lof_pred,
            "lof_score": round(lof_score, 4),
        }
