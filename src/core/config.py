import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
ONLINE_DATASETS_DIR = DATA_DIR / "online_datasets"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
STATIC_DIR = SRC_DIR / "web" / "static"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ONLINE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_PATH = DATA_DIR / "raw.csv"
WITH_TEXT_DATA_PATH = DATA_DIR / "with_text.csv"
MASTER_BENCHMARK_PATH = DATA_DIR / "master_benchmark.csv"
MONITORING_DB_PATH = BASE_DIR / "monitoring.db"
ONLINE_CATALOG_PATH = ONLINE_DATASETS_DIR / "catalog.json"

# Model Paths
SUPERVISED_PIPELINE_PATH = MODELS_DIR / "model_v1.joblib"
SUPERVISED_ENSEMBLE_PATH = MODELS_DIR / "ensemble_model.joblib"
MODEL_BENCHMARK_PATH = MODELS_DIR / "model_benchmark.json"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_v1.joblib"
GMM_MODEL_PATH = MODELS_DIR / "gmm_v1.joblib"
PCA_MODEL_PATH = MODELS_DIR / "pca_v1.joblib"
ISOLATION_FOREST_PATH = MODELS_DIR / "anomaly_v1.joblib"
LOF_MODEL_PATH = MODELS_DIR / "lof_v1.joblib"
AUTOENCODER_PATH = MODELS_DIR / "autoencoder.pt"
VAE_PATH = MODELS_DIR / "vae.pt"
TABULAR_RESNET_PATH = MODELS_DIR / "tabular_resnet.pt"
BASELINE_STATS_PATH = MODELS_DIR / "baseline_stats.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf.joblib"
TEXT_EMBEDDINGS_PATH = MODELS_DIR / "text_embeddings.npy"

# Feature Definitions
NUMERICAL_COLS = [
    "Age",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "WorkLifeBalance",
    "YearsAtCompany",
    "TotalWorkingYears",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "PerformanceRating",
    "PercentSalaryHike",
    "TrainingTimesLastYear",
]

CATEGORICAL_COLS = [
    "Department",
    "EducationField",
    "MaritalStatus",
    "JobRole",
    "OverTime",
    "BusinessTravel",
]

TARGET_COL = "Attrition"

# Risk Thresholds
RISK_THRESHOLDS = {
    "MINIMAL": 0.15,
    "LOW": 0.35,
    "MODERATE": 0.60,
    "HIGH": 0.80,
    "CRITICAL": 1.00,
}

# Turnover Cost Multipliers (fraction of annual salary)
TURNOVER_COST_RATES = {
    "ENTRY": 0.50,      # Junior / Entry level
    "MID": 0.75,        # Mid-level individual contributor
    "SENIOR": 1.25,     # Senior specialist / Lead
    "EXECUTIVE": 1.75,  # Director / Executive
}
