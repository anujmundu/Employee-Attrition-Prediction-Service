import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import numpy as np

from src.core.config import (
    MASTER_BENCHMARK_PATH,
    RAW_DATA_PATH,
    DATASETS_DIR,
    SUPERVISED_PIPELINE_PATH,
    TARGET_COL,
)
from src.data_generator.industry_datasets import generate_all_datasets
from src.models.supervised import SupervisedModelSuite
from src.models.unsupervised import UnsupervisedModelSuite
from src.models.deep_learning import train_deep_models
from src.models.nlp_engine import NLPEngine
from src.mlops.drift_detector import save_baseline_statistics
from src.mlops.monitoring_db import init_db


def main():
    print("=" * 70)
    print("  EMPLOYEE ATTRITION PREDICTION SERVICE - MASTER TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Ensure Data Exists
    if not MASTER_BENCHMARK_PATH.exists() or len(list(DATASETS_DIR.glob("*.csv"))) < 20:
        print("\n[Step 1/6] Generating 20 specialized industry benchmark datasets...")
        generate_all_datasets()
    else:
        print(f"\n[Step 1/6] Loading existing master benchmark dataset from {MASTER_BENCHMARK_PATH}")
        
    df = pd.read_csv(MASTER_BENCHMARK_PATH)
    print(f"Loaded {len(df)} total employee records.")
    
    # 2. Train Supervised Model Suite
    print("\n[Step 2/6] Training Supervised Model Suite & Calibrated Ensemble...")
    supervised_suite = SupervisedModelSuite()
    benchmarks = supervised_suite.fit_and_evaluate(df)
    
    pipeline = joblib.load(SUPERVISED_PIPELINE_PATH)
    preprocessor = pipeline.named_steps["preprocessing"]
    
    # Prepare preprocessed matrix for unsupervised & deep learning
    available_cols = preprocessor.feature_names_in_
    X_raw = df[available_cols]
    X_processed = preprocessor.transform(X_raw)
    y = df[TARGET_COL].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).values
    
    # 3. Train Unsupervised Models
    print("\n[Step 3/6] Training Unsupervised Models (KMeans, GMM, PCA, IsoForest, LOF)...")
    unsupervised_suite = UnsupervisedModelSuite(n_clusters=4)
    unsupervised_suite.fit(X_processed)
    
    # 4. Train Deep Learning Suite (PyTorch)
    print("\n[Step 4/6] Training PyTorch Deep Learning Models (Autoencoder, VAE, ResNet)...")
    train_deep_models(X_processed, y=y, epochs=12)
    
    # 5. Train NLP Engine
    print("\n[Step 5/6] Training NLP Engine & Aspect Sentiment Lexicons...")
    nlp = NLPEngine()
    if "EmployeeFeedback" in df.columns:
        nlp.fit(df["EmployeeFeedback"].dropna().tolist())
    else:
        nlp.fit(["Good working conditions.", "High workload and overtime."])
        
    # 6. Save Drift Baseline Statistics & Init DB
    print("\n[Step 6/6] Computing Statistical Drift Baselines & Initializing DB...")
    save_baseline_statistics(df)
    init_db()
    
    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE! ALL MODELS & BENCHMARKS SAVED.")
    print("=" * 70)
    for model_name, metrics in benchmarks.items():
        print(f"  * {model_name:22s} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
