import joblib
import numpy as np
import pandas as pd

BASELINE_PATH = "models/baseline_stats.joblib"
PIPELINE_PATH = "models/model_v1.joblib"

baseline = joblib.load(BASELINE_PATH)
pipeline = joblib.load(PIPELINE_PATH)
preprocessor = pipeline.named_steps["preprocessing"]


def check_drift(new_data: list):
    df = pd.DataFrame(new_data)
    X_new = preprocessor.transform(df)

    mean_diff = np.abs(X_new.mean(axis=0) - baseline["mean"])
    std_diff = np.abs(X_new.std(axis=0) - baseline["std"])

    drift_score = float(mean_diff.mean() + std_diff.mean())

    return drift_score
