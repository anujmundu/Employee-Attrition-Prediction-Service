import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

SUPERVISED_MODEL_PATH = "models/model_v1.joblib"
CLUSTER_MODEL_PATH = "models/kmeans_v1.joblib"
ANOMALY_MODEL_PATH = "models/anomaly_v1.joblib"
DATA_PATH = "data/raw.csv"


def main():
    # Load trained supervised pipeline
    pipeline = joblib.load(SUPERVISED_MODEL_PATH)

    # Extract preprocessing only
    preprocessor = pipeline.named_steps["preprocessing"]

    # Load raw data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Attrition"])

    # Transform data using the SAME preprocessing
    X_processed = preprocessor.transform(X)

    # ---- KMeans for segmentation ----
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_processed)

    # ---- Isolation Forest for anomaly detection ----
    anomaly_model = IsolationForest(contamination=0.05, random_state=42)
    anomaly_model.fit(X_processed)

    # Save models
    joblib.dump(kmeans, CLUSTER_MODEL_PATH)
    joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)

    print("Unsupervised models trained and saved.")


if __name__ == "__main__":
    main()
