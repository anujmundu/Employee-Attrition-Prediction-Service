import joblib
import pandas as pd
import torch
import torch.nn as nn

PIPELINE_PATH = "models/model_v1.joblib"
CLUSTER_PATH = "models/kmeans_v1.joblib"
ANOMALY_PATH = "models/anomaly_v1.joblib"
AUTOENCODER_PATH = "models/autoencoder.pt"

pipeline = joblib.load(PIPELINE_PATH)
kmeans = joblib.load(CLUSTER_PATH)
anomaly_model = joblib.load(ANOMALY_PATH)

preprocessor = pipeline.named_steps["preprocessing"]


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# Load autoencoder
input_dim = preprocessor.transform(
    pd.DataFrame([{
        "Age": 30,
        "DistanceFromHome": 10,
        "Education": 3,
        "EnvironmentSatisfaction": 3,
        "JobSatisfaction": 3,
        "MonthlyIncome": 50000,
        "NumCompaniesWorked": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "Department": "Sales",
        "EducationField": "Life Sciences",
        "MaritalStatus": "Married"
    }])
).shape[1]

autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH))
autoencoder.eval()


def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(df)

    # Supervised
    prob = pipeline.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    # Clustering
    cluster = int(kmeans.predict(X_processed)[0])

    # Isolation Forest anomaly
    iso_anomaly = int(anomaly_model.predict(X_processed)[0] == -1)

    # Autoencoder anomaly
    x_tensor = torch.tensor(X_processed, dtype=torch.float32)
    with torch.no_grad():
        recon = autoencoder(x_tensor)
        recon_error = torch.mean((recon - x_tensor) ** 2).item()

    deep_anomaly = int(recon_error > 1.0)  # simple threshold

    return {
        "attrition_probability": float(prob),
        "attrition_prediction": pred,
        "cluster_id": cluster,
        "is_isolation_forest_anomaly": iso_anomaly,
        "reconstruction_error": recon_error,
        "is_deep_anomaly": deep_anomaly
    }
