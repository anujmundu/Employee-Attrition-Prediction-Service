import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI
from io import BytesIO
from fastapi.responses import StreamingResponse
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from drift import check_drift


DB_PATH = "monitoring.db"

app = FastAPI()


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df


@app.get("/metrics")
def metrics():
    df = load_data()

    if len(df) < 10:
        return {"message": "Not enough data"}

    # Drift
    drift = check_drift(df[[
        "Age","DistanceFromHome","Education","EnvironmentSatisfaction","JobSatisfaction",
        "MonthlyIncome","NumCompaniesWorked","WorkLifeBalance","YearsAtCompany",
        "Department","EducationField","MaritalStatus"
    ]].to_dict(orient="records"))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Probability distribution
    axs[0, 0].hist(df["probability"], bins=20)
    axs[0, 0].set_title("Attrition Probability")

    # 2. Anomaly rate
    axs[0, 1].bar(["Normal","Anomalous"], [
        (df["iso_anomaly"] == 0).sum(),
        (df["iso_anomaly"] == 1).sum()
    ])
    axs[0, 1].set_title("Isolation Forest Anomalies")

    # 3. Deep anomaly
    axs[1, 0].bar(["Normal","Anomalous"], [
        (df["deep_anomaly"] == 0).sum(),
        (df["deep_anomaly"] == 1).sum()
    ])
    axs[1, 0].set_title("Deep Anomalies")

    # 4. Cluster traffic
    df["cluster"].value_counts().plot(kind="bar", ax=axs[1, 1])
    axs[1, 1].set_title("Cluster Distribution")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")
