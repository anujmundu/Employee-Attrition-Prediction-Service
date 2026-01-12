import sqlite3
import pandas as pd
from drift import check_drift
import subprocess

DB_PATH = "monitoring.db"
DRIFT_THRESHOLD = 1.0


def get_recent_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
    SELECT
        Age,
        DistanceFromHome,
        Education,
        EnvironmentSatisfaction,
        JobSatisfaction,
        MonthlyIncome,
        NumCompaniesWorked,
        WorkLifeBalance,
        YearsAtCompany,
        Department,
        EducationField,
        MaritalStatus
    FROM predictions
    """, conn)
    conn.close()

    return df


def main():
    df = get_recent_data()

    if len(df) < 50:
        print("Not enough data to evaluate drift")
        return

    drift_score = check_drift(df.to_dict(orient="records"))
    print("Drift score:", drift_score)

    if drift_score > DRIFT_THRESHOLD:
        print("Drift detected. Retraining model...")
        subprocess.run(["python", "src/train.py"])
    else:
        print("No significant drift.")


if __name__ == "__main__":
    main()
