import sqlite3
from datetime import datetime

DB_PATH = "monitoring.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            Age INTEGER,
            DistanceFromHome INTEGER,
            Education INTEGER,
            EnvironmentSatisfaction INTEGER,
            JobSatisfaction INTEGER,
            MonthlyIncome INTEGER,
            NumCompaniesWorked INTEGER,
            WorkLifeBalance INTEGER,
            YearsAtCompany INTEGER,
            Department TEXT,
            EducationField TEXT,
            MaritalStatus TEXT,
            cluster INTEGER,
            iso_anomaly INTEGER,
            deep_anomaly INTEGER,
            probability REAL,
            prediction INTEGER
        )

    """)
    conn.commit()
    conn.close()


def log_prediction(data, result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
    "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            data["Age"],
            data["DistanceFromHome"],
            data["Education"],
            data["EnvironmentSatisfaction"],
            data["JobSatisfaction"],
            data["MonthlyIncome"],
            data["NumCompaniesWorked"],
            data["WorkLifeBalance"],
            data["YearsAtCompany"],
            data["Department"],
            data["EducationField"],
            data["MaritalStatus"],
            result["cluster_id"],
            result["is_isolation_forest_anomaly"],
            result["is_deep_anomaly"],
            result["attrition_probability"],
            result["attrition_prediction"],
        ),
    )

    conn.commit()
    conn.close()
