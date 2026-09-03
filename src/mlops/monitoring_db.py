import json
import sqlite3
from datetime import datetime
from src.core.config import MONITORING_DB_PATH


def get_connection():
    return sqlite3.connect(str(MONITORING_DB_PATH))


def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Individual predictions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            employee_id TEXT,
            department TEXT,
            job_role TEXT,
            monthly_income REAL,
            probability REAL,
            prediction INTEGER,
            risk_tier TEXT,
            trust_score REAL,
            cluster_id INTEGER,
            iso_anomaly INTEGER,
            deep_anomaly INTEGER,
            lof_anomaly INTEGER,
            replacement_cost REAL,
            expected_loss REAL,
            data_json TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_tier ON predictions(risk_tier)")
    
    # 2. Batch runs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS batch_runs (
            batch_id TEXT PRIMARY KEY,
            timestamp TEXT,
            file_name TEXT,
            total_records INTEGER,
            avg_probability REAL,
            high_risk_count INTEGER,
            total_loss_at_risk REAL
        )
    """)
    
    # 3. Drift monitoring logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS drift_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ks_drift_score REAL,
            drift_detected INTEGER,
            drifted_features_json TEXT
        )
    """)
    
    conn.commit()
    conn.close()


def log_prediction(data: dict, result: dict, employee_id: str = None) -> int:
    conn = get_connection()
    c = conn.cursor()
    
    fin = result.get("financials", {})
    now = datetime.utcnow().isoformat()
    emp_id = employee_id or f"EMP-{int(datetime.utcnow().timestamp() * 1000) % 100000:05d}"
    
    c.execute("""
        INSERT INTO predictions (
            timestamp, employee_id, department, job_role, monthly_income,
            probability, prediction, risk_tier, trust_score, cluster_id,
            iso_anomaly, deep_anomaly, lof_anomaly, replacement_cost,
            expected_loss, data_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        now,
        emp_id,
        data.get("Department", "General"),
        data.get("JobRole", "Specialist"),
        float(data.get("MonthlyIncome", 5000)),
        float(result.get("attrition_probability", 0.0)),
        int(result.get("attrition_prediction", 0)),
        result.get("risk_tier", "LOW"),
        float(result.get("data_trust_score", 100.0)),
        int(result.get("cluster_id", 0)),
        int(result.get("is_isolation_forest_anomaly", 0)),
        int(result.get("is_deep_anomaly", 0)),
        int(result.get("is_lof_anomaly", 0)),
        float(fin.get("replacement_cost", 0.0)),
        float(fin.get("expected_loss_at_risk", 0.0)),
        json.dumps(data)
    ))
    
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id


def log_batch_run(batch_id: str, file_name: str, total_records: int, avg_prob: float, high_risk_count: int, total_loss: float):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO batch_runs VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (batch_id, datetime.utcnow().isoformat(), file_name, total_records, avg_prob, high_risk_count, total_loss))
    conn.commit()
    conn.close()


def log_drift_event(ks_score: float, drift_detected: bool, drifted_features: list):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO drift_logs (timestamp, ks_drift_score, drift_detected, drifted_features_json)
        VALUES (?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), ks_score, int(drift_detected), json.dumps(drifted_features)))
    conn.commit()
    conn.close()


def get_kpi_summary():
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*), AVG(probability), SUM(expected_loss) FROM predictions")
    row = c.fetchone()
    total_predictions = row[0] or 0
    avg_probability = round(row[1] or 0.0, 3)
    total_loss_at_risk = round(row[2] or 0.0, 2)
    
    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_tier IN ('HIGH', 'CRITICAL')")
    high_risk_count = c.fetchone()[0] or 0
    
    c.execute("SELECT AVG(trust_score) FROM predictions")
    avg_trust = round(c.fetchone()[0] or 95.0, 1)
    
    conn.close()
    return {
        "total_predictions": total_predictions,
        "average_attrition_risk": avg_probability,
        "total_loss_at_risk": total_loss_at_risk,
        "high_risk_count": high_risk_count,
        "average_trust_score": avg_trust,
    }


def get_recent_predictions(limit: int = 50):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM predictions ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
