import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
from src.financial.turnover_cost import TurnoverCostCalculator
from src.mlops.monitoring_db import MonitoringDB
from src.mlops.drift_detector import DriftDetector


def test_financial_turnover_cost():
    calculator = TurnoverCostCalculator()
    
    # Entry level
    res_entry = calculator.calculate_cost(
        job_role="Laboratory Technician",
        department="Research & Development",
        monthly_income=3500,
        years_at_company=1
    )
    assert res_entry["replacement_cost"] > 0
    assert res_entry["seniority_tier"] in ["ENTRY", "MID", "SENIOR", "EXECUTIVE"]
    assert "cost_breakdown" in res_entry
    assert "hiring_cost" in res_entry["cost_breakdown"]
    assert "lost_productivity_cost" in res_entry["cost_breakdown"]
    
    # Executive level
    res_exec = calculator.calculate_cost(
        job_role="Manager",
        department="Sales",
        monthly_income=18000,
        years_at_company=12
    )
    assert res_exec["replacement_cost"] > res_entry["replacement_cost"]
    assert res_exec["seniority_tier"] in ["SENIOR", "EXECUTIVE"]


def test_monitoring_db_and_drift():
    db = MonitoringDB()
    stats_initial = db.get_statistics()
    assert isinstance(stats_initial, dict)
    
    # Log sample prediction
    features = {
        "Age": 32,
        "MonthlyIncome": 5200,
        "OverTime": "No",
        "DistanceFromHome": 5
    }
    log_id = db.log_prediction(
        features=features,
        prediction=0.25,
        risk_tier="LOW",
        trust_score=94.5,
        latency_ms=12.4
    )
    assert log_id is not None
    
    detector = DriftDetector()
    drift_status = detector.check_drift()
    assert "drift_detected" in drift_status
    assert "summary" in drift_status or "feature_details" in drift_status or "composite_drift_score" in drift_status


