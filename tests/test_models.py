import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
from src.predict import get_engine


def test_prediction_engine_single():
    engine = get_engine()
    sample = {
        "Age": 38,
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "MonthlyIncome": 5400,
        "OverTime": "No",
        "DistanceFromHome": 6,
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsSinceLastPromotion": 1,
    }
    
    result = engine.predict_single(sample)
    
    assert "attrition_probability" in result
    assert 0.0 <= result["attrition_probability"] <= 1.0
    assert result["risk_tier"] in ["MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "data_trust_score" in result
    assert result["data_trust_score"] >= 0.0
    assert "cluster_id" in result
    assert "persona_name" in result
    assert "financials" in result
    assert result["financials"]["replacement_cost"] > 0
    assert "retention_playbook" in result
    assert len(result["retention_playbook"]) > 0


def test_prediction_engine_anomalous():
    engine = get_engine()
    # Outlier / extreme profile
    extreme_sample = {
        "Age": 68,
        "MonthlyIncome": 95000,
        "DistanceFromHome": 95,
        "YearsAtCompany": 35,
        "NumCompaniesWorked": 18,
    }
    result = engine.predict_single(extreme_sample)
    assert "data_trust_score" in result
    assert "reconstruction_error" in result
