import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
from src.financial.turnover_cost import calculate_turnover_financials, determine_seniority_tier
from src.explainability.playbooks import generate_retention_playbook


def test_turnover_financials():
    fin = calculate_turnover_financials(
        monthly_income=6000,
        job_role="Senior Engineer",
        years_at_company=4,
        attrition_probability=0.75,
    )
    assert fin["annual_salary"] == 72000.0
    assert fin["seniority_tier"] in ["MID", "SENIOR"]
    assert fin["replacement_cost"] > 0
    assert fin["expected_loss_at_risk"] > 0
    assert "net_retention_roi_percent" in fin


def test_seniority_tiers():
    assert determine_seniority_tier("VP of Technology", 18000, 8) == "EXECUTIVE"
    assert determine_seniority_tier("Junior Associate", 2800, 1) == "ENTRY"


def test_retention_playbooks_high_risk():
    emp = {
        "OverTime": "Yes",
        "WorkLifeBalance": 1,
        "MonthlyIncome": 3200,
        "YearsSinceLastPromotion": 5,
        "DistanceFromHome": 35,
    }
    playbook = generate_retention_playbook(emp)
    assert len(playbook) >= 3
    pillars = [p["pillar"] for p in playbook]
    assert any("Burnout" in p for p in pillars)
    assert any("Compensation" in p for p in pillars)
