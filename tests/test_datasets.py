import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
import pandas as pd
from src.data_generator.industry_datasets import INDUSTRY_PROFILES, generate_industry_dataset


def test_industry_profiles_count():
    assert len(INDUSTRY_PROFILES) == 20, "Must have exactly 20 specialized industry profiles"


def test_industry_dataset_generation():
    profile = INDUSTRY_PROFILES[0]
    df = generate_industry_dataset(profile, n_samples=30)
    
    assert len(df) == 30
    assert "Attrition" in df.columns
    assert "MonthlyIncome" in df.columns
    assert "Department" in df.columns
    assert "EmployeeFeedback" in df.columns
    assert df["Attrition"].isin(["Yes", "No"]).all()
    assert (df["MonthlyIncome"] > 0).all()
