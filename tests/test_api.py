import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "version" in data


def test_list_datasets():
    res = client.get("/v1/datasets")
    assert res.status_code == 200
    data = res.json()
    assert data["total_datasets"] == 20
    assert len(data["datasets"]) == 20


def test_kpi_endpoint():
    res = client.get("/v1/kpis")
    assert res.status_code == 200
    data = res.json()
    assert "total_predictions" in data
    assert "average_trust_score" in data


def test_list_online_datasets():
    res = client.get("/v1/online-datasets")
    assert res.status_code == 200
    data = res.json()
    assert data["total_online_datasets"] == 20
    assert len(data["datasets"]) == 20
