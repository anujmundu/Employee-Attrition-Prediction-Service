import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import io
import json
import uuid
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.config import (
    STATIC_DIR,
    DATASETS_DIR,
    MODEL_BENCHMARK_PATH,
    SUPERVISED_PIPELINE_PATH,
)
from src.predict import get_engine, predict
from src.mlops.monitoring_db import (
    init_db,
    log_prediction,
    log_batch_run,
    get_kpi_summary,
    get_recent_predictions,
    log_drift_event,
)
from src.mlops.drift_detector import detect_data_drift
from src.data_generator.industry_datasets import INDUSTRY_PROFILES


app = FastAPI(
    title="Enterprise Employee Attrition Intelligence Service",
    version="2.0.0",
    description="Production ML System combining Supervised Ensembles, Unsupervised Personas, Deep Learning Trust Shields, SHAP Explainability, Prescriptive HR Playbooks, and Real-Time Financial ROI Modeling.",
)

# Enable CORS for enterprise integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    init_db()
    if SUPERVISED_PIPELINE_PATH.exists():
        try:
            get_engine().load_models()
            print("Production prediction models preloaded successfully.")
        except Exception as e:
            print(f"Warning during model preload: {e}")


# -------------------------------------------------------------
# Pydantic Request Schemas
# -------------------------------------------------------------
class EmployeeInput(BaseModel):
    Age: Optional[int] = Field(default=35, ge=18, le=75)
    Department: Optional[str] = "Research & Development"
    JobRole: Optional[str] = "Research Scientist"
    DistanceFromHome: Optional[int] = Field(default=8, ge=0, le=120)
    Education: Optional[int] = Field(default=3, ge=1, le=5)
    EducationField: Optional[str] = "Life Sciences"
    EnvironmentSatisfaction: Optional[int] = Field(default=3, ge=1, le=4)
    JobSatisfaction: Optional[int] = Field(default=3, ge=1, le=4)
    MaritalStatus: Optional[str] = "Married"
    MonthlyIncome: Optional[float] = Field(default=5000, ge=500, le=100000)
    NumCompaniesWorked: Optional[int] = Field(default=2, ge=0, le=25)
    WorkLifeBalance: Optional[int] = Field(default=3, ge=1, le=4)
    YearsAtCompany: Optional[int] = Field(default=4, ge=0, le=50)
    TotalWorkingYears: Optional[int] = Field(default=8, ge=0, le=55)
    YearsInCurrentRole: Optional[int] = Field(default=3, ge=0, le=40)
    YearsSinceLastPromotion: Optional[int] = Field(default=1, ge=0, le=30)
    YearsWithCurrManager: Optional[int] = Field(default=2, ge=0, le=30)
    PerformanceRating: Optional[int] = Field(default=3, ge=1, le=4)
    PercentSalaryHike: Optional[int] = Field(default=14, ge=0, le=100)
    TrainingTimesLastYear: Optional[int] = Field(default=3, ge=0, le=10)
    OverTime: Optional[str] = "No"
    BusinessTravel: Optional[str] = "Travel_Rarely"
    EmployeeFeedback: Optional[str] = ""
    EmployeeID: Optional[str] = None


class ScenarioSimulationInput(BaseModel):
    baseline_employee: Dict[str, Any]
    modifications: Dict[str, Any]


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.head("/")
def serve_dashboard():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Employee Attrition Prediction Service is Running</h1><p>Visit /docs for OpenAPI documentation.</p>"


@app.get("/health")
@app.head("/health")
def health_check():
    engine = get_engine()
    return {
        "status": "HEALTHY",
        "service": "Employee Attrition Prediction Service",
        "version": "2.0.0",
        "models_loaded": engine._loaded,
    }


@app.post("/v1/predict")
def predict_employee(data: EmployeeInput):
    """
    Computes multi-layer attrition intelligence:
    - Supervised probability & risk tier
    - Unsupervised persona segment & PCA coords
    - Anomaly flags (Isolation Forest, LOF, Autoencoder) & Data Trust Index
    - Aspect sentiment NLP
    - SHAP-style feature attributions
    - Prescriptive HR retention playbooks
    - Turnover cost & ROI model
    """
    input_dict = data.dict()
    emp_id = input_dict.pop("EmployeeID", None)
    
    try:
        result = get_engine().predict_single(input_dict)
        log_prediction(input_dict, result, employee_id=emp_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/simulate")
def simulate_intervention(payload: ScenarioSimulationInput):
    """What-If scenario analysis comparing baseline vs proposed interventions."""
    try:
        sim_res = get_engine().simulate_scenario(payload.baseline_employee, payload.modifications)
        return JSONResponse(content=sim_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Upload a CSV file of employee records for instant batch scoring."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        engine = get_engine()
        scored_df = engine.predict_batch(df)
        
        batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
        avg_prob = float(scored_df["Attrition_Probability"].mean())
        high_risk = int((scored_df["Risk_Tier"].isin(["HIGH", "CRITICAL"])).sum())
        total_loss = float(scored_df["Expected_Loss_At_Risk"].sum())
        
        log_batch_run(batch_id, file.filename, len(scored_df), avg_prob, high_risk, total_loss)
        
        return {
            "batch_id": batch_id,
            "filename": file.filename,
            "total_records": len(scored_df),
            "average_attrition_risk": round(avg_prob, 3),
            "high_risk_count": high_risk,
            "total_loss_at_risk": round(total_loss, 2),
            "records": scored_df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


@app.get("/v1/datasets")
def list_datasets():
    """Returns metadata for the 20 specialized industry datasets."""
    datasets = []
    for p in INDUSTRY_PROFILES:
        path = DATASETS_DIR / f"{p['id']}.csv"
        rows = 400
        if path.exists():
            rows = len(pd.read_csv(path))
        datasets.append({
            "id": p["id"],
            "name": p["name"],
            "departments": p["departments"],
            "roles": p["roles"],
            "base_attrition_rate": p["base_attrition_rate"],
            "row_count": rows,
        })
    return {"total_datasets": len(datasets), "datasets": datasets}


@app.get("/v1/datasets/{dataset_id}/sample")
def get_dataset_sample(dataset_id: str, n: int = 15):
    """Returns sample records from a specific industry dataset."""
    path = DATASETS_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found.")
    df = pd.read_csv(path)
    return df.head(n).to_dict(orient="records")


@app.get("/v1/online-datasets")
def list_online_datasets():
    """Returns catalog of the 20 real online downloaded GitHub datasets."""
    from src.core.config import ONLINE_CATALOG_PATH, ONLINE_DATASETS_DIR
    if ONLINE_CATALOG_PATH.exists():
        with open(ONLINE_CATALOG_PATH, "r") as f:
            return {"total_online_datasets": 20, "datasets": json.load(f)}
    
    files = list(ONLINE_DATASETS_DIR.glob("*.csv"))
    return {"total_online_datasets": len(files), "datasets": [{"id": f.stem, "name": f.stem.replace('_', ' ').title(), "local_file": f.name} for f in files]}


@app.get("/v1/online-datasets/{dataset_id}/sample")
def get_online_dataset_sample(dataset_id: str, n: int = 15):
    """Returns sample records from a specific downloaded online dataset."""
    from src.core.config import ONLINE_DATASETS_DIR
    path = ONLINE_DATASETS_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Online dataset not found.")
    df = pd.read_csv(path, low_memory=False)
    # Sanitize NaN values for JSON serialization
    df = df.head(n).fillna("")
    return df.to_dict(orient="records")


@app.get("/v1/model-benchmarks")
def get_benchmarks():
    """Returns comparative model benchmark metrics."""
    if MODEL_BENCHMARK_PATH.exists():
        with open(MODEL_BENCHMARK_PATH, "r") as f:
            return json.load(f)
    return {"message": "Model benchmarks not generated yet. Run train_all_models.py."}


@app.get("/v1/kpis")
def get_kpis():
    """Returns executive dashboard KPIs from logged predictions."""
    return get_kpi_summary()


@app.get("/v1/recent-predictions")
def get_recent(limit: int = 30):
    """Returns the most recent predictions for the live roster."""
    return get_recent_predictions(limit=limit)


@app.get("/v1/drift-status")
def get_drift():
    """Performs real-time Kolmogorov-Smirnov and TVD drift detection on recent predictions."""
    recent = get_recent_predictions(limit=100)
    if not recent:
        return {"drift_detected": False, "message": "No prediction traffic recorded yet."}
        
    records = []
    for r in recent:
        if "data_json" in r and r["data_json"]:
            records.append(json.loads(r["data_json"]))
            
    if len(records) < 5:
        return {"drift_detected": False, "message": "Fewer than 5 records; awaiting more traffic."}
        
    report = detect_data_drift(records)
    log_drift_event(report["composite_drift_score"], report["drift_detected"], report["feature_details"])
    return report
