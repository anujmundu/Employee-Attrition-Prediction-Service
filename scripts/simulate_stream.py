"""
Streaming Event Simulation Utility.
Simulates real-time HR events and prediction telemetry to test SQLite audit logging and MLOps KS drift detection.
"""

import random
import sys
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import get_engine
from src.mlops.monitoring_db import MonitoringDB
from src.mlops.drift_detector import DriftDetector


def generate_random_employee_event(drift_mode: bool = False) -> dict:
    """Generate a realistic employee feature record, optionally injecting drift."""
    dept = random.choice(["Sales", "Research & Development", "Human Resources"])
    job_roles = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": ["Research Scientist", "Software Engineer", "Laboratory Technician"],
        "Human Resources": ["HR Specialist", "Recruiter", "HR Manager"]
    }
    
    # In drift mode, skew features (e.g. extreme overtime, low satisfaction, high distance)
    if drift_mode:
        return {
            "Age": random.randint(21, 30),
            "Department": dept,
            "JobRole": random.choice(job_roles[dept]),
            "MonthlyIncome": random.randint(2000, 4500),
            "OverTime": "Yes",
            "DistanceFromHome": random.randint(35, 75),
            "JobSatisfaction": random.choice([1, 2]),
            "WorkLifeBalance": 1,
            "YearsAtCompany": random.randint(1, 3),
            "YearsSinceLastPromotion": random.randint(2, 4),
            "ExitSurveyNotes": "Severe burnout, extreme commute, lack of recognition."
        }
    else:
        return {
            "Age": random.randint(24, 60),
            "Department": dept,
            "JobRole": random.choice(job_roles[dept]),
            "MonthlyIncome": random.randint(4000, 18000),
            "OverTime": random.choice(["No", "No", "Yes"]),
            "DistanceFromHome": random.randint(1, 25),
            "JobSatisfaction": random.choice([2, 3, 4]),
            "WorkLifeBalance": random.choice([2, 3, 4]),
            "YearsAtCompany": random.randint(1, 15),
            "YearsSinceLastPromotion": random.randint(0, 5),
            "ExitSurveyNotes": "Normal work environment, standard career progression."
        }


def run_stream_simulation(num_events: int = 50, drift_start: int = 35, delay: float = 0.05):
    """Run stream simulation and log telemetry."""
    print("=" * 60)
    print(f" Starting HR Stream Telemetry Simulation ({num_events} events)")
    print("=" * 60)
    
    engine = get_engine()
    db = MonitoringDB()
    detector = DriftDetector()
    
    for i in range(1, num_events + 1):
        is_drift = i >= drift_start
        sample = generate_random_employee_event(drift_mode=is_drift)
        
        start_time = time.time()
        result = engine.predict_single(sample)
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to monitoring DB
        db.log_prediction(
            features=sample,
            prediction=result["attrition_probability"],
            risk_tier=result["risk_tier"],
            trust_score=result["data_trust_score"],
            latency_ms=latency_ms
        )
        
        status_tag = "[DRIFT REGIME]" if is_drift else "[NORMAL]"
        print(f"Event #{i:02d} {status_tag} | Risk: {result['attrition_probability']*100:.1f}% ({result['risk_tier']}) | Latency: {latency_ms:.1f}ms")
        
        if delay > 0:
            time.sleep(delay)
            
    print("\n--- Running Post-Stream Drift Check ---")
    drift_result = detector.check_drift()
    print(f"Drift Detected: {drift_result.get('drift_detected')}")
    print(f"Drift Summary:  {drift_result.get('summary', 'Evaluation completed.')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_stream_simulation(num_events=30, drift_start=20, delay=0.01)
