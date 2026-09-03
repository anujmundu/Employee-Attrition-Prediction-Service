"""
Model Latency and Throughput Performance Benchmarking Script.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import get_engine


def benchmark_inference(iterations: int = 100):
    print("=" * 60)
    print(f" BENCHMARKING PREDICTION ENGINE ({iterations} iterations)")
    print("=" * 60)
    
    engine = get_engine()
    sample = {
        "Age": 36,
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "MonthlyIncome": 6200,
        "OverTime": "No",
        "DistanceFromHome": 8,
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 4,
        "YearsSinceLastPromotion": 1,
        "ExitSurveyNotes": "Positive team culture, moderate work pace."
    }
    
    # Warmup
    for _ in range(10):
        engine.predict_single(sample)
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.predict_single(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    latencies = np.array(latencies)
    print(f"P50 Latency:  {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 Latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency:  {np.percentile(latencies, 99):.2f} ms")
    print(f"Mean Latency: {np.mean(latencies):.2f} ms (+/- {np.std(latencies):.2f} ms)")
    print(f"Throughput:   {1000 / np.mean(latencies):.1f} predictions/sec (single thread)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    benchmark_inference(100)
