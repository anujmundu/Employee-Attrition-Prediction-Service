import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.core.config import SUPERVISED_PIPELINE_PATH, MASTER_BENCHMARK_PATH


def main():
    print("=" * 70)
    print("   RETAIN-AI ENTERPRISE | EMPLOYEE ATTRITION INTELLIGENCE SERVICE")
    print("=" * 70)
    
    # Check if models are trained
    if not SUPERVISED_PIPELINE_PATH.exists() or not MASTER_BENCHMARK_PATH.exists():
        print("\n[Notice] Trained model artifacts or benchmark datasets not detected.")
        print("Initiating automated pipeline: generating 20 industry datasets and training all models...")
        import src.train_all_models
        src.train_all_models.main()
    else:
        print("\n[OK] Model artifacts and datasets detected and ready.")
        
    print("\nStarting RetainAI Enterprise FastAPI Service & Executive Web Portal...")
    print("Access the Interactive Dashboard at:  http://127.0.0.1:8000")
    print("Access the Interactive OpenAPI Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
