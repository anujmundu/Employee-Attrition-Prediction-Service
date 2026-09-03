import sys
import json
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.core.config import DATA_DIR

ONLINE_DATASETS_DIR = DATA_DIR / "online_datasets"
ONLINE_CATALOG_PATH = ONLINE_DATASETS_DIR / "catalog.json"

ONLINE_SOURCES = [
    {
        "id": "01_ibm_watson_original",
        "name": "IBM Watson HR Employee Attrition (Original)",
        "source_repo": "mrc03/IBM-HR-Analytics-Employee-Attrition-Performance",
        "url": "https://raw.githubusercontent.com/mrc03/IBM-HR-Analytics-Employee-Attrition-Performance/master/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "category": "Cross-Industry Corporate Benchmark",
        "description": "Canonical 35-feature IBM Watson employee attrition dataset with satisfaction, compensation, and tenure metrics."
    },
    {
        "id": "02_healthcare_attrition",
        "name": "Watson Healthcare Staff & Clinical Attrition",
        "source_repo": "marcello-calabrese/EDAHealthcareEmployeeAttrition",
        "url": "https://raw.githubusercontent.com/marcello-calabrese/EDAHealthcareEmployeeAttrition/main/watson_healthcare_modified.csv",
        "category": "Healthcare & Clinical Services",
        "description": "Modified Watson HR dataset modeled specifically on clinical nursing shifts, medical staff burnout, and hospital turnover."
    },
    {
        "id": "03_ibm_performance_revised",
        "name": "IBM HR Analytics Performance-Revised",
        "source_repo": "shantanu1109/IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction",
        "url": "https://raw.githubusercontent.com/shantanu1109/IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction/main/DATASET/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv",
        "category": "Performance Management",
        "description": "Revised performance ratings, promotion cycles, and tenure milestones linked to attrition outcomes."
    },
    {
        "id": "04_ibm_performance_baseline",
        "name": "IBM HR Performance Baseline Dataset",
        "source_repo": "shantanu1109/IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction",
        "url": "https://raw.githubusercontent.com/shantanu1109/IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction/main/DATASET/IBM-HR-Analytics-Employee-Attrition-and-Performance.csv",
        "category": "General Corporate",
        "description": "Baseline performance and satisfaction dataset for employee retention modeling."
    },
    {
        "id": "05_employee_attrition_large",
        "name": "Enterprise High-Volume Attrition Dataset (49k rows)",
        "source_repo": "raju5162/EmployeeAttrition",
        "url": "https://raw.githubusercontent.com/raju5162/EmployeeAttrition/main/new%20dataset%20for%20python.csv",
        "category": "Large-Scale Enterprise",
        "description": "Massive 49,624-record enterprise dataset for stress-testing large workforce churn and batch inference."
    },
    {
        "id": "06_douglas_attrition",
        "name": "Douglas HR Attrition Benchmark",
        "source_repo": "DouglasRFLeite/EmployeeAttritionPrediction",
        "url": "https://raw.githubusercontent.com/DouglasRFLeite/EmployeeAttritionPrediction/main/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "category": "Corporate HR",
        "description": "Cleaned workforce turnover dataset with standard demographic and engagement features."
    },
    {
        "id": "07_ahmed_hr_dataset",
        "name": "Ahmed Enterprise HR Attrition Predictor Dataset",
        "source_repo": "ahmed-alameldin/IBM-HR-Employee-Attrition-Predictor",
        "url": "https://raw.githubusercontent.com/ahmed-alameldin/IBM-HR-Employee-Attrition-Predictor/main/Data/HR%20Employee%20Attrition%20Dataset.csv",
        "category": "Talent Analytics",
        "description": "Standardized HR attrition dataset for feature engineering and predictive risk modeling."
    },
    {
        "id": "08_ibm_aif360_fairness",
        "name": "IBM AIF360 Algorithmic Fairness HR Dataset",
        "source_repo": "IBM/employee-attrition-aif360",
        "url": "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
        "category": "Ethical AI & Fairness",
        "description": "Curated by IBM Research for AI Fairness 360 (AIF360) to evaluate algorithmic bias across gender, age, and marital status."
    },
    {
        "id": "09_sarthak_knn_split",
        "name": "Sarthak KNN Partitioned HR Dataset",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/KNN/ibm.csv",
        "category": "Model Benchmarking",
        "description": "Partitioned baseline dataset for nearest-neighbor distance and density evaluation."
    },
    {
        "id": "10_sarthak_knn_smote",
        "name": "SMOTE Class-Balanced KNN Workforce Dataset",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/KNN/smote.csv",
        "category": "Class Imbalance & SMOTE",
        "description": "Synthetically oversampled (SMOTE) dataset with 2,466 records balancing the minority attrition class."
    },
    {
        "id": "11_sarthak_lr_split",
        "name": "Sarthak Logistic Regression Split Dataset",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/LR/ibm.csv",
        "category": "Linear Benchmarks",
        "description": "Normalized dataset for testing logistic regression odds ratios and linear decision boundaries."
    },
    {
        "id": "12_sarthak_nn_split",
        "name": "Sarthak Neural Network Workforce Partition",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/Neural_Networks/ibm.csv",
        "category": "Deep Learning",
        "description": "Clean split designed for neural network feature representation and MLP benchmarking."
    },
    {
        "id": "13_sarthak_nn_smote",
        "name": "SMOTE Neural Network Balanced Dataset",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/Neural_Networks/smote.csv",
        "category": "Deep Learning & SMOTE",
        "description": "Balanced 2,466-record dataset tailored for training deep tabular neural networks without majority bias."
    },
    {
        "id": "14_sarthak_smote_split",
        "name": "Sarthak SMOTE Evaluation Benchmark",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/SMOTE/ibm.csv",
        "category": "Imbalance Research",
        "description": "Pre-oversampling holdout partition for measuring recall on rare minority attrition instances."
    },
    {
        "id": "15_sarthak_smote_balanced",
        "name": "SMOTE Fully Balanced Corporate Dataset",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/SMOTE/smote.csv",
        "category": "Class Imbalance",
        "description": "Fully balanced synthetic workforce distribution with equal representation of active and departed employees."
    },
    {
        "id": "16_sarthak_tree_split",
        "name": "Tree-Based Algorithm HR Benchmark",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/Tree-Based/ibm.csv",
        "category": "Decision Trees & Ensembles",
        "description": "Dataset optimized for Gradient Boosting and Random Forest tree splitting algorithms."
    },
    {
        "id": "17_sarthak_canonical",
        "name": "Sarthak Canonical Attrition Dataset",
        "source_repo": "sarthakbabbar3/IBM_employee_attrition_prediction",
        "url": "https://raw.githubusercontent.com/sarthakbabbar3/IBM_employee_attrition_prediction/master/Codes/ibm.csv",
        "category": "Cross-Validation",
        "description": "Standardized canonical IBM HR dataset for cross-validation consistency."
    },
    {
        "id": "18_pavan_attrition_mirror",
        "name": "Pavan HR Analytics Attrition Mirror",
        "source_repo": "mragpavank/ibm-hr-analytics-attrition-dataset",
        "url": "https://raw.githubusercontent.com/mragpavank/ibm-hr-analytics-attrition-dataset/master/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "category": "Open Data Mirror",
        "description": "High-availability mirror of the IBM Watson HR Analytics dataset."
    },
    {
        "id": "19_anuj_raw_benchmark",
        "name": "Anuj Mundu Service Raw HR Benchmark",
        "source_repo": "anujmundu/Employee-Attrition-Prediction-Service",
        "url": "https://raw.githubusercontent.com/anujmundu/Employee-Attrition-Prediction-Service/main/data/raw.csv",
        "category": "Core Service Dataset",
        "description": "The foundational 12-feature core benchmark dataset from the original repository."
    },
    {
        "id": "20_anuj_text_augmented",
        "name": "Anuj Mundu NLP Multimodal Text-Augmented Dataset",
        "source_repo": "anujmundu/Employee-Attrition-Prediction-Service",
        "url": "https://raw.githubusercontent.com/anujmundu/Employee-Attrition-Prediction-Service/main/data/with_text.csv",
        "category": "Multimodal NLP & Feedback",
        "description": "Tabular HR dataset augmented with unstructured qualitative employee pulse feedback text."
    }
]


def download_all_online_datasets():
    """Downloads all 20 verified online datasets from GitHub and generates catalog.json."""
    ONLINE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []
    
    print(f"Downloading 20 verified real-world datasets from online sources into {ONLINE_DATASETS_DIR}...")
    for idx, item in enumerate(ONLINE_SOURCES, 1):
        dest = ONLINE_DATASETS_DIR / f"{item['id']}.csv"
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp, open(dest, "wb") as f:
                f.write(resp.read())
                
            df = pd.read_csv(dest, low_memory=False)
            catalog_entry = {
                **item,
                "local_file": f"{item['id']}.csv",
                "row_count": len(df),
                "column_count": len(df.columns),
                "file_size_kb": round(dest.stat().st_size / 1024, 1),
                "columns_sample": list(df.columns)[:8],
            }
            catalog.append(catalog_entry)
            print(f"  [{idx}/20] Downloaded {item['name']} ({len(df)} rows, {len(df.columns)} cols) -> {dest.name}")
        except Exception as e:
            print(f"  [{idx}/20] Failed to download {item['id']}: {e}")
            
    with open(ONLINE_CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)
        
    print(f"\nSuccessfully downloaded {len(catalog)}/20 online datasets!")
    print(f"Catalog saved to {ONLINE_CATALOG_PATH}")
    return catalog


if __name__ == "__main__":
    download_all_online_datasets()
