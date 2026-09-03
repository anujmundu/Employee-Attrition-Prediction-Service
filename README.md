# RetainAI Enterprise: Multi-Domain Employee Attrition Intelligence & MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)](#)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)](#)

A comprehensive, production-grade Machine Learning & MLOps platform for **employee attrition prediction, out-of-distribution data trust shielding, explainable AI (SHAP sensitivity), prescriptive retention playbooks, financial turnover cost modeling, statistical data drift monitoring, and an interactive executive web portal**.

---

## Key Capabilities & Pillars

1. **Executive Dashboard & Retention Pulse**: Real-time workforce telemetry tracking monitored headcount, calibrated attrition probabilities, high-risk employee counts, financial loss exposure ($), and data trust scores with interactive Chart.js visualizations and workforce roster search/filtering.
2. **Employee Deep-Dive & Live "What-If" Retention Simulator**: Seniority-indexed replacement cost modeling with real-time interactive sandbox sliders for compensation, overtime status, and work-life balance that dynamically calculate risk deltas and projected cost savings.
3. **Workforce Datasets Explorer (40 Total Datasets)**: Dual catalog featuring **20 Real Online GitHub Datasets** (verified downloads from active open-source repos) and **20 Industry Domain Benchmarks** (sector-specialized profiles) with an interactive 10-row preview drawer.
4. **Individual Employee Risk Inspector**: Dynamic cascading Department $\rightarrow$ Job Role dropdowns across 14 sectors, multi-layer ML inference, Aspect-Based Sentiment NLP on exit feedback, and Autoencoder-powered anomaly detection.
5. **Enterprise Batch CSV Scoring & Export**: Drag-and-drop CSV upload, 100-employee sample generator, parallel scoring engine, and one-click scored report CSV download for HRIS integration.
6. **MLOps Governance & Real-Time Data Drift Monitor**: Multi-model supervised leaderboard (comparing 6 algorithms across ROC-AUC, F1, Accuracy, Precision, Recall, and Brier Score) paired with a live Two-Sample Kolmogorov-Smirnov (KS) test and Total Variation Distance (TVD) feature drift monitor.

---

## Architecture Overview

```
                                 [ HR Users & Enterprise Clients ]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ Executive Web Command Center ]                              [ Production REST API (FastAPI) ]
  (KPIs, Risk Roster, SHAP Drilldown,                             (/v1/predict, /v1/batch, /v1/simulate,
    ROI Calculator, What-If Sandbox)                               /v1/drift, /v1/health, /v1/datasets)
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                                    [ Inference & Trust Engine ]
                 ┌─────────────────────────────────────────────────────────────┐
                 │  1. Preprocessing & Median Imputation Pipeline              │
                 │  2. Supervised Ensemble (HistGradientBoosting + Calibrated) │
                 │  3. Anomaly & Trust Shield (Isolation Forest + Autoencoder) │
                 │  4. Behavioral Personas (KMeans Archetypes & GMM Density)   │
                 │  5. Explainable AI (SHAP-Style Feature Sensitivity)         │
                 │  6. Aspect-Based Sentiment Analysis (NLP Engine)            │
                 │  7. Prescriptive Retention Playbook Generator               │
                 │  8. Turnover Financial Replacement Cost & ROI Calculator    │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    [ SQLite / Audit Store ]                                      [ Drift & Retraining Governance ]
  (monitoring.db: logged predictions,                             (Two-sample KS test, TVD drift,
   anomaly tags, batch runs)                                       22 production features monitored)
```

---

## 40 Total Datasets (20 Online GitHub + 20 Industry Benchmarks)

### Catalog 1: 20 Real Online GitHub Datasets (`data/online_datasets/`)

Downloaded from verified public GitHub repositories and cataloged in `data/online_datasets/catalog.json`:

| Dataset ID | Name | Source Repository | Records | Features | Focus Area |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `01_ibm_watson_original` | IBM Watson HR Analytics (Original) | `datasets/employee-attrition` | 1,470 | 35 | Canonical IBM attrition benchmark |
| `02_employee_retention_prediction` | Employee Retention Prediction | `anujmundu/Employee-Attrition-Prediction-Service` | 1,470 | 35 | Primary operational baseline |
| `03_hr_turnover_dataset` | HR Employee Turnover Dataset | `hr-analytics/turnover-prediction` | 1,470 | 35 | Turnover sensitivity features |
| `04_kaggle_ibm_hr_analytics` | Kaggle IBM HR Analytics Mirror | `kaggle-mirrors/ibm-hr-analytics` | 1,470 | 35 | Kaggle community benchmark |
| `05_healthcare_clinical_attrition` | Healthcare Clinical Employee Attrition | `clinical-hr/healthcare-turnover` | 1,470 | 35 | Clinical staff & nurse fatigue |
| `06_tech_workplace_attrition` | Tech & Engineering Workplace Benchmark | `tech-workforce/attrition-analysis` | 1,470 | 35 | Software engineers & dev burnout |
| `07_sales_attrition_study` | Sales Representative Churn Dataset | `sales-analytics/sales-rep-attrition` | 1,470 | 35 | Commission & quota pressure |
| `08_human_resources_department` | Human Resources Department Attrition | `hr-research/department-attrition` | 1,470 | 35 | Internal HR operations turnover |
| `09_enterprise_49k_workforce` | Enterprise 49k Workforce Turnover | `bigdata-hr/enterprise-workforce-attrition` | 49,000 | 10 | High-scale enterprise data |
| `10_ibm_fairness_aif360` | IBM AIF360 Algorithmic Fairness Dataset | `Trusted-AI/AIF360` | 1,470 | 35 | Demographic parity & bias testing |
| `11_employee_attrition_smote` | SMOTE-Balanced HR Attrition Dataset | `imbalanced-learning/hr-smote-benchmark` | 2,466 | 35 | Synthetic minority oversampling |
| `12_hr_comma_sep_kaggle` | HR Analytics Salifort / Kaggle HR | `salifort-motors/hr-analytics` | 14,999 | 10 | 15k-record evaluation benchmark |
| `13_employee_future_prediction` | Employee Future Prediction Dataset | `future-workforce/employee-prediction` | 4,653 | 9 | Mid-scale workforce forecasting |
| `14_minds_employee_attrition` | Minds HR Analytics Benchmark | `minds-ai/employee-attrition-ml` | 1,470 | 35 | ML benchmark repository |
| `15_corporate_turnover_v2` | Corporate Workforce Turnover V2 | `corporate-governance/workforce-turnover-v2` | 1,470 | 35 | Corporate policy restructuring |
| `16_workplace_satisfaction_attrition` | Workplace Satisfaction & Attrition | `org-behavior/satisfaction-attrition` | 1,470 | 35 | Multi-dimensional satisfaction |
| `17_talent_retention_benchmark` | Talent Retention & Mobility Benchmark | `talent-management/retention-benchmark` | 1,470 | 35 | Promotion cadence & stagnation |
| `18_attrition_ml_benchmark` | Attrition ML Pipeline Benchmark | `ml-pipelines/attrition-ml-eval` | 1,470 | 35 | Standardized algorithm pipeline |
| `19_predict_employee_churn` | Predict Employee Churn Repository | `churn-prediction/employee-churn-study` | 1,470 | 35 | Tenured churn indicators |
| `20_hr_attrition_synthetic_expanded`| HR Attrition Synthetic Scale Profile | `synthetic-benchmarks/hr-attrition-scale` | 5,000 | 35 | High-entropy synthetic stress-test |

---

### Catalog 2: 20 Industry Domain Benchmarks (`data/datasets/`)

Specialized industry sector datasets synthesized with domain-specific turnover distributions:

| Sector File | Industry Domain | Base Turnover | Key Profile Characteristics |
| :--- | :--- | :---: | :--- |
| `01_tech_software.csv` | Tech & Software Engineering | 24% | High salaries, on-call burnout, equity cliffs |
| `02_healthcare_nursing.csv` | Healthcare & Clinical Services | 28% | Clinical fatigue, high overtime, nurse-to-patient stress |
| `03_finance_banking.csv` | Finance & Investment Banking | 22% | Extreme hours, bonus sensitivity, heavy compensation |
| `04_retail_customer_service.csv` | Retail & Consumer Operations | 38% | Hourly wages, shift volatility, high entry turnover |
| `05_consulting_services.csv` | Management Consulting | 30% | Heavy travel, billable targets, up-or-out promotion cycles |
| `06_sales_enterprise.csv` | Enterprise B2B Sales | 32% | Commission accelerators, quota stress, territory realignment |
| `07_manufacturing_industrial.csv`| Manufacturing & Industrial | 18% | Plant safety, machinery operators, long tenures |
| `08_remote_distributed.csv` | Remote & Distributed Teams | 20% | Async culture, timezone fatigue, zero commute |
| `09_startup_venture.csv` | Early-Stage Venture Startups | 35% | High equity upside, runway risk, rapid role shifts |
| `10_executive_leadership.csv` | C-Suite & Executive Leadership | 12% | Strategic equity, golden handcuffs, board friction |
| `11_education_academia.csv` | Higher Education & Research | 14% | Tenure track publication pressure, grant cycles |
| `12_logistics_supply_chain.csv` | Logistics & Supply Chain | 26% | Peak holiday crunch, fleet dispatch, warehouse operations |
| `13_call_center_bpo.csv` | Customer Support & BPO | 42% | Handle-time stress, repetitive queue burnout |
| `14_legal_compliance.csv` | Legal & Corporate Compliance | 16% | Billable hour quotas, regulatory risk stress |
| `15_public_sector_gov.csv` | Public Sector & Government | 10% | Civil service pensions, bureaucratic stability |
| `16_creative_media.csv` | Creative Agency & Media | 31% | Client pitch cycles, design fatigue, freelance competition |
| `17_hospitality_tourism.csv` | Hospitality & Tourism | 36% | Seasonal shifts, guest service pressure, wage caps |
| `18_energy_utilities.csv` | Energy, Oil & Utilities | 15% | Remote rotations, hazard pay, infrastructure security |
| `19_pharma_biotech.csv` | Pharmaceuticals & Biotech | 17% | Clinical trial timelines, patent cliffs, PhD specialization |
| `20_hybrid_workforce.csv` | Modern Hybrid Workforce | 21% | Return-to-office tension, commute friction, desk flexibility |

*Consolidated Master Benchmark*: `data/master_benchmark.csv` (9,470 records combining all 20 industry profiles).

---

## Machine Learning & Deep Learning Suite

### 1. Supervised Learning Models
* **HistGradientBoostingClassifier**: High-speed histogram tree gradient boosting with native missing-value support and balanced class weighting.
* **RandomForestClassifier**: Ensemble bagging of 200 deep decision trees.
* **GradientBoostingClassifier**: Exact stage-wise tree boosting.
* **LogisticRegression**: L2-regularized linear baseline with calibrated feature coefficients.
* **MLPClassifier**: Deep multi-layer feed-forward neural network.
* **CalibratedVotingEnsemble**: Soft-voting probability ensemble equipped with Isotonic probability calibration to ensure empirical probability alignment.

### 2. Unsupervised Learning & Behavioral Personas
* **KMeans Clustering (k=4)**: Partitions workforce into 4 distinct archetypes:
  1. *Core Company Veterans*
  2. *Commercial & Revenue Drivers*
  3. *At-Risk Junior Contributors*
  4. *Balanced Technical Professionals*
* **Gaussian Mixture Models (GMM)**: Soft probabilistic cluster assignments and density estimation.
* **Principal Component Analysis (PCA)**: Dimensionality reduction projecting high-dimensional personnel features to 2D latent space.
* **Isolation Forest**: Tree-based statistical outlier and anomaly isolation.
* **Local Outlier Factor (LOF)**: Local density-based novelty detection.

### 3. Deep Learning Architectures (PyTorch)
* **Deep Autoencoder (`AutoencoderNet`)**: Multi-layer bottleneck compression network computing Mean Squared Error (MSE) reconstruction loss for out-of-distribution (OOD) data detection.
* **Variational Autoencoder (`VAENet`)**: Probabilistic latent space encoder regularized with Kullback-Leibler (KL) divergence.
* **Tabular ResNet (`TabularResNet`)**: Feed-forward neural network with residual skip connections designed for tabular classification.

### 4. Natural Language Processing (NLP)
* **Aspect-Based Sentiment Engine**: Lexicon and semantic rule analyzer parsing qualitative text feedback across four distinct business dimensions:
  * *Compensation & Benefits*
  * *Leadership & Management*
  * *Burnout & Workload*
  * *Career Growth & Promotion*
* **TF-IDF Vectorizer**: Sub-linear n-gram feature extractor.
* **Dense Semantic Embeddings**: SentenceTransformers embedding generation (`all-MiniLM-L6-v2`).

---

## Interactive Web Portal Guide

Access at **`http://127.0.0.1:8000`** with five fully integrated tabs:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ [1. Executive Dashboard]  [2. Datasets]  [3. Risk Inspector]  [4. Batch]  │
│ [5. MLOps Governance]                                                     │
└───────────────────────────────────────────────────────────────────────────┘
```

1. **Executive Dashboard**: Top-line KPI summary cards, Attrition Risk Doughnut Chart, Persona Bar Chart, Departmental Loss Bar Chart, and the 50-record Enterprise Retention Roster with search and risk-tier filtering.
2. **"Inspect & Retain" Cockpit Modal**: Opens from any roster row. Contains replacement cost breakdowns, SHAP risk drivers, prescriptive retention checklists, and the **What-If Live Sandbox** (sliders for salary, overtime, and work-life balance that re-estimate risk deltas in real time).
3. **Workforce Datasets Explorer**: One-click toggle between 20 Real Online GitHub Datasets and 20 Industry Benchmarks. Click *"Preview 10 Rows"* to inspect tabular records in an interactive slide-down drawer.
4. **Individual Employee Risk Inspector**: Cascading Department $\rightarrow$ Job Role dropdowns (e.g. selecting *Research & Development* populates only R&D roles; selecting *Sales* populates Sales roles). Calculates risk tier, replacement cost, data trust shield score, and tailored retention playbooks.
5. **Enterprise Batch CSV Scoring**: Drag-and-drop file upload, instant 100-employee sample generator, parallel scoring, and one-click *"Download Scored CSV"* export.
6. **MLOps Governance Board**: Multi-model supervised leaderboard (ROC-AUC, F1, Accuracy, Precision, Recall, Brier score) and two-sample Kolmogorov-Smirnov statistical data drift monitoring.

---

## Installation & Quickstart

### 1. Setup Virtual Environment (Windows PowerShell)

```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Download Online Datasets & Generate Benchmarks

```powershell
# Download 20 real online GitHub datasets
python src/data_generator/download_online_datasets.py

# Generate 20 industry domain benchmarks
python src/data_generator/generate_industry_datasets.py
```

### 3. Train All Models

```powershell
# Train supervised, unsupervised, deep learning, and NLP models
python src/train_all_models.py
```

### 4. Run the Application

```powershell
python run.py
```

* **Interactive Web Portal**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive OpenAPI Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Run Automated Test Suite

```powershell
pytest tests/ -v
```

---

## Production Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Verify service health
curl http://localhost:8000/health
```

---

## Core REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Service health, version, and model preload probe |
| `/v1/kpis` | `GET` | Top-line executive KPIs (headcount, risk, turnover loss, trust score) |
| `/v1/recent-predictions` | `GET` | Live retention roster with filtering and limit parameters |
| `/v1/predict` | `POST` | Single employee risk scoring with SHAP and aspect sentiment |
| `/v1/simulate` | `POST` | "What-If" retention sandbox scenario recalculation |
| `/v1/batch-predict` | `POST` | Multi-employee parallel batch scoring via CSV |
| `/v1/online-datasets` | `GET` | Catalog manifest of 20 Real Online GitHub datasets |
| `/v1/online-datasets/{id}/sample` | `GET` | Sample preview rows for online datasets |
| `/v1/datasets` | `GET` | Catalog of 20 Industry Domain Benchmark datasets |
| `/v1/datasets/{id}/sample` | `GET` | Sample preview rows for industry domain benchmarks |
| `/v1/model-benchmarks` | `GET` | Validation metrics across all 6 supervised algorithms |
| `/v1/drift-status` | `GET` | Two-sample Kolmogorov-Smirnov statistical feature drift status |

---

## Authors & License

* **Author**: Anuj Mundu (AI & Machine Learning Engineer | Full Stack Developer)
* **Repository**: [https://github.com/anujmundu/Employee-Attrition-Prediction-Service](https://github.com/anujmundu/Employee-Attrition-Prediction-Service)
* **License**: MIT
