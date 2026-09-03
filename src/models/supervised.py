import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.core.config import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    SUPERVISED_PIPELINE_PATH,
    SUPERVISED_ENSEMBLE_PATH,
    MODEL_BENCHMARK_PATH,
)


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Creates a robust preprocessor with imputation, scaling, and one-hot encoding."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    return preprocessor


class SupervisedModelSuite:
    """
    Manages training, benchmarking, and selection of multiple supervised classifiers.
    Includes HistGradientBoosting, Random Forest, Gradient Boosting, Logistic Regression,
    MLP, and a Soft-Voting Calibrated Ensemble.
    """

    def __init__(self):
        self.models = {
            "HistGradientBoosting": HistGradientBoostingClassifier(
                random_state=42, class_weight="balanced", max_iter=150
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=120, random_state=42, class_weight="balanced", max_depth=10
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42, learning_rate=0.08
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced", C=0.8
            ),
            "MLPClassifier": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=400, random_state=42, early_stopping=True
            ),
        }
        self.benchmark_results = {}
        self.best_model_name = None
        self.best_pipeline = None
        self.preprocessor = None

    def fit_and_evaluate(self, df: pd.DataFrame) -> dict:
        """Fits all models, computes evaluation metrics, and saves the best model pipeline."""
        available_num = [c for c in NUMERICAL_COLS if c in df.columns]
        available_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
        
        X = df[available_num + available_cat]
        y = df[TARGET_COL].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        self.preprocessor = build_preprocessor(available_num, available_cat)
        
        # Benchmark individual models
        best_roc_auc = -1.0
        pipelines = {}
        
        for name, clf in self.models.items():
            pipe = Pipeline([
                ("preprocessing", self.preprocessor),
                ("model", clf),
            ])
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            
            auc = round(float(roc_auc_score(y_test, y_prob)), 4)
            f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            acc = round(float(accuracy_score(y_test, y_pred)), 4)
            prec = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
            rec = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
            brier = round(float(brier_score_loss(y_test, y_prob)), 4)
            
            self.benchmark_results[name] = {
                "roc_auc": auc,
                "f1_score": f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "brier_score": brier,
            }
            pipelines[name] = pipe
            
            if auc > best_roc_auc:
                best_roc_auc = auc
                self.best_model_name = name
                self.best_pipeline = pipe
                
        # Build Calibrated Soft-Voting Ensemble from top 3 models
        ensemble_estimators = [
            ("hgb", self.models["HistGradientBoosting"]),
            ("rf", self.models["RandomForest"]),
            ("lr", self.models["LogisticRegression"]),
        ]
        voting_clf = VotingClassifier(estimators=ensemble_estimators, voting="soft")
        ensemble_pipe = Pipeline([
            ("preprocessing", self.preprocessor),
            ("model", CalibratedClassifierCV(voting_clf, method="isotonic", cv=3)),
        ])
        ensemble_pipe.fit(X_train, y_train)
        
        y_ens_prob = ensemble_pipe.predict_proba(X_test)[:, 1]
        y_ens_pred = (y_ens_prob >= 0.40).astype(int)
        
        ens_auc = round(float(roc_auc_score(y_test, y_ens_prob)), 4)
        ens_f1 = round(float(f1_score(y_test, y_ens_pred, zero_division=0)), 4)
        ens_acc = round(float(accuracy_score(y_test, y_ens_pred)), 4)
        ens_prec = round(float(precision_score(y_test, y_ens_pred, zero_division=0)), 4)
        ens_rec = round(float(recall_score(y_test, y_ens_pred, zero_division=0)), 4)
        ens_brier = round(float(brier_score_loss(y_test, y_ens_prob)), 4)
        
        self.benchmark_results["CalibratedEnsemble"] = {
            "roc_auc": ens_auc,
            "f1_score": ens_f1,
            "accuracy": ens_acc,
            "precision": ens_prec,
            "recall": ens_rec,
            "brier_score": ens_brier,
        }
        
        # If ensemble outperforms or is within 1% of best, prefer ensemble for robustness
        if ens_auc >= best_roc_auc - 0.01:
            self.best_model_name = "CalibratedEnsemble"
            self.best_pipeline = ensemble_pipe
            
        # Serialize artifacts
        joblib.dump(self.best_pipeline, SUPERVISED_PIPELINE_PATH)
        joblib.dump(ensemble_pipe, SUPERVISED_ENSEMBLE_PATH)
        with open(MODEL_BENCHMARK_PATH, "w") as f:
            json.dump(self.benchmark_results, f, indent=2)
            
        print(f"Model Suite Trained. Best Model: {self.best_model_name} (ROC-AUC: {self.benchmark_results[self.best_model_name]['roc_auc']})")
        return self.benchmark_results
