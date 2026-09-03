"""
CLI diagnostic and batch scoring utility for Employee Attrition Prediction Service.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import get_engine
from src.mlops.drift_detector import DriftDetector
from src.mlops.monitoring_db import MonitoringDB


def run_single(args):
    """Run attrition prediction on single employee inputs."""
    engine = get_engine()
    data = {
        "Age": args.age,
        "Department": args.department,
        "JobRole": args.job_role,
        "MonthlyIncome": args.monthly_income,
        "OverTime": args.overtime,
        "DistanceFromHome": args.distance,
        "JobSatisfaction": args.job_satisfaction,
        "WorkLifeBalance": args.work_life_balance,
        "YearsAtCompany": args.years_at_company,
        "YearsSinceLastPromotion": args.years_since_promotion,
        "ExitSurveyNotes": args.notes or ""
    }
    
    result = engine.predict_single(data)
    print("\n" + "=" * 60)
    print(" EMPLOYEE ATTRITION DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Attrition Probability: {result['attrition_probability'] * 100:.2f}%")
    print(f"Risk Tier:             {result['risk_tier']}")
    print(f"Behavioral Persona:    {result['persona_name']}")
    print(f"Data Trust Score:      {result['data_trust_score']:.1f} / 100")
    print(f"Replacement Cost:      ${result['financials']['replacement_cost']:,.2f}")
    
    print("\n--- Key Risk Drivers (Top SHAP Impacts) ---")
    for impact in result.get("top_risk_factors", [])[:3]:
        print(f" • {impact['feature']}: {impact['impact_direction']} ({impact['value']})")
        
    print("\n--- Prescriptive Retention Action ---")
    for rec in result.get("retention_playbook", [])[:2]:
        print(f" [{rec['priority']}] {rec['title']} (Expected ROI: {rec['estimated_roi']})")
        print(f"     -> {rec['action']}")
    print("=" * 60 + "\n")


def run_batch(args):
    """Run batch prediction on input CSV file."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    engine = get_engine()
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records from {input_path}...")
    
    results = []
    for _, row in df.iterrows():
        res = engine.predict_single(row.to_dict())
        results.append({
            "Attrition_Probability": res["attrition_probability"],
            "Risk_Tier": res["risk_tier"],
            "Persona": res["persona_name"],
            "Data_Trust_Score": res["data_trust_score"],
            "Replacement_Cost": res["financials"]["replacement_cost"]
        })
        
    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    output_path = args.output or f"batch_results_{input_path.name}"
    out_df.to_csv(output_path, index=False)
    print(f"Scoring complete! Results saved to '{output_path}'.")


def run_status(args):
    """Check MLOps telemetry and drift metrics."""
    db = MonitoringDB()
    stats = db.get_statistics()
    print("\n" + "=" * 60)
    print(" MLOPS MONITORING & DRIFT STATUS")
    print("=" * 60)
    print(f"Total Prediction Logs: {stats.get('total_predictions', 0)}")
    print(f"Average Predicted Risk: {stats.get('avg_risk_probability', 0) * 100:.2f}%")
    print(f"Average Trust Score:    {stats.get('avg_trust_score', 0):.2f}")
    
    detector = DriftDetector()
    drift_status = detector.check_drift()
    print(f"Feature Drift Alert:    {'YES - DRIFT DETECTED' if drift_status.get('drift_detected') else 'NO - WITHIN TOLERANCE'}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Utility for Enterprise Employee Attrition Prediction Service"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Single scoring
    single_parser = subparsers.add_parser("score-single", help="Predict attrition for a single employee")
    single_parser.add_argument("--age", type=int, default=35, help="Employee age")
    single_parser.add_argument("--department", type=str, default="Research & Development")
    single_parser.add_argument("--job-role", type=str, default="Software Engineer")
    single_parser.add_argument("--monthly-income", type=float, default=6500)
    single_parser.add_argument("--overtime", type=str, choices=["Yes", "No"], default="No")
    single_parser.add_argument("--distance", type=int, default=10, help="Distance from home (km/miles)")
    single_parser.add_argument("--job-satisfaction", type=int, choices=[1, 2, 3, 4], default=3)
    single_parser.add_argument("--work-life-balance", type=int, choices=[1, 2, 3, 4], default=3)
    single_parser.add_argument("--years-at-company", type=int, default=4)
    single_parser.add_argument("--years-since-promotion", type=int, default=1)
    single_parser.add_argument("--notes", type=str, default="", help="Exit/pulse interview text")
    single_parser.set_defaults(func=run_single)

    # Batch scoring
    batch_parser = subparsers.add_parser("score-batch", help="Batch score a CSV dataset")
    batch_parser.add_argument("-i", "--input-file", required=True, help="Input CSV path")
    batch_parser.add_argument("-o", "--output", help="Output CSV path")
    batch_parser.set_defaults(func=run_batch)

    # Status & Drift
    status_parser = subparsers.add_parser("status", help="Inspect MLOps drift and monitoring audit logs")
    status_parser.set_defaults(func=run_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
