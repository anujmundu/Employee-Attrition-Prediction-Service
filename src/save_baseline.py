import joblib
import pandas as pd
from preprocessing import get_preprocessor

DATA_PATH = "data/raw.csv"
BASELINE_PATH = "models/baseline_stats.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Attrition"])

    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    baseline = {
        "mean": X_processed.mean(axis=0),
        "std": X_processed.std(axis=0),
    }

    joblib.dump(baseline, BASELINE_PATH)
    print("Baseline saved.")


if __name__ == "__main__":
    main()
