import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import get_preprocessor, TARGET_COL


DATA_PATH = "data/raw.csv"
MODEL_PATH = "models/model_v1.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor()

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
