import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from preprocessing import get_preprocessor, TARGET_COL
from nlp_features import train_nlp

DATA_PATH = "data/with_text.csv"
MODEL_PATH = "models/model_v2_text.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    texts = df["EmployeeFeedback"].tolist()
    X_tabular = df.drop(columns=[TARGET_COL, "EmployeeFeedback"])
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # NLP features
    X_tfidf, embeddings = train_nlp(texts)

    # Tabular preprocessing
    preprocessor = get_preprocessor()
    X_tab = preprocessor.fit_transform(X_tabular)

    # Fuse features
    X_full = np.hstack([X_tab, X_tfidf.toarray(), embeddings])

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump((preprocessor, model), MODEL_PATH)
    print("Text enhanced model saved.")


if __name__ == "__main__":
    main()
