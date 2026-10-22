
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "UCI_Credit_Card.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "BILL_AMT1",
    "PAY_AMT1",
]

CATEGORICAL_FEATURES = ["EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = [
    feature for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]
TARGET = "default.payment.next.month"


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Pastikan file CSV berada di folder data/."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Kolom target '{TARGET}' tidak ditemukan. "
            "Periksa nama kolom pada dataset."
        )

    # Bersihkan data dengan aturan yang digunakan notebook.
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    # Kelompokkan kode kategori tidak umum ke kategori "lainnya".
    df["EDUCATION"] = df["EDUCATION"].where(
        df["EDUCATION"].isin([1, 2, 3]), 4
    )
    df["MARRIAGE"] = df["MARRIAGE"].where(
        df["MARRIAGE"].isin([1, 2]), 3
    )

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    baseline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    categorical_indices = [
        FEATURES.index(feature)
        for feature in CATEGORICAL_FEATURES
    ]

    smote_model = ImbPipeline([
        (
            "smote",
            SMOTENC(
                categorical_features=categorical_indices,
                random_state=42,
            ),
        ),
        ("preprocess", make_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    models = {
        "Logistic Regression": baseline,
        "Logistic Regression + SMOTENC": smote_model,
    }

    results = []
    fitted_models = {}

    for name, candidate in models.items():
        print(f"\nMelatih: {name}")
        candidate.fit(X_train, y_train)

        pred = candidate.predict(X_test)
        proba = candidate.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, proba),
        }

        results.append(metrics)
        fitted_models[name] = candidate

        print(pd.Series(metrics).to_string())
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred))

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        by=["F1-Score", "Recall", "ROC-AUC"],
        ascending=False,
    ).reset_index(drop=True)

    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        MODEL_DIR / "credit_risk_model.joblib",
        compress=3,
    )

    comparison.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False,
    )

    metadata = {
        "model_name": best_name,
        "dataset": DATA_PATH.name,
        "rows_after_cleaning": int(len(df)),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "positive_class": 1,
        "threshold": 0.5,
        "random_state": 42,
        "test_size": 0.2,
        "selection_criteria": [
            "F1-Score",
            "Recall",
            "ROC-AUC",
        ],
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\n=== PERBANDINGAN MODEL ===")
    print(comparison.to_string(index=False))
    print(f"\nModel terpilih: {best_name}")
    print(f"Model tersimpan di: {MODEL_DIR}")


if __name__ == "__main__":
    main()
