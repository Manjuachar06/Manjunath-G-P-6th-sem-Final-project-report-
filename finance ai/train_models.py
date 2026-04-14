"""
================================================================================
MODEL TRAINING SCRIPT - Train and evaluate ML models
================================================================================

PURPOSE:
  Trains Random Forest and Gradient Boosting classifiers on German Credit Dataset.
  Saves trained models and preprocessors (encoders, scalers) to app/models/
  Creates evaluation report with accuracy, precision, recall, F1-score metrics.

HOW TO RUN:
  python train_models.py
  
  This script will:
  1. Load german_credit_data.csv
  2. Preprocess features (label encode categorical, standard scale)
  3. Train Random Forest and Gradient Boosting models
  4. Evaluate on test set
  5. Save model files to app/models/ directory
  6. Generate model_evaluation_report.txt

OUTPUT:
  - app/models/random_forest_model.pkl
  - app/models/gradient_boosting_model.pkl
  - app/models/label_encoders.pkl
  - app/models/scaler.pkl
  - model_evaluation_report.txt (metrics summary)

WHEN TO RUN THIS:
  ✓ First time setup (before running app.py)
  ✓ After installing scikit-learn (prevents model load errors)
  ✓ When retraining with new data
  ✓ When you get "ModuleNotFoundError" in app.py

EXPECTED OUTPUT:
  Loading dataset from app/data/german_credit_data.csv
  Dataset contains 2000 rows
  Training models...
  Saving models and preprocessors...
  Evaluating models...
  Random Forest metrics: {...}
  Gradient Boosting metrics: {...}

MODELS TRAINED:
  1. Random Forest: 200 estimators, random_state=42
  2. Gradient Boosting: 200 estimators, random_state=42
  
TARGET VARIABLE:
  Risk column: 'good' (0) = approved, 'bad' (1) = rejected

DATA COLUMNS USED:
  Age, Sex, Job, Housing, Saving_accounts, Checking_account, 
  Credit_amount, Duration, Purpose

================================================================================
"""
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "app", "data", "german_credit_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "app", "models")
REPORT_PATH = os.path.join(PROJECT_ROOT, "model_evaluation_report.txt")

CATEGORICAL_COLUMNS = [
    "Sex",
    "Housing",
    "Saving_accounts",
    "Checking_account",
    "Purpose",
]

EXPECTED_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving_accounts",
    "Checking_account",
    "Credit_amount",
    "Duration",
    "Purpose",
]


def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(" ", "_")

    # Drop the index column if it exists
    if "Unnamed:_0" in df.columns:
        df = df.drop(columns=["Unnamed:_0"])

    if "Risk" not in df.columns:
        raise ValueError("Expected target column 'Risk' in dataset")

    # Convert Risk column: 'good' -> 0, 'bad' -> 1 (only if needed)
    if df["Risk"].dtype == 'object':
        df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})
    else:
        # Ensure it's integer type
        df["Risk"] = df["Risk"].astype(int)

    # If this is the original Kaggle file with 1000 rows, duplicate it to 2000 rows.
    if len(df) == 1000:
        df = pd.concat([df, df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"Expanded dataset to {len(df)} rows and saved back to {path}")

    return df


def prepare_features(df):
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "_")
    missing_cols = [col for col in EXPECTED_COLUMNS + ["Risk"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    X = df.drop(columns=["Risk"])
    y = df["Risk"].astype(int)

    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        label_encoders[col] = encoder

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.to_numpy(), label_encoders, scaler


def train_and_save_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    return rf, gb


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def write_report(rf_metrics, gb_metrics):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("RANDOM FOREST\n")
        for name, value in rf_metrics.items():
            f.write(f"• {name.replace('_', ' ').title()}: {value:.4f}\n")
        f.write("\n")
        f.write("GRADIENT BOOSTING\n")
        for name, value in gb_metrics.items():
            f.write(f"• {name.replace('_', ' ').title()}: {value:.4f}\n")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading dataset from {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    print(f"Dataset contains {len(df)} rows")

    X, y, label_encoders, scaler = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")
    rf_model, gb_model = train_and_save_models(X_train, y_train)

    print("Saving models and preprocessors...")
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(gb_model, os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Evaluating models...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    gb_metrics = evaluate_model(gb_model, X_test, y_test)
    write_report(rf_metrics, gb_metrics)

    print("Training complete.")
    print(f"Random Forest metrics: {rf_metrics}")
    print(f"Gradient Boosting metrics: {gb_metrics}")
