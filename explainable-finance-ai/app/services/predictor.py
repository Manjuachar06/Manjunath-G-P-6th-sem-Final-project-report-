# app/services/predictor.py

import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# models live under app/models (relative to this file: app/services/predictor.py)
BASE_APP_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_APP_DIR, "models")

# safe load with helpful errors
def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

# Load the trained models and preprocessing objects
rf_model = _load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
gb_model = _load(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"))

try:
    label_encoders = _load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
except Exception:
    label_encoders = {}

try:
    scaler = _load(os.path.join(MODEL_DIR, "scaler.pkl"))
except Exception:
    scaler = None


def _validate_credit_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dict")

    required_fields = {
        "Age": (int, float),
        "Job": (int,),
        "Credit amount": (int, float),
        "Duration": (int, float),
        "Sex": (str,),
        "Housing": (str,),
        "Saving accounts": (str,),
        "Checking account": (str,),
        "Purpose": (str,),
    }

    for key, expected_types in required_fields.items():
        if key not in payload:
            raise ValueError(f"Missing required feature: {key}")

        value = payload[key]
        if value is None:
            raise ValueError(f"Feature '{key}' cannot be None")

        if not isinstance(value, expected_types) or isinstance(value, bool):
            expected_names = ", ".join([t.__name__ for t in expected_types])
            raise TypeError(f"Feature '{key}' must be {expected_names}, got {type(value).__name__}")

    return True


def _prepare_input(X):
    """Prepare input with label encoding and scaling"""
    if isinstance(X, dict):
        _validate_credit_payload(X)
        # Normalize column names to use underscores (matching training data)
        X = {k.replace(' ', '_'): v for k, v in X.items()}
        df = pd.DataFrame([X])
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
        df.columns = df.columns.str.replace(' ', '_')
    else:
        df = pd.DataFrame([X])
        df.columns = df.columns.str.replace(' ', '_')

    # Ensure correct column order as used during training (from CSV)
    expected_columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving_accounts', 'Checking_account', 'Credit_amount', 'Duration', 'Purpose']
    df = df[expected_columns]

    # Encode categorical variables using the saved label encoders
    categorical_columns = ['Sex', 'Housing', 'Saving_accounts', 'Checking_account', 'Purpose']
    
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except Exception as e:
                raise ValueError(f"Error encoding {col}: {str(e)}")

    # Scale features using the saved scaler
    if scaler:
        df_scaled = scaler.transform(df)
        # Return numpy array without column names to match training format
        return df_scaled

    # Return numpy array without column names
    return df.values


def predict_credit(X):
    df = _prepare_input(X)
    pred = rf_model.predict(df)
    return int(pred[0]) if hasattr(pred, "__len__") else int(pred)


def predict_both_models(X):
    """Return predictions from both Random Forest and Gradient Boosting models"""
    df = _prepare_input(X)
    
    # Random Forest predictions
    rf_proba = rf_model.predict_proba(df)[0][1]
    
    # Gradient Boosting predictions
    gb_proba = gb_model.predict_proba(df)[0][1]
    
    # Manual risk score for hybrid approach
    manual_proba = _manual_credit_risk_score(X if isinstance(X, dict) else X.iloc[0].to_dict())
    
    # Combine model scores: weighted average of both models with manual rules
    rf_combined = (0.6 * rf_proba) + (0.4 * manual_proba)
    gb_combined = (0.6 * gb_proba) + (0.4 * manual_proba)
    
    return {
        "random_forest": {
            "probability": float(min(1.0, max(0.0, rf_combined))),
            "percentage": f"{min(1.0, max(0.0, rf_combined)):.2%}",
            "label": "High Risk" if rf_combined >= 0.5 else "Low Risk"
        },
        "gradient_boosting": {
            "probability": float(min(1.0, max(0.0, gb_combined))),
            "percentage": f"{min(1.0, max(0.0, gb_combined)):.2%}",
            "label": "High Risk" if gb_combined >= 0.5 else "Low Risk"
        }
    }


def _manual_credit_risk_score(payload):
    # Normalize and combine feature-based risk contributions (0..1)
    age = max(18, min(100, payload.get("Age", 30)))
    job = payload.get("Job", 1)
    amount = max(0, float(payload.get("Credit amount", 10000)))
    duration = max(1, min(120, float(payload.get("Duration", 12))))
    sex = payload.get("Sex", "male").lower()
    housing = payload.get("Housing", "own").lower()
    saving_accounts = payload.get("Saving accounts", "little").lower()
    checking_account = payload.get("Checking account", "moderate").lower()
    purpose = payload.get("Purpose", "education").lower()

    age_risk = 1.0 if age < 25 or age > 60 else 0.2 + 0.8 * abs(age - 30) / 30
    age_risk = min(1.0, max(0.0, age_risk))

    job_risk_map = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.3, 4: 0.6}
    job_risk = job_risk_map.get(job, 0.7)

    amount_risk = min(1.0, amount / 50000)
    duration_risk = min(1.0, duration / 72)

    sex_risk = 0.5

    housing_risk_map = {"own": 0.2, "rent": 0.6, "free": 0.4}
    housing_risk = housing_risk_map.get(housing, 0.5)

    saving_risk_map = {"unknown": 0.8, "little": 0.7, "moderate": 0.4, "quite rich": 0.2, "rich": 0.1}
    saving_risk = saving_risk_map.get(saving_accounts, 0.6)

    checking_risk_map = {"unknown": 0.8, "little": 0.7, "moderate": 0.4, "rich": 0.2}
    checking_risk = checking_risk_map.get(checking_account, 0.6)

    purpose_risk_map = {
        "business": 0.6,
        "car": 0.5,
        "domestic appliances": 0.4,
        "education": 0.3,
        "furniture/equipment": 0.5,
        "radio/tv": 0.4,
        "repairs": 0.5,
        "vacation/others": 0.7,
    }
    purpose_risk = purpose_risk_map.get(purpose, 0.5)

    manual_score = (
        0.18 * age_risk
        + 0.12 * job_risk
        + 0.18 * amount_risk
        + 0.18 * duration_risk
        + 0.08 * housing_risk
        + 0.08 * saving_risk
        + 0.08 * checking_risk
        + 0.1 * purpose_risk
    )
    return min(1.0, max(0.0, manual_score))


def predict_credit_proba(X):
    """Return probability of class 1 (high risk) from Random Forest model"""
    df = _prepare_input(X)
    proba = rf_model.predict_proba(df)
    model_proba = float(proba[0][1])

    # Combine model score and manual rules to ensure all fields impact result
    manual_proba = _manual_credit_risk_score(X if isinstance(X, dict) else X.iloc[0].to_dict())
    combined = (0.6 * model_proba) + (0.4 * manual_proba)
    return float(min(1.0, max(0.0, combined)))
