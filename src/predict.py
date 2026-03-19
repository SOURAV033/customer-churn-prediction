"""
predict.py
==========
Load the trained best model and predict churn for new customers.
Can be used as a CLI tool or imported as a module.

Author : Sourav Nayak
GitHub : github.com/SOURAV033

Usage:
    python predict.py
"""

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder


def load_model(path="models/best_model.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[INFO] Loaded model: {bundle['model_name']}")
    return bundle


def predict_single(customer_dict: dict, bundle: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_dict : dict  — raw customer feature values
    bundle        : dict  — loaded model bundle from best_model.pkl

    Returns
    -------
    dict with 'churn_prediction', 'churn_probability', 'risk_level'
    """
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features"]

    df = pd.DataFrame([customer_dict])

    # Engineer same features as training
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1,12,24,48,72],
        labels=["New (0-12m)","Growing (1-2yr)","Established (2-4yr)","Loyal (4yr+)"])
    df["avg_monthly_spend"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["has_bundle"] = (
        (df["PhoneService"]=="Yes") &
        (df["InternetService"].isin(["DSL","Fiber optic"]))
    ).astype(int)
    df["high_value"]    = (df["MonthlyCharges"] > 75).astype(int)
    contract_map        = {"Month-to-month":2,"One year":1,"Two year":0}
    df["contract_risk"] = df["Contract"].map(contract_map)

    # Encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object","category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Align columns to training feature order
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Scale numeric
    num_cols = df.select_dtypes(include="number").columns.tolist()
    try:
        df[num_cols] = scaler.transform(df[num_cols])
    except Exception:
        pass

    prob  = model.predict_proba(df)[0][1]
    pred  = int(prob >= 0.5)
    risk  = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.40 else "LOW"

    return {
        "churn_prediction": "Will Churn" if pred else "Will Stay",
        "churn_probability": round(prob * 100, 2),
        "risk_level":        risk,
    }


def demo_predictions(bundle):
    """Run demo predictions on 3 customer profiles."""
    customers = [
        {
            "name": "Profile A — High Risk (New, Fiber, Month-to-month)",
            "data": {
                "gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No",
                "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
                "InternetService":"Fiber optic","OnlineSecurity":"No",
                "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
                "StreamingTV":"Yes","StreamingMovies":"Yes",
                "Contract":"Month-to-month","PaperlessBilling":"Yes",
                "PaymentMethod":"Electronic check",
                "MonthlyCharges":95.50,"TotalCharges":191.00,
            }
        },
        {
            "name": "Profile B — Low Risk (Loyal, Two-year, DSL)",
            "data": {
                "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"Yes",
                "tenure":60,"PhoneService":"Yes","MultipleLines":"Yes",
                "InternetService":"DSL","OnlineSecurity":"Yes",
                "OnlineBackup":"Yes","DeviceProtection":"Yes","TechSupport":"Yes",
                "StreamingTV":"No","StreamingMovies":"No",
                "Contract":"Two year","PaperlessBilling":"No",
                "PaymentMethod":"Bank transfer",
                "MonthlyCharges":55.20,"TotalCharges":3312.00,
            }
        },
        {
            "name": "Profile C — Medium Risk (1yr contract, no tech support)",
            "data": {
                "gender":"Male","SeniorCitizen":1,"Partner":"No","Dependents":"No",
                "tenure":18,"PhoneService":"Yes","MultipleLines":"No",
                "InternetService":"Fiber optic","OnlineSecurity":"No",
                "OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No",
                "StreamingTV":"Yes","StreamingMovies":"No",
                "Contract":"One year","PaperlessBilling":"Yes",
                "PaymentMethod":"Credit card",
                "MonthlyCharges":79.85,"TotalCharges":1437.30,
            }
        },
    ]

    print("\n" + "=" * 55)
    print(" CHURN PREDICTION — DEMO CUSTOMER PROFILES")
    print("=" * 55)
    for c in customers:
        result = predict_single(c["data"], bundle)
        print(f"\n{c['name']}")
        print(f"  Prediction   : {result['churn_prediction']}")
        print(f"  Probability  : {result['churn_probability']}%")
        print(f"  Risk Level   : {result['risk_level']}")
    print("\n[DONE] Predictions complete.")


if __name__ == "__main__":
    bundle = load_model()
    demo_predictions(bundle)
