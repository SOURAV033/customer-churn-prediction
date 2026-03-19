"""
preprocess.py
=============
Data loading, cleaning, and feature engineering pipeline
for the Customer Churn Prediction project.

Author : Sourav Nayak
GitHub : github.com/SOURAV033
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    - Drop customerID (non-predictive)
    - Convert TotalCharges to numeric
    - Handle missing/empty values
    - Fill missing TotalCharges with 0 (new customers with 0 tenure)
    """
    df = df.copy()

    # Drop identifier column
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges can have whitespace strings in real data
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna().sum()
    if missing > 0:
        print(f"[INFO] Filling {missing} missing TotalCharges with 0 (new customers)")
        df["TotalCharges"].fillna(0, inplace=True)

    # Binary target encoding
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    print(f"[INFO] Churn distribution:\n{df['Churn'].value_counts()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - tenure_group: bucket tenure into lifecycle stages
    - avg_monthly_spend: MonthlyCharges / (tenure+1) ratio
    - has_multiple_services: flag for bundled services
    - high_value_flag: MonthlyCharges > 75th percentile
    """
    df = df.copy()

    # Tenure lifecycle buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New (0-12m)", "Growing (1-2yr)", "Established (2-4yr)", "Loyal (4yr+)"]
    )

    # Average monthly spend
    df["avg_monthly_spend"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Bundled services flag (has both phone and internet)
    df["has_bundle"] = (
        (df["PhoneService"] == "Yes") &
        (df["InternetService"].isin(["DSL", "Fiber optic"]))
    ).astype(int)

    # High-value customer flag
    threshold = df["MonthlyCharges"].quantile(0.75)
    df["high_value"] = (df["MonthlyCharges"] > threshold).astype(int)

    # Contract risk score (month-to-month = highest risk)
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk"] = df["Contract"].map(contract_map)

    print(f"[INFO] Feature engineering complete. New shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical variables and scale numeric features.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # Separate target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Identify column types
    cat_cols  = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols  = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"[INFO] Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"[INFO] Numerical columns  ({len(num_cols)}): {num_cols}")

    # Label-encode categoricals
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Train/test split (stratified to preserve churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler


if __name__ == "__main__":
    df = load_data("data/telco_churn.csv")
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features, scaler = encode_features(df)
    print("\n[DONE] Preprocessing pipeline complete.")
    print(f"       Features: {len(features)}")
    print(f"       Train churn rate: {y_train.mean():.3f}")
    print(f"       Test  churn rate: {y_test.mean():.3f}")
