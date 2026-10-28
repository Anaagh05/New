import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/churn_processed.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, handle missing values, fix dtypes."""
    df = df.drop_duplicates()

    # TotalCharges is sometimes loaded as object due to whitespace
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID — not predictive
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    print(f"[INFO] After cleaning: {df.shape[0]} rows")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using LabelEncoder."""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [ENC] {col}")

    return df


def scale_numeric(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Standard-scale numeric columns (except target)."""
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[INFO] Scaled {len(num_cols)} numeric columns")
    return df


def split_and_save(df: pd.DataFrame, output_path: str):
    """Save processed dataframe."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed data → {output_path}")


if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_numeric(df)
    split_and_save(df, PROCESSED_DATA_PATH)
