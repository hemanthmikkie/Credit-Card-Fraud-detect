"""Data cleaning module for sanitizing raw credit card transaction data."""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH
from src.data_loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate expected column presence and data types."""
    required_cols = ["Time", "Amount", "Class"]
    pca_cols = [f"V{i}" for i in range(1, 29)]
    all_expected = required_cols + pca_cols
    
    missing_cols = [col for col in all_expected if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    
    # Check invalid target values
    unique_targets = set(df["Class"].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column contains invalid values: {unique_targets}. Expected subset of {{0, 1}}.")
    logger.info("Schema validation passed successfully.")


def clean_missing_and_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Check and treat missing or infinite values."""
    df_clean = df.copy()
    
    # Replace inf and -inf with NaN
    num_inf = np.isinf(df_clean.select_dtypes(include=[np.number])).values.sum()
    if num_inf > 0:
        logger.warning(f"Detected {num_inf} infinite values. Converting to NaN.")
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute missing if any (median for numerical)
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Detected {missing_count} missing values. Imputing with median.")
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0 and col != "Class":
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        logger.info("Zero missing or infinite values found.")
        
    return df_clean


def handle_duplicates(df: pd.DataFrame, drop: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    Detect and handle duplicate transactions.
    
    Note: In financial datasets with anonymized PCA features, exact duplicate rows
    can indicate system retry glitches or identical logging. We document the duplicate
    count and remove exact duplicates to ensure training integrity.
    """
    dup_count = int(df.duplicated().sum())
    logger.info(f"Duplicate rows detected: {dup_count:,}")
    if drop and dup_count > 0:
        df_dedup = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dropped {dup_count:,} duplicate rows. Remaining: {len(df_dedup):,}")
        return df_dedup, dup_count
    return df, dup_count


def analyze_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Investigate outliers in Transaction Amount.
    
    Crucial Rule: Do NOT blindly remove high-amount transactions.
    In fraud detection, extreme values often represent high-value enterprise transactions
    or massive fraud bursts. Removing them causes significant survivorship bias.
    """
    amount = df["Amount"]
    q1 = amount.quantile(0.25)
    q3 = amount.quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    
    outliers_mask = amount > upper_bound
    outliers_df = df[outliers_mask]
    
    total_outliers = int(outliers_mask.sum())
    fraud_in_outliers = int(outliers_df["Class"].sum()) if "Class" in outliers_df.columns else 0
    fraud_rate_in_outliers = (fraud_in_outliers / max(1, total_outliers)) * 100
    
    summary = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "upper_bound": float(upper_bound),
        "total_outliers": total_outliers,
        "outlier_percentage": round((total_outliers / len(df)) * 100, 2),
        "fraud_count_in_outliers": fraud_in_outliers,
        "fraud_rate_in_outliers": round(fraud_rate_in_outliers, 2),
        "recommendation": (
            "Retain all outliers. High transaction amounts include both legitimate high-net-worth "
            "spending and high-impact fraud attempts. Robust scaling / log-transformation will be "
            "applied during feature engineering to mitigate sensitivity without losing fraud signals."
        )
    }
    logger.info(
        f"Outlier Analysis: {total_outliers:,} outliers found (> ${upper_bound:.2f}). "
        f"Contains {fraud_in_outliers} fraudulent transactions. Verdict: RETAIN."
    )
    return summary


def clean_dataset(input_path: Path = RAW_DATA_PATH, output_path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    End-to-end data cleaning pipeline.
    
    1. Validate schema
    2. Handle missing & inf values
    3. Remove exact duplicate transactions
    4. Profile outliers without dropping
    5. Save cleaned dataset to processed directory
    """
    logger.info("Starting data cleaning pipeline...")
    df = load_raw_data(input_path)
    validate_schema(df)
    
    df_clean = clean_missing_and_inf(df)
    df_clean, _ = handle_duplicates(df_clean, drop=True)
    _ = analyze_outliers(df_clean)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved to: {output_path.resolve()} ({len(df_clean):,} rows)")
    return df_clean


if __name__ == "__main__":
    cleaned_df = clean_dataset()
    print("Data cleaning execution complete.")

