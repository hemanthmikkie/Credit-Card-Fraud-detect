"""Data loader module for ingesting, validating, and profiling raw credit card transaction data."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw credit card dataset from CSV.
    
    Args:
        file_path: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame containing raw transactions.
        
    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If the file cannot be parsed or is empty.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"Raw dataset not found at {path.resolve()}. Please verify the download."
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading raw dataset from {path.resolve()}...")
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Failed to read dataset from {path}: {e}")
        raise ValueError(f"Corrupted or unreadable CSV file: {e}") from e


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data profiling metrics for the transaction dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary containing profiling metrics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_summary = pd.DataFrame({"MissingCount": missing_counts, "MissingPercent": missing_pct})
    
    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / total_rows) * 100
    
    # Target distribution
    target_col = "Class" if "Class" in df.columns else None
    target_dist = {}
    if target_col:
        counts = df[target_col].value_counts().to_dict()
        percentages = (df[target_col].value_counts(normalize=True) * 100).to_dict()
        target_dist = {
            "counts": counts,
            "percentages": percentages,
            "fraud_count": int(counts.get(1, 0)),
            "legitimate_count": int(counts.get(0, 0)),
            "fraud_percentage": float(percentages.get(1, 0.0))
        }

    profile = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round(duplicate_pct, 4),
        "missing_summary": missing_summary,
        "total_missing_cells": int(missing_counts.sum()),
        "target_distribution": target_dist,
        "numeric_summary": df.describe().to_dict()
    }
    return profile


def print_data_quality_report(df: pd.DataFrame) -> None:
    """Print a clean human-readable data quality report."""
    profile = profile_data(df)
    print("=" * 70)
    print(" CREDIT CARD FRAUD DETECTION - DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Total Transactions (Rows)   : {profile['total_rows']:,}")
    print(f"Total Attributes (Columns)   : {profile['total_columns']}")
    print(f"Duplicate Transactions       : {profile['duplicate_rows']:,} ({profile['duplicate_percentage']}%)")
    print(f"Total Missing Values         : {profile['total_missing_cells']}")
    
    td = profile["target_distribution"]
    if td:
        print("-" * 70)
        print("TARGET VARIABLE DISTRIBUTION ('Class'):")
        print(f"  Legitimate (0) : {td['legitimate_count']:,} ({100 - td['fraud_percentage']:.4f}%)")
        print(f"  Fraudulent (1) : {td['fraud_count']:,} ({td['fraud_percentage']:.4f}%)")
        print(f"  Imbalance Ratio: ~1 : {int(td['legitimate_count'] / max(1, td['fraud_count']))}")
    print("=" * 70)


if __name__ == "__main__":
    raw_df = load_raw_data()
    print_data_quality_report(raw_df)

