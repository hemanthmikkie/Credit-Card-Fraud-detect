"""Unit tests for data loading and cleaning pipelines."""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import profile_data
from src.data_cleaning import validate_schema, clean_missing_and_inf, handle_duplicates, analyze_outliers


@pytest.fixture
def sample_raw_df():
    """Create a controlled mini-dataset mimicking credit card data schema."""
    np.random.seed(42)
    n = 100
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.linspace(0, 3600, n)
    data["Amount"] = np.random.exponential(scale=50, size=n)
    # 98 legit, 2 fraud
    classes = np.zeros(n, dtype=int)
    classes[10] = 1
    classes[50] = 1
    data["Class"] = classes
    return pd.DataFrame(data)


def test_schema_validation_success(sample_raw_df):
    """Schema validation should pass for properly structured datasets."""
    validate_schema(sample_raw_df)


def test_schema_validation_missing_col(sample_raw_df):
    """Schema validation should raise ValueError when mandatory columns are missing."""
    corrupted = sample_raw_df.drop(columns=["V14"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(corrupted)


def test_schema_validation_invalid_class(sample_raw_df):
    """Schema validation should catch invalid target labels."""
    corrupted = sample_raw_df.copy()
    corrupted.loc[0, "Class"] = 99
    with pytest.raises(ValueError, match="invalid values"):
        validate_schema(corrupted)


def test_clean_missing_and_inf(sample_raw_df):
    """Missing and infinite values should be handled cleanly without crashing."""
    corrupted = sample_raw_df.copy()
    corrupted.loc[5, "V1"] = np.nan
    corrupted.loc[6, "Amount"] = np.inf
    
    cleaned = clean_missing_and_inf(corrupted)
    assert not cleaned.isnull().values.any()
    assert not np.isinf(cleaned.select_dtypes(include=[np.number])).values.any()


def test_handle_duplicates(sample_raw_df):
    """Duplicate detection should accurately count and remove identical entries."""
    dup_df = pd.concat([sample_raw_df, sample_raw_df.iloc[[0, 1]]], ignore_index=True)
    deduped, count = handle_duplicates(dup_df, drop=True)
    assert count == 2
    assert len(deduped) == len(sample_raw_df)


def test_analyze_outliers_does_not_drop(sample_raw_df):
    """Outlier analysis must produce summary statistics without truncating data."""
    summary = analyze_outliers(sample_raw_df)
    assert "total_outliers" in summary
    assert "outlier_percentage" in summary
    assert "recommendation" in summary

