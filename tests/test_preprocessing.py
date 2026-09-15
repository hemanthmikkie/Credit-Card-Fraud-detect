"""Unit tests for feature engineering, scaling, and imbalanced SMOTE balancing."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FraudFeatureEngineer
from src.preprocessing import prepare_data_pipeline, apply_smote


@pytest.fixture
def synthetic_df():
    """Create a synthetic dataset with known distribution."""
    np.random.seed(42)
    n = 200
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.linspace(0, 86400, n)
    data["Amount"] = np.random.uniform(1, 500, n)
    classes = np.zeros(n, dtype=int)
    classes[:10] = 1  # 5% fraud
    data["Class"] = classes
    return pd.DataFrame(data)


def test_feature_engineering_transforms(synthetic_df):
    """Transformer should compute cyclical time, log_amount, and amount_ratio."""
    fe = FraudFeatureEngineer()
    fe.fit(synthetic_df)
    transformed = fe.transform(synthetic_df)
    
    assert "hour_of_day" in transformed.columns
    assert "sin_hour" in transformed.columns
    assert "cos_hour" in transformed.columns
    assert "log_amount" in transformed.columns
    assert "amount_ratio" in transformed.columns
    
    # Hours should be bounded between 0 and 24
    assert (transformed["hour_of_day"] >= 0).all() and (transformed["hour_of_day"] <= 24).all()


def test_prepare_data_pipeline_no_leakage(synthetic_df):
    """Verify that train/test split isolates splits cleanly and scales appropriately."""
    X_train, X_test, y_train, y_test, artifact = prepare_data_pipeline(synthetic_df, test_size=0.25, random_state=42)
    
    assert len(X_train) == 150
    assert len(X_test) == 50
    assert "Time" not in X_train.columns
    assert "Time" not in X_test.columns
    assert list(X_train.columns) == list(X_test.columns)
    
    # Stratification check: fraud ratio should be roughly preserved
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_smote_applied_strictly_to_train(synthetic_df):
    """Verify SMOTE resamples training data without touching test splits."""
    X_train, X_test, y_train, y_test, _ = prepare_data_pipeline(synthetic_df, test_size=0.20, random_state=42)
    
    original_fraud_count = y_train.sum()
    X_res, y_res = apply_smote(X_train, y_train, sampling_strategy=0.20, random_state=42)
    
    # Training size should increase due to synthetic samples
    assert len(X_res) > len(X_train)
    assert y_res.sum() > original_fraud_count
    # Test set remains completely unaltered
    assert len(X_test) == 40

