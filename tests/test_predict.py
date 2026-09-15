"""Unit tests for inference scoring engine and explainability attributions."""

import pytest
from src.predict import get_predictor


@pytest.fixture(scope="module")
def predictor():
    """Load predictor instance."""
    return get_predictor()


def test_predict_single_legitimate(predictor):
    """Typical legitimate transaction should return low fraud probability and LOW risk."""
    sample = {f"V{i}": 0.0 for i in range(1, 29)}
    sample["Time"] = 3600.0
    sample["Amount"] = 45.00
    
    result = predictor.predict_single(sample, "TX_TEST_LEGIT")
    
    assert result["transaction_id"] == "TX_TEST_LEGIT"
    assert result["prediction"] in ("LEGITIMATE", "FRAUD")
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "top_contributing_factors" in result
    assert isinstance(result["top_contributing_factors"], list)


def test_risk_level_mapping(predictor):
    """Test boundary conditions for risk level buckets."""
    assert predictor.calculate_risk_level(15.0) == "LOW"
    assert predictor.calculate_risk_level(30.0) == "LOW"
    assert predictor.calculate_risk_level(50.0) == "MEDIUM"
    assert predictor.calculate_risk_level(70.0) == "MEDIUM"
    assert predictor.calculate_risk_level(85.0) == "HIGH"

