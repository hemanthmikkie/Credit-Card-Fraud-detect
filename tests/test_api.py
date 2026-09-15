"""Integration tests for FastAPI endpoints, request validation, and database tracking."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client using context manager to trigger lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Root endpoint should return 200 and system metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_check_endpoint(client):
    """Health endpoint should confirm model loading and DB connectivity."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["model_loaded"] is True


def test_predict_endpoint_valid_transaction(client):
    """POST /predict should score valid transaction and persist to database."""
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["transaction_id"] = "TX_PYTEST_001"
    payload["Time"] = 7200.0
    payload["Amount"] = 150.00
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 201
    data = response.json()
    
    assert data["transaction_id"] == "TX_PYTEST_001"
    assert data["prediction"] in ("LEGITIMATE", "FRAUD")
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "top_contributing_factors" in data


def test_predict_endpoint_negative_amount_validation_error(client):
    """Pydantic should reject negative transaction amounts with 422 status."""
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["transaction_id"] = "TX_PYTEST_INVALID"
    payload["Time"] = 100.0
    payload["Amount"] = -25.00  # Invalid
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_features_validation_error(client):
    """Pydantic should reject requests missing required PCA components."""
    payload = {
        "transaction_id": "TX_PYTEST_INCOMPLETE",
        "Time": 500.0,
        "Amount": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_transactions_listing_and_retrieval(client):
    """GET /transactions should return list and GET /transactions/{id} should return details."""
    # List transactions
    response = client.get("/transactions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    
    # Fetch specific transaction created in earlier test
    tx_resp = client.get("/transactions/TX_PYTEST_001")
    assert tx_resp.status_code == 200
    tx_data = tx_resp.json()
    assert tx_data["transaction_id"] == "TX_PYTEST_001"


def test_fraud_summary_endpoint(client):
    """GET /fraud-summary should return aggregate operational indicators."""
    response = client.get("/fraud-summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "fraud_rate_pct" in data
    assert "risk_distribution" in data
    assert data["total_transactions"] >= 1


def test_high_risk_endpoint(client):
    """GET /high-risk-transactions should return high risk alerts list."""
    response = client.get("/high-risk-transactions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

