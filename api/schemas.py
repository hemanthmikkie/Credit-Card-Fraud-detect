"""Pydantic V2 schemas for input validation, API responses, and analytics contracts."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ContributingFactor(BaseModel):
    feature: str = Field(..., description="Feature identifier (e.g. V14, Amount)")
    impact: str = Field(..., description="Direction of risk attribution")
    weight: float = Field(..., description="Magnitude of contribution")


class TransactionInput(BaseModel):
    """Transaction scoring payload. Requires Time, Amount, and 28 PCA-transformed factors."""
    transaction_id: str = Field(..., description="Unique alphanumeric transaction reference", min_length=3, max_length=64)
    Time: float = Field(..., description="Seconds elapsed since reference baseline", ge=0.0)
    Amount: float = Field(..., description="Monetary transaction amount", ge=0.0)
    actual_class: Optional[int] = Field(None, description="Optional ground truth (0 or 1) for historical validation")
    
    # PCA components V1 through V28
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")

    @field_validator("actual_class")
    @classmethod
    def validate_actual_class(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("actual_class must be either 0 (legitimate) or 1 (fraud)")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TX_TEST_9901",
                "Time": 42000.0,
                "Amount": 249.50,
                "actual_class": None,
                "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
                "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986,
                "V9": 0.3637, "V10": 0.0907, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
                "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
                "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210
            }
        }
    )


class PredictionResponse(BaseModel):
    """Standardized response contract for scored transactions."""
    transaction_id: str = Field(..., description="Echoed transaction ID")
    prediction: str = Field(..., description="Model classification: LEGITIMATE or FRAUD")
    fraud_probability: float = Field(..., description="Estimated fraud probability [0.0 - 1.0]")
    risk_score: float = Field(..., description="Normalized risk score [0 - 100]")
    risk_level: str = Field(..., description="Risk classification: LOW, MEDIUM, or HIGH")
    threshold_applied: float = Field(..., description="Operational decision threshold used")
    top_contributing_factors: Optional[List[ContributingFactor]] = Field(
        default=None,
        description="Key predictive signals driving this risk calculation"
    )


class TransactionRecordResponse(BaseModel):
    """Database entity serialization."""
    id: int
    transaction_id: str
    transaction_time: float
    amount: float
    actual_class: Optional[int]
    predicted_class: str
    fraud_probability: float
    risk_score: float
    risk_level: str
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class FraudSummaryResponse(BaseModel):
    """Aggregate KPI dashboard response."""
    total_transactions: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_rate_pct: float
    total_fraud_exposure_amount: float
    avg_transaction_amount: float
    risk_distribution: Dict[str, int]


class HealthResponse(BaseModel):
    """Service status and dependency health check."""
    status: str
    environment: str
    model_loaded: bool
    model_algorithm: str
    database_connected: bool
    database_engine: str
    timestamp: datetime

