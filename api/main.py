"""FastAPI application for real-time Credit Card Fraud Detection and Risk Intelligence."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db_session, get_prediction_engine
from api.schemas import (
    TransactionInput,
    PredictionResponse,
    TransactionRecordResponse,
    FraudSummaryResponse,
    HealthResponse
)
from database.database import init_db, engine
from database.crud import (
    create_transaction_record,
    get_transaction_by_id,
    get_transactions,
    get_high_risk_transactions,
    get_fraud_summary_metrics
)
from src.predict import FraudPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize DB tables and warm up ML pipeline."""
    logger.info("Initializing Credit Card Fraud Detection API service...")
    try:
        init_db()
        # Warm up predictor
        _ = get_prediction_engine()
        logger.info("Database initialized and ML prediction engine ready.")
    except Exception as e:
        logger.error(f"Startup initialization warning: {e}")
    yield
    logger.info("Shutting down API service.")


app = FastAPI(
    title="Credit Card Fraud Detection & Risk Intelligence API",
    description=(
        "Production-grade REST API providing real-time credit card fraud risk scoring, "
        "SHAP-based predictive attributions, persistent transaction auditing, and analytics summary KPIs."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend or dashboard consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    """Root metadata endpoint with direct links to documentation."""
    return {
        "system": "Credit Card Fraud Detection & Risk Intelligence System",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "transactions": "GET /transactions",
            "fraud_summary": "GET /fraud-summary",
            "high_risk": "GET /high-risk-transactions"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    db: Session = Depends(get_db_session),
    predictor: FraudPredictor = Depends(get_prediction_engine)
):
    """Health check validating ML model readiness and database connectivity."""
    db_connected = False
    try:
        # Check database connectivity
        db.execute(sqlalchemy_text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_connected = False

    model_loaded = predictor.model is not None and predictor.preprocessing is not None
    model_name = predictor.config.get("model_name", "XGBoost Classifier")

    return HealthResponse(
        status="healthy" if (db_connected and model_loaded) else "degraded",
        environment="production-ready",
        model_loaded=model_loaded,
        model_algorithm=model_name,
        database_connected=db_connected,
        database_engine=engine.name,
        timestamp=datetime.now(timezone.utc)
    )


# Helper for raw text queries
from sqlalchemy import text as sqlalchemy_text


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Scoring Engine"]
)
def predict_transaction(
    payload: TransactionInput,
    db: Session = Depends(get_db_session),
    predictor: FraudPredictor = Depends(get_prediction_engine)
):
    """
    Score a credit card transaction in real time.
    
    1. Validates input schema via Pydantic.
    2. Applies feature engineering and robust scaling.
    3. Computes posterior fraud probability and risk score (0-100).
    4. Categorizes into LOW, MEDIUM, or HIGH risk tier.
    5. Calculates top predictive SHAP factor attributions.
    6. Persists transaction into PostgreSQL / SQLite database.
    7. Returns clean risk intelligence payload.
    """
    try:
        data_dict = payload.model_dump()
        tx_id = data_dict["transaction_id"]
        
        # 1. Generate Prediction & Explainability
        prediction_result = predictor.predict_single(data_dict, transaction_id=tx_id)
        
        # 2. Persist to Database
        record_data = {
            "transaction_id": tx_id,
            "transaction_time": data_dict["Time"],
            "amount": data_dict["Amount"],
            "actual_class": data_dict.get("actual_class"),
            "predicted_class": prediction_result["prediction"],
            "fraud_probability": prediction_result["fraud_probability"],
            "risk_score": prediction_result["risk_score"],
            "risk_level": prediction_result["risk_level"]
        }
        create_transaction_record(db, record_data)
        
        return PredictionResponse(**prediction_result)
        
    except Exception as e:
        logger.error(f"Prediction failure on {payload.transaction_id}: {e}", exc_info=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error processing transaction: {str(e)}"
        )


@app.get("/transactions", response_model=List[TransactionRecordResponse], tags=["Surveillance"])
def list_transactions(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=500, description="Maximum items to return"),
    risk_level: Optional[str] = Query(None, description="Filter by risk tier: LOW, MEDIUM, or HIGH"),
    db: Session = Depends(get_db_session)
):
    """Retrieve scored transactions with pagination and optional risk level filter."""
    records = get_transactions(db, skip=skip, limit=limit, risk_level=risk_level)
    return [TransactionRecordResponse(**r.to_dict()) for r in records]


@app.get("/transactions/{transaction_id}", response_model=TransactionRecordResponse, tags=["Surveillance"])
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db_session)
):
    """Retrieve audit record for a specific transaction ID."""
    record = get_transaction_by_id(db, transaction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID '{transaction_id}' was not found."
        )
    return TransactionRecordResponse(**record.to_dict())


@app.get("/fraud-summary", response_model=FraudSummaryResponse, tags=["Analytics"])
def fraud_summary(db: Session = Depends(get_db_session)):
    """Executive overview metrics across all captured transactions."""
    metrics = get_fraud_summary_metrics(db)
    return FraudSummaryResponse(**metrics)


@app.get("/high-risk-transactions", response_model=List[TransactionRecordResponse], tags=["Surveillance"])
def high_risk_transactions(
    limit: int = Query(50, ge=1, le=200, description="Maximum high risk alerts to return"),
    db: Session = Depends(get_db_session)
):
    """Fetch high-risk transactions sorted by descending fraud probability for immediate investigation."""
    records = get_high_risk_transactions(db, limit=limit)
    return [TransactionRecordResponse(**r.to_dict()) for r in records]

