"""FastAPI dependency injection providers for database sessions and ML predictors."""

from typing import Generator
from sqlalchemy.orm import Session
from database.database import get_db
from src.predict import get_predictor, FraudPredictor


def get_prediction_engine() -> FraudPredictor:
    """Dependency provider returning singleton FraudPredictor instance."""
    return get_predictor()


def get_db_session() -> Generator[Session, None, None]:
    """Dependency provider for scoped database sessions."""
    yield from get_db()

