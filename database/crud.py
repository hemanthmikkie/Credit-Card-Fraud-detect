"""CRUD (Create, Read, Update, Delete) operations for transaction tracking and risk monitoring."""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case

from database.models import TransactionRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_transaction_record(db: Session, record_data: Dict[str, Any]) -> TransactionRecord:
    """Insert a newly scored transaction into the database."""
    # Check if transaction_id already exists
    existing = db.query(TransactionRecord).filter(
        TransactionRecord.transaction_id == record_data["transaction_id"]
    ).first()
    
    if existing:
        for key, value in record_data.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing

    record = TransactionRecord(
        transaction_id=record_data["transaction_id"],
        transaction_time=record_data["transaction_time"],
        amount=record_data["amount"],
        actual_class=record_data.get("actual_class"),
        predicted_class=record_data["predicted_class"],
        fraud_probability=record_data["fraud_probability"],
        risk_score=record_data["risk_score"],
        risk_level=record_data["risk_level"]
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_transaction_by_id(db: Session, transaction_id: str) -> Optional[TransactionRecord]:
    """Fetch single transaction by its unique tracking ID."""
    return db.query(TransactionRecord).filter(
        TransactionRecord.transaction_id == transaction_id
    ).first()


def get_transactions(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    risk_level: Optional[str] = None
) -> List[TransactionRecord]:
    """Retrieve paginated list of transactions with optional risk tier filtering."""
    query = db.query(TransactionRecord)
    if risk_level:
        query = query.filter(TransactionRecord.risk_level == risk_level.upper())
    return query.order_by(desc(TransactionRecord.created_at)).offset(skip).limit(limit).all()


def get_high_risk_transactions(db: Session, limit: int = 50) -> List[TransactionRecord]:
    """Retrieve critical high-risk alerts prioritizing highest probability."""
    return db.query(TransactionRecord).filter(
        TransactionRecord.risk_level == "HIGH"
    ).order_by(desc(TransactionRecord.fraud_probability)).limit(limit).all()


def get_fraud_summary_metrics(db: Session) -> Dict[str, Any]:
    """Compute aggregate analytics KPIs across stored transactions."""
    total_tx = db.query(func.count(TransactionRecord.id)).scalar() or 0
    if total_tx == 0:
        return {
            "total_transactions": 0,
            "fraud_transactions": 0,
            "legitimate_transactions": 0,
            "fraud_rate_pct": 0.0,
            "total_fraud_exposure_amount": 0.0,
            "avg_transaction_amount": 0.0,
            "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        }

    fraud_tx = db.query(func.count(TransactionRecord.id)).filter(
        TransactionRecord.predicted_class == "FRAUD"
    ).scalar() or 0
    
    total_fraud_amount = db.query(func.sum(TransactionRecord.amount)).filter(
        TransactionRecord.predicted_class == "FRAUD"
    ).scalar() or 0.0
    
    avg_amount = db.query(func.avg(TransactionRecord.amount)).scalar() or 0.0
    
    # Risk level counts
    low_cnt = db.query(func.count(TransactionRecord.id)).filter(TransactionRecord.risk_level == "LOW").scalar() or 0
    med_cnt = db.query(func.count(TransactionRecord.id)).filter(TransactionRecord.risk_level == "MEDIUM").scalar() or 0
    high_cnt = db.query(func.count(TransactionRecord.id)).filter(TransactionRecord.risk_level == "HIGH").scalar() or 0

    return {
        "total_transactions": int(total_tx),
        "fraud_transactions": int(fraud_tx),
        "legitimate_transactions": int(total_tx - fraud_tx),
        "fraud_rate_pct": round((fraud_tx / total_tx) * 100, 2),
        "total_fraud_exposure_amount": round(float(total_fraud_amount), 2),
        "avg_transaction_amount": round(float(avg_amount), 2),
        "risk_distribution": {
            "LOW": int(low_cnt),
            "MEDIUM": int(med_cnt),
            "HIGH": int(high_cnt)
        }
    }

