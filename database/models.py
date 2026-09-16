"""SQLAlchemy ORM models for transaction persistence."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Index,
    CheckConstraint
)
from database.database import Base


class TransactionRecord(Base):
    """
    Transactions table stored in the configured SQLAlchemy database.
    Strictly devoid of PII or raw card numbers.
    """
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, nullable=False, index=True)
    transaction_time = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    actual_class = Column(Integer, nullable=True)
    predicted_class = Column(String(20), nullable=False, index=True)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False, index=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    __table_args__ = (
        CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH')", name="check_risk_level"),
        CheckConstraint("predicted_class IN ('LEGITIMATE', 'FRAUD')", name="check_predicted_class"),
        CheckConstraint("fraud_probability >= 0.0 AND fraud_probability <= 1.0", name="check_prob_range"),
        CheckConstraint("risk_score >= 0.0 AND risk_score <= 100.0", name="check_risk_range"),
        Index("idx_risk_created", "risk_level", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "transaction_time": self.transaction_time,
            "amount": self.amount,
            "actual_class": self.actual_class,
            "predicted_class": self.predicted_class,
            "fraud_probability": self.fraud_probability,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
