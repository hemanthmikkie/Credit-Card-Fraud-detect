"""Database engine configuration with PostgreSQL support and automated SQLite fallback."""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

from src.config import DATABASE_URL, SQLITE_FALLBACK_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Base = declarative_base()

# Attempt connection to PostgreSQL; fallback to local SQLite if unreachable
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    # Test connection
    with engine.connect() as conn:
        logger.info(f"Connected to PostgreSQL database: {DATABASE_URL.split('@')[-1]}")
except Exception as e:
    logger.warning(
        f"Could not connect to PostgreSQL at {DATABASE_URL} ({e}). "
        f"Falling back to local SQLite database: {SQLITE_FALLBACK_URL}"
    )
    engine = create_engine(
        SQLITE_FALLBACK_URL,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all registered database tables and indexes."""
    from database.models import TransactionRecord
    logger.info("Initializing database tables and indexes...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully.")


def get_db():
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

