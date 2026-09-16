"""Database engine configuration with an explicit local SQLite fallback."""

import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

from src.config import ALLOW_SQLITE_FALLBACK, DATABASE_URL, SQLITE_FALLBACK_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Base = declarative_base()

database_url = make_url(DATABASE_URL)
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,     # Recycle connections after 1 hour (prevents stale connections)
        pool_size=10,
        max_overflow=20
    )
    # Test connection
    with engine.connect() as conn:
        db_info = database_url.render_as_string(hide_password=True)
        logger.info("Connected to %s database: %s", database_url.drivername, db_info)
except SQLAlchemyError as e:
    if not ALLOW_SQLITE_FALLBACK:
        raise RuntimeError(
            f"Unable to connect to configured database ({database_url.drivername}). "
            "SQLite fallback is disabled; verify DATABASE_URL and database availability."
        ) from e

    logger.warning(
        "Could not connect to configured database (%s). "
        "Falling back to local SQLite database: %s",
        e,
        SQLITE_FALLBACK_URL
    )
    engine = create_engine(
        SQLITE_FALLBACK_URL,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all registered database tables."""
    from database.models import TransactionRecord  # noqa: F401 - registers the ORM model
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized on engine: {engine.name}")


def get_db():
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
