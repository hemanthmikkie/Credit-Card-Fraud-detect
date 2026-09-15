"""Configuration module for paths, hyper-parameters, and runtime settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Directory Paths
DATASET_RAW_DIR = BASE_DIR / "dataset" / "raw"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# File Paths
RAW_DATA_PATH = DATASET_RAW_DIR / "creditcard.csv"
CLEANED_DATA_PATH = DATASET_PROCESSED_DIR / "creditcard_cleaned.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.pkl"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Model & Evaluation Configuration
RANDOM_STATE = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = 0.20
STRATIFY = True

# Risk Scoring Configuration
RISK_THRESHOLD_LOW = float(os.getenv("RISK_THRESHOLD_LOW", 30.0))
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", 70.0))
OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_DECISION_THRESHOLD", 0.35))

# Database Settings
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/credit_card_fraud_db"
)

# SQLite fallback path for local offline development/testing
SQLITE_FALLBACK_URL = f"sqlite:///{BASE_DIR / 'database' / 'credit_card_fraud.db'}"

# API Settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))

