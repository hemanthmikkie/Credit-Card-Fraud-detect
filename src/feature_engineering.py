"""Feature engineering module for extracting meaningful fraud indicators without data leakage."""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for credit card transactions.
    
    Transforms:
    1. Extracts 'hour_of_day' from elapsed seconds (Time // 3600 % 24).
    2. Encodes cyclical periodic diurnal patterns via 'sin_hour' and 'cos_hour'.
    3. Calculates 'log_amount' using log1p transformation to temper extreme right-skew.
    4. Computes 'amount_per_hour_ratio' against historical median amount learned strictly from training data.
    """

    def __init__(self):
        self.fitted_median_amount_ = 0.0
        self.engineered_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer parameters strictly on training split."""
        df = X.copy()
        if "Amount" in df.columns:
            self.fitted_median_amount_ = float(df["Amount"].median())
        else:
            self.fitted_median_amount_ = 22.0  # Fallback default
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations without data leakage."""
        df = X.copy()
        
        # 1. Temporal Feature Extraction
        if "Time" in df.columns:
            # Time is recorded as seconds elapsed from the first transaction in the dataset
            total_hours = df["Time"] / 3600.0
            hour_of_day = (total_hours % 24).astype(float)
            df["hour_of_day"] = hour_of_day
            
            # Cyclical encoding: ensures hour 23 and hour 0 are mathematically adjacent
            df["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24.0)
            df["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24.0)
        else:
            df["hour_of_day"] = 12.0
            df["sin_hour"] = 0.0
            df["cos_hour"] = -1.0

        # 2. Amount Non-Linear Transformation
        if "Amount" in df.columns:
            # log1p handles 0 amounts safely and controls extreme outliers
            df["log_amount"] = np.log1p(np.maximum(0, df["Amount"].values))
            
            # Relative amount ratio compared to baseline training median
            denom = max(1.0, self.fitted_median_amount_)
            df["amount_ratio"] = df["Amount"] / denom
        else:
            df["log_amount"] = 0.0
            df["amount_ratio"] = 1.0

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all feature columns excluding Target and Raw Time."""
    exclude = ["Class", "Time"]
    return [col for col in df.columns if col not in exclude]

