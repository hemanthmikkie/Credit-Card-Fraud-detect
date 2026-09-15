"""Data preprocessing module implementing stratified splitting, scaling, and SMOTE balancing without data leakage."""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    CLEANED_DATA_PATH,
    PREPROCESSING_PATH,
    TEST_SIZE,
    RANDOM_STATE
)
from src.feature_engineering import FraudFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PreprocessingArtifact:
    """Container holding all fitted transformers, scalers, and column schemas for inference."""
    def __init__(self, feature_engineer: FraudFeatureEngineer, scaler: RobustScaler, feature_names: List[str]):
        self.feature_engineer = feature_engineer
        self.scaler = scaler
        self.feature_names = feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and robust scaling in exact fitted order."""
        # 1. Feature Engineering
        fe_df = self.feature_engineer.transform(df)
        
        # 2. Select only expected features
        X_df = fe_df[self.feature_names].copy()
        
        # 3. Apply Scaling
        scaled_array = self.scaler.transform(X_df)
        return pd.DataFrame(scaled_array, columns=self.feature_names, index=df.index)


def prepare_data_pipeline(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingArtifact]:
    """
    Perform stratified split, feature engineering, and robust scaling strictly preventing data leakage.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_artifact
    """
    logger.info("Splitting dataset into stratified train and test sets...")
    
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' missing from DataFrame.")
        
    X_raw = df.drop(columns=["Class"])
    y = df["Class"]
    
    # 1. Stratified Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    logger.info(f"Train split: {len(X_train_raw):,} samples | Test split: {len(X_test_raw):,} samples.")
    logger.info(f"Train fraud count: {y_train.sum()} ({y_train.mean()*100:.3f}%) | Test fraud count: {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    
    # 2. Fit Feature Engineering STRICTLY on Train split
    fe = FraudFeatureEngineer()
    fe.fit(X_train_raw)
    
    X_train_fe = fe.transform(X_train_raw)
    X_test_fe = fe.transform(X_test_raw)
    
    # Exclude raw 'Time' if present from model inputs
    feature_cols = [c for c in X_train_fe.columns if c != "Time"]
    
    X_train_selected = X_train_fe[feature_cols]
    X_test_selected = X_test_fe[feature_cols]
    
    # 3. Fit RobustScaler STRICTLY on Train split
    scaler = RobustScaler()
    scaler.fit(X_train_selected)
    
    X_train_scaled_arr = scaler.transform(X_train_selected)
    X_test_scaled_arr = scaler.transform(X_test_selected)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=feature_cols, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=feature_cols, index=X_test_raw.index)
    
    artifact = PreprocessingArtifact(feature_engineer=fe, scaler=scaler, feature_names=feature_cols)
    return X_train_scaled, X_test_scaled, y_train, y_test, artifact


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Synthesize minority fraud cases using SMOTE ONLY on the training dataset.
    
    NEVER apply SMOTE to the test dataset!
    Sampling strategy=0.1 balances the training set to ~10% minority fraud cases,
    avoiding synthetic noise while boosting model sensitivity to rare anomalies.
    """
    logger.info(f"Applying SMOTE to training split (original fraud ratio: {y_train.mean()*100:.3f}%)...")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled_arr, y_resampled_arr = smote.fit_resample(X_train, y_train)
    
    X_resampled = pd.DataFrame(X_resampled_arr, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled_arr, name=y_train.name)
    
    logger.info(
        f"SMOTE complete. Resampled Train: {len(X_resampled):,} samples "
        f"(Legitimate: {(y_resampled == 0).sum():,}, Fraud: {(y_resampled == 1).sum():,}, "
        f"Fraud Ratio: {y_resampled.mean()*100:.2f}%)"
    )
    return X_resampled, y_resampled


def save_preprocessing_artifact(artifact: PreprocessingArtifact, output_path: Path = PREPROCESSING_PATH) -> None:
    """Serialize the fitted preprocessing artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"Saved preprocessing artifact to {output_path.resolve()}")


def load_preprocessing_artifact(input_path: Path = PREPROCESSING_PATH) -> PreprocessingArtifact:
    """Load the serialized preprocessing artifact."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing artifact not found at {path.resolve()}")
    return joblib.load(path)

