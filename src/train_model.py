"""Model training and selection module comparing Logistic Regression, Decision Tree, Random Forest, and XGBoost."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.config import (
    CLEANED_DATA_PATH,
    MODEL_PATH,
    PREPROCESSING_PATH,
    THRESHOLD_CONFIG_PATH,
    MODEL_COMPARISON_PATH,
    RANDOM_STATE,
    OPTIMAL_THRESHOLD
)
from src.data_cleaning import clean_dataset
from src.preprocessing import prepare_data_pipeline, apply_smote, save_preprocessing_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate core classification metrics for imbalanced evaluation."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "PR-AUC": round(float(average_precision_score(y_true, y_prob)), 4)
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Any, str]:
    """
    Train and rigorously benchmark 4 ML algorithms using identical test splits.
    
    Returns:
        comparison_df, best_model, best_model_name
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=1.0,  # Training on SMOTE resampled data
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "=" * 80)
    print(f"{'MODEL TRAINING & EVALUATION BENCHMARK':^80}")
    print("=" * 80)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        # Test predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["TrainTimeSec"] = round(elapsed, 2)
        metrics["Model"] = name
        
        results.append(metrics)
        trained_models[name] = model
        logger.info(
            f"{name} Results -> Precision: {metrics['Precision']:.4f} | "
            f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}"
        )
        
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TrainTimeSec"]
    comparison_df = pd.DataFrame(results)[cols].sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    logger.info(f"Top performing model based on PR-AUC & F1-score: {best_model_name}")
    
    return comparison_df, best_model, best_model_name


def save_model_artifacts(
    model: Any,
    best_name: str,
    feature_names: list,
    optimal_threshold: float = OPTIMAL_THRESHOLD
) -> None:
    """Save trained model, feature configuration, and optimal threshold."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Saved best model ({best_name}) to {MODEL_PATH.resolve()}")
    
    config = {
        "model_name": best_name,
        "optimal_threshold": optimal_threshold,
        "feature_names": feature_names,
        "risk_levels": {
            "low_max": 30.0,
            "medium_max": 70.0,
            "high_min": 70.01
        }
    }
    with open(THRESHOLD_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved threshold configuration to {THRESHOLD_CONFIG_PATH.resolve()}")


def execute_training_pipeline() -> Tuple[pd.DataFrame, str]:
    """Execute complete training workflow from data loading to artifact persistence."""
    logger.info("Executing training pipeline...")
    
    # 1. Load or clean dataset
    if not CLEANED_DATA_PATH.exists():
        df = clean_dataset()
    else:
        df = pd.read_csv(CLEANED_DATA_PATH)
        
    # 2. Stratified Preprocessing & Scaler fit (NO DATA LEAKAGE)
    X_train_scaled, X_test_scaled, y_train, y_test, prep_artifact = prepare_data_pipeline(df)
    save_preprocessing_artifact(prep_artifact)
    
    # 3. SMOTE strictly on training split
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train, sampling_strategy=0.10)
    
    # 4. Train & Compare Models
    comparison_df, best_model, best_name = train_and_compare_models(
        X_train_resampled,
        y_train_resampled,
        X_test_scaled,
        y_test
    )
    
    # 5. Persist comparison report and models
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    save_model_artifacts(best_model, best_name, prep_artifact.feature_names)
    
    print("\n" + "=" * 80)
    print(" FINAL MODEL COMPARISON MATRIX")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    return comparison_df, best_name


if __name__ == "__main__":
    execute_training_pipeline()
