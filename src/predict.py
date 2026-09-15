"""Prediction and real-time risk intelligence inference engine with SHAP explainability."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    MODEL_PATH,
    PREPROCESSING_PATH,
    THRESHOLD_CONFIG_PATH,
    OPTIMAL_THRESHOLD,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH
)
from src.preprocessing import PreprocessingArtifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Singleton inference engine for scoring credit card transactions.
    
    Loads model and preprocessing artifacts once at initialization.
    """

    def __init__(self):
        self.model = None
        self.preprocessing: PreprocessingArtifact = None
        self.config: Dict[str, Any] = {}
        self.explainer = None
        self.optimal_threshold = OPTIMAL_THRESHOLD
        self.risk_low = RISK_THRESHOLD_LOW
        self.risk_high = RISK_THRESHOLD_HIGH
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model, preprocessing pipeline, and configuration."""
        if not MODEL_PATH.exists() or not PREPROCESSING_PATH.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {MODEL_PATH.parent}. "
                "Please run train_model.py first."
            )
            
        logger.info("Loading model and preprocessing artifacts...")
        self.model = joblib.load(MODEL_PATH)
        self.preprocessing = joblib.load(PREPROCESSING_PATH)
        
        if THRESHOLD_CONFIG_PATH.exists():
            try:
                with open(THRESHOLD_CONFIG_PATH, "r") as f:
                    self.config = json.load(f)
                    self.optimal_threshold = self.config.get("optimal_threshold", self.optimal_threshold)
                    risk_cfg = self.config.get("risk_levels", {})
                    self.risk_low = risk_cfg.get("low_max", self.risk_low)
                    self.risk_high = risk_cfg.get("medium_max", self.risk_high)
            except Exception as e:
                logger.warning(f"Could not read threshold config: {e}. Using defaults.")

        # Initialize SHAP TreeExplainer for instant, low-latency explanations
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer successfully initialized for real-time explanations.")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer not supported directly on model: {e}")
            self.explainer = None

    def calculate_risk_level(self, risk_score: float) -> str:
        """Categorize risk score into LOW, MEDIUM, or HIGH risk bands."""
        if risk_score <= self.risk_low:
            return "LOW"
        elif risk_score <= self.risk_high:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_top_contributing_features(self, X_transformed: pd.DataFrame, top_n: int = 4) -> List[Dict[str, Any]]:
        """
        Compute top contributing features for this prediction using SHAP or tree weights.
        Explains model attributions without claiming real-world causal certainty.
        """
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_transformed)
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0]
                elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                    vals = shap_vals[0, :, 1]
                else:
                    vals = shap_vals[0]
                    
                contrib_df = pd.DataFrame({
                    "feature": X_transformed.columns,
                    "contribution": vals
                }).sort_values(by="contribution", ascending=False)
                
                # Take features that positively push towards fraud risk
                top_features = contrib_df.head(top_n).to_dict(orient="records")
                return [
                    {
                        "feature": row["feature"],
                        "impact": "Increases Fraud Risk" if row["contribution"] > 0 else "Reduces Fraud Risk",
                        "weight": round(float(abs(row["contribution"])), 4)
                    }
                    for row in top_features
                ]
            except Exception as e:
                logger.debug(f"SHAP local attribution exception: {e}")

        # Fallback to model feature importances if available
        if hasattr(self.model, "feature_importances_"):
            top_indices = np.argsort(self.model.feature_importances_)[::-1][:top_n]
            return [
                {
                    "feature": X_transformed.columns[i],
                    "impact": "High Relative Model Importance",
                    "weight": round(float(self.model.feature_importances_[i]), 4)
                }
                for i in top_indices
            ]
        return []

    def predict_single(self, transaction_data: Dict[str, Any], transaction_id: str = "TX_LIVE") -> Dict[str, Any]:
        """
        Score a single transaction dictionary.
        
        Args:
            transaction_data: Dict containing Time, Amount, V1..V28.
            transaction_id: Unique tracking ID.
            
        Returns:
            Structured risk intelligence response dictionary.
        """
        df_input = pd.DataFrame([transaction_data])
        
        # 1. Transform features via fitted preprocessing artifact
        X_trans = self.preprocessing.transform(df_input)
        
        # 2. Predict probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        risk_score = round(prob * 100, 2)
        risk_level = self.calculate_risk_level(risk_score)
        prediction = "FRAUD" if prob >= self.optimal_threshold else "LEGITIMATE"
        
        # 3. Explainability factors
        top_factors = self.get_top_contributing_features(X_trans, top_n=4)
        
        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "fraud_probability": round(prob, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "threshold_applied": self.optimal_threshold,
            "top_contributing_factors": top_factors
        }


# Global predictor instance for FastAPI dependency injection
_predictor_instance = None

def get_predictor() -> FraudPredictor:
    """Singleton getter for the FraudPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor()
    return _predictor_instance

