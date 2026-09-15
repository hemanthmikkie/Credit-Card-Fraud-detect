"""Model evaluation and threshold optimization module generating publication-quality visual diagnostics."""

import logging
from pathlib import Path
from typing import Dict, Any, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from src.config import (
    CLEANED_DATA_PATH,
    MODEL_PATH,
    PREPROCESSING_PATH,
    REPORTS_DIR,
    CLASSIFICATION_REPORT_PATH,
    OPTIMAL_THRESHOLD
)
from src.preprocessing import prepare_data_pipeline, load_preprocessing_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set clean aesthetic styling for Matplotlib/Seaborn
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot high-contrast normalized and count confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Format annotations: Count + (Percentage)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j] * 100
            annot[i, j] = f"{c:,}\n({p:.1f}%)"
            
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        linewidths=1.5,
        linecolor="#e2e8f0"
    )
    ax.set_title("Confusion Matrix (Credit Card Fraud Detection)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Class", fontsize=11, fontweight="bold")
    ax.set_xticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=10)
    ax.set_yticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=10, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path.resolve()}")


def plot_roc_and_pr_curves(y_true: pd.Series, y_prob: np.ndarray, reports_dir: Path) -> None:
    """Generate separate standalone ROC and Precision-Recall diagnostic curves."""
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curve (Crucial for Imbalanced Datasets)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#059669", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="#dc2626", lw=1.5, linestyle="--", label=f"Baseline (Prevalence = {baseline:.4f})")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall (Fraud Coverage)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Fraud Identification Accuracy)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curve.png", dpi=300)
    plt.close()
    logger.info(f"Saved ROC & PR curves to {reports_dir.resolve()}")


def evaluate_threshold_tradeoffs(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Analyze the operational trade-offs across prediction probability thresholds.
    
    Thresholds: [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    Demonstrates the trade-off between False Positives (customer friction) and False Negatives (financial fraud loss).
    """
    thresholds = [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    records = []
    
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = 2 * (prec * rec) / max(1e-6, (prec + rec))
        
        records.append({
            "Threshold": thresh,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "TruePositives (Caught Fraud)": tp,
            "FalsePositives (Customer Friction)": fp,
            "FalseNegatives (Missed Fraud Loss)": fn,
            "TrueNegatives": tn
        })
        
    df_thresh = pd.DataFrame(records)
    logger.info("Threshold optimization evaluation table generated.")
    return df_thresh


def generate_shap_summary(model: Any, X_sample: pd.DataFrame, output_path: Path) -> None:
    """Generate SHAP feature importance summary plot."""
    logger.info("Computing SHAP feature attributions on representative test sample...")
    try:
        # Use TreeExplainer for tree-based models (Random Forest / XGBoost)
        explainer = shap.TreeExplainer(model)
        # Sample 500 rows for swift computation without lag
        sample = X_sample.sample(n=min(500, len(X_sample)), random_state=42)
        shap_values = explainer.shap_values(sample)
        
        # If binary classification returns list of [neg, pos], select positive class (fraud)
        if isinstance(shap_values, list):
            values_to_plot = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            values_to_plot = shap_values[:, :, 1]
        else:
            values_to_plot = shap_values

        plt.figure(figsize=(9, 7))
        shap.summary_plot(values_to_plot, sample, show=False, max_display=15)
        plt.title("SHAP Feature Importance (Fraud Predictors)", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP summary plot to {output_path.resolve()}")
    except Exception as e:
        logger.warning(f"SHAP generation encountered note: {e}. Generating fallback bar plot.")
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_sample.columns).sort_values(ascending=False).head(15)
            plt.figure(figsize=(9, 6))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis")
            plt.title("Top Feature Importances (Tree Model)", fontsize=13, fontweight="bold")
            plt.xlabel("Gini / Gain Importance")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


def run_full_evaluation():
    """Execute complete evaluation, save visual artifacts and threshold reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data and preprocessing
    df = pd.read_csv(CLEANED_DATA_PATH)
    _, X_test_scaled, _, y_test, _ = prepare_data_pipeline(df)
    
    # 2. Load model
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model for evaluation from {MODEL_PATH}")
    
    # 3. Predict on unseen test set
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_optimal = (y_prob >= OPTIMAL_THRESHOLD).astype(int)
    
    # 4. Generate Confusion Matrix & Curves
    plot_confusion_matrix(y_test, y_pred_optimal, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_and_pr_curves(y_test, y_prob, REPORTS_DIR)
    
    # 5. Classification Report
    cr_text = classification_report(y_test, y_pred_optimal, target_names=["Legitimate", "Fraudulent"], digits=4)
    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("CREDIT CARD FRAUD DETECTION - TEST SET CLASSIFICATION REPORT\n")
        f.write(f"Decision Threshold: {OPTIMAL_THRESHOLD}\n")
        f.write("=" * 65 + "\n")
        f.write(cr_text)
        f.write("\n" + "=" * 65 + "\n")
    logger.info(f"Saved classification report to {CLASSIFICATION_REPORT_PATH.resolve()}")
    
    # 6. Threshold Trade-Off Analysis
    thresh_df = evaluate_threshold_tradeoffs(y_test, y_prob)
    thresh_df.to_csv(REPORTS_DIR / "threshold_tradeoffs.csv", index=False)
    
    # 7. SHAP Feature Importance
    generate_shap_summary(model, X_test_scaled, REPORTS_DIR / "shap_summary.png")
    
    print("\n" + "=" * 80)
    print(" THRESHOLD SENSITIVITY & BUSINESS TRADE-OFF ANALYSIS")
    print("=" * 80)
    print(thresh_df.to_markdown(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_full_evaluation()

