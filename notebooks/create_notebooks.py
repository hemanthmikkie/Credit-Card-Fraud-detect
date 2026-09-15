"""Script to generate publication-grade Jupyter Notebooks for the Fraud Detection system."""

import json
from pathlib import Path

notebooks_dir = Path(r"c:\10k\data science projects\Credit_Card_Fraud_Detection\notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)


def make_cell(cell_type: str, source: str):
    """Format single Jupyter cell."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def save_nb(filename: str, cells: list):
    """Write Jupyter Notebook JSON."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(notebooks_dir / filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {filename}")


# 01. Data Understanding
nb1 = [
    make_cell("markdown", "# 01. Data Understanding & Profiling\n\n## Credit Card Fraud Detection & Risk Intelligence System\n\n### Notebook Purpose:\n1. Ingest raw transactional dataset without altering the original file.\n2. Profile dataset structure: dimensions, column types, and schema validity.\n3. Compute completeness metrics, missing cell counts, and exact duplicate rows.\n4. Analyze target variable distribution (`Class`) and calculate class imbalance ratios."),
    make_cell("code", "import sys\nfrom pathlib import Path\n# Add parent directory to path\nsys.path.append(str(Path.cwd().parent))\n\nimport pandas as pd\nimport numpy as np\nfrom src.config import RAW_DATA_PATH\nfrom src.data_loader import load_raw_data, profile_data, print_data_quality_report\n\ndf = load_raw_data(RAW_DATA_PATH)\nprint_data_quality_report(df)"),
    make_cell("markdown", "### Dataset Attributes & Memory Footprint"),
    make_cell("code", "df.info()"),
    make_cell("markdown", "### Descriptive Summary of Transaction Features\n\nNotice that V1–V28 are numerical features obtained through PCA (Principal Component Analysis). `Time` contains elapsed seconds, and `Amount` represents transaction value in USD."),
    make_cell("code", "df.describe().T"),
    make_cell("markdown", "### Target Imbalance Analysis\n\nLegitimate transactions (`Class = 0`) constitute 99.83% of the dataset, while fraudulent transactions (`Class = 1`) represent only ~0.17% (a 577:1 imbalance ratio)."),
    make_cell("code", "counts = df['Class'].value_counts()\npcts = df['Class'].value_counts(normalize=True) * 100\npd.DataFrame({'Count': counts, 'Percentage': pcts})")
]
save_nb("01_data_understanding.ipynb", nb1)

# 02. Data Cleaning
nb2 = [
    make_cell("markdown", "# 02. Data Cleaning & Outlier Analysis\n\n## Objectives:\n1. Verify schema consistency and non-null constraints.\n2. Detect and handle exact duplicate transaction records.\n3. Conduct rigorous statistical outlier investigation on `Amount`.\n4. Explain why outliers must NOT be dropped blindly in fraud detection.\n5. Save sanitized data to `dataset/processed/`."),
    make_cell("code", "import sys\nfrom pathlib import Path\nsys.path.append(str(Path.cwd().parent))\n\nimport pandas as pd\nimport numpy as np\nfrom src.config import RAW_DATA_PATH, CLEANED_DATA_PATH\nfrom src.data_loader import load_raw_data\nfrom src.data_cleaning import validate_schema, clean_missing_and_inf, handle_duplicates, analyze_outliers\n\ndf = load_raw_data(RAW_DATA_PATH)\nvalidate_schema(df)"),
    make_cell("markdown", "### Handling Missing & Infinite Values"),
    make_cell("code", "df_clean = clean_missing_and_inf(df)\nprint('Total missing cells after check:', df_clean.isnull().sum().sum())"),
    make_cell("markdown", "### Duplicate Record Detection\n\nExact duplicate records often occur from payment network retry bursts or dual-logging."),
    make_cell("code", "df_dedup, dup_count = handle_duplicates(df_clean, drop=True)\nprint(f'Dropped {dup_count:,} duplicate transactions. Remaining clean rows: {len(df_dedup):,}')"),
    make_cell("markdown", "### Outlier Analysis: Why We Retain Financial Outliers\n\nIn standard regression, extreme outliers (> Q3 + 1.5*IQR) are frequently truncated. In financial fraud analytics, this causes severe survivorship bias because high-value transactions harbor critical fraud cases."),
    make_cell("code", "outlier_analysis = analyze_outliers(df_dedup)\nfor k, v in outlier_analysis.items():\n    print(f'{k}: {v}')"),
    make_cell("markdown", "### Persisting the Cleaned Dataset"),
    make_cell("code", "CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)\ndf_dedup.to_csv(CLEANED_DATA_PATH, index=False)\nprint(f'Cleaned dataset successfully saved to: {CLEANED_DATA_PATH}')")
]
save_nb("02_data_cleaning.ipynb", nb2)

# 03. Exploratory Data Analysis
nb3 = [
    make_cell("markdown", "# 03. Exploratory Data Analysis (EDA)\n\n## Objectives:\n1. Compare transaction amounts between legitimate and fraudulent transactions.\n2. Explore cyclical transaction activity across hours of the day.\n3. Analyze discriminative power of key PCA features (V14, V17, V12, V4).\n4. Highlight key statistical insights for business stakeholders."),
    make_cell("code", "import sys\nfrom pathlib import Path\nsys.path.append(str(Path.cwd().parent))\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.config import CLEANED_DATA_PATH\n\nplt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\ndf = pd.read_csv(CLEANED_DATA_PATH)\nprint(f'Loaded {len(df):,} cleaned records.')"),
    make_cell("markdown", "### Transaction Amount Distribution (Log-Scale Contrast)"),
    make_cell("code", "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\nsns.histplot(df[df['Class'] == 0]['Amount'], bins=50, kde=True, ax=axes[0], color='#10B981', log_scale=True)\naxes[0].set_title('Legitimate Transactions (Amount)', fontweight='bold')\naxes[0].set_xlabel('Amount ($)')\n\nsns.histplot(df[df['Class'] == 1]['Amount'], bins=50, kde=True, ax=axes[1], color='#EF4444', log_scale=True)\naxes[1].set_title('Fraudulent Transactions (Amount)', fontweight='bold')\naxes[1].set_xlabel('Amount ($)')\nplt.tight_layout()\nplt.show()"),
    make_cell("markdown", "### Temporal Fraud Distribution by Hour of Day"),
    make_cell("code", "df['hour'] = ((df['Time'] // 3600) % 24).astype(int)\nhourly = df.groupby('hour')['Class'].agg(['count', 'sum']).reset_index()\nhourly.columns = ['Hour', 'Total', 'Fraud']\nhourly['Fraud_Rate_Pct'] = (hourly['Fraud'] / hourly['Total']) * 100\n\nplt.figure(figsize=(12, 4.5))\nsns.barplot(data=hourly, x='Hour', y='Fraud_Rate_Pct', color='#2563EB')\nplt.title('Fraud Rate (%) Across 24-Hour Diurnal Cycle', fontweight='bold')\nplt.ylabel('Fraud Rate (%)')\nplt.xlabel('Hour of Day (0 - 23)')\nplt.tight_layout()\nplt.show()"),
    make_cell("markdown", "### Top Discriminative PCA Components (V14, V17, V12, V4)"),
    make_cell("code", "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\nfeats = ['V14', 'V17', 'V12', 'V4']\nfor idx, f in enumerate(feats):\n    ax = axes[idx // 2, idx % 2]\n    sns.kdeplot(df[df['Class'] == 0][f], label='Legit (0)', color='#10B981', fill=True, alpha=0.3, ax=ax)\n    sns.kdeplot(df[df['Class'] == 1][f], label='Fraud (1)', color='#EF4444', fill=True, alpha=0.3, ax=ax)\n    ax.set_title(f'Distribution of {f}', fontweight='bold')\n    ax.legend()\nplt.tight_layout()\nplt.show()")
]
save_nb("03_eda.ipynb", nb3)

# 04. Feature Engineering
nb4 = [
    make_cell("markdown", "# 04. Feature Engineering & Leakage Prevention\n\n## Objectives:\n1. Extract temporal cyclical features (`sin_hour`, `cos_hour`).\n2. Temper right-skewed amounts via `log_amount = log1p(Amount)`.\n3. Calculate relative amount ratio relative to baseline training median.\n4. Encapsulate transformations in a Scikit-Learn compatible transformer to guarantee zero data leakage."),
    make_cell("code", "import sys\nfrom pathlib import Path\nsys.path.append(str(Path.cwd().parent))\n\nimport pandas as pd\nimport numpy as np\nfrom src.config import CLEANED_DATA_PATH\nfrom src.feature_engineering import FraudFeatureEngineer\n\ndf = pd.read_csv(CLEANED_DATA_PATH)\nfe = FraudFeatureEngineer()\nfe.fit(df)\ndf_fe = fe.transform(df)\n\nprint('Engineered features created:', [c for c in df_fe.columns if c not in df.columns])\ndf_fe[['Time', 'hour_of_day', 'sin_hour', 'cos_hour', 'Amount', 'log_amount', 'amount_ratio']].head()"),
    make_cell("markdown", "### Mathematical Verification of Cyclical Continuity"),
    make_cell("code", "import matplotlib.pyplot as plt\nplt.figure(figsize=(6, 6))\nplt.scatter(df_fe['sin_hour'].iloc[:600], df_fe['cos_hour'].iloc[:600], color='#2563EB', alpha=0.5)\nplt.title('Cyclical 24-Hour Projection (sin_hour vs cos_hour)', fontweight='bold')\nplt.xlabel('sin(2 * pi * hour / 24)')\nplt.ylabel('cos(2 * pi * hour / 24)')\nplt.grid(True)\nplt.show()")
]
save_nb("04_feature_engineering.ipynb", nb4)

# 05. Model Training
nb5 = [
    make_cell("markdown", "# 05. Model Training & Class Imbalance Benchmark\n\n## Objectives:\n1. Stratified 80/20 train/test split preserving rare fraud prevalence.\n2. RobustScaler fitted strictly on training data.\n3. Demonstrate the severe danger of raw Accuracy on imbalanced data.\n4. Address class imbalance via SMOTE on the training split only.\n5. Train and benchmark 4 algorithms:\n   - Logistic Regression\n   - Decision Tree\n   - Random Forest\n   - XGBoost\n6. Save optimal model and pipeline artifacts."),
    make_cell("code", "import sys\nfrom pathlib import Path\nsys.path.append(str(Path.cwd().parent))\n\nimport pandas as pd\nfrom src.config import CLEANED_DATA_PATH\nfrom src.preprocessing import prepare_data_pipeline, apply_smote\nfrom src.train_model import train_and_compare_models, save_model_artifacts\n\ndf = pd.read_csv(CLEANED_DATA_PATH)\nX_train, X_test, y_train, y_test, artifact = prepare_data_pipeline(df)\nprint(f'Train split: {len(X_train):,} rows | Test split: {len(X_test):,} rows')"),
    make_cell("markdown", "### Applying SMOTE Strictly to Training Split\n\n**CRITICAL RULE**: SMOTE must NEVER be applied to test data. Doing so synthetically inflates evaluation scores."),
    make_cell("code", "X_train_res, y_train_res = apply_smote(X_train, y_train, sampling_strategy=0.10)\nprint(f'Resampled Train: {len(X_train_res):,} samples (Fraud Ratio: {y_train_res.mean()*100:.2f}%)')"),
    make_cell("markdown", "### Model Benchmark Comparison"),
    make_cell("code", "comparison_df, best_model, best_name = train_and_compare_models(X_train_res, y_train_res, X_test, y_test)\ncomparison_df")
]
save_nb("05_model_training.ipynb", nb5)

# 06. Model Evaluation
nb6 = [
    make_cell("markdown", "# 06. Model Evaluation, Threshold Optimization & Explainable AI\n\n## Objectives:\n1. Benchmark Precision, Recall, F1, ROC-AUC, and PR-AUC on holdout test data.\n2. Plot Confusion Matrix, ROC curve, and Precision-Recall curve.\n3. Business Threshold Tuning: examine thresholds [0.20 – 0.70] and analyze trade-offs between customer friction (FP) and fraud loss (FN).\n4. Explainable AI with SHAP: global importance and transaction-level attribution."),
    make_cell("code", "import sys\nfrom pathlib import Path\nsys.path.append(str(Path.cwd().parent))\n\nimport pandas as pd\nimport numpy as np\nimport joblib\nimport matplotlib.pyplot as plt\nimport shap\nfrom src.config import CLEANED_DATA_PATH, MODEL_PATH, OPTIMAL_THRESHOLD\nfrom src.preprocessing import prepare_data_pipeline\nfrom src.evaluate_model import evaluate_threshold_tradeoffs\n\ndf = pd.read_csv(CLEANED_DATA_PATH)\n_, X_test, _, y_test, _ = prepare_data_pipeline(df)\nmodel = joblib.load(MODEL_PATH)\n\ny_prob = model.predict_proba(X_test)[:, 1]\ny_pred = (y_prob >= OPTIMAL_THRESHOLD).astype(int)\nprint(f'Model loaded. Holdout test set evaluated at threshold: {OPTIMAL_THRESHOLD}')"),
    make_cell("markdown", "### Business Threshold Sensitivity & Trade-off Matrix"),
    make_cell("code", "tradeoffs = evaluate_threshold_tradeoffs(y_test, y_prob)\ntradeoffs"),
    make_cell("markdown", "### Explainable AI (SHAP Summary Plot)"),
    make_cell("code", "explainer = shap.TreeExplainer(model)\nsample = X_test.sample(n=300, random_state=42)\nshap_values = explainer.shap_values(sample)\n\nplt.figure(figsize=(9, 6))\nshap.summary_plot(shap_values, sample, show=False)\nplt.title('SHAP Feature Importance (Fraud Attributions)', fontweight='bold')\nplt.tight_layout()\nplt.show()")
]
save_nb("06_model_evaluation.ipynb", nb6)

print("All 6 Jupyter Notebooks successfully generated.")

