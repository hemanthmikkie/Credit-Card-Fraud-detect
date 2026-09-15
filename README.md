# Credit Card Fraud Detection & Risk Intelligence System

A production-grade, end-to-end Machine Learning, Risk Intelligence, and Surveillance Platform for real-time credit card fraud detection. Built for financial enterprise scale and placement interview excellence.

![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Framework](https://img.shields.io/badge/FastAPI-0.128-green.svg)
![ML Engine](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)
![Database](https://img.shields.io/badge/PostgreSQL%20%7C%20SQLite-Compatible-blue.svg)
![BI](https://img.shields.io/badge/Power%20BI-Ready-yellow.svg)
![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passing-brightgreen.svg)

---

## 1. Overview

The **Credit Card Fraud Detection & Risk Intelligence System** is an enterprise-ready risk surveillance solution designed to detect fraudulent credit card transactions in real time while minimizing false alarms on legitimate cardholders.

Beyond standard classification, the system introduces a **Risk Intelligence Framework**:
- Calibrated posterior **Fraud Probability**
- Normalized **Fraud Risk Score (0–100)**
- Dynamic **Risk Level Bands** (`LOW`, `MEDIUM`, `HIGH`)
- **Explainable AI (XAI)** attributions powered by **SHAP** TreeExplainer
- Sub-10ms **FastAPI REST API** with Pydantic V2 validation
- Persistent transaction auditing in **PostgreSQL / SQLAlchemy**
- Executive & analytical business intelligence via **Power BI**

---

## 2. Business Problem

Credit card issuers process billions of dollars across millions of transactions daily. However, fraudulent transactions represent a minute fraction (~0.17%) of all transactions. 

Financial organizations face a dual-sided financial optimization dilemma:
1. **False Negatives (Missed Fraud)**: Financial write-offs, chargeback recovery costs, merchant dispute fees, and regulatory penalties.
2. **False Positives (Customer Insult)**: Freezing legitimate cards, causing customer embarrassment, lost transaction fees, and high call-center investigation costs.

This platform solves the dilemma through **cost-sensitive learning**, **SMOTE-balanced training**, **threshold optimization**, and **explainable triage queues**.

---

## 3. Project Objectives

- Ingest and profile 284,807 transactions without mutating raw source data.
- Eliminate exact duplicates and assess financial outliers without introducing survivorship bias.
- Engineer non-leaking temporal cyclical (`sin_hour`, `cos_hour`) and non-linear amount features.
- Solve class imbalance strictly on training data using **SMOTE** and **Class Weighting**.
- Benchmark 4 algorithms: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.
- Evaluate models using imbalanced metrics (**PR-AUC**, **ROC-AUC**, **Recall**, **Precision**, **F1**).
- Optimize decision thresholds based on financial trade-offs (0.35 selected).
- Provide real-time inference and transaction auditing via **FastAPI** and **PostgreSQL**.
- Enable visual surveillance and executive monitoring in **Power BI**.

---

## 4. Dataset

- **Source**: Publicly benchmarked European Cardholders Credit Card Fraud Dataset.
- **Records**: 284,807 transactions across 48 hours.
- **Features**: 31 columns:
  - `Time`: Elapsed seconds from first transaction.
  - `V1` – `V28`: Principal Component Analysis (PCA) numerical components (due to confidentiality).
  - `Amount`: Transaction amount in USD.
  - `Class`: Ground truth label (`0` = Legitimate, `1` = Fraudulent).
- **Target Distribution**:
  - Legitimate (`0`): 284,315 (99.827%)
  - Fraudulent (`1`): 492 (0.173%)
  - Imbalance Ratio: ~577 to 1

---

## 5. Technology Stack

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-Learn, Imbalanced-Learn, XGBoost, Joblib |
| **Explainable AI** | SHAP (SHapley Additive exPlanations) |
| **Data Visualization** | Matplotlib, Seaborn |
| **Backend API** | FastAPI, Uvicorn, Pydantic V2, Starlette |
| **Database & ORM** | PostgreSQL, SQLite (graceful fallback), SQLAlchemy 2.0, Psycopg2 |
| **Business Intelligence** | Power BI Desktop, DAX |
| **Testing & Quality** | Pytest, HTTPX, Coverage |
| **Configuration** | Python-Dotenv |

---

## 6. Project Architecture

```
Raw Transaction Feed (284,807 Rows)
         │
         ▼
Data Validation & Cleaning (Schema Check, Deduplication, Outlier Rationale)
         │
         ▼
Exploratory Data Analysis (Amounts, Hourly Diurnal Patterns, PCA KDEs)
         │
         ▼
Stratified Train/Test Split (80% Train / 20% Test, Seed=42)
         │
         ▼
Feature Engineering & RobustScaler (Fitted STRICTLY on Train split - NO LEAKAGE)
         │
         ▼
Imbalance Handling (SMOTE on Train split ONLY)
         │
         ▼
Model Benchmarking (Logistic Regression vs Decision Tree vs Random Forest vs XGBoost)
         │
         ▼
Model Evaluation & Selection (PR-AUC, ROC-AUC, Confusion Matrix, Threshold Tuning)
         │
         ▼
Explainable AI (SHAP TreeExplainer attributions)
         │
         ▼
Model Persistence (fraud_model.pkl, preprocessing.pkl, threshold_config.json)
         │
         ├──────────────────────────────┬──────────────────────────────┐
         ▼                              ▼                              ▼
FastAPI Microservice          PostgreSQL Database              Power BI Dashboard
(/predict, /health)          (transactions table)             (Executive & Surveillance)
```

---

## 7. Project Structure

```
Credit_Card_Fraud_Detection/
│
├── dataset/
│   ├── raw/
│   │   └── creditcard.csv                 # Raw pristine transaction dataset
│   └── processed/
│       └── creditcard_cleaned.csv         # Cleaned and deduplicated dataset
│
├── notebooks/
│   ├── 01_data_understanding.ipynb        # Data profiling and imbalance analysis
│   ├── 02_data_cleaning.ipynb             # Deduplication and outlier investigation
│   ├── 03_eda.ipynb                       # Comprehensive statistical visualizations
│   ├── 04_feature_engineering.ipynb       # Cyclical time & log amount transforms
│   ├── 05_model_training.ipynb            # SMOTE and 4-model benchmark
│   └── 06_model_evaluation.ipynb          # PR-AUC, confusion matrix, and SHAP
│
├── src/
│   ├── __init__.py                        # Package root
│   ├── config.py                          # Centralized paths and risk thresholds
│   ├── data_loader.py                     # Ingestion, validation, profiling
│   ├── data_cleaning.py                   # Deduplication and outlier analysis
│   ├── feature_engineering.py             # Cyclical time and ratio feature transforms
│   ├── preprocessing.py                   # Stratified split, RobustScaler, SMOTE
│   ├── train_model.py                     # Benchmark training and artifact saving
│   ├── evaluate_model.py                  # Metrics, diagnostic plots, threshold tuning
│   └── predict.py                         # Production inference engine with SHAP
│
├── models/
│   ├── fraud_model.pkl                    # Serialized optimal XGBoost classifier
│   ├── preprocessing.pkl                  # Fitted transformers & scalers
│   └── threshold_config.json              # Calibrated decision threshold & risk bands
│
├── api/
│   ├── __init__.py                        # REST service module
│   ├── main.py                            # FastAPI application and routing
│   ├── schemas.py                         # Pydantic V2 input & response schemas
│   └── dependencies.py                    # Database and model singleton injectors
│
├── database/
│   ├── __init__.py                        # DB package
│   ├── database.py                        # Engine configuration with SQLite fallback
│   ├── models.py                          # SQLAlchemy ORM transactions table
│   └── crud.py                            # Persistence and aggregation queries
│
├── sql/
│   ├── schema.sql                         # PostgreSQL DDL with indexes and constraints
│   └── fraud_analysis.sql                 # 13 documented production analytics queries
│
├── dashboard/
│   ├── Fraud_Detection_Dashboard.pbix     # Power BI starter project
│   ├── dax_measures.dax                   # 17 production DAX measures
│   ├── powerbi_transactions_dataset.csv   # Curated scored reporting dataset
│   └── POWER_BI_DASHBOARD_GUIDE.md        # Comprehensive visual design layout guide
│
├── reports/
│   ├── model_comparison.csv               # Empirical benchmark results
│   ├── classification_report.txt          # Holdout test set metrics
│   ├── threshold_tradeoffs.csv            # Threshold sensitivity analysis table
│   ├── confusion_matrix.png               # High-contrast confusion matrix
│   ├── roc_curve.png                      # Receiver Operating Characteristic
│   ├── pr_curve.png                       # Precision-Recall curve
│   └── shap_summary.png                   # Global SHAP feature attributions
│
├── tests/
│   ├── test_data.py                       # Data integrity and cleaning unit tests
│   ├── test_preprocessing.py              # Feature engineering and leakage tests
│   ├── test_predict.py                    # Risk score and inference unit tests
│   └── test_api.py                        # FastAPI integration tests
│
├── requirements.txt                       # Project dependencies
├── .env.example                           # Configuration environment template
├── .gitignore                             # Git ignore rules
├── INTERVIEW_PREP.md                      # Comprehensive placement interview Q&As
└── README.md                              # This documentation
```

---

## 8. Data Cleaning & Outlier Strategy

### Data Profiling Results
- Total rows: 284,807
- Total columns: 31
- Total missing cells: **0**
- Infinite values: **0**
- Duplicate rows: **1,081** (purged to prevent artificial weighting)
- Cleaned dataset: **283,726 rows**

### Why Financial Outliers Are Retained
In standard data cleaning, values beyond $Q_3 + 1.5 \times \text{IQR}$ are often dropped. In this dataset, 31,685 transactions exceed the IQR upper bound of $185.38. Crucially:
- **87 confirmed fraudulent transactions** exist within this outlier subset.
- Truncating these outliers would discard **18.4% of all fraud cases**, causing severe survivorship bias.
- Instead of removing them, we applied `log1p(Amount)` transformation and `RobustScaler` (which uses median and IQR), protecting models from outlier sensitivity without discarding fraud signals.

---

## 9. Exploratory Data Analysis (EDA)

Key statistical findings:
1. **Amount Skewness**: Legitimate transactions have a median of \$22.00, while fraudulent transactions show wide bi-modal dispersion (micro-testing fraud < \$2.00 and high-value bursts > \$1,000).
2. **Temporal Cyclicality**: Transactions recorded across seconds 0 to 172,792 (48 hours). When mapped to `hour_of_day`, fraud occurrence rates surge during early morning hours (2:00 AM – 5:00 AM) when cardholders are asleep.
3. **PCA Discriminative Power**: Features **V14**, **V17**, and **V12** show stark distribution divergence, with fraudulent transactions displaying extreme negative deviations (e.g., $V14 < -5$).

---

## 10. Feature Engineering

1. **`hour_of_day`**: Derived from elapsed seconds:
   $$\text{hour} = \left(\frac{\text{Time}}{3600}\right) \pmod{24}$$
2. **Cyclical Periodic Coordinates**: Ensures continuity across midnight (hour 23 and hour 0 are Euclidean neighbors):
   $$\text{sin\_hour} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{cos\_hour} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$
3. **`log_amount`**: Compresses power-law amount skewness:
   $$\text{log\_amount} = \ln(1 + \text{Amount})$$
4. **`amount_ratio`**: Ratio of transaction amount relative to baseline median learned strictly from the training set.

---

## 11. Imbalanced Data Handling

- Original training split: **226,980 samples**, of which only **378** are fraud (0.167%).
- **Technique**: **SMOTE** (Synthetic Minority Over-sampling Technique) with `sampling_strategy = 0.10`, expanding fraud representation to 22,660 synthetic cases.
- **Strict Leakage Prevention**: SMOTE is applied **strictly to `X_train`**. The holdout test set (56,746 samples with 95 frauds) was kept completely unaltered.

---

## 12. Model Benchmark & Results

All 4 algorithms were evaluated on the **same holdout test split (56,746 transactions)**:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.9992** | **0.7576** | **0.7895** | **0.7732** | **0.9767** | **0.8028** | **2.28s** |
| **Random Forest** | 0.9992 | 0.7308 | 0.8000 | 0.7638 | 0.9843 | 0.7946 | 26.52s |
| **Logistic Regression** | 0.9743 | 0.0542 | 0.8737 | 0.1020 | 0.9614 | 0.6842 | 1.86s |
| **Decision Tree** | 0.9840 | 0.0788 | 0.8000 | 0.1435 | 0.8869 | 0.4667 | 7.93s |

*Note: XGBoost achieved top PR-AUC (0.8028) and superior inference speed, making it the selected production model.*

---

## 13. Threshold Sensitivity & Business Trade-offs

Evaluating probabilities at varying decision cutoffs:

| Threshold | Precision | Recall | F1 Score | Caught Fraud (TP) | False Alarms (FP) | Missed Fraud (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | 0.4246 | 0.8000 | 0.5547 | 76 | 103 | 19 |
| 0.30 | 0.5802 | 0.8000 | 0.6726 | 76 | 55 | 19 |
| **0.35 (Optimal)** | **0.6441** | **0.8000** | **0.7136** | **76** | **42** | **19** |
| 0.40 | 0.6726 | 0.8000 | 0.7308 | 76 | 37 | 19 |
| 0.50 (Default) | 0.7576 | 0.7895 | 0.7732 | 75 | 24 | 20 |
| 0.60 | 0.8242 | 0.7895 | 0.8065 | 75 | 16 | 20 |
| 0.70 | 0.8523 | 0.7895 | 0.8197 | 75 | 13 | 20 |

**Operational Rationale**:  
By shifting threshold from 0.50 to **0.35**, the system captures **76 out of 95 frauds (80.00% Recall)**, stopping an additional high-dollar fraud event with only 42 false alarms across 56,651 legitimate purchases.

---

## 14. Risk Score Formulation

$$\text{Risk Score} = \text{Fraud Probability} \times 100$$

- **`LOW` (0 – 30)**: Transaction approved automatically.
- **`MEDIUM` (31 – 70)**: Step-up authentication triggered (e.g., SMS OTP, 3D-Secure).
- **`HIGH` (71 – 100)**: Transaction held; routed to high-priority investigator triage queue.

---

## 15. Explainable AI (SHAP)

Real-time attributions explain model outputs to investigators:
- **Top Fraud Signals**: Extreme negative deviations on $V14$, $V17$, $V12$, and atypical $Amount$.
- Generated plot: `reports/shap_summary.png`.

---

## 16. FastAPI Production REST Service

### Key Endpoints:
- `GET /`: Service metadata and API routing directory.
- `GET /health`: Model status and database connectivity check.
- `POST /predict`: Real-time transaction risk scoring with SHAP factors.
- `GET /transactions`: Paginated surveillance feed with risk filtering.
- `GET /transactions/{transaction_id}`: Single transaction audit record.
- `GET /fraud-summary`: Aggregate KPI metrics.
- `GET /high-risk-transactions`: Priority triage queue.

### Example Request (`POST /predict`):
```json
{
  "transaction_id": "TX_PROD_1001",
  "Time": 45000.0,
  "Amount": 149.99,
  "V1": -2.312, "V2": 1.951, "V3": -1.609, "V4": 3.997,
  "V5": -0.522, "V6": -1.426, "V7": -2.537, "V8": 1.391,
  "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.899,
  "V13": -0.595, "V14": -4.289, "V15": 0.389, "V16": -1.140,
  "V17": -2.830, "V18": -0.016, "V19": 0.416, "V20": 0.126,
  "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
  "V25": 0.044, "V26": 0.177, "V27": 0.261, "V28": -0.143
}
```

### Example Response:
```json
{
  "transaction_id": "TX_PROD_1001",
  "prediction": "FRAUD",
  "fraud_probability": 0.9412,
  "risk_score": 94.12,
  "risk_level": "HIGH",
  "threshold_applied": 0.35,
  "top_contributing_factors": [
    {"feature": "V14", "impact": "Increases Fraud Risk", "weight": 1.4215},
    {"feature": "V12", "impact": "Increases Fraud Risk", "weight": 0.9841},
    {"feature": "V17", "impact": "Increases Fraud Risk", "weight": 0.8712},
    {"feature": "Amount", "impact": "Increases Fraud Risk", "weight": 0.3245}
  ]
}
```

---

## 17. PostgreSQL Database & SQL Analytics

- Schema defined in `sql/schema.sql`.
- Includes unique index on `transaction_id`, B-tree index on `risk_level`, and composite index on `(created_at, risk_level)`.
- 13 advanced analytics queries in `sql/fraud_analysis.sql` computing:
  - Gross fraud financial exposure
  - Hourly fraud rate
  - Day-over-Day fraud velocity using `LAG()` window functions
  - Risk tier ranking using `DENSE_RANK()`
  - Fraud probability decile calibrations

---

## 18. Power BI Dashboard

Four specialized dashboard pages:
1. **Executive Overview**: High-level KPIs, fraud rate, monetary exposure, and macro distribution.
2. **Fraud Analysis**: 24-hour cycle trends, binned amount distribution, and probability scatter plots.
3. **Model Performance**: Precision, Recall, ROC-AUC, PR-AUC, Confusion Matrix, and ROC/PR curves.
4. **Transaction Monitoring**: Streaming surveillance table with risk level conditional formatting.

Complete DAX measures provided in `dashboard/dax_measures.dax` and visual layout instructions in `dashboard/POWER_BI_DASHBOARD_GUIDE.md`.

---

## 19. Installation & Quickstart (Windows PowerShell)

### Step 1: Clone Repository
```powershell
git clone <YOUR_GITHUB_REPOSITORY>
cd Credit_Card_Fraud_Detection
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```powershell
cp .env.example .env
```
*(Optionally adjust `DATABASE_URL` for PostgreSQL; if PostgreSQL is not active, the system automatically falls back to local SQLite).*

### Step 5: Run Automated Tests
```powershell
python -m pytest tests/ -v
```
*(All 19 tests will execute and pass).*

### Step 6: Start FastAPI REST Service
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
- Open Swagger Documentation: **`http://127.0.0.1:8000/docs`**
- Open ReDoc: **`http://127.0.0.1:8000/redoc`**
- Check Health Endpoint: **`http://127.0.0.1:8000/health`**

---

## 20. Placement Interview Preparation

For in-depth placement interview questions and technical answers covering Python, Statistics, Machine Learning, SQL, FastAPI, PostgreSQL, and Power BI, refer to:  
👉 **[`INTERVIEW_PREP.md`](file:///c:/10k/data%20science%20projects/Credit_Card_Fraud_Detection/INTERVIEW_PREP.md)** (Contains 20+ comprehensive interview Q&As).

---

## License

Distributed under the MIT License. Developed for data science portfolio presentation and enterprise fraud detection benchmarking.
