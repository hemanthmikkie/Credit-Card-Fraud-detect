# Power BI Fraud Analytics & Risk Intelligence Dashboard Specification

This document provides step-by-step layout specifications, visual configurations, DAX measure assignments, and color palettes for constructing the **`Fraud_Detection_Dashboard.pbix`** in Power BI Desktop.

---

## 1. Data Source Connection

1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Select `dashboard/powerbi_transactions_dataset.csv`.
4. Verify column types:
   - `Transaction_ID`: Text
   - `Transaction_Time_Seconds`: Decimal Number
   - `Hour_of_Day`: Whole Number
   - `Amount`: Fixed Decimal Number (Currency: `$`)
   - `Actual_Class`: Whole Number
   - `Actual_Label`: Text
   - `Prediction`: Text (`LEGITIMATE` / `FRAUD`)
   - `Fraud_Probability`: Decimal Number (Format: `0.00%` or `0.0000`)
   - `Risk_Score`: Decimal Number (0–100)
   - `Risk_Level`: Text (`LOW`, `MEDIUM`, `HIGH`)
   - `Top_Predictive_Feature`: Text
5. Click **Load**.
6. Create a New Table named `_Measures` and paste the DAX measures defined in `dashboard/dax_measures.dax`.

---

## 2. Color Palette & Styling System

| Element | Color Name | Hex Code | Purpose |
| :--- | :--- | :--- | :--- |
| **High Risk / Fraud** | Crimson Alert | `#EF4444` | Highlight confirmed fraud, high-risk transactions |
| **Medium Risk** | Amber Warning | `#F59E0B` | Suspicious transactions requiring human review |
| **Low Risk / Legit** | Emerald Safe | `#10B981` | Verified legitimate transactions |
| **Primary Theme** | Dark Slate Navy | `#1E293B` | Headers, KPI card borders, text |
| **Accent Blue** | Cobalt Blue | `#2563EB` | Trend lines, primary bar charts |
| **Background** | Clean Off-White | `#F8FAFC` | Canvas background |

---

## 3. Page Layouts & Visual Configurations

### PAGE 1 — Executive Overview
**Objective**: Provide senior risk officers and executives with a high-level summary of fraud volume, monetary exposure, and macro distribution.

#### Top Header & Filters
- **Title**: *Executive Fraud Risk Intelligence & Surveillance*
- **Slicers**:
  - `Risk_Level` (Dropdown / Horizontal Buttons: All, LOW, MEDIUM, HIGH)
  - `Hour_of_Day` (Slider: 0–23)

#### Top KPI Cards (6 Cards)
1. **Total Transactions**: Measure `[Total Transactions]` (Formatted: `#,##0`)
2. **Fraud Transactions**: Measure `[Fraud Transactions]` (Color: Red `#EF4444`)
3. **Legitimate Transactions**: Measure `[Legitimate Transactions]` (Color: Green `#10B981`)
4. **Fraud Rate**: Measure `[Fraud Rate]` (Formatted: `0.00%`)
5. **Fraud Exposure Amount**: Measure `[Fraud Amount]` (Formatted: `$#,##0.00`)
6. **High-Risk Alerts**: Measure `[High Risk Transactions]` (Color: Red `#EF4444`)

#### Main Visuals (2x2 Grid)
1. **Line Chart**: *Hourly Fraud Trend*
   - X-axis: `Hour_of_Day` (0 to 23)
   - Y-axis: Measure `[Fraud Transactions]` and `[Fraud Rate]`
2. **Donut Chart**: *Fraud vs. Legitimate Volume*
   - Legend: `Prediction` (`LEGITIMATE` vs `FRAUD`)
   - Values: `[Total Transactions]`
   - Colors: `#10B981` (Legitimate), `#EF4444` (Fraud)
3. **Clustered Bar Chart**: *Monetary Exposure by Risk Level*
   - Y-axis: `Risk_Level` (`HIGH`, `MEDIUM`, `LOW`)
   - X-axis: `SUM(Amount)`
   - Color: Dynamic based on Risk Level (`HIGH` -> `#EF4444`, `MEDIUM` -> `#F59E0B`, `LOW` -> `#10B981`)
4. **Stacked Column Chart**: *Risk Level Breakdown by Transaction Tier*
   - X-axis: Binned Amount
   - Y-axis: Count of Transactions
   - Legend: `Risk_Level`

---

### PAGE 2 — Fraud Analysis & Deep Dive
**Objective**: Enable fraud analysts to detect behavioral anomalies, cyclical fraud bursts, and high-exposure spikes.

#### Visuals
1. **Area Chart**: *Fraud Rate Across the 24-Hour Cycle*
   - X-axis: `Hour_of_Day`
   - Y-axis: `[Fraud Rate]`
   - Highlight: Late-night spikes (Hours 2 AM – 5 AM)
2. **Histogram / Binned Column Chart**: *Transaction Amount Distribution by Fraud Status*
   - X-axis: Amount Ranges (`$0-$50`, `$50-$200`, `$200-$500`, `$500+`)
   - Y-axis: Count of Fraud Transactions
3. **Density / Scatter Plot**: *Fraud Probability vs Transaction Amount*
   - X-axis: `Amount`
   - Y-axis: `Fraud_Probability`
   - Color: `Risk_Level`
4. **Interactive High-Risk Triage Queue**:
   - Filter: `Risk_Level = "HIGH"`
   - Columns: `Transaction_ID`, `Amount`, `Fraud_Probability`, `Risk_Score`, `Top_Predictive_Feature`
   - Sort: `Fraud_Probability` Descending

---

### PAGE 3 — Model Performance & Evaluation
**Objective**: Demonstrate machine learning rigor, threshold trade-offs, and calibration for placement interviews and model validation auditors.

#### KPI Cards
1. **Model Algorithm**: `XGBoost Classifier + RobustScaler + SMOTE`
2. **Precision**: `75.76%`
3. **Recall (Fraud Coverage)**: `78.95%` (80.00% at optimal 0.35 threshold)
4. **ROC-AUC**: `0.9767`
5. **PR-AUC**: `0.8028`

#### Embedded Images / Custom Visuals
1. **Image Visual**: `reports/confusion_matrix.png`
2. **Image Visual**: `reports/roc_curve.png`
3. **Image Visual**: `reports/pr_curve.png`
4. **Table Visual**: *Algorithm Benchmark Comparison*
   - Data Source: `reports/model_comparison.csv`
   - Columns: `Model`, `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`, `PR-AUC`, `TrainTimeSec`

---

### PAGE 4 — Real-Time Transaction Surveillance
**Objective**: Operational monitoring interface for Tier-1 fraud analysts investigating incoming streaming transactions.

#### Table Visual (Full Width)
- **Columns**:
  - `Transaction_ID`
  - `Transaction_Time_Seconds`
  - `Hour_of_Day`
  - `Amount`
  - `Prediction`
  - `Fraud_Probability`
  - `Risk_Score`
  - `Risk_Level`
  - `Top_Predictive_Feature`
- **Conditional Formatting Rules**:
  - `Risk_Level`:
    - `"HIGH"` -> Background `#FEE2E2`, Font `#991B1B` (Bold)
    - `"MEDIUM"` -> Background `#FEF3C7`, Font `#92400E`
    - `"LOW"` -> Background `#D1FAE5`, Font `#065F46`
  - `Risk_Score`: Data Bars (Red to Green gradient)

