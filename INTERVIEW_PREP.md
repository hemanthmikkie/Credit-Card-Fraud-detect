# Credit Card Fraud Detection & Risk Intelligence System
## Master Placement & Senior Data Science Interview Preparation Guide

This guide compiles **comprehensive, senior-level interview questions and interviewer-calibrated responses** across Python, Statistics, Machine Learning, SQL, FastAPI, PostgreSQL, Power BI, and project architecture.

---

## 1. Python & Data Engineering

### Q1: Why use Pandas and NumPy instead of native Python lists and dictionaries?
**Answer:**  
Native Python objects incur significant memory overhead due to dynamic typing (pointer indirection and Python object headers) and execute in interpreted loops that cannot be vectorized by the CPU.  
- **NumPy** allocates contiguous blocks of C-level memory and leverages SIMD (Single Instruction, Multiple Data) processor instructions, enabling element-wise array operations orders of magnitude faster.  
- **Pandas** builds on NumPy arrays to provide columnar Series and DataFrames with indexing, alignment, categorical encoding, and vectorized string/datetime manipulation. In our 284,807-row fraud dataset, Pandas performed full-table deduplication and aggregation in under 1.5 seconds.

### Q2: How did you detect and handle duplicates, and what do duplicates mean in financial transactions?
**Answer:**  
We used `df.duplicated().sum()` to find 1,081 identical records across all 30 features. In credit card networks, exact duplicates typically represent idempotent retry bursts (a terminal or payment gateway retrying an unconfirmed network request) or multi-region database logging synchronization anomalies. We purged exact duplicates using `df.drop_duplicates().reset_index(drop=True)` to prevent the model from learning inflated weights for repeated transactions.

### Q3: How did you prevent data leakage during feature engineering and scaling?
**Answer:**  
Data leakage occurs when information from outside the training split (such as test set statistics) contaminates the model training process.  
We prevented leakage by:
1. Conducting stratified train/test splitting (80% train, 20% test) **first**.
2. Fitting our custom `FraudFeatureEngineer` (which computes historical median amounts) **strictly on the training split**.
3. Fitting the `RobustScaler` **strictly on `X_train`**, and using the fitted transformer to project `X_test` and real-time incoming API requests.
4. Applying SMOTE **only to `X_train`**; the test set was left untouched in its natural imbalanced distribution.

---

## 2. Statistics & Imbalanced Data Metrics

### Q4: Why is Accuracy misleading for imbalanced classification?
**Answer:**  
In our dataset, fraud prevalence is **0.167%** (only 378 frauds in 226,980 training rows). A naive dummy classifier that predicts `Class = 0` (Legitimate) for every transaction achieves **99.833% Accuracy**, yet captures **0% of fraudulent attacks**, resulting in catastrophic financial loss. Accuracy computes $\frac{TP + TN}{TP + TN + FP + FN}$. When $TN \gg TP$, the metric is dominated by True Negatives, obscuring model failure on the minority positive class.

### Q5: What is the difference between Precision and Recall, and which is more important in fraud detection?
**Answer:**  
- **Precision** $= \frac{TP}{TP + FP}$: Out of all transactions flagged as fraud, what proportion was actually fraudulent? Low precision causes false alarms, freezing innocent customer cards (customer friction/insult).
- **Recall (Sensitivity)** $= \frac{TP}{TP + FN}$: Out of all actual fraud events, what proportion did our system detect? Low recall means fraudulent charges slip through, causing direct chargeback write-offs.
- In financial risk management, **Recall is prioritized** to protect financial capital, but Precision must remain above acceptable thresholds (e.g., > 60-75%) so fraud investigation teams are not overwhelmed by false alerts.

### Q6: What is the F1-score and when does it fall short?
**Answer:**  
F1 is the harmonic mean of Precision and Recall:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It penalizes extreme imbalances between the two. However, F1 weights Precision and Recall equally. When a business values Recall over Precision, the generalized $F_\beta$ score (e.g., $F_2$, where recall is weighted twice as heavily as precision) is a more appropriate statistical criterion.

### Q7: Why is PR-AUC (Precision-Recall AUC) superior to ROC-AUC for fraud detection?
**Answer:**  
ROC-AUC evaluates False Positive Rate (FPR) on the x-axis: $\text{FPR} = \frac{FP}{FP + TN}$. Because $TN$ is massive (~56,650 in our test split), even hundreds of False Positives produce an infinitesimal FPR, keeping the ROC curve close to 1.0 (e.g., 0.984).  
Conversely, the Precision-Recall curve plots Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{TP + FN}$). Because $TN$ is absent from both Precision and Recall, the PR curve directly isolates the minority class performance and penalizes false positives heavily, providing an unvarnished view of model capability.

---

## 3. Machine Learning & Model Architecture

### Q8: Why compare Logistic Regression, Decision Tree, Random Forest, and XGBoost?
**Answer:**  
Benchmarking multiple algorithm families establishes an empirical baseline:
1. **Logistic Regression (Linear)**: Fast, interpretable baseline. With `class_weight='balanced'`, it yielded 87.37% recall but poor precision (5.42%) due to linear decision boundaries unable to separate intricate multivariate PCA manifolds.
2. **Decision Tree (Non-linear single estimator)**: Prone to high variance and overfitting on local data pockets, achieving only 7.88% precision and 0.4667 PR-AUC.
3. **Random Forest (Bagging Ensemble)**: Reduces variance by averaging 100 decorrelated deep trees trained on bootstrap samples. Achieved 73.08% precision, 80.00% recall, and 0.7946 PR-AUC.
4. **XGBoost (Gradient Boosting)**: Sequentially trains shallow trees to minimize the gradient of the loss function, using second-order Taylor expansions and regularization. Achieved top performance with **75.76% precision, 78.95% recall (80.00% at optimal 0.35 threshold), and 0.8028 PR-AUC** in just 2.28 seconds training time.

### Q9: What is SMOTE and why must it NEVER be applied to test data?
**Answer:**  
**SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic minority instances along the line segments joining the $k$ nearest neighbors of minority class samples.  
**Critical Rule**: Applying SMOTE to the test dataset is a fatal methodological error because:
1. Test data must reflect the real-world production prior distribution ($\sim 0.17\%$ fraud).
2. Synthesizing test points introduces artificial samples that the model was partially trained to recognize, falsely inflating test scores.
3. It distorts the true base rate, invalidating Precision and PR-AUC.

### Q10: How did you select the optimal decision threshold rather than defaulting to 0.50?
**Answer:**  
Default 0.50 probability assumes equal misclassification costs ($\text{Cost}(FP) = \text{Cost}(FN)$). In fraud, the cost of a False Negative (e.g., losing a \$1,500 unauthorized transaction) is substantially higher than the cost of a False Positive (e.g., sending an automated SMS confirmation costing \$0.02).  
We ran a threshold sensitivity analysis from 0.20 to 0.70:
- At **0.50**: 75 caught fraud, 24 false alarms, 20 missed frauds.
- At **0.35**: 76 caught fraud, 42 false alarms, 19 missed frauds (catching an extra fraud with manageable operational overhead).
- Thus, 0.35 was configured as the operational decision threshold.

### Q11: How does Explainable AI (SHAP) work in your pipeline?
**Answer:**  
We integrated `shap.TreeExplainer`, which computes exact **Shapley values** from cooperative game theory. For each scored transaction, the TreeExplainer calculates each feature's marginal contribution to pushing the log-odds away from the base value towards fraud. We return the top 4 contributing features (e.g., V14, V17, Amount) in the API response, explaining model attributions to fraud investigators without making unverified causal claims.

---

## 4. SQL & Database Architecture

### Q12: Why use PostgreSQL for the transactional store?
**Answer:**  
PostgreSQL offers full ACID (Atomicity, Consistency, Isolation, Durability) guarantees, strong typing, check constraints, robust concurrent write handling (MVCC), and JSONB capabilities. In financial surveillance, we must guarantee that once an alert is recorded, it is never lost or partially written.

### Q13: Explain your index strategy on the `transactions` table.
**Answer:**  
1. **Unique B-Tree Index on `transaction_id`**: Guarantees fast $O(\log N)$ idempotent lookups and prevents duplicate scoring.
2. **Index on `risk_level` and `predicted_class`**: Powers real-time dashboard filtering where queries frequently filter `WHERE risk_level = 'HIGH'`.
3. **Composite Index on `(created_at DESC, risk_level)`**: Optimizes time-window audits and day-over-day trend queries without full table scans.
4. **Index on `fraud_probability DESC`**: Accelerates top-N suspicious transaction ranking queries for investigator triage queues.

### Q14: Explain the use of window functions like `LAG` and `DENSE_RANK` in your SQL analytics.
**Answer:**  
- **`LAG(daily_fraud_count, 1) OVER (ORDER BY tx_date)`**: Fetches the previous day's fraud count in a single pass without expensive self-joins, enabling immediate Day-over-Day (DoD) delta calculations.
- **`DENSE_RANK() OVER (PARTITION BY risk_level ORDER BY fraud_probability DESC, amount DESC)`**: Assigns ranks within each risk tier so fraud operations can extract the top 10 most critical transactions per risk category.

---

## 5. FastAPI & Production REST APIs

### Q15: Why FastAPI over Flask or Django?
**Answer:**  
1. **High Performance**: Built on Starlette and Uvicorn, FastAPI utilizes Python `asyncio` for non-blocking asynchronous I/O, matching Node.js/Go speeds.
2. **Data Validation via Pydantic**: Eliminates manual request parsing; validates types, ranges (e.g. `Amount >= 0.0`), and returns automatic 422 errors for malformed requests.
3. **Automated Documentation**: Generates interactive OpenAPI (Swagger UI) at `/docs` and ReDoc at `/redoc` out of the box.
4. **Dependency Injection System**: Allows clean injection of database sessions and ML predictor singletons.

### Q16: How do you prevent model reloading latency on every API request?
**Answer:**  
We implemented a **Singleton Pattern** via FastAPI's `lifespan` context manager. On server startup, `FraudPredictor` loads the model (`fraud_model.pkl`), preprocessing pipeline (`preprocessing.pkl`), and SHAP TreeExplainer into memory once. Incoming requests reuse the warmed-up pipeline, enabling sub-10ms response times.

---

## 6. Power BI & Business Intelligence

### Q17: What is the difference between Filter Context and Row Context in DAX?
**Answer:**  
- **Row Context**: Exists during calculated columns or row iterators (like `SUMX`). It refers to the current row being evaluated.
- **Filter Context**: The set of active filters applied to the data model via visual elements, slicers, cross-filtering, and `CALCULATE` statements.  
In our dashboard, measures like `[Fraud Transactions] = CALCULATE(COUNTROWS('Transactions'), 'Transactions'[Prediction] = "FRAUD")` modify the existing filter context to isolate fraudulent transactions regardless of outer table filtering.

### Q18: Why use `DIVIDE` in DAX instead of `/`?
**Answer:**  
The `/` operator in DAX throws an error or returns infinity on division by zero. The `DIVIDE(Numerator, Denominator, AlternateResult)` function performs safe division, returning 0 or an alternate value if the denominator is 0 or blank, preventing dashboard card crashes.

---

## 7. 20 Project-Specific Placement Interview Questions & Answers

1. **Why did you transform Amount using `log1p`?**  
   *Answer*: Transaction amounts follow an extreme Pareto/power-law distribution with high skewness. `log1p(x) = ln(1 + x)` handles $0 transactions safely and compresses the dynamic range, preventing large outliers from dominating gradient calculations in distance- and linear-based estimators.

2. **Why encode hour of day cyclically with sine and cosine?**  
   *Answer*: Numerical hours 23 (11 PM) and 0 (midnight) have a Euclidean difference of 23, but temporally they are only 1 hour apart. Transforming into $(\sin(2\pi h/24), \cos(2\pi h/24))$ creates continuous circular coordinates where 23:00 and 00:00 are adjacent in feature space.

3. **Why did you use `RobustScaler` instead of `StandardScaler`?**  
   *Answer*: `StandardScaler` uses sample mean and standard deviation, which are heavily distorted by extreme financial transaction amounts. `RobustScaler` subtracts the median and divides by the Interquartile Range (IQR, 25th–75th percentiles), ensuring scaling is unaffected by extreme outliers.

4. **What was the exact class imbalance ratio?**  
   *Answer*: 283,726 transactions after deduplication had 473 fraud cases (~0.167%), or approximately 1 fraud in every 600 transactions.

5. **Why not undersample the majority class instead of SMOTE?**  
   *Answer*: Random undersampling to 1:1 ratio would discard >99% of our legitimate transactions (over 225,000 data points), destroying valuable customer behavioral patterns. SMOTE with a conservative 10% sampling strategy boosted minority representation without sacrificing majority variance.

6. **What were the top predictive features identified by SHAP?**  
   *Answer*: V14, V17, V12, and V10 consistently showed the strongest negative attribution towards fraud (extreme negative values indicating high fraud probability), while V4 and Amount showed positive attribution.

7. **How does the system translate probability into a Risk Score?**  
   *Answer*: $\text{Risk Score} = \text{Fraud Probability} \times 100$, mapped to 3 configurable bands: LOW (0–30), MEDIUM (31–70), and HIGH (71–100).

8. **How does the API handle database disconnections?**  
   *Answer*: The database engine tests connectivity on startup. If the primary PostgreSQL instance is unreachable, it logs a warning and falls back to a local SQLite database, maintaining 100% service uptime for local evaluations and test runners.

9. **How did you prevent SQL injection in your application?**  
   *Answer*: All database queries use SQLAlchemy ORM with parameterized binding, completely preventing string concatenation and SQL injection vulnerabilities.

10. **How are real-time API transactions validated against malformed data?**  
    *Answer*: Pydantic V2 schemas enforce strict types, valid numeric intervals ($Time \ge 0, Amount \ge 0$), and ensure all 28 PCA components are present before invoking the ML pipeline.

11. **What is the difference between `test_size=0.20` with and without stratification?**  
    *Answer*: Without stratification, random splitting could result in a test set with very few or zero fraud cases by pure chance. `stratify=y` guarantees that both training and testing splits maintain the exact 0.167% fraud prevalence.

12. **What would you do if model inference latency exceeded 200ms?**  
    *Answer*: We would precompute and cache SHAP background summaries, convert the XGBoost model to ONNX runtime or Treelite for compiled C++ inference, and deploy with multi-worker Gunicorn/Uvicorn processes.

13. **How does your system handle transactions with negative amounts?**  
    *Answer*: Both data cleaning and Pydantic validation reject negative amounts as data entry or schema violations with an explicit 422 HTTP error response.

14. **Why is `scale_pos_weight` used in XGBoost?**  
    *Answer*: `scale_pos_weight` specifies the ratio of negative to positive samples, scaling the gradient on positive instances. Since we pre-balanced our training set to 10% with SMOTE, we calibrated `scale_pos_weight` accordingly.

15. **What is the business impact of selecting threshold 0.35 over 0.50?**  
    *Answer*: In our holdout test set of 56,746 transactions, 0.35 increased fraud detection from 75 to 76 caught cases while keeping false alarms to 42 across 56,651 legitimate purchases.

16. **How would you monitor model drift in production?**  
    *Answer*: Track Population Stability Index (PSI) on incoming features weekly, monitor rolling fraud rate and risk level distribution via `/fraud-summary`, and trigger automated retraining alerts if PSI exceeds 0.25.

17. **What is the role of the `id` vs `transaction_id` columns in PostgreSQL?**  
    *Answer*: `id` is a synthetic serial integer primary key optimized for fast internal B-tree indexing and joins; `transaction_id` is a client-provided alphanumeric tracking key indexed with a UNIQUE constraint.

18. **How did you test the system?**  
    *Answer*: We implemented 19 automated tests using `pytest` covering data validation, non-leakage preprocessing, SMOTE isolation, inference logic, and FastAPI endpoints via `TestClient`.

19. **How would you scale this architecture to 10,000 transactions per second?**  
    *Answer*: Decouple ingestion from scoring using an event streaming bus (Apache Kafka), run stateless FastAPI scoring workers behind a Kubernetes Load Balancer, and write transaction logs asynchronously via Kafka consumers to PostgreSQL read-replicas or ClickHouse.

20. **What is the most significant lesson learned from this project?**  
    *Answer*: Model performance in fraud detection is dictated more by data hygiene, leak-free preprocessing, and imbalanced metric selection (PR-AUC over ROC-AUC) than by simply using more complex algorithms.

