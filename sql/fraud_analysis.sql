-- ============================================================================
-- CREDIT CARD FRAUD DETECTION & RISK INTELLIGENCE SYSTEM
-- Analytical SQL Queries for Fraud Ops, Risk Officers, and Reporting
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Total Transactions Count
-- Computes total volume of all transactions captured in the surveillance store.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_transactions
FROM transactions;


-- ----------------------------------------------------------------------------
-- 2. Fraud Transactions Count
-- Counts total transactions classified as FRAUD by model prediction or ground truth.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_fraud_transactions
FROM transactions
WHERE predicted_class = 'FRAUD';


-- ----------------------------------------------------------------------------
-- 3. Legitimate Transactions Count
-- Counts all normal, legitimate card activities.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_legitimate_transactions
FROM transactions
WHERE predicted_class = 'LEGITIMATE';


-- ----------------------------------------------------------------------------
-- 4. Overall Fraud Rate Percentage
-- Percentage of fraudulent transactions relative to overall transaction throughput.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(
        100.0 * SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 
        4
    ) AS fraud_rate_percentage
FROM transactions;


-- ----------------------------------------------------------------------------
-- 5. Total Fraud Financial Amount (Exposure at Risk)
-- Measures gross monetary exposure in transactions flagged as fraudulent.
-- ----------------------------------------------------------------------------
SELECT 
    ROUND(CAST(SUM(amount) AS NUMERIC), 2) AS total_fraud_exposure_amount
FROM transactions
WHERE predicted_class = 'FRAUD';


-- ----------------------------------------------------------------------------
-- 6. Average Transaction Amount: Legitimate vs Fraudulent
-- Compares average ticket size between legitimate purchases and fraud attempts.
-- ----------------------------------------------------------------------------
SELECT 
    predicted_class,
    COUNT(*) AS transaction_count,
    ROUND(CAST(AVG(amount) AS NUMERIC), 2) AS avg_amount,
    ROUND(CAST(MIN(amount) AS NUMERIC), 2) AS min_amount,
    ROUND(CAST(MAX(amount) AS NUMERIC), 2) AS max_amount,
    ROUND(CAST(STDDEV(amount) AS NUMERIC), 2) AS std_amount
FROM transactions
GROUP BY predicted_class;


-- ----------------------------------------------------------------------------
-- 7. Fraud Analysis by Risk Tier (LOW, MEDIUM, HIGH)
-- Aggregates volume, percentage, and financial exposure across risk bands.
-- ----------------------------------------------------------------------------
SELECT 
    risk_level,
    COUNT(*) AS volume,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total_volume,
    ROUND(CAST(SUM(amount) AS NUMERIC), 2) AS total_amount,
    ROUND(CAST(AVG(fraud_probability) AS NUMERIC), 4) AS avg_fraud_probability,
    SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_count
FROM transactions
GROUP BY risk_level
ORDER BY 
    CASE risk_level 
        WHEN 'HIGH' THEN 1 
        WHEN 'MEDIUM' THEN 2 
        WHEN 'LOW' THEN 3 
        ELSE 4 
    END;


-- ----------------------------------------------------------------------------
-- 8. Fraud by Time of Day (Hour-by-Hour Breakdown)
-- Extracts cyclical hour-of-day pattern from elapsed transaction seconds.
-- ----------------------------------------------------------------------------
SELECT 
    FLOOR((transaction_time / 3600) % 24) AS hour_of_day,
    COUNT(*) AS total_hourly_volume,
    SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        3
    ) AS hourly_fraud_rate_pct,
    ROUND(CAST(SUM(CASE WHEN predicted_class = 'FRAUD' THEN amount ELSE 0 END) AS NUMERIC), 2) AS fraud_amount
FROM transactions
GROUP BY FLOOR((transaction_time / 3600) % 24)
ORDER BY hour_of_day ASC;


-- ----------------------------------------------------------------------------
-- 9. Real-Time High-Risk Transactions (Immediate Triage Queue)
-- Extracts high-risk alerts with amount > $100 for manual investigator review.
-- ----------------------------------------------------------------------------
SELECT 
    transaction_id,
    amount,
    fraud_probability,
    risk_score,
    risk_level,
    created_at
FROM transactions
WHERE risk_level = 'HIGH'
  AND amount > 100.0
ORDER BY fraud_probability DESC, amount DESC
LIMIT 50;


-- ----------------------------------------------------------------------------
-- 10. Daily Fraud Trends & Day-over-Day Velocity (Window Function)
-- Tracks rolling daily volume, fraud totals, and percentage changes.
-- ----------------------------------------------------------------------------
WITH daily_summary AS (
    SELECT 
        DATE_TRUNC('day', created_at) AS tx_date,
        COUNT(*) AS daily_volume,
        SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) AS daily_fraud_count,
        ROUND(CAST(SUM(CASE WHEN predicted_class = 'FRAUD' THEN amount ELSE 0 END) AS NUMERIC), 2) AS daily_fraud_amount
    FROM transactions
    GROUP BY DATE_TRUNC('day', created_at)
)
SELECT 
    tx_date,
    daily_volume,
    daily_fraud_count,
    daily_fraud_amount,
    LAG(daily_fraud_count, 1) OVER (ORDER BY tx_date) AS prev_day_fraud,
    daily_fraud_count - COALESCE(LAG(daily_fraud_count, 1) OVER (ORDER BY tx_date), 0) AS dod_fraud_change
FROM daily_summary
ORDER BY tx_date DESC;


-- ----------------------------------------------------------------------------
-- 11. Monthly Fraud Trends and Aggregate Exposure
-- Aggregates monthly transaction counts and total fraud monetary loss.
-- ----------------------------------------------------------------------------
SELECT 
    TO_CHAR(created_at, 'YYYY-MM') AS report_month,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        4
    ) AS fraud_incidence_rate,
    ROUND(CAST(SUM(CASE WHEN predicted_class = 'FRAUD' THEN amount ELSE 0 END) AS NUMERIC), 2) AS total_fraud_amount
FROM transactions
GROUP BY TO_CHAR(created_at, 'YYYY-MM')
ORDER BY report_month DESC;


-- ----------------------------------------------------------------------------
-- 12. Top Suspicious Transactions per Risk Band (DENSE_RANK Window Function)
-- Uses window functions to rank transactions within their respective risk tiers.
-- ----------------------------------------------------------------------------
WITH ranked_transactions AS (
    SELECT 
        transaction_id,
        amount,
        risk_level,
        fraud_probability,
        risk_score,
        created_at,
        DENSE_RANK() OVER (
            PARTITION BY risk_level 
            ORDER BY fraud_probability DESC, amount DESC
        ) AS tier_rank
    FROM transactions
)
SELECT 
    risk_level,
    tier_rank,
    transaction_id,
    amount,
    fraud_probability,
    risk_score,
    created_at
FROM ranked_transactions
WHERE tier_rank <= 10
ORDER BY 
    CASE risk_level 
        WHEN 'HIGH' THEN 1 
        WHEN 'MEDIUM' THEN 2 
        ELSE 3 
    END, 
    tier_rank ASC;


-- ----------------------------------------------------------------------------
-- 13. Fraud Probability Distribution by Decile Bins
-- Discretizes model probabilities into decile bands to inspect calibration.
-- ----------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN fraud_probability < 0.10 THEN '0.00 - 0.10'
        WHEN fraud_probability < 0.20 THEN '0.10 - 0.20'
        WHEN fraud_probability < 0.30 THEN '0.20 - 0.30'
        WHEN fraud_probability < 0.40 THEN '0.30 - 0.40'
        WHEN fraud_probability < 0.50 THEN '0.40 - 0.50'
        WHEN fraud_probability < 0.60 THEN '0.50 - 0.60'
        WHEN fraud_probability < 0.70 THEN '0.60 - 0.70'
        WHEN fraud_probability < 0.80 THEN '0.70 - 0.80'
        WHEN fraud_probability < 0.90 THEN '0.80 - 0.90'
        ELSE '0.90 - 1.00'
    END AS probability_bin,
    COUNT(*) AS transaction_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
    SUM(CASE WHEN predicted_class = 'FRAUD' THEN 1 ELSE 0 END) AS confirmed_fraud_count,
    ROUND(CAST(AVG(amount) AS NUMERIC), 2) AS avg_amount
FROM transactions
GROUP BY 
    CASE 
        WHEN fraud_probability < 0.10 THEN '0.00 - 0.10'
        WHEN fraud_probability < 0.20 THEN '0.10 - 0.20'
        WHEN fraud_probability < 0.30 THEN '0.20 - 0.30'
        WHEN fraud_probability < 0.40 THEN '0.30 - 0.40'
        WHEN fraud_probability < 0.50 THEN '0.40 - 0.50'
        WHEN fraud_probability < 0.60 THEN '0.50 - 0.60'
        WHEN fraud_probability < 0.70 THEN '0.60 - 0.70'
        WHEN fraud_probability < 0.80 THEN '0.70 - 0.80'
        WHEN fraud_probability < 0.90 THEN '0.80 - 0.90'
        ELSE '0.90 - 1.00'
    END
ORDER BY MIN(fraud_probability) ASC;

