-- ============================================================================
-- CREDIT CARD FRAUD DETECTION & RISK INTELLIGENCE SYSTEM
-- PostgreSQL Database Schema Definition
-- Database: credit_card_fraud_db
-- ============================================================================

-- Create database (run separately if not exists)
-- CREATE DATABASE credit_card_fraud_db;

-- Drop table if exists during migrations
DROP TABLE IF EXISTS transactions CASCADE;

-- ----------------------------------------------------------------------------
-- Table: transactions
-- Stores both historical ground-truth records and real-time scored API events.
-- PII Compliance: Strictly no credit card PANs, CVVs, or personal identities.
-- ----------------------------------------------------------------------------
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    transaction_time DOUBLE PRECISION NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    actual_class INTEGER CHECK (actual_class IN (0, 1) OR actual_class IS NULL),
    predicted_class VARCHAR(20) NOT NULL CHECK (predicted_class IN ('LEGITIMATE', 'FRAUD')),
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability >= 0.0 AND fraud_probability <= 1.0),
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 100.0),
    risk_level VARCHAR(10) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- ----------------------------------------------------------------------------
-- High-Performance Indexes for Fraud Surveillance & Query Optimization
-- ----------------------------------------------------------------------------
-- Unique lookup index for transaction verification
CREATE UNIQUE INDEX idx_transactions_txid ON transactions (transaction_id);

-- Filter index for rapid real-time high-risk investigation queues
CREATE INDEX idx_transactions_risk_level ON transactions (risk_level);

-- Index on prediction label for operational reporting
CREATE INDEX idx_transactions_predicted_class ON transactions (predicted_class);

-- Composite index for temporal risk analysis and audit range scans
CREATE INDEX idx_transactions_created_risk ON transactions (created_at DESC, risk_level);

-- Index on fraud probability for percentile ranking and top suspicious triage
CREATE INDEX idx_transactions_fraud_prob ON transactions (fraud_probability DESC);

-- Comment metadata
COMMENT ON TABLE transactions IS 'Repository of credit card transactions scored for fraud probability and risk level.';
COMMENT ON COLUMN transactions.transaction_id IS 'Unique idempotency tracking key for the transaction.';
COMMENT ON COLUMN transactions.actual_class IS 'Historical ground truth (0: Legitimate, 1: Fraudulent, NULL: Live prediction).';
COMMENT ON COLUMN transactions.predicted_class IS 'Binary model determination (LEGITIMATE or FRAUD) based on business threshold.';
COMMENT ON COLUMN transactions.fraud_probability IS 'Model estimated posterior probability of fraudulent activity.';
COMMENT ON COLUMN transactions.risk_score IS 'Normalized risk score on scale 0-100 (probability * 100).';
COMMENT ON COLUMN transactions.risk_level IS 'Categorical risk tier: LOW (<=30), MEDIUM (31-70), HIGH (>70).';

