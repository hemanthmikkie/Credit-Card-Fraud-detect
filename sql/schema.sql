-- ============================================================================
-- CREDIT CARD FRAUD DETECTION & RISK INTELLIGENCE SYSTEM
-- MySQL Schema Definition
-- Database: Credit_Card_Fraud_Detection
-- ============================================================================

-- Create database if not exists (run separately if needed)
CREATE DATABASE IF NOT EXISTS Credit_Card_Fraud_Detection
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE Credit_Card_Fraud_Detection;

-- Drop existing table for clean migration
DROP TABLE IF EXISTS transactions;

-- ----------------------------------------------------------------------------
-- Table: transactions
-- Stores real-time scored and historical transaction audit records.
-- PII Compliance: No PANs, CVVs, or personal identities stored.
-- ----------------------------------------------------------------------------
CREATE TABLE transactions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id  VARCHAR(64)     NOT NULL UNIQUE,
    transaction_time DOUBLE          NOT NULL COMMENT 'Elapsed seconds from reference baseline',
    amount          DOUBLE          NOT NULL COMMENT 'Transaction amount in currency units',
    actual_class    TINYINT         NULL     COMMENT '0 = Legitimate, 1 = Fraudulent, NULL = live prediction',
    predicted_class VARCHAR(20)     NOT NULL COMMENT 'LEGITIMATE or FRAUD',
    fraud_probability DOUBLE        NOT NULL COMMENT 'Model estimated probability [0.0 - 1.0]',
    risk_score      DOUBLE          NOT NULL COMMENT 'Normalized risk score [0 - 100]',
    risk_level      VARCHAR(10)     NOT NULL COMMENT 'LOW, MEDIUM, or HIGH',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'UTC record timestamp',

    -- Check constraints (MySQL 8.0.16+)
    CONSTRAINT check_risk_level     CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    CONSTRAINT check_predicted_class CHECK (predicted_class IN ('LEGITIMATE', 'FRAUD')),
    CONSTRAINT check_prob_range     CHECK (fraud_probability >= 0.0 AND fraud_probability <= 1.0),
    CONSTRAINT check_risk_range     CHECK (risk_score >= 0.0 AND risk_score <= 100.0)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Fraud-scored credit card transaction surveillance store';

-- ----------------------------------------------------------------------------
-- Indexes for High-Performance Fraud Analytics Queries
-- ----------------------------------------------------------------------------
CREATE INDEX idx_risk_level        ON transactions (risk_level);
CREATE INDEX idx_predicted_class   ON transactions (predicted_class);
CREATE INDEX idx_fraud_probability ON transactions (fraud_probability DESC);
CREATE INDEX idx_created_at        ON transactions (created_at DESC);
CREATE INDEX idx_risk_created      ON transactions (risk_level, created_at DESC);
