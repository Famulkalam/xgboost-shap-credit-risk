/*
================================================================
Schema for the credit-risk project (Microsoft SQL Server)
Run this once, then run push_to_sql.py to load the data,
then run create_views.sql.
================================================================
*/

-- CREATE DATABASE CreditRiskDB;
-- GO
-- USE CreditRiskDB;
-- GO

IF OBJECT_ID('dbo.vw_LoanRiskSummary', 'V') IS NOT NULL DROP VIEW dbo.vw_LoanRiskSummary;
IF OBJECT_ID('dbo.LoanPredictions', 'U') IS NOT NULL DROP TABLE dbo.LoanPredictions;
IF OBJECT_ID('dbo.Loans', 'U') IS NOT NULL DROP TABLE dbo.Loans;
GO

-- One row per loan in the raw Kaggle file. loan_id is the 1-based row
-- number assigned in data_prep.load_raw(), so it matches predictions.csv.
CREATE TABLE dbo.Loans (
    loan_id INT NOT NULL PRIMARY KEY,
    person_age INT NOT NULL,
    person_income INT NOT NULL,
    person_home_ownership VARCHAR(20),
    person_emp_length FLOAT NULL,
    loan_intent VARCHAR(30),
    loan_grade CHAR(1),
    loan_amnt INT NOT NULL,
    loan_int_rate FLOAT NULL,
    loan_status INT NOT NULL,            -- 1 = defaulted, 0 = repaid
    loan_percent_income FLOAT,
    cb_person_default_on_file CHAR(1),
    cb_person_cred_hist_length INT
);

-- Model output for the held-out test set (one row per test loan).
CREATE TABLE dbo.LoanPredictions (
    loan_id INT NOT NULL PRIMARY KEY,
    probability_default FLOAT NOT NULL,  -- XGBoost PD
    scorecard_pd FLOAT NOT NULL,         -- WoE logistic scorecard PD
    scorecard_score INT NOT NULL,        -- points; higher = lower risk
    risk_band VARCHAR(10) NOT NULL,      -- Low < 10% <= Medium < 30% <= High
    is_high_risk BIT NOT NULL,
    created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT FK_Predictions_Loans FOREIGN KEY (loan_id) REFERENCES dbo.Loans(loan_id)
);
GO
