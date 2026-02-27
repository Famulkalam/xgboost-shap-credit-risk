/*
================================================================
TASK: Design an MS SQL schema for the Credit Risk Project
PROJECT: XGBoost-SHAP Credit Risk Scorecard
================================================================
*/

-- 1. Create the Database (Run this separately if needed)
-- CREATE DATABASE CreditRiskDB;
-- GO
-- USE CreditRiskDB;
-- GO

-- 2. Create the Clean Fact Table
IF OBJECT_ID('dbo.Loans', 'U') IS NOT NULL DROP TABLE dbo.Loans;
CREATE TABLE dbo.Loans (
    loan_id INT IDENTITY(1,1) PRIMARY KEY, -- Internal unique ID
    person_age INT NOT NULL,
    person_income INT NOT NULL,
    person_home_ownership VARCHAR(50),
    person_emp_length FLOAT,
    loan_intent VARCHAR(100),
    loan_grade CHAR(1),
    loan_amnt INT NOT NULL,
    loan_int_rate FLOAT,
    loan_status INT NOT NULL, -- Target: 0 (Good), 1 (Bad)
    loan_percent_income FLOAT,
    cb_person_default_on_file CHAR(1),
    cb_person_cred_hist_length INT
);

-- 3. Create the Predictions Table
IF OBJECT_ID('dbo.LoanPredictions', 'U') IS NOT NULL DROP TABLE dbo.LoanPredictions;
CREATE TABLE dbo.LoanPredictions (
    prediction_id INT IDENTITY(1,1) PRIMARY KEY,
    loan_id INT NOT NULL, -- Foreign key to dbo.Loans
    probability_default FLOAT NOT NULL,
    risk_band VARCHAR(20), -- Low, Medium, High
    is_high_risk BIT, -- 1 if probability > threshold
    created_at DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_Predictions_Loans FOREIGN KEY (loan_id) REFERENCES dbo.Loans(loan_id)
);

-- 4. Example INSERT Statement
/*
INSERT INTO dbo.Loans (person_age, person_income, person_home_ownership, person_emp_length, 
                       loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status, 
                       loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length)
VALUES (25, 55000, 'RENT', 4, 'PERSONAL', 'B', 10000, 11.5, 0, 0.18, 'N', 3);
*/

-- 5. BULK INSERT Guide
/*
STEP-BY-STEP BULK LOAD:
1. Ensure your 'credit_risk_dataset.csv' is cleaned (no headers or proper mapping).
2. Run the following command in SSMS:

BULK INSERT dbo.Loans
FROM 'C:\Path\To\Your\dataset_no_header.csv'
WITH (
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);
*/
