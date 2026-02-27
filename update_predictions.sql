/*
================================================================
TASK: Update SQL tables with XGBoost predictions
PROJECT: XGBoost-SHAP Credit Risk Scorecard
================================================================
*/

-- 1. Load the Predictions from CSV (Temp Table Approach)
IF OBJECT_ID('tempdb..#TempPredictions') IS NOT NULL DROP TABLE #TempPredictions;
CREATE TABLE #TempPredictions (
    temp_id INT, -- This would be the index/loan_id from your CSV
    probability_default FLOAT,
    is_high_risk BIT,
    risk_band VARCHAR(20)
);

-- BULK INSERT the predictions.csv
/*
BULK INSERT #TempPredictions
FROM 'C:\Path\To\predictions.csv'
WITH (
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);
*/

-- 2. Insert into the permanent LoanPredictions table
INSERT INTO dbo.LoanPredictions (loan_id, probability_default, risk_band, is_high_risk)
SELECT 
    temp_id, -- Maps to the loan_id in dbo.Loans
    probability_default,
    risk_band,
    is_high_risk
FROM #TempPredictions;

-- 3. CALCULATE EXPECTED LOSS (Impressive SQL Feature)
-- Formula: Expected Loss = PD (Prob of Default) * LGD (Loss Given Default) * EAD (Exposure at Default)
-- For this project, let's assume LGD = 0.5 (50% loss if they default)
-- EAD = loan_amnt

ALTER TABLE dbo.LoanPredictions ADD expected_loss AS (probability_default * 0.5 * (SELECT loan_amnt FROM dbo.Loans WHERE dbo.Loans.loan_id = dbo.LoanPredictions.loan_id));

-- 4. Create a View for Power BI
IF OBJECT_ID('dbo.vw_LoanRiskSummary', 'V') IS NOT NULL DROP VIEW dbo.vw_LoanRiskSummary;
GO
CREATE VIEW dbo.vw_LoanRiskSummary AS
SELECT 
    L.loan_id,
    L.person_age,
    L.person_income,
    L.loan_amnt,
    L.loan_status as actual_default,
    P.probability_default,
    P.risk_band,
    P.is_high_risk,
    P.expected_loss
FROM dbo.Loans L
JOIN dbo.LoanPredictions P ON L.loan_id = P.loan_id;
GO
