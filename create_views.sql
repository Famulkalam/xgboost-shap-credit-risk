/*
================================================================
Reporting view for Power BI (Microsoft SQL Server)
Run after push_to_sql.py has loaded dbo.Loans and dbo.LoanPredictions.

Expected loss = PD x LGD x EAD
  PD  = model probability of default
  LGD = 45%, a simplifying assumption (the dataset has no recovery data;
        45% is the Basel foundation-IRB value for senior unsecured exposures)
  EAD = loan amount (no amortisation or undrawn balances in the data)
================================================================
*/

IF OBJECT_ID('dbo.vw_LoanRiskSummary', 'V') IS NOT NULL DROP VIEW dbo.vw_LoanRiskSummary;
GO

CREATE VIEW dbo.vw_LoanRiskSummary AS
SELECT
    L.loan_id,
    L.person_age,
    L.person_income,
    L.person_home_ownership,
    L.loan_intent,
    L.loan_grade,
    L.loan_amnt,
    L.loan_status            AS actual_default,
    P.probability_default,
    P.scorecard_pd,
    P.scorecard_score,
    P.risk_band,
    P.is_high_risk,
    CAST(0.45 AS FLOAT)      AS lgd_assumption,
    P.probability_default * 0.45 * L.loan_amnt AS expected_loss
FROM dbo.Loans AS L
JOIN dbo.LoanPredictions AS P
    ON L.loan_id = P.loan_id;
GO
