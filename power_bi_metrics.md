# Power BI dashboard: measures and layout

The dashboard connects to one table, the SQL view `dbo.vw_LoanRiskSummary`
(see `create_views.sql`). In Power BI the table is referred to below as
`Risk`. It contains test-set loans only, because those are the loans the
model did not see during training.

## Core measures (DAX)

| Measure | DAX |
| :--- | :--- |
| Loans | `Loans = COUNTROWS(Risk)` |
| Funded volume | `Funded = SUM(Risk[loan_amnt])` |
| Average loan | `Avg Loan = AVERAGE(Risk[loan_amnt])` |
| Observed default rate | `Observed DR = DIVIDE(CALCULATE(COUNTROWS(Risk), Risk[actual_default] = 1), [Loans])` |
| Mean predicted PD | `Predicted PD = AVERAGE(Risk[probability_default])` |
| High-risk loans | `High Risk Loans = CALCULATE(COUNTROWS(Risk), Risk[is_high_risk] = TRUE())` |
| High-risk exposure | `High Risk Exposure = CALCULATE(SUM(Risk[loan_amnt]), Risk[is_high_risk] = TRUE())` |
| Expected loss | `Expected Loss = SUM(Risk[expected_loss])` |
| EL ratio | `EL Ratio = DIVIDE([Expected Loss], [Funded])` |
| Approval rate at 10% PD cut-off | `Approval Rate = DIVIDE(CALCULATE(COUNTROWS(Risk), Risk[probability_default] < 0.10), [Loans])` |

`Observed DR` next to `Predicted PD` is the simplest calibration check on
the dashboard: the two should be close for any slice of the portfolio.

## Layout (3 pages)

**1. Portfolio overview:** cards for Funded, Predicted PD, Observed DR and
Expected Loss; donut of risk bands; bar chart of funded volume and observed
default rate by loan intent; slicers for home ownership and loan grade.

**2. Risk segmentation:** scatter of income against loan amount coloured by
PD; matrix of loan grade by risk band (average PD); predicted against observed
default rate by scorecard score band.

**3. Stress view:** a what-if parameter scales PD and recalculates expected loss.

## Stress measure

Create a numeric what-if parameter called `Stress Factor` (1.0 to 2.0, step 0.1).
PD is scaled loan by loan and capped at 100%:

```dax
Stressed EL =
SUMX(
    Risk,
    MIN(Risk[probability_default] * 'Stress Factor'[Stress Factor Value], 1)
        * Risk[lgd_assumption]
        * Risk[loan_amnt]
)
```

This is a simple sensitivity, not a macroeconomic stress test: it assumes
every borrower's PD rises by the same multiple and that LGD does not change.
