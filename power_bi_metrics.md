# Power BI Risk Dashboard Design Guide

This guide defines the KPIs and DAX measures for the "XGBoost-SHAP Credit Risk Dashboard".

## 1. Core KPIs (12-15 Metrics)

| KPI Name | Description | DAX Formula Idea |
| :--- | :--- | :--- |
| **Total Loan Portfolio** | Total number of loans in the dataset. | `Total Loans = COUNT(Loans[loan_id])` |
| **Total Funded Volume** | Sum of all loan amounts. | `Total Funded = SUM(Loans[loan_amnt])` |
| **Average Loan Amount** | Mean value of loans. | `Avg Loan = AVERAGE(Loans[loan_amnt])` |
| **Actual Default Rate** | % of loans that have actually defaulted (Historical). | `Actual Default Rate = DIVIDE(CALCULATE(COUNT(Loans[loan_id]), Loans[loan_status]=1), [Total Loans])` |
| **Predicted Default Rate** | Mean probability of default across the portfolio. | `Predicted PD = AVERAGE(LoanPredictions[probability_default])` |
| **High Risk Loans** | Count of loans with IsHighRisk = 1. | `High Risk Count = CALCULATE(COUNT(Loans[loan_id]), LoanPredictions[is_high_risk]=1)` |
| **High Risk Exposure** | Sum of loan amounts for high-risk loans. | `High Risk Volume = CALCULATE(SUM(Loans[loan_amnt]), LoanPredictions[is_high_risk]=1)` |
| **Portfolio Expected Loss** | Total expected loss based on model predictions. | `Total Expected Loss = SUM(LoanPredictions[expected_loss])` |
| **EL Ratio** | Expected loss as a % of total funded volume. | `EL Ratio = DIVIDE([Total Expected Loss], [Total Funded])` |
| **Avg PD by Risk Band** | Breakdown of probability per segment. | `Avg PD per Band = AVERAGE(LoanPredictions[probability_default])` |
| **Approval Rate** | (Simulated) Rate of "Good" loans predicted. | `Approval Rate = DIVIDE(CALCULATE(COUNT(Loans[loan_id]), LoanPredictions[probability_default] < 0.5), [Total Loans])` |
| **Model Precision** | (If actuals are known) | `Precision = ... (DAX mapping to Confusion Matrix)` |

## 2. Dashboard Layout (3 Pages)

### Page 1: Executive Overview
- **Large Tiles**: Total Funded, Predicted PD, Total Expected Loss.
- **Donut Chart**: Risk Band Distribution (Low vs Medium vs High).
- **Bar Chart**: Funding Volume vs Actual Default Rate by Loan Intent.
- **Map/Slicer**: Filter by Home Ownership or Age Group.

### Page 2: Risk Segmentation
- **Scatter Plot**: Income vs Loan Amount (Color-coded by Probability of Default).
- **Heatmap**: Loan Grade vs Age Group (Values: Avg PD).
- **Line Chart**: Predicted vs Actual Default Rate by Credit History Length.

### Page 3: Model & Stress Test
- **Gauge**: Portfolio PD vs Target (e.g., Target 5%).
- **Slicer (What-If Parameter)**: "Stress Factor" (range 1.0 to 2.0). 
- **Visual**: "Stress-Adjusted Expected Loss".

## 3. Stress Test DAX (Parameter-Based)

Create a Parameter in Power BI called `StressFactor` (Numerical).

```dax
Stress Adjusted PD = 
VAR BasePD = AVERAGE(LoanPredictions[probability_default])
VAR AdjustedPD = BasePD * 'Stress Parameter'[Stress Parameter Value]
RETURN
IF(AdjustedPD > 1, 1, AdjustedPD)

Stress Adjusted EL = 
SUMX(
    LoanPredictions,
    [Stress Adjusted PD] * LoanPredictions[loan_amnt] * 0.5
)
```

> [!IMPORTANT]
> Use the **Value** of the stress parameter to dynamically scale your `expected_loss` visual. This allows risk managers to see how a "20% increase in default probability" affects the bottom line.
