# Model validation note

A short validation summary loosely following the structure of the PRA's
model risk management principles (SS1/23): purpose, data, methodology,
performance, limitations and conclusion. This is a portfolio project on a
public dataset, not a model in production use.

## 1. Purpose and scope

Estimate the probability that a consumer loan defaults, using variables
available at application. Two models are compared:

| Role | Model | Why |
| :--- | :--- | :--- |
| Benchmark | Weight-of-Evidence logistic regression scorecard | Transparent, the industry standard for application scoring |
| Challenger | XGBoost (gradient-boosted trees), explained with SHAP | Captures interactions the scorecard cannot |

## 2. Data

- Source: Kaggle "Credit Risk Dataset", 32,581 loans, 12 columns, default rate 21.8%.
- No loan dates, no lender identity, no data dictionary from the original owner.
- Cleaning (all logged in `outputs/metrics.json`):

| Rule | Rows affected |
| :--- | ---: |
| Exact duplicate rows dropped | 165 |
| Age over 100 dropped (maximum was 144) | 5 |
| Employment length longer than age minus 14 set to missing | 2 |
| Rows used for modelling | 32,411 |

- Missing values (employment length 2.8%, interest rate 9.6%) are imputed
  with medians learned on the training split only. Interest rate is imputed
  by loan grade. A missing-value flag is kept for both.
- Split: stratified random 80/20 (25,928 train, 6,483 test). XGBoost early
  stopping uses a validation split taken from the training data, so the
  test set is used only once, for the final numbers.

## 3. Performance on the test set

| Metric | WoE scorecard | XGBoost |
| :--- | ---: | ---: |
| AUC | 0.869 | 0.944 |
| AUC 95% bootstrap interval | 0.857 to 0.881 | 0.937 to 0.952 |
| Gini | 0.739 | 0.888 |
| KS | 0.618 | 0.750 |
| Brier score | 0.102 | 0.053 |
| Mean predicted PD vs observed default rate | 21.8% vs 21.9% | 21.9% vs 21.9% |
| PSI, train vs test score distribution | 0.002 | 0.001 |

- XGBoost 5-fold cross-validated AUC on the training data: 0.948 (std 0.003),
  consistent with the single test-set figure.
- XGBoost train AUC is 0.972 against 0.944 on test, so there is some
  overfitting; the scorecard's gap is small (0.874 vs 0.869).
- Calibration by decile is in `outputs/calibration_*.csv` and
  `figures/calibration.png`. XGBoost is well calibrated across deciles. The
  scorecard under-predicts in the lowest deciles (for example 0.9% predicted
  against 2.0% observed in decile 1).

## 4. Sensitivity checks

**Lender-assigned variables.** Loan grade and interest rate are set by the
lender's own risk assessment, so they partly encode the answer. They are
legitimate application-time inputs, but a model leaning on them adds less
than its AUC suggests. Retraining XGBoost without them gives a test AUC of
**0.904** (Gini 0.809): still strong, which shows the borrower variables
carry most of the signal.

**Information Value.** Loan grade (0.87), loan-to-income (0.78) and interest
rate (0.71) all exceed the 0.5 level usually treated as a leakage warning.
None is a post-outcome variable, so they are kept, but the very high IVs are
one reason the results below are not representative of real portfolios.

## 5. Limitations

1. **Not representative of a real lending book.** Discrimination of
   Gini 0.89 is well above what application scorecards usually achieve
   (Gini of roughly 0.4 to 0.6). In the top decile XGBoost predicts 99.3%
   and observes 100% defaults, which suggests near-deterministic segments in
   this dataset. Results should not be read as evidence the method would
   perform this well on bank data.
2. **No out-of-time validation.** There are no dates, so the model cannot
   be tested on a later period, which is the standard test for
   application scorecards. The PSI figure only compares two random samples
   from the same period, so it says little about real stability.
3. **No through-the-cycle calibration.** PDs are point-in-time for this
   sample. They are not suitable for regulatory capital (IRB) or IFRS 9
   lifetime ECL without further work.
4. **Simplified expected loss.** The SQL view uses LGD = 45% (the Basel
   foundation-IRB value for senior unsecured exposures) and EAD = loan
   amount, because the data has no recovery or balance information.
5. **Fairness.** Age is a model input. A real deployment would need a
   review under UK equality and consumer-credit rules before using it.

## 6. Conclusion

The XGBoost challenger outperforms the scorecard benchmark on every
discrimination and calibration measure, and the gap survives removing the
lender-assigned variables. Given the limitations above, especially the
absence of out-of-time data, the appropriate next step would be to repeat
the exercise on a dated public dataset (for example Freddie Mac
single-family loan-level data) before drawing conclusions about real
portfolios.
