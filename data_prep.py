"""Loading and cleaning for the Kaggle credit_risk_dataset.csv (32,581 loans).

Cleaning rules are deliberately simple and documented so that every row
removed or changed can be explained:

1. Exact duplicate rows are dropped (165 in the raw file).
2. Applicants older than 100 are dropped as data-entry errors (5 rows, max 144).
3. Employment length longer than (age - 14) years is impossible, so it is set
   to missing and then imputed like any other missing value (2 rows, both 123).
4. Missing values are imputed with statistics learned on the TRAINING split only,
   so no information from the test set leaks into the model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "credit_risk_dataset.csv"
TARGET = "loan_status"
RANDOM_STATE = 42

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
# Variables set by the lender's own pricing / grading process. They are
# legitimate application-time inputs, but they already encode the lender's
# view of risk, so the model is also tested without them (see MODEL_VALIDATION.md).
LENDER_ASSIGNED_COLS = ["loan_grade", "loan_int_rate"]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "loan_id", np.arange(1, len(df) + 1))  # stable key for SQL / Power BI
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the row-level cleaning rules and return a log of what changed."""
    log = {"rows_in": int(len(df))}
    feature_cols = [c for c in df.columns if c != "loan_id"]

    dup_mask = df.duplicated(subset=feature_cols)
    log["duplicates_dropped"] = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    age_mask = df["person_age"] > 100
    log["age_over_100_dropped"] = int(age_mask.sum())
    df = df.loc[~age_mask].copy()

    emp_mask = df["person_emp_length"] > (df["person_age"] - 14)
    log["impossible_emp_length_set_missing"] = int(emp_mask.sum())
    df.loc[emp_mask, "person_emp_length"] = np.nan

    log["rows_out"] = int(len(df))
    log["default_rate"] = float(df[TARGET].mean())
    return df, log


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified random split. The dataset has no dates, so an out-of-time
    split is not possible; this limitation is stated in MODEL_VALIDATION.md."""
    return train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )


class Imputer:
    """Median imputation fitted on training data only.

    Interest rate is imputed with the median for the same loan grade, because
    rates are priced off grade; everything else uses the overall median.
    A missing-value flag is kept for both imputed columns.
    """

    def fit(self, train: pd.DataFrame) -> "Imputer":
        self.rate_by_grade_ = train.groupby("loan_grade")["loan_int_rate"].median()
        self.rate_overall_ = train["loan_int_rate"].median()
        self.emp_median_ = train["person_emp_length"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
        df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
        df["person_emp_length"] = df["person_emp_length"].fillna(self.emp_median_)
        grade_rate = df["loan_grade"].map(self.rate_by_grade_).fillna(self.rate_overall_)
        df["loan_int_rate"] = df["loan_int_rate"].fillna(grade_rate)
        return df


def encode_for_trees(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode categoricals for XGBoost. Loan grade is ordinal (A best,
    G worst) so it is mapped to 1-7 rather than one-hot encoded."""
    out = df[columns].copy()
    if "loan_grade" in out:
        out["loan_grade"] = out["loan_grade"].map({g: i for i, g in enumerate("ABCDEFG", 1)})
    other_cats = [c for c in CATEGORICAL_COLS if c in out and c != "loan_grade"]
    out = pd.get_dummies(out, columns=other_cats, dtype=int)
    return out
