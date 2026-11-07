"""Train, validate and explain the credit-risk models.

Benchmark: WoE logistic regression scorecard (scorecard.py)
Challenger: XGBoost gradient-boosted trees, explained with SHAP

Run:  python credit_risk_model.py
Outputs go to outputs/ (tables, metrics.json) and figures/ (plots);
predictions.csv is written to the repo root for the SQL / Power BI step.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

import metrics
from data_prep import (
    CATEGORICAL_COLS,
    LENDER_ASSIGNED_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    Imputer,
    clean,
    encode_for_trees,
    load_raw,
    split,
)
from scorecard import Scorecard

OUT_DIR = "outputs"
FIG_DIR = "figures"
PREDICTIONS_CSV = "predictions.csv"
EXTRA_NUMERIC = ["emp_length_missing", "int_rate_missing"]

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=RANDOM_STATE,
    n_jobs=4,
)


def fit_xgb(X_train, y_train):
    """Fit XGBoost with early stopping on a validation split carved out of
    the training data, so the test set is never used for model selection."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def cross_validated_auc(X, y, n_splits=5):
    """5-fold CV AUC on the training data, as a check that the single
    test-set number is not a lucky split."""
    aucs = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in folds.split(X, y):
        m = fit_xgb(X.iloc[tr], y.iloc[tr])
        aucs.append(metrics.summary(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1])["auc"])
    return float(np.mean(aucs)), float(np.std(aucs))


def risk_band(pd_value: float) -> str:
    if pd_value < 0.10:
        return "Low"
    if pd_value < 0.30:
        return "Medium"
    return "High"


def plot_calibration(tables: dict, path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, t in tables.items():
        ax.plot(t["predicted_pd"], t["observed_dr"], marker="o", label=name)
    lim = max(max(t["predicted_pd"].max(), t["observed_dr"].max()) for t in tables.values())
    ax.plot([0, lim], [0, lim], "--", color="grey", label="Perfect calibration")
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: dict, path: str):
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (Gini {metrics.gini(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Data
    raw = load_raw()
    df, cleaning_log = clean(raw)
    train, test = split(df)
    imputer = Imputer().fit(train)
    train, test = imputer.transform(train), imputer.transform(test)
    y_train, y_test = train[TARGET], test[TARGET]
    print("Cleaning:", cleaning_log)

    # 2. Benchmark: WoE logistic scorecard
    sc = Scorecard(NUMERIC_COLS, CATEGORICAL_COLS).fit(train, y_train)
    pd_sc_train, pd_sc_test = sc.predict_pd(train), sc.predict_pd(test)
    sc.binner.iv_table().to_csv(f"{OUT_DIR}/information_value.csv", index=False)
    sc.points_table().to_csv(f"{OUT_DIR}/scorecard_points.csv", index=False)

    # 3. Challenger: XGBoost
    feature_cols = NUMERIC_COLS + EXTRA_NUMERIC + CATEGORICAL_COLS
    X_train = encode_for_trees(train, feature_cols)
    X_test = encode_for_trees(test, feature_cols).reindex(columns=X_train.columns, fill_value=0)
    xgb_model = fit_xgb(X_train, y_train)
    pd_xgb_train = xgb_model.predict_proba(X_train)[:, 1]
    pd_xgb_test = xgb_model.predict_proba(X_test)[:, 1]
    cv_mean, cv_std = cross_validated_auc(X_train, y_train)

    # 4. Sensitivity: XGBoost without lender-assigned grade and interest rate
    reduced_cols = [c for c in feature_cols if c not in LENDER_ASSIGNED_COLS + ["int_rate_missing"]]
    Xr_train = encode_for_trees(train, reduced_cols)
    Xr_test = encode_for_trees(test, reduced_cols).reindex(columns=Xr_train.columns, fill_value=0)
    xgb_reduced = fit_xgb(Xr_train, y_train)
    pd_reduced_test = xgb_reduced.predict_proba(Xr_test)[:, 1]

    # 5. Metrics
    results = {
        "cleaning": cleaning_log,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "scorecard": {
            "train": metrics.summary(y_train, pd_sc_train),
            "test": metrics.summary(y_test, pd_sc_test),
            "test_auc_95ci": metrics.bootstrap_auc_ci(y_test, pd_sc_test),
            "psi_train_vs_test": metrics.psi(pd_sc_train, pd_sc_test),
        },
        "xgboost": {
            "train": metrics.summary(y_train, pd_xgb_train),
            "test": metrics.summary(y_test, pd_xgb_test),
            "test_auc_95ci": metrics.bootstrap_auc_ci(y_test, pd_xgb_test),
            "cv5_auc_mean": cv_mean,
            "cv5_auc_std": cv_std,
            "psi_train_vs_test": metrics.psi(pd_xgb_train, pd_xgb_test),
            "best_iteration": int(xgb_model.best_iteration),
        },
        "xgboost_without_grade_and_rate": {
            "test": metrics.summary(y_test, pd_reduced_test),
        },
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    cal_sc = metrics.calibration_table(y_test, pd_sc_test)
    cal_xgb = metrics.calibration_table(y_test, pd_xgb_test)
    cal_sc.assign(model="scorecard").to_csv(f"{OUT_DIR}/calibration_scorecard.csv", index=False)
    cal_xgb.assign(model="xgboost").to_csv(f"{OUT_DIR}/calibration_xgboost.csv", index=False)

    # 6. Plots
    plot_roc(
        {"WoE scorecard": (y_test, pd_sc_test), "XGBoost": (y_test, pd_xgb_test)},
        f"{FIG_DIR}/roc_curves.png",
    )
    plot_calibration({"WoE scorecard": cal_sc, "XGBoost": cal_xgb}, f"{FIG_DIR}/calibration.png")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("SHAP: global feature impact (XGBoost, test set)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150)
    plt.close()

    highest_risk = int(np.argmax(pd_xgb_test))
    shap.plots.waterfall(shap_values[highest_risk], show=False, max_display=12)
    plt.title(f"SHAP: highest-PD test loan (loan_id {int(test['loan_id'].iloc[highest_risk])})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_waterfall_highest_risk.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Predictions for SQL / Power BI (test set only, keyed by loan_id)
    preds = pd.DataFrame(
        {
            "loan_id": test["loan_id"].values,
            "probability_default": np.round(pd_xgb_test, 6),
            "scorecard_pd": np.round(pd_sc_test, 6),
            "scorecard_score": sc.score(test),
        }
    )
    preds["risk_band"] = preds["probability_default"].map(risk_band)
    preds["is_high_risk"] = (preds["risk_band"] == "High").astype(int)
    preds.sort_values("loan_id").to_csv(PREDICTIONS_CSV, index=False)

    print(json.dumps({k: v for k, v in results.items() if k != "cleaning"}, indent=2))


if __name__ == "__main__":
    main()
