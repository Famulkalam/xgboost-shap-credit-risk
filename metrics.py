"""Credit-risk model metrics: discrimination, calibration and stability."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def gini(y_true, y_score) -> float:
    """Gini (accuracy ratio) = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    """Maximum distance between the cumulative score distributions of
    defaulters and non-defaulters."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score distributions.
    Bin edges are deciles of the expected (development) distribution.
    Rule of thumb: < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by decile of predicted PD."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    table = df.groupby("decile", observed=True).agg(
        n=("y", "size"), predicted_pd=("p", "mean"), observed_dr=("y", "mean")
    )
    return table.reset_index()


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42):
    """95% bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summary(y_true, y_prob) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "gini": float(gini(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "mean_pd": float(np.mean(y_prob)),
        "observed_dr": float(np.mean(y_true)),
    }
