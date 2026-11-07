"""Weight-of-Evidence logistic regression scorecard (the benchmark model).

Each variable is binned (quantile bins for numeric variables, one bin per
category for categoricals), each bin is replaced by its Weight of Evidence,
and a logistic regression is fitted on the WoE values. Coefficients are then
scaled to points: base score 600 at odds of 50:1 (good:bad), 20 points to
double the odds (PDO = 20).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 0.5  # smoothing so empty cells do not give infinite WoE


class WoEBinner:
    def __init__(self, numeric_cols, categorical_cols, n_bins: int = 6):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.n_bins = n_bins

    def _bin(self, col: str, values: pd.Series) -> pd.Series:
        if col in self.numeric_cols:
            return pd.cut(values, self.edges_[col], include_lowest=True).astype(str)
        return values.astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        self.edges_, self.woe_, self.iv_ = {}, {}, {}
        total_bad = y.sum()
        total_good = len(y) - total_bad
        for col in self.numeric_cols + self.categorical_cols:
            if col in self.numeric_cols:
                edges = np.unique(np.quantile(X[col], np.linspace(0, 1, self.n_bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges_[col] = edges
            bins = self._bin(col, X[col])
            grp = pd.DataFrame({"bin": bins, "y": y.values}).groupby("bin")["y"].agg(["sum", "size"])
            bad = grp["sum"] + EPS
            good = grp["size"] - grp["sum"] + EPS
            dist_bad = bad / (total_bad + EPS)
            dist_good = good / (total_good + EPS)
            woe = np.log(dist_good / dist_bad)
            self.woe_[col] = woe.to_dict()
            self.iv_[col] = float(((dist_good - dist_bad) * woe).sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.numeric_cols + self.categorical_cols:
            out[col] = self._bin(col, X[col]).map(self.woe_[col]).fillna(0.0)
        return out

    def iv_table(self) -> pd.DataFrame:
        table = pd.Series(self.iv_, name="information_value").sort_values(ascending=False)
        def strength(iv):
            if iv < 0.02:
                return "not predictive"
            if iv < 0.1:
                return "weak"
            if iv < 0.3:
                return "medium"
            if iv < 0.5:
                return "strong"
            return "very strong (check for leakage)"
        return table.to_frame().assign(strength=table.map(strength)).rename_axis("variable").reset_index()


class Scorecard:
    def __init__(self, numeric_cols, categorical_cols, base_score=600, base_odds=50, pdo=20):
        self.binner = WoEBinner(numeric_cols, categorical_cols)
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binner.fit(X, y)
        Xw = self.binner.transform(X)
        self.model_ = LogisticRegression(C=1.0, max_iter=1000)
        self.model_.fit(Xw, y)
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        return self

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.binner.transform(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score = offset + factor * ln(good/bad odds). Higher score = lower risk."""
        pd_ = np.clip(self.predict_pd(X), 1e-6, 1 - 1e-6)
        return np.round(self.offset_ + self.factor_ * np.log((1 - pd_) / pd_)).astype(int)

    def points_table(self) -> pd.DataFrame:
        """Points per bin. Since the model predicts P(default), a positive
        WoE (more goods) earns points via -coef * woe * factor."""
        cols = self.binner.numeric_cols + self.binner.categorical_cols
        n = len(cols)
        rows = []
        for coef, col in zip(self.model_.coef_[0], cols):
            for bin_label, woe in self.binner.woe_[col].items():
                points = -coef * woe * self.factor_ + (self.offset_ - self.factor_ * self.model_.intercept_[0]) / n
                rows.append({"variable": col, "bin": bin_label, "woe": round(woe, 4), "points": int(round(points))})
        return pd.DataFrame(rows)
