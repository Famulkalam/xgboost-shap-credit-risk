import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics  # noqa: E402
from data_prep import (  # noqa: E402
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    Imputer,
    clean,
    encode_for_trees,
    load_raw,
    split,
)
from scorecard import Scorecard  # noqa: E402


@pytest.fixture(scope="module")
def data():
    raw = load_raw()
    df, log = clean(raw)
    train, test = split(df)
    imp = Imputer().fit(train)
    return raw, df, log, imp.transform(train), imp.transform(test)


def test_cleaning_removes_known_errors(data):
    raw, df, log, _, _ = data
    assert log["duplicates_dropped"] == 165
    assert df["person_age"].max() <= 100
    assert not (df["person_emp_length"] > df["person_age"] - 14).any()
    assert log["rows_out"] == len(df) == len(raw) - 165 - log["age_over_100_dropped"]


def test_loan_ids_unique_and_stable(data):
    raw, df, _, train, test = data
    assert raw["loan_id"].is_unique
    assert raw["loan_id"].iloc[0] == 1
    assert set(train["loan_id"]).isdisjoint(set(test["loan_id"]))


def test_no_missing_after_imputation(data):
    *_, train, test = data
    for frame in (train, test):
        assert not frame[NUMERIC_COLS].isna().any().any()


def test_target_not_in_features(data):
    *_, train, _ = data
    X = encode_for_trees(train, NUMERIC_COLS + CATEGORICAL_COLS)
    assert TARGET not in X.columns
    assert "loan_id" not in X.columns


def test_loan_grade_is_ordinal(data):
    *_, train, _ = data
    X = encode_for_trees(train, ["loan_grade"])
    assert set(X["loan_grade"].unique()) <= set(range(1, 8))


def test_ks_and_gini_known_values():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.ks_statistic(y, perfect) == pytest.approx(1.0)
    assert metrics.gini(y, perfect) == pytest.approx(1.0)
    assert metrics.gini(y, -perfect) == pytest.approx(-1.0)


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).uniform(size=5000)
    assert metrics.psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert metrics.psi(x, x + 0.5) > 0.25


def test_scorecard_higher_score_means_lower_pd(data):
    *_, train, test = data
    sc = Scorecard(NUMERIC_COLS, CATEGORICAL_COLS).fit(train, train[TARGET])
    pd_test = sc.predict_pd(test)
    score = sc.score(test)
    assert np.corrcoef(pd_test, score)[0, 1] < -0.9
    # points table reproduces the score
    pts = sc.points_table()
    assert set(pts["variable"]) == set(NUMERIC_COLS + CATEGORICAL_COLS)


def test_points_add_up_to_score(data):
    *_, train, test = data
    sc = Scorecard(NUMERIC_COLS, CATEGORICAL_COLS).fit(train, train[TARGET])
    pts = sc.points_table().set_index(["variable", "bin"])["points"]
    row = test.iloc[[0]]
    total = 0
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        label = sc.binner._bin(col, row[col]).iloc[0]
        total += pts[(col, label)]
    # rounding each bin's points can move the total by a few points
    assert abs(total - sc.score(row)[0]) <= len(NUMERIC_COLS + CATEGORICAL_COLS)
