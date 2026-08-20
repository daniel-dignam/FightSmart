"""Tests for src.models.baseline."""

from __future__ import annotations

import numpy as np

from src.models.baseline import temporal_split, train_evaluate


def test_temporal_split_order_preserved():
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20)
    Xtr, Xte, ytr, yte = temporal_split(X, y, train_frac=0.7)
    assert len(ytr) == 14 and len(yte) == 6
    # Chronological: training is the first part.
    assert ytr[-1] < yte[0]


def test_train_evaluate_returns_metrics_and_model():
    # Synthetic separable features: positive class has high first-feature value.
    rng = np.random.RandomState(0)
    n = 200
    X = rng.rand(n, 4)
    y = (X[:, 0] > 0.8).astype(int)  # first feature separates classes
    res = train_evaluate(X, y, train_frac=0.7, random_state=1)
    assert "clf" in res
    assert res["n_test"] == 60
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["f1"] <= 1.0
    assert len(res["y_pred"]) == res["n_test"]
