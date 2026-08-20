"""Train and evaluate a supervised baseline classifier (Phase 6).

A Random Forest is a good first classifier here: it is robust to the small,
imbalanced, single-fight dataset, needs no feature scaling, and exposes
feature importances (useful to talk about).

Split choice: we use a **stratified random split** and keep it reproducible.
This is the pragmatic choice for a single-fight proof of concept. The honest
caveat is that a proper production evaluation should hold out whole fights
(temporal/event grouping) so adjacent bins of the same highlight event do
not leak across the split — that needs more than one fight, which is exactly
the Phase 5 goal of expanding the dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def temporal_split(
    X: np.ndarray, y: np.ndarray, train_frac: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split chronologically (first `train_frac` of rows for training).

    Kept for reference; not used as the default because the raw timeline
    includes lots of post-fight footage with no highlights, which makes a
    naive chronological holdout empty of positives.
    """
    cut = int(len(y) * train_frac)
    return X[:cut], X[cut:], y[:cut], y[cut:]


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split preserving class proportions; returns train/test
    indices so callers can evaluate any predictor on the same held-out rows."""
    idx = np.arange(len(y))
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, idx, stratify=y, train_size=train_frac, random_state=random_state
    )
    return X_tr, X_te, y_tr, y_te, tr_idx, te_idx


def train_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """Train a Random Forest on a stratified split and evaluate on the holdout.

    `class_weight="balanced"` handles the class imbalance (highlights are a
    small fraction of time bins). Returns the fitted model, held-out arrays,
    predictions, test indices, and precision/recall/F1 on the holdout.
    """
    X_tr, X_te, y_tr, y_te, _, te_idx = stratified_split(
        X, y, train_frac, random_state=random_state
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average="binary", zero_division=0
    )
    return {
        "clf": clf,
        "X_test": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "test_index": te_idx,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_test": int(len(y_te)),
    }
