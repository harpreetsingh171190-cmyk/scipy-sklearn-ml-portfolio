"""v3.5: class-imbalance strategies compared on a deliberately skewed dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ml_portfolio import load_dataset

__all__ = ["make_imbalanced", "compare_imbalance_strategies", "compare_strategies_xy", "STRATEGIES"]

STRATEGIES = ["baseline", "class_weight", "oversample", "undersample"]


def make_imbalanced(X, y, minority_fraction: float = 0.1, random_state: int = 42):
    """Keep all majority rows and subsample the minority class to the requested overall fraction."""
    if not 0.01 <= minority_fraction <= 0.5:
        raise ValueError("minority_fraction must be between 0.01 and 0.5")
    X, y = np.asarray(X), np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("a binary dataset is required")
    minor, major = classes[np.argmin(counts)], classes[np.argmax(counts)]
    n_major = int((y == major).sum())
    n_minor = max(5, int(round(n_major * minority_fraction / (1 - minority_fraction))))
    rng = np.random.default_rng(random_state)
    minor_idx = np.flatnonzero(y == minor)
    keep_minor = rng.choice(minor_idx, size=min(n_minor, len(minor_idx)), replace=False)
    idx = np.concatenate([np.flatnonzero(y == major), keep_minor])
    rng.shuffle(idx)
    return X[idx], (y[idx] == minor).astype(int)   # minority class becomes label 1


def _resample(X, y, strategy: str, rng):
    if strategy not in ("oversample", "undersample"):
        return X, y
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if strategy == "oversample":
        pos = rng.choice(pos, size=len(neg), replace=True)
    else:
        neg = rng.choice(neg, size=len(pos), replace=False)
    idx = np.concatenate([pos, neg])
    return X[idx], y[idx]


def compare_strategies_xy(X, y, model: str = "logistic_regression", random_state: int = 42) -> pd.DataFrame:
    """Compare strategies on numeric X and binary y (minority class labelled 1)."""
    if model not in ("logistic_regression", "random_forest"):
        raise ValueError("model must be 'logistic_regression' or 'random_forest'")
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if len(np.unique(y)) != 2 or min(np.bincount(y.astype(int))) < 10:
        raise ValueError("need a binary target with at least 10 rows in the minority class")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    rng = np.random.default_rng(random_state)
    rows = []
    for strategy in STRATEGIES:
        weight = "balanced" if strategy == "class_weight" else None
        est = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=weight))
               if model == "logistic_regression" else
               RandomForestClassifier(n_estimators=200, class_weight=weight, random_state=random_state))
        Xs, ys = _resample(X_tr, y_tr, strategy, rng)
        est.fit(Xs, ys)
        pred, prob = est.predict(X_te), est.predict_proba(X_te)[:, 1]
        rows.append({"strategy": strategy, "minority_recall": recall_score(y_te, pred, zero_division=0),
                     "minority_f1": f1_score(y_te, pred, zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                     "pr_auc": average_precision_score(y_te, prob),
                     "minority_share_test": float(y_te.mean())})
    return pd.DataFrame(rows).sort_values("minority_f1", ascending=False).reset_index(drop=True)


def compare_imbalance_strategies(dataset: str = "breast_cancer", minority_fraction: float = 0.1,
                                 model: str = "logistic_regression", random_state: int = 42) -> pd.DataFrame:
    """Held-out minority-class metrics for baseline, class weights, over- and under-sampling.

    Resampling happens on the training split only, so the test set keeps its natural imbalance.
    """
    data = load_dataset(dataset)
    X, y = make_imbalanced(data.data, data.target, minority_fraction, random_state)
    return compare_strategies_xy(X, y, model, random_state)
