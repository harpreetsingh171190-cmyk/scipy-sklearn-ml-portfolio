"""SHAP-style explanations without extra dependencies (Monte Carlo Shapley values).

For one row, random feature orderings switch features from a background row to the
explained row one at a time; the change in predicted probability is credited to the
feature just switched. Averaging over orderings approximates Shapley values, and
values + base_value always sum exactly to the model output for the row.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ml_portfolio import load_dataset

__all__ = ["shapley_values", "explain_row", "global_importance", "explain_regression"]

MAX_PERMUTATIONS = 200


def _proba(model: Any, rows: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must support predict_proba for explanations")
    return model.predict_proba(rows)


def _shapley_core(f, x: np.ndarray, background: np.ndarray, n_permutations: int, random_state: int):
    d = x.size
    rng = np.random.default_rng(random_state)
    phi = np.zeros(d)
    base = 0.0
    for _ in range(n_permutations):
        order = rng.permutation(d)
        rows = np.tile(background[rng.integers(len(background))], (d + 1, 1))
        for k, feat in enumerate(order, start=1):
            rows[k:, feat] = x[feat]
        p = np.asarray(f(rows), dtype=float)
        phi[order] += np.diff(p)
        base += p[0]
    return phi / n_permutations, base / n_permutations


def shapley_values(model: Any, x, background, class_index: int | None = None,
                   n_permutations: int = 50, random_state: int = 42) -> dict[str, Any]:
    """Approximate Shapley values of one row for one class probability."""
    x = np.asarray(x, dtype=float).ravel()
    background = np.asarray(background, dtype=float)
    if background.ndim != 2 or background.shape[1] != x.size or len(background) == 0:
        raise ValueError("background must be a non-empty 2D array matching the row width")
    if not 1 <= n_permutations <= MAX_PERMUTATIONS:
        raise ValueError(f"n_permutations must be between 1 and {MAX_PERMUTATIONS}")
    probs_x = _proba(model, x[None, :])[0]
    if class_index is None:
        class_index = int(np.argmax(probs_x))
    if not 0 <= class_index < len(probs_x):
        raise ValueError("class_index out of range")
    phi, base = _shapley_core(lambda r: _proba(model, r)[:, class_index], x, background,
                              n_permutations, random_state)
    return {"values": phi, "base_value": float(base), "prediction": float(probs_x[class_index]),
            "class_index": int(class_index)}


def explain_row(model: Any, x, background, feature_names: list[str] | None = None,
                class_index: int | None = None, n_permutations: int = 50,
                random_state: int = 42, top_k: int | None = None) -> dict[str, Any]:
    """Shapley values as a ranked table plus base value and prediction."""
    res = shapley_values(model, x, background, class_index, n_permutations, random_state)
    d = len(res["values"])
    names = feature_names or [f"f{i}" for i in range(d)]
    table = pd.DataFrame({"feature": names, "value": np.asarray(x, float).ravel(),
                          "contribution": res["values"]})
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    if top_k is not None:
        table = table.head(max(1, top_k))
    res["table"] = table.reset_index(drop=True)
    return res


def global_importance(model: Any, X, background, feature_names: list[str] | None = None,
                      n_rows: int = 20, n_permutations: int = 20, random_state: int = 42) -> pd.DataFrame:
    """Mean |Shapley value| over a sample of rows (predicted class of each row)."""
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        raise ValueError("X is empty")
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    acc = np.mean([np.abs(shapley_values(model, X[i], background, None, n_permutations,
                                         random_state + int(i))["values"]) for i in idx], axis=0)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    return (pd.DataFrame({"feature": names, "mean_abs_contribution": acc})
            .sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True))


def dataset_background(dataset: str, size: int = 50, random_state: int = 42) -> np.ndarray:
    data = load_dataset(dataset)
    rng = np.random.default_rng(random_state)
    return data.data[rng.choice(len(data.data), size=min(size, len(data.data)), replace=False)]


def explain_regression(model: Any, x, background, feature_names: list[str] | None = None,
                       n_permutations: int = 50, random_state: int = 42,
                       top_k: int | None = None) -> dict[str, Any]:
    """Shapley contributions to a regressor's predicted value (sum = prediction - base_value)."""
    x = np.asarray(x, dtype=float).ravel()
    background = np.asarray(background, dtype=float)
    if background.ndim != 2 or background.shape[1] != x.size or len(background) == 0:
        raise ValueError("background must be a non-empty 2D array matching the row width")
    if not 1 <= n_permutations <= MAX_PERMUTATIONS:
        raise ValueError(f"n_permutations must be between 1 and {MAX_PERMUTATIONS}")
    phi, base = _shapley_core(model.predict, x, background, n_permutations, random_state)
    names = feature_names or [f"f{i}" for i in range(x.size)]
    table = pd.DataFrame({"feature": names, "value": x, "contribution": phi})
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    if top_k is not None:
        table = table.head(max(1, top_k))
    return {"table": table.reset_index(drop=True), "base_value": float(base),
            "prediction": float(model.predict(x[None, :])[0]), "values": phi}
