"""v3.6: explanations and imbalance checks for your own CSV files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .explain import MAX_PERMUTATIONS, _shapley_core
from .extras import train_from_csv
from .imbalance import compare_strategies_xy

__all__ = ["explain_csv_row", "csv_imbalance"]


def explain_csv_row(path: str | Path, target: str, row: int = 0, top_k: int = 8,
                    n_permutations: int = 30, random_state: int = 42) -> dict[str, Any]:
    """Train on a CSV, then explain the predicted class probability for one row (SHAP-style)."""
    if not 1 <= n_permutations <= MAX_PERMUTATIONS:
        raise ValueError(f"n_permutations must be between 1 and {MAX_PERMUTATIONS}")
    out = train_from_csv(path, target, random_state=random_state)
    df = pd.read_csv(path).dropna(subset=[target]).reset_index(drop=True)
    X = df.drop(columns=[target])
    if not 0 <= row < len(X):
        raise ValueError(f"row must be between 0 and {len(X) - 1}")
    model, cols, dtypes = out["model"], list(X.columns), X.dtypes.to_dict()

    def frame(rows: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=cols).astype(dtypes)

    x = X.iloc[row].to_numpy(dtype=object)
    probs = model.predict_proba(frame(x[None, :]))[0]
    ci = int(np.argmax(probs))
    rng = np.random.default_rng(random_state)
    bg = X.iloc[rng.choice(len(X), size=min(50, len(X)), replace=False)].to_numpy(dtype=object)
    phi, base = _shapley_core(lambda r: model.predict_proba(frame(r))[:, ci], x, bg, n_permutations, random_state)
    table = pd.DataFrame({"feature": cols, "value": [str(v) for v in x], "contribution": phi})
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).head(max(1, top_k))
    return {"predicted_class": str(model.classes_[ci]), "prediction": float(probs[ci]),
            "base_value": float(base), "accuracy": out["accuracy"], "table": table.reset_index(drop=True)}


def csv_imbalance(path: str | Path, target: str, model: str = "logistic_regression",
                  random_state: int = 42) -> pd.DataFrame:
    """Compare imbalance strategies on a CSV with a binary target (numeric/one-hot features)."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    counts = df[target].value_counts()
    if len(counts) != 2:
        raise ValueError("csv_imbalance needs a target with exactly two classes")
    y = (df[target] == counts.idxmin()).astype(int).to_numpy()
    X = pd.get_dummies(df.drop(columns=[target]), dtype=float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("no usable feature columns")
    return compare_strategies_xy(X.to_numpy(), y, model, random_state)
