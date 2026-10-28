"""v3.3 features: data profiling, drift detection, anomaly detection, threshold tuning."""
from __future__ import annotations

import hmac
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .ml_portfolio import build_model, load_dataset


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Quick data-quality profile: shape, missing values, duplicates, constants, class balance."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    missing = df.isna().mean().round(4)
    report: dict[str, Any] = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": [c for c in df.columns if c not in df.select_dtypes(include="number").columns],
        "missing_fraction": {c: float(v) for c, v in missing.items() if v > 0},
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "high_cardinality": [c for c in df.select_dtypes(exclude="number").columns
                             if df[c].nunique() > 0.5 * len(df) and len(df) > 20],
    }
    warnings = []
    if report["duplicate_rows"]:
        warnings.append(f"{report['duplicate_rows']} duplicate rows")
    if report["constant_columns"]:
        warnings.append(f"constant columns: {report['constant_columns']}")
    worst = max(report["missing_fraction"].items(), key=lambda kv: kv[1], default=None)
    if worst and worst[1] > 0.3:
        warnings.append(f"column {worst[0]!r} is {worst[1]:.0%} missing")
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} not found")
        shares = df[target].value_counts(normalize=True)
        report["class_balance"] = {str(k): round(float(v), 4) for k, v in shares.items()}
        if len(shares) > 1 and shares.min() < 0.1:
            warnings.append(f"imbalanced target: smallest class is {shares.min():.1%}")
    report["warnings"] = warnings
    return report


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference and a new sample (rule of thumb: <0.1 stable, >0.25 major shift)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2:
        raise ValueError("Need at least 2 non-missing values in each sample")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """Per-feature drift table using PSI and a two-sample Kolmogorov-Smirnov test."""
    cols = [c for c in reference.select_dtypes(include="number").columns if c in current.columns]
    if not cols:
        raise ValueError("No shared numeric columns to compare")
    rows = []
    for c in cols:
        ref, cur = reference[c].dropna(), current[c].dropna()
        psi = population_stability_index(ref, cur)
        ks = ks_2samp(ref, cur)
        status = "major" if psi > 0.25 else "moderate" if psi > 0.1 else "stable"
        rows.append({"feature": c, "psi": round(psi, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_pvalue": float(ks.pvalue), "drift": status, "ks_significant": bool(ks.pvalue < alpha)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def detect_anomalies(X: pd.DataFrame | np.ndarray, contamination: float = 0.05,
                     random_state: int = 42) -> pd.DataFrame:
    """Isolation Forest anomaly scores. Returns DataFrame(index, score, is_anomaly); higher score = more unusual."""
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or len(arr) < 10:
        raise ValueError("X must be 2-D with at least 10 rows")
    if np.isnan(arr).any():
        raise ValueError("X contains missing values; impute them first")
    scaled = StandardScaler().fit_transform(arr)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state).fit(scaled)
    scores = -iso.score_samples(scaled)
    return pd.DataFrame({"index": np.arange(len(arr)), "score": scores,
                         "is_anomaly": iso.predict(scaled) == -1}).sort_values("score", ascending=False).reset_index(drop=True)


def tune_threshold(dataset: str = "breast_cancer", model_name: str = "logistic_regression",
                   cost_fn: float = 5.0, cost_fp: float = 1.0, cv: int = 5, random_state: int = 42) -> dict[str, Any]:
    """Pick a decision threshold from out-of-fold probabilities by minimising cost (binary datasets only).

    cost_fn / cost_fp: relative cost of a false negative / false positive for the positive class (label 1).
    """
    data = load_dataset(dataset)
    if len(np.unique(data.target)) != 2:
        raise ValueError("Threshold tuning needs a binary dataset (e.g. breast_cancer)")
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError("costs must be positive")
    model = build_model(model_name, random_state)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name!r} has no predict_proba")
    splitter = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    proba = cross_val_predict(model, data.data, data.target, cv=splitter, method="predict_proba")[:, 1]
    y = data.target
    rows = []
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        rows.append({"threshold": round(float(t), 2), "cost": cost_fn * fn + cost_fp * fp, "fn": fn, "fp": fp,
                     "precision": float(precision_score(y, pred, zero_division=0)),
                     "recall": float(recall_score(y, pred, zero_division=0)),
                     "f1": float(f1_score(y, pred, zero_division=0))})
    table = pd.DataFrame(rows)
    best = table.loc[table["cost"].idxmin()]
    default = table.loc[(table["threshold"] - 0.5).abs().idxmin()]
    return {"table": table, "best_threshold": float(best["threshold"]), "best_cost": float(best["cost"]),
            "default_cost": float(default["cost"]), "note": "Out-of-fold estimate; confirm on a separate test set."}


def check_api_key(provided: str | None, expected: str | None) -> bool:
    """Constant-time API key check. If no key is configured, access is open (local development)."""
    if not expected:
        return True
    return provided is not None and hmac.compare_digest(provided.encode(), expected.encode())
