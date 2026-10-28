"""v3.4 features: ensembles, probability calibration, PCA summary, feature selection."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .ml_portfolio import build_model, load_dataset

__all__ = ["compare_ensembles", "calibration_report", "expected_calibration_error",
           "pca_summary", "select_features"]

_BASE = ["logistic_regression", "random_forest", "svm_rbf"]


def _check_cv(cv: int) -> None:
    if cv < 2:
        raise ValueError("cv must be at least 2")


def compare_ensembles(dataset: str = "iris", cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Cross-validated accuracy of base models vs soft-voting and stacking ensembles."""
    _check_cv(cv)
    data = load_dataset(dataset)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    est = [(n, build_model(n, random_state)) for n in _BASE]
    candidates = {n: m for n, m in est}
    candidates["voting_soft"] = VotingClassifier(est, voting="soft")
    candidates["stacking"] = StackingClassifier(
        est, final_estimator=LogisticRegression(max_iter=2000), cv=3)
    rows = []
    for name, model in candidates.items():
        s = cross_val_score(model, data.data, data.target, cv=splitter, scoring="accuracy")
        rows.append({"model": name, "accuracy_mean": s.mean(), "accuracy_std": s.std(),
                     "kind": "ensemble" if name in ("voting_soft", "stacking") else "base"})
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)


def expected_calibration_error(y_true, prob, bins: int = 10) -> float:
    """ECE for binary probabilities of the positive class."""
    y_true, prob = np.asarray(y_true, float), np.asarray(prob, float)
    if y_true.shape != prob.shape or y_true.size == 0:
        raise ValueError("y_true and prob must be non-empty and the same length")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def calibration_report(dataset: str = "breast_cancer", model_name: str = "random_forest",
                       method: str = "sigmoid", random_state: int = 42) -> dict[str, Any]:
    """Compare Brier score and ECE before/after probability calibration (binary datasets)."""
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    data = load_dataset(dataset)
    if len(np.unique(data.target)) != 2:
        raise ValueError("calibration_report needs a binary dataset (e.g. breast_cancer)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.3, random_state=random_state, stratify=data.target)
    raw = build_model(model_name, random_state).fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(build_model(model_name, random_state), method=method, cv=3).fit(X_tr, y_tr)
    out: dict[str, Any] = {"dataset": dataset, "model": model_name, "method": method}
    for label, m in (("raw", raw), ("calibrated", cal)):
        p = m.predict_proba(X_te)[:, 1]
        out[f"{label}_brier"] = float(brier_score_loss(y_te, p))
        out[f"{label}_ece"] = expected_calibration_error(y_te, p)
    return out


def pca_summary(dataset: str = "wine", variance: float = 0.95) -> dict[str, Any]:
    """Components needed to explain `variance` of the (standardised) data."""
    if not 0 < variance <= 1:
        raise ValueError("variance must be in (0, 1]")
    data = load_dataset(dataset)
    X = StandardScaler().fit_transform(data.data)
    ratio = PCA().fit(X).explained_variance_ratio_
    cum = np.cumsum(ratio)
    n = int(np.searchsorted(cum, variance - 1e-12) + 1)
    return {"dataset": dataset, "n_features": int(X.shape[1]), "components_needed": n,
            "target_variance": variance,
            "explained_ratio": [round(float(r), 4) for r in ratio[:min(10, len(ratio))]],
            "cumulative": [round(float(c), 4) for c in cum[:min(10, len(cum))]]}


def select_features(dataset: str = "wine", k: int = 5, method: str = "f_test") -> pd.DataFrame:
    """Rank features by ANOVA F-test or mutual information; mark the top k."""
    data = load_dataset(dataset)
    n = data.data.shape[1]
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}")
    funcs = {"f_test": f_classif, "mutual_info": mutual_info_classif}
    if method not in funcs:
        raise ValueError(f"method must be one of {sorted(funcs)}")
    sel = SelectKBest(funcs[method], k=k).fit(data.data, data.target)
    names = getattr(data, "feature_names", [f"f{i}" for i in range(n)])
    df = pd.DataFrame({"feature": [str(x) for x in names], "score": sel.scores_,
                       "selected": sel.get_support()})
    return df.sort_values("score", ascending=False).reset_index(drop=True)
