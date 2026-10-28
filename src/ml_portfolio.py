"""Reusable, testable utilities for the 2026 ML portfolio (v3).

Features: dataset registry, model registry, cross-validated model comparison,
held-out permutation importance, JSON experiment logging, and model
persistence with metadata + SHA-256 integrity checking.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = [
    "DATASETS", "MODELS", "load_dataset", "build_model", "build_iris_pipeline",
    "train_model", "train_iris_model", "compare_models", "feature_importance",
    "log_experiment", "read_experiments", "save_model", "load_model",
]

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def _scaled(estimator: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


MODELS = {
    "logistic_regression": lambda rs: _scaled(LogisticRegression(max_iter=2000, random_state=rs)),
    "random_forest": lambda rs: RandomForestClassifier(n_estimators=250, random_state=rs, n_jobs=1),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(random_state=rs),
    "svm_rbf": lambda rs: _scaled(SVC(probability=True, random_state=rs)),
    "knn": lambda rs: _scaled(KNeighborsClassifier(n_neighbors=5)),
}


def load_dataset(name: str):
    """Return a scikit-learn Bunch for a bundled dataset (no network needed)."""
    key = name.lower().replace(" ", "_")
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(DATASETS)}")
    return DATASETS[key]()


def build_model(name: str, random_state: int = 42) -> Any:
    """Create an unfitted model from the registry."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {sorted(MODELS)}")
    return MODELS[name](random_state)


def build_iris_pipeline(random_state: int = 42) -> Pipeline:
    """Backwards-compatible v2 helper: scaler + logistic regression."""
    return build_model("logistic_regression", random_state)


def _check_test_size(test_size: float) -> None:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")


def train_model(dataset: str = "iris", model_name: str = "logistic_regression",
                test_size: float = 0.25, random_state: int = 42) -> dict[str, Any]:
    """Train any registered model on any bundled dataset; return model + metrics."""
    _check_test_size(test_size)
    data = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size,
        random_state=random_state, stratify=data.target)
    model = build_model(model_name, random_state).fit(X_train, y_train)
    pred = model.predict(X_test)
    names = [str(n) for n in getattr(data, "target_names", np.unique(data.target))]
    return {
        "model": model, "dataset": dataset, "model_name": model_name,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "predictions": pred,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "feature_names": [str(n) for n in getattr(data, "feature_names", [])],
        "target_names": names,
    }


def train_iris_model(test_size: float = 0.25, random_state: int = 42) -> dict[str, Any]:
    """Backwards-compatible v2 helper."""
    return train_model("iris", "logistic_regression", test_size, random_state)


def compare_models(dataset: str = "iris", models: list[str] | None = None,
                   cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Cross-validated leaderboard (mean ± std accuracy and macro-F1, fit time)."""
    if cv < 2:
        raise ValueError("cv must be at least 2")
    data = load_dataset(dataset)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name in models or list(MODELS):
        scores = cross_validate(
            build_model(name, random_state), data.data, data.target, cv=splitter,
            scoring={"accuracy": "accuracy", "f1": "f1_macro"}, n_jobs=1)
        rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "macro_f1_mean": scores["test_f1"].mean(),
            "macro_f1_std": scores["test_f1"].std(),
            "fit_seconds": scores["fit_time"].mean(),
        })
    return (pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)
            .reset_index(drop=True))


def feature_importance(result: dict[str, Any], n_repeats: int = 10,
                       random_state: int = 42) -> pd.DataFrame:
    """Held-out permutation importance for a result from ``train_model``."""
    imp = permutation_importance(result["model"], result["X_test"], result["y_test"],
                                 n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    names = result["feature_names"] or [f"f{i}" for i in range(len(imp.importances_mean))]
    return (pd.DataFrame({"feature": names, "importance_mean": imp.importances_mean,
                          "importance_std": imp.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def log_experiment(result: dict[str, Any], path: str | Path = "artifacts/experiments.jsonl") -> Path:
    """Append one run summary (JSON line) so experiments can be compared later."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": result["dataset"], "model": result["model_name"],
        "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
        "n_train": int(len(result["y_train"])), "n_test": int(len(result["y_test"])),
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return destination


def read_experiments(path: str | Path = "artifacts/experiments.jsonl") -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=["timestamp", "dataset", "model", "accuracy", "macro_f1",
                                     "n_train", "n_test"])
    return pd.read_json(source, lines=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Save a model plus a ``.meta.json`` sidecar with a SHA-256 checksum.

    Never load joblib/pickle files from untrusted sources.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    import sklearn
    meta = {"sha256": _sha256(destination), "sklearn_version": sklearn.__version__,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(metadata or {})}
    destination.with_suffix(destination.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_model(path: str | Path, verify: bool = True) -> Any:
    """Load a model created by this project; verify checksum when a sidecar exists."""
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".meta.json")
    if verify and sidecar.exists():
        expected = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256")
        if expected and expected != _sha256(source):
            raise ValueError("Checksum mismatch: model file changed since it was saved")
    return joblib.load(source)
