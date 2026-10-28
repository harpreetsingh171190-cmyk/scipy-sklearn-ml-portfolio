"""v3.1 extras: regression, hyperparameter tuning, clustering, CSV training, reports."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .ml_portfolio import build_model, load_dataset

REGRESSORS = {
    "ridge": lambda rs: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "random_forest": lambda rs: RandomForestRegressor(n_estimators=200, random_state=rs, n_jobs=1),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingRegressor(random_state=rs),
}

# Search spaces keyed by classifier name (prefix "classifier__" for scaled pipelines).
PARAM_GRIDS = {
    "logistic_regression": {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    "svm_rbf": {"classifier__C": [0.1, 1, 10], "classifier__gamma": ["scale", 0.01, 0.1]},
    "knn": {"classifier__n_neighbors": [3, 5, 7, 11, 15]},
    "random_forest": {"max_depth": [None, 3, 5, 8], "n_estimators": [100, 250]},
    "hist_gradient_boosting": {"learning_rate": [0.03, 0.1, 0.3], "max_depth": [None, 3, 6]},
}


def tune_model(dataset: str = "iris", model_name: str = "logistic_regression", cv: int = 5,
               method: str = "grid", n_iter: int = 8, random_state: int = 42) -> dict[str, Any]:
    """Hyperparameter search on a training split, scored on a held-out test split."""
    if model_name not in PARAM_GRIDS:
        raise ValueError(f"No search space for {model_name!r}. Choose from {sorted(PARAM_GRIDS)}")
    if method not in {"grid", "random"}:
        raise ValueError("method must be 'grid' or 'random'")
    data = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=random_state, stratify=data.target)
    splitter = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    base = build_model(model_name, random_state)
    if method == "grid":
        search = GridSearchCV(base, PARAM_GRIDS[model_name], cv=splitter, n_jobs=1)
    else:
        search = RandomizedSearchCV(base, PARAM_GRIDS[model_name], n_iter=n_iter, cv=splitter,
                                    n_jobs=1, random_state=random_state)
    search.fit(X_train, y_train)
    return {"best_params": search.best_params_, "cv_score": float(search.best_score_),
            "test_score": float(search.score(X_test, y_test)), "model": search.best_estimator_,
            "results": pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score",
                                                          "rank_test_score"]].sort_values("rank_test_score")}


def compare_regressors(cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Leaderboard on the bundled diabetes regression dataset (R², MAE, RMSE)."""
    if cv < 2:
        raise ValueError("cv must be at least 2")
    data = load_diabetes()
    splitter = KFold(cv, shuffle=True, random_state=random_state)
    rows = []
    for name, factory in REGRESSORS.items():
        s = cross_validate(factory(random_state), data.data, data.target, cv=splitter,
                           scoring={"r2": "r2", "mae": "neg_mean_absolute_error",
                                    "rmse": "neg_root_mean_squared_error"})
        rows.append({"model": name, "r2_mean": s["test_r2"].mean(), "r2_std": s["test_r2"].std(),
                     "mae": -s["test_mae"].mean(), "rmse": -s["test_rmse"].mean()})
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)


def train_regressor(model_name: str = "ridge", test_size: float = 0.25, random_state: int = 42) -> dict[str, Any]:
    if model_name not in REGRESSORS:
        raise ValueError(f"Unknown regressor {model_name!r}. Choose from {sorted(REGRESSORS)}")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state)
    model = REGRESSORS[model_name](random_state).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"model": model, "predictions": pred, "y_test": y_test, "X_test": X_test,
            "r2": float(r2_score(y_test, pred)), "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "feature_names": list(data.feature_names)}


def cluster_analysis(dataset: str = "iris", k_range: range | None = None, random_state: int = 42) -> pd.DataFrame:
    """KMeans inertia and silhouette per k, to help choose the number of clusters."""
    k_range = range(2, 8) if k_range is None else k_range
    if k_range.start < 2:
        raise ValueError("k must start at 2 or more")
    X = StandardScaler().fit_transform(load_dataset(dataset).data)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(X, km.labels_))})
    return pd.DataFrame(rows)


def train_from_csv(path: str | Path, target: str, test_size: float = 0.25,
                   random_state: int = 42) -> dict[str, Any]:
    """Train a classifier on your own CSV: numeric + categorical columns, missing values handled."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
    if y.nunique() < 2:
        raise ValueError("Target needs at least two classes")
    if y.value_counts().min() < 2:
        raise ValueError("Every class needs at least 2 rows")
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    model.fit(X_train, y_train)
    return {"model": model, "accuracy": float(model.score(X_test, y_test)), "n_rows": len(df),
            "numeric": numeric, "categorical": categorical, "classes": [str(c) for c in model.classes_]}


def generate_report(dataset: str = "iris", cv: int = 5, random_state: int = 42,
                    path: str | Path | None = None) -> str:
    """Markdown model report: leaderboard plus a short, honest interpretation."""
    from .ml_portfolio import compare_models
    board = compare_models(dataset, cv=cv, random_state=random_state)
    best = board.iloc[0]
    lines = [f"# Model report: {dataset}", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · {cv}-fold stratified CV · seed {random_state}", "",
             "| Model | Accuracy | Macro F1 | Fit (s) |", "|---|---|---|---|"]
    for r in board.itertuples():
        lines.append(f"| {r.model} | {r.accuracy_mean:.3f} ± {r.accuracy_std:.3f} | "
                     f"{r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f} | {r.fit_seconds:.3f} |")
    close = board[board.accuracy_mean >= best.accuracy_mean - best.accuracy_std]["model"].tolist()
    lines += ["", f"**Best mean accuracy:** {best.model} ({best.accuracy_mean:.3f}).",
              f"Models within one standard deviation of the best: {', '.join(close)}. "
              "Differences that small may not be meaningful.", "",
              "_Educational report on a bundled dataset; not a production or medical decision tool._"]
    text = "\n".join(lines) + "\n"
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text
