"""Framework-free serving logic (testable without FastAPI). api/main.py is a thin wrapper."""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .advanced import profile_dataframe
from .csvtools import csv_imbalance, explain_csv_row
from .explain import dataset_background, explain_row
from .forecast import forecast_series, load_series
from .htmlreport import build_comparison_html
from .imbalance import compare_imbalance_strategies
from .ml_portfolio import DATASETS, MODELS, load_dataset, load_model, save_model, train_model
from .more import pca_summary, select_features
from .text import predict_texts, train_text_model

MAX_BATCH = 100
MAX_TEXT_CHARS = 2000
MAX_CSV_BYTES = 2_000_000
MAX_CSV_ROWS = 50_000


def parse_csv(text: str) -> pd.DataFrame:
    """Parse uploaded CSV text with size and row limits."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("CSV body is empty")
    if len(text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError(f"CSV is larger than {MAX_CSV_BYTES // 1_000_000} MB")
    try:
        df = pd.read_csv(io.StringIO(text), nrows=MAX_CSV_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if len(df) > MAX_CSV_ROWS:
        raise ValueError(f"CSV has more than {MAX_CSV_ROWS} rows")
    if df.empty:
        raise ValueError("CSV has no data rows")
    return df


class ModelService:
    """Holds one tabular model per dataset and one text model; trains on first use."""

    def __init__(self, model_dir: str | Path = "artifacts/serving", default_model: str = "logistic_regression"):
        self.model_dir = Path(model_dir)
        self.default_model = default_model
        self._tabular: dict[str, Any] = {}
        self._text = None

    def tabular_model(self, dataset: str):
        if dataset not in self._tabular:
            path = self.model_dir / f"{dataset}_{self.default_model}.joblib"
            if path.exists():
                self._tabular[dataset] = load_model(path)  # checksum-verified
            else:
                result = train_model(dataset, self.default_model)
                save_model(result["model"], path, {"dataset": dataset, "accuracy": result["accuracy"]})
                self._tabular[dataset] = result["model"]
        return self._tabular[dataset]

    def text_model(self):
        if self._text is None:
            self._text = train_text_model()["model"]
        return self._text

    def predict_tabular(self, dataset: str, rows: list[list[float]]) -> list[dict[str, Any]]:
        data = load_dataset(dataset)
        width = data.data.shape[1]
        if not rows or len(rows) > MAX_BATCH:
            raise ValueError(f"rows must contain between 1 and {MAX_BATCH} items")
        for r in rows:
            if len(r) != width:
                raise ValueError(f"each row needs exactly {width} features for {dataset!r}")
        model = self.tabular_model(dataset)
        names = [str(n) for n in getattr(data, "target_names", [])]
        preds = model.predict(rows)
        probs = model.predict_proba(rows) if hasattr(model, "predict_proba") else None
        out = []
        for i, p in enumerate(preds):
            item = {"class_index": int(p), "class_name": names[int(p)] if names else str(p)}
            if probs is not None:
                item["probabilities"] = [round(float(x), 4) for x in probs[i]]
            out.append(item)
        return out

    def explain_tabular(self, dataset: str, row: list[float], class_index: int | None = None,
                        top_k: int = 5, n_permutations: int = 30) -> dict[str, Any]:
        """SHAP-style per-feature contributions for one row."""
        data = load_dataset(dataset)
        if len(row) != data.data.shape[1]:
            raise ValueError(f"row needs exactly {data.data.shape[1]} features for {dataset!r}")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        model = self.tabular_model(dataset)
        res = explain_row(model, row, dataset_background(dataset), [str(n) for n in getattr(data, "feature_names", [])] or None,
                          class_index, n_permutations, top_k=top_k)
        names = [str(n) for n in getattr(data, "target_names", [])]
        ci = res["class_index"]
        return {"dataset": dataset, "class_index": ci, "class_name": names[ci] if names else str(ci),
                "base_value": round(res["base_value"], 4), "prediction": round(res["prediction"], 4),
                "contributions": [{"feature": r.feature, "value": round(float(r.value), 4),
                                   "contribution": round(float(r.contribution), 4)}
                                  for r in res["table"].itertuples()]}

    def feature_ranking(self, dataset: str, k: int = 5, method: str = "f_test") -> list[dict[str, Any]]:
        df = select_features(dataset, k, method)
        return [{"feature": r.feature, "score": round(float(r.score), 4), "selected": bool(r.selected)}
                for r in df.itertuples()]

    def imbalance(self, dataset: str = "breast_cancer", minority_fraction: float = 0.1) -> list[dict[str, Any]]:
        df = compare_imbalance_strategies(dataset, minority_fraction)
        return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                for r in df.to_dict("records")]

    @staticmethod
    def _with_csv_file(text: str, fn):
        df = parse_csv(text)
        fd, name = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                df.to_csv(handle, index=False)
            return fn(name)
        finally:
            os.unlink(name)

    def csv_profile(self, text: str, target: str | None = None) -> dict[str, Any]:
        return profile_dataframe(parse_csv(text), target)

    def csv_explain(self, text: str, target: str, row: int = 0, top_k: int = 8) -> dict[str, Any]:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        ex = self._with_csv_file(text, lambda path: explain_csv_row(path, target, row, top_k, 20))
        return {"predicted_class": ex["predicted_class"], "prediction": round(ex["prediction"], 4),
                "base_value": round(ex["base_value"], 4), "model_accuracy": round(ex["accuracy"], 4),
                "contributions": [{"feature": r.feature, "value": r.value, "contribution": round(float(r.contribution), 4)}
                                  for r in ex["table"].itertuples()]}

    def csv_imbalance(self, text: str, target: str, model: str = "logistic_regression") -> list[dict[str, Any]]:
        df = self._with_csv_file(text, lambda path: csv_imbalance(path, target, model))
        return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                for r in df.to_dict("records")]

    def csv_forecast(self, text: str, value: str, date: str | None = None, horizon: int = 12,
                     model: str = "auto", season: int | None = None) -> dict[str, Any]:
        y = load_series(parse_csv(text), value, date)
        res = forecast_series(y, horizon, model, season)
        fc = res["forecast"]
        return {"model": res["model"], "season": res["season"], "horizon": horizon, "note": res["note"],
                "forecast": [{"step": i + 1, "time": str(ts), "forecast": round(float(r.forecast), 4),
                              "lower": round(float(r.lower), 4), "upper": round(float(r.upper), 4)}
                             for i, (ts, r) in enumerate(fc.iterrows())],
                "backtest": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                             for r in res["backtest"].to_dict("records")]}

    def html_report(self, dataset: str = "iris", cv: int = 3) -> str:
        if not 2 <= cv <= 5:
            raise ValueError("cv must be between 2 and 5")
        return build_comparison_html(dataset, cv=cv)

    def pca(self, dataset: str, variance: float = 0.95) -> dict[str, Any]:
        return pca_summary(dataset, variance)

    def predict_text(self, texts: list[str]) -> list[dict[str, Any]]:
        if len(texts) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} texts per request")
        if any(isinstance(t, str) and len(t) > MAX_TEXT_CHARS for t in texts):
            raise ValueError(f"each text must be at most {MAX_TEXT_CHARS} characters")
        return predict_texts(self.text_model(), texts)

    def info(self) -> dict[str, Any]:
        return {"datasets": sorted(DATASETS), "models": sorted(MODELS),
                "default_model": self.default_model, "text_labels": ["ham", "spam"]}
