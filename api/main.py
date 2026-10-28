"""FastAPI service. Run: uvicorn api.main:app --reload   (docs at /docs)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.advanced import check_api_key
from src.serving import MAX_BATCH, MAX_CSV_BYTES, ModelService

app = FastAPI(title="ML Portfolio API", version="3.8.0",
              description="Educational model-serving API. Not for medical or production decisions.")
service = ModelService()


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Optional auth: set ML_API_KEY in the environment to require an X-API-Key header."""
    if not check_api_key(x_api_key, os.environ.get("ML_API_KEY")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class TabularRequest(BaseModel):
    dataset: str = Field("iris", examples=["iris"])
    rows: list[list[float]] = Field(..., min_length=1, max_length=MAX_BATCH,
                                    examples=[[[5.1, 3.5, 1.4, 0.2]]])


class ExplainRequest(BaseModel):
    dataset: str = Field("iris", examples=["iris"])
    row: list[float] = Field(..., min_length=1, max_length=100, examples=[[5.1, 3.5, 1.4, 0.2]])
    class_index: int | None = Field(None, ge=0)
    top_k: int = Field(5, ge=1, le=50)


class TextRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                             examples=[["WINNER!! Claim your free prize now"]])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return service.info()


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: TabularRequest):
    try:
        return {"dataset": req.dataset, "predictions": service.predict_tabular(req.dataset, req.rows)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict-text", dependencies=[Depends(require_api_key)])
def predict_text(req: TextRequest):
    try:
        return {"predictions": service.predict_text(req.texts)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/explain", dependencies=[Depends(require_api_key)])
def explain(req: ExplainRequest):
    """SHAP-style contributions of each feature to the predicted (or chosen) class probability."""
    try:
        return service.explain_tabular(req.dataset, req.row, req.class_index, req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/feature-ranking", dependencies=[Depends(require_api_key)])
def feature_ranking(dataset: str = "wine", k: int = 5, method: str = "f_test"):
    try:
        return {"dataset": dataset, "features": service.feature_ranking(dataset, k, method)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/pca", dependencies=[Depends(require_api_key)])
def pca(dataset: str = "wine", variance: float = 0.95):
    try:
        return service.pca(dataset, variance)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/imbalance", dependencies=[Depends(require_api_key)])
def imbalance(dataset: str = "breast_cancer", minority_fraction: float = 0.1):
    """Compare class-imbalance strategies (baseline, class weights, over/under-sampling)."""
    try:
        return {"dataset": dataset, "results": service.imbalance(dataset, minority_fraction)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _csv_body(request: Request) -> str:
    """Read an uploaded CSV sent as the raw request body (Content-Type: text/csv)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV too large")
    raw = await request.body()
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV too large")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="CSV must be UTF-8 text") from exc


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/csv/profile", dependencies=[Depends(require_api_key)])
async def csv_profile(request: Request, target: str | None = None):
    """Upload a CSV as the raw body (curl --data-binary @file.csv) to get a data-quality profile."""
    return _run(service.csv_profile, await _csv_body(request), target)


@app.post("/csv/explain", dependencies=[Depends(require_api_key)])
async def csv_explain(request: Request, target: str, row: int = 0, top_k: int = 8):
    return _run(service.csv_explain, await _csv_body(request), target, row, top_k)


@app.post("/csv/imbalance", dependencies=[Depends(require_api_key)])
async def csv_imbalance(request: Request, target: str, model: str = "logistic_regression"):
    return {"results": _run(service.csv_imbalance, await _csv_body(request), target, model)}


@app.post("/csv/forecast", dependencies=[Depends(require_api_key)])
async def csv_forecast(request: Request, value: str, date: str | None = None, horizon: int = 12,
                       model: str = "auto", season: int | None = None):
    """Time-series forecast from an uploaded CSV with a numeric value column and optional date column."""
    return _run(service.csv_forecast, await _csv_body(request), value, date, horizon, model, season)


@app.get("/report.html", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
def report_html(dataset: str = "iris", cv: int = 3):
    """Self-contained HTML model-comparison report."""
    return HTMLResponse(_run(service.html_report, dataset, cv))
