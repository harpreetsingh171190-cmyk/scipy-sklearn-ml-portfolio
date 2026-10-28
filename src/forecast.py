"""v3.7: time-series forecasting with honest rolling-origin backtests.

Models: naive, seasonal naive, Ridge on lag features, and Random Forest on lagged differences.
Multi-step forecasts are recursive; the uncertainty band is a rough one built from backtest errors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["demo_series", "load_series", "detect_period", "backtest", "forecast_series", "MODEL_NAMES"]

MODEL_NAMES = ["naive", "seasonal_naive", "ridge_lags", "random_forest_lags"]
MIN_POINTS = 24
MAX_HORIZON = 365


def demo_series(n: int = 120, period: int = 12, seed: int = 42) -> pd.Series:
    """Synthetic monthly-style series: trend + seasonality + noise (no download needed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 100 + 0.8 * t + 15 * np.sin(2 * np.pi * t / period) + rng.normal(0, 3, n)
    return pd.Series(y, index=pd.date_range("2016-01-01", periods=n, freq="MS"), name="value")


def load_series(df: pd.DataFrame, value: str, date: str | None = None) -> pd.Series:
    """Pick a numeric value column (and optional date column) from a DataFrame."""
    if value not in df.columns:
        raise ValueError(f"Value column {value!r} not found. Columns: {list(df.columns)}")
    y = pd.to_numeric(df[value], errors="coerce")
    index = None
    if date is not None:
        if date not in df.columns:
            raise ValueError(f"Date column {date!r} not found. Columns: {list(df.columns)}")
        index = pd.to_datetime(df[date], errors="coerce")
        keep = index.notna() & y.notna()
        y, index = y[keep], index[keep]
        order = np.argsort(index.to_numpy())
        y, index = y.iloc[order], pd.DatetimeIndex(index.iloc[order])
        if index.has_duplicates:
            raise ValueError("Date column contains duplicate dates")
    else:
        y = y.dropna()
    y = y.astype(float)
    y.index = index if index is not None else pd.RangeIndex(len(y))
    y.name = value
    if len(y) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} valid observations, got {len(y)}")
    return y


def detect_period(y, max_period: int | None = None) -> int:
    """Seasonal period of the detrended series (>=2), else 1.

    Picks the smallest autocorrelation peak that is nearly as strong as the strongest one, so a
    series with period 12 is not reported as 24 just because noise made lag 24 slightly higher.
    """
    v = np.asarray(y, dtype=float)
    d = np.diff(v)
    d = d - d.mean()
    top = min(max_period or len(d) // 3, len(d) // 3)
    if top < 2 or d.std() == 0:
        return 1
    lags = np.arange(2, top + 1)
    acf = np.array([np.corrcoef(d[:-k], d[k:])[0, 1] for k in lags])
    acf = np.nan_to_num(acf, nan=-1.0)
    peak = acf.max()
    if peak <= 0.3:
        return 1
    padded = np.concatenate([[-np.inf], acf, [-np.inf]])
    is_peak = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:])
    candidates = lags[is_peak & (acf >= 0.9 * peak)]
    return int(candidates[0])


def _lags(season: int, n: int) -> int:
    return int(max(2, min(max(season, 5), n // 3)))


def _naive(y, h, season, rs):
    return np.repeat(y[-1], h)


def _seasonal_naive(y, h, season, rs):
    if season <= 1 or season > len(y):
        return _naive(y, h, season, rs)
    return np.array([y[-season + (i % season)] for i in range(h)])


def _lag_matrix(v: np.ndarray, lags: int):
    X = np.array([v[i - lags:i] for i in range(lags, len(v))])
    return X, v[lags:]


def _ridge(y, h, season, rs):
    lags = _lags(season, len(y))
    X, t = _lag_matrix(y, lags)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, t)
    hist = list(y[-lags:])
    out = []
    for _ in range(h):
        nxt = float(model.predict(np.array(hist[-lags:])[None, :])[0])
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def _forest(y, h, season, rs):
    d = np.diff(y)
    lags = _lags(season, len(d))
    X, t = _lag_matrix(d, lags)
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=rs, n_jobs=1).fit(X, t)
    hist, level, out = list(d[-lags:]), float(y[-1]), []
    for _ in range(h):
        step = float(model.predict(np.array(hist[-lags:])[None, :])[0])
        hist.append(step)
        level += step
        out.append(level)
    return np.array(out)


_FUNCS = {"naive": _naive, "seasonal_naive": _seasonal_naive,
          "ridge_lags": _ridge, "random_forest_lags": _forest}


def _check(y: pd.Series, horizon: int) -> np.ndarray:
    v = np.asarray(y, dtype=float)
    if len(v) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} observations, got {len(v)}")
    if not np.isfinite(v).all():
        raise ValueError("Series contains NaN or infinite values")
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON}")
    return v


def backtest(y: pd.Series, horizon: int = 12, n_splits: int = 3, season: int | None = None,
             random_state: int = 42) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rolling-origin evaluation. Returns (metrics table, per-model error arrays [splits x horizon])."""
    v = _check(y, horizon)
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    season = detect_period(v) if season is None else int(season)
    step = max(1, horizon // 2)
    origins = [len(v) - horizon - i * step for i in range(n_splits)][::-1]
    origins = [o for o in origins if o >= MIN_POINTS - 4]
    if not origins:
        raise ValueError("Series too short for this horizon; lower the horizon")
    errors: dict[str, list] = {m: [] for m in MODEL_NAMES}
    for o in origins:
        actual = v[o:o + horizon]
        for name, fn in _FUNCS.items():
            errors[name].append(actual - fn(v[:o], horizon, season, random_state))
    rows, err_arrays = [], {}
    scale = np.mean(np.abs(np.diff(v))) or 1.0
    for name, e in errors.items():
        e = np.array(e)
        err_arrays[name] = e
        rows.append({"model": name, "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "mase": float(np.abs(e).mean() / scale), "origins": len(origins)})
    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return table, err_arrays


def _future_index(index: pd.Index, horizon: int) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        freq = pd.infer_freq(index) if len(index) >= 3 else None
        if freq:
            return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]
        delta = pd.Series(index).diff().median()
        return pd.DatetimeIndex([index[-1] + delta * (i + 1) for i in range(horizon)])
    return pd.RangeIndex(len(index), len(index) + horizon)


def forecast_series(y: pd.Series, horizon: int = 12, model: str = "auto", season: int | None = None,
                    n_splits: int = 3, random_state: int = 42) -> dict[str, Any]:
    """Forecast `horizon` steps ahead with the chosen model (or the best on the backtest)."""
    if model != "auto" and model not in _FUNCS:
        raise ValueError(f"model must be 'auto' or one of {MODEL_NAMES}")
    v = _check(y, horizon)
    season = detect_period(v) if season is None else int(season)
    if season < 1:
        raise ValueError("season must be at least 1")
    table, errs = backtest(y, horizon, n_splits, season, random_state)
    chosen = table.loc[0, "model"] if model == "auto" else model
    pred = _FUNCS[chosen](v, horizon, season, random_state)
    sigma = np.sqrt((errs[chosen] ** 2).mean(axis=0))
    frame = pd.DataFrame({"forecast": pred, "lower": pred - 1.96 * sigma, "upper": pred + 1.96 * sigma},
                         index=_future_index(y.index, horizon))
    return {"model": chosen, "season": season, "horizon": horizon, "forecast": frame, "backtest": table,
            "note": "Bands are rough (from backtest errors); forecasts are not guarantees."}
