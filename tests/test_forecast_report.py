import numpy as np
import pandas as pd
import pytest

from src.cli import main
from src.forecast import MODEL_NAMES, backtest, demo_series, detect_period, forecast_series, load_series
from src.htmlreport import build_comparison_html, write_comparison_html
from src.serving import MAX_CSV_ROWS, ModelService, parse_csv


def _csv_text(n=60):
    y = demo_series(n)
    return pd.DataFrame({"day": y.index.strftime("%Y-%m-%d"), "sales": y.values}).to_csv(index=False)


def test_detect_period_and_models_beat_naive():
    y = demo_series(96)
    assert detect_period(y) == 12
    table, errs = backtest(y, horizon=12, n_splits=3)
    assert set(table["model"]) == set(MODEL_NAMES) and errs["naive"].shape == (3, 12)
    mae = table.set_index("model")["mae"]
    assert mae["ridge_lags"] < mae["naive"] and mae["seasonal_naive"] < mae["naive"]


def test_forecast_shape_dates_and_bands():
    y = demo_series(60)
    res = forecast_series(y, horizon=6)
    f = res["forecast"]
    assert len(f) == 6 and f.index[0] > y.index[-1] and (f["lower"] <= f["forecast"]).all()
    assert (f["forecast"] <= f["upper"]).all() and res["model"] in MODEL_NAMES
    assert forecast_series(y, 3, model="naive")["model"] == "naive"
    plain = pd.Series(np.arange(40, dtype=float) + np.sin(np.arange(40)))
    assert isinstance(forecast_series(plain, 4)["forecast"].index, pd.RangeIndex)


def test_forecast_validation():
    y = demo_series(60)
    with pytest.raises(ValueError):
        forecast_series(y, horizon=0)
    with pytest.raises(ValueError):
        forecast_series(y, horizon=12, model="magic")
    with pytest.raises(ValueError):
        forecast_series(demo_series(60)[:10], horizon=3)
    with pytest.raises(ValueError):
        forecast_series(y, horizon=55)
    with pytest.raises(ValueError):
        load_series(pd.DataFrame({"a": [1.0] * 30}), "missing")
    dup = pd.DataFrame({"d": ["2020-01-01"] * 30, "v": range(30)})
    with pytest.raises(ValueError):
        load_series(dup, "v", "d")


def test_html_report_is_self_contained_and_escaped(tmp_path):
    html_text = build_comparison_html("iris", cv=3, include_regression=False)
    assert html_text.startswith("<!doctype html>") and "<svg" in html_text and "Leaderboard" in html_text
    assert "<script" not in html_text and "http://" not in html_text and "https://" not in html_text
    out = write_comparison_html(tmp_path / "sub" / "r.html", "wine", cv=3, include_regression=False)
    assert out.exists() and "wine" in out.read_text(encoding="utf-8")


def test_parse_csv_limits():
    assert len(parse_csv("a,b\n1,2\n3,4\n")) == 2
    for bad in ("", "   ", "a,b\n"):
        with pytest.raises(ValueError):
            parse_csv(bad)
    big = "a\n" + "1\n" * (MAX_CSV_ROWS + 1)
    with pytest.raises(ValueError):
        parse_csv(big)


def test_service_csv_endpoints_logic(tmp_path):
    svc = ModelService(tmp_path)
    text = _csv_text()
    fc = svc.csv_forecast(text, "sales", "day", horizon=4)
    assert len(fc["forecast"]) == 4 and fc["backtest"][0]["model"] in MODEL_NAMES
    assert svc.csv_profile(text)["rows"] == 60
    with pytest.raises(ValueError):
        svc.csv_forecast(text, "nope")
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = np.where(df.x1 + rng.normal(0, .5, n) > 0.9, "yes", "no")
    body = df.to_csv(index=False)
    ex = svc.csv_explain(body, "y", row=0, top_k=2)
    assert len(ex["contributions"]) == 2
    assert len(svc.csv_imbalance(body, "y")) == 4
    with pytest.raises(ValueError):
        svc.csv_explain(body, "y", top_k=0)
    assert "<svg" in svc.html_report("iris", 2)
    with pytest.raises(ValueError):
        svc.html_report("iris", 99)


def test_cli_forecast_and_report(tmp_path, capsys):
    assert main(["forecast", "--horizon", "3"]) == 0
    path = tmp_path / "ts.csv"
    path.write_text(_csv_text(), encoding="utf-8")
    assert main(["forecast", "--path", str(path), "--value", "sales", "--date", "day", "--horizon", "3"]) == 0
    assert main(["forecast", "--path", str(path), "--value", "nope"]) == 1
    out = tmp_path / "r.html"
    assert main(["html-report", "--dataset", "iris", "--cv", "3", "--no-regression", "--out", str(out)]) == 0
    assert out.exists()


def test_api_csv_upload_and_report(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import api.main as api_main
    monkeypatch.setattr(api_main, "service", ModelService(tmp_path))
    c = TestClient(api_main.app)
    hdr = {"Content-Type": "text/csv"}
    r = c.post("/csv/forecast?value=sales&date=day&horizon=3", content=_csv_text(), headers=hdr)
    assert r.status_code == 200 and len(r.json()["forecast"]) == 3
    assert c.post("/csv/forecast?value=nope", content=_csv_text(), headers=hdr).status_code == 422
    assert c.post("/csv/profile", content="", headers=hdr).status_code == 422
    assert c.post("/csv/profile", content=_csv_text(), headers=hdr).json()["rows"] == 60
    assert c.post("/csv/profile", content=b"\xff\xfe\x00bad", headers=hdr).status_code == 422
    html_resp = c.get("/report.html?dataset=iris&cv=2")
    assert html_resp.status_code == 200 and "text/html" in html_resp.headers["content-type"]
    assert c.get("/report.html?dataset=iris&cv=99").status_code == 422
