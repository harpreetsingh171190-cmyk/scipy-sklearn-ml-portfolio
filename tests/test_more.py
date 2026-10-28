import numpy as np
import pytest

from src.cli import main
from src.more import (
    calibration_report,
    compare_ensembles,
    expected_calibration_error,
    pca_summary,
    select_features,
)


def test_ensembles_include_voting_and_stacking():
    df = compare_ensembles("iris", cv=3)
    assert {"voting_soft", "stacking"} <= set(df["model"])
    assert df["accuracy_mean"].between(0, 1).all()
    with pytest.raises(ValueError):
        compare_ensembles("iris", cv=1)


def test_ece_perfect_and_bad():
    assert expected_calibration_error([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == 0.0
    assert expected_calibration_error([0, 0, 0, 0], [1.0] * 4) == 1.0
    with pytest.raises(ValueError):
        expected_calibration_error([], [])


def test_calibration_report_and_errors():
    r = calibration_report("breast_cancer", "random_forest")
    assert 0 <= r["raw_brier"] <= 1 and 0 <= r["calibrated_ece"] <= 1
    with pytest.raises(ValueError):
        calibration_report("iris")
    with pytest.raises(ValueError):
        calibration_report("breast_cancer", method="bad")


def test_pca_and_select():
    r = pca_summary("wine", 0.95)
    assert 1 <= r["components_needed"] <= r["n_features"]
    assert r["cumulative"] == sorted(r["cumulative"])
    df = select_features("wine", k=3)
    assert int(df["selected"].sum()) == 3 and df["score"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        select_features("wine", k=99)
    with pytest.raises(ValueError):
        pca_summary("wine", 0)


def test_cli_new_commands(capsys):
    for argv in (["ensemble", "--cv", "3"], ["calibrate"], ["pca"], ["select", "--k", "3"]):
        assert main(argv) == 0
    assert "stacking" in capsys.readouterr().out


def test_shapley_efficiency_and_signal():
    from src.explain import explain_row, shapley_values
    from src.ml_portfolio import build_model, load_dataset
    d = load_dataset("breast_cancer")
    model = build_model("logistic_regression").fit(d.data, d.target)
    res = shapley_values(model, d.data[0], d.data[:30], n_permutations=10)
    assert abs(res["values"].sum() - (res["prediction"] - res["base_value"])) < 1e-9
    t = explain_row(model, d.data[0], d.data[:30], list(d.feature_names), top_k=3)["table"]
    assert len(t) == 3 and abs(t["contribution"]).is_monotonic_decreasing
    with pytest.raises(ValueError):
        shapley_values(model, d.data[0][:5], d.data[:30])
    with pytest.raises(ValueError):
        shapley_values(model, d.data[0], d.data[:30], n_permutations=0)


def test_service_explain_and_ranking(tmp_path):
    from src.serving import ModelService
    svc = ModelService(tmp_path)
    out = svc.explain_tabular("iris", [5.1, 3.5, 1.4, 0.2], top_k=2, n_permutations=10)
    assert out["class_name"] == "setosa" and len(out["contributions"]) == 2
    assert svc.feature_ranking("wine", 3)[0]["selected"] is True
    assert svc.pca("wine")["components_needed"] >= 1
    with pytest.raises(ValueError):
        svc.explain_tabular("iris", [1, 2])
    with pytest.raises(ValueError):
        svc.explain_tabular("iris", [5.1, 3.5, 1.4, 0.2], top_k=0)


def test_api_new_endpoints(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import api.main as api_main
    from src.serving import ModelService
    monkeypatch.setattr(api_main, "service", ModelService(tmp_path))
    c = TestClient(api_main.app)
    ok = c.post("/explain", json={"dataset": "iris", "row": [5.1, 3.5, 1.4, 0.2], "top_k": 2})
    assert ok.status_code == 200 and len(ok.json()["contributions"]) == 2
    assert c.post("/explain", json={"dataset": "iris", "row": [1.0]}).status_code == 422
    assert c.get("/feature-ranking?dataset=wine&k=3").status_code == 200
    assert c.get("/feature-ranking?dataset=wine&k=99").status_code == 422
    assert c.get("/pca?dataset=wine&variance=0").status_code == 422


def test_make_imbalanced_and_strategies():
    from src.imbalance import STRATEGIES, compare_imbalance_strategies, make_imbalanced
    from src.ml_portfolio import load_dataset
    d = load_dataset("breast_cancer")
    X, y = make_imbalanced(d.data, d.target, 0.1)
    assert 0.07 < y.mean() < 0.13 and len(X) == len(y)
    df = compare_imbalance_strategies("breast_cancer", 0.1)
    assert set(df["strategy"]) == set(STRATEGIES)
    assert df["minority_recall"].between(0, 1).all() and df["pr_auc"].between(0, 1).all()
    with pytest.raises(ValueError):
        make_imbalanced(d.data, d.target, 0.9)
    with pytest.raises(ValueError):
        make_imbalanced(load_dataset("iris").data, load_dataset("iris").target)
    with pytest.raises(ValueError):
        compare_imbalance_strategies("breast_cancer", model="knn")


def test_regression_explanation_efficiency():
    from src.explain import explain_regression
    from src.extras import train_regressor
    r = train_regressor("ridge")
    ex = explain_regression(r["model"], r["X_test"][0], r["X_test"], r["feature_names"], n_permutations=10, top_k=3)
    assert len(ex["table"]) == 3
    full = explain_regression(r["model"], r["X_test"][0], r["X_test"], r["feature_names"], n_permutations=10)
    assert abs(full["values"].sum() - (full["prediction"] - full["base_value"])) < 1e-6
    with pytest.raises(ValueError):
        explain_regression(r["model"], r["X_test"][0], r["X_test"][:, :3])


def test_cli_imbalance_and_regression_explain():
    assert main(["imbalance", "--minority-fraction", "0.1"]) == 0
    assert main(["explain-regression", "--row", "1", "--top", "3"]) == 0
    assert main(["explain-regression", "--row", "9999"]) == 1


def test_api_imbalance_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import api.main as api_main
    from src.serving import ModelService
    monkeypatch.setattr(api_main, "service", ModelService(tmp_path))
    c = TestClient(api_main.app)
    ok = c.get("/imbalance?dataset=breast_cancer&minority_fraction=0.1")
    assert ok.status_code == 200 and len(ok.json()["results"]) == 4
    assert c.get("/imbalance?dataset=iris").status_code == 422


def _sample_csv(tmp_path):
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({"age": rng.integers(20, 70, n).astype(float), "income": rng.normal(50, 15, n),
                       "city": rng.choice(["A", "B", "C"], n)})
    df.loc[rng.choice(n, 10, replace=False), "income"] = np.nan
    df["churn"] = np.where(df.income.fillna(50) < 42, "yes", "no")
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_csv_explain_and_imbalance(tmp_path):
    from src.csvtools import csv_imbalance, explain_csv_row
    path = _sample_csv(tmp_path)
    ex = explain_csv_row(path, "churn", row=1, top_k=2, n_permutations=10)
    assert len(ex["table"]) == 2 and ex["predicted_class"] in {"yes", "no"}
    assert 0 <= ex["prediction"] <= 1
    assert set(csv_imbalance(path, "churn")["strategy"]) == {"baseline", "class_weight", "oversample", "undersample"}
    with pytest.raises(ValueError):
        explain_csv_row(path, "churn", row=9999)
    with pytest.raises(ValueError):
        explain_csv_row(path, "nope")
    with pytest.raises(ValueError):
        csv_imbalance(path, "city")


def test_cli_csv_commands(tmp_path):
    path = str(_sample_csv(tmp_path))
    assert main(["csv-explain", "--path", path, "--target", "churn", "--row", "0", "--top", "2"]) == 0
    assert main(["csv-imbalance", "--path", path, "--target", "churn"]) == 0
    assert main(["csv-explain", "--path", path, "--target", "missing"]) == 1
