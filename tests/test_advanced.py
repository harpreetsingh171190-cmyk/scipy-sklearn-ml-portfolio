import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from src.advanced import (
    check_api_key,
    detect_anomalies,
    detect_drift,
    population_stability_index,
    profile_dataframe,
    tune_threshold,
)
from src.cli import main


def test_profile_flags_problems():
    df = pd.DataFrame({"a": [1, 2, 2, None] * 10, "const": [1] * 40, "y": ["x"] * 38 + ["z"] * 2})
    rep = profile_dataframe(pd.concat([df, df.iloc[:3]]), target="y")
    assert rep["duplicate_rows"] >= 3 and "const" in rep["constant_columns"]
    assert rep["missing_fraction"]["a"] > 0.2 and any("imbalanced" in w for w in rep["warnings"])
    with pytest.raises(ValueError):
        profile_dataframe(df, target="missing")
    with pytest.raises(ValueError):
        profile_dataframe(pd.DataFrame())


def test_psi_detects_shift():
    rng = np.random.RandomState(0)
    base = rng.normal(0, 1, 2000)
    assert population_stability_index(base, rng.normal(0, 1, 2000)) < 0.1
    assert population_stability_index(base, rng.normal(2, 1, 2000)) > 0.25
    with pytest.raises(ValueError):
        population_stability_index([1.0], [1.0, 2.0])


def test_detect_drift_table():
    rng = np.random.RandomState(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(5, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(8, 1, 500)})
    t = detect_drift(ref, cur)
    assert t.iloc[0]["feature"] == "y" and t.iloc[0]["drift"] == "major"
    assert t.set_index("feature").loc["x", "drift"] == "stable"
    with pytest.raises(ValueError):
        detect_drift(ref, pd.DataFrame({"z": [1.0]}))


def test_anomalies_find_planted_outlier():
    X = np.vstack([np.random.RandomState(0).normal(0, 1, (200, 3)), [[25, 25, 25]]])
    res = detect_anomalies(X, contamination=0.02)
    assert res.iloc[0]["index"] == 200 and bool(res.iloc[0]["is_anomaly"])
    assert 0 < res["is_anomaly"].sum() <= 8
    assert len(detect_anomalies(load_wine().data)) == 178
    for bad in (np.zeros((3, 2)), np.array([[np.nan, 1.0]] * 20)):
        with pytest.raises(ValueError):
            detect_anomalies(bad)
    with pytest.raises(ValueError):
        detect_anomalies(X, contamination=0.9)


def test_threshold_tuning_reduces_cost_when_misses_are_expensive():
    r = tune_threshold("breast_cancer", cost_fn=10, cost_fp=1, cv=3)
    assert r["best_cost"] <= r["default_cost"] and 0.05 <= r["best_threshold"] <= 0.95
    assert r["best_threshold"] < 0.5 and len(r["table"]) == 91
    with pytest.raises(ValueError):
        tune_threshold("iris")
    with pytest.raises(ValueError):
        tune_threshold("breast_cancer", cost_fn=0)


def test_api_key_check():
    assert check_api_key(None, None) and check_api_key("x", "")
    assert check_api_key("secret", "secret")
    assert not check_api_key("wrong", "secret") and not check_api_key(None, "secret")


def test_cli_new_commands(tmp_path, capsys):
    ref = tmp_path / "ref.csv"; cur = tmp_path / "cur.csv"
    pd.DataFrame({"v": np.random.RandomState(0).normal(0, 1, 300)}).to_csv(ref, index=False)
    pd.DataFrame({"v": np.random.RandomState(1).normal(3, 1, 300)}).to_csv(cur, index=False)
    assert main(["drift", "--reference", str(ref), "--current", str(cur)]) == 0
    assert "major" in capsys.readouterr().out
    assert main(["anomaly", "--dataset", "wine"]) == 0
    assert main(["threshold", "--cost-fn", "8"]) == 0
    assert main(["profile", "--path", str(ref)]) == 0
    assert main(["threshold", "--dataset", "iris"]) == 1
    assert main(["profile", "--path", str(tmp_path / "none.csv")]) == 1
