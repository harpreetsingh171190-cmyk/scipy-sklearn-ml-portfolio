import pytest

from src.cli import main
from src.extras import (
    cluster_analysis,
    compare_regressors,
    generate_report,
    train_from_csv,
    train_regressor,
    tune_model,
)


def test_tune_grid_and_random():
    g = tune_model("iris", "knn", cv=3)
    assert 0.8 <= g["test_score"] <= 1.0 and "classifier__n_neighbors" in g["best_params"]
    r = tune_model("wine", "logistic_regression", cv=3, method="random", n_iter=3)
    assert len(r["results"]) == 3


def test_tune_validation():
    with pytest.raises(ValueError):
        tune_model("iris", "nope")
    with pytest.raises(ValueError):
        tune_model("iris", "knn", method="bayes")


def test_regressors():
    board = compare_regressors(cv=3)
    assert len(board) == 3 and board["r2_mean"].is_monotonic_decreasing
    r = train_regressor("ridge")
    assert r["rmse"] >= r["mae"] > 0 and r["r2"] > 0.2
    with pytest.raises(ValueError):
        train_regressor("nope")


def test_cluster_analysis():
    t = cluster_analysis("iris", range(2, 5))
    assert list(t["k"]) == [2, 3, 4] and t["silhouette"].between(-1, 1).all()
    with pytest.raises(ValueError):
        cluster_analysis("iris", range(1, 3))


def test_train_from_csv_mixed_types_with_missing(tmp_path):
    rows = ["age,city,score,label"]
    for i in range(60):
        age = "" if i % 10 == 0 else 20 + i % 30
        rows.append(f"{age},{'Amritsar' if i % 2 else 'Delhi'},{i % 7},{'yes' if i % 3 else 'no'}")
    path = tmp_path / "d.csv"
    path.write_text("\n".join(rows))
    r = train_from_csv(path, "label")
    assert r["categorical"] == ["city"] and 0 <= r["accuracy"] <= 1
    with pytest.raises(ValueError):
        train_from_csv(path, "missing_col")


def test_report_written(tmp_path):
    out = tmp_path / "r" / "report.md"
    text = generate_report("iris", cv=3, path=out)
    assert out.read_text(encoding="utf-8") == text and "| Model |" in text


def test_new_cli_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["tune", "--dataset", "iris", "--model", "knn", "--cv", "3"]) == 0
    assert main(["regress"]) == 0
    assert main(["cluster", "--dataset", "iris"]) == 0
    assert main(["report", "--dataset", "iris", "--out", str(tmp_path / "r.md")]) == 0
    assert main(["csv", "--path", str(tmp_path / "none.csv"), "--target", "x"]) == 1
