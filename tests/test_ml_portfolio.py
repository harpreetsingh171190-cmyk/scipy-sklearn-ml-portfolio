import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris

from src.cli import main
from src.ml_portfolio import (
    DATASETS,
    MODELS,
    build_iris_pipeline,
    build_model,
    compare_models,
    feature_importance,
    load_dataset,
    load_model,
    log_experiment,
    read_experiments,
    save_model,
    train_iris_model,
    train_model,
)


def test_pipeline_trains_and_predicts():
    data = load_iris()
    model = build_iris_pipeline().fit(data.data, data.target)
    predictions = model.predict(data.data[:8])
    assert predictions.shape == (8,)
    assert set(predictions).issubset(set(data.target))


def test_training_is_reproducible_and_metrics_are_valid():
    first, second = train_iris_model(), train_iris_model()
    assert first["accuracy"] == second["accuracy"]
    assert 0.8 <= first["accuracy"] <= 1.0
    assert first["confusion_matrix"].shape == (3, 3)
    assert len(first["predictions"]) == len(first["y_test"])


def test_invalid_test_size_raises():
    with pytest.raises(ValueError):
        train_iris_model(test_size=0.8)


@pytest.mark.parametrize("dataset", ["iris", "wine", "breast_cancer"])
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_every_model_trains_on_every_small_dataset(dataset, model_name):
    result = train_model(dataset, model_name)
    assert 0.7 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        load_dataset("nope")
    with pytest.raises(ValueError):
        build_model("nope")


def test_digits_dataset_is_registered():
    assert "digits" in DATASETS
    assert load_dataset("digits").data.shape[1] == 64


def test_compare_models_leaderboard():
    table = compare_models("iris", models=["logistic_regression", "knn"], cv=3)
    assert list(table["model"].sort_values()) == ["knn", "logistic_regression"]
    assert table["accuracy_mean"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        compare_models("iris", cv=1)


def test_feature_importance_columns():
    table = feature_importance(train_model("iris", "random_forest"), n_repeats=3)
    assert len(table) == 4
    assert {"feature", "importance_mean", "importance_std"} <= set(table.columns)


def test_save_load_round_trip_with_checksum(tmp_path):
    data = load_iris()
    model = build_iris_pipeline().fit(data.data, data.target)
    path = save_model(model, tmp_path / "models" / "iris.joblib", {"dataset": "iris"})
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    assert len(meta["sha256"]) == 64 and meta["dataset"] == "iris"
    loaded = load_model(path)
    np.testing.assert_array_equal(model.predict(data.data[:10]), loaded.predict(data.data[:10]))


def test_tampered_model_is_rejected(tmp_path):
    data = load_iris()
    path = save_model(build_iris_pipeline().fit(data.data, data.target), tmp_path / "m.joblib")
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_model(path)


def test_experiment_log(tmp_path):
    log = tmp_path / "runs.jsonl"
    assert read_experiments(log).empty
    log_experiment(train_model("wine", "knn"), log)
    log_experiment(train_model("iris", "svm_rbf"), log)
    runs = read_experiments(log)
    assert len(runs) == 2 and set(runs["dataset"]) == {"wine", "iris"}


def test_cli_train_predict_and_compare(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "m.joblib"
    assert main(["train", "--dataset", "iris", "--out", str(out)]) == 0
    assert main(["predict", "--model-path", str(out), "--features", "5.1,3.5,1.4,0.2"]) == 0
    assert '"prediction"' in capsys.readouterr().out
    assert main(["compare", "--dataset", "iris", "--cv", "3"]) == 0
    assert main(["runs"]) == 0


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["train", "--test-size", "0.9", "--out", str(tmp_path / "x.joblib")]) == 1
    assert main(["predict", "--model-path", str(tmp_path / "missing.joblib"), "--features", "1,2,3,4"]) == 1


def test_notebooks_are_valid_json():
    root = Path(__file__).resolve().parents[1]
    notebook_files = list((root / "notebooks").glob("*.ipynb"))
    assert len(notebook_files) >= 33
    for notebook in notebook_files:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        assert "cells" in document, notebook.name
        assert document.get("nbformat", 0) >= 4, notebook.name
