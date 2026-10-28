import json
import re
from pathlib import Path

import pytest

from src.cli import main
from src.serving import MAX_BATCH, ModelService
from src.text import compare_text_models, load_sms, predict_texts, top_terms, train_text_model


def test_sms_loads_with_two_labels():
    df = load_sms()
    assert set(df["label"]) == {"ham", "spam"} and len(df) > 5000


def test_text_models_are_accurate_and_reproducible():
    a, b = train_text_model(), train_text_model()
    assert a["accuracy"] == b["accuracy"] and a["accuracy"] > 0.95 and a["spam_recall"] > 0.8
    for name in ["naive_bayes", "linear_svm"]:
        assert train_text_model(name)["macro_f1"] > 0.9
    with pytest.raises(ValueError):
        train_text_model("nope")


def test_compare_text_models_sorted():
    table = compare_text_models(cv=3)
    assert len(table) == 3 and table["macro_f1_mean"].is_monotonic_decreasing


def test_predictions_and_terms():
    model = train_text_model()["model"]
    out = predict_texts(model, ["WINNER!! Call now to claim your free prize", "See you at lunch tomorrow"])
    assert [o["label"] for o in out] == ["spam", "ham"] and out[0]["spam_score"] > out[1]["spam_score"]
    assert "free" in top_terms(model, 15)["spam"] or "call" in top_terms(model, 15)["spam"]
    with pytest.raises(ValueError):
        predict_texts(model, [])
    with pytest.raises(ValueError):
        predict_texts(model, ["   "])


def test_service_tabular_and_validation(tmp_path):
    svc = ModelService(tmp_path)
    out = svc.predict_tabular("iris", [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
    assert [o["class_name"] for o in out] == ["setosa", "virginica"]
    assert (tmp_path / "iris_logistic_regression.joblib.meta.json").exists()
    assert ModelService(tmp_path).predict_tabular("iris", [[5.1, 3.5, 1.4, 0.2]])[0]["class_index"] == 0
    for bad in ([], [[1, 2, 3]], [[0.0] * 4] * (MAX_BATCH + 1)):
        with pytest.raises(ValueError):
            svc.predict_tabular("iris", bad)
    with pytest.raises(ValueError):
        svc.predict_tabular("nope", [[1, 2, 3, 4]])


def test_service_text_limits(tmp_path):
    svc = ModelService(tmp_path)
    assert svc.predict_text(["free prize call now"])[0]["label"] in {"spam", "ham"}
    with pytest.raises(ValueError):
        svc.predict_text(["x" * 5000])
    assert "iris" in svc.info()["datasets"]


def test_cli_text_commands(capsys):
    assert main(["text-predict", "--text", "Free entry! Text WIN to 80086 now", "--text", "ok see you soon"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [x["label"] for x in lines] == ["spam", "ham"]
    assert main(["text-compare", "--cv", "3"]) == 0


def test_api_endpoints(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import api.main as api_main
    monkeypatch.setattr(api_main, "service", ModelService(tmp_path))
    client = TestClient(api_main.app)
    assert client.get("/health").json() == {"status": "ok"}
    ok = client.post("/predict", json={"dataset": "iris", "rows": [[5.1, 3.5, 1.4, 0.2]]})
    assert ok.status_code == 200 and ok.json()["predictions"][0]["class_name"] == "setosa"
    assert client.post("/predict", json={"dataset": "iris", "rows": [[1, 2]]}).status_code == 422
    assert client.post("/predict-text", json={"texts": ["win a free prize now"]}).status_code == 200
    assert client.post("/predict-text", json={"texts": []}).status_code == 422


REMOVED_APIS = [r"get_feature_names\(\)", r"sklearn\.cross_validation", r"sklearn\.grid_search", r"sklearn\.externals",
                r"metrics\.scorer", r"neighbors\.kde", r"cm\.spectral", r"affinity=", r"\.threshold_\b",
                r"sklearn\.tree\.export\b", r"\bImputer\(", r"load_boston", r"fetch_mldata"]


def test_notebooks_and_helpers_avoid_removed_apis():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    files = list(root.glob("*.ipynb")) + list(root.glob("solutions/*.py")) + list(root.glob("figures/*.py"))
    offenders = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".ipynb":
            text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"] if c["cell_type"] == "code")
        offenders += [f"{f.name}: {p}" for p in REMOVED_APIS if re.search(p, text)]
    assert not offenders, offenders
