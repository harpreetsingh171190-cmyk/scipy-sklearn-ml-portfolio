"""Text classification module (TF-IDF + linear models) using the bundled SMS spam dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SMS_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "datasets" / "smsspam" / "SMSSpamCollection"

TEXT_MODELS = {
    "logistic_regression": lambda rs: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
    "naive_bayes": lambda rs: MultinomialNB(alpha=0.1),
    "linear_svm": lambda rs: LinearSVC(class_weight="balanced", random_state=rs),
}


def load_sms(path: str | Path = SMS_PATH) -> pd.DataFrame:
    """Load the tab-separated SMS Spam Collection as a DataFrame(label, text)."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"SMS dataset not found at {source}")
    df = pd.read_csv(source, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    return df.dropna().reset_index(drop=True)


def build_text_model(name: str = "logistic_regression", random_state: int = 42,
                     ngram_range: tuple[int, int] = (1, 2)) -> Pipeline:
    if name not in TEXT_MODELS:
        raise ValueError(f"Unknown text model {name!r}. Choose from {sorted(TEXT_MODELS)}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=2,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("clf", TEXT_MODELS[name](random_state)),
    ])


def train_text_model(model_name: str = "logistic_regression", test_size: float = 0.25,
                     random_state: int = 42, df: pd.DataFrame | None = None) -> dict[str, Any]:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    df = load_sms() if df is None else df
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"])
    model = build_text_model(model_name, random_state).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"model": model, "predictions": pred, "y_test": y_test.to_numpy(), "X_test": X_test,
            "accuracy": float(accuracy_score(y_test, pred)),
            "spam_precision": float(precision_score(y_test, pred, pos_label="spam", zero_division=0)),
            "spam_recall": float(recall_score(y_test, pred, pos_label="spam", zero_division=0)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0))}


def compare_text_models(cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    if cv < 2:
        raise ValueError("cv must be at least 2")
    df = load_sms()
    splitter = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    rows = []
    for name in TEXT_MODELS:
        s = cross_validate(build_text_model(name, random_state), df["text"], df["label"], cv=splitter,
                           scoring={"acc": "accuracy", "f1": "f1_macro"})
        rows.append({"model": name, "accuracy_mean": s["test_acc"].mean(), "macro_f1_mean": s["test_f1"].mean(),
                     "macro_f1_std": s["test_f1"].std()})
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def predict_texts(model: Pipeline, texts: list[str]) -> list[dict[str, Any]]:
    """Predict labels (with spam probability/score when available) for a list of messages."""
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("texts must be a non-empty list of non-empty strings")
    labels = model.predict(texts)
    clf = model.named_steps["clf"]
    scores = None
    if hasattr(model, "predict_proba"):
        spam_idx = list(clf.classes_).index("spam")
        scores = model.predict_proba(texts)[:, spam_idx]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
    return [{"label": str(lab), **({"spam_score": round(float(scores[i]), 4)} if scores is not None else {})}
            for i, lab in enumerate(labels)]


def top_terms(model: Pipeline, n: int = 10) -> dict[str, list[str]]:
    """Most spam-indicative and ham-indicative terms for linear models."""
    clf = model.named_steps["clf"]
    names = model.named_steps["tfidf"].get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        raise ValueError("Model does not expose term weights")
    spam_class_positive = list(clf.classes_)[1] == "spam"
    if not spam_class_positive:
        weights = -weights
    order = np.argsort(weights)
    return {"spam": [str(names[i]) for i in order[::-1][:n]], "ham": [str(names[i]) for i in order[:n]]}
