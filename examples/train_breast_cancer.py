"""A second practical classification example using scikit-learn's bundled dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Dataset: {data.DESCR.splitlines()[0]}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=data.target_names, zero_division=0))

if __name__ == "__main__":
    main()
