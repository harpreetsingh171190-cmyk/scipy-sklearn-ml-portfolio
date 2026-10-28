"""Train a reproducible Iris classifier and print key evaluation metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml_portfolio import save_model, train_iris_model

if __name__ == "__main__":
    result = train_iris_model()
    print(f"Accuracy: {result['accuracy']:.3f}")
    print("Confusion matrix:")
    print(result["confusion_matrix"])
    path = save_model(result["model"], "artifacts/iris_pipeline.joblib")
    print(f"Saved model to: {path}")
