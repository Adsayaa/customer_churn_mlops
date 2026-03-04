import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    os.makedirs("reports", exist_ok=True)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_roc_curve(model, X_test, y_test, path):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    params = load_params()

    data_path = params["data"]["cleaned_path"]
    target_col = params["data"]["target_col"]
    test_size = float(params["data"]["test_size"])
    random_state = int(params["data"]["random_state"])

    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run training first.")

    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y #
    )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = compute_metrics(y_test, y_pred, y_prob)

    ensure_dirs()
    out = {
        "evaluated_model_path": model_path,
        "metrics": metrics
    }

    with open("reports/eval_metrics.json", "w") as f:
        json.dump(out, f, indent=4)

    save_confusion_matrix(y_test, y_pred, "reports/eval_confusion_matrix.png")
    save_roc_curve(model, X_test, y_test, "reports/eval_roc_curve.png")

    print("Evaluation complete")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
