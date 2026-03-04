import os
import json
import yaml
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

def plot_and_save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_and_save_roc_curve(model, X_test, y_test, path):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def build_models(train_params):
    # 3 models required by Member 2: Logistic Regression, Random Forest, (XGBoost alternative) GradientBoosting
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=int(train_params.get("max_iter", 2000)),
            class_weight=train_params.get("class_weight", None),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=train_params.get("class_weight", None),
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        )
    }
    return models

def tune_best_model(best_name, base_model, X_train, y_train):
    # Minimal but strong GridSearch for the selected best model
    if best_name == "logistic_regression":
        grid = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [2000, 5000],
        }
    elif best_name == "random_forest":
        grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5],
        }
    else:  # gradient_boosting
        grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        verbose=0
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)

def main():
    params = load_params()

    data_path = params["data"]["cleaned_path"]
    target_col = params["data"]["target_col"]
    test_size = float(params["data"]["test_size"])
    random_state = int(params["data"]["random_state"])

    train_params = params["train"]
    exp_name = params["mlflow"]["experiment_name"]

    ensure_dirs()

    # Load
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)[:10]}...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    mlflow.set_experiment(exp_name)

    summary = {
        "model_comparison": {},
        "best_model": {},
        "tuning": {}
    }

    with mlflow.start_run(run_name="member2_training"):

        # Compare 3 models
        candidates = build_models(train_params)
        best_name, best_auc = None, -1.0
        best_model = None

        for name, model in candidates.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            if y_prob is None:
                # fallback, should not happen for these models
                y_prob = y_pred

            metrics = compute_metrics(y_test, y_pred, y_prob)
            summary["model_comparison"][name] = metrics

            # log each model’s metrics under different prefixes
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = name
                best_model = model

        # Tune best model using GridSearchCV
        tuned_model, best_params, cv_best_auc = tune_best_model(best_name, best_model, X_train, y_train)

        # Evaluate tuned best model
        tuned_model.fit(X_train, y_train)
        y_pred = tuned_model.predict(X_test)
        y_prob = tuned_model.predict_proba(X_test)[:, 1]

        tuned_metrics = compute_metrics(y_test, y_pred, y_prob)

        summary["best_model"] = {
            "name": best_name,
            "test_metrics": tuned_metrics
        }
        summary["tuning"] = {
            "best_params": best_params,
            "cv_best_roc_auc": cv_best_auc
        }

        # Save artifacts
        best_model_path = "models/best_model.pkl"
        joblib.dump(tuned_model, best_model_path)

        metrics_path = "reports/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=4)

        cm_path = "reports/confusion_matrix.png"
        roc_path = "reports/roc_curve.png"
        plot_and_save_confusion_matrix(y_test, y_pred, cm_path)
        plot_and_save_roc_curve(tuned_model, X_test, y_test, roc_path)

        # MLflow logging (best model + artifacts)
        mlflow.log_param("best_model_name", best_name)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", cv_best_auc)
        mlflow.log_metrics({f"best_test_{k}": v for k, v in tuned_metrics.items()})

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        mlflow.sklearn.log_model(tuned_model, "best_model")

        print("✅ Member 2 training complete")
        print(json.dumps(summary["best_model"], indent=2))

if __name__ == "__main__":
    main()
