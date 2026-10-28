import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)
import os

DATA_PATH = "data/processed/test_split.csv"
MODEL_DIR = "models"
TARGET = "Churn"
RESULTS_DIR = "results"


def load_test_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate_model(name: str, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"Model: {name.upper()}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc, "y_prob": y_prob}


def plot_roc_curves(results: list, y_test, output_dir: str):
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Churn Prediction Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()
    print(f"[PLOT] ROC curve saved → {output_dir}/roc_curves.png")


def plot_confusion_matrix(name: str, model, X_test, y_test, output_dir: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{name}.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix saved → {output_dir}/cm_{name}.png")


if __name__ == "__main__":
    X_test, y_test = load_test_data(DATA_PATH)
    results = []

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    for mf in model_files:
        name = mf.replace(".pkl", "")
        model = joblib.load(os.path.join(MODEL_DIR, mf))
        r = evaluate_model(name, model, X_test, y_test)
        results.append(r)
        plot_confusion_matrix(name, model, X_test, y_test, RESULTS_DIR)

    plot_roc_curves(results, y_test, RESULTS_DIR)

    # Summary table
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "y_prob"} for r in results])
    summary = summary.sort_values("auc", ascending=False)
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
