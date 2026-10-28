import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

DATA_PATH = "data/processed/churn_processed.csv"
MODEL_DIR = "models"
TARGET = "Churn"
RANDOM_STATE = 42


def load_processed(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"[INFO] Features: {X.shape[1]}  |  Positive class rate: {y.mean():.2%}")
    return X, y


def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
    }

    trained = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  [{name}] CV AUC-ROC: {scores.mean():.4f} ± {scores.std():.4f}")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def save_models(models: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  [SAVED] {path}")


if __name__ == "__main__":
    X, y = load_processed(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\n[TRAIN] Training models...")
    trained_models = train_models(X_train, y_train)

    print("\n[SAVE] Saving models...")
    save_models(trained_models, MODEL_DIR)

    # Save test split for evaluation
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv("data/processed/test_split.csv", index=False)
    print("\n[INFO] Test split saved → data/processed/test_split.csv")
