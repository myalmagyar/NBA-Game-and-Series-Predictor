# src/nhl_train_model.py

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.nhl_features import FEATURE_COLUMNS


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
FEATURES_PATH = DATA_DIR / "nhl_model_features.csv"
MODEL_PATH = MODELS_DIR / "nhl_game_winner_model.joblib"
METRICS_PATH = DATA_DIR / "nhl_model_metrics.csv"


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities) if len(set(y_test)) > 1 else 0.5,
        "Brier_Score": brier_score_loss(y_test, probabilities),
        "Log_Loss": log_loss(y_test, probabilities),
        "Estimator": model,
    }


def train_model(features: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    rows = features.dropna(subset=["HOME_WIN"]).copy()
    rows = rows.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    for column in FEATURE_COLUMNS:
        if column not in rows.columns:
            rows[column] = 0.0
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)

    split_index = max(int(len(rows) * 0.8), 1)
    train = rows.iloc[:split_index]
    test = rows.iloc[split_index:]
    if test.empty:
        test = train

    x_train = train[FEATURE_COLUMNS]
    y_train = train["HOME_WIN"].astype(int)
    x_test = test[FEATURE_COLUMNS]
    y_test = test["HOME_WIN"].astype(int)
    candidates = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=12,
                random_state=42,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=180,
                max_depth=2,
                learning_rate=0.04,
                random_state=42,
            ),
        ),
    ]
    metrics = []

    for name, model in candidates:
        print(f"Training {name}...")
        metrics.append(evaluate_model(name, model, x_train, y_train, x_test, y_test))

    metrics_df = pd.DataFrame([{k: v for k, v in row.items() if k != "Estimator"} for row in metrics])
    best = max(metrics, key=lambda row: (row["ROC_AUC"], row["Accuracy"]))
    bundle = {
        "model": best["Estimator"],
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best["Model"],
        "metrics": {k: v for k, v in best.items() if k != "Estimator"},
    }
    return bundle, metrics_df.sort_values(["ROC_AUC", "Accuracy"], ascending=False)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Missing data/nhl_model_features.csv. Run: python src/nhl_features.py")

    features = pd.read_csv(FEATURES_PATH)
    bundle, metrics = train_model(features)
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    metrics.to_csv(METRICS_PATH, index=False)
    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nBest model: {bundle['model_name']}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
