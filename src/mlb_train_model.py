# src/mlb_train_model.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.mlb_features import FEATURES_PATH, get_feature_columns
except ModuleNotFoundError:
    from mlb_features import FEATURES_PATH, get_feature_columns


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "mlb_game_winner_model.joblib"
METRICS_PATH = DATA_DIR / "mlb_model_metrics.csv"


def load_features() -> pd.DataFrame:
    """Load MLB model-ready feature rows."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/mlb_model_features.csv. Run: python src/mlb_features.py"
        )

    data = pd.read_csv(FEATURES_PATH)
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    return data.sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)


def build_models() -> dict[str, Pipeline]:
    """Create candidate MLB classifiers."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=7,
                        min_samples_leaf=12,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=42,
                    ),
                )
            ]
        ),
    }


def chronological_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically."""
    split_index = int(len(data) * (1 - test_size))
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    return (
        train[feature_columns],
        test[feature_columns],
        train["HOME_WIN"],
        test["HOME_WIN"],
    )


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    elo_probabilities: pd.Series | None = None,
) -> dict:
    """Evaluate a model with probability metrics."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "Brier_Score": brier_score_loss(y_test, probabilities),
        "Log_Loss": log_loss(y_test, probabilities),
    }

    if elo_probabilities is not None:
        best_weight, best_brier = optimize_model_elo_blend(
            model_probabilities=probabilities,
            elo_probabilities=elo_probabilities,
            y_true=y_test,
        )
        blended = (best_weight * probabilities) + ((1 - best_weight) * elo_probabilities)
        result["Optimized_Model_Weight"] = best_weight
        result["Blend_Accuracy"] = accuracy_score(y_test, blended >= 0.5)
        result["Blend_ROC_AUC"] = roc_auc_score(y_test, blended)
        result["Blend_Brier_Score"] = best_brier
        result["Blend_Log_Loss"] = log_loss(y_test, blended)

    return result


def optimize_model_elo_blend(
    model_probabilities,
    elo_probabilities,
    y_true: pd.Series,
) -> tuple[float, float]:
    """Find the model/Elo blend weight with the best Brier score."""
    best_weight = 1.0
    best_brier = float("inf")

    for step in range(0, 21):
        weight = step / 20
        blended = (weight * model_probabilities) + ((1 - weight) * elo_probabilities)
        brier = brier_score_loss(y_true, blended)

        if brier < best_brier:
            best_weight = weight
            best_brier = brier

    return best_weight, best_brier


def train_model() -> Pipeline:
    """Train MLB game winner model and save bundle."""
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    data = load_features()

    if len(data) < 200:
        raise RuntimeError("Need at least 200 MLB games before training.")

    feature_columns = get_feature_columns(data)
    x_train, x_test, y_train, y_test = chronological_split(data, feature_columns)
    split_index = int(len(data) * 0.8)
    test_rows = data.iloc[split_index:].copy()
    elo_test_probabilities = test_rows["HOME_ELO_WIN_PROB"].astype(float)
    models = build_models()
    trained_models = {}
    metrics = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        trained_models[model_name] = model
        metrics.append(evaluate_model(model_name, model, x_test, y_test, elo_test_probabilities))

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["Brier_Score", "Log_Loss", "ROC_AUC"],
        ascending=[True, True, False],
    )
    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]
    best_metric = metrics_df.iloc[0]
    model_weight = float(best_metric.get("Optimized_Model_Weight", 0.8))
    elo_weight = 1 - model_weight

    metrics_df.to_csv(METRICS_PATH, index=False)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "metrics": metrics_df.to_dict(orient="records"),
            "blend_settings": {
                "model_probability_weight": model_weight,
                "elo_probability_weight": elo_weight,
                "optimization_metric": "Brier_Score",
            },
            "notes": {
                "sport": "MLB",
                "split_type": "chronological",
                "feature_version": (
                    "Team form, Elo, starter proxy, bullpen fatigue proxy, "
                    "home/away splits, recent form, lineup-strength proxy, and park factors."
                ),
            },
        },
        MODEL_PATH,
    )

    print()
    print("Model comparison:")
    print(metrics_df.to_string(index=False))
    print()
    print(f"Best model: {best_model_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

    return best_model


if __name__ == "__main__":
    train_model()
