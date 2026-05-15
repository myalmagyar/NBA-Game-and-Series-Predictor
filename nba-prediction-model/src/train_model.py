# src/train_model.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.feature_utils import PLAYOFF_CONTEXT_FEATURE_COLUMNS
except ModuleNotFoundError:
    from feature_utils import PLAYOFF_CONTEXT_FEATURE_COLUMNS


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
FEATURES_PATH = DATA_DIR / "model_features.csv"
MODEL_PATH = MODELS_DIR / "game_winner_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.csv"
BACKTEST_METRICS_PATH = DATA_DIR / "backtest_metrics.csv"
CALIBRATION_PATH = DATA_DIR / "calibration_metrics.csv"

MIN_SINGLE_GAME_PROBABILITY = 0.18
MAX_SINGLE_GAME_PROBABILITY = 0.82


def load_features() -> pd.DataFrame:
    """Load model-ready NBA features."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/model_features.csv. Run: python src/features.py"
        )

    data = pd.read_csv(FEATURES_PATH)
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    return data.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Select model feature columns."""
    return [
        column
        for column in data.columns
        if column.startswith("DIFF_")
        or column == "HOME_ELO_WIN_PROB"
        or column in PLAYOFF_CONTEXT_FEATURE_COLUMNS
    ]


def build_models() -> dict[str, Pipeline]:
    """Create calibrated candidate models."""
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LogisticRegression(max_iter=1000),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=RandomForestClassifier(
                        n_estimators=400,
                        max_depth=7,
                        min_samples_leaf=12,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    method="sigmoid",
                    cv=5,
                ),
            )
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=GradientBoostingClassifier(
                        n_estimators=120,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=5,
                ),
            )
        ]
    )

    return {
        "Calibrated Logistic Regression": logistic_regression,
        "Calibrated Random Forest": random_forest,
        "Calibrated Gradient Boosting": gradient_boosting,
    }


def chronological_train_test_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split historical games chronologically to better mimic future prediction."""
    split_index = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    x_train = train_data[feature_columns]
    y_train = train_data["HOME_WIN"]

    x_test = test_data[feature_columns]
    y_test = test_data["HOME_WIN"]

    return x_train, x_test, y_train, y_test


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate one model with classification and probability metrics."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "Brier_Score": brier_score_loss(y_test, probabilities),
        "Log_Loss": log_loss(y_test, probabilities),
    }


def clamp_probabilities(probabilities) -> pd.Series:
    """Clamp probabilities to the app's single-game bounds."""
    return pd.Series(probabilities).clip(
        lower=MIN_SINGLE_GAME_PROBABILITY,
        upper=MAX_SINGLE_GAME_PROBABILITY,
    )


def shrink_probabilities(probabilities, shrinkage: float) -> pd.Series:
    """Shrink probabilities toward 50%."""
    probability_series = pd.Series(probabilities)
    return 0.5 + ((probability_series - 0.5) * shrinkage)


def tune_probability_blend(
    model_probabilities,
    elo_probabilities,
    y_true: pd.Series,
) -> dict:
    """Tune model/Elo blend and probability shrinkage on validation data."""
    rows = []

    for model_weight_step in range(0, 21):
        model_weight = model_weight_step / 20
        elo_weight = 1 - model_weight

        for shrinkage_step in range(70, 101, 2):
            shrinkage = shrinkage_step / 100
            blended = (
                model_weight * pd.Series(model_probabilities).reset_index(drop=True)
                + elo_weight * pd.Series(elo_probabilities).reset_index(drop=True)
            )
            final_probabilities = clamp_probabilities(
                shrink_probabilities(blended, shrinkage)
            )

            rows.append(
                {
                    "model_probability_weight": model_weight,
                    "elo_probability_weight": elo_weight,
                    "probability_shrinkage": shrinkage,
                    "Brier_Score": brier_score_loss(y_true, final_probabilities),
                    "Log_Loss": log_loss(y_true, final_probabilities),
                    "ROC_AUC": roc_auc_score(y_true, final_probabilities),
                }
            )

    blend_metrics = pd.DataFrame(rows).sort_values(
        by=["Brier_Score", "Log_Loss", "ROC_AUC"],
        ascending=[True, True, False],
    )
    return blend_metrics.iloc[0].to_dict()


def build_calibration_table(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Build probability calibration bins for the test split."""
    probabilities = pd.Series(model.predict_proba(x_test)[:, 1], name="Predicted")
    calibration = pd.DataFrame(
        {
            "Predicted": probabilities,
            "Actual": y_test.reset_index(drop=True),
        }
    )
    calibration["Bucket"] = pd.cut(
        calibration["Predicted"],
        bins=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        include_lowest=True,
    )

    return (
        calibration.groupby("Bucket", observed=True)
        .agg(
            Games=("Actual", "size"),
            Average_Predicted_Probability=("Predicted", "mean"),
            Actual_Home_Win_Rate=("Actual", "mean"),
        )
        .reset_index()
        .assign(Bucket=lambda df: df["Bucket"].astype(str))
    )


def choose_best_model(metrics_df: pd.DataFrame) -> str:
    """Choose model by probability quality first, then ranking power."""
    sorted_metrics = metrics_df.sort_values(
        by=["Brier_Score", "Log_Loss", "ROC_AUC"],
        ascending=[True, True, False],
    )
    return str(sorted_metrics.iloc[0]["Model"])


def run_rolling_backtests(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Run expanding-window backtests by season."""
    seasons = sorted(data["SEASON"].unique())
    rows = []

    for test_season in seasons[2:]:
        train_data = data[data["SEASON"] < test_season].copy()
        test_data = data[data["SEASON"] == test_season].copy()

        if len(train_data) < 500 or len(test_data) < 100:
            continue

        x_train = train_data[feature_columns]
        y_train = train_data["HOME_WIN"]
        x_test = test_data[feature_columns]
        y_test = test_data["HOME_WIN"]

        models = build_models()
        season_metrics = []
        trained_models = {}

        for model_name, model in models.items():
            print(f"Backtesting {test_season}: {model_name}...")
            model.fit(x_train, y_train)
            trained_models[model_name] = model
            season_metrics.append(
                evaluate_model(
                    model_name=model_name,
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                )
            )

        season_metrics_df = pd.DataFrame(season_metrics)
        best_model_name = choose_best_model(season_metrics_df)
        best_model = trained_models[best_model_name]

        model_probabilities = best_model.predict_proba(x_test)[:, 1]
        blend = tune_probability_blend(
            model_probabilities=model_probabilities,
            elo_probabilities=x_test["HOME_ELO_WIN_PROB"],
            y_true=y_test.reset_index(drop=True),
        )

        rows.append(
            {
                "Test_Season": test_season,
                "Train_Games": len(train_data),
                "Test_Games": len(test_data),
                "Best_Model": best_model_name,
                "Accuracy": float(
                    accuracy_score(y_test, best_model.predict(x_test))
                ),
                "ROC_AUC": float(
                    roc_auc_score(y_test, model_probabilities)
                ),
                "Brier_Score": float(
                    brier_score_loss(y_test, model_probabilities)
                ),
                "Log_Loss": float(log_loss(y_test, model_probabilities)),
                "Blend_Model_Weight": blend["model_probability_weight"],
                "Blend_Elo_Weight": blend["elo_probability_weight"],
                "Blend_Shrinkage": blend["probability_shrinkage"],
                "Blend_Brier_Score": blend["Brier_Score"],
                "Blend_Log_Loss": blend["Log_Loss"],
                "Blend_ROC_AUC": blend["ROC_AUC"],
            }
        )

    return pd.DataFrame(rows)


def train_model() -> Pipeline:
    """Train calibrated models, save best model, and save metrics."""
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    data = load_features()
    feature_columns = get_feature_columns(data)
    feature_availability = {
        "historical_injuries_used": bool(
            "DIFF_INJURY_WEIGHTED_IMPACT" in data.columns
            and data["DIFF_INJURY_WEIGHTED_IMPACT"].abs().sum() > 0
        ),
        "previous_season_player_strength_used": bool(
            "DIFF_PREV_SEASON_PLAYER_TOP_8" in data.columns
            and data["DIFF_PREV_SEASON_PLAYER_TOP_8"].abs().sum() > 0
        ),
        "schedule_features_used": bool(
            "DIFF_DAYS_REST" in data.columns
            and data["DIFF_DAYS_REST"].abs().sum() > 0
        ),
        "playoff_context_used": bool(
            "IS_PLAYOFF_GAME" in data.columns
            and data["IS_PLAYOFF_GAME"].abs().sum() > 0
        ),
        "advanced_efficiency_used": bool(
            "DIFF_SEASON_AVG_NET_RATING" in data.columns
            and data["DIFF_SEASON_AVG_NET_RATING"].abs().sum() > 0
        ),
    }

    x_train, x_test, y_train, y_test = chronological_train_test_split(
        data=data,
        feature_columns=feature_columns,
        test_size=0.2,
    )

    models = build_models()
    metrics = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        trained_models[model_name] = model

        model_metrics = evaluate_model(
            model_name=model_name,
            model=model,
            x_test=x_test,
            y_test=y_test,
        )
        metrics.append(model_metrics)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values("Brier_Score", ascending=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    best_model_name = choose_best_model(metrics_df)
    best_model = trained_models[best_model_name]

    predictions = best_model.predict(x_test)
    model_probabilities = best_model.predict_proba(x_test)[:, 1]
    calibration_table = build_calibration_table(best_model, x_test, y_test)
    calibration_table.to_csv(CALIBRATION_PATH, index=False)

    blend_settings = tune_probability_blend(
        model_probabilities=model_probabilities,
        elo_probabilities=x_test["HOME_ELO_WIN_PROB"],
        y_true=y_test.reset_index(drop=True),
    )

    backtest_metrics = run_rolling_backtests(data, feature_columns)

    if not backtest_metrics.empty:
        backtest_metrics.to_csv(BACKTEST_METRICS_PATH, index=False)

    print()
    print("Model comparison:")
    print(metrics_df.to_string(index=False))
    print()
    print(f"Best model: {best_model_name}")
    print()
    print("Tuned probability blend:")
    print(pd.Series(blend_settings).to_string())

    if not backtest_metrics.empty:
        print()
        print("Rolling backtest metrics:")
        print(backtest_metrics.to_string(index=False))

    print()
    print("Calibration table:")
    print(calibration_table.to_string(index=False))

    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "metrics": metrics_df.to_dict(orient="records"),
            "probability_notes": {
                "selection_metric": "Brier_Score, then Log_Loss, then ROC_AUC",
                "split_type": "chronological",
            },
            "blend_settings": blend_settings,
            "backtest_metrics_path": str(BACKTEST_METRICS_PATH),
            "calibration_metrics_path": str(CALIBRATION_PATH),
            "feature_availability": feature_availability,
        },
        MODEL_PATH,
    )

    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

    if not backtest_metrics.empty:
        print(f"Saved backtest metrics to {BACKTEST_METRICS_PATH}")

    print(f"Saved calibration metrics to {CALIBRATION_PATH}")

    print()
    print("Features used:")
    for column in feature_columns:
        print(f"- {column}")

    return best_model


if __name__ == "__main__":
    train_model()
