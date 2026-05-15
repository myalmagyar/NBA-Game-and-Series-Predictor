# src/predict.py

from pathlib import Path

import joblib
import pandas as pd


DATA_DIR = Path("data")
MODELS_DIR = Path("models")

FEATURES_PATH = DATA_DIR / "model_features.csv"
MODEL_PATH = MODELS_DIR / "game_winner_model.joblib"


def load_model_bundle() -> dict:
    """Load trained model and feature columns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing models/game_winner_model.joblib. Run: python src/train_model.py"
        )

    return joblib.load(MODEL_PATH)


def load_features() -> pd.DataFrame:
    """Load matchup feature data."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/model_features.csv. Run: python src/features.py"
        )

    features = pd.read_csv(FEATURES_PATH)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])
    return features


def get_latest_team_row(team_name: str, features: pd.DataFrame) -> pd.Series:
    """Find the latest row where the team appears."""
    team_name_lower = team_name.lower()

    rows = features[
        (features["HOME_TEAM"].str.lower() == team_name_lower)
        | (features["AWAY_TEAM"].str.lower() == team_name_lower)
    ].copy()

    if rows.empty:
        available_teams = sorted(
            set(features["HOME_TEAM"].unique()) | set(features["AWAY_TEAM"].unique())
        )
        raise ValueError(
            f"Team not found: {team_name}\n\nAvailable examples:\n"
            f"{available_teams[:15]}"
        )

    return rows.sort_values("GAME_DATE").iloc[-1]


def team_feature_direction(team_name: str, row: pd.Series, feature_column: str) -> float:
    """Convert a saved home-away diff feature into this team's perspective."""
    if row["HOME_TEAM"].lower() == team_name.lower():
        return float(row[feature_column])

    return -float(row[feature_column])


def build_prediction_row(
    home_team: str,
    away_team: str,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build one model input row for a future matchup."""
    home_latest = get_latest_team_row(home_team, features)
    away_latest = get_latest_team_row(away_team, features)

    prediction_values = {}

    for column in feature_columns:
        home_value = team_feature_direction(home_team, home_latest, column)
        away_value = team_feature_direction(away_team, away_latest, column)
        prediction_values[column] = (home_value + away_value) / 2

    return pd.DataFrame([prediction_values], columns=feature_columns)


def predict_game(home_team: str, away_team: str) -> None:
    """Predict a game winner."""
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    features = load_features()

    prediction_row = build_prediction_row(
        home_team=home_team,
        away_team=away_team,
        features=features,
        feature_columns=feature_columns,
    )

    home_win_probability = model.predict_proba(prediction_row)[0][1]
    away_win_probability = 1 - home_win_probability

    predicted_winner = home_team if home_win_probability >= 0.5 else away_team

    print()
    print("NBA Game Prediction")
    print("-------------------")
    print(f"Matchup: {home_team} vs {away_team}")
    print(f"Predicted winner: {predicted_winner}")
    print(f"{home_team} win probability: {home_win_probability:.1%}")
    print(f"{away_team} win probability: {away_win_probability:.1%}")
    print()


def main() -> None:
    """Run one sample prediction."""
    predict_game(
        home_team="Boston Celtics",
        away_team="Denver Nuggets",
    )


if __name__ == "__main__":
    main()