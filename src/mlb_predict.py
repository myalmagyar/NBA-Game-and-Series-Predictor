# src/mlb_predict.py

from pathlib import Path

import joblib
import pandas as pd

try:
    from src.mlb_features import DEFAULT_RUN_ENVIRONMENT, HOME_ELO_ADVANTAGE, expected_score, get_park_run_factor
except ModuleNotFoundError:
    from mlb_features import DEFAULT_RUN_ENVIRONMENT, HOME_ELO_ADVANTAGE, expected_score, get_park_run_factor


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TEAM_STRENGTH_PATH = DATA_DIR / "mlb_current_team_strength.csv"
MODEL_PATH = MODELS_DIR / "mlb_game_winner_model.joblib"

FEATURE_COLUMN_ALIASES = {
    "SEASON_RUN_PER_GAME": "SEASON_AVG_RUNS_FOR",
    "SEASON_OPP_RUN_PER_GAME": "SEASON_AVG_RUNS_AGAINST",
    "ROLLING_RUN_PER_GAME_10": "ROLLING_RUNS_FOR_10",
    "ROLLING_OPP_RUN_PER_GAME_10": "ROLLING_RUNS_AGAINST_10",
    "ROLLING_RUN_PER_GAME_20": "ROLLING_RUNS_FOR_20",
    "ROLLING_OPP_RUN_PER_GAME_20": "ROLLING_RUNS_AGAINST_20",
}


def load_model_bundle() -> dict:
    """Load trained MLB model bundle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing models/mlb_game_winner_model.joblib. Run: python src/mlb_train_model.py"
        )

    return joblib.load(MODEL_PATH)


def load_team_strength() -> pd.DataFrame:
    """Load current MLB team strength."""
    if not TEAM_STRENGTH_PATH.exists():
        raise FileNotFoundError(
            "Missing data/mlb_current_team_strength.csv. Run: python src/mlb_team_strength.py"
        )

    return pd.read_csv(TEAM_STRENGTH_PATH)


def get_team_strength_row(team_name: str, strength: pd.DataFrame) -> pd.Series:
    """Return one MLB team strength row."""
    rows = strength[strength["TEAM_NAME"].str.lower().eq(team_name.lower())]

    if rows.empty:
        raise ValueError(f"MLB team not found: {team_name}")

    return rows.iloc[0]


def neutral_feature_value(column: str) -> float:
    """Return a neutral value for features unavailable in CLI predictions."""
    if column == "ELO":
        return 1500.0
    if "WIN_PCT" in column or column == "PITCHER_TEAM_WIN_PCT":
        return 0.5
    if "RUNS_FOR" in column or "RUNS_AGAINST" in column:
        return float(DEFAULT_RUN_ENVIRONMENT)
    if column in {"PROJECTED_LINEUP_STRENGTH", "PITCHER_RUN_SUPPORT_PER_START", "PITCHER_RUNS_ALLOWED_PER_START"}:
        return float(DEFAULT_RUN_ENVIRONMENT)
    if column in {"DAYS_REST", "PITCHER_DAYS_REST"}:
        return 3.0
    return 0.0


def numeric_row_value(row: pd.Series, column: str) -> float:
    """Read a numeric row value with neutral fallback."""
    if column not in row.index:
        return neutral_feature_value(column)

    try:
        return float(row[column])
    except (TypeError, ValueError):
        return neutral_feature_value(column)


def build_prediction_row(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build one model input row from current strength."""
    home = get_team_strength_row(home_team, strength)
    away = get_team_strength_row(away_team, strength)
    values = {}

    for column in feature_columns:
        if column == "HOME_ELO_WIN_PROB":
            values[column] = expected_score(
                float(home["ELO"]) + HOME_ELO_ADVANTAGE,
                float(away["ELO"]),
            )
            continue

        if column == "HOME_PARK_RUN_FACTOR":
            values[column] = get_park_run_factor(home_team)
            continue

        strength_column = column.removeprefix("DIFF_")
        strength_column = FEATURE_COLUMN_ALIASES.get(strength_column, strength_column)

        values[column] = numeric_row_value(home, strength_column) - numeric_row_value(away, strength_column)

    return pd.DataFrame([values], columns=feature_columns)


def predict_game(home_team: str, away_team: str) -> tuple[str, float, float]:
    """Predict one MLB game."""
    bundle = load_model_bundle()
    strength = load_team_strength()
    row = build_prediction_row(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        feature_columns=bundle["feature_columns"],
    )
    probability = float(bundle["model"].predict_proba(row)[0][1])
    winner = home_team if probability >= 0.5 else away_team
    return winner, probability, 1 - probability


def main() -> None:
    """Run one sample MLB prediction."""
    winner, home_probability, away_probability = predict_game(
        home_team="Los Angeles Dodgers",
        away_team="New York Yankees",
    )
    print("MLB Game Prediction")
    print("-------------------")
    print(f"Predicted winner: {winner}")
    print(f"Home win probability: {home_probability:.1%}")
    print(f"Away win probability: {away_probability:.1%}")


if __name__ == "__main__":
    main()
