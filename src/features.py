# src/features.py

from pathlib import Path

import pandas as pd

try:
    from src.feature_utils import (
        ADVANCED_STAT_COLUMNS,
        INJURY_FEATURE_COLUMNS,
        PLAYER_STRENGTH_FEATURE_COLUMNS,
        PLAYOFF_CONTEXT_FEATURE_COLUMNS,
        SCHEDULE_FEATURE_COLUMNS,
        add_advanced_game_stats,
        add_historical_injury_features,
        add_playoff_context_features,
        add_pregame_schedule_features,
        add_previous_season_player_strength,
    )
except ModuleNotFoundError:
    from feature_utils import (
        ADVANCED_STAT_COLUMNS,
        INJURY_FEATURE_COLUMNS,
        PLAYER_STRENGTH_FEATURE_COLUMNS,
        PLAYOFF_CONTEXT_FEATURE_COLUMNS,
        SCHEDULE_FEATURE_COLUMNS,
        add_advanced_game_stats,
        add_historical_injury_features,
        add_playoff_context_features,
        add_pregame_schedule_features,
        add_previous_season_player_strength,
    )


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "raw_games.csv"
FEATURES_PATH = DATA_DIR / "model_features.csv"

BASE_ELO = 1500.0
HOME_ELO_ADVANTAGE = 65.0
K_FACTOR = 20.0
ROLLING_WINDOWS = [5, 10]

BASE_STAT_COLUMNS = [
    "PTS",
    "PLUS_MINUS",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "REB",
    "AST",
    "TOV",
    *ADVANCED_STAT_COLUMNS,
]


def load_raw_games() -> pd.DataFrame:
    """Load raw NBA game logs."""
    if not RAW_GAMES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/raw_games.csv. Run: python src/collect_data.py"
        )

    games = pd.read_csv(RAW_GAMES_PATH)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    return games


def expected_score(team_elo: float, opponent_elo: float) -> float:
    """Calculate Elo expected win probability."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def add_team_form_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and season-to-date features using only previous games."""
    games = add_advanced_game_stats(games)
    games = games.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"]).copy()
    games["WIN"] = games["WL"].eq("W").astype(int)

    for window in ROLLING_WINDOWS:
        for column in BASE_STAT_COLUMNS:
            games[f"ROLLING_{column}_{window}"] = (
                games.groupby("TEAM_ID")[column]
                .transform(lambda series: series.shift(1).rolling(window).mean())
            )

        games[f"ROLLING_WIN_PCT_{window}"] = (
            games.groupby("TEAM_ID")["WIN"]
            .transform(lambda series: series.shift(1).rolling(window).mean())
        )

    for column in BASE_STAT_COLUMNS:
        games[f"SEASON_AVG_{column}"] = (
            games.groupby(["TEAM_ID", "SEASON"])[column]
            .transform(lambda series: series.shift(1).expanding().mean())
        )

    games["SEASON_WIN_PCT"] = (
        games.groupby(["TEAM_ID", "SEASON"])["WIN"]
        .transform(lambda series: series.shift(1).expanding().mean())
    )

    return games


def get_feature_source_columns() -> list[str]:
    """Return team-level feature columns used to create matchup differences."""
    columns = []

    for window in ROLLING_WINDOWS:
        for stat_column in BASE_STAT_COLUMNS:
            columns.append(f"ROLLING_{stat_column}_{window}")

        columns.append(f"ROLLING_WIN_PCT_{window}")

    for stat_column in BASE_STAT_COLUMNS:
        columns.append(f"SEASON_AVG_{stat_column}")

    columns.append("SEASON_WIN_PCT")
    columns.extend(SCHEDULE_FEATURE_COLUMNS)

    return columns


def create_base_matchup_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Convert team-game rows into one matchup row per game."""
    games = add_team_form_features(games)
    games = add_pregame_schedule_features(games)

    team_feature_columns = get_feature_source_columns()

    home_games = games[games["IS_HOME"] == 1].copy()
    away_games = games[games["IS_HOME"] == 0].copy()

    home_columns = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ID",
        "TEAM_NAME",
        "WL",
        *team_feature_columns,
    ]

    away_columns = [
        "GAME_ID",
        "TEAM_ID",
        "TEAM_NAME",
        *team_feature_columns,
    ]

    home_games = home_games[home_columns].rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_NAME": "HOME_TEAM",
            "WL": "HOME_WL",
            **{column: f"HOME_{column}" for column in team_feature_columns},
        }
    )

    away_games = away_games[away_columns].rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "TEAM_NAME": "AWAY_TEAM",
            **{column: f"AWAY_{column}" for column in team_feature_columns},
        }
    )

    matchups = home_games.merge(away_games, on="GAME_ID", how="inner")
    matchups["HOME_WIN"] = matchups["HOME_WL"].eq("W").astype(int)

    for column in team_feature_columns:
        matchups[f"DIFF_{column}"] = (
            matchups[f"HOME_{column}"] - matchups[f"AWAY_{column}"]
        )

    return matchups.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def add_elo_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """Add pregame Elo features and update ratings after each game."""
    elo_ratings: dict[int, float] = {}

    home_pre_elo_values = []
    away_pre_elo_values = []
    home_elo_probabilities = []

    for _, row in matchups.iterrows():
        home_team_id = int(row["HOME_TEAM_ID"])
        away_team_id = int(row["AWAY_TEAM_ID"])

        home_elo = elo_ratings.get(home_team_id, BASE_ELO)
        away_elo = elo_ratings.get(away_team_id, BASE_ELO)

        home_expected = expected_score(home_elo + HOME_ELO_ADVANTAGE, away_elo)
        home_actual = float(row["HOME_WIN"])
        elo_change = K_FACTOR * (home_actual - home_expected)

        home_pre_elo_values.append(home_elo)
        away_pre_elo_values.append(away_elo)
        home_elo_probabilities.append(home_expected)

        elo_ratings[home_team_id] = home_elo + elo_change
        elo_ratings[away_team_id] = away_elo - elo_change

    matchups = matchups.copy()
    matchups["HOME_ELO"] = home_pre_elo_values
    matchups["AWAY_ELO"] = away_pre_elo_values
    matchups["DIFF_ELO"] = matchups["HOME_ELO"] - matchups["AWAY_ELO"]
    matchups["HOME_ELO_WIN_PROB"] = home_elo_probabilities

    return matchups


def create_matchup_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Create final model-ready matchup rows."""
    matchups = create_base_matchup_rows(games)
    matchups = add_playoff_context_features(matchups)
    matchups = add_previous_season_player_strength(matchups)
    matchups = add_historical_injury_features(matchups)
    matchups = add_elo_features(matchups)

    feature_columns = [
        column
        for column in matchups.columns
        if column.startswith("DIFF_")
        or column == "HOME_ELO_WIN_PROB"
        or column in PLAYOFF_CONTEXT_FEATURE_COLUMNS
    ]

    output_columns = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_WIN",
        *feature_columns,
    ]

    return matchups[output_columns].dropna().reset_index(drop=True)


def build_features() -> pd.DataFrame:
    """Build model-ready features."""
    DATA_DIR.mkdir(exist_ok=True)

    games = load_raw_games()
    features = create_matchup_rows(games)
    features.to_csv(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} matchup rows to {FEATURES_PATH}")
    print()
    print("Feature columns:")
    for column in features.columns:
        if column.startswith("DIFF_") or column == "HOME_ELO_WIN_PROB":
            print(f"- {column}")

    print()
    print("Added accuracy feature groups:")
    print(f"- Schedule/rest: {', '.join(SCHEDULE_FEATURE_COLUMNS)}")
    print(f"- Advanced efficiency: {', '.join(ADVANCED_STAT_COLUMNS)}")
    print(f"- Playoff context: {', '.join(PLAYOFF_CONTEXT_FEATURE_COLUMNS)}")
    print(f"- Historical injuries: {', '.join(INJURY_FEATURE_COLUMNS)}")
    print(f"- Previous-season player strength: {', '.join(PLAYER_STRENGTH_FEATURE_COLUMNS)}")

    return features


if __name__ == "__main__":
    build_features()
