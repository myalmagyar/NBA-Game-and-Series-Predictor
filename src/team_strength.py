# src/team_strength.py

from pathlib import Path

import pandas as pd

try:
    from src.feature_utils import (
        ADVANCED_STAT_COLUMNS,
        CURRENT_PLAYER_STRENGTH_COLUMNS,
        SCHEDULE_FEATURE_COLUMNS,
        add_advanced_game_stats,
        calculate_current_schedule_state,
        load_current_player_strength,
    )
except ModuleNotFoundError:
    from feature_utils import (
        ADVANCED_STAT_COLUMNS,
        CURRENT_PLAYER_STRENGTH_COLUMNS,
        SCHEDULE_FEATURE_COLUMNS,
        add_advanced_game_stats,
        calculate_current_schedule_state,
        load_current_player_strength,
    )


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "raw_games.csv"
TEAM_STRENGTH_PATH = DATA_DIR / "current_team_strength.csv"

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


def expected_score(team_elo: float, opponent_elo: float) -> float:
    """Calculate Elo expected win probability."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


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


def add_current_team_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add current rolling and season-to-date team features."""
    games = add_advanced_game_stats(games)
    games = games.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"]).copy()
    games["WIN"] = games["WL"].eq("W").astype(int)

    for window in ROLLING_WINDOWS:
        for column in BASE_STAT_COLUMNS:
            games[f"ROLLING_{column}_{window}"] = (
                games.groupby("TEAM_ID")[column]
                .transform(lambda series: series.rolling(window).mean())
            )

        games[f"ROLLING_WIN_PCT_{window}"] = (
            games.groupby("TEAM_ID")["WIN"]
            .transform(lambda series: series.rolling(window).mean())
        )

    for column in BASE_STAT_COLUMNS:
        games[f"SEASON_AVG_{column}"] = (
            games.groupby(["TEAM_ID", "SEASON"])[column]
            .transform(lambda series: series.expanding().mean())
        )

    games["SEASON_WIN_PCT"] = (
        games.groupby(["TEAM_ID", "SEASON"])["WIN"]
        .transform(lambda series: series.expanding().mean())
    )

    return games


def calculate_latest_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Calculate current Elo for every team."""
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.").astype(int)

    home_games = games[games["IS_HOME"] == 1][
        ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "WL"]
    ].rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_NAME": "HOME_TEAM",
            "WL": "HOME_WL",
        }
    )

    away_games = games[games["IS_HOME"] == 0][
        ["GAME_ID", "TEAM_ID", "TEAM_NAME"]
    ].rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "TEAM_NAME": "AWAY_TEAM",
        }
    )

    matchups = home_games.merge(away_games, on="GAME_ID", how="inner")
    matchups = matchups.sort_values(["GAME_DATE", "GAME_ID"])

    elo_ratings: dict[int, float] = {}
    team_names: dict[int, str] = {}

    for _, row in matchups.iterrows():
        home_team_id = int(row["HOME_TEAM_ID"])
        away_team_id = int(row["AWAY_TEAM_ID"])

        team_names[home_team_id] = row["HOME_TEAM"]
        team_names[away_team_id] = row["AWAY_TEAM"]

        home_elo = elo_ratings.get(home_team_id, BASE_ELO)
        away_elo = elo_ratings.get(away_team_id, BASE_ELO)

        home_expected = expected_score(home_elo + HOME_ELO_ADVANTAGE, away_elo)
        home_actual = 1.0 if row["HOME_WL"] == "W" else 0.0
        elo_change = K_FACTOR * (home_actual - home_expected)

        elo_ratings[home_team_id] = home_elo + elo_change
        elo_ratings[away_team_id] = away_elo - elo_change

    return pd.DataFrame(
        [
            {
                "TEAM_ID": team_id,
                "TEAM_NAME": team_names[team_id],
                "ELO": rating,
            }
            for team_id, rating in elo_ratings.items()
        ]
    )


def get_team_strength_columns() -> list[str]:
    """Return team strength columns needed by the app."""
    columns = []

    for window in ROLLING_WINDOWS:
        for stat_column in BASE_STAT_COLUMNS:
            columns.append(f"ROLLING_{stat_column}_{window}")

        columns.append(f"ROLLING_WIN_PCT_{window}")

    for stat_column in BASE_STAT_COLUMNS:
        columns.append(f"SEASON_AVG_{stat_column}")

    columns.append("SEASON_WIN_PCT")

    return columns


def build_current_team_strength() -> pd.DataFrame:
    """Build current team strength table for future predictions."""
    DATA_DIR.mkdir(exist_ok=True)

    games = load_raw_games()
    games_with_features = add_current_team_features(games)

    latest_rows = (
        games_with_features.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])
        .groupby("TEAM_ID")
        .tail(1)
        .copy()
    )

    feature_columns = get_team_strength_columns()

    latest_strength = latest_rows[
        [
            "TEAM_ID",
            "TEAM_NAME",
            "TEAM_ABBREVIATION",
            "GAME_DATE",
            *feature_columns,
        ]
    ].copy()

    elo_rankings = calculate_latest_elo(games)
    schedule_state = calculate_current_schedule_state(games)
    player_strength = load_current_player_strength()

    strength = latest_strength.merge(
        elo_rankings[["TEAM_ID", "ELO"]],
        on="TEAM_ID",
        how="left",
    )
    strength = strength.merge(schedule_state, on="TEAM_ID", how="left")

    if not player_strength.empty:
        strength = strength.merge(
            player_strength[["TEAM_ID", *CURRENT_PLAYER_STRENGTH_COLUMNS]],
            on="TEAM_ID",
            how="left",
        )
    else:
        for column in CURRENT_PLAYER_STRENGTH_COLUMNS:
            strength[column] = 0.0

    for column in [
        "DAYS_REST",
        "IS_BACK_TO_BACK",
        "IS_THIRD_IN_FOUR_DAYS",
        "GAMES_LAST_7_DAYS",
        "CURRENT_ROAD_STREAK",
        *CURRENT_PLAYER_STRENGTH_COLUMNS,
    ]:
        strength[column] = strength[column].fillna(0.0)

    strength = strength.dropna().sort_values("ELO", ascending=False).reset_index(drop=True)
    strength["ELO"] = strength["ELO"].round(1)

    strength.to_csv(TEAM_STRENGTH_PATH, index=False)

    print(f"Saved current team strength to {TEAM_STRENGTH_PATH}")
    print()
    print(
        strength[
            [
                "TEAM_NAME",
                "ELO",
                "SEASON_WIN_PCT",
                "ROLLING_WIN_PCT_10",
                "ROLLING_PLUS_MINUS_10",
                "ROLLING_PLUS_MINUS_5",
                "DAYS_REST",
                "PLAYER_TOP_8",
            ]
        ].head(15)
    )

    return strength


if __name__ == "__main__":
    build_current_team_strength()
