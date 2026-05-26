# src/mlb_features.py

from pathlib import Path
import math

import pandas as pd


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "mlb_raw_games.csv"
FEATURES_PATH = DATA_DIR / "mlb_model_features.csv"

BASE_ELO = 1500.0
HOME_ELO_ADVANTAGE = 24.0
K_FACTOR = 12.0
ROLLING_WINDOWS = [5, 10, 20]
DEFAULT_RUN_ENVIRONMENT = 4.4
DEFAULT_PARK_RUN_FACTOR = 1.0

TEAM_PARK_RUN_FACTORS = {
    "Arizona Diamondbacks": 1.03,
    "Athletics": 0.96,
    "Oakland Athletics": 0.96,
    "Atlanta Braves": 1.01,
    "Baltimore Orioles": 0.97,
    "Boston Red Sox": 1.05,
    "Chicago Cubs": 1.02,
    "Chicago White Sox": 0.98,
    "Cincinnati Reds": 1.06,
    "Cleveland Guardians": 0.96,
    "Colorado Rockies": 1.18,
    "Detroit Tigers": 0.97,
    "Houston Astros": 0.99,
    "Kansas City Royals": 1.01,
    "Los Angeles Angels": 0.98,
    "Los Angeles Dodgers": 1.00,
    "Miami Marlins": 0.94,
    "Milwaukee Brewers": 1.00,
    "Minnesota Twins": 0.99,
    "New York Mets": 0.97,
    "New York Yankees": 1.03,
    "Philadelphia Phillies": 1.02,
    "Pittsburgh Pirates": 0.96,
    "San Diego Padres": 0.95,
    "San Francisco Giants": 0.94,
    "Seattle Mariners": 0.95,
    "St. Louis Cardinals": 0.99,
    "Tampa Bay Rays": 0.97,
    "Texas Rangers": 1.03,
    "Toronto Blue Jays": 1.01,
    "Washington Nationals": 1.00,
}

TEAM_STRENGTH_COLUMNS = [
    "ELO",
    "SEASON_WIN_PCT",
    "SEASON_RUN_DIFF_PER_GAME",
    "SEASON_AVG_RUNS_FOR",
    "SEASON_AVG_RUNS_AGAINST",
    "SEASON_HOME_WIN_PCT",
    "SEASON_AWAY_WIN_PCT",
    "SEASON_HOME_RUN_DIFF_PER_GAME",
    "SEASON_AWAY_RUN_DIFF_PER_GAME",
    "ROLLING_WIN_PCT_5",
    "ROLLING_RUN_DIFF_5",
    "ROLLING_RUNS_FOR_5",
    "ROLLING_RUNS_AGAINST_5",
    "ROLLING_WIN_PCT_10",
    "ROLLING_RUN_DIFF_10",
    "ROLLING_RUNS_FOR_10",
    "ROLLING_RUNS_AGAINST_10",
    "ROLLING_WIN_PCT_20",
    "ROLLING_RUN_DIFF_20",
    "ROLLING_RUNS_FOR_20",
    "ROLLING_RUNS_AGAINST_20",
    "DAYS_REST",
    "GAMES_LAST_3_DAYS",
    "GAMES_LAST_5_DAYS",
    "RUNS_ALLOWED_LAST_3_DAYS",
    "BULLPEN_FATIGUE_PROXY",
    "PROJECTED_LINEUP_STRENGTH",
]

PITCHER_STRENGTH_COLUMNS = [
    "PITCHER_TEAM_WIN_PCT",
    "PITCHER_RUNS_ALLOWED_PER_START",
    "PITCHER_RUN_SUPPORT_PER_START",
    "PITCHER_STARTS",
    "PITCHER_DAYS_REST",
]


def load_raw_games() -> pd.DataFrame:
    """Load final MLB game rows."""
    if not RAW_GAMES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/mlb_raw_games.csv. Run: python src/mlb_collect_data.py"
        )

    games = pd.read_csv(RAW_GAMES_PATH)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return games.sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)


def expected_score(team_elo: float, opponent_elo: float) -> float:
    """Calculate Elo expected win probability."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def clamp_rest_days(value: float | int | None) -> float:
    """Cap rest days so long layoffs do not dominate."""
    if value is None or pd.isna(value):
        return 3.0
    return float(min(max(value, 0), 7))


def rolling_average(values: list[float], window: int, default: float) -> float:
    """Average the latest N values with a neutral default when unavailable."""
    if not values:
        return default
    selected = values[-window:]
    return float(sum(selected) / len(selected))


def normalize_pitcher_name(value: object) -> str:
    """Normalize pitcher names for historical starter proxies."""
    return str(value or "").strip().lower()


def get_probable_pitcher_name(game: pd.Series, side: str) -> str:
    """Return probable pitcher name from a game row."""
    column = "HOME_PROBABLE_PITCHER" if side == "home" else "AWAY_PROBABLE_PITCHER"
    return str(game.get(column, "") or "").strip()


def get_park_run_factor(team_name: str) -> float:
    """Return a static ballpark run-factor baseline."""
    return float(TEAM_PARK_RUN_FACTORS.get(str(team_name), DEFAULT_PARK_RUN_FACTOR))


def recent_rows_within_days(
    history: list[dict[str, float]],
    current_date: pd.Timestamp,
    days: int,
) -> list[dict[str, float]]:
    """Return historical team rows within a recent day window."""
    rows = []

    for row in history:
        game_date = pd.to_datetime(row["GAME_DATE"])

        if 0 < (current_date - game_date).days <= days:
            rows.append(row)

    return rows


def average_or_default(values: list[float], default: float) -> float:
    """Average values with a neutral default when empty."""
    return float(sum(values) / len(values)) if values else float(default)


def build_team_snapshot(
    team_id: int,
    season: int,
    current_date: pd.Timestamp,
    histories: dict[tuple[int, int], list[dict[str, float]]],
    elo_ratings: dict[int, float],
) -> dict[str, float]:
    """Build pregame team-form features from prior games only."""
    history = histories.get((season, team_id), [])
    wins = [float(row["WIN"]) for row in history]
    run_diffs = [float(row["RUN_DIFF"]) for row in history]
    runs_for = [float(row["RUNS_FOR"]) for row in history]
    runs_against = [float(row["RUNS_AGAINST"]) for row in history]
    home_rows = [row for row in history if bool(row.get("IS_HOME"))]
    away_rows = [row for row in history if not bool(row.get("IS_HOME"))]
    recent_3 = recent_rows_within_days(history, current_date, 3)
    recent_5 = recent_rows_within_days(history, current_date, 5)

    if history:
        season_win_pct = sum(wins) / len(wins)
        season_run_diff = sum(run_diffs) / len(run_diffs)
        season_runs_for = sum(runs_for) / len(runs_for)
        season_runs_against = sum(runs_against) / len(runs_against)
        last_game_date = pd.to_datetime(history[-1]["GAME_DATE"])
        days_rest = clamp_rest_days((current_date - last_game_date).days)
    else:
        season_win_pct = 0.5
        season_run_diff = 0.0
        season_runs_for = DEFAULT_RUN_ENVIRONMENT
        season_runs_against = DEFAULT_RUN_ENVIRONMENT
        days_rest = 3.0

    home_wins = [float(row["WIN"]) for row in home_rows]
    away_wins = [float(row["WIN"]) for row in away_rows]
    home_run_diffs = [float(row["RUN_DIFF"]) for row in home_rows]
    away_run_diffs = [float(row["RUN_DIFF"]) for row in away_rows]
    rolling_runs_for_10 = rolling_average(runs_for, 10, DEFAULT_RUN_ENVIRONMENT)
    rolling_runs_against_10 = rolling_average(runs_against, 10, DEFAULT_RUN_ENVIRONMENT)
    runs_allowed_last_3 = sum(float(row["RUNS_AGAINST"]) for row in recent_3)
    bullpen_fatigue = (
        len(recent_3) * 0.45
        + len(recent_5) * 0.20
        + runs_allowed_last_3 * 0.04
    )

    snapshot = {
        "ELO": float(elo_ratings.get(team_id, BASE_ELO)),
        "SEASON_WIN_PCT": float(season_win_pct),
        "SEASON_RUN_DIFF_PER_GAME": float(season_run_diff),
        "SEASON_AVG_RUNS_FOR": float(season_runs_for),
        "SEASON_AVG_RUNS_AGAINST": float(season_runs_against),
        "SEASON_HOME_WIN_PCT": average_or_default(home_wins, 0.5),
        "SEASON_AWAY_WIN_PCT": average_or_default(away_wins, 0.5),
        "SEASON_HOME_RUN_DIFF_PER_GAME": average_or_default(home_run_diffs, 0.0),
        "SEASON_AWAY_RUN_DIFF_PER_GAME": average_or_default(away_run_diffs, 0.0),
        "DAYS_REST": float(days_rest),
        "GAMES_LAST_3_DAYS": float(len(recent_3)),
        "GAMES_LAST_5_DAYS": float(len(recent_5)),
        "RUNS_ALLOWED_LAST_3_DAYS": float(runs_allowed_last_3),
        "BULLPEN_FATIGUE_PROXY": float(bullpen_fatigue),
        "PROJECTED_LINEUP_STRENGTH": float(
            (season_runs_for * 0.45)
            + (rolling_runs_for_10 * 0.35)
            + ((DEFAULT_RUN_ENVIRONMENT * 2 - rolling_runs_against_10) * 0.20)
        ),
    }

    for window in ROLLING_WINDOWS:
        snapshot[f"ROLLING_WIN_PCT_{window}"] = rolling_average(wins, window, 0.5)
        snapshot[f"ROLLING_RUN_DIFF_{window}"] = rolling_average(run_diffs, window, 0.0)
        snapshot[f"ROLLING_RUNS_FOR_{window}"] = rolling_average(runs_for, window, DEFAULT_RUN_ENVIRONMENT)
        snapshot[f"ROLLING_RUNS_AGAINST_{window}"] = rolling_average(runs_against, window, DEFAULT_RUN_ENVIRONMENT)

    return snapshot


def build_pitcher_snapshot(
    pitcher_name: str,
    season: int,
    current_date: pd.Timestamp,
    pitcher_histories: dict[tuple[int, str], list[dict[str, float]]],
) -> dict[str, float]:
    """Build pregame starter proxy features from prior starts only."""
    pitcher_key = normalize_pitcher_name(pitcher_name)
    history = pitcher_histories.get((season, pitcher_key), []) if pitcher_key else []

    if history:
        wins = [float(row["TEAM_WIN"]) for row in history]
        runs_allowed = [float(row["RUNS_ALLOWED"]) for row in history]
        run_support = [float(row["RUN_SUPPORT"]) for row in history]
        last_start = pd.to_datetime(history[-1]["GAME_DATE"])
        days_rest = clamp_rest_days((current_date - last_start).days)
        return {
            "PITCHER_TEAM_WIN_PCT": average_or_default(wins, 0.5),
            "PITCHER_RUNS_ALLOWED_PER_START": average_or_default(runs_allowed, DEFAULT_RUN_ENVIRONMENT),
            "PITCHER_RUN_SUPPORT_PER_START": average_or_default(run_support, DEFAULT_RUN_ENVIRONMENT),
            "PITCHER_STARTS": float(min(len(history), 30)),
            "PITCHER_DAYS_REST": float(days_rest),
        }

    return {
        "PITCHER_TEAM_WIN_PCT": 0.5,
        "PITCHER_RUNS_ALLOWED_PER_START": DEFAULT_RUN_ENVIRONMENT,
        "PITCHER_RUN_SUPPORT_PER_START": DEFAULT_RUN_ENVIRONMENT,
        "PITCHER_STARTS": 0.0,
        "PITCHER_DAYS_REST": 5.0,
    }


def update_team_history(
    histories: dict[tuple[int, int], list[dict[str, float]]],
    season: int,
    team_id: int,
    game_date: pd.Timestamp,
    runs_for: int,
    runs_against: int,
    is_home: bool,
) -> None:
    """Append a completed team-game result."""
    histories.setdefault((season, team_id), []).append(
        {
            "GAME_DATE": game_date,
            "WIN": float(runs_for > runs_against),
            "RUN_DIFF": float(runs_for - runs_against),
            "RUNS_FOR": float(runs_for),
            "RUNS_AGAINST": float(runs_against),
            "IS_HOME": bool(is_home),
        }
    )


def update_pitcher_history(
    pitcher_histories: dict[tuple[int, str], list[dict[str, float]]],
    season: int,
    pitcher_name: str,
    game_date: pd.Timestamp,
    runs_for: int,
    runs_against: int,
) -> None:
    """Append a probable-starter result for proxy pitcher features."""
    pitcher_key = normalize_pitcher_name(pitcher_name)

    if not pitcher_key:
        return

    pitcher_histories.setdefault((season, pitcher_key), []).append(
        {
            "GAME_DATE": game_date,
            "TEAM_WIN": float(runs_for > runs_against),
            "RUNS_ALLOWED": float(runs_against),
            "RUN_SUPPORT": float(runs_for),
        }
    )


def create_matchup_rows(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model-ready matchup rows and latest team snapshots."""
    games = games.copy()
    games = games.dropna(subset=["HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_SCORE", "AWAY_SCORE"])
    games["HOME_TEAM_ID"] = games["HOME_TEAM_ID"].astype(int)
    games["AWAY_TEAM_ID"] = games["AWAY_TEAM_ID"].astype(int)
    games["HOME_SCORE"] = games["HOME_SCORE"].astype(int)
    games["AWAY_SCORE"] = games["AWAY_SCORE"].astype(int)
    games["SEASON"] = games["SEASON"].astype(int)
    games = games.sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)

    histories: dict[tuple[int, int], list[dict[str, float]]] = {}
    pitcher_histories: dict[tuple[int, str], list[dict[str, float]]] = {}
    elo_ratings: dict[int, float] = {}
    team_names: dict[int, str] = {}
    latest_snapshots: dict[int, dict] = {}
    rows = []

    for _, game in games.iterrows():
        home_id = int(game["HOME_TEAM_ID"])
        away_id = int(game["AWAY_TEAM_ID"])
        season = int(game["SEASON"])
        game_date = pd.to_datetime(game["GAME_DATE"])
        home_score = int(game["HOME_SCORE"])
        away_score = int(game["AWAY_SCORE"])
        home_pitcher = get_probable_pitcher_name(game, "home")
        away_pitcher = get_probable_pitcher_name(game, "away")

        team_names[home_id] = str(game["HOME_TEAM"])
        team_names[away_id] = str(game["AWAY_TEAM"])

        home_snapshot = build_team_snapshot(home_id, season, game_date, histories, elo_ratings)
        away_snapshot = build_team_snapshot(away_id, season, game_date, histories, elo_ratings)
        home_pitcher_snapshot = build_pitcher_snapshot(home_pitcher, season, game_date, pitcher_histories)
        away_pitcher_snapshot = build_pitcher_snapshot(away_pitcher, season, game_date, pitcher_histories)
        home_elo_probability = expected_score(
            home_snapshot["ELO"] + HOME_ELO_ADVANTAGE,
            away_snapshot["ELO"],
        )
        home_win = int(home_score > away_score)

        row = {
            "GAME_PK": str(game["GAME_PK"]),
            "GAME_DATE": game_date,
            "SEASON": season,
            "GAME_TYPE": str(game.get("GAME_TYPE", "")),
            "HOME_TEAM": str(game["HOME_TEAM"]),
            "AWAY_TEAM": str(game["AWAY_TEAM"]),
            "HOME_WIN": home_win,
            "HOME_ELO_WIN_PROB": home_elo_probability,
            "HOME_PARK_RUN_FACTOR": get_park_run_factor(str(game["HOME_TEAM"])),
        }

        for column in TEAM_STRENGTH_COLUMNS:
            row[f"DIFF_{column}"] = home_snapshot[column] - away_snapshot[column]

        for column in PITCHER_STRENGTH_COLUMNS:
            row[f"DIFF_{column}"] = home_pitcher_snapshot[column] - away_pitcher_snapshot[column]

        rows.append(row)

        home_actual = float(home_win)
        elo_change = K_FACTOR * (home_actual - home_elo_probability)
        elo_ratings[home_id] = home_snapshot["ELO"] + elo_change
        elo_ratings[away_id] = away_snapshot["ELO"] - elo_change

        update_team_history(histories, season, home_id, game_date, home_score, away_score, is_home=True)
        update_team_history(histories, season, away_id, game_date, away_score, home_score, is_home=False)
        update_pitcher_history(pitcher_histories, season, home_pitcher, game_date, home_score, away_score)
        update_pitcher_history(pitcher_histories, season, away_pitcher, game_date, away_score, home_score)

        latest_snapshots[home_id] = {
            "TEAM_ID": home_id,
            "TEAM_NAME": team_names[home_id],
            "LAST_GAME_DATE": game_date,
            **build_team_snapshot(home_id, season, game_date, histories, elo_ratings),
        }
        latest_snapshots[away_id] = {
            "TEAM_ID": away_id,
            "TEAM_NAME": team_names[away_id],
            "LAST_GAME_DATE": game_date,
            **build_team_snapshot(away_id, season, game_date, histories, elo_ratings),
        }

    features = pd.DataFrame(rows)
    team_strength = pd.DataFrame(latest_snapshots.values())

    if not team_strength.empty:
        team_strength = team_strength.sort_values("ELO", ascending=False).reset_index(drop=True)

    return features, team_strength


def build_current_pitcher_strength(games: pd.DataFrame) -> pd.DataFrame:
    """Build latest probable-starter proxy rows for future predictions."""
    games = games.copy()
    games = games.dropna(subset=["HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_SCORE", "AWAY_SCORE"])

    if games.empty:
        return pd.DataFrame()

    games["HOME_SCORE"] = games["HOME_SCORE"].astype(int)
    games["AWAY_SCORE"] = games["AWAY_SCORE"].astype(int)
    games["SEASON"] = games["SEASON"].astype(int)
    games = games.sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)
    pitcher_histories: dict[tuple[int, str], list[dict[str, float]]] = {}
    latest_rows: dict[tuple[int, str], dict] = {}

    for _, game in games.iterrows():
        season = int(game["SEASON"])
        game_date = pd.to_datetime(game["GAME_DATE"])

        for side in ["home", "away"]:
            pitcher_name = get_probable_pitcher_name(game, side)
            pitcher_key = normalize_pitcher_name(pitcher_name)

            if not pitcher_key:
                continue

            if side == "home":
                team_id = int(game["HOME_TEAM_ID"])
                team_name = str(game["HOME_TEAM"])
                runs_for = int(game["HOME_SCORE"])
                runs_against = int(game["AWAY_SCORE"])
                pitcher_id = game.get("HOME_PROBABLE_PITCHER_ID")
            else:
                team_id = int(game["AWAY_TEAM_ID"])
                team_name = str(game["AWAY_TEAM"])
                runs_for = int(game["AWAY_SCORE"])
                runs_against = int(game["HOME_SCORE"])
                pitcher_id = game.get("AWAY_PROBABLE_PITCHER_ID")

            update_pitcher_history(
                pitcher_histories,
                season,
                pitcher_name,
                game_date,
                runs_for,
                runs_against,
            )
            latest_rows[(season, pitcher_key)] = {
                "SEASON": season,
                "PITCHER_NAME": pitcher_name,
                "PITCHER_KEY": pitcher_key,
                "PITCHER_ID": pitcher_id,
                "TEAM_ID": team_id,
                "TEAM_NAME": team_name,
                "LAST_START_DATE": game_date,
                **build_pitcher_snapshot(
                    pitcher_name,
                    season,
                    game_date + pd.Timedelta(days=1),
                    pitcher_histories,
                ),
            }

    pitcher_strength = pd.DataFrame(latest_rows.values())

    if pitcher_strength.empty:
        return pitcher_strength

    return pitcher_strength.sort_values(
        ["SEASON", "PITCHER_STARTS", "PITCHER_RUNS_ALLOWED_PER_START"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return model feature columns."""
    return [
        column
        for column in data.columns
        if column.startswith("DIFF_") or column in {"HOME_ELO_WIN_PROB", "HOME_PARK_RUN_FACTOR"}
    ]


def build_features() -> pd.DataFrame:
    """Build model-ready MLB matchup features."""
    DATA_DIR.mkdir(exist_ok=True)
    games = load_raw_games()
    features, _ = create_matchup_rows(games)
    features.to_csv(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} MLB matchup rows to {FEATURES_PATH}")
    print()
    print("Feature columns:")
    for column in get_feature_columns(features):
        print(f"- {column}")

    return features


if __name__ == "__main__":
    build_features()
