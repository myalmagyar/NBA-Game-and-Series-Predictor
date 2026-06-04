# src/nhl_features.py

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "nhl_raw_games.csv"
FEATURES_PATH = DATA_DIR / "nhl_model_features.csv"

HOME_ELO_ADVANTAGE = 42.0
ELO_K = 24.0
STARTING_ELO = 1500.0

FEATURE_COLUMNS = [
    "HOME_ELO_WIN_PROB",
    "DIFF_ELO",
    "DIFF_SEASON_WIN_PCT",
    "DIFF_SEASON_GOAL_DIFF_PER_GAME",
    "DIFF_SEASON_AVG_GOALS_FOR",
    "DIFF_SEASON_AVG_GOALS_AGAINST",
    "DIFF_ROLLING_WIN_PCT_5",
    "DIFF_ROLLING_GOAL_DIFF_5",
    "DIFF_ROLLING_GOALS_FOR_5",
    "DIFF_ROLLING_GOALS_AGAINST_5",
    "DIFF_ROLLING_WIN_PCT_10",
    "DIFF_ROLLING_GOAL_DIFF_10",
    "DIFF_ROLLING_GOALS_FOR_10",
    "DIFF_ROLLING_GOALS_AGAINST_10",
    "DIFF_DAYS_REST",
    "IS_PLAYOFF_GAME",
]


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def empty_team_state() -> dict:
    return {
        "games": 0,
        "wins": 0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "last_date": None,
        "history": [],
    }


def rolling_value(history: list[dict], key: str, window: int) -> float:
    selected = history[-window:]
    if not selected:
        return 0.0
    return float(sum(row[key] for row in selected) / len(selected))


def team_pregame_features(state: dict, game_date: pd.Timestamp, elo: float) -> dict[str, float]:
    games = max(int(state["games"]), 1)
    last_date = state.get("last_date")
    days_rest = 7.0
    if last_date is not None and not pd.isna(last_date):
        days_rest = min(max((game_date - last_date).days, 0), 7)

    history = state["history"]
    return {
        "ELO": float(elo),
        "SEASON_WIN_PCT": float(state["wins"] / games) if state["games"] else 0.5,
        "SEASON_GOAL_DIFF_PER_GAME": float((state["goals_for"] - state["goals_against"]) / games) if state["games"] else 0.0,
        "SEASON_AVG_GOALS_FOR": float(state["goals_for"] / games) if state["games"] else 0.0,
        "SEASON_AVG_GOALS_AGAINST": float(state["goals_against"] / games) if state["games"] else 0.0,
        "ROLLING_WIN_PCT_5": rolling_value(history, "win", 5),
        "ROLLING_GOAL_DIFF_5": rolling_value(history, "goal_diff", 5),
        "ROLLING_GOALS_FOR_5": rolling_value(history, "goals_for", 5),
        "ROLLING_GOALS_AGAINST_5": rolling_value(history, "goals_against", 5),
        "ROLLING_WIN_PCT_10": rolling_value(history, "win", 10),
        "ROLLING_GOAL_DIFF_10": rolling_value(history, "goal_diff", 10),
        "ROLLING_GOALS_FOR_10": rolling_value(history, "goals_for", 10),
        "ROLLING_GOALS_AGAINST_10": rolling_value(history, "goals_against", 10),
        "DAYS_REST": float(days_rest),
    }


def update_team_state(state: dict, game_date: pd.Timestamp, goals_for: float, goals_against: float) -> None:
    win = 1 if goals_for > goals_against else 0
    state["games"] += 1
    state["wins"] += win
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    state["last_date"] = game_date
    state["history"].append(
        {
            "win": float(win),
            "goals_for": float(goals_for),
            "goals_against": float(goals_against),
            "goal_diff": float(goals_for - goals_against),
        }
    )


def build_nhl_features(raw_games: pd.DataFrame) -> pd.DataFrame:
    games = raw_games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games["HOME_SCORE"] = pd.to_numeric(games["HOME_SCORE"], errors="coerce")
    games["AWAY_SCORE"] = pd.to_numeric(games["AWAY_SCORE"], errors="coerce")
    games = games[
        games["GAME_TYPE"].isin([2, 3])
        & games["HOME_SCORE"].notna()
        & games["AWAY_SCORE"].notna()
        & games["GAME_DATE"].notna()
    ].copy()
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    elos: dict[str, float] = {}
    states: dict[tuple[object, str], dict] = {}
    rows = []

    for _, game in games.iterrows():
        season = game["SEASON"]
        home_team = str(game["HOME_TEAM"])
        away_team = str(game["AWAY_TEAM"])
        game_date = game["GAME_DATE"]
        home_score = float(game["HOME_SCORE"])
        away_score = float(game["AWAY_SCORE"])
        home_key = (season, home_team)
        away_key = (season, away_team)
        states.setdefault(home_key, empty_team_state())
        states.setdefault(away_key, empty_team_state())
        home_elo = elos.get(home_team, STARTING_ELO)
        away_elo = elos.get(away_team, STARTING_ELO)
        home_features = team_pregame_features(states[home_key], game_date, home_elo)
        away_features = team_pregame_features(states[away_key], game_date, away_elo)
        home_elo_prob = expected_score(home_elo + HOME_ELO_ADVANTAGE, away_elo)

        row = {
            "GAME_ID": game["GAME_ID"],
            "GAME_DATE": game_date,
            "SEASON": season,
            "GAME_TYPE": game["GAME_TYPE"],
            "HOME_TEAM": home_team,
            "AWAY_TEAM": away_team,
            "HOME_SCORE": home_score,
            "AWAY_SCORE": away_score,
            "HOME_WIN": 1 if home_score > away_score else 0,
            "HOME_ELO_WIN_PROB": home_elo_prob,
            "DIFF_ELO": home_elo - away_elo,
            "IS_PLAYOFF_GAME": 1 if int(game["GAME_TYPE"]) == 3 else 0,
        }

        for feature_name in [
            "SEASON_WIN_PCT",
            "SEASON_GOAL_DIFF_PER_GAME",
            "SEASON_AVG_GOALS_FOR",
            "SEASON_AVG_GOALS_AGAINST",
            "ROLLING_WIN_PCT_5",
            "ROLLING_GOAL_DIFF_5",
            "ROLLING_GOALS_FOR_5",
            "ROLLING_GOALS_AGAINST_5",
            "ROLLING_WIN_PCT_10",
            "ROLLING_GOAL_DIFF_10",
            "ROLLING_GOALS_FOR_10",
            "ROLLING_GOALS_AGAINST_10",
            "DAYS_REST",
        ]:
            row[f"DIFF_{feature_name}"] = home_features[feature_name] - away_features[feature_name]

        rows.append(row)

        actual = 1.0 if home_score > away_score else 0.0
        margin_multiplier = min(abs(home_score - away_score), 3.0) / 3.0
        k = ELO_K * (0.85 + margin_multiplier)
        change = k * (actual - home_elo_prob)
        elos[home_team] = home_elo + change
        elos[away_team] = away_elo - change
        update_team_state(states[home_key], game_date, home_score, away_score)
        update_team_state(states[away_key], game_date, away_score, home_score)

    features = pd.DataFrame(rows)
    if features.empty:
        return features

    for column in FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
    return features


def main() -> None:
    if not RAW_GAMES_PATH.exists():
        raise FileNotFoundError("Missing data/nhl_raw_games.csv. Run: python src/nhl_collect_data.py")

    raw = pd.read_csv(RAW_GAMES_PATH)
    features = build_nhl_features(raw)
    DATA_DIR.mkdir(exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    print(f"Saved {len(features):,} NHL matchup rows to {FEATURES_PATH}")
    print("\nFeature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")


if __name__ == "__main__":
    main()
