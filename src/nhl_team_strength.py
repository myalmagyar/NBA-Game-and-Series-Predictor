# src/nhl_team_strength.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nhl_features import (
    ELO_K,
    HOME_ELO_ADVANTAGE,
    STARTING_ELO,
    empty_team_state,
    expected_score,
    team_pregame_features,
    update_team_state,
)


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "nhl_raw_games.csv"
TEAM_STRENGTH_PATH = DATA_DIR / "nhl_current_team_strength.csv"


def build_current_team_strength(raw_games: pd.DataFrame) -> pd.DataFrame:
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

    if games.empty:
        return pd.DataFrame()

    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    latest_season = games.sort_values("GAME_DATE")["SEASON"].dropna().iloc[-1]
    elos: dict[str, float] = {}
    states: dict[tuple[object, str], dict] = {}
    team_ids: dict[str, object] = {}
    team_abbrevs: dict[str, object] = {}

    for _, game in games.iterrows():
        season = game["SEASON"]
        home_team = str(game["HOME_TEAM"])
        away_team = str(game["AWAY_TEAM"])
        game_date = game["GAME_DATE"]
        home_score = float(game["HOME_SCORE"])
        away_score = float(game["AWAY_SCORE"])
        team_ids[home_team] = game.get("HOME_TEAM_ID")
        team_ids[away_team] = game.get("AWAY_TEAM_ID")
        team_abbrevs[home_team] = game.get("HOME_TEAM_ABBREV")
        team_abbrevs[away_team] = game.get("AWAY_TEAM_ABBREV")
        home_key = (season, home_team)
        away_key = (season, away_team)
        states.setdefault(home_key, empty_team_state())
        states.setdefault(away_key, empty_team_state())
        home_elo = elos.get(home_team, STARTING_ELO)
        away_elo = elos.get(away_team, STARTING_ELO)
        home_elo_prob = expected_score(home_elo + HOME_ELO_ADVANTAGE, away_elo)
        actual = 1.0 if home_score > away_score else 0.0
        margin_multiplier = min(abs(home_score - away_score), 3.0) / 3.0
        change = ELO_K * (0.85 + margin_multiplier) * (actual - home_elo_prob)
        elos[home_team] = home_elo + change
        elos[away_team] = away_elo - change
        update_team_state(states[home_key], game_date, home_score, away_score)
        update_team_state(states[away_key], game_date, away_score, home_score)

    season_games = games[games["SEASON"].eq(latest_season)].copy()
    teams = sorted(
        set(season_games["HOME_TEAM"].dropna().astype(str))
        | set(season_games["AWAY_TEAM"].dropna().astype(str))
    )
    rows = []

    for team in teams:
        state = states.get((latest_season, team), empty_team_state())
        latest = team_pregame_features(
            state=state,
            game_date=games["GAME_DATE"].max(),
            elo=elos.get(team, STARTING_ELO),
        )
        rows.append(
            {
                "TEAM_ID": team_ids.get(team),
                "TEAM_ABBREV": team_abbrevs.get(team),
                "TEAM_NAME": team,
                "SEASON": latest_season,
                "ELO": round(float(latest["ELO"]), 1),
                "SEASON_WIN_PCT": latest["SEASON_WIN_PCT"],
                "SEASON_GOAL_DIFF_PER_GAME": latest["SEASON_GOAL_DIFF_PER_GAME"],
                "SEASON_AVG_GOALS_FOR": latest["SEASON_AVG_GOALS_FOR"],
                "SEASON_AVG_GOALS_AGAINST": latest["SEASON_AVG_GOALS_AGAINST"],
                "ROLLING_WIN_PCT_5": latest["ROLLING_WIN_PCT_5"],
                "ROLLING_GOAL_DIFF_5": latest["ROLLING_GOAL_DIFF_5"],
                "ROLLING_GOALS_FOR_5": latest["ROLLING_GOALS_FOR_5"],
                "ROLLING_GOALS_AGAINST_5": latest["ROLLING_GOALS_AGAINST_5"],
                "ROLLING_WIN_PCT_10": latest["ROLLING_WIN_PCT_10"],
                "ROLLING_GOAL_DIFF_10": latest["ROLLING_GOAL_DIFF_10"],
                "ROLLING_GOALS_FOR_10": latest["ROLLING_GOALS_FOR_10"],
                "ROLLING_GOALS_AGAINST_10": latest["ROLLING_GOALS_AGAINST_10"],
                "DAYS_REST": latest["DAYS_REST"],
                "LAST_GAME_DATE": state.get("last_date"),
                "GAMES_PLAYED": state.get("games", 0),
                "WINS": state.get("wins", 0),
                "LOSSES": int(state.get("games", 0)) - int(state.get("wins", 0)),
            }
        )

    strength = pd.DataFrame(rows)
    return strength.sort_values("ELO", ascending=False).reset_index(drop=True)


def main() -> None:
    if not RAW_GAMES_PATH.exists():
        raise FileNotFoundError("Missing data/nhl_raw_games.csv. Run: python src/nhl_collect_data.py")

    raw = pd.read_csv(RAW_GAMES_PATH)
    strength = build_current_team_strength(raw)
    DATA_DIR.mkdir(exist_ok=True)
    strength.to_csv(TEAM_STRENGTH_PATH, index=False)
    print(f"Saved NHL current team strength to {TEAM_STRENGTH_PATH}")
    if not strength.empty:
        print()
        print(
            strength[
                [
                    "TEAM_NAME",
                    "ELO",
                    "SEASON_WIN_PCT",
                    "SEASON_GOAL_DIFF_PER_GAME",
                    "ROLLING_WIN_PCT_10",
                    "DAYS_REST",
                ]
            ].head(15).to_string(index=False)
        )


if __name__ == "__main__":
    main()
