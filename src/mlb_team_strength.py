# src/mlb_team_strength.py

from pathlib import Path
from datetime import date

import pandas as pd

try:
    from src.mlb_features import create_matchup_rows, build_current_pitcher_strength, load_raw_games
except ModuleNotFoundError:
    from mlb_features import create_matchup_rows, build_current_pitcher_strength, load_raw_games


DATA_DIR = Path("data")
TEAM_STRENGTH_PATH = DATA_DIR / "mlb_current_team_strength.csv"
PITCHER_STRENGTH_PATH = DATA_DIR / "mlb_current_pitcher_strength.csv"


def build_current_team_strength() -> pd.DataFrame:
    """Build current MLB team-strength table for future predictions."""
    DATA_DIR.mkdir(exist_ok=True)
    games = load_raw_games()
    _, team_strength = create_matchup_rows(games)
    pitcher_strength = build_current_pitcher_strength(games)

    if team_strength.empty:
        raise RuntimeError("No MLB team strength rows were created.")

    team_strength["LAST_GAME_DATE"] = pd.to_datetime(team_strength["LAST_GAME_DATE"])
    team_strength["DAYS_REST"] = (
        pd.Timestamp(date.today()) - team_strength["LAST_GAME_DATE"]
    ).dt.days.clip(lower=0, upper=7)
    team_strength["ELO"] = team_strength["ELO"].round(1)
    team_strength.to_csv(TEAM_STRENGTH_PATH, index=False)

    if not pitcher_strength.empty:
        pitcher_strength["LAST_START_DATE"] = pd.to_datetime(pitcher_strength["LAST_START_DATE"])
        pitcher_strength["PITCHER_DAYS_REST"] = (
            pd.Timestamp(date.today()) - pitcher_strength["LAST_START_DATE"]
        ).dt.days.clip(lower=0, upper=7)
        pitcher_strength.to_csv(PITCHER_STRENGTH_PATH, index=False)

    print(f"Saved MLB current team strength to {TEAM_STRENGTH_PATH}")
    if not pitcher_strength.empty:
        print(f"Saved MLB current pitcher strength to {PITCHER_STRENGTH_PATH}")
    print()
    print(
        team_strength[
            [
                "TEAM_NAME",
                "ELO",
                "SEASON_WIN_PCT",
                "SEASON_RUN_DIFF_PER_GAME",
                "ROLLING_WIN_PCT_10",
                "ROLLING_RUN_DIFF_10",
                "DAYS_REST",
            ]
        ].head(15).to_string(index=False)
    )

    return team_strength


if __name__ == "__main__":
    build_current_team_strength()
