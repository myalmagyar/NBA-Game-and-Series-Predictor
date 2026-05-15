# src/player_strength.py

from pathlib import Path
from time import sleep

import pandas as pd

try:
    from src.feature_utils import TEAM_PLAYER_STRENGTH_BY_SEASON_PATH, aggregate_player_strength
    from src.player_impact import calculate_player_impact, fetch_player_stats
except ModuleNotFoundError:
    from feature_utils import TEAM_PLAYER_STRENGTH_BY_SEASON_PATH, aggregate_player_strength
    from player_impact import calculate_player_impact, fetch_player_stats


DATA_DIR = Path("data")

# Previous-season player strength is used to avoid leaking full current-season stats
# into early historical games.
START_YEAR = 2017
END_YEAR = 2024


def format_season(year: int) -> str:
    """Convert a start year into NBA season format."""
    return f"{year}-{str(year + 1)[-2:]}"


def build_team_player_strength_for_season(season: str) -> pd.DataFrame:
    """Fetch player stats and aggregate team player strength for one season."""
    stats = fetch_player_stats(season)
    player_impact = calculate_player_impact(stats)
    team_strength = aggregate_player_strength(player_impact)
    team_strength["SEASON"] = season
    return team_strength[
        [
            "SEASON",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "PLAYER_TOP_5",
            "PLAYER_TOP_8",
            "PLAYER_DEPTH",
            "STAR_COUNT",
        ]
    ]


def build_team_player_strength_by_season(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """Build team player-strength features by season."""
    DATA_DIR.mkdir(exist_ok=True)

    rows = []

    for year in range(start_year, end_year + 1):
        season = format_season(year)
        print(f"Fetching player strength for {season}...")

        try:
            rows.append(build_team_player_strength_for_season(season))
        except Exception as error:
            print(f"Skipping {season}: {error}")
            continue

        sleep(1)

    if not rows:
        raise RuntimeError("No team player-strength data was collected.")

    player_strength = pd.concat(rows, ignore_index=True)
    player_strength.to_csv(TEAM_PLAYER_STRENGTH_BY_SEASON_PATH, index=False)

    print(f"Saved player-strength data to {TEAM_PLAYER_STRENGTH_BY_SEASON_PATH}")
    print()
    print(player_strength.head(25).to_string(index=False))

    return player_strength


if __name__ == "__main__":
    build_team_player_strength_by_season()
