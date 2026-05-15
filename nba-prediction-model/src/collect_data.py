# src/collect_data.py

from pathlib import Path
from time import sleep

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "raw_games.csv"

START_YEAR = 2018
END_YEAR = 2025
SEASON_TYPES = ["Regular Season", "Playoffs"]


def format_season(year: int) -> str:
    """Convert a start year into NBA season format."""
    return f"{year}-{str(year + 1)[-2:]}"


def fetch_season_games(season: str, season_type: str) -> pd.DataFrame:
    """Fetch NBA team game logs for one season and season type."""
    response = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
        timeout=60,
    )

    games = response.get_data_frames()[0]

    if games.empty:
        raise ValueError(f"No {season_type} games returned for season {season}")

    games["SEASON"] = season
    games["SEASON_TYPE"] = season_type
    return games


def collect_games(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    season_types: list[str] | None = None,
) -> pd.DataFrame:
    """Collect NBA regular-season and playoff team game logs."""
    DATA_DIR.mkdir(exist_ok=True)
    season_types = season_types or SEASON_TYPES

    all_games = []

    for year in range(start_year, end_year + 1):
        season = format_season(year)

        for season_type in season_types:
            print(f"Fetching {season} {season_type}...")

            try:
                season_games = fetch_season_games(season, season_type)
            except Exception as error:
                print(f"Skipping {season} {season_type}: {error}")
                continue

            all_games.append(season_games)
            sleep(1)

    if not all_games:
        raise RuntimeError("No NBA game data was collected.")

    games = pd.concat(all_games, ignore_index=True)
    games.to_csv(RAW_GAMES_PATH, index=False)

    print(f"Saved {len(games)} team-game rows to {RAW_GAMES_PATH}")
    print()
    print("Seasons included:")
    print(sorted(games["SEASON"].unique()))

    if "SEASON_TYPE" in games.columns:
        print()
        print("Season types included:")
        print(games["SEASON_TYPE"].value_counts().to_string())

    return games


if __name__ == "__main__":
    collect_games()
