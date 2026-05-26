# src/mlb_collect_data.py

from datetime import date
from pathlib import Path
from time import sleep
import argparse

import pandas as pd

try:
    from src.mlb_api import fetch_schedule, iter_schedule_games, normalize_schedule_game
except ModuleNotFoundError:
    from mlb_api import fetch_schedule, iter_schedule_games, normalize_schedule_game


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "mlb_raw_games.csv"

START_YEAR = 2021
END_YEAR = date.today().year
SEASON_START_MONTH_DAY = "02-15"
SEASON_END_MONTH_DAY = "11-30"
DEFAULT_GAME_TYPES = {"R", "F", "D", "L", "W"}


def collect_mlb_games(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    final_only: bool = True,
    include_spring_training: bool = False,
) -> pd.DataFrame:
    """Collect MLB game rows from the public Stats API."""
    DATA_DIR.mkdir(exist_ok=True)
    rows = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-{SEASON_START_MONTH_DAY}"
        end_date = f"{year}-{SEASON_END_MONTH_DAY}"
        print(f"Fetching MLB schedule {start_date} to {end_date}...")

        payload = fetch_schedule(start_date=start_date, end_date=end_date)

        for game in iter_schedule_games(payload):
            row = normalize_schedule_game(game)

            if final_only and not row["IS_FINAL"]:
                continue

            if not include_spring_training and row["GAME_TYPE"] not in DEFAULT_GAME_TYPES:
                continue

            if not row["HOME_TEAM"] or not row["AWAY_TEAM"]:
                continue

            if row["HOME_SCORE"] is None or row["AWAY_SCORE"] is None:
                continue

            row["SEASON"] = year
            rows.append(row)

        sleep(1)

    if not rows:
        raise RuntimeError("No MLB games were collected.")

    games = pd.DataFrame(rows).sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)
    games.to_csv(RAW_GAMES_PATH, index=False)

    print(f"Saved {len(games)} MLB game rows to {RAW_GAMES_PATH}")
    print()
    print("Seasons included:")
    print(sorted(games["SEASON"].unique()))
    print()
    print("Game types:")
    print(games["GAME_TYPE"].value_counts().to_string())

    return games


def main() -> None:
    """Collect MLB data from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Save unfinished schedule rows too. Training ignores them.",
    )
    parser.add_argument(
        "--include-spring-training",
        action="store_true",
        help="Include spring-training games. Default is regular season and postseason only.",
    )
    args = parser.parse_args()

    collect_mlb_games(
        start_year=args.start_year,
        end_year=args.end_year,
        final_only=not args.include_unfinished,
        include_spring_training=args.include_spring_training,
    )


if __name__ == "__main__":
    main()
