# src/nhl_collect_data.py

from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse
import time

import pandas as pd

from src.nhl_api import TEAM_ABBREVIATIONS, current_nhl_season, load_club_schedule, parse_game


DATA_DIR = Path("data")
RAW_GAMES_PATH = DATA_DIR / "nhl_raw_games.csv"
DEFAULT_START_SEASON = 20212022


def iter_seasons(start_season: int = DEFAULT_START_SEASON, end_season: int | None = None) -> list[int]:
    end_season = end_season or current_nhl_season(date.today())
    start_year = int(str(start_season)[:4])
    end_year = int(str(end_season)[:4])
    return [int(f"{year}{year + 1}") for year in range(start_year, end_year + 1)]


def collect_nhl_games(
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
    pause_seconds: float = 0.08,
) -> pd.DataFrame:
    """Collect NHL regular season and playoff games from team schedule feeds."""
    rows = []
    seen_game_ids: set[str] = set()

    for season in iter_seasons(start_season, end_season):
        print(f"Fetching NHL season {season}...")

        for team_abbrev in TEAM_ABBREVIATIONS:
            try:
                payload = load_club_schedule(team_abbrev, season)
            except Exception as error:
                print(f"Skipping {team_abbrev} {season}: {error}")
                continue

            for game in payload.get("games", []):
                game_id = str(game.get("id") or "")
                if not game_id or game_id in seen_game_ids:
                    continue

                row = parse_game(game)
                if row.get("GAME_TYPE") not in {2, 3}:
                    continue

                seen_game_ids.add(game_id)
                rows.append(row)

            if pause_seconds:
                time.sleep(pause_seconds)

    games = pd.DataFrame(rows)
    if games.empty:
        return games

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return games


def save_nhl_games(games: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    games.to_csv(RAW_GAMES_PATH, index=False)
    print(f"Saved {len(games):,} NHL game rows to {RAW_GAMES_PATH}")
    if not games.empty:
        print("\nSeasons included:")
        print(sorted(games["SEASON"].dropna().unique().tolist()))
        print("\nGame types:")
        print(games["GAME_TYPE"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--pause-seconds", type=float, default=0.08)
    args = parser.parse_args()

    save_nhl_games(
        collect_nhl_games(
            start_season=args.start_season,
            end_season=args.end_season,
            pause_seconds=args.pause_seconds,
        )
    )


if __name__ == "__main__":
    main()
