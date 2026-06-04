# src/nhl_api.py

from __future__ import annotations

from datetime import date
import json
from urllib.parse import quote
from urllib.request import Request, urlopen


BASE_URL = "https://api-web.nhle.com/v1"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

TEAM_ABBREVIATIONS = [
    "ANA",
    "ARI",
    "BOS",
    "BUF",
    "CAR",
    "CBJ",
    "CGY",
    "CHI",
    "COL",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NJD",
    "NSH",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SEA",
    "SJS",
    "STL",
    "TBL",
    "TOR",
    "UTA",
    "VAN",
    "VGK",
    "WPG",
    "WSH",
]


def current_nhl_season(today: date | None = None) -> int:
    """Return NHL season id like 20252026."""
    today = today or date.today()
    start_year = today.year if today.month >= 9 else today.year - 1
    return int(f"{start_year}{start_year + 1}")


def season_label(season: int | str) -> str:
    text = str(season)
    return f"{text[:4]}-{text[-2:]}" if len(text) == 8 else text


def fetch_json(path: str) -> dict:
    """Fetch one NHL API JSON payload."""
    url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def localized_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("default") or value.get("en") or "").strip()
    return str(value or "").strip()


def team_full_name(team: dict) -> str:
    place = localized_name(team.get("placeName"))
    common = localized_name(team.get("commonName"))
    if place and common:
        return f"{place} {common}".strip()
    return common or place or str(team.get("abbrev") or "").strip()


def load_schedule(start_date: str) -> dict:
    return fetch_json(f"schedule/{quote(start_date)}")


def load_score(game_date: str) -> dict:
    return fetch_json(f"score/{quote(game_date)}")


def load_standings(game_date: str) -> dict:
    return fetch_json(f"standings/{quote(game_date)}")


def load_club_schedule(team_abbrev: str, season: int | str) -> dict:
    return fetch_json(f"club-schedule-season/{quote(team_abbrev)}/{season}")


def parse_game(game: dict) -> dict:
    away = game.get("awayTeam") or {}
    home = game.get("homeTeam") or {}
    return {
        "GAME_ID": game.get("id"),
        "SEASON": game.get("season"),
        "SEASON_LABEL": season_label(game.get("season", "")),
        "GAME_TYPE": game.get("gameType"),
        "GAME_DATE": game.get("gameDate") or str(game.get("startTimeUTC", ""))[:10],
        "GAME_DATETIME": game.get("startTimeUTC"),
        "STATUS": game.get("gameState") or game.get("gameScheduleState") or "",
        "HOME_TEAM_ID": home.get("id"),
        "HOME_TEAM_ABBREV": home.get("abbrev"),
        "HOME_TEAM": team_full_name(home),
        "AWAY_TEAM_ID": away.get("id"),
        "AWAY_TEAM_ABBREV": away.get("abbrev"),
        "AWAY_TEAM": team_full_name(away),
        "HOME_SCORE": home.get("score"),
        "AWAY_SCORE": away.get("score"),
        "VENUE": localized_name((game.get("venue") or {})),
        "LAST_PERIOD_TYPE": (game.get("gameOutcome") or {}).get("lastPeriodType"),
        "SERIES_STATUS": localized_name(game.get("seriesStatus") or {}),
    }


def parse_score_game(game: dict) -> dict:
    row = parse_game(game)
    away_odds, away_provider = extract_moneyline_odds(game.get("awayTeam") or {})
    home_odds, home_provider = extract_moneyline_odds(game.get("homeTeam") or {})
    row["CLOCK"] = (game.get("clock") or {}).get("timeRemaining") or ""
    row["PERIOD"] = (game.get("periodDescriptor") or {}).get("number")
    row["AWAY_ODDS"] = away_odds
    row["HOME_ODDS"] = home_odds
    row["ODDS_PROVIDER"] = home_provider or away_provider
    row["ODDS_PARTNER_IDS"] = ",".join(
        sorted(
            {
                str(odd.get("providerId"))
                for team_key in ["awayTeam", "homeTeam"]
                for odd in ((game.get(team_key) or {}).get("odds") or [])
                if odd.get("providerId") is not None
            }
        )
    )
    return row


def extract_moneyline_odds(team: dict) -> tuple[object, str]:
    """Return American moneyline odds from NHL embedded odds rows."""
    odds = team.get("odds") or []
    preferred_provider_ids = [9, 7]
    for provider_id in preferred_provider_ids:
        for row in odds:
            if row.get("providerId") == provider_id:
                value = row.get("value")
                if isinstance(value, str) and value.startswith(("+", "-")):
                    provider = "DraftKings" if provider_id == 9 else "FanDuel"
                    return value, provider

    for row in odds:
        value = row.get("value")
        if isinstance(value, str) and value.startswith(("+", "-")):
            return value, f"Provider {row.get('providerId')}"

    return None, ""


def iter_schedule_games(payload: dict) -> list[dict]:
    rows = []
    for day in payload.get("gameWeek", []):
        for game in day.get("games", []):
            rows.append(parse_game(game))
    return rows


def iter_score_games(payload: dict) -> list[dict]:
    return [parse_score_game(game) for game in payload.get("games", [])]
