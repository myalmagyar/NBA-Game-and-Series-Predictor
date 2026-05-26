# src/mlb_api.py

from __future__ import annotations

from datetime import date, datetime
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_API_V11_BASE = "https://statsapi.mlb.com/api/v1.1"
OPEN_METEO_API_BASE = "https://api.open-meteo.com/v1/forecast"
ODDS_API_BASE = "https://api.theoddsapi.com/odds/"
SHARPSPORTS_API_BASE = "https://api.sharpsports.io/v1"
MLB_SPORT_ID = 1
DEFAULT_TIMEOUT_SECONDS = 20


def fetch_url_json(
    url: str,
    params: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    headers: dict | None = None,
) -> dict:
    """Fetch JSON from one public API URL."""
    params = params or {}

    if params:
        url = f"{url}?{urlencode(params)}"

    request_headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    request_headers.update(headers or {})
    request = Request(
        url,
        headers=request_headers,
    )

    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_url_json(
    url: str,
    payload: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    headers: dict | None = None,
) -> dict:
    """Post JSON to one API URL and return JSON."""
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    request_headers.update(headers or {})
    request = Request(
        url,
        data=json.dumps(payload or {}).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_json(endpoint: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Fetch JSON from the public MLB Stats API v1."""
    url = f"{MLB_API_BASE}/{endpoint.lstrip('/')}"
    return fetch_url_json(url, params=params, timeout=timeout)


def fetch_live_game_feed(game_pk: str | int, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Fetch one MLB game live feed, including box score, lineups, and play-by-play."""
    game_pk_text = str(game_pk or "").strip()

    if not game_pk_text:
        return {}

    return fetch_url_json(f"{MLB_API_V11_BASE}/game/{game_pk_text}/feed/live", timeout=timeout)


def fetch_player_stats(
    person_id: str | int,
    season: int,
    group: str = "pitching",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Fetch one player's season stats from the public MLB Stats API."""
    person_id_text = str(person_id or "").strip()

    if not person_id_text:
        return {}

    return fetch_json(
        f"people/{person_id_text}/stats",
        {
            "stats": "season",
            "group": group,
            "season": int(season),
        },
        timeout=timeout,
    )


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout: int = 10,
) -> dict:
    """Fetch hourly ballpark weather from Open-Meteo."""
    return fetch_url_json(
        OPEN_METEO_API_BASE,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "precipitation_probability",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "weather_code",
                ]
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
        },
        timeout=timeout,
    )


def fetch_schedule(
    start_date: str | date,
    end_date: str | date | None = None,
    hydrate: str = "team,probablePitcher,linescore",
) -> dict:
    """Fetch MLB schedule data for a date or date range."""
    if isinstance(start_date, date):
        start_date = start_date.strftime("%Y-%m-%d")

    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, date):
        end_date = end_date.strftime("%Y-%m-%d")

    return fetch_json(
        "schedule",
        {
            "sportId": MLB_SPORT_ID,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": hydrate,
        },
    )


def fetch_standings(
    season: int,
    league_ids: tuple[int, ...] = (103, 104),
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Fetch MLB regular-season standings for the selected leagues."""
    return fetch_json(
        "standings",
        {
            "sportId": MLB_SPORT_ID,
            "season": int(season),
            "leagueId": ",".join(str(league_id) for league_id in league_ids),
            "standingsTypes": "regularSeason",
            "hydrate": "team",
        },
        timeout=timeout,
    )


def fetch_mlb_odds(
    api_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict]:
    """Fetch MLB odds from The Odds API when the user provides an API key."""
    api_key_text = str(api_key or "").strip()

    if not api_key_text:
        return []

    payload = fetch_url_json(
        ODDS_API_BASE,
        {
            "sport_key": "baseball_mlb",
        },
        headers={"x-api-key": api_key_text},
        timeout=timeout,
    )

    if isinstance(payload, list):
        return payload

    events = payload.get("data", []) if isinstance(payload, dict) else []
    normalized_events = []

    for event in events or []:
        if not isinstance(event, dict):
            continue

        bookmakers_by_key = {}

        for book in event.get("books", []) or []:
            if not isinstance(book, dict):
                continue

            book_key = str(book.get("book", "") or "").strip()

            if not book_key:
                continue

            bookmaker = bookmakers_by_key.setdefault(
                book_key,
                {
                    "key": book_key,
                    "title": book_key.replace("_", " ").title(),
                    "markets": [],
                },
            )
            market_key = str(book.get("market", "") or "h2h")
            bookmaker["markets"].append(
                {
                    "key": market_key,
                    "last_update": book.get("updated_at", ""),
                    "outcomes": book.get("outcomes", []) or [],
                }
            )

        normalized_events.append(
            {
                "id": event.get("event_id"),
                "sport_key": event.get("sport"),
                "sport_title": event.get("league"),
                "commence_time": event.get("start_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "bookmakers": list(bookmakers_by_key.values()),
            }
        )

    return normalized_events


def create_sharpsports_context(
    api_key: str,
    payload: dict,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Create a SharpSports hosted linking or betPlace context."""
    api_key_text = str(api_key or "").strip()

    if not api_key_text:
        return {}

    return post_url_json(
        f"{SHARPSPORTS_API_BASE}/context",
        payload=payload,
        headers={"Authorization": f"Token {api_key_text}"},
        timeout=timeout,
    )


def fetch_sharpsports_bet_slips(
    api_key: str,
    params: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Fetch synced bet slips from SharpSports when credentials are configured."""
    api_key_text = str(api_key or "").strip()

    if not api_key_text:
        return {}

    return fetch_url_json(
        f"{SHARPSPORTS_API_BASE}/betSlips",
        params=params or {},
        headers={"Authorization": f"Token {api_key_text}"},
        timeout=timeout,
    )


def iter_schedule_games(schedule_payload: dict) -> list[dict]:
    """Return flattened games from an MLB schedule payload."""
    games = []

    for date_row in schedule_payload.get("dates", []) or []:
        games.extend(date_row.get("games", []) or [])

    return games


def parse_game_datetime(value: object) -> datetime | None:
    """Parse MLB UTC game timestamp."""
    text = str(value or "").strip()

    if not text:
        return None

    try:
        if text.endswith("Z"):
            text = text.replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def get_team_payload(game: dict, side: str) -> dict:
    """Return home or away team payload."""
    return ((game.get("teams", {}) or {}).get(side, {}) or {})


def get_team_name(game: dict, side: str) -> str:
    """Return home or away team name."""
    team = get_team_payload(game, side).get("team", {}) or {}
    return str(team.get("name", "")).strip()


def get_team_id(game: dict, side: str) -> int | None:
    """Return home or away MLB team id."""
    team = get_team_payload(game, side).get("team", {}) or {}

    try:
        return int(team.get("id"))
    except (TypeError, ValueError):
        return None


def get_team_score(game: dict, side: str) -> int | None:
    """Return home or away score when present."""
    payload = get_team_payload(game, side)

    try:
        return int(payload.get("score"))
    except (TypeError, ValueError):
        return None


def get_probable_pitcher(game: dict, side: str) -> str:
    """Return probable pitcher name when the schedule feed provides one."""
    pitcher = get_team_payload(game, side).get("probablePitcher", {}) or {}
    return str(pitcher.get("fullName", "")).strip()


def get_probable_pitcher_id(game: dict, side: str) -> int | None:
    """Return probable pitcher MLB id when the schedule feed provides one."""
    pitcher = get_team_payload(game, side).get("probablePitcher", {}) or {}

    try:
        return int(pitcher.get("id"))
    except (TypeError, ValueError):
        return None


def get_status_text(game: dict) -> str:
    """Return the most readable MLB game status."""
    status = game.get("status", {}) or {}
    return (
        str(status.get("detailedState", "")).strip()
        or str(status.get("abstractGameState", "")).strip()
        or str(status.get("statusCode", "")).strip()
    )


def get_status_code(game: dict) -> str:
    """Return MLB status code when present."""
    status = game.get("status", {}) or {}
    return str(status.get("statusCode", "")).strip()


def is_final_game(game: dict) -> bool:
    """Return whether an MLB schedule game is final."""
    status = game.get("status", {}) or {}
    abstract_state = str(status.get("abstractGameState", "")).lower()
    detailed_state = str(status.get("detailedState", "")).lower()
    status_code = str(status.get("statusCode", "")).upper()
    return (
        abstract_state == "final"
        or detailed_state in {"final", "game over"}
        or status_code in {"F", "O"}
    )


def is_live_game(game: dict) -> bool:
    """Return whether an MLB schedule game is in progress."""
    status = game.get("status", {}) or {}
    abstract_state = str(status.get("abstractGameState", "")).lower()
    status_code = str(status.get("statusCode", "")).upper()
    return abstract_state == "live" or status_code in {"I", "M", "N"}


def normalize_schedule_game(game: dict) -> dict:
    """Normalize one MLB schedule game into the app/data schema."""
    game_datetime = parse_game_datetime(game.get("gameDate"))
    home_score = get_team_score(game, "home")
    away_score = get_team_score(game, "away")

    return {
        "GAME_PK": str(game.get("gamePk", "")).strip(),
        "GAME_DATE": str(game.get("officialDate", "")).strip(),
        "GAME_DATETIME": game_datetime.isoformat() if game_datetime else "",
        "GAME_TYPE": str(game.get("gameType", "")).strip(),
        "STATUS": get_status_text(game),
        "STATUS_CODE": get_status_code(game),
        "IS_FINAL": is_final_game(game),
        "IS_LIVE": is_live_game(game),
        "HOME_TEAM_ID": get_team_id(game, "home"),
        "HOME_TEAM": get_team_name(game, "home"),
        "AWAY_TEAM_ID": get_team_id(game, "away"),
        "AWAY_TEAM": get_team_name(game, "away"),
        "HOME_SCORE": home_score,
        "AWAY_SCORE": away_score,
        "HOME_PROBABLE_PITCHER": get_probable_pitcher(game, "home"),
        "AWAY_PROBABLE_PITCHER": get_probable_pitcher(game, "away"),
        "HOME_PROBABLE_PITCHER_ID": get_probable_pitcher_id(game, "home"),
        "AWAY_PROBABLE_PITCHER_ID": get_probable_pitcher_id(game, "away"),
        "VENUE": str((game.get("venue", {}) or {}).get("name", "")).strip(),
        "DOUBLEHEADER": str(game.get("doubleHeader", "")).strip(),
        "GAME_NUMBER": game.get("gameNumber", ""),
    }
