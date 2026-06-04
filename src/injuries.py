# src/injuries.py

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import argparse
import re
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from pypdf import PdfReader


DATA_DIR = Path("data")
TEAM_STRENGTH_PATH = DATA_DIR / "current_team_strength.csv"
CURRENT_INJURIES_PATH = DATA_DIR / "current_injuries.csv"
HISTORICAL_INJURIES_PATH = DATA_DIR / "historical_injuries.csv"

OFFICIAL_INJURY_REPORT_PAGE = (
    "https://official.nba.com/nba-injury-report-2025-26-season/"
)
HISTORICAL_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

STATUS_WEIGHTS = {
    "Out": 1.00,
    "Doubtful": 0.75,
    "Questionable": 0.50,
    "Probable": 0.10,
    "Available": 0.00,
}

IGNORE_MARKERS = [
    ("NOT", "YET", "SUBMITTED"),
    ("NO", "INJURIES", "REPORTED"),
]

HEADER_TOKENS = {
    "Injury",
    "Report:",
    "Page",
    "of",
    "Game",
    "Date",
    "Time",
    "Matchup",
    "Team",
    "Player",
    "Name",
    "Current",
    "Status",
    "Reason",
}

INJURY_COLUMNS = [
    "REPORT_TIMESTAMP",
    "SOURCE_URL",
    "GAME_DATE",
    "GAME_TIME",
    "MATCHUP",
    "TEAM",
    "PLAYER_NAME_REPORT",
    "PLAYER_NAME",
    "CURRENT_STATUS",
    "STATUS_WEIGHT",
    "REASON",
]


def fetch_url(url: str) -> bytes:
    """Fetch bytes from a URL with a browser-like user agent."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def parse_report_datetime_from_url(url: str) -> datetime:
    """Parse the report timestamp embedded in an official NBA PDF URL."""
    match = re.search(
        r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})(?:_(\d{2}))?([AP]M)\.pdf",
        url,
    )

    if not match:
        raise ValueError(f"Could not parse injury report timestamp from URL: {url}")

    date_part, hour_part, minute_part, meridiem = match.groups()
    minute_part = minute_part or "00"
    timestamp_text = f"{date_part} {hour_part}:{minute_part} {meridiem}"
    return datetime.strptime(timestamp_text, "%Y-%m-%d %I:%M %p")


def get_report_urls(page_url: str) -> list[str]:
    """Return official NBA injury-report PDF URLs linked from a season page."""
    html = fetch_url(page_url).decode("utf-8", errors="replace")
    links = re.findall(r'href=["\']([^"\']*Injury-Report_[^"\']+\.pdf)["\']', html)

    if not links:
        raise RuntimeError(f"No official injury-report PDF links found at {page_url}")

    return sorted(
        {urljoin(page_url, link) for link in links},
        key=parse_report_datetime_from_url,
    )


def get_latest_report_url(page_url: str = OFFICIAL_INJURY_REPORT_PAGE) -> str:
    """Return the latest linked official NBA injury-report PDF URL."""
    return max(get_report_urls(page_url), key=parse_report_datetime_from_url)


def get_season_report_page(season: str) -> str:
    """Return the official NBA injury-report page URL for a season."""
    return f"https://official.nba.com/nba-injury-report-{season}-season/"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from an official NBA injury-report PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_team_names() -> list[str]:
    """Load current NBA team names from the team strength table."""
    if not TEAM_STRENGTH_PATH.exists():
        raise FileNotFoundError(
            "Missing data/current_team_strength.csv. Run: python src/team_strength.py"
        )

    strength = pd.read_csv(TEAM_STRENGTH_PATH)
    return sorted(strength["TEAM_NAME"].dropna().unique(), key=len, reverse=True)


def is_date_token(token: str) -> bool:
    """Return whether a token is an injury-report game date."""
    return bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", token))


def is_time_at(lines: list[str], index: int) -> bool:
    """Return whether tokens at index represent a game time."""
    return (
        index + 1 < len(lines)
        and bool(re.fullmatch(r"\d{1,2}:\d{2}", lines[index]))
        and lines[index + 1] == "(ET)"
    )


def is_matchup_token(token: str) -> bool:
    """Return whether a token is an NBA matchup abbreviation."""
    return bool(re.fullmatch(r"[A-Z]{2,3}@[A-Z]{2,3}", token))


def split_team_name(team_name: str) -> tuple[str, ...]:
    """Split a team name into report tokens."""
    return tuple(team_name.split())


def match_team_at(
    lines: list[str],
    index: int,
    team_token_map: dict[tuple[str, ...], str],
) -> tuple[str | None, int]:
    """Match a team name starting at the current token."""
    for tokens, team_name in team_token_map.items():
        end_index = index + len(tokens)

        if tuple(lines[index:end_index]) == tokens:
            return team_name, len(tokens)

    return None, 0


def has_ignore_marker_at(lines: list[str], index: int) -> bool:
    """Return whether report tokens at index are a non-player team note."""
    for marker in IGNORE_MARKERS:
        if tuple(lines[index : index + len(marker)]) == marker:
            return True

    return False


def skip_ignore_marker(lines: list[str], index: int) -> int:
    """Move past a non-player team note."""
    for marker in IGNORE_MARKERS:
        if tuple(lines[index : index + len(marker)]) == marker:
            return index + len(marker)

    return index


def is_report_boundary(
    lines: list[str],
    index: int,
    team_token_map: dict[tuple[str, ...], str],
) -> bool:
    """Return whether a token begins a new report context."""
    if index >= len(lines):
        return True

    token = lines[index]

    if is_date_token(token) or is_time_at(lines, index) or is_matchup_token(token):
        return True

    if has_ignore_marker_at(lines, index):
        return True

    team_name, _ = match_team_at(lines, index, team_token_map)
    return team_name is not None


def is_potential_player_start(
    lines: list[str],
    index: int,
    team_token_map: dict[tuple[str, ...], str],
) -> bool:
    """Return whether tokens at index look like the next player row."""
    if "," not in lines[index]:
        return False

    search_end = min(index + 8, len(lines))

    for status_index in range(index + 1, search_end):
        if is_report_boundary(lines, status_index, team_token_map):
            return False

        if lines[status_index] in STATUS_WEIGHTS:
            return True

    return False


def find_status_index(lines: list[str], index: int) -> int | None:
    """Find the status token for a player row."""
    search_end = min(index + 8, len(lines))

    for status_index in range(index + 1, search_end):
        if lines[status_index] in STATUS_WEIGHTS:
            return status_index

    return None


def normalize_report_player_name(report_name: str) -> str:
    """Convert NBA report names from 'Last, First' to 'First Last'."""
    if "," not in report_name:
        return report_name

    last_name, first_names = [part.strip() for part in report_name.split(",", 1)]
    return f"{first_names} {last_name}".strip()


def clean_reason(tokens: list[str]) -> str:
    """Join reason tokens into readable report text."""
    reason = " ".join(tokens)
    reason = reason.replace(" - ", " - ")
    reason = re.sub(r"\s+;", ";", reason)
    reason = re.sub(r"\s+", " ", reason)
    return reason.strip()


def parse_injury_report_text(
    text: str,
    source_url: str,
    report_timestamp: datetime,
    team_names: list[str] | None = None,
) -> pd.DataFrame:
    """Parse extracted official NBA injury-report text into a table."""
    if team_names is None:
        team_names = load_team_names()

    team_token_map = {split_team_name(team_name): team_name for team_name in team_names}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    rows = []
    current_date = None
    current_time = None
    current_matchup = None
    current_team = None

    index = 0

    while index < len(lines):
        token = lines[index]

        if token in HEADER_TOKENS or token.isdigit():
            index += 1
            continue

        if is_date_token(token):
            current_date = token
            index += 1
            continue

        if is_time_at(lines, index):
            current_time = f"{lines[index]} {lines[index + 1]}"
            index += 2
            continue

        if is_matchup_token(token):
            current_matchup = token
            index += 1
            continue

        team_name, consumed = match_team_at(lines, index, team_token_map)

        if team_name:
            current_team = team_name
            index += consumed
            continue

        if has_ignore_marker_at(lines, index):
            index = skip_ignore_marker(lines, index)
            continue

        status_index = find_status_index(lines, index)

        if (
            current_date
            and current_time
            and current_matchup
            and current_team
            and status_index is not None
        ):
            report_name = " ".join(lines[index:status_index])
            status = lines[status_index]
            reason_start = status_index + 1
            reason_end = reason_start

            while reason_end < len(lines):
                if is_report_boundary(lines, reason_end, team_token_map):
                    break

                if is_potential_player_start(lines, reason_end, team_token_map):
                    break

                reason_end += 1

            rows.append(
                {
                    "REPORT_TIMESTAMP": report_timestamp.isoformat(timespec="minutes"),
                    "SOURCE_URL": source_url,
                    "GAME_DATE": current_date,
                    "GAME_TIME": current_time,
                    "MATCHUP": current_matchup,
                    "TEAM": current_team,
                    "PLAYER_NAME_REPORT": report_name,
                    "PLAYER_NAME": normalize_report_player_name(report_name),
                    "CURRENT_STATUS": status,
                    "STATUS_WEIGHT": STATUS_WEIGHTS[status],
                    "REASON": clean_reason(lines[reason_start:reason_end]),
                }
            )
            index = reason_end
            continue

        index += 1

    return pd.DataFrame(rows, columns=INJURY_COLUMNS)


def fetch_latest_injury_report() -> pd.DataFrame:
    """Fetch and parse the latest official NBA injury report."""
    report_url = get_latest_report_url()
    report_timestamp = parse_report_datetime_from_url(report_url)
    pdf_bytes = fetch_url(report_url)
    text = extract_pdf_text(pdf_bytes)
    return parse_injury_report_text(
        text=text,
        source_url=report_url,
        report_timestamp=report_timestamp,
    )


def fetch_injury_report_url(
    report_url: str,
    team_names: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch and parse one official NBA injury-report PDF URL."""
    report_timestamp = parse_report_datetime_from_url(report_url)
    pdf_bytes = fetch_url(report_url)
    text = extract_pdf_text(pdf_bytes)
    return parse_injury_report_text(
        text=text,
        source_url=report_url,
        report_timestamp=report_timestamp,
        team_names=team_names,
    )


def build_current_injuries_file() -> pd.DataFrame:
    """Build the current injuries CSV used by the Streamlit app."""
    DATA_DIR.mkdir(exist_ok=True)
    injuries = fetch_latest_injury_report()
    injuries = injuries.reindex(columns=INJURY_COLUMNS)
    injuries.to_csv(CURRENT_INJURIES_PATH, index=False)

    print(f"Saved {len(injuries)} injury rows to {CURRENT_INJURIES_PATH}")

    if not injuries.empty:
        print(f"Latest report: {injuries.iloc[0]['REPORT_TIMESTAMP']}")
        print(f"Source: {injuries.iloc[0]['SOURCE_URL']}")
        print()
        print(injuries.head(25).to_string(index=False))

    return injuries


def collect_historical_injury_reports(
    seasons: list[str] | None = None,
    max_reports_per_season: int | None = None,
) -> pd.DataFrame:
    """Collect official injury reports for multiple seasons into one CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    seasons = seasons or HISTORICAL_SEASONS
    team_names = load_team_names()
    frames = []

    for season in seasons:
        page_url = get_season_report_page(season)
        print(f"Finding official injury reports for {season}...")

        try:
            report_urls = get_report_urls(page_url)
        except Exception as error:
            print(f"Skipping {season}: {error}")
            continue

        if max_reports_per_season is not None:
            report_urls = report_urls[-max_reports_per_season:]

        for index, report_url in enumerate(report_urls, start=1):
            print(f"Parsing {season} report {index:,}/{len(report_urls):,}")

            try:
                report = fetch_injury_report_url(report_url, team_names=team_names)
            except Exception as error:
                print(f"Skipping report {report_url}: {error}")
                continue

            if not report.empty:
                frames.append(report)

    if not frames:
        raise RuntimeError("No historical injury report rows were collected.")

    injuries = pd.concat(frames, ignore_index=True)
    injuries["REPORT_TIMESTAMP_SORT"] = pd.to_datetime(injuries["REPORT_TIMESTAMP"])
    injuries = injuries.sort_values("REPORT_TIMESTAMP_SORT")
    injuries = injuries.drop_duplicates(
        ["GAME_DATE", "MATCHUP", "TEAM", "PLAYER_NAME"],
        keep="last",
    )
    injuries = injuries.drop(columns=["REPORT_TIMESTAMP_SORT"])
    injuries.to_csv(HISTORICAL_INJURIES_PATH, index=False)

    print(f"Saved {len(injuries)} historical injury rows to {HISTORICAL_INJURIES_PATH}")
    return injuries


def main() -> None:
    """Run injury report collection from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Collect historical official injury reports instead of only the latest report.",
    )
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Season to collect, such as 2024-25. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-reports-per-season",
        type=int,
        default=None,
        help="Limit each season to the latest N reports while testing the parser.",
    )
    args = parser.parse_args()

    if args.historical:
        collect_historical_injury_reports(
            seasons=args.seasons,
            max_reports_per_season=args.max_reports_per_season,
        )
    else:
        build_current_injuries_file()


if __name__ == "__main__":
    main()
