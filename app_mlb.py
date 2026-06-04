# app_mlb.py

from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo
import html
import json
import math
import os
import random

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.mlb_api import (
    create_sharpsports_context,
    fetch_live_game_feed,
    fetch_mlb_odds,
    fetch_player_stats,
    fetch_schedule,
    fetch_sharpsports_bet_slips,
    fetch_standings,
    fetch_weather_forecast,
    iter_schedule_games,
    normalize_schedule_game,
)
from src.mlb_features import DEFAULT_RUN_ENVIRONMENT, HOME_ELO_ADVANTAGE, expected_score, get_park_run_factor


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TEAM_STRENGTH_PATH = DATA_DIR / "mlb_current_team_strength.csv"
PITCHER_STRENGTH_PATH = DATA_DIR / "mlb_current_pitcher_strength.csv"
FEATURES_PATH = DATA_DIR / "mlb_model_features.csv"
RAW_GAMES_PATH = DATA_DIR / "mlb_raw_games.csv"
MODEL_PATH = MODELS_DIR / "mlb_game_winner_model.joblib"
METRICS_PATH = DATA_DIR / "mlb_model_metrics.csv"
BET_TRACKER_PATH = DATA_DIR / "mlb_bet_tracker.csv"

MLB_TEAM_ABBREVIATIONS_BY_ID = {
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "ATH",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}

MLB_TEAM_ABBREVIATIONS_BY_NAME = {
    "arizona diamondbacks": "ARI",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "oakland athletics": "ATH",
    "athletics": "ATH",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st. louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}

MLB_LEAGUE_NAMES_BY_ID = {
    103: "American League",
    104: "National League",
}

MLB_DIVISION_NAMES_BY_ID = {
    200: "AL West",
    201: "AL East",
    202: "AL Central",
    203: "NL West",
    204: "NL East",
    205: "NL Central",
}

MLB_DIVISION_ORDER_BY_LEAGUE = {
    "American League": ["AL East", "AL Central", "AL West"],
    "National League": ["NL East", "NL Central", "NL West"],
}

MODEL_WEIGHT = 0.80
ELO_WEIGHT = 0.20
MIN_GAME_PROBABILITY = 0.10
MAX_GAME_PROBABILITY = 0.90
LIVE_REFRESH_SECONDS = 3
LIVE_FEED_CACHE_SECONDS = 3
SCHEDULE_CACHE_SECONDS = 10
STANDINGS_CACHE_SECONDS = 60
ODDS_CACHE_SECONDS = 30
ODDS_REFRESH_SECONDS = 30
WEATHER_CACHE_SECONDS = 900
GAME_DETAIL_STATE_KEY = "mlb_selected_game_detail"
WIN_PROB_DIALOG_STATE_KEY = "mlb_selected_win_probability_game"
WIN_PROB_QUERY_PARAM = "mlb_wp_game"
BET_SLIP_STATE_KEY = "mlb_bet_slip"
PREGAME_OVERDUE_WINDOW_HOURS = 6
SCORE_SIMULATION_COUNT = 1000
DEFAULT_SERIES_SIMULATIONS = 2000
DEFAULT_BRACKET_SIMULATIONS = 1000
BET_TRACKER_COLUMNS = [
    "Bet ID",
    "Date Logged",
    "Game Date",
    "Matchup",
    "Market",
    "Selection",
    "Odds",
    "Stake",
    "Model Probability",
    "Market Probability",
    "Edge",
    "EV/Unit",
    "Kelly",
    "Result",
    "Profit",
    "Closing Odds",
    "CLV",
    "Notes",
]
SPORTSBOOK_LINKS = {
    "draftkings": {
        "label": "DraftKings",
        "abbr": "dk",
        "url": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
    },
    "fanduel": {
        "label": "FanDuel",
        "abbr": "fd",
        "url": "https://sportsbook.fanduel.com/navigation/mlb",
    },
    "betmgm": {
        "label": "BetMGM",
        "abbr": "mg",
        "url": "https://sports.betmgm.com/en/sports/baseball-23",
    },
    "caesars": {
        "label": "Caesars",
        "abbr": "ca",
        "url": "https://www.caesars.com/sportsbook-and-casino/sport/baseball",
    },
}
SPORTSBOOK_KEY_ALIASES = {
    "draftkings": "draftkings",
    "draft kings": "draftkings",
    "dk": "draftkings",
    "fanduel": "fanduel",
    "fan duel": "fanduel",
    "fd": "fanduel",
    "betmgm": "betmgm",
    "bet mgm": "betmgm",
    "mgm": "betmgm",
    "mg": "betmgm",
    "caesars": "caesars",
    "caesars sportsbook": "caesars",
    "william hill": "caesars",
    "ca": "caesars",
    "betrivers": "betrivers",
    "bet rivers": "betrivers",
    "br": "betrivers",
    "fanatics": "fanatics",
    "fanatics sportsbook": "fanatics",
    "fb": "fanatics",
    "espnbet": "espnbet",
    "espn bet": "espnbet",
    "pe": "espnbet",
}
PLAYER_STATS_KEYS = [
    "AVG",
    "OPS",
    "HR",
    "RBI",
    "SB",
    "ERA",
    "WHIP",
    "SO",
    "W",
    "L",
    "GS",
    "IP",
]
TEAM_COLOR_PALETTE = [
    "#1d4ed8",
    "#b91c1c",
    "#166534",
    "#7c3aed",
    "#be123c",
    "#0f766e",
    "#c2410c",
    "#4338ca",
]

BALLPARK_WEATHER_COORDS = {
    108: {"name": "Angel Stadium", "latitude": 33.8003, "longitude": -117.8827},
    109: {"name": "Chase Field", "latitude": 33.4455, "longitude": -112.0667},
    110: {"name": "Oriole Park at Camden Yards", "latitude": 39.2840, "longitude": -76.6217},
    111: {"name": "Fenway Park", "latitude": 42.3467, "longitude": -71.0972},
    112: {"name": "Wrigley Field", "latitude": 41.9484, "longitude": -87.6553},
    113: {"name": "Great American Ball Park", "latitude": 39.0974, "longitude": -84.5066},
    114: {"name": "Progressive Field", "latitude": 41.4962, "longitude": -81.6852},
    115: {"name": "Coors Field", "latitude": 39.7562, "longitude": -104.9942},
    116: {"name": "Comerica Park", "latitude": 42.3390, "longitude": -83.0485},
    117: {"name": "Daikin Park", "latitude": 29.7573, "longitude": -95.3555},
    118: {"name": "Kauffman Stadium", "latitude": 39.0517, "longitude": -94.4803},
    119: {"name": "Dodger Stadium", "latitude": 34.0739, "longitude": -118.2400},
    120: {"name": "Nationals Park", "latitude": 38.8730, "longitude": -77.0074},
    121: {"name": "Citi Field", "latitude": 40.7571, "longitude": -73.8458},
    133: {"name": "Sutter Health Park", "latitude": 38.5802, "longitude": -121.5133},
    134: {"name": "PNC Park", "latitude": 40.4469, "longitude": -80.0057},
    135: {"name": "Petco Park", "latitude": 32.7073, "longitude": -117.1566},
    136: {"name": "T-Mobile Park", "latitude": 47.5914, "longitude": -122.3325},
    137: {"name": "Oracle Park", "latitude": 37.7786, "longitude": -122.3893},
    138: {"name": "Busch Stadium", "latitude": 38.6226, "longitude": -90.1928},
    139: {"name": "Tropicana Field", "latitude": 27.7682, "longitude": -82.6534},
    140: {"name": "Globe Life Field", "latitude": 32.7473, "longitude": -97.0842},
    141: {"name": "Rogers Centre", "latitude": 43.6414, "longitude": -79.3894},
    142: {"name": "Target Field", "latitude": 44.9817, "longitude": -93.2776},
    143: {"name": "Citizens Bank Park", "latitude": 39.9057, "longitude": -75.1665},
    144: {"name": "Truist Park", "latitude": 33.8907, "longitude": -84.4677},
    145: {"name": "Rate Field", "latitude": 41.8300, "longitude": -87.6338},
    146: {"name": "loanDepot park", "latitude": 25.7781, "longitude": -80.2197},
    147: {"name": "Yankee Stadium", "latitude": 40.8296, "longitude": -73.9262},
    158: {"name": "American Family Field", "latitude": 43.0280, "longitude": -87.9712},
}

WEATHER_CODE_LABELS = {
    0: "Clear",
    1: "Mostly clear",
    2: "Partly cloudy",
    3: "Cloudy",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy showers",
    95: "Thunderstorms",
    96: "Thunderstorms",
    99: "Thunderstorms",
}

FEATURE_COLUMN_ALIASES = {
    "SEASON_RUN_PER_GAME": "SEASON_AVG_RUNS_FOR",
    "SEASON_OPP_RUN_PER_GAME": "SEASON_AVG_RUNS_AGAINST",
    "ROLLING_RUN_PER_GAME_10": "ROLLING_RUNS_FOR_10",
    "ROLLING_OPP_RUN_PER_GAME_10": "ROLLING_RUNS_AGAINST_10",
    "ROLLING_RUN_PER_GAME_20": "ROLLING_RUNS_FOR_20",
    "ROLLING_OPP_RUN_PER_GAME_20": "ROLLING_RUNS_AGAINST_20",
}


APP_CSS = """
<style>
    :root {
        --page: #f6f8fb;
        --surface: #ffffff;
        --ink: #182033;
        --muted: #667085;
        --line: #d9e2ec;
        --accent: #166534;
        --accent-blue: #1d4ed8;
        --accent-red: #b91c1c;
        --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #fbfdff 0%, var(--page) 340px, var(--page) 100%);
        color: var(--ink);
    }

    .sidebar-brand {
        align-items: center;
        display: flex;
        gap: 0.65rem;
        margin: 0.2rem 0 1rem;
    }

    .sidebar-brand-mark {
        align-items: center;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        color: #166534;
        display: flex;
        font-size: 1.05rem;
        font-weight: 950;
        height: 2.3rem;
        justify-content: center;
        width: 2.3rem;
    }

    .sidebar-brand-name {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 950;
        line-height: 1.1;
    }

    .sidebar-brand-sub, .sidebar-nav-title {
        color: var(--muted);
        font-size: 0.54rem;
        font-weight: 850;
        line-height: 1.2;
        text-transform: uppercase;
    }

    .sidebar-nav-title {
        margin: 0.75rem 0 0.35rem;
    }

    div[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="collapsedControl"] {
        display: none;
    }

    .sidebar-toggle-wrap {
        margin: 0.35rem 0 0.75rem;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.35rem !important;
    }

    .mlb-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        display: flex;
        flex-direction: column;
        margin-bottom: 0.85rem;
        min-height: 585px;
        padding: 0.9rem;
    }

    .mlb-topline {
        align-items: flex-start;
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
        min-height: 7rem;
    }

    .mlb-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .mlb-status {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        color: #166534;
        font-size: 0.72rem;
        font-weight: 900;
        padding: 0.22rem 0.55rem;
        white-space: nowrap;
    }

    .mlb-countdown {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        color: #1d4ed8;
        display: inline-flex;
        font-size: 0.72rem;
        font-weight: 950;
        margin-top: 0.35rem;
        padding: 0.22rem 0.55rem;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .mlb-matchup {
        align-items: start;
        display: grid;
        gap: 0.6rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin-bottom: 0.75rem;
        min-height: 7.05rem;
    }

    .mlb-team {
        min-width: 0;
    }

    .mlb-team-heading {
        align-items: center;
        display: flex;
        gap: 0.55rem;
        min-width: 0;
    }

    .mlb-team-heading.home {
        justify-content: flex-start;
    }

    .mlb-logo-frame {
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        box-sizing: border-box;
        display: flex;
        flex: 0 0 auto;
        height: 54px;
        justify-content: center;
        width: 54px;
    }

    .mlb-logo {
        display: block;
        height: 34px;
        object-fit: contain;
        width: 34px;
    }

    .mlb-team-name {
        color: var(--ink);
        display: -webkit-box;
        font-size: 1.02rem;
        font-weight: 950;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.1;
        min-height: 2.25em;
        overflow: hidden;
        overflow-wrap: anywhere;
    }

    .mlb-team .mlb-meta {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        min-height: 2.1em;
        overflow: hidden;
    }

    .mlb-meta {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 750;
        margin-top: 0.18rem;
        overflow-wrap: anywhere;
    }

    .mlb-score {
        color: var(--accent-blue);
        font-size: 1.4rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 0.22rem;
        min-height: 1.4rem;
    }

    .mlb-vs {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 950;
    }

    .mlb-live-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 750;
        height: 3.25rem;
        line-height: 1.35;
        margin: 0 0 0.75rem;
        overflow: hidden;
        padding: 0.55rem 0.65rem;
    }

    .mlb-live-note span {
        color: var(--accent-red);
        font-size: 0.54rem;
        font-weight: 950;
        margin-right: 0.35rem;
        text-transform: uppercase;
    }

    .mlb-live-count {
        align-items: flex-start;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        color: #9a3412;
        display: flex;
        flex-direction: column;
        font-size: 0.72rem;
        font-weight: 950;
        gap: 0.22rem;
        margin-top: 0.35rem;
        min-height: 1.45rem;
        padding: 0.22rem 0.55rem;
        white-space: nowrap;
    }

    .mlb-live-count-line {
        align-items: center;
        display: inline-flex;
        gap: 0.35rem;
        line-height: 1;
    }

    .mlb-live-count span {
        color: #c2410c;
        font-size: 0.66rem;
        text-transform: uppercase;
    }

    .mlb-out-dots {
        align-items: center;
        display: flex;
        gap: 0.22rem;
        min-height: 0.48rem;
    }

    .mlb-out-dot {
        background: #ffffff;
        border: 1px solid #fb923c;
        border-radius: 999px;
        display: block;
        height: 0.42rem;
        width: 0.42rem;
    }

    .mlb-out-dot.active {
        background: #ea580c;
        border-color: #c2410c;
    }

    .game-center-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        display: grid;
        gap: 0.9rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin: 0.75rem 0;
        padding: 0.85rem;
    }

    .game-center-block {
        min-width: 0;
    }

    .game-center-label {
        color: var(--muted);
        font-size: 0.64rem;
        font-weight: 950;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .game-center-value {
        color: var(--ink);
        font-size: 0.88rem;
        font-weight: 900;
        line-height: 1.25;
        margin-top: 0.25rem;
        overflow-wrap: anywhere;
    }

    .game-center-diamond {
        align-items: center;
        display: flex;
        justify-content: center;
        min-width: 4.2rem;
    }

    .game-center-panel .base-diamond {
        height: 3.7rem;
        width: 3.7rem;
    }

    .game-center-panel .base-node.first {
        right: 0.2rem;
        top: 1.42rem;
    }

    .game-center-panel .base-node.second {
        left: 1.42rem;
        top: 0.18rem;
    }

    .game-center-panel .base-node.third {
        left: 0.2rem;
        top: 1.42rem;
    }

    .game-center-panel .base-node.home {
        bottom: 0.15rem;
        left: 1.42rem;
    }

    .game-today-row {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.45rem;
        padding: 0.55rem 0.65rem;
    }

    .game-today-stats {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 800;
        margin-top: 0.16rem;
        overflow-wrap: anywhere;
    }

    .standings-header {
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin: 0.5rem 0 0.9rem;
        padding: 0.8rem 0.9rem;
    }

    .standings-header-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.2;
    }

    .standings-header-note {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 750;
        margin-top: 0.18rem;
    }

    .standings-summary-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0.35rem 0 0.9rem;
    }

    .standings-summary-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--card-color, #1d4ed8);
        border-radius: 8px;
        min-width: 0;
        padding: 0.75rem;
    }

    .standings-card-label {
        color: var(--muted);
        font-size: 0.66rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .standings-card-team {
        align-items: center;
        color: var(--ink);
        display: flex;
        font-size: 0.9rem;
        font-weight: 950;
        gap: 0.45rem;
        line-height: 1.15;
        margin-top: 0.35rem;
        min-width: 0;
    }

    .standings-logo {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        flex: 0 0 auto;
        height: 26px;
        object-fit: contain;
        padding: 0.15rem;
        width: 26px;
    }

    .standings-card-meta {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .standings-table-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-top: 0.65rem;
        overflow-x: auto;
    }

    .standings-table {
        border-collapse: collapse;
        min-width: 760px;
        width: 100%;
    }

    .standings-table th,
    .standings-table td {
        border-bottom: 1px solid #e2e8f0;
        padding: 0.55rem 0.65rem;
        text-align: left;
        white-space: nowrap;
    }

    .standings-table th {
        background: #f8fafc;
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .standings-table td {
        color: var(--ink);
        font-size: 0.8rem;
        font-weight: 800;
    }

    .standings-table tr:last-child td {
        border-bottom: 0;
    }

    .standings-team-cell {
        align-items: center;
        display: flex;
        gap: 0.5rem;
        min-width: 13rem;
    }

    .standings-rank {
        align-items: center;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 999px;
        color: #3730a3;
        display: inline-flex;
        font-size: 0.72rem;
        font-weight: 950;
        height: 1.35rem;
        justify-content: center;
        min-width: 1.35rem;
        padding: 0 0.35rem;
    }

    .betting-panel {
        background: var(--surface);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin: 0.65rem 0 0.9rem;
        padding: 0.85rem;
    }

    .betting-layout {
        align-items: start;
        display: grid;
        gap: 1rem;
        grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
    }

    .betting-board-grid {
        display: grid;
        gap: 0.8rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .betting-filter-bar {
        align-items: center;
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        box-shadow: var(--shadow);
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0.5rem 0 0.85rem;
        padding: 0.75rem;
    }

    .betting-hero {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-left: 4px solid #0f766e;
        border-radius: 8px;
        box-shadow: var(--shadow);
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(0, 1fr) auto;
        margin: 0.45rem 0 1rem;
        padding: 0.9rem;
    }

    .betting-hero-title {
        color: var(--ink);
        font-size: 1.1rem;
        font-weight: 950;
        line-height: 1.15;
    }

    .betting-hero-note {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 750;
        line-height: 1.35;
        margin-top: 0.25rem;
    }

    .betting-hero-stat {
        align-self: center;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        color: #065f46;
        min-width: 8rem;
        padding: 0.65rem 0.75rem;
        text-align: right;
    }

    .betting-hero-stat-label {
        font-size: 0.64rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .betting-hero-stat-value {
        font-size: 1.15rem;
        font-weight: 950;
        line-height: 1.15;
        margin-top: 0.12rem;
    }

    .betting-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin: 0.75rem 0 1rem;
        overflow: hidden;
    }

    .betting-card-shell {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin-bottom: 0.8rem;
        overflow: hidden;
    }

    .betting-card-head {
        align-items: center;
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(0, 1fr) auto;
        padding: 0.82rem 0.9rem;
    }

    .betting-game-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.2;
    }

    .betting-game-meta {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }

    .betting-matchup-logos {
        align-items: center;
        display: flex;
        gap: 0.35rem;
    }

    .betting-matchup-logo {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        height: 2.15rem;
        object-fit: contain;
        padding: 0.2rem;
        width: 2.15rem;
    }

    .betting-rank-pill {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 999px;
        color: #3730a3;
        display: inline-flex;
        font-size: 0.68rem;
        font-weight: 950;
        padding: 0.22rem 0.52rem;
        white-space: nowrap;
    }

    .betting-pick-body {
        padding: 0.85rem 0.9rem 0.95rem;
    }

    .betting-side-grid {
        display: grid;
        gap: 0.65rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .betting-side-tile {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        min-width: 0;
        padding: 0.7rem;
    }

    .betting-side-selected {
        background: #ecfdf5;
        border-color: #86efac;
        box-shadow: inset 0 0 0 1px rgba(22, 163, 74, 0.14);
    }

    .betting-side-top {
        align-items: center;
        display: flex;
        gap: 0.5rem;
        justify-content: space-between;
    }

    .betting-side-name {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 950;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .betting-side-odds {
        color: #0f766e;
        font-size: 1.35rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 0.35rem;
    }

    .betting-side-meta-grid {
        display: grid;
        gap: 0.38rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: 0.55rem;
    }

    .betting-side-meta {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        min-width: 0;
        padding: 0.38rem 0.42rem;
    }

    .betting-side-meta-label {
        color: var(--muted);
        font-size: 0.58rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .betting-side-meta-value {
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 950;
        line-height: 1.1;
        margin-top: 0.14rem;
    }

    .betting-card-actions {
        align-items: center;
        border-top: 1px solid #e2e8f0;
        display: grid;
        gap: 0.55rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        padding: 0.7rem 0.9rem 0.85rem;
    }

    .betting-pick-simple {
        align-items: center;
        display: grid;
        gap: 0.8rem;
        grid-template-columns: auto minmax(0, 1fr);
    }

    .betting-pick-logo {
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        display: flex;
        height: 4.1rem;
        justify-content: center;
        width: 4.1rem;
    }

    .betting-pick-logo img {
        height: 3.1rem;
        object-fit: contain;
        width: 3.1rem;
    }

    .betting-pick-row {
        align-items: center;
        display: grid;
        gap: 0.45rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: 0.65rem;
    }

    .betting-pick-stat {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .betting-primary-action {
        margin-top: 0.7rem;
    }

    .bet-slip {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-left: 4px solid #0f766e;
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 0.85rem;
        position: sticky;
        top: 0.75rem;
    }

    .bet-slip-empty {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 750;
        line-height: 1.35;
        padding: 0.8rem;
    }

    .bet-slip-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.15;
    }

    .bet-slip-market {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 750;
        line-height: 1.3;
        margin-top: 0.2rem;
    }

    .bet-slip-stat-grid {
        display: grid;
        gap: 0.45rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin: 0.65rem 0;
    }

    .bet-slip-stat {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.52rem;
    }

    .bet-slip-warning {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        color: #9a3412;
        font-size: 0.74rem;
        font-weight: 800;
        line-height: 1.35;
        margin-top: 0.55rem;
        padding: 0.58rem 0.65rem;
    }

    .saved-pick-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-left: 4px solid var(--pick-accent, #0f766e);
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin: 0.55rem 0 0.85rem;
        min-width: 0;
        padding: 0.85rem;
    }

    .saved-pick-top {
        align-items: flex-start;
        display: grid;
        gap: 0.65rem;
        grid-template-columns: minmax(0, 1fr) auto;
    }

    .saved-pick-title {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 950;
        line-height: 1.16;
        overflow-wrap: anywhere;
    }

    .saved-pick-meta {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 780;
        line-height: 1.3;
        margin-top: 0.2rem;
    }

    .saved-pick-result {
        border-radius: 999px;
        display: inline-flex;
        font-size: 0.68rem;
        font-weight: 950;
        line-height: 1;
        padding: 0.32rem 0.55rem;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .saved-pick-result.open {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
    }

    .saved-pick-result.win {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
    }

    .saved-pick-result.loss {
        background: #fee2e2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }

    .saved-pick-result.push,
    .saved-pick-result.review {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
    }

    .saved-pick-detail-grid {
        display: grid;
        gap: 0.45rem;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-top: 0.72rem;
    }

    .saved-pick-detail {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        min-width: 0;
        padding: 0.52rem;
    }

    .saved-pick-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 760;
        line-height: 1.35;
        margin-top: 0.62rem;
        padding: 0.58rem 0.65rem;
    }

    .betting-line-table {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin-top: 1rem;
        padding: 0.85rem;
    }

    .sportsbook-open-link {
        align-items: center;
        background: #0f766e;
        border: 1px solid #0f766e;
        border-radius: 8px;
        color: #ffffff !important;
        display: flex;
        font-size: 0.92rem;
        font-weight: 950;
        justify-content: center;
        line-height: 1.2;
        margin: 0.55rem 0;
        min-height: 2.55rem;
        padding: 0.62rem 0.8rem;
        text-align: center;
        text-decoration: none !important;
        width: 100%;
    }

    .sportsbook-open-link:hover {
        background: #0b5f59;
        border-color: #0b5f59;
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .betting-pick-main {
        display: grid;
        gap: 0.8rem;
        grid-template-columns: minmax(180px, 1.3fr) repeat(3, minmax(120px, 0.8fr));
    }

    .betting-pick-primary {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 0.75rem;
    }

    .betting-pick-label {
        color: #047857;
        font-size: 0.65rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .betting-pick-team {
        color: #064e3b;
        font-size: 1.15rem;
        font-weight: 950;
        line-height: 1.15;
        margin-top: 0.18rem;
        overflow-wrap: anywhere;
    }

    .betting-pick-sub {
        color: #047857;
        font-size: 0.75rem;
        font-weight: 800;
        line-height: 1.3;
        margin-top: 0.25rem;
    }

    .betting-simple-stat {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        min-height: 5.2rem;
        padding: 0.65rem 0.7rem;
    }

    .betting-simple-label {
        color: var(--muted);
        font-size: 0.64rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .betting-simple-value {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 950;
        line-height: 1.15;
        margin-top: 0.2rem;
        overflow-wrap: anywhere;
    }

    .betting-simple-note {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 750;
        line-height: 1.25;
        margin-top: 0.2rem;
    }

    .betting-meter {
        background: #e2e8f0;
        border-radius: 999px;
        height: 0.42rem;
        margin-top: 0.48rem;
        overflow: hidden;
    }

    .betting-meter-fill {
        background: #0f766e;
        border-radius: 999px;
        height: 100%;
        width: var(--meter-width, 50%);
    }

    .betting-action-row {
        align-items: stretch;
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(190px, 1fr) minmax(190px, 1fr) minmax(190px, 1fr);
        margin-top: 0.75rem;
    }

    .betting-handoff-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        min-height: 4.6rem;
        padding: 0.65rem 0.75rem;
    }

    .betting-handoff-label {
        color: var(--muted);
        font-size: 0.64rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .betting-handoff-value {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.15;
        margin-top: 0.18rem;
    }

    .betting-handoff-note {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 750;
        line-height: 1.25;
        margin-top: 0.2rem;
    }

    .edge-pill {
        border-radius: 999px;
        display: inline-flex;
        font-size: 0.72rem;
        font-weight: 950;
        padding: 0.2rem 0.5rem;
        white-space: nowrap;
    }

    .edge-strong {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
    }

    .edge-small {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
    }

    .edge-none {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #475569;
    }

    .betting-note {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 750;
        line-height: 1.35;
        margin-top: 0.35rem;
    }

    .prop-result {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        box-shadow: var(--shadow);
        border-radius: 8px;
        margin-top: 0.75rem;
        padding: 0.8rem;
    }

    @media (max-width: 720px) {
        .game-center-panel, .betting-hero, .betting-layout, .betting-board-grid,
        .betting-filter-bar, .betting-card-head, .betting-side-grid,
        .betting-card-actions, .betting-pick-simple, .betting-pick-row,
        .betting-pick-main, .betting-action-row, .saved-pick-top,
        .saved-pick-detail-grid {
            grid-template-columns: 1fr;
        }

        .betting-hero-stat {
            text-align: left;
        }

        .game-center-diamond {
            justify-content: flex-start;
        }

        .standings-header {
            align-items: flex-start;
            flex-direction: column;
        }

        .standings-summary-grid {
            grid-template-columns: 1fr;
        }
    }

    .mlb-live-state {
        align-items: flex-start;
        display: flex;
        flex-direction: column;
        gap: 0.24rem;
        margin-top: 0.35rem;
        min-height: 3.15rem;
    }

    .mlb-live-state .mlb-countdown,
    .mlb-live-state .mlb-live-count {
        margin-top: 0;
    }

    .mlb-live-pitch {
        color: var(--muted);
        display: -webkit-box;
        font-size: 0.72rem;
        font-weight: 850;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        min-height: 1rem;
        overflow: hidden;
        overflow-wrap: anywhere;
    }

    .mlb-live-pitch span {
        color: #c2410c;
        font-size: 0.66rem;
        font-weight: 950;
        margin-right: 0.28rem;
        text-transform: uppercase;
    }

    .live-side-panel {
        align-items: flex-end;
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        min-width: 5.6rem;
    }

    .base-diamond {
        height: 3.15rem;
        position: relative;
        width: 3.15rem;
    }

    .base-node {
        background: #e2e8f0;
        border: 1px solid #cbd5e1;
        height: 0.76rem;
        position: absolute;
        transform: rotate(45deg);
        width: 0.76rem;
    }

    .base-node.occupied {
        background: #16a34a;
        border-color: #15803d;
    }

    .base-node.first {
        right: 0.15rem;
        top: 1.2rem;
    }

    .base-node.second {
        left: 1.2rem;
        top: 0.12rem;
    }

    .base-node.third {
        left: 0.15rem;
        top: 1.2rem;
    }

    .base-node.home {
        background: #ffffff;
        bottom: 0.1rem;
        left: 1.2rem;
    }

    .live-win-prob {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        color: #3730a3;
        display: block;
        font-size: 0.4rem;
        font-weight: 950;
        line-height: 1;
        padding: 0.1rem 0.18rem;
        text-align: right;
        white-space: nowrap;
    }

    div[class*="st-key-mlb-game-card-wrap-"] {
        position: relative;
    }

    div[class*="st-key-mlb-live-wp-wrap-"] {
        position: absolute;
        right: 0.9rem;
        top: 6.58rem;
        width: 5.8rem;
        z-index: 5;
    }

    div[class*="st-key-mlb-live-wp-wrap-"] div.stButton {
        width: 100%;
    }

    div[class*="st-key-mlb-live-wp-wrap-"] button {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        color: #3730a3;
        align-items: center;
        display: flex;
        font-size: 0.65rem !important;
        font-weight: 950 !important;
        height: 1.32rem;
        justify-content: center;
        line-height: 1 !important;
        min-height: 1.32rem;
        padding: 0.12rem 0.22rem;
        text-align: center;
        width: 100%;
    }

    div[class*="st-key-mlb-live-wp-wrap-"] button:hover {
        background: #e0e7ff;
        border-color: #a5b4fc;
        color: #3730a3;
    }

    div[class*="st-key-mlb-live-wp-wrap-"] button div[data-testid="stMarkdownContainer"] {
        white-space: nowrap !important;
    }

    div[class*="st-key-mlb-live-wp-wrap-"] button div[data-testid="stMarkdownContainer"] p,
    div[class*="st-key-mlb-live-wp-wrap-"] button p,
    div[class*="st-key-mlb-live-wp-wrap-"] button span {
        font-size: 0.65rem !important;
        font-weight: 950 !important;
        letter-spacing: 0 !important;
        line-height: 1 !important;
        margin: 0 !important;
        overflow-wrap: normal !important;
        text-align: center !important;
        white-space: nowrap !important;
    }

    .current-matchup-row {
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        display: grid;
        gap: 0.65rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin: 0 0 0.75rem;
        min-height: 4.65rem;
        padding: 0.6rem 0.65rem;
    }

    .card-slot-empty {
        visibility: hidden;
    }

    .current-player {
        align-items: center;
        display: flex;
        gap: 0.5rem;
        min-width: 0;
    }

    .current-player.pitcher {
        justify-content: flex-end;
        text-align: right;
    }

    .current-headshot {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        flex: 0 0 auto;
        height: 54px;
        object-fit: cover;
        object-position: center 28%;
        width: 44px;
    }

    .current-role {
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .current-name {
        color: var(--ink);
        font-size: 0.84rem;
        font-weight: 950;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .current-stat {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 0.12rem;
    }

    .current-vs {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 950;
        text-align: center;
    }

    .mlb-prediction-grid {
        display: grid;
        gap: 0.55rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: auto;
    }

    .mlb-prediction-cell {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-top: 3px solid var(--accent);
        border-radius: 8px;
        min-height: 4.15rem;
        padding: 0.5rem 0.6rem;
    }

    .mlb-prediction-label {
        color: var(--muted);
        font-size: 0.65rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .mlb-prediction-value {
        color: var(--ink);
        display: -webkit-box;
        font-size: 0.95rem;
        font-weight: 950;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-top: 0.18rem;
        overflow: hidden;
        overflow-wrap: anywhere;
    }

    .prediction-explain {
        color: var(--muted);
        display: -webkit-box;
        font-size: 0.72rem;
        font-weight: 750;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.3;
        margin-top: 0.48rem;
        min-height: 1.9rem;
        overflow: hidden;
        overflow-wrap: anywhere;
    }

    .section-kicker {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 950;
        letter-spacing: 0.08em;
        margin: 0.9rem 0 0.4rem;
        text-transform: uppercase;
    }

    .dashboard-grid, .mlb-status-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        margin-bottom: 1rem;
    }

    .dashboard-card, .mlb-status-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--card-color, var(--accent-blue));
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 0.8rem;
    }

    .dashboard-label {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .dashboard-value {
        color: var(--ink);
        font-size: 1.3rem;
        font-weight: 950;
        line-height: 1.15;
        margin-top: 0.2rem;
        overflow-wrap: anywhere;
    }

    .dashboard-note {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1.3;
        margin-top: 0.2rem;
        overflow-wrap: anywhere;
    }

    .home-signal-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        margin-bottom: 1rem;
    }

    .home-signal-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--card-color, var(--accent-blue));
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 0.85rem;
    }

    .home-signal-title {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .home-signal-main {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.15;
        margin-top: 0.28rem;
        overflow-wrap: anywhere;
    }

    .home-signal-value {
        color: var(--accent-blue);
        font-size: 1.35rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 0.45rem;
    }

    .home-two-column {
        display: grid;
        gap: 0.9rem;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }

    .team-mini-row {
        align-items: center;
        background: var(--surface);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        display: grid;
        gap: 0.55rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        margin-bottom: 0.45rem;
        padding: 0.55rem 0.65rem;
    }

    .team-mini-logo {
        height: 32px;
        object-fit: contain;
        width: 32px;
    }

    .team-mini-name {
        color: var(--ink);
        font-size: 0.86rem;
        font-weight: 950;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .team-mini-meta {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 750;
        line-height: 1.25;
        margin-top: 0.1rem;
    }

    .team-mini-score {
        color: var(--accent-blue);
        font-size: 1rem;
        font-weight: 950;
        white-space: nowrap;
    }

    .home-game-row {
        align-items: center;
        background: var(--surface);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        display: grid;
        gap: 0.65rem;
        grid-template-columns: minmax(0, 1fr) auto;
        margin-bottom: 0.5rem;
        padding: 0.65rem 0.75rem;
    }

    .home-game-matchup {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        min-width: 0;
    }

    .home-game-logo {
        height: 26px;
        object-fit: contain;
        width: 26px;
    }

    .home-game-team {
        color: var(--ink);
        font-size: 0.86rem;
        font-weight: 950;
        overflow-wrap: anywhere;
    }

    .home-game-at {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 950;
    }

    .home-game-meta {
        color: var(--muted);
        font-size: 0.73rem;
        font-weight: 750;
        line-height: 1.3;
        margin-top: 0.12rem;
    }

    .home-game-score {
        color: var(--accent-blue);
        font-size: 1rem;
        font-weight: 950;
        text-align: right;
        white-space: nowrap;
    }

    .matchup-preview {
        align-items: center;
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin: 0.7rem 0 1rem;
    }

    .preview-team-card {
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--team-color, var(--accent-blue));
        border-radius: 8px;
        box-shadow: var(--shadow);
        display: flex;
        gap: 0.7rem;
        min-width: 0;
        padding: 0.85rem;
    }

    .preview-logo {
        height: 44px;
        object-fit: contain;
        width: 44px;
    }

    .preview-role {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .preview-name {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .preview-center {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 950;
    }

    .prediction-card, .score-sim-card, .profile-hero, .score-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--team-color, var(--accent));
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        padding: 0.95rem;
    }

    .prediction-topline, .score-topline {
        align-items: center;
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.65rem;
    }

    .prediction-main, .score-sim-board {
        align-items: center;
        display: grid;
        gap: 0.75rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
    }

    .prediction-logo, .score-sim-logo, .profile-logo {
        height: 54px;
        object-fit: contain;
        width: 54px;
    }

    .prediction-winner, .profile-name {
        color: var(--ink);
        font-size: 1.3rem;
        font-weight: 950;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .prediction-sub {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 750;
        margin-top: 0.15rem;
    }

    .prediction-prob, .score-sim-points, .score-wins {
        color: var(--accent-blue);
        font-size: 1.65rem;
        font-weight: 950;
        line-height: 1;
    }

    .probability-meter {
        background: #e2e8f0;
        border-radius: 999px;
        height: 0.55rem;
        margin: 0.75rem 0;
        overflow: hidden;
    }

    .probability-fill {
        background: var(--team-color, var(--accent));
        height: 100%;
        width: var(--probability-width, 50%);
    }

    .signal-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }

    .signal-pill, .score-chip {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        color: var(--ink);
        font-size: 0.7rem;
        font-weight: 850;
        padding: 0.18rem 0.5rem;
        white-space: nowrap;
    }

    .matchup-card-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin-bottom: 1rem;
    }

    .team-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--team-color, var(--accent-blue));
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 0.85rem;
        text-align: center;
    }

    .team-logo {
        height: 52px;
        object-fit: contain;
        width: 52px;
    }

    .team-name {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 950;
        margin-top: 0.25rem;
        overflow-wrap: anywhere;
    }

    .team-probability {
        color: var(--accent-blue);
        font-size: 1.45rem;
        font-weight: 950;
        margin-top: 0.25rem;
    }

    .logo-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.75rem 0 1rem;
    }

    .logo-tile {
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid var(--team-color, var(--accent-blue));
        border-radius: 8px;
        display: flex;
        gap: 0.35rem;
        padding: 0.3rem 0.45rem;
    }

    .logo-tile img {
        height: 24px;
        object-fit: contain;
        width: 24px;
    }

    .logo-tile span {
        color: var(--ink);
        font-size: 0.72rem;
        font-weight: 850;
    }

    .profile-hero {
        align-items: center;
        display: flex;
        gap: 0.85rem;
    }

    .profile-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.55rem;
    }

    .profile-stat {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.4rem 0.55rem;
    }

    .profile-label {
        color: var(--muted);
        font-size: 0.65rem;
        font-weight: 950;
        text-transform: uppercase;
    }

    .profile-value {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 950;
    }

    .score-row {
        align-items: center;
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        padding: 0.45rem 0;
    }

    .score-team-wrap {
        align-items: center;
        display: flex;
        gap: 0.45rem;
        min-width: 0;
    }

    .score-logo {
        height: 30px;
        object-fit: contain;
        width: 30px;
    }

    .score-team {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 900;
        overflow-wrap: anywhere;
    }

    .score-meta {
        border-top: 1px solid #e2e8f0;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
    }

    .score-meta-line {
        color: var(--muted);
        display: block;
        font-size: 0.76rem;
        font-weight: 750;
        line-height: 1.35;
    }

    .score-sim-team {
        align-items: center;
        display: flex;
        gap: 0.55rem;
        min-width: 0;
    }

    .score-sim-name {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 950;
        overflow-wrap: anywhere;
    }

    .score-sim-role {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 750;
    }

    .score-sim-center {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 950;
        text-align: center;
    }

    .mlb-detail-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin-top: 0.5rem;
        padding: 1rem;
    }

    .mlb-detail-title {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 950;
        margin-bottom: 0.25rem;
    }

    .mlb-play-row, .mlb-pitch-row, .mlb-player-row, .mlb-lineup-row {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.45rem;
        padding: 0.55rem 0.65rem;
    }

    .mlb-row-title {
        color: var(--ink);
        font-size: 0.86rem;
        font-weight: 900;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .mlb-row-meta {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 750;
        line-height: 1.3;
        margin-top: 0.12rem;
        overflow-wrap: anywhere;
    }

    .pitcher-card {
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--team-color, var(--accent-blue));
        border-radius: 8px;
        box-shadow: var(--shadow);
        display: grid;
        gap: 0.7rem;
        grid-template-columns: auto minmax(0, 1fr);
        margin-bottom: 0.75rem;
        padding: 0.85rem;
    }

    .pitcher-photo {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        height: 86px;
        object-fit: cover;
        object-position: center 28%;
        width: 72px;
    }

    .pitcher-name {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .pitcher-stat-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-top: 0.45rem;
    }

    .pitcher-stat {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        color: var(--ink);
        font-size: 0.7rem;
        font-weight: 850;
        padding: 0.18rem 0.45rem;
        white-space: nowrap;
    }

    .next-up-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--accent-blue);
        border-radius: 8px;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        padding: 0.95rem;
    }

    .next-up-head {
        align-items: flex-start;
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.8rem;
    }

    .next-up-title {
        color: var(--ink);
        font-size: 1.08rem;
        font-weight: 950;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .next-up-meta {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 750;
        line-height: 1.35;
        margin-top: 0.18rem;
    }

    .next-up-pitchers {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    @media (max-width: 760px) {
        .mlb-matchup, .current-matchup-row, .mlb-prediction-grid, .dashboard-grid, .mlb-status-grid,
        .matchup-preview, .prediction-main, .score-sim-board, .matchup-card-grid,
        .home-two-column, .home-game-row, .pitcher-card, .next-up-pitchers {
            grid-template-columns: 1fr;
        }

        .current-player.pitcher {
            justify-content: flex-start;
            text-align: left;
        }

        .next-up-head {
            display: block;
        }

        .mlb-vs {
            display: none;
        }
    }
</style>
"""


def get_model_mtime() -> float | None:
    """Return model file mtime for cache invalidation."""
    if not MODEL_PATH.exists():
        return None

    return MODEL_PATH.stat().st_mtime


@st.cache_resource
def load_model_bundle(file_mtime: float | None = None) -> dict:
    """Load trained MLB model bundle."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_team_strength() -> pd.DataFrame:
    """Load current MLB team strength."""
    if not TEAM_STRENGTH_PATH.exists():
        return pd.DataFrame()

    strength = pd.read_csv(TEAM_STRENGTH_PATH)

    if "LAST_GAME_DATE" in strength.columns:
        strength["LAST_GAME_DATE"] = pd.to_datetime(strength["LAST_GAME_DATE"])

    return strength


@st.cache_data
def load_pitcher_strength() -> pd.DataFrame:
    """Load current MLB probable-starter strength proxies."""
    if not PITCHER_STRENGTH_PATH.exists():
        return pd.DataFrame()

    pitchers = pd.read_csv(PITCHER_STRENGTH_PATH)

    if "LAST_START_DATE" in pitchers.columns:
        pitchers["LAST_START_DATE"] = pd.to_datetime(pitchers["LAST_START_DATE"])

    return pitchers


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Load MLB model metrics."""
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_model_features() -> pd.DataFrame:
    """Load MLB model feature rows."""
    if not FEATURES_PATH.exists():
        return pd.DataFrame()

    features = pd.read_csv(FEATURES_PATH)

    if "GAME_DATE" in features.columns:
        features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])

    return features.sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)


@st.cache_data
def load_raw_games() -> pd.DataFrame:
    """Load historical MLB game rows."""
    if not RAW_GAMES_PATH.exists():
        return pd.DataFrame()

    games = pd.read_csv(RAW_GAMES_PATH)

    if "GAME_DATE" in games.columns:
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    if "GAME_DATETIME" in games.columns:
        games["GAME_DATETIME"] = pd.to_datetime(games["GAME_DATETIME"], errors="coerce", utc=True)

    return games.sort_values(["GAME_DATE", "GAME_PK"]).reset_index(drop=True)


@st.cache_data(ttl=SCHEDULE_CACHE_SECONDS)
def load_mlb_schedule(start_date: str, end_date: str) -> pd.DataFrame:
    """Load MLB schedule rows for app display."""
    try:
        payload = fetch_schedule(start_date=start_date, end_date=end_date)
    except Exception:
        return pd.DataFrame()

    rows = [normalize_schedule_game(game) for game in iter_schedule_games(payload)]
    rows = [
        row
        for row in rows
        if row["HOME_TEAM"] and row["AWAY_TEAM"]
    ]

    if not rows:
        return pd.DataFrame()

    games = pd.DataFrame(rows)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["GAME_DATETIME"] = pd.to_datetime(games["GAME_DATETIME"], errors="coerce", utc=True)
    return games.sort_values(["GAME_DATE", "GAME_DATETIME", "GAME_PK"]).reset_index(drop=True)


@st.cache_data(ttl=LIVE_FEED_CACHE_SECONDS, show_spinner=False)
def load_live_game_feed(game_pk: str) -> dict:
    """Load a live MLB game feed for scores, lineups, and play-by-play."""
    try:
        return fetch_live_game_feed(game_pk)
    except Exception:
        return {}


@st.cache_data(ttl=STANDINGS_CACHE_SECONDS, show_spinner=False)
def load_mlb_standings(season: int) -> tuple[pd.DataFrame, str]:
    """Load live MLB standings from the public MLB Stats API."""
    try:
        payload = fetch_standings(season=season)
    except Exception:
        return pd.DataFrame(), ""

    rows = build_standings_rows(payload)
    loaded_at = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%b %-d, %-I:%M:%S %p ET")

    if not rows:
        return pd.DataFrame(), loaded_at

    standings = pd.DataFrame(rows)
    standings["Division Sort"] = standings["Division"].map(
        {
            division: index
            for league_divisions in MLB_DIVISION_ORDER_BY_LEAGUE.values()
            for index, division in enumerate(league_divisions)
        }
    ).fillna(99)
    standings["Rank Sort"] = standings["Rank"].map(lambda value: safe_int(value) or 99)
    standings = standings.sort_values(["League", "Division Sort", "Rank Sort", "Team"]).reset_index(drop=True)
    return standings, loaded_at


@st.cache_data(ttl=ODDS_CACHE_SECONDS, show_spinner=False)
def load_mlb_odds(api_key: str, markets: str = "h2h,spreads,totals") -> list[dict]:
    """Load MLB betting odds when an API key is configured."""
    try:
        return fetch_mlb_odds(api_key=api_key, markets=markets)
    except Exception:
        return []


@st.cache_data(ttl=ODDS_CACHE_SECONDS, show_spinner=False)
def load_mlb_odds_with_status(api_key: str, markets: str = "h2h,spreads,totals") -> tuple[list[dict], str]:
    """Load MLB betting odds and return a user-facing error when the feed fails."""
    try:
        return fetch_mlb_odds(api_key=api_key, markets=markets), ""
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            message = str(payload.get("message") or payload.get("error_code") or body)
        except Exception:
            message = str(exc)

        return [], f"Odds API error {exc.code}: {message}"
    except URLError as exc:
        return [], f"Could not reach the odds API: {exc.reason}"
    except Exception as exc:
        return [], f"Could not load odds: {exc}"


@st.cache_data(ttl=ODDS_CACHE_SECONDS, show_spinner=False)
def load_sharpsports_bet_slips(api_key: str, status: str = "", limit: int = 50) -> dict:
    """Load synced SharpSports bet slips when an API key is configured."""
    params = {
        "sport": "baseball",
        "league": "MLB",
        "limit": int(limit),
    }

    if status:
        params["status"] = status

    try:
        return fetch_sharpsports_bet_slips(api_key=api_key, params=params)
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_player_season_stats(person_id: str, season: int, group: str = "pitching") -> dict:
    """Load one MLB player's season stat payload."""
    try:
        return fetch_player_stats(person_id=person_id, season=season, group=group)
    except Exception:
        return {}


@st.cache_data(ttl=WEATHER_CACHE_SECONDS, show_spinner=False)
def load_ballpark_weather(team_id: object, game_datetime: object) -> dict:
    """Load nearest hourly forecast for the home ballpark."""
    coords = get_ballpark_coordinates(team_id)

    if not coords:
        return {}

    parsed = pd.to_datetime(game_datetime, errors="coerce")

    if pd.isna(parsed):
        return {}

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize("UTC")

    game_utc = parsed.tz_convert(ZoneInfo("UTC"))

    try:
        payload = fetch_weather_forecast(
            latitude=float(coords["latitude"]),
            longitude=float(coords["longitude"]),
            start_date=game_utc.date().isoformat(),
            end_date=game_utc.date().isoformat(),
        )
    except Exception:
        return {}

    return extract_nearest_weather_hour(payload, game_utc, str(coords["name"]))


def clear_mlb_caches() -> None:
    """Clear MLB app caches."""
    for loader in [
        load_model_bundle,
        load_team_strength,
        load_pitcher_strength,
        load_model_metrics,
        load_model_features,
        load_raw_games,
        build_team_backtest_summary,
        build_probability_calibration,
        load_mlb_schedule,
        load_live_game_feed,
        load_mlb_odds,
        load_sharpsports_bet_slips,
        load_player_season_stats,
        load_ballpark_weather,
    ]:
        loader.clear()


def get_available_teams() -> list[str]:
    """Return teams known to the MLB model."""
    strength = load_team_strength()

    if strength.empty:
        return []

    return sorted(strength["TEAM_NAME"].dropna().unique())


def filter_known_games(games: pd.DataFrame) -> pd.DataFrame:
    """Keep only games where both teams are in the local strength table."""
    teams = set(get_available_teams())

    if games.empty or not teams:
        return pd.DataFrame()

    return games[
        games["HOME_TEAM"].isin(teams) & games["AWAY_TEAM"].isin(teams)
    ].copy().reset_index(drop=True)


def load_today_games() -> pd.DataFrame:
    """Load today's MLB games."""
    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    games = load_mlb_schedule(today.isoformat(), today.isoformat())
    return filter_known_games(games)


def load_next_upcoming_games() -> pd.DataFrame:
    """Load games on the next future day with MLB games."""
    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    start = today + timedelta(days=1)
    end = today + timedelta(days=14)
    games = filter_known_games(load_mlb_schedule(start.isoformat(), end.isoformat()))

    if games.empty:
        return pd.DataFrame()

    future_dates = games["GAME_DATE"].dt.date
    next_date = future_dates.min()
    return games[future_dates.eq(next_date)].copy().reset_index(drop=True)


def get_team_strength_row(team_name: str, strength: pd.DataFrame) -> pd.Series:
    """Return one MLB strength row."""
    rows = strength[strength["TEAM_NAME"].str.lower().eq(team_name.lower())]

    if rows.empty:
        raise ValueError(f"MLB team not found: {team_name}")

    return rows.iloc[0]


def clamp_probability(probability: float) -> float:
    """Clamp one-game MLB probability."""
    return min(max(probability, MIN_GAME_PROBABILITY), MAX_GAME_PROBABILITY)


def neutral_feature_value(column: str) -> float:
    """Return a neutral value for model features missing from old strength files."""
    if column == "ELO":
        return 1500.0
    if "WIN_PCT" in column or column == "PITCHER_TEAM_WIN_PCT":
        return 0.5
    if "RUNS_FOR" in column or "RUNS_AGAINST" in column:
        return float(DEFAULT_RUN_ENVIRONMENT)
    if column in {"PROJECTED_LINEUP_STRENGTH", "PITCHER_RUN_SUPPORT_PER_START", "PITCHER_RUNS_ALLOWED_PER_START"}:
        return float(DEFAULT_RUN_ENVIRONMENT)
    if column in {"DAYS_REST", "PITCHER_DAYS_REST"}:
        return 3.0
    return 0.0


def numeric_row_value(row: pd.Series, column: str) -> float:
    """Read a numeric row value with neutral fallback."""
    if column not in row.index:
        return neutral_feature_value(column)

    parsed = safe_float(row[column], default=None)
    return neutral_feature_value(column) if parsed is None else float(parsed)


def get_pitcher_strength_row(pitcher_name: str) -> pd.Series | None:
    """Return one pitcher proxy row by normalized name."""
    pitchers = load_pitcher_strength()

    if pitchers.empty or not pitcher_name:
        return None

    key = str(pitcher_name).strip().lower()
    rows = pitchers[pitchers["PITCHER_KEY"].astype(str).str.lower().eq(key)]

    if rows.empty:
        rows = pitchers[pitchers["PITCHER_NAME"].astype(str).str.lower().eq(key)]

    if rows.empty:
        return None

    return rows.sort_values("LAST_START_DATE", ascending=False).iloc[0]


def pitcher_context_from_game(game_context: pd.Series | dict | None, side: str) -> dict[str, float]:
    """Build starter proxy context from the schedule row or neutral defaults."""
    prefix = "HOME" if side == "home" else "AWAY"
    pitcher_name = ""

    if game_context is not None:
        pitcher_name = str(
            game_context.get(f"{prefix}_PROBABLE_PITCHER", "")
            if hasattr(game_context, "get")
            else ""
        ).strip()

    row = get_pitcher_strength_row(pitcher_name)
    columns = [
        "PITCHER_TEAM_WIN_PCT",
        "PITCHER_RUNS_ALLOWED_PER_START",
        "PITCHER_RUN_SUPPORT_PER_START",
        "PITCHER_STARTS",
        "PITCHER_DAYS_REST",
    ]

    if row is None:
        return {column: neutral_feature_value(column) for column in columns}

    return {column: numeric_row_value(row, column) for column in columns}


def weather_run_factor_adjustment(weather: dict) -> float:
    """Return a small run-environment adjustment from ballpark weather."""
    if not weather:
        return 0.0

    adjustment = 0.0
    temperature = safe_float(weather.get("temperature"), default=None)
    wind_speed = safe_float(weather.get("wind_speed"), default=None)
    precipitation = safe_float(weather.get("precipitation_probability"), default=None)

    if temperature is not None:
        adjustment += min(max((float(temperature) - 70.0) * 0.0012, -0.035), 0.045)

    if wind_speed is not None and float(wind_speed) >= 12:
        adjustment += min((float(wind_speed) - 12.0) * 0.001, 0.018)

    if precipitation is not None and float(precipitation) >= 30:
        adjustment -= min((float(precipitation) - 30.0) * 0.0006, 0.03)

    try:
        weather_code = int(float(weather.get("weather_code")))
    except (TypeError, ValueError):
        weather_code = None

    if weather_code in {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99}:
        adjustment -= 0.012

    return min(max(adjustment, -0.06), 0.06)


def build_weather_prediction_context(
    home_team: str,
    game_context: pd.Series | dict | None,
) -> pd.Series | dict | None:
    """Attach hidden weather-derived values for schedule-based predictions."""
    if game_context is None or not hasattr(game_context, "get"):
        return game_context

    context = game_context.to_dict() if isinstance(game_context, pd.Series) else dict(game_context)
    weather = context.get("WEATHER_FORECAST")

    if not isinstance(weather, dict):
        weather = load_ballpark_weather(context.get("HOME_TEAM_ID"), context.get("GAME_DATETIME"))

    base_factor = safe_float(context.get("HOME_PARK_RUN_FACTOR"), default=None)

    if base_factor is None:
        base_factor = float(get_park_run_factor(home_team))

    adjustment = weather_run_factor_adjustment(weather)
    context["WEATHER_RUN_FACTOR_ADJUSTMENT"] = adjustment
    context["WEATHER_ADJUSTED_PARK_RUN_FACTOR"] = min(max(float(base_factor) + adjustment, 0.85), 1.20)
    return context


def home_park_factor_from_context(home_team: str, game_context: pd.Series | dict | None) -> float:
    """Return park factor for a prediction row."""
    if game_context is not None and hasattr(game_context, "get"):
        weather_adjusted = safe_float(game_context.get("WEATHER_ADJUSTED_PARK_RUN_FACTOR"), default=None)

        if weather_adjusted is not None:
            return float(weather_adjusted)

        venue_factor = safe_float(game_context.get("HOME_PARK_RUN_FACTOR"), default=None)

        if venue_factor is not None:
            return float(venue_factor)

    return float(get_park_run_factor(home_team))


def build_prediction_row(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
    feature_columns: list[str],
    game_context: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Build one MLB model input row."""
    home = get_team_strength_row(home_team, strength)
    away = get_team_strength_row(away_team, strength)
    home_pitcher = pitcher_context_from_game(game_context, "home")
    away_pitcher = pitcher_context_from_game(game_context, "away")
    values = {}

    for column in feature_columns:
        if column == "HOME_ELO_WIN_PROB":
            values[column] = expected_score(
                float(home["ELO"]) + HOME_ELO_ADVANTAGE,
                float(away["ELO"]),
            )
            continue

        if column == "HOME_PARK_RUN_FACTOR":
            values[column] = home_park_factor_from_context(home_team, game_context)
            continue

        strength_column = column.removeprefix("DIFF_")
        strength_column = FEATURE_COLUMN_ALIASES.get(strength_column, strength_column)

        if strength_column.startswith("PITCHER_"):
            values[column] = home_pitcher.get(
                strength_column,
                neutral_feature_value(strength_column),
            ) - away_pitcher.get(
                strength_column,
                neutral_feature_value(strength_column),
            )
            continue

        values[column] = numeric_row_value(home, strength_column) - numeric_row_value(away, strength_column)

    return pd.DataFrame([values], columns=feature_columns)


def build_prediction_explanation(
    home_team: str,
    away_team: str,
    winner: str,
    home_strength: pd.Series,
    away_strength: pd.Series,
    game_context: pd.Series | dict | None = None,
) -> list[str]:
    """Build short, human-readable reasons for a predicted winner."""
    winner_is_home = winner == home_team
    winner_strength = home_strength if winner_is_home else away_strength
    opponent_strength = away_strength if winner_is_home else home_strength
    reasons: list[str] = []

    def add_numeric_edge(
        label: str,
        column: str,
        threshold: float,
        digits: int = 1,
        higher_is_better: bool = True,
        suffix: str = "",
    ) -> None:
        if len(reasons) >= 4:
            return

        winner_value = numeric_row_value(winner_strength, column)
        opponent_value = numeric_row_value(opponent_strength, column)
        edge = winner_value - opponent_value if higher_is_better else opponent_value - winner_value

        if abs(edge) < threshold:
            return

        formatted = f"{edge:+.{digits}f}{suffix}" if digits else f"{edge:+.0f}{suffix}"
        reasons.append(f"{label} {formatted}")

    add_numeric_edge("Power rating edge", "ELO", 15, digits=0)
    add_numeric_edge("Run diff edge", "SEASON_RUN_DIFF_PER_GAME", 0.2, suffix="/G")
    add_numeric_edge("Last 10 form", "ROLLING_RUN_DIFF_10", 0.3)

    if len(reasons) < 4:
        win_pct_edge = numeric_row_value(winner_strength, "SEASON_WIN_PCT") - numeric_row_value(
            opponent_strength,
            "SEASON_WIN_PCT",
        )

        if abs(win_pct_edge) >= 0.04:
            reasons.append(f"Record edge {win_pct_edge * 100:+.0f} pts")

    if len(reasons) < 4 and game_context is not None:
        home_pitcher = pitcher_context_from_game(game_context, "home")
        away_pitcher = pitcher_context_from_game(game_context, "away")
        winner_pitcher = home_pitcher if winner_is_home else away_pitcher
        opponent_pitcher = away_pitcher if winner_is_home else home_pitcher
        run_allowed_edge = opponent_pitcher.get("PITCHER_RUNS_ALLOWED_PER_START", DEFAULT_RUN_ENVIRONMENT) - winner_pitcher.get(
            "PITCHER_RUNS_ALLOWED_PER_START",
            DEFAULT_RUN_ENVIRONMENT,
        )

        if run_allowed_edge >= 0.3:
            reasons.append(f"Starter edge {run_allowed_edge:+.1f} runs/start")

    if len(reasons) < 4 and winner_is_home:
        reasons.append("Home-field edge")

    if not reasons:
        reasons.append("Pricing blend sees a narrow edge")

    return reasons[:4]


def predict_game_details(
    home_team: str,
    away_team: str,
    game_context: pd.Series | dict | None = None,
) -> dict:
    """Return MLB game prediction details."""
    bundle = load_model_bundle(get_model_mtime())
    strength = load_team_strength()
    feature_columns = bundle["feature_columns"]
    prediction_context = build_weather_prediction_context(home_team, game_context)
    prediction_row = build_prediction_row(home_team, away_team, strength, feature_columns, prediction_context)
    model_probability = float(bundle["model"].predict_proba(prediction_row)[0][1])
    home_strength = get_team_strength_row(home_team, strength)
    away_strength = get_team_strength_row(away_team, strength)
    elo_probability = expected_score(
        float(home_strength["ELO"]) + HOME_ELO_ADVANTAGE,
        float(away_strength["ELO"]),
    )
    blend_settings = bundle.get("blend_settings", {}) or {}
    model_weight = float(blend_settings.get("model_probability_weight", MODEL_WEIGHT))
    elo_weight = float(blend_settings.get("elo_probability_weight", ELO_WEIGHT))
    total_weight = model_weight + elo_weight

    if total_weight <= 0:
        model_weight = MODEL_WEIGHT
        elo_weight = ELO_WEIGHT
        total_weight = model_weight + elo_weight

    model_weight = model_weight / total_weight
    elo_weight = elo_weight / total_weight
    blended = clamp_probability(
        (model_weight * model_probability) + (elo_weight * elo_probability)
    )
    away_probability = 1 - blended
    winner = home_team if blended >= 0.5 else away_team
    winner_probability = max(blended, away_probability)
    explanation = build_prediction_explanation(
        home_team=home_team,
        away_team=away_team,
        winner=winner,
        home_strength=home_strength,
        away_strength=away_strength,
        game_context=prediction_context,
    )
    return {
        "winner": winner,
        "winner_probability": winner_probability,
        "home_probability": blended,
        "away_probability": away_probability,
        "model_probability": model_probability,
        "elo_probability": elo_probability,
        "model_weight": model_weight,
        "elo_weight": elo_weight,
        "weather_run_factor_adjustment": (
            prediction_context.get("WEATHER_RUN_FACTOR_ADJUSTMENT", 0.0)
            if hasattr(prediction_context, "get")
            else 0.0
        ),
        "explanation": explanation,
    }


def build_game_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Predict a schedule frame."""
    if games.empty or not MODEL_PATH.exists() or load_team_strength().empty:
        return pd.DataFrame()

    rows = []

    for _, game in games.iterrows():
        details = predict_game_details(str(game["HOME_TEAM"]), str(game["AWAY_TEAM"]), game)
        rows.append(
            {
                **game.to_dict(),
                "PREDICTED_WINNER": details["winner"],
                "WINNER_PROBABILITY": details["winner_probability"],
                "HOME_WIN_PROBABILITY": details["home_probability"],
                "AWAY_WIN_PROBABILITY": details["away_probability"],
                "MODEL_PROBABILITY": details["model_probability"],
                "ELO_PROBABILITY": details["elo_probability"],
                "PREDICTION_EXPLANATION": " / ".join(details.get("explanation", [])),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def build_team_backtest_summary(model_mtime: float | None = None) -> pd.DataFrame:
    """Build team-level historical model reliability from saved features."""
    features = load_model_features()

    if features.empty or not MODEL_PATH.exists():
        return pd.DataFrame()

    try:
        bundle = load_model_bundle(model_mtime)
    except Exception:
        return pd.DataFrame()

    feature_columns = bundle.get("feature_columns", [])

    if not feature_columns or any(column not in features.columns for column in feature_columns):
        return pd.DataFrame()

    probabilities = bundle["model"].predict_proba(features[feature_columns])[:, 1]
    rows = []

    for (_, game), home_probability in zip(features.iterrows(), probabilities):
        home_team = str(game["HOME_TEAM"])
        away_team = str(game["AWAY_TEAM"])
        home_win = int(game["HOME_WIN"]) == 1
        predicted_home = float(home_probability) >= 0.5
        predicted_winner = home_team if predicted_home else away_team
        actual_winner = home_team if home_win else away_team
        correct = int(predicted_winner == actual_winner)
        confidence = max(float(home_probability), 1 - float(home_probability))

        for team in [home_team, away_team]:
            rows.append(
                {
                    "Team": team,
                    "Season": int(game["SEASON"]),
                    "Games": 1,
                    "Correct": correct,
                    "Avg Confidence": confidence,
                }
            )

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows).groupby(["Team", "Season"], as_index=False).agg(
        Games=("Games", "sum"),
        Correct=("Correct", "sum"),
        Avg_Confidence=("Avg Confidence", "mean"),
    )
    summary["Accuracy"] = summary["Correct"] / summary["Games"]
    return summary


@st.cache_data
def build_probability_calibration(model_mtime: float | None = None) -> pd.DataFrame:
    """Build predicted home probability calibration buckets."""
    features = load_model_features()

    if features.empty or not MODEL_PATH.exists():
        return pd.DataFrame()

    try:
        bundle = load_model_bundle(model_mtime)
    except Exception:
        return pd.DataFrame()

    feature_columns = bundle.get("feature_columns", [])

    if not feature_columns or any(column not in features.columns for column in feature_columns):
        return pd.DataFrame()

    calibration = features[["HOME_WIN"]].copy()
    calibration["Predicted"] = bundle["model"].predict_proba(features[feature_columns])[:, 1]
    calibration["Bucket"] = pd.cut(
        calibration["Predicted"],
        bins=[0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0],
        include_lowest=True,
    ).astype(str)
    return calibration.groupby("Bucket", as_index=False).agg(
        Predicted=("Predicted", "mean"),
        Actual=("HOME_WIN", "mean"),
        Games=("HOME_WIN", "size"),
    )


def format_game_time(value: object) -> str:
    """Format game time in Eastern time."""
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return "Time TBD"

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize("UTC")

    eastern = parsed.tz_convert(ZoneInfo("America/New_York"))
    return eastern.strftime("%b %-d, %-I:%M %p ET")


def format_start_countdown(value: object, show_starting_soon: bool = False) -> str:
    """Format a countdown until first pitch."""
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return ""

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize("UTC")

    now = pd.Timestamp.now(tz=ZoneInfo("UTC"))
    seconds = int((parsed.tz_convert(ZoneInfo("UTC")) - now).total_seconds())

    if seconds <= 0:
        return "Starting soon" if show_starting_soon else ""

    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"Starts in {days}d {hours}h"

    if hours:
        return f"Starts in {hours}h {minutes}m"

    return f"Starts in {minutes}m {seconds}s"


def format_score(value: object) -> str:
    """Format a baseball score if present."""
    if value is None or pd.isna(value):
        return ""

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def get_nested(payload: dict, path: list[str], default: object = None) -> object:
    """Read one nested dictionary value."""
    current: object = payload

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

    return default if current is None else current


def normalize_player_id(value: object) -> str:
    """Normalize MLB player ids from live-feed keys and values."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)

        if parsed.is_integer():
            return str(int(parsed))

    text = "" if value is None else str(value).strip()

    if text.startswith("ID"):
        text = text[2:]

    if text.lower() in {"nan", "none", "<na>"}:
        return ""

    try:
        parsed = float(text)

        if parsed.is_integer():
            return str(int(parsed))
    except ValueError:
        pass

    return text


def safe_float(value: object, default: float | None = 0.0) -> float | None:
    """Parse a numeric MLB stat value."""
    text = "" if value is None else str(value).strip()

    if not text or text in {"-", "--", ".---", "None", "nan", "<NA>"}:
        return default

    try:
        return float(text.replace("%", ""))
    except (TypeError, ValueError):
        return default


def display_stat(value: object, fallback: str = "-") -> str:
    """Display one MLB stat without losing baseball decimal formatting."""
    text = "" if value is None else str(value).strip()
    return text if text and text not in {"None", "nan", "<NA>"} else fallback


def format_whole_number(value: object) -> str:
    """Display a numeric stat as an integer when possible."""
    parsed = safe_float(value, default=None)

    if parsed is None:
        return "-"

    return str(int(parsed))


def format_batting_slot(value: object) -> int | None:
    """Convert MLB batting-order slots like 100/200 into 1/2."""
    try:
        slot = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    if slot <= 0:
        return None

    return max(1, slot // 100)


def team_logo_url(team_id: object) -> str:
    """Return a public MLB static team-logo URL."""
    try:
        if pd.isna(team_id):
            return ""
        team_id_number = int(float(team_id))
    except (TypeError, ValueError):
        return ""

    return f"https://www.mlbstatic.com/team-logos/{team_id_number}.svg"


def get_ballpark_coordinates(team_id: object) -> dict:
    """Return static home-ballpark coordinates for one MLB team id."""
    try:
        if team_id is None or pd.isna(team_id):
            return {}

        team_id_number = int(float(team_id))
    except (TypeError, ValueError):
        return {}

    return BALLPARK_WEATHER_COORDS.get(team_id_number, {})


def weather_code_label(value: object) -> str:
    """Return a readable Open-Meteo weather-code label."""
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "Weather unavailable"

    return WEATHER_CODE_LABELS.get(code, f"Weather code {code}")


def extract_nearest_weather_hour(payload: dict, game_utc: pd.Timestamp, ballpark_name: str) -> dict:
    """Extract the weather hour nearest to game time."""
    hourly = payload.get("hourly", {}) or {}
    times = hourly.get("time", []) or []

    if not times:
        return {}

    weather_times = pd.Series(pd.to_datetime(times, errors="coerce", utc=True, format="ISO8601"))
    valid_times = weather_times[weather_times.notna()]

    if valid_times.empty:
        return {}

    target = game_utc.tz_convert(ZoneInfo("UTC"))
    deltas = (valid_times - target).abs()
    index = int(deltas.idxmin())

    def hourly_value(key: str) -> object:
        values = hourly.get(key, []) or []
        return values[index] if index < len(values) else None

    temperature = safe_float(hourly_value("temperature_2m"), default=None)
    wind_speed = safe_float(hourly_value("wind_speed_10m"), default=None)
    precipitation = safe_float(hourly_value("precipitation_probability"), default=None)
    weather_code = hourly_value("weather_code")

    return {
        "ballpark": ballpark_name,
        "time": weather_times.iloc[index].isoformat(),
        "temperature": temperature,
        "wind_speed": wind_speed,
        "wind_direction": hourly_value("wind_direction_10m"),
        "precipitation_probability": precipitation,
        "weather_code": weather_code,
        "label": weather_code_label(weather_code),
    }


def format_weather_summary(weather: dict) -> str:
    """Format weather data for non-UI diagnostics."""
    if not weather:
        return ""

    parts = []
    temperature = weather.get("temperature")
    wind_speed = weather.get("wind_speed")
    precipitation = weather.get("precipitation_probability")
    label = str(weather.get("label", "") or "")

    if temperature is not None:
        parts.append(f"{float(temperature):.0f}°F")

    if label:
        parts.append(label)

    if wind_speed is not None:
        parts.append(f"Wind {float(wind_speed):.0f} mph")

    if precipitation is not None:
        parts.append(f"Rain {float(precipitation):.0f}%")

    return " / ".join(parts)


def player_headshot_url(player_id: object) -> str:
    """Return a public MLB player headshot URL."""
    player_id_text = normalize_player_id(player_id)

    if not player_id_text:
        return ""

    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_180,d_people:generic:headshot:67:current.png,q_auto:best/"
        f"v1/people/{player_id_text}/headshot/67/current"
    )


def render_team_logo(team_id: object, team_name: str) -> str:
    """Return escaped team-logo HTML."""
    logo_url = team_logo_url(team_id)

    if not logo_url:
        return ""

    return (
        '<span class="mlb-logo-frame">'
        f'<img class="mlb-logo" src="{html.escape(logo_url)}" '
        f'alt="{html.escape(team_name)} logo">'
        "</span>"
    )


def get_live_team_score(feed: dict, side: str) -> int | None:
    """Return live score from the game feed when present."""
    value = get_nested(feed, ["liveData", "linescore", "teams", side, "runs"])

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_live_team_total(feed: dict, side: str, stat_name: str) -> int | None:
    """Return one live team total from the linescore."""
    value = get_nested(feed, ["liveData", "linescore", "teams", side, stat_name])

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> int | None:
    """Parse an integer value from MLB feed payloads."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_out_count(outs: object) -> str:
    """Format an out count."""
    parsed = safe_int(outs)

    if parsed is None:
        return ""

    return "1 out" if parsed == 1 else f"{parsed} outs"


def clamp_float(value: float, lower: float, upper: float) -> float:
    """Clamp a float between two bounds."""
    return min(max(value, lower), upper)


def format_count_summary(count: dict) -> str:
    """Format the live balls and strikes count."""
    balls = safe_int(count.get("balls")) if isinstance(count, dict) else None
    strikes = safe_int(count.get("strikes")) if isinstance(count, dict) else None

    if balls is not None and strikes is not None:
        return f"Count {balls}-{strikes}"

    return ""


def get_live_count_state(feed: dict) -> dict:
    """Return the current live count from current play or linescore."""
    current_count = get_nested(feed, ["liveData", "plays", "currentPlay", "count"], default={})
    linescore = get_nested(feed, ["liveData", "linescore"], default={}) or {}
    count = current_count if isinstance(current_count, dict) else {}
    return {
        "balls": count.get("balls", linescore.get("balls")),
        "strikes": count.get("strikes", linescore.get("strikes")),
        "outs": count.get("outs", linescore.get("outs")),
    }


def get_live_base_state(feed: dict) -> dict:
    """Return current occupied-base state from the MLB live feed."""
    offense = get_nested(feed, ["liveData", "linescore", "offense"], default={})

    if not isinstance(offense, dict):
        offense = {}

    return {
        "first": bool(offense.get("first")),
        "second": bool(offense.get("second")),
        "third": bool(offense.get("third")),
    }


def get_live_inning_state(feed: dict) -> dict:
    """Return current inning state from the MLB live feed."""
    linescore = get_nested(feed, ["liveData", "linescore"], default={})

    if not isinstance(linescore, dict):
        linescore = {}

    return {
        "inning": safe_int(linescore.get("currentInning")),
        "half": str(linescore.get("inningHalf") or "").strip().lower(),
        "state": str(linescore.get("inningState") or "").strip().lower(),
    }


def base_run_expectancy(base_state: dict, outs: object) -> float:
    """Return a small current-inning run-expectancy proxy."""
    value = 0.0

    if base_state.get("first"):
        value += 0.28
    if base_state.get("second"):
        value += 0.45
    if base_state.get("third"):
        value += 0.65

    parsed_outs = safe_int(outs)

    if parsed_outs is not None:
        value *= max(0.35, 1 - (0.22 * parsed_outs))

    return value


def logit_probability(probability: float) -> float:
    """Return the logit for a probability."""
    probability = clamp_float(probability, 0.01, 0.99)
    return math.log(probability / (1 - probability))


def inverse_logit(value: float) -> float:
    """Return probability from logit value."""
    return 1 / (1 + math.exp(-value))


def estimate_home_probability_from_state(
    pregame_home_probability: float,
    home_score: object,
    away_score: object,
    inning_state: dict,
    count_state: dict,
    base_state: dict | None = None,
) -> float | None:
    """Estimate home win probability from one game state."""
    home_runs = safe_int(home_score)
    away_runs = safe_int(away_score)

    if home_runs is None or away_runs is None:
        return None

    base_state = base_state or {}
    pregame_home = clamp_float(float(pregame_home_probability), 0.01, 0.99)
    inning = safe_int(inning_state.get("inning")) or 1
    half = str(inning_state.get("half") or "").lower()
    outs = safe_int(count_state.get("outs")) or 0
    inning = max(1, min(inning, 12))
    half_index = (inning - 1) * 2 + (1 if half == "bottom" else 0)
    elapsed_half_innings = clamp_float(half_index + (outs / 3), 0.0, 18.0)
    elapsed_fraction = clamp_float(elapsed_half_innings / 18.0, 0.0, 1.0)
    remaining_fraction = clamp_float(1.0 - elapsed_fraction, 0.04, 1.0)
    raw_margin = home_runs - away_runs
    margin = raw_margin
    base_adjustment = base_run_expectancy(base_state, outs)

    if half == "top":
        margin -= base_adjustment
    elif half == "bottom":
        margin += base_adjustment

    score_weight = 0.24 + (1.28 * (elapsed_fraction**1.85))
    pregame_weight = 0.08 + (0.55 * (remaining_fraction**1.25))
    probability = inverse_logit((logit_probability(pregame_home) * pregame_weight) + (margin * score_weight))

    score_margin = abs(raw_margin)

    if score_margin <= 1 and remaining_fraction >= 0.16:
        one_run_cap = 0.62 + (0.11 * elapsed_fraction)
        probability = clamp_float(probability, 1 - one_run_cap, one_run_cap)
    elif score_margin <= 2 and remaining_fraction >= 0.40:
        two_run_cap = 0.68 + (0.12 * elapsed_fraction)
        probability = clamp_float(probability, 1 - two_run_cap, two_run_cap)
    elif score_margin <= 3 and remaining_fraction >= 0.45:
        three_run_cap = 0.78 + (0.10 * elapsed_fraction)
        probability = clamp_float(probability, 1 - three_run_cap, three_run_cap)

    return clamp_float(probability, 0.01, 0.99)


def estimate_live_home_probability(
    game: pd.Series,
    home_score: object,
    away_score: object,
    inning_state: dict,
    count_state: dict,
    base_state: dict,
    feed_is_live: bool,
    feed_is_final: bool = False,
) -> float | None:
    """Estimate in-game home win probability from live score and base state."""
    home_runs = safe_int(home_score)
    away_runs = safe_int(away_score)

    if home_runs is None or away_runs is None:
        return None

    if feed_is_final or is_final_status_value(game.get("STATUS", ""), game.get("STATUS_CODE", ""), game.get("IS_FINAL")):
        if home_runs == away_runs:
            return 0.5

        return 1.0 if home_runs > away_runs else 0.0

    if not feed_is_live:
        return None

    pregame_home = safe_float(game.get("HOME_WIN_PROBABILITY"), default=None)

    if pregame_home is None:
        pregame_home = 0.5

    return estimate_home_probability_from_state(
        pregame_home_probability=float(pregame_home),
        home_score=home_runs,
        away_score=away_runs,
        inning_state=inning_state,
        count_state=count_state,
        base_state=base_state,
    )


def format_matchup_names(play: dict) -> str:
    """Return batter vs pitcher for one play."""
    batter = str(get_nested(play, ["matchup", "batter", "fullName"], "") or "").strip()
    pitcher = str(get_nested(play, ["matchup", "pitcher", "fullName"], "") or "").strip()

    if batter and pitcher:
        return f"{batter} vs {pitcher}"

    return batter or pitcher


def extract_person_summary(person: object) -> dict:
    """Return a small player summary from an MLB person payload."""
    if not isinstance(person, dict):
        return {}

    player_id = normalize_player_id(person.get("id"))
    name = str(person.get("fullName") or person.get("name") or "").strip()

    if not name:
        return {}

    return {
        "id": player_id,
        "name": name,
        "photo_url": player_headshot_url(player_id),
    }


def extract_current_batter_pitcher(feed: dict) -> dict:
    """Return current batter-vs-pitcher data from the MLB live feed."""
    current_play = get_nested(feed, ["liveData", "plays", "currentPlay"], default={})

    if not isinstance(current_play, dict):
        return {}

    batter = extract_person_summary(get_nested(current_play, ["matchup", "batter"], {}))
    pitcher = extract_person_summary(get_nested(current_play, ["matchup", "pitcher"], {}))
    half = str(get_nested(current_play, ["about", "halfInning"], "") or "").strip().lower()
    batter_side = "home" if half == "bottom" else "away" if half == "top" else ""
    pitcher_side = "away" if half == "bottom" else "home" if half == "top" else ""

    if batter and batter_side:
        boxscore_batter = find_boxscore_player(feed, batter_side, batter.get("id"), batter.get("name", ""))
        batter_average = display_stat(get_season_stat_group(boxscore_batter, "batting").get("avg"))

        if batter_average != "-":
            batter["avg"] = batter_average

    if pitcher and pitcher_side:
        boxscore_pitcher = find_boxscore_player(feed, pitcher_side, pitcher.get("id"), pitcher.get("name", ""))
        game_pitching = get_game_stat_group(boxscore_pitcher, "pitching")
        pitch_count = (
            game_pitching.get("pitchesThrown")
            or game_pitching.get("numberOfPitches")
            or game_pitching.get("pitchCount")
            or game_pitching.get("pitches")
        )
        pitch_count_label = format_whole_number(pitch_count)

        if pitch_count_label != "-":
            pitcher["pitch_count"] = pitch_count_label

    if not batter and not pitcher:
        return {}

    return {
        "batter": batter,
        "pitcher": pitcher,
        "context": format_play_context(current_play),
    }


def format_play_context(play: dict) -> str:
    """Format inning context for one MLB play."""
    about = play.get("about", {}) or {}
    half = str(about.get("halfInning", "") or "").strip().title()
    inning = str(about.get("inningOrdinal", "") or about.get("inning", "") or "").strip()

    if half and inning:
        return f"{half} {inning}"

    return inning or half


def is_pitch_event(event: dict) -> bool:
    """Return whether one play event is a pitch."""
    event_type = event.get("type")

    if isinstance(event_type, dict):
        event_type_text = str(event_type.get("code") or event_type.get("description") or "")
    else:
        event_type_text = str(event_type or "")

    return (
        event.get("isPitch") is True
        or event_type_text.lower() == "pitch"
        or isinstance(event.get("pitchData"), dict)
        or event.get("pitchNumber") is not None
    )


def format_pitch_speed(event: dict) -> str:
    """Format pitch speed when available."""
    speed = safe_float(get_nested(event, ["pitchData", "startSpeed"], None), default=None)

    if speed is None:
        return ""

    return f"{float(speed):.0f} mph"


def format_pitch_type(event: dict) -> str:
    """Format pitch type when available."""
    pitch_type = str(
        get_nested(event, ["details", "type", "description"], "")
        or get_nested(event, ["details", "type", "code"], "")
        or ""
    ).strip()
    pitch_type_lookup = {
        "FF": "Four-Seam Fastball",
        "FT": "Two-Seam Fastball",
        "SI": "Sinker",
        "FC": "Cutter",
        "SL": "Slider",
        "ST": "Sweeper",
        "CH": "Changeup",
        "CU": "Curveball",
        "KC": "Knuckle Curve",
        "FS": "Splitter",
        "FO": "Forkball",
        "KN": "Knuckleball",
        "EP": "Eephus",
        "SC": "Screwball",
    }
    return pitch_type_lookup.get(pitch_type.upper(), pitch_type)


def format_pitch_summary(pitch_type: str, speed: str) -> str:
    """Format a readable pitch label."""
    if pitch_type and speed:
        return f"{speed} {pitch_type}"

    if pitch_type:
        return pitch_type

    if speed:
        return speed

    return "Pitch"


def extract_recent_pitches(feed: dict, limit: int = 80) -> list[dict]:
    """Extract recent pitch-by-pitch rows from the MLB live feed."""
    all_plays = get_nested(feed, ["liveData", "plays", "allPlays"], default=[]) or []
    current_play = get_nested(feed, ["liveData", "plays", "currentPlay"], default={})
    source_plays = []

    if isinstance(current_play, dict) and current_play:
        source_plays.append(current_play)

    source_plays.extend([play for play in reversed(all_plays) if isinstance(play, dict)])
    rows = []
    seen = set()

    for play in source_plays:
        context = format_play_context(play)
        matchup = format_matchup_names(play)
        at_bat_index = str(get_nested(play, ["about", "atBatIndex"], "") or "")
        events = play.get("playEvents", []) or []

        for event in reversed(events):
            if not isinstance(event, dict) or not is_pitch_event(event):
                continue

            details = event.get("details", {}) or {}
            description = str(
                details.get("description")
                or get_nested(details, ["call", "description"], "")
                or "Pitch"
            ).strip()
            pitch_number = str(event.get("pitchNumber") or event.get("index") or "")
            dedupe_key = (at_bat_index, pitch_number, description)

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            count_label = format_count_summary(event.get("count", {}) or {})
            pitch_type = format_pitch_type(event)
            speed = format_pitch_speed(event)
            summary = format_pitch_summary(pitch_type, speed)
            meta = " / ".join(
                [
                    value
                    for value in [
                        context,
                        matchup,
                        f"Pitch {pitch_number}" if pitch_number else "",
                        description,
                        count_label,
                    ]
                    if value
                ]
            )
            rows.append(
                {
                    "DESCRIPTION": description,
                    "SUMMARY": summary,
                    "META": meta,
                    "COUNT": count_label,
                    "PITCH_TYPE": pitch_type,
                    "SPEED": speed,
                }
            )

            if len(rows) >= limit:
                return rows

    return rows


def extract_recent_plays(feed: dict, limit: int = 60) -> list[dict]:
    """Extract recent play-by-play rows from one MLB live feed."""
    plays = get_nested(feed, ["liveData", "plays", "allPlays"], default=[]) or []
    rows = []

    for play in reversed(plays):
        if not isinstance(play, dict):
            continue

        result = play.get("result", {}) or {}
        count = play.get("count", {}) or {}
        description = str(result.get("description", "") or "").strip()

        if not description:
            continue

        score_bits = []
        away_score = result.get("awayScore")
        home_score = result.get("homeScore")

        if away_score is not None and home_score is not None:
            score_bits.append(f"{away_score}-{home_score}")

        outs = count.get("outs")

        if outs is not None:
            score_bits.append(f"{outs} out")

        rows.append(
            {
                "CONTEXT": format_play_context(play),
                "EVENT": str(result.get("event", "") or "").strip(),
                "DESCRIPTION": description,
                "SCORE": " / ".join(score_bits),
            }
        )

        if len(rows) >= limit:
            break

    return rows


def format_inning_display(feed: dict) -> str:
    """Return the current inning label for a live game."""
    linescore = get_nested(feed, ["liveData", "linescore"], default={}) or {}
    state = str(linescore.get("inningState", "") or "").strip()
    half = str(linescore.get("inningHalf", "") or "").strip()
    inning = str(linescore.get("currentInningOrdinal", "") or "").strip()

    if inning and state:
        return f"{state} {inning}"

    if inning and half:
        return f"{half} {inning}"

    return inning or state or half


def get_season_stat_group(player: dict, group: str) -> dict:
    """Return one player's season batting or pitching stats."""
    season_stats = player.get("seasonStats", {}) or {}
    return season_stats.get(group, {}) or {}


def get_game_stat_group(player: dict, group: str) -> dict:
    """Return one player's current-game batting or pitching stats."""
    game_stats = player.get("stats", {}) or {}
    return game_stats.get(group, {}) or {}


def current_mlb_season() -> int:
    """Return the active MLB season year."""
    return int(pd.Timestamp.now(tz=ZoneInfo("America/New_York")).year)


def short_division_name(division: dict) -> str:
    """Return compact MLB division text."""
    division_id = safe_int((division or {}).get("id"))

    if division_id in MLB_DIVISION_NAMES_BY_ID:
        return MLB_DIVISION_NAMES_BY_ID[division_id]

    name = str((division or {}).get("name", "") or "").strip()

    for source, replacement in {"American League": "AL", "National League": "NL"}.items():
        name = name.replace(source, replacement)

    return name or "Division"


def standings_split_record(team_record: dict, split_type: str) -> str:
    """Return a split record from one MLB standings team row."""
    split_records = get_nested(team_record, ["records", "splitRecords"], default=[]) or []

    for split in split_records:
        if not isinstance(split, dict):
            continue

        if str(split.get("type", "") or "").lower() == split_type.lower():
            wins = safe_int(split.get("wins"))
            losses = safe_int(split.get("losses"))

            if wins is not None and losses is not None:
                return f"{wins}-{losses}"

    return "-"


def format_games_back(value: object, leader_zero: bool = True) -> str:
    """Format standings games-back text."""
    text = display_stat(value)

    if text in {"0.0", "0.00"}:
        return "0"

    if text == "-":
        return "0" if leader_zero else "-"

    return text


def format_run_differential(value: object, runs_for: object = None, runs_against: object = None) -> str:
    """Format run differential with a plus sign when possible."""
    parsed = safe_int(value)

    if parsed is None:
        scored = safe_int(runs_for)
        allowed = safe_int(runs_against)

        if scored is not None and allowed is not None:
            parsed = scored - allowed

    if parsed is None:
        return "-"

    return f"{parsed:+d}"


def build_standings_rows(payload: dict) -> list[dict]:
    """Normalize MLB standings payload into table rows."""
    rows = []

    for record in payload.get("records", []) or []:
        if not isinstance(record, dict):
            continue

        league = record.get("league", {}) or {}
        division = record.get("division", {}) or {}
        league_id = safe_int(league.get("id"))
        league_name = MLB_LEAGUE_NAMES_BY_ID.get(
            league_id,
            str(league.get("name", "") or "League").strip(),
        )
        division_name = short_division_name(division)

        for team_record in record.get("teamRecords", []) or []:
            if not isinstance(team_record, dict):
                continue

            team = team_record.get("team", {}) or {}
            team_name = str(team.get("name", "") or "").strip()

            if not team_name:
                continue

            rows.append(
                {
                    "League": league_name,
                    "Division": division_name,
                    "Team ID": team.get("id"),
                    "Rank": display_stat(team_record.get("divisionRank")),
                    "Team": team_name,
                    "W": display_stat(team_record.get("wins")),
                    "L": display_stat(team_record.get("losses")),
                    "Pct": display_stat(team_record.get("winningPercentage")),
                    "GB": format_games_back(team_record.get("gamesBack")),
                    "WC GB": format_games_back(team_record.get("wildCardGamesBack"), leader_zero=False),
                    "Last 10": standings_split_record(team_record, "lastTen"),
                    "Streak": display_stat(
                        get_nested(team_record, ["streak", "streakCode"], None)
                        or get_nested(team_record, ["streak", "streakDescription"], None)
                    ),
                    "Run Diff": format_run_differential(
                        team_record.get("runDifferential"),
                        team_record.get("runsScored"),
                        team_record.get("runsAllowed"),
                    ),
                    "League Rank": display_stat(team_record.get("leagueRank")),
                    "Overall Rank": display_stat(team_record.get("sportRank")),
                }
            )

    return rows


def extract_season_stat_from_payload(payload: dict) -> dict:
    """Extract the stat dictionary from an MLB people stats payload."""
    stats = payload.get("stats", []) or []

    for stat_group in stats:
        splits = stat_group.get("splits", []) or []

        if splits:
            return splits[0].get("stat", {}) or {}

    return {}


def extract_pitching_stat_row(player_id: object, player: dict | None = None) -> dict:
    """Build a normalized pitching stat row from live feed or player endpoint data."""
    player = player or {}
    pitching = get_season_stat_group(player, "pitching")

    if not pitching:
        player_id_text = normalize_player_id(player_id)
        pitching = extract_season_stat_from_payload(
            load_player_season_stats(player_id_text, current_mlb_season(), "pitching")
        )

    return {
        "ERA": display_stat(pitching.get("era")),
        "WHIP": display_stat(pitching.get("whip")),
        "SO": format_whole_number(pitching.get("strikeOuts")),
        "W": format_whole_number(pitching.get("wins")),
        "L": format_whole_number(pitching.get("losses")),
        "GS": format_whole_number(pitching.get("gamesStarted")),
        "IP": display_stat(pitching.get("inningsPitched")),
    }


def get_probable_pitcher_payload_from_feed(feed: dict, side: str) -> dict:
    """Return probable pitcher payload from live feed when present."""
    return get_nested(feed, ["gameData", "probablePitchers", side], default={}) or {}


def find_boxscore_player(feed: dict, side: str, player_id: object = None, name: str = "") -> dict:
    """Find a boxscore player by id or name."""
    players = get_nested(feed, ["liveData", "boxscore", "teams", side, "players"], default={}) or {}
    player_id_text = normalize_player_id(player_id)
    name_text = str(name or "").strip().lower()

    if player_id_text:
        player = players.get(f"ID{player_id_text}") or players.get(player_id_text)

        if isinstance(player, dict):
            return player

    for player in players.values():
        if not isinstance(player, dict):
            continue

        player_name = str((player.get("person", {}) or {}).get("fullName", "") or "").strip().lower()

        if name_text and player_name == name_text:
            return player

    return {}


def build_probable_pitcher_profile(game: pd.Series, feed: dict, side: str) -> dict:
    """Build pitcher profile data for a probable starter."""
    prefix = "AWAY" if side == "away" else "HOME"
    feed_pitcher = get_probable_pitcher_payload_from_feed(feed, side)
    pitcher_id = (
        normalize_player_id(game.get(f"{prefix}_PROBABLE_PITCHER_ID"))
        or normalize_player_id(feed_pitcher.get("id"))
    )
    pitcher_name = (
        str(game.get(f"{prefix}_PROBABLE_PITCHER", "") or "").strip()
        or str(feed_pitcher.get("fullName", "") or "").strip()
    )
    boxscore_player = find_boxscore_player(feed, side, pitcher_id, pitcher_name)

    if not pitcher_id and boxscore_player:
        pitcher_id = normalize_player_id((boxscore_player.get("person", {}) or {}).get("id"))

    if not pitcher_name and boxscore_player:
        pitcher_name = str((boxscore_player.get("person", {}) or {}).get("fullName", "") or "").strip()

    if not pitcher_name:
        pitcher_name = "Probable pitcher TBD"

    stats = extract_pitching_stat_row(pitcher_id, boxscore_player)
    return {
        "PLAYER_ID": pitcher_id,
        "PLAYER_NAME": pitcher_name,
        "POSITION": "SP",
        "BATTING_SLOT": None,
        "LINEUP_LABEL": "SP",
        "TEAM_SIDE": side,
        "IS_STARTING_PITCHER": True,
        "PHOTO_URL": player_headshot_url(pitcher_id),
        "AVG": "-",
        "OPS": "-",
        "HR": "-",
        "RBI": "-",
        "SB": "-",
        "GAME_H": "-",
        "GAME_AB": "-",
        "GAME_RBI": "-",
        "GAME_IP": "-",
        "GAME_SO": "-",
        **stats,
    }


def pitcher_stat_line(profile: dict) -> str:
    """Return compact pitcher stat text."""
    parts = [
        f"ERA {display_stat(profile.get('ERA'))}",
        f"WHIP {display_stat(profile.get('WHIP'))}",
        f"SO {display_stat(profile.get('SO'))}",
        f"W-L {display_stat(profile.get('W'))}-{display_stat(profile.get('L'))}",
    ]
    return " / ".join(parts)


def render_pitcher_profile_card(profile: dict, team_name: str) -> None:
    """Render one probable pitcher profile card."""
    st.html(build_pitcher_profile_card_html(profile, team_name))


def build_pitcher_profile_card_html(profile: dict, team_name: str) -> str:
    """Return one probable pitcher profile card as HTML."""
    name = str(profile.get("PLAYER_NAME", "") or "Probable pitcher TBD")
    photo_url = str(profile.get("PHOTO_URL", "") or "")
    photo_html = (
        f'<img class="pitcher-photo" src="{html.escape(photo_url)}" alt="{html.escape(name)} headshot">'
        if photo_url
        else '<div class="pitcher-photo"></div>'
    )
    stat_labels = [
        ("ERA", profile.get("ERA")),
        ("WHIP", profile.get("WHIP")),
        ("SO", profile.get("SO")),
        ("W-L", f"{display_stat(profile.get('W'))}-{display_stat(profile.get('L'))}"),
        ("GS", profile.get("GS")),
        ("IP", profile.get("IP")),
    ]
    stats_html = "".join(
        f'<span class="pitcher-stat">{html.escape(label)} {html.escape(display_stat(value))}</span>'
        for label, value in stat_labels
        if display_stat(value) != "-"
    )

    if not stats_html:
        stats_html = '<span class="pitcher-stat">Stats unavailable</span>'

    return f"""
        <div class="pitcher-card" style="--team-color: {html.escape(get_team_color(team_name))};">
            {photo_html}
            <div>
                <div class="dashboard-label">{html.escape(team_name)} starter</div>
                <div class="pitcher-name">{html.escape(name)}</div>
                <div class="pitcher-stat-grid">{stats_html}</div>
            </div>
        </div>
    """


def render_probable_pitcher_matchup(game: pd.Series, feed: dict) -> None:
    """Render away/home probable starter cards."""
    away_team = str(game.get("AWAY_TEAM", "") or "Away")
    home_team = str(game.get("HOME_TEAM", "") or "Home")
    away_profile = build_probable_pitcher_profile(game, feed, "away")
    home_profile = build_probable_pitcher_profile(game, feed, "home")
    away_col, home_col = st.columns(2)

    with away_col:
        render_pitcher_profile_card(away_profile, away_team)

    with home_col:
        render_pitcher_profile_card(home_profile, home_team)


def add_starting_pitcher_to_lineup(lineup: list[dict], pitcher_profile: dict) -> list[dict]:
    """Prepend the starting pitcher to a projected/official lineup display."""
    pitcher_name = str(pitcher_profile.get("PLAYER_NAME", "") or "").strip()

    if not pitcher_name or pitcher_name == "Probable pitcher TBD":
        return lineup

    existing_ids = {normalize_player_id(player.get("PLAYER_ID")) for player in lineup}
    pitcher_id = normalize_player_id(pitcher_profile.get("PLAYER_ID"))

    if pitcher_id and pitcher_id in existing_ids:
        return lineup

    return [pitcher_profile, *lineup]


def player_star_score(row: dict) -> float:
    """Score players for a compact stars list."""
    avg = safe_float(row.get("AVG"), 0.0) or 0.0
    ops = safe_float(row.get("OPS"), 0.0) or 0.0
    hr = safe_float(row.get("HR"), 0.0) or 0.0
    rbi = safe_float(row.get("RBI"), 0.0) or 0.0
    era = safe_float(row.get("ERA"), None)
    whip = safe_float(row.get("WHIP"), None)
    strikeouts = safe_float(row.get("SO"), 0.0) or 0.0
    wins = safe_float(row.get("W"), 0.0) or 0.0
    batting_score = (ops * 1000) + (avg * 100) + (hr * 4) + (rbi * 0.35)
    pitching_score = (strikeouts * 0.2) + (wins * 3)

    if era is not None:
        pitching_score += max(0.0, 20 - (era * 3))

    if whip is not None:
        pitching_score += max(0.0, 5 - whip)

    lineup_slot = row.get("BATTING_SLOT")

    if lineup_slot:
        batting_score += max(0, 10 - int(lineup_slot))

    return max(batting_score, pitching_score)


def extract_team_players(feed: dict, side: str) -> list[dict]:
    """Extract player stat rows for one team from a live feed box score."""
    box_team = get_nested(feed, ["liveData", "boxscore", "teams", side], default={}) or {}
    players = box_team.get("players", {}) or {}
    game_players = feed.get("gameData", {}).get("players", {}) or {}
    batting_order = box_team.get("battingOrder", []) or []
    order_by_player_id = {
        normalize_player_id(player_id): index + 1
        for index, player_id in enumerate(batting_order)
    }
    rows = []

    for key, player in players.items():
        if not isinstance(player, dict):
            continue

        player_id = normalize_player_id((player.get("person", {}) or {}).get("id") or key)
        game_player = game_players.get(f"ID{player_id}", {}) or {}
        person = player.get("person", {}) or game_player
        name = str(person.get("fullName", "") or game_player.get("fullName", "") or "").strip()

        if not name:
            continue

        position = (
            (player.get("position", {}) or {}).get("abbreviation")
            or (person.get("primaryPosition", {}) or {}).get("abbreviation")
            or (game_player.get("primaryPosition", {}) or {}).get("abbreviation")
            or ""
        )
        batting = get_season_stat_group(player, "batting")
        pitching = get_season_stat_group(player, "pitching")
        game_batting = get_game_stat_group(player, "batting")
        game_pitching = get_game_stat_group(player, "pitching")
        batting_slot = format_batting_slot(player.get("battingOrder"))

        if batting_slot is None:
            batting_slot = order_by_player_id.get(player_id)

        row = {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": name,
            "POSITION": str(position or "").strip(),
            "BATTING_SLOT": batting_slot,
            "AVG": display_stat(batting.get("avg")),
            "OPS": display_stat(batting.get("ops")),
            "HR": format_whole_number(batting.get("homeRuns")),
            "RBI": format_whole_number(batting.get("rbi")),
            "SB": format_whole_number(batting.get("stolenBases")),
            "ERA": display_stat(pitching.get("era")),
            "WHIP": display_stat(pitching.get("whip")),
            "SO": format_whole_number(pitching.get("strikeOuts")),
            "W": format_whole_number(pitching.get("wins")),
            "L": format_whole_number(pitching.get("losses")),
            "GS": format_whole_number(pitching.get("gamesStarted")),
            "IP": display_stat(pitching.get("inningsPitched")),
            "GAME_R": format_whole_number(game_batting.get("runs")),
            "GAME_H": format_whole_number(game_batting.get("hits")),
            "GAME_AB": format_whole_number(game_batting.get("atBats")),
            "GAME_RBI": format_whole_number(game_batting.get("rbi")),
            "GAME_BB": format_whole_number(game_batting.get("baseOnBalls")),
            "GAME_HR": format_whole_number(game_batting.get("homeRuns")),
            "GAME_SB": format_whole_number(game_batting.get("stolenBases")),
            "GAME_IP": display_stat(game_pitching.get("inningsPitched")),
            "GAME_P_H": format_whole_number(game_pitching.get("hits")),
            "GAME_ER": format_whole_number(game_pitching.get("earnedRuns")),
            "GAME_P_BB": format_whole_number(game_pitching.get("baseOnBalls")),
            "GAME_SO": format_whole_number(game_pitching.get("strikeOuts")),
            "TEAM_SIDE": side,
        }
        row["STAR_SCORE"] = player_star_score(row)
        rows.append(row)

    return rows


def get_star_players(feed: dict, side: str, limit: int = 5) -> list[dict]:
    """Return top player rows for one team."""
    players = extract_team_players(feed, side)

    return sorted(
        players,
        key=lambda row: (float(row.get("STAR_SCORE", 0.0)), bool(row.get("BATTING_SLOT"))),
        reverse=True,
    )[:limit]


def get_lineup_players(feed: dict, side: str, limit: int = 9) -> tuple[list[dict], str]:
    """Return official lineup rows, or a best available fallback."""
    players = extract_team_players(feed, side)
    lineup = [player for player in players if player.get("BATTING_SLOT")]

    if lineup:
        return sorted(lineup, key=lambda row: int(row.get("BATTING_SLOT") or 99))[:limit], "Official Lineup"

    fallback = [
        player
        for player in players
        if str(player.get("POSITION", "")).upper() not in {"P", "SP", "RP"}
    ]

    return sorted(fallback, key=lambda row: float(row.get("STAR_SCORE", 0.0)), reverse=True)[:limit], "Projected Lineup"


def player_stat_line(player: dict) -> str:
    """Return a compact player stat line."""
    hitter_stats = [
        f"AVG {display_stat(player.get('AVG'))}",
        f"OPS {display_stat(player.get('OPS'))}",
        f"HR {display_stat(player.get('HR'))}",
        f"RBI {display_stat(player.get('RBI'))}",
    ]
    pitcher_stats = [
        f"ERA {display_stat(player.get('ERA'))}",
        f"WHIP {display_stat(player.get('WHIP'))}",
        f"SO {display_stat(player.get('SO'))}",
        f"W-L {display_stat(player.get('W'))}-{display_stat(player.get('L'))}",
    ]
    has_pitching = any(display_stat(player.get(key)) != "-" for key in ["ERA", "WHIP", "SO"])

    if has_pitching and str(player.get("POSITION", "")).upper() in {"P", "SP", "RP", "CL"}:
        return " / ".join(pitcher_stats)

    return " / ".join(hitter_stats)


def player_today_stat_line(player: dict) -> str:
    """Return one player's current-game box-score line."""
    pitching_values = [
        ("IP", player.get("GAME_IP")),
        ("H", player.get("GAME_P_H")),
        ("ER", player.get("GAME_ER")),
        ("BB", player.get("GAME_P_BB")),
        ("SO", player.get("GAME_SO")),
    ]
    batting_values = [
        ("AB", player.get("GAME_AB")),
        ("R", player.get("GAME_R")),
        ("H", player.get("GAME_H")),
        ("RBI", player.get("GAME_RBI")),
        ("BB", player.get("GAME_BB")),
        ("HR", player.get("GAME_HR")),
        ("SB", player.get("GAME_SB")),
    ]
    has_pitching = any(display_stat(value) != "-" for _, value in pitching_values)
    has_batting = any(display_stat(value) != "-" for _, value in batting_values)

    if has_pitching and str(player.get("POSITION", "")).upper() in {"P", "SP", "RP", "CL"}:
        return " / ".join(
            f"{label} {display_stat(value)}"
            for label, value in pitching_values
            if display_stat(value) != "-"
        )

    if has_batting:
        return " / ".join(
            f"{label} {display_stat(value)}"
            for label, value in batting_values
            if display_stat(value) != "-"
        )

    if has_pitching:
        return " / ".join(
            f"{label} {display_stat(value)}"
            for label, value in pitching_values
            if display_stat(value) != "-"
        )

    return "No game stats yet"


def sort_today_players(players: list[dict]) -> list[dict]:
    """Sort current-game player rows by lineup, then game involvement."""
    def sort_key(player: dict) -> tuple[int, int, str]:
        batting_slot = player.get("BATTING_SLOT")
        has_today_stats = player_today_stat_line(player) != "No game stats yet"

        if batting_slot:
            try:
                return (0, int(batting_slot), str(player.get("PLAYER_NAME", "")))
            except (TypeError, ValueError):
                return (0, 99, str(player.get("PLAYER_NAME", "")))

        return (1 if has_today_stats else 2, 99, str(player.get("PLAYER_NAME", "")))

    return sorted(players, key=sort_key)


def get_optional_team_strength(team_name: str) -> pd.Series | None:
    """Return one strength row, or None if local team data is unavailable."""
    strength = load_team_strength()

    if strength.empty or "TEAM_NAME" not in strength.columns:
        return None

    rows = strength[strength["TEAM_NAME"].str.lower().eq(team_name.lower())]

    if rows.empty:
        return None

    return rows.iloc[0]


def format_team_percent(value: object) -> str:
    """Format a team percentage stat."""
    parsed = safe_float(value, default=None)

    if parsed is None:
        return "-"

    return f"{parsed:.0%}"


def format_team_decimal(value: object, digits: int = 1) -> str:
    """Format a team decimal stat."""
    parsed = safe_float(value, default=None)

    if parsed is None:
        return "-"

    return f"{parsed:.{digits}f}"


def build_team_stats_table(game: pd.Series, snapshot: dict) -> pd.DataFrame:
    """Build away/home team stats for the game-detail popup."""
    feed = snapshot.get("FEED", {}) or {}
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    away_strength = get_optional_team_strength(away_team)
    home_strength = get_optional_team_strength(home_team)

    def strength_value(row: pd.Series | None, column: str) -> object:
        if row is None or column not in row.index:
            return None

        return row[column]

    rows = [
        {
            "Stat": "Score",
            away_team: format_score(snapshot.get("AWAY_SCORE")) or "-",
            home_team: format_score(snapshot.get("HOME_SCORE")) or "-",
        },
        {
            "Stat": "Hits",
            away_team: display_stat(get_live_team_total(feed, "away", "hits")),
            home_team: display_stat(get_live_team_total(feed, "home", "hits")),
        },
        {
            "Stat": "Errors",
            away_team: display_stat(get_live_team_total(feed, "away", "errors")),
            home_team: display_stat(get_live_team_total(feed, "home", "errors")),
        },
        {
            "Stat": "Season Win %",
            away_team: format_team_percent(strength_value(away_strength, "SEASON_WIN_PCT")),
            home_team: format_team_percent(strength_value(home_strength, "SEASON_WIN_PCT")),
        },
        {
            "Stat": "Elo",
            away_team: format_team_decimal(strength_value(away_strength, "ELO"), 0),
            home_team: format_team_decimal(strength_value(home_strength, "ELO"), 0),
        },
        {
            "Stat": "Run Diff/G",
            away_team: format_team_decimal(strength_value(away_strength, "SEASON_RUN_DIFF_PER_GAME"), 1),
            home_team: format_team_decimal(strength_value(home_strength, "SEASON_RUN_DIFF_PER_GAME"), 1),
        },
        {
            "Stat": "Runs/G",
            away_team: format_team_decimal(strength_value(away_strength, "SEASON_AVG_RUNS_FOR"), 1),
            home_team: format_team_decimal(strength_value(home_strength, "SEASON_AVG_RUNS_FOR"), 1),
        },
        {
            "Stat": "Runs Allowed/G",
            away_team: format_team_decimal(strength_value(away_strength, "SEASON_AVG_RUNS_AGAINST"), 1),
            home_team: format_team_decimal(strength_value(home_strength, "SEASON_AVG_RUNS_AGAINST"), 1),
        },
        {
            "Stat": "Last 10 Win %",
            away_team: format_team_percent(strength_value(away_strength, "ROLLING_WIN_PCT_10")),
            home_team: format_team_percent(strength_value(home_strength, "ROLLING_WIN_PCT_10")),
        },
        {
            "Stat": "Last 10 Run Diff",
            away_team: format_team_decimal(strength_value(away_strength, "ROLLING_RUN_DIFF_10"), 1),
            home_team: format_team_decimal(strength_value(home_strength, "ROLLING_RUN_DIFF_10"), 1),
        },
        {
            "Stat": "Rest",
            away_team: format_team_decimal(strength_value(away_strength, "DAYS_REST"), 0),
            home_team: format_team_decimal(strength_value(home_strength, "DAYS_REST"), 0),
        },
        {
            "Stat": "Probable Pitcher",
            away_team: display_stat(game.get("AWAY_PROBABLE_PITCHER")),
            home_team: display_stat(game.get("HOME_PROBABLE_PITCHER")),
        },
    ]
    return pd.DataFrame(rows)


def build_live_game_snapshot(game: pd.Series, include_pregame_feed: bool = False) -> dict:
    """Build the live-display payload for one schedule row."""
    game_pk = str(game.get("GAME_PK", "") or "").strip()
    should_fetch_feed = include_pregame_feed

    if game_pk and not should_fetch_feed:
        game_time = game_datetime_utc(game)
        now_utc = pd.Timestamp.now(tz=ZoneInfo("UTC"))
        near_live_window = (
            game_time is not None
            and now_utc - timedelta(hours=6) <= game_time <= now_utc + timedelta(minutes=5)
        )
        should_fetch_feed = is_live_schedule_game(game) or (
            near_live_window and not is_final_schedule_game(game)
        )

    feed = load_live_game_feed(game_pk) if game_pk and should_fetch_feed else {}
    live_status = str(get_nested(feed, ["gameData", "status", "detailedState"], "") or "").strip()
    live_status_code = str(get_nested(feed, ["gameData", "status", "statusCode"], "") or "").strip()
    abstract_state = str(get_nested(feed, ["gameData", "status", "abstractGameState"], "") or "").strip()
    away_score = get_live_team_score(feed, "away")
    home_score = get_live_team_score(feed, "home")
    plays = extract_recent_plays(feed)
    pitches = extract_recent_pitches(feed)
    feed_is_live = is_live_status_value(
        live_status or game.get("STATUS", ""),
        live_status_code or game.get("STATUS_CODE", ""),
        game.get("IS_LIVE"),
    )
    feed_is_final = is_final_status_value(
        live_status or game.get("STATUS", ""),
        live_status_code or game.get("STATUS_CODE", ""),
        game.get("IS_FINAL"),
    )
    count_state = get_live_count_state(feed) if feed_is_live else {}
    count_summary = format_count_summary(count_state) if feed_is_live else ""
    current_matchup = extract_current_batter_pitcher(feed) if feed_is_live else {}
    base_state = get_live_base_state(feed) if feed_is_live else {"first": False, "second": False, "third": False}
    inning_state = get_live_inning_state(feed) if feed_is_live else {}

    if away_score is None:
        away_score = game.get("AWAY_SCORE")

    if home_score is None:
        home_score = game.get("HOME_SCORE")

    live_home_probability = estimate_live_home_probability(
        game=game,
        home_score=home_score,
        away_score=away_score,
        inning_state=inning_state,
        count_state=count_state,
        base_state=base_state,
        feed_is_live=feed_is_live,
        feed_is_final=feed_is_final,
    )

    return {
        "GAME_PK": game_pk,
        "FEED": feed,
        "STATUS": live_status or str(game.get("STATUS", "Scheduled") or "Scheduled"),
        "ABSTRACT_STATE": abstract_state,
        "IS_FINAL": feed_is_final,
        "INNING": format_inning_display(feed),
        "AWAY_SCORE": away_score,
        "HOME_SCORE": home_score,
        "BALLS": safe_int(count_state.get("balls")),
        "STRIKES": safe_int(count_state.get("strikes")),
        "OUTS": safe_int(count_state.get("outs")),
        "COUNT_SUMMARY": count_summary,
        "BASE_STATE": base_state,
        "LIVE_HOME_WIN_PROBABILITY": live_home_probability,
        "CURRENT_MATCHUP": current_matchup,
        "LATEST_PLAY": plays[0]["DESCRIPTION"] if plays else "",
        "LATEST_PITCH": pitches[0] if pitches else {},
        "PLAYS": plays,
        "PITCHES": pitches,
    }


def render_section_kicker(label: str, note: str | None = None) -> None:
    """Render a compact section label."""
    text = html.escape(label)

    if note:
        text += f" / {html.escape(note)}"

    st.html(f'<div class="section-kicker">{text}</div>')


def get_team_color(team_name: str) -> str:
    """Return a stable accent color for a team."""
    index = sum((position + 1) * ord(char) for position, char in enumerate(team_name))
    return TEAM_COLOR_PALETTE[index % len(TEAM_COLOR_PALETTE)]


def get_team_id_from_strength(team_name: str) -> int | None:
    """Return MLB team id from local strength data."""
    row = get_optional_team_strength(team_name)

    if row is None or "TEAM_ID" not in row.index:
        return None

    try:
        return int(row["TEAM_ID"])
    except (TypeError, ValueError):
        return None


def render_dashboard_cards(cards: list[dict[str, str]]) -> None:
    """Render compact dashboard cards."""
    html_rows = []

    for card in cards:
        html_rows.append(
            f"""
            <div class="dashboard-card" style="--card-color: {html.escape(card.get('color', '#1d4ed8'))};">
                <div class="dashboard-label">{html.escape(str(card.get('label', '')))}</div>
                <div class="dashboard-value">{html.escape(str(card.get('value', '')))}</div>
                <div class="dashboard-note">{html.escape(str(card.get('note', '')))}</div>
            </div>
            """
        )

    st.html(f'<div class="dashboard-grid">{"".join(html_rows)}</div>')


def render_status_grid(cards: list[dict[str, str]]) -> None:
    """Render model/status cards."""
    html_rows = []

    for card in cards:
        html_rows.append(
            f"""
            <div class="mlb-status-card" style="--card-color: {html.escape(card.get('color', '#166534'))};">
                <div class="dashboard-label">{html.escape(str(card.get('label', '')))}</div>
                <div class="dashboard-value">{html.escape(str(card.get('value', '')))}</div>
                <div class="dashboard-note">{html.escape(str(card.get('note', '')))}</div>
            </div>
            """
        )

    st.html(f'<div class="mlb-status-grid">{"".join(html_rows)}</div>')


def get_game_confidence_label(probability: float) -> str:
    """Return a readable confidence label."""
    favorite_probability = max(probability, 1 - probability)

    if favorite_probability >= 0.68:
        return "High"
    if favorite_probability >= 0.58:
        return "Medium"
    if favorite_probability >= 0.53:
        return "Lean"

    return "Toss-up"


def estimate_prediction_interval(probability: float) -> tuple[float, float]:
    """Return a practical uncertainty interval around a displayed probability."""
    probability = min(max(probability, 0.01), 0.99)
    closeness = 1 - min(abs(max(probability, 1 - probability) - 0.5) * 2, 1)
    margin = 0.045 + (0.04 * closeness)
    return max(0.01, probability - margin), min(0.99, probability + margin)


def render_prediction_result_card(
    label: str,
    winner: str,
    probability: float,
    details: dict | None = None,
    note: str | None = None,
) -> None:
    """Render a visual MLB prediction result."""
    lower, upper = estimate_prediction_interval(probability)
    confidence = get_game_confidence_label(probability)
    team_color = get_team_color(winner)
    logo_url = team_logo_url(get_team_id_from_strength(winner))
    signals = [
        f"Range {lower:.0%}-{upper:.0%}",
        confidence,
    ]

    if details:
        if "model_probability" in details:
            signals.append(f"Win {float(details['model_probability']):.0%}")
        if "elo_probability" in details:
            signals.append(f"Power {float(details['elo_probability']):.0%}")

    explanation = []

    if details:
        explanation = details.get("explanation", []) or []

    signal_html = "".join(
        f'<span class="signal-pill">{html.escape(signal)}</span>'
        for signal in signals
    )
    explanation_html = (
        f'<div class="prediction-explain">Why: {html.escape(" / ".join(explanation))}</div>'
        if explanation
        else ""
    )
    logo_html = (
        f'<img class="prediction-logo" src="{html.escape(logo_url)}" alt="{html.escape(winner)} logo">'
        if logo_url
        else ""
    )
    note_html = f'<div class="prediction-sub">{html.escape(note)}</div>' if note else ""

    st.html(
        f"""
        <div class="prediction-card" style="--team-color: {html.escape(team_color)};">
            <div class="prediction-topline">
                <div class="dashboard-label">{html.escape(label)}</div>
                <span class="mlb-status">{html.escape(confidence)}</span>
            </div>
            <div class="prediction-main">
                {logo_html}
                <div>
                    <div class="prediction-winner">{html.escape(winner)}</div>
                    <div class="prediction-sub">Projected win probability with pricing confidence</div>
                    {note_html}
                </div>
                <div class="prediction-prob">{probability:.0%}</div>
            </div>
            <div class="probability-meter">
                <div class="probability-fill" style="--probability-width: {probability * 100:.1f}%;"></div>
            </div>
            <div class="signal-grid">{signal_html}</div>
            {explanation_html}
        </div>
        """
    )


def render_matchup_preview(team_a: str, team_b: str, label_a: str, label_b: str) -> None:
    """Render selected teams before a prediction is run."""
    team_a_logo = team_logo_url(get_team_id_from_strength(team_a))
    team_b_logo = team_logo_url(get_team_id_from_strength(team_b))
    team_a_logo_html = (
        f'<img class="preview-logo" src="{html.escape(team_a_logo)}" alt="{html.escape(team_a)} logo">'
        if team_a_logo
        else ""
    )
    team_b_logo_html = (
        f'<img class="preview-logo" src="{html.escape(team_b_logo)}" alt="{html.escape(team_b)} logo">'
        if team_b_logo
        else ""
    )
    st.html(
        f"""
        <div class="matchup-preview">
            <div class="preview-team-card" style="--team-color: {html.escape(get_team_color(team_a))};">
                {team_a_logo_html}
                <div>
                    <div class="preview-role">{html.escape(label_a)}</div>
                    <div class="preview-name">{html.escape(team_a)}</div>
                </div>
            </div>
            <div class="preview-center">VS</div>
            <div class="preview-team-card" style="--team-color: {html.escape(get_team_color(team_b))};">
                {team_b_logo_html}
                <div>
                    <div class="preview-role">{html.escape(label_b)}</div>
                    <div class="preview-name">{html.escape(team_b)}</div>
                </div>
            </div>
        </div>
        """
    )


def render_team_card(team_name: str, probability: float, is_winner: bool = False) -> None:
    """Render one team probability card."""
    logo_url = team_logo_url(get_team_id_from_strength(team_name))
    logo_html = (
        f'<img class="team-logo" src="{html.escape(logo_url)}" alt="{html.escape(team_name)} logo">'
        if logo_url
        else ""
    )
    winner_html = '<span class="score-chip">Projected Winner</span>' if is_winner else ""

    st.html(
        f"""
        <div class="team-card" style="--team-color: {html.escape(get_team_color(team_name))};">
            {winner_html}
            {logo_html}
            <div class="team-name">{html.escape(team_name)}</div>
            <div class="dashboard-note">Win probability</div>
            <div class="team-probability">{probability:.1%}</div>
        </div>
        """
    )


def render_matchup_cards(
    team_a: str,
    team_b: str,
    probability_a: float,
    probability_b: float,
    winner: str,
) -> None:
    """Render side-by-side matchup probabilities."""
    col1, col2 = st.columns(2)

    with col1:
        render_team_card(team_a, probability_a, is_winner=winner == team_a)

    with col2:
        render_team_card(team_b, probability_b, is_winner=winner == team_b)


def render_team_logo_strip(teams: list[str]) -> None:
    """Render a compact strip of team logos."""
    rows = []

    for team in teams:
        logo_url = team_logo_url(get_team_id_from_strength(team))
        logo_html = (
            f'<img src="{html.escape(logo_url)}" alt="{html.escape(team)} logo">'
            if logo_url
            else ""
        )
        rows.append(
            f"""
            <div class="logo-tile" style="--team-color: {html.escape(get_team_color(team))};">
                {logo_html}
                <span>{html.escape(team)}</span>
            </div>
            """
        )

    st.html(f'<div class="logo-strip">{"".join(rows)}</div>')


def render_team_mini_rows(rows: list[dict]) -> None:
    """Render compact team rows for dashboard summaries."""
    row_html = []

    for row in rows:
        team = str(row.get("team", ""))
        logo_url = team_logo_url(get_team_id_from_strength(team))
        logo_html = (
            f'<img class="team-mini-logo" src="{html.escape(logo_url)}" alt="{html.escape(team)} logo">'
            if logo_url
            else '<div class="team-mini-logo"></div>'
        )
        row_html.append(
            f"""
            <div class="team-mini-row">
                {logo_html}
                <div>
                    <div class="team-mini-name">{html.escape(team)}</div>
                    <div class="team-mini-meta">{html.escape(str(row.get('meta', '')))}</div>
                </div>
                <div class="team-mini-score">{html.escape(str(row.get('score', '')))}</div>
            </div>
            """
        )

    st.html("".join(row_html))


def render_home_signal_cards(cards: list[dict]) -> None:
    """Render home-only signal cards."""
    if not cards:
        st.info("No pick signals are available for the current slate.")
        return

    card_html = []

    for card in cards:
        card_html.append(
            f"""
            <div class="home-signal-card" style="--card-color: {html.escape(card.get('color', '#1d4ed8'))};">
                <div class="home-signal-title">{html.escape(str(card.get('label', '')))}</div>
                <div class="home-signal-main">{html.escape(str(card.get('title', '')))}</div>
                <div class="home-signal-value">{html.escape(str(card.get('value', '')))}</div>
                <div class="dashboard-note">{html.escape(str(card.get('note', '')))}</div>
            </div>
            """
        )

    st.html(f'<div class="home-signal-grid">{"".join(card_html)}</div>')


def create_prediction_report(title: str, rows: list[str]) -> str:
    """Create a plain-text report."""
    return "\n".join([title, "=" * len(title), "", *rows])


def render_report_download(report_text: str, file_name: str, key: str) -> None:
    """Render a report download button."""
    st.download_button(
        "Download report",
        data=report_text.encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        key=key,
        width="stretch",
    )


def get_configured_odds_api_key() -> str:
    """Return an odds API key from Streamlit secrets or environment."""
    try:
        secret_value = st.secrets.get("ODDS_API_KEY", "")

        if secret_value:
            return str(secret_value).strip()
    except Exception:
        pass

    return str(os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY") or "").strip()


def get_configured_secret_or_env(*names: str) -> str:
    """Return the first configured Streamlit secret or environment value."""
    for name in names:
        try:
            secret_value = st.secrets.get(name, "")

            if secret_value:
                return str(secret_value).strip()
        except Exception:
            pass

        env_value = os.environ.get(name)

        if env_value:
            return str(env_value).strip()

    return ""


def get_configured_sharpsports_public_key() -> str:
    """Return SharpSports public key when configured."""
    return get_configured_secret_or_env("SHARPSPORTS_PUBLIC_API_KEY", "SHARPSPORTS_API_KEY")


def get_configured_sharpsports_private_key() -> str:
    """Return SharpSports private key when configured."""
    return get_configured_secret_or_env("SHARPSPORTS_PRIVATE_API_KEY")


def get_configured_betting_region() -> str:
    """Return the default betting region/state code for provider links."""
    return (get_configured_secret_or_env("BETTING_REGION", "SHARPSPORTS_REGION") or "az").lower()


def american_odds_to_implied_probability(odds: object) -> float | None:
    """Convert American odds to implied probability before vig removal."""
    parsed = safe_float(odds, default=None)

    if parsed is None or parsed == 0:
        return None

    if parsed > 0:
        return 100 / (parsed + 100)

    return abs(parsed) / (abs(parsed) + 100)


def american_odds_profit_per_unit(odds: object) -> float | None:
    """Return profit on a one-unit winning bet at American odds."""
    parsed = safe_float(odds, default=None)

    if parsed is None or parsed == 0:
        return None

    if parsed > 0:
        return parsed / 100

    return 100 / abs(parsed)


def probability_to_american_odds(probability: object) -> str:
    """Return fair American odds from a probability."""
    parsed = safe_float(probability, default=None)

    if parsed is None:
        return "-"

    parsed = clamp_float(float(parsed), 0.01, 0.99)

    if parsed >= 0.5:
        return str(int(round(-(parsed / (1 - parsed)) * 100)))

    return f"+{int(round(((1 - parsed) / parsed) * 100))}"


def remove_two_way_vig(probability_a: float | None, probability_b: float | None) -> tuple[float | None, float | None]:
    """Normalize two implied probabilities to remove sportsbook hold."""
    if probability_a is None or probability_b is None:
        return None, None

    total = probability_a + probability_b

    if total <= 0:
        return None, None

    return probability_a / total, probability_b / total


def expected_value_per_unit(model_probability: object, odds: object) -> float | None:
    """Return expected profit per one unit risked."""
    probability = safe_float(model_probability, default=None)
    profit = american_odds_profit_per_unit(odds)

    if probability is None or profit is None:
        return None

    return (float(probability) * profit) - (1 - float(probability))


def kelly_fraction(model_probability: object, odds: object) -> float | None:
    """Return full Kelly fraction for American odds."""
    probability = safe_float(model_probability, default=None)
    profit = american_odds_profit_per_unit(odds)

    if probability is None or profit is None or profit <= 0:
        return None

    fraction = ((profit * float(probability)) - (1 - float(probability))) / profit
    return clamp_float(fraction, 0.0, 1.0)


def edge_strength_label(edge: object) -> tuple[str, str]:
    """Return display label and CSS class for a model edge."""
    parsed = safe_float(edge, default=None)

    if parsed is None:
        return "No odds", "edge-none"

    if parsed >= 0.05:
        return "Strong edge", "edge-strong"

    if parsed >= 0.02:
        return "Small edge", "edge-small"

    return "No edge", "edge-none"


def format_signed_percent(value: object) -> str:
    """Format a signed percentage."""
    parsed = safe_float(value, default=None)

    if parsed is None:
        return "-"

    return f"{parsed:+.1%}"


def odds_team_key(name: object) -> str:
    """Normalize team names for matching odds events to schedule games."""
    text = str(name or "").strip().lower()
    text = text.replace(".", "")
    text = text.replace("  ", " ")
    return text


def sportsbook_key_from_bookmaker(value: object) -> str:
    """Normalize a bookmaker title/key into a supported sportsbook key."""
    text = str(value or "").strip().lower()
    compact = " ".join(text.replace("_", " ").replace("-", " ").split())
    alnum = "".join(ch for ch in compact if ch.isalnum())
    return SPORTSBOOK_KEY_ALIASES.get(compact) or SPORTSBOOK_KEY_ALIASES.get(alnum) or alnum


def sportsbook_label(value: object) -> str:
    """Return a display label for a sportsbook key or bookmaker name."""
    key = sportsbook_key_from_bookmaker(value)
    config = SPORTSBOOK_LINKS.get(key)

    if config:
        return str(config["label"])

    return str(value or "Sportsbook").strip() or "Sportsbook"


def sportsbook_abbreviation(value: object) -> str:
    """Return a provider-facing sportsbook abbreviation when known."""
    key = sportsbook_key_from_bookmaker(value)
    config = SPORTSBOOK_LINKS.get(key)
    return str(config.get("abbr", key)) if config else key


def sportsbook_external_url(value: object) -> str:
    """Return a safe external sportsbook page for MLB betting."""
    key = sportsbook_key_from_bookmaker(value)
    config = SPORTSBOOK_LINKS.get(key)
    return str(config.get("url", "")) if config else str(SPORTSBOOK_LINKS["draftkings"]["url"])


def is_supported_sportsbook(value: object) -> bool:
    """Return whether the sportsbook should appear in the beginner betting UI."""
    return sportsbook_key_from_bookmaker(value) in SPORTSBOOK_LINKS


def render_sportsbook_open_link(label: object, url: str) -> None:
    """Render a direct external sportsbook website link."""
    safe_url = html.escape(str(url or SPORTSBOOK_LINKS["draftkings"]["url"]), quote=True)
    safe_label = html.escape(str(label or "Sportsbook"))
    st.markdown(
        f"""
        <a class="sportsbook-open-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">
            Open {safe_label} website
        </a>
        """,
        unsafe_allow_html=True,
    )


def betting_game_key(away_team: object, home_team: object) -> str:
    """Return a schedule/odds matching key."""
    return f"{odds_team_key(away_team)}|{odds_team_key(home_team)}"


def choose_better_american_odds(current: object, candidate: object) -> object:
    """Return the better bettor-facing American price."""
    current_float = safe_float(current, default=None)
    candidate_float = safe_float(candidate, default=None)

    if candidate_float is None:
        return current

    if current_float is None:
        return candidate

    return candidate if candidate_float > current_float else current


def extract_moneyline_odds_map(events: list[dict]) -> dict[str, dict]:
    """Return best available moneyline odds by matchup."""
    odds_map: dict[str, dict] = {}

    for event in events or []:
        home_team = str(event.get("home_team", "") or "")
        away_team = str(event.get("away_team", "") or "")

        if not home_team or not away_team:
            continue

        key = betting_game_key(away_team, home_team)
        row = odds_map.setdefault(
            key,
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_moneyline": None,
                "away_moneyline": None,
                "sportsbook": "",
            },
        )

        for bookmaker in event.get("bookmakers", []) or []:
            sportsbook = str(bookmaker.get("title", "") or bookmaker.get("key", "") or "")

            if not is_supported_sportsbook(sportsbook):
                continue

            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []) or []:
                    outcome_name = str(outcome.get("name", "") or "")
                    price = outcome.get("price")

                    if odds_team_key(outcome_name) == odds_team_key(home_team):
                        chosen = choose_better_american_odds(row["home_moneyline"], price)
                        row["home_moneyline"] = chosen

                        if safe_float(chosen, default=None) == safe_float(price, default=None):
                            row["home_sportsbook"] = sportsbook
                    elif odds_team_key(outcome_name) == odds_team_key(away_team):
                        chosen = choose_better_american_odds(row["away_moneyline"], price)
                        row["away_moneyline"] = chosen

                        if safe_float(chosen, default=None) == safe_float(price, default=None):
                            row["away_sportsbook"] = sportsbook

                    if sportsbook:
                        row["sportsbook"] = sportsbook

    return odds_map


def extract_moneyline_sportsbook_rows(events: list[dict]) -> dict[str, list[dict]]:
    """Return book-by-book moneyline prices by matchup."""
    rows_by_game: dict[str, list[dict]] = {}

    for event in events or []:
        home_team = str(event.get("home_team", "") or "")
        away_team = str(event.get("away_team", "") or "")

        if not home_team or not away_team:
            continue

        game_key = betting_game_key(away_team, home_team)

        for bookmaker in event.get("bookmakers", []) or []:
            bookmaker_key = str(bookmaker.get("key", "") or bookmaker.get("title", "") or "")
            bookmaker_title = str(bookmaker.get("title", "") or bookmaker_key or "Sportsbook")
            sportsbook_key = sportsbook_key_from_bookmaker(bookmaker_key or bookmaker_title)

            if sportsbook_key not in SPORTSBOOK_LINKS:
                continue

            row = {
                "sportsbook_key": sportsbook_key,
                "sportsbook": sportsbook_label(bookmaker_title),
                "bookmaker_key": bookmaker_key,
                "home_moneyline": None,
                "away_moneyline": None,
                "last_update": bookmaker.get("last_update", ""),
            }

            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue

                if market.get("last_update"):
                    row["last_update"] = market.get("last_update")

                for outcome in market.get("outcomes", []) or []:
                    outcome_name = str(outcome.get("name", "") or "")
                    price = outcome.get("price")

                    if odds_team_key(outcome_name) == odds_team_key(home_team):
                        row["home_moneyline"] = price
                    elif odds_team_key(outcome_name) == odds_team_key(away_team):
                        row["away_moneyline"] = price

            if row["home_moneyline"] is not None or row["away_moneyline"] is not None:
                rows_by_game.setdefault(game_key, []).append(row)

    for game_key, rows in rows_by_game.items():
        rows_by_game[game_key] = sorted(rows, key=lambda row: str(row.get("sportsbook", "")))

    return rows_by_game


def sportsbook_price_for_selection(book_row: dict, selection: str, home_team: str, away_team: str) -> object:
    """Return the sportsbook moneyline for the selected team."""
    if odds_team_key(selection) == odds_team_key(home_team):
        return book_row.get("home_moneyline")

    if odds_team_key(selection) == odds_team_key(away_team):
        return book_row.get("away_moneyline")

    return None


def sportsbook_market_probability_for_selection(
    book_row: dict,
    selection: str,
    home_team: str,
    away_team: str,
) -> float | None:
    """Return no-vig market probability for one selected team at one sportsbook."""
    away_implied = american_odds_to_implied_probability(book_row.get("away_moneyline"))
    home_implied = american_odds_to_implied_probability(book_row.get("home_moneyline"))
    away_market, home_market = remove_two_way_vig(away_implied, home_implied)

    if odds_team_key(selection) == odds_team_key(home_team):
        return home_market

    if odds_team_key(selection) == odds_team_key(away_team):
        return away_market

    return None


def best_book_row_for_selection(
    book_rows: list[dict],
    selection: str,
    home_team: str,
    away_team: str,
) -> dict:
    """Return the best available sportsbook row for one selection."""
    best_row = {}
    best_price = None

    for row in book_rows:
        if not is_supported_sportsbook(row.get("sportsbook_key") or row.get("sportsbook")):
            continue

        price = safe_float(
            sportsbook_price_for_selection(row, selection, home_team, away_team),
            default=None,
        )

        if price is None:
            continue

        if best_price is None or price > best_price:
            best_price = price
            best_row = row

    return best_row


def sportsbook_price_table(book_rows: list[dict], away_team: str, home_team: str) -> pd.DataFrame:
    """Build a display table for sportsbook moneylines."""
    rows = []

    for row in book_rows:
        away_odds = row.get("away_moneyline")
        home_odds = row.get("home_moneyline")
        away_implied = american_odds_to_implied_probability(away_odds)
        home_implied = american_odds_to_implied_probability(home_odds)
        hold = (
            (away_implied + home_implied - 1)
            if away_implied is not None and home_implied is not None
            else None
        )
        rows.append(
            {
                "Sportsbook": row.get("sportsbook", "Sportsbook"),
                team_abbreviation(away_team): format_american_odds(away_odds),
                team_abbreviation(home_team): format_american_odds(home_odds),
                "Hold": f"{hold:.1%}" if hold is not None else "-",
                "Updated": format_game_time(row.get("last_update")) if row.get("last_update") else "-",
            }
        )

    return pd.DataFrame(rows)


def sportsbook_option_rows(
    book_rows: list[dict],
    selection: str,
    home_team: str,
    away_team: str,
) -> list[dict]:
    """Return sportsbook options sorted by the selected team's available price."""
    options = []

    for key, config in SPORTSBOOK_LINKS.items():
        options.append(
            {
                "sportsbook_key": key,
                "sportsbook": config["label"],
                "price": None,
                "row": {},
            }
        )

    for row in book_rows:
        key = sportsbook_key_from_bookmaker(row.get("sportsbook_key") or row.get("sportsbook"))

        if key not in SPORTSBOOK_LINKS:
            continue

        price = sportsbook_price_for_selection(row, selection, home_team, away_team)
        matched = next((option for option in options if option["sportsbook_key"] == key), None)

        if matched is None:
            matched = {
                "sportsbook_key": key,
                "sportsbook": sportsbook_label(row.get("sportsbook")),
                "price": None,
                "row": {},
            }
            options.append(matched)

        matched["price"] = price
        matched["row"] = row

    def sort_key(option: dict) -> tuple:
        price = safe_float(option.get("price"), default=None)
        preferred_index = list(SPORTSBOOK_LINKS).index(option.get("sportsbook_key")) if option.get("sportsbook_key") in SPORTSBOOK_LINKS else 99
        return (
            0 if price is not None else 1,
            -float(price) if price is not None else 99999,
            preferred_index,
            str(option.get("sportsbook", "")),
        )

    return sorted(options, key=sort_key)


def sharpsports_standalone_place_url(market_selection_id: str, sportsbook: object) -> str:
    """Build a SharpSports standalone betPlace URL from a known marketSelection id."""
    market_id = str(market_selection_id or "").strip()
    book_abbr = sportsbook_abbreviation(sportsbook)

    if not market_id or not book_abbr:
        return ""

    return f"https://ui.sharpsports.io/place/{market_id}/{book_abbr}"


def extract_sharpsports_context_url(context: dict, mode: str = "place") -> str:
    """Extract a usable hosted SharpSports URL from a context response."""
    for key in ["url", "link", "redirectUrl"]:
        value = str(context.get(key, "") or "").strip()

        if value:
            return value if value.startswith("http") else f"https://{value.lstrip('/')}"

    context_id = str(context.get("cid") or context.get("id") or context.get("contextId") or "").strip()

    if context_id:
        return f"https://ui.sharpsports.io/{mode}/{context_id}"

    return ""


def format_american_odds(value: object) -> str:
    """Format American odds with plus sign for positive prices."""
    parsed = safe_float(value, default=None)

    if parsed is None or parsed == 0:
        return "-"

    rounded = int(round(parsed))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def load_bet_tracker() -> pd.DataFrame:
    """Load saved MLB bet tracker rows."""
    if not BET_TRACKER_PATH.exists():
        return pd.DataFrame(columns=BET_TRACKER_COLUMNS)

    tracker = pd.read_csv(BET_TRACKER_PATH, dtype=str).fillna("")

    for column in BET_TRACKER_COLUMNS:
        if column not in tracker.columns:
            tracker[column] = ""

    tracker = tracker[BET_TRACKER_COLUMNS].copy()
    text_columns = [
        "Bet ID",
        "Date Logged",
        "Game Date",
        "Matchup",
        "Market",
        "Selection",
        "Result",
        "Notes",
    ]

    for column in text_columns:
        tracker[column] = tracker[column].fillna("").astype(str)

    return tracker


def save_bet_tracker(tracker: pd.DataFrame) -> None:
    """Persist MLB bet tracker rows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tracker = tracker.copy()

    for column in BET_TRACKER_COLUMNS:
        if column not in tracker.columns:
            tracker[column] = ""

    for column in ["Result", "Notes", "Selection", "Matchup", "Market"]:
        tracker[column] = tracker[column].fillna("").astype(str)

    tracker[BET_TRACKER_COLUMNS].to_csv(BET_TRACKER_PATH, index=False)


def next_bet_id(tracker: pd.DataFrame) -> int:
    """Return the next bet tracker id."""
    if tracker.empty or "Bet ID" not in tracker.columns:
        return 1

    ids = pd.to_numeric(tracker["Bet ID"], errors="coerce").dropna()
    return int(ids.max()) + 1 if not ids.empty else 1


def calculate_bet_profit(result: object, odds: object, stake: object) -> float | None:
    """Calculate profit for a settled bet."""
    result_text = str(result or "").strip().lower()
    parsed_stake = safe_float(stake, default=None)
    profit_per_unit = american_odds_profit_per_unit(odds)

    if parsed_stake is None:
        return None

    if result_text == "win" and profit_per_unit is not None:
        return parsed_stake * profit_per_unit

    if result_text == "loss":
        return -parsed_stake

    if result_text == "push":
        return 0.0

    return None


def recalculate_tracker_results(tracker: pd.DataFrame) -> pd.DataFrame:
    """Recalculate profit and CLV fields for edited bet tracker rows."""
    tracker = tracker.copy()

    for column in ["Profit", "CLV"]:
        if column in tracker.columns:
            tracker[column] = tracker[column].astype("object")

    for index, row in tracker.iterrows():
        profit = calculate_bet_profit(row.get("Result"), row.get("Odds"), row.get("Stake"))

        if profit is not None:
            tracker.at[index, "Profit"] = round(float(profit), 2)

        closing_probability = american_odds_to_implied_probability(row.get("Closing Odds"))
        market_probability = safe_float(row.get("Market Probability"), default=None)

        if closing_probability is not None and market_probability is not None:
            tracker.at[index, "CLV"] = round(float(closing_probability) - float(market_probability), 4)

    return tracker


def parse_tracker_game_date(value: object) -> date | None:
    """Return a date from a tracker row date value."""
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.date()


def parse_tracker_matchup(matchup: object) -> tuple[str, str] | None:
    """Split the saved matchup label into away/home teams."""
    parts = str(matchup or "").split(" at ", 1)

    if len(parts) != 2:
        return None

    away_team = parts[0].strip()
    home_team = parts[1].strip()

    if not away_team or not home_team:
        return None

    return away_team, home_team


def tracker_score_lookup_from_games(games: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Build a lookup of final score rows by game date and matchup."""
    lookup: dict[tuple[str, str], pd.Series] = {}

    if games.empty:
        return lookup

    for _, game in games.iterrows():
        game_date = parse_tracker_game_date(game.get("GAME_DATE"))

        if game_date is None:
            continue

        matchup_key = odds_team_key(format_matchup(game))
        key = (game_date.isoformat(), matchup_key)
        lookup[key] = game

    return lookup


def load_tracker_score_lookup(tracker: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Load score rows that can settle saved moneyline picks."""
    frames = []
    raw_games = load_raw_games()

    if not raw_games.empty:
        frames.append(raw_games)

    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    tracker_dates = [
        parsed
        for parsed in tracker["Game Date"].map(parse_tracker_game_date).tolist()
        if parsed is not None and parsed <= today
    ]

    if tracker_dates:
        start_date = max(min(tracker_dates), today - timedelta(days=14))
        schedule = load_mlb_schedule(start_date.isoformat(), today.isoformat())

        if not schedule.empty:
            frames.append(schedule)

    if not frames:
        return {}

    games = pd.concat(frames, ignore_index=True)

    if "GAME_PK" in games.columns:
        games = games.drop_duplicates(subset=["GAME_PK"], keep="last")

    return tracker_score_lookup_from_games(games)


def append_tracker_note(existing: object, note: str) -> str:
    """Append a short note once."""
    current = str(existing or "").strip()

    if not note or note in current:
        return current

    return f"{current}; {note}" if current else note


def settle_moneyline_pick(row: pd.Series, game: pd.Series) -> tuple[str | None, str]:
    """Return the win/loss/push result for a saved moneyline pick."""
    home_score = safe_int(game.get("HOME_SCORE"))
    away_score = safe_int(game.get("AWAY_SCORE"))

    if home_score is None or away_score is None:
        return None, ""

    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    final_note = f"Final: {away_team} {away_score}, {home_team} {home_score}"

    if home_score == away_score:
        return "Push", final_note

    winner = home_team if home_score > away_score else away_team
    selection = str(row.get("Selection", "") or "")
    result = "Win" if odds_team_key(selection) == odds_team_key(winner) else "Loss"
    return result, final_note


def auto_settle_bet_tracker(tracker: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Settle open moneyline tracker rows when final MLB scores are available."""
    tracker = tracker.copy()
    changed = False

    if tracker.empty:
        return recalculate_tracker_results(tracker), changed

    score_lookup = load_tracker_score_lookup(tracker)

    if not score_lookup:
        return recalculate_tracker_results(tracker), changed

    for index, row in tracker.iterrows():
        result = str(row.get("Result", "") or "Open").strip().lower()

        if result in {"win", "loss", "push"}:
            continue

        if str(row.get("Market", "") or "").strip().lower() != "moneyline":
            continue

        game_date = parse_tracker_game_date(row.get("Game Date"))
        matchup = parse_tracker_matchup(row.get("Matchup"))

        if game_date is None or matchup is None:
            continue

        matchup_key = odds_team_key(f"{matchup[0]} at {matchup[1]}")
        game = score_lookup.get((game_date.isoformat(), matchup_key))

        if game is None:
            continue

        if not is_final_status_value(game.get("STATUS"), game.get("STATUS_CODE"), game.get("IS_FINAL")):
            continue

        settled_result, final_note = settle_moneyline_pick(row, game)

        if not settled_result:
            continue

        tracker.at[index, "Result"] = settled_result
        tracker.at[index, "Notes"] = append_tracker_note(row.get("Notes"), final_note)
        changed = True

    return recalculate_tracker_results(tracker), changed


def load_bet_tracker_with_auto_settlement() -> pd.DataFrame:
    """Load tracker rows and persist automatic moneyline settlements."""
    tracker, changed = auto_settle_bet_tracker(load_bet_tracker())

    if changed:
        save_bet_tracker(tracker)

    return tracker


def delete_bet_tracker_row(bet_id: object) -> None:
    """Delete one saved tracker row by id."""
    tracker = load_bet_tracker()
    bet_id_text = str(bet_id or "").strip()

    if tracker.empty or not bet_id_text:
        return

    tracker = tracker[tracker["Bet ID"].astype(str).ne(bet_id_text)].copy()
    save_bet_tracker(recalculate_tracker_results(tracker))


def append_bet_tracker_row(row: dict) -> None:
    """Append one bet to the tracker CSV."""
    tracker = load_bet_tracker()
    row = {column: row.get(column, "") for column in BET_TRACKER_COLUMNS}
    row["Bet ID"] = next_bet_id(tracker)
    updated = pd.concat([tracker, pd.DataFrame([row])], ignore_index=True)
    save_bet_tracker(updated)


def poisson_cdf(k: int, rate: float) -> float:
    """Return Poisson CDF for small prop-projection helpers."""
    if k < 0:
        return 0.0

    rate = max(float(rate), 0.0)
    total = 0.0

    for value in range(k + 1):
        total += math.exp(-rate) * (rate**value) / math.factorial(value)

    return total


def probability_over_poisson(line: float, projection: float) -> float:
    """Approximate probability of going over a prop line."""
    threshold = math.floor(float(line))
    return clamp_float(1 - poisson_cdf(threshold, max(float(projection), 0.0)), 0.0, 1.0)


def parse_baseball_innings(value: object) -> float | None:
    """Parse baseball innings where .1 and .2 mean thirds of an inning."""
    text = display_stat(value)

    if text == "-":
        return None

    parts = text.split(".")

    try:
        whole = int(parts[0])
    except ValueError:
        return safe_float(value, default=None)

    if len(parts) == 1:
        return float(whole)

    outs = 0

    try:
        outs = int(parts[1][:1])
    except ValueError:
        outs = 0

    return whole + (min(max(outs, 0), 2) / 3)


def project_player_prop(player: dict, prop_type: str) -> tuple[float | None, str]:
    """Return a simple player prop projection and note."""
    prop = str(prop_type or "")

    if prop == "Pitcher strikeouts":
        strikeouts = safe_float(player.get("SO"), default=None)
        innings = parse_baseball_innings(player.get("IP"))

        if strikeouts is None or not innings:
            return None, "Missing pitcher strikeout or innings data."

        projection = (strikeouts / innings) * 5.4
        return projection, "Uses season strikeouts per inning with a 5.4 IP starter baseline."

    if prop == "Batter hits":
        avg = safe_float(player.get("AVG"), default=None)

        if avg is None:
            return None, "Missing batting average."

        return avg * 4.1, "Uses batting average with a 4.1 at-bat baseline."

    if prop == "Total bases":
        avg = safe_float(player.get("AVG"), default=None)
        ops = safe_float(player.get("OPS"), default=None)

        if avg is None:
            return None, "Missing batting average."

        power_factor = 1.35 + (0.35 if ops and ops >= 0.8 else 0.0)
        return avg * 4.1 * power_factor, "Uses projected hits with a simple power adjustment."

    if prop == "RBIs":
        rbi = safe_float(player.get("RBI"), default=None)

        if rbi is None:
            return None, "Missing RBI data."

        return rbi / 135, "Uses season RBI pace with a 135-game playing baseline."

    if prop == "Home runs":
        hr = safe_float(player.get("HR"), default=None)

        if hr is None:
            return None, "Missing home run data."

        return hr / 135, "Uses season HR pace with a 135-game playing baseline."

    return None, "Choose a prop type."


def current_matchup_html(snapshot: dict, reserve_space: bool = False) -> str:
    """Return current batter-vs-pitcher HTML for live games."""
    matchup = snapshot.get("CURRENT_MATCHUP", {}) or {}
    batter = matchup.get("batter", {}) or {}
    pitcher = matchup.get("pitcher", {}) or {}

    if not batter.get("name") or not pitcher.get("name"):
        if not reserve_space:
            return ""

        return (
            '<div class="current-matchup-row card-slot-empty" aria-hidden="true">'
            '<div class="current-player"><div class="current-headshot"></div><div>'
            '<div class="current-role">Current batter</div><div class="current-name">Player</div>'
            '</div></div><div class="current-vs">VS</div>'
            '<div class="current-player pitcher"><div>'
            '<div class="current-role">Current pitcher</div><div class="current-name">Player</div>'
            '</div><div class="current-headshot"></div></div></div>'
        )

    def player_html(player: dict, role: str, extra_class: str = "") -> str:
        name = str(player.get("name", "") or "")
        photo_url = str(player.get("photo_url", "") or "")
        average = str(player.get("avg", "") or "").strip()
        pitch_count = str(player.get("pitch_count", "") or "").strip()
        stat_bits = []

        if average:
            stat_bits.append(f"AVG {average}")

        if pitch_count:
            stat_bits.append(f"Pitches thrown: {pitch_count}")

        stat_html = (
            f'<div class="current-stat">{html.escape(" / ".join(stat_bits))}</div>'
            if stat_bits
            else ""
        )
        headshot_html = (
            f'<img class="current-headshot" src="{html.escape(photo_url)}" alt="{html.escape(name)} headshot">'
            if photo_url
            else '<div class="current-headshot"></div>'
        )

        if extra_class == "pitcher":
            return (
                f'<div class="current-player {extra_class}">'
                '<div>'
                f'<div class="current-role">{html.escape(role)}</div>'
                f'<div class="current-name">{html.escape(name)}</div>'
                f'{stat_html}'
                '</div>'
                f'{headshot_html}'
                '</div>'
            )

        return (
            f'<div class="current-player {extra_class}">'
            f'{headshot_html}'
            '<div>'
            f'<div class="current-role">{html.escape(role)}</div>'
            f'<div class="current-name">{html.escape(name)}</div>'
            f'{stat_html}'
            '</div>'
            '</div>'
        )

    return (
        '<div class="current-matchup-row">'
        f'{player_html(batter, "Current batter")}'
        '<div class="current-vs">VS</div>'
        f'{player_html(pitcher, "Current pitcher", "pitcher")}'
        '</div>'
    )


def base_diamond_html(snapshot: dict, reserve_space: bool = False) -> str:
    """Return a compact base-occupancy diamond."""
    base_state = snapshot.get("BASE_STATE", {}) or {}
    has_live_context = bool(snapshot.get("COUNT_SUMMARY")) or snapshot.get("LIVE_HOME_WIN_PROBABILITY") is not None

    if not any(base_state.values()) and not has_live_context and not reserve_space:
        return ""

    empty_class = " card-slot-empty" if reserve_space and not has_live_context else ""

    def base_class(base: str) -> str:
        occupied = " occupied" if base_state.get(base) else ""
        return f"base-node {base}{occupied}"

    return (
        f'<div class="base-diamond{empty_class}" aria-label="Base runners">'
        f'<div class="{base_class("second")}"></div>'
        f'<div class="{base_class("third")}"></div>'
        f'<div class="{base_class("first")}"></div>'
        '<div class="base-node home"></div>'
        '</div>'
    )


def safe_streamlit_dom_key(value: object) -> str:
    """Return a key segment that is stable for Streamlit CSS classes."""
    safe_key = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "game"))
    safe_key = "-".join(part for part in safe_key.split("-") if part)
    return safe_key[:72] or "game"


def team_abbreviation(team_name: object, team_id: object = None) -> str:
    """Return a compact MLB team abbreviation for tight UI labels."""
    try:
        parsed_team_id = int(float(team_id))
    except (TypeError, ValueError):
        parsed_team_id = None

    if parsed_team_id in MLB_TEAM_ABBREVIATIONS_BY_ID:
        return MLB_TEAM_ABBREVIATIONS_BY_ID[parsed_team_id]

    name = str(team_name or "").strip()
    mapped_name = MLB_TEAM_ABBREVIATIONS_BY_NAME.get(name.lower())

    if mapped_name:
        return mapped_name

    words = [word for word in name.replace(".", "").split() if word.lower() not in {"the"}]
    fallback = "".join(word[:1].upper() for word in words)[:3]
    return fallback or "TEAM"


def is_final_game_snapshot(game: pd.Series, snapshot: dict) -> bool:
    """Return whether a game row/snapshot represents a completed game."""
    return is_final_status_value(
        snapshot.get("STATUS", game.get("STATUS", "")),
        game.get("STATUS_CODE", ""),
        snapshot.get("IS_FINAL", game.get("IS_FINAL")),
    )


def final_winner_label(game: pd.Series, snapshot: dict) -> str:
    """Return a compact final winner label for the win-probability button."""
    if not is_final_game_snapshot(game, snapshot):
        return ""

    home_score = safe_int(snapshot.get("HOME_SCORE", game.get("HOME_SCORE")))
    away_score = safe_int(snapshot.get("AWAY_SCORE", game.get("AWAY_SCORE")))

    if home_score is None or away_score is None or home_score == away_score:
        return ""

    if home_score > away_score:
        winner = team_abbreviation(game.get("HOME_TEAM", "Home"), game.get("HOME_TEAM_ID"))
    else:
        winner = team_abbreviation(game.get("AWAY_TEAM", "Away"), game.get("AWAY_TEAM_ID"))

    return f"{winner} won"


def live_win_probability_label(game: pd.Series, snapshot: dict) -> str:
    """Return the live win percentage tile text."""
    winner_label = final_winner_label(game, snapshot)

    if winner_label:
        return winner_label

    probability = safe_float(snapshot.get("LIVE_HOME_WIN_PROBABILITY"), default=None)

    if probability is None:
        return ""

    is_home_favorite = float(probability) >= 0.5
    favorite_team = game.get("HOME_TEAM") if is_home_favorite else game.get("AWAY_TEAM")
    favorite_team_id = game.get("HOME_TEAM_ID") if is_home_favorite else game.get("AWAY_TEAM_ID")
    favorite_abbreviation = team_abbreviation(favorite_team, favorite_team_id)
    favorite_probability = max(float(probability), 1 - float(probability))
    return f"Win %: {favorite_abbreviation} {favorite_probability:.0%}"


def live_win_probability_html(game: pd.Series, snapshot: dict, reserve_space: bool = False) -> str:
    """Return non-clickable live win-probability HTML for card layout."""
    label = live_win_probability_label(game, snapshot)

    if not label:
        if not reserve_space:
            return ""

        return '<div class="live-win-prob card-slot-empty" aria-hidden="true">Win %: 50%</div>'

    return (
        '<div class="live-win-prob card-slot-empty" aria-hidden="true">'
        f"{html.escape(label).replace(chr(10), '<br>')}"
        "</div>"
    )


def outs_dots_html(outs: object, reserve_space: bool = False) -> str:
    """Return three dot indicators for the current out count."""
    parsed_outs = safe_int(outs)

    if parsed_outs is None:
        if not reserve_space:
            return ""

        parsed_outs = 0
        empty_class = " card-slot-empty"
        aria_attrs = 'aria-hidden="true"'
    else:
        parsed_outs = max(0, min(parsed_outs, 3))
        empty_class = ""
        aria_attrs = f'aria-label="{parsed_outs} outs"'

    dots = "".join(
        f'<span class="mlb-out-dot{" active" if index < parsed_outs else ""}"></span>'
        for index in range(3)
    )
    return f'<div class="mlb-out-dots{empty_class}" {aria_attrs}>{dots}</div>'


def game_row_key(game: pd.Series) -> str:
    """Return a stable key for one game row."""
    game_pk = str(game.get("GAME_PK", "") or "").strip()

    if game_pk:
        return game_pk

    return "|".join(
        [
            str(game.get("GAME_DATETIME", "") or ""),
            str(game.get("AWAY_TEAM", "") or ""),
            str(game.get("HOME_TEAM", "") or ""),
        ]
    )


def build_win_probability_timeline(game: pd.Series, snapshot: dict) -> pd.DataFrame:
    """Build a win-probability path from the MLB play log plus current live state."""
    feed = snapshot.get("FEED", {}) or {}
    plays = get_nested(feed, ["liveData", "plays", "allPlays"], default=[]) or []
    pregame_home = safe_float(game.get("HOME_WIN_PROBABILITY"), default=None)

    if pregame_home is None:
        pregame_home = 0.5

    rows = []

    for play in plays:
        if not isinstance(play, dict):
            continue

        result = play.get("result", {}) or {}
        about = play.get("about", {}) or {}
        home_score = result.get("homeScore")
        away_score = result.get("awayScore")

        if home_score is None or away_score is None:
            continue

        probability = estimate_home_probability_from_state(
            pregame_home_probability=float(pregame_home),
            home_score=home_score,
            away_score=away_score,
            inning_state={
                "inning": about.get("inning"),
                "half": about.get("halfInning"),
            },
            count_state=play.get("count", {}) or {},
            base_state={},
        )

        if probability is None:
            continue

        rows.append(
            {
                "Point": len(rows) + 1,
                "Game State": format_play_context(play) or f"Play {len(rows) + 1}",
                "Event": str(result.get("event") or "Play"),
                "Home Win %": float(probability) * 100,
                "Away Win %": (1 - float(probability)) * 100,
            }
        )

    current_probability = safe_float(snapshot.get("LIVE_HOME_WIN_PROBABILITY"), default=None)

    if current_probability is not None:
        if not rows or abs(float(rows[-1]["Home Win %"]) - (float(current_probability) * 100)) >= 0.1:
            rows.append(
                {
                    "Point": len(rows) + 1,
                    "Game State": str(snapshot.get("INNING", "") or "Live"),
                    "Event": "Current",
                    "Home Win %": float(current_probability) * 100,
                    "Away Win %": (1 - float(current_probability)) * 100,
                }
            )

    return pd.DataFrame(rows)


def render_win_probability_chart(timeline: pd.DataFrame, home_team: str, away_team: str, chart_key: str) -> None:
    """Render an interactive win-probability chart with percentage-only hover readouts."""
    chart_points = []

    for _, row in timeline.iterrows():
        home_probability = safe_float(row.get("Home Win %"), default=None)
        away_probability = safe_float(row.get("Away Win %"), default=None)

        if home_probability is None or away_probability is None:
            continue

        chart_points.append(
            {
                "home": round(float(home_probability), 1),
                "away": round(float(away_probability), 1),
            }
        )

    if not chart_points:
        st.info("Win probability will appear once the MLB live feed publishes plays with score state.")
        return

    safe_chart_id = "".join(ch if ch.isalnum() else "_" for ch in str(chart_key))[:72] or "game"
    chart_id = f"wp_chart_{safe_chart_id}_{len(chart_points)}"
    payload = json.dumps(chart_points)
    home_label = html.escape(home_team)
    away_label = html.escape(away_team)

    components.html(
        f"""
        <div id="{chart_id}" class="wp-chart-wrap">
            <div class="wp-readout">
                <div class="wp-readout-card home">
                    <span>{home_label}</span>
                    <strong id="homePct">--%</strong>
                </div>
                <div class="wp-readout-card away">
                    <span>{away_label}</span>
                    <strong id="awayPct">--%</strong>
                </div>
            </div>
            <svg class="wp-chart" viewBox="0 0 760 320" role="img" aria-label="Win probability chart">
                <g id="gridLayer"></g>
                <path id="homePath" class="prob-line home-line"></path>
                <path id="awayPath" class="prob-line away-line"></path>
                <line id="hoverLine" class="hover-rule" y1="20" y2="274"></line>
                <circle id="homeDot" class="prob-dot home-dot" r="5"></circle>
                <circle id="awayDot" class="prob-dot away-dot" r="5"></circle>
                <rect id="hitbox" class="hitbox" x="58" y="20" width="674" height="254"></rect>
            </svg>
            <div id="wpChartState" class="wp-chart-state">Current win probability</div>
        </div>
        <style>
            .wp-chart-wrap {{
                color: #0f172a;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                width: 100%;
            }}

            .wp-readout {{
                display: grid;
                gap: 0.75rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-bottom: 0.65rem;
            }}

            .wp-readout-card {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.75rem 0.85rem;
            }}

            .wp-readout-card span {{
                color: #64748b;
                display: block;
                font-size: 0.78rem;
                font-weight: 800;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}

            .wp-readout-card strong {{
                display: block;
                font-size: 2rem;
                font-weight: 950;
                line-height: 1;
                margin-top: 0.25rem;
            }}

            .wp-readout-card.home strong {{
                color: #2563eb;
            }}

            .wp-readout-card.away strong {{
                color: #dc2626;
            }}

            .wp-chart {{
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                height: 320px;
                width: 100%;
            }}

            .prob-line {{
                fill: none;
                stroke-linecap: round;
                stroke-linejoin: round;
                stroke-width: 4;
            }}

            .home-line {{
                stroke: #2563eb;
            }}

            .away-line {{
                stroke: #dc2626;
            }}

            .prob-dot {{
                stroke: #ffffff;
                stroke-width: 2;
            }}

            .home-dot {{
                fill: #2563eb;
            }}

            .away-dot {{
                fill: #dc2626;
            }}

            .hover-rule {{
                opacity: 0;
                stroke: #475569;
                stroke-dasharray: 4 4;
                stroke-width: 1.5;
            }}

            .hover-rule.visible {{
                opacity: 1;
            }}

            .grid-line {{
                stroke: #e2e8f0;
                stroke-width: 1;
            }}

            .mid-line {{
                stroke: #94a3b8;
                stroke-dasharray: 5 5;
            }}

            .axis-text {{
                fill: #64748b;
                font-size: 12px;
                font-weight: 750;
            }}

            .hitbox {{
                cursor: crosshair;
                fill: transparent;
            }}

            .wp-chart-state {{
                color: #64748b;
                font-size: 0.82rem;
                font-weight: 750;
                margin-top: 0.45rem;
                text-align: center;
            }}

        </style>
        <script>
            (() => {{
                const root = document.getElementById({json.dumps(chart_id)});
                const data = {payload};
                const svg = root.querySelector(".wp-chart");
                const gridLayer = root.querySelector("#gridLayer");
                const homePath = root.querySelector("#homePath");
                const awayPath = root.querySelector("#awayPath");
                const hoverLine = root.querySelector("#hoverLine");
                const homeDot = root.querySelector("#homeDot");
                const awayDot = root.querySelector("#awayDot");
                const hitbox = root.querySelector("#hitbox");
                const homePct = root.querySelector("#homePct");
                const awayPct = root.querySelector("#awayPct");
                const chartState = root.querySelector("#wpChartState");
                const width = 760;
                const plot = {{ left: 58, right: 28, top: 20, bottom: 46 }};
                const plotWidth = width - plot.left - plot.right;
                const plotHeight = 320 - plot.top - plot.bottom;

                function svgNode(name) {{
                    return document.createElementNS("http://www.w3.org/2000/svg", name);
                }}

                function formatPct(value) {{
                    return Math.round(Number(value)) + "%";
                }}

                function xAt(index) {{
                    if (data.length <= 1) {{
                        return plot.left + plotWidth / 2;
                    }}

                    return plot.left + (index / (data.length - 1)) * plotWidth;
                }}

                function yAt(value) {{
                    return plot.top + ((100 - Number(value)) / 100) * plotHeight;
                }}

                function buildPath(key) {{
                    return data
                        .map((row, index) => (index === 0 ? "M" : "L") + xAt(index).toFixed(2) + " " + yAt(row[key]).toFixed(2))
                        .join(" ");
                }}

                function drawGrid() {{
                    [100, 75, 50, 25, 0].forEach((pct) => {{
                        const y = yAt(pct);
                        const line = svgNode("line");
                        line.setAttribute("x1", plot.left);
                        line.setAttribute("x2", width - plot.right);
                        line.setAttribute("y1", y);
                        line.setAttribute("y2", y);
                        line.setAttribute("class", pct === 50 ? "grid-line mid-line" : "grid-line");
                        gridLayer.appendChild(line);

                        const label = svgNode("text");
                        label.setAttribute("x", plot.left - 10);
                        label.setAttribute("y", y + 4);
                        label.setAttribute("text-anchor", "end");
                        label.setAttribute("class", "axis-text");
                        label.textContent = pct + "%";
                        gridLayer.appendChild(label);
                    }});

                    const startLabel = svgNode("text");
                    startLabel.setAttribute("x", plot.left);
                    startLabel.setAttribute("y", 306);
                    startLabel.setAttribute("class", "axis-text");
                    startLabel.textContent = "Start";
                    gridLayer.appendChild(startLabel);

                    const currentLabel = svgNode("text");
                    currentLabel.setAttribute("x", width - plot.right);
                    currentLabel.setAttribute("y", 306);
                    currentLabel.setAttribute("text-anchor", "end");
                    currentLabel.setAttribute("class", "axis-text");
                    currentLabel.textContent = "Current";
                    gridLayer.appendChild(currentLabel);
                }}

                function nearestIndex(mouseX) {{
                    let winner = 0;
                    let smallestDistance = Number.POSITIVE_INFINITY;

                    data.forEach((row, index) => {{
                        const distance = Math.abs(xAt(index) - mouseX);

                        if (distance < smallestDistance) {{
                            smallestDistance = distance;
                            winner = index;
                        }}
                    }});

                    return winner;
                }}

                function updateReadout(index, isHovering) {{
                    const row = data[index] || data[data.length - 1];
                    const x = xAt(index);
                    homePct.textContent = formatPct(row.home);
                    awayPct.textContent = formatPct(row.away);
                    chartState.textContent = isHovering ? "Selected moment" : "Current win probability";

                    hoverLine.setAttribute("x1", x);
                    hoverLine.setAttribute("x2", x);
                    hoverLine.classList.toggle("visible", isHovering);
                    homeDot.setAttribute("cx", x);
                    homeDot.setAttribute("cy", yAt(row.home));
                    awayDot.setAttribute("cx", x);
                    awayDot.setAttribute("cy", yAt(row.away));
                }}

                if (data.length) {{
                    drawGrid();
                    homePath.setAttribute("d", buildPath("home"));
                    awayPath.setAttribute("d", buildPath("away"));
                    updateReadout(data.length - 1, false);

                    hitbox.addEventListener("pointermove", (event) => {{
                        const rect = svg.getBoundingClientRect();
                        const mouseX = ((event.clientX - rect.left) / rect.width) * width;
                        updateReadout(nearestIndex(mouseX), true);
                    }});

                    hitbox.addEventListener("pointerleave", () => {{
                        updateReadout(data.length - 1, false);
                    }});
                }}
            }})();
        </script>
        """,
        height=455,
    )


def render_win_probability_dialog(game_row: dict) -> None:
    """Open a live win-probability graph dialog."""
    game = pd.Series(game_row)
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")

    def clear_win_probability_dialog() -> None:
        st.session_state.pop(WIN_PROB_DIALOG_STATE_KEY, None)
        try:
            if WIN_PROB_QUERY_PARAM in st.query_params:
                del st.query_params[WIN_PROB_QUERY_PARAM]
        except Exception:
            pass

    @st.dialog(
        f"Win Probability: {away_team} at {home_team}",
        width="large",
        dismissible=True,
        on_dismiss=clear_win_probability_dialog,
    )
    def _dialog() -> None:
        snapshot = build_live_game_snapshot(game, include_pregame_feed=True)
        timeline = build_win_probability_timeline(game, snapshot)

        if timeline.empty:
            st.info("Win probability will appear once the MLB live feed publishes plays with score state.")
            return

        render_win_probability_chart(timeline, home_team, away_team, game_row_key(game))

        with st.expander("Win probability log", expanded=False):
            display = timeline.copy()
            display["Home Win %"] = display["Home Win %"].map("{:.1f}%".format)
            display["Away Win %"] = display["Away Win %"].map("{:.1f}%".format)
            st.dataframe(display.tail(30), width="stretch", hide_index=True)

    _dialog()


def render_game_card(game: pd.Series, live_snapshot: dict | None = None) -> None:
    """Render one MLB game card."""
    live_snapshot = live_snapshot or build_live_game_snapshot(game)
    away_team = str(game["AWAY_TEAM"])
    home_team = str(game["HOME_TEAM"])
    away_score = format_score(live_snapshot.get("AWAY_SCORE"))
    home_score = format_score(live_snapshot.get("HOME_SCORE"))
    status = str(live_snapshot.get("STATUS") or game.get("STATUS", "Scheduled") or "Scheduled")
    inning = str(live_snapshot.get("INNING", "") or "").strip()
    count_summary = str(live_snapshot.get("COUNT_SUMMARY", "") or "").strip()
    latest_play = str(live_snapshot.get("LATEST_PLAY", "") or "").strip()
    latest_pitch = live_snapshot.get("LATEST_PITCH", {}) or {}
    outs_html = outs_dots_html(live_snapshot.get("OUTS"), reserve_space=True)
    game_time = format_game_time(game.get("GAME_DATETIME"))
    countdown = format_game_start_countdown(game, status_override=status)
    winner = str(game.get("PREDICTED_WINNER", "Prediction unavailable"))
    winner_probability = game.get("WINNER_PROBABILITY", math.nan)
    home_probability = game.get("HOME_WIN_PROBABILITY", math.nan)
    home_pitcher = str(game.get("HOME_PROBABLE_PITCHER", "") or "")
    away_pitcher = str(game.get("AWAY_PROBABLE_PITCHER", "") or "")
    away_logo = render_team_logo(game.get("AWAY_TEAM_ID"), away_team)
    home_logo = render_team_logo(game.get("HOME_TEAM_ID"), home_team)
    prediction_explanation = str(game.get("PREDICTION_EXPLANATION", "") or "").strip()

    if pd.isna(winner_probability):
        prediction_html = '<div class="mlb-meta">Run the MLB data refresh commands to enable picks.</div>'
    else:
        explanation_html = (
            f'<div class="prediction-explain">Why: {html.escape(prediction_explanation)}</div>'
            if prediction_explanation
            else '<div class="prediction-explain card-slot-empty" aria-hidden="true">Why: prediction reason</div>'
        )
        prediction_html = f"""
            <div class="mlb-prediction-grid">
                <div class="mlb-prediction-cell">
                    <div class="mlb-prediction-label">Pick</div>
                    <div class="mlb-prediction-value">{html.escape(winner)}</div>
                </div>
                <div class="mlb-prediction-cell">
                    <div class="mlb-prediction-label">Win chance</div>
                    <div class="mlb-prediction-value">{float(winner_probability):.0%}</div>
                </div>
                <div class="mlb-prediction-cell">
                    <div class="mlb-prediction-label">Home chance</div>
                    <div class="mlb-prediction-value">{float(home_probability):.0%}</div>
                </div>
            </div>
            {explanation_html}
        """

    away_pitcher_line = f"Probable: {away_pitcher}" if away_pitcher else "Probable pitcher TBD"
    home_pitcher_line = f"Probable: {home_pitcher}" if home_pitcher else "Probable pitcher TBD"
    away_score_html = (
        f'<div class="mlb-score">{html.escape(away_score)}</div>'
        if away_score
        else '<div class="mlb-score card-slot-empty" aria-hidden="true">0</div>'
    )
    home_score_html = (
        f'<div class="mlb-score">{html.escape(home_score)}</div>'
        if home_score
        else '<div class="mlb-score card-slot-empty" aria-hidden="true">0</div>'
    )
    inning_html = (
        f'<div class="mlb-meta">{html.escape(inning)}</div>'
        if inning
        else '<div class="mlb-meta card-slot-empty" aria-hidden="true">Inning</div>'
    )
    latest_pitch_summary = str(latest_pitch.get("SUMMARY", "") or "").strip()
    latest_pitch_html = (
        f'<div class="mlb-live-pitch"><span>Pitch</span>{html.escape(latest_pitch_summary)}</div>'
        if latest_pitch_summary
        else '<div class="mlb-live-pitch card-slot-empty" aria-hidden="true"><span>Pitch</span>Pitch type</div>'
    )

    if countdown:
        game_signal_html = (
            '<div class="mlb-live-state">'
            f'<div class="mlb-countdown">{html.escape(countdown)}</div>'
            '<div class="mlb-live-pitch card-slot-empty" aria-hidden="true"><span>Pitch</span>Pitch type</div>'
            '</div>'
        )
    elif count_summary:
        game_signal_html = (
            '<div class="mlb-live-state">'
            '<div class="mlb-live-count">'
            f'<div class="mlb-live-count-line"><span>Live</span>{html.escape(count_summary)}</div>'
            f'{outs_html}'
            '</div>'
            f'{latest_pitch_html}'
            '</div>'
        )
    else:
        game_signal_html = (
            '<div class="mlb-live-state card-slot-empty" aria-hidden="true">'
            '<div class="mlb-live-count">'
            '<div class="mlb-live-count-line"><span>Live</span>Count 0-0</div>'
            f'{outs_dots_html(0)}'
            '</div>'
            '<div class="mlb-live-pitch"><span>Pitch</span>Pitch type</div>'
            '</div>'
        )

    current_matchup = current_matchup_html(live_snapshot, reserve_space=True)
    live_panel_html = (
        '<div class="live-side-panel">'
        f'<span class="mlb-status">{html.escape(status)}</span>'
        f'{base_diamond_html(live_snapshot, reserve_space=True)}'
        f'{live_win_probability_html(game, live_snapshot, reserve_space=True)}'
        '</div>'
    )
    latest_play_html = (
        '<div class="mlb-live-note card-slot-empty" aria-hidden="true">'
        '<span>Latest</span> Play update placeholder'
        '</div>'
    )

    if latest_play:
        latest_play_html = (
            '<div class="mlb-live-note">'
            '<span>Latest</span>'
            f"{html.escape(latest_play)}"
            "</div>"
        )

    st.html(
        f"""
        <div class="mlb-card">
            <div class="mlb-topline">
                <div>
                    <div class="mlb-label">{html.escape(game_time)}</div>
                    {game_signal_html}
                    <div class="mlb-meta">{html.escape(str(game.get("VENUE", "")))}</div>
                    {inning_html}
                </div>
                {live_panel_html}
            </div>
            <div class="mlb-matchup">
                <div class="mlb-team">
                    <div class="mlb-team-heading">
                        {away_logo}
                        <div class="mlb-team-name">{html.escape(away_team)}</div>
                    </div>
                    {away_score_html}
                    <div class="mlb-meta">Away / {html.escape(away_pitcher_line)}</div>
                </div>
                <div class="mlb-vs">AT</div>
                <div class="mlb-team">
                    <div class="mlb-team-heading home">
                        {home_logo}
                        <div class="mlb-team-name">{html.escape(home_team)}</div>
                    </div>
                    {home_score_html}
                    <div class="mlb-meta">Home / {html.escape(home_pitcher_line)}</div>
                </div>
            </div>
            {current_matchup}
            {latest_play_html}
            {prediction_html}
        </div>
        """
    )


def render_play_by_play(snapshot: dict) -> None:
    """Render recent play-by-play for the selected game."""
    plays = snapshot.get("PLAYS", []) or []

    if not plays:
        st.info("Play-by-play will appear here once the MLB live feed publishes plays.")
        return

    for play in plays:
        context = html.escape(str(play.get("CONTEXT", "") or ""))
        event = html.escape(str(play.get("EVENT", "") or "Play"))
        description = html.escape(str(play.get("DESCRIPTION", "") or ""))
        score = html.escape(str(play.get("SCORE", "") or ""))
        meta = " / ".join([value for value in [context, score] if value])
        st.html(
            f"""
            <div class="mlb-play-row">
                <div class="mlb-row-title">{event}</div>
                <div class="mlb-row-meta">{meta}</div>
                <div class="mlb-row-meta">{description}</div>
            </div>
            """
        )


def render_pitch_by_pitch(snapshot: dict) -> None:
    """Render recent pitch-by-pitch feed for the selected game."""
    count_summary = str(snapshot.get("COUNT_SUMMARY", "") or "").strip()
    outs_html = outs_dots_html(snapshot.get("OUTS"))
    pitches = snapshot.get("PITCHES", []) or []

    if count_summary:
        st.html(
            f"""
            <div class="mlb-live-count">
                <div class="mlb-live-count-line"><span>Live count</span>{html.escape(count_summary)}</div>
                {outs_html}
            </div>
            """
        )

    if not pitches:
        st.info("Pitch-by-pitch data will appear here once the MLB live feed publishes pitches.")
        return

    for pitch in pitches:
        description = html.escape(
            str(pitch.get("PITCH_TYPE", "") or pitch.get("SUMMARY", "") or "Pitch")
        )
        st.html(
            f"""
            <div class="mlb-pitch-row">
                <div class="mlb-row-title">{description}</div>
            </div>
            """
        )


def render_game_center_panel(game: pd.Series, snapshot: dict) -> None:
    """Render the live game-center header inside the game dialog."""
    count_summary = str(snapshot.get("COUNT_SUMMARY", "") or "").strip()
    latest_pitch = snapshot.get("LATEST_PITCH", {}) or {}
    latest_pitch_summary = str(latest_pitch.get("SUMMARY", "") or "").strip() or "No pitch yet"
    latest_play = str(snapshot.get("LATEST_PLAY", "") or "").strip() or "No play update yet"
    inning = str(snapshot.get("INNING", "") or "").strip() or str(snapshot.get("STATUS", "") or "Scheduled")
    win_label = live_win_probability_label(game, snapshot) or "Win % unavailable"
    count_html = (
        '<div class="mlb-live-count">'
        f'<div class="mlb-live-count-line"><span>Live count</span>{html.escape(count_summary)}</div>'
        f'{outs_dots_html(snapshot.get("OUTS"), reserve_space=True)}'
        '</div>'
        if count_summary
        else (
            '<div class="mlb-live-count card-slot-empty" aria-hidden="true">'
            '<div class="mlb-live-count-line"><span>Live count</span>Count 0-0</div>'
            f'{outs_dots_html(0)}'
            '</div>'
        )
    )

    st.html(
        f"""
        <div class="game-center-panel">
            <div class="game-center-block">
                <div class="game-center-label">{html.escape(inning)}</div>
                {count_html}
                <div class="game-center-value">{html.escape(win_label)}</div>
            </div>
            <div class="game-center-diamond">
                {base_diamond_html(snapshot, reserve_space=True)}
            </div>
            <div class="game-center-block">
                <div class="game-center-label">Latest Pitch</div>
                <div class="game-center-value">{html.escape(latest_pitch_summary)}</div>
                <div class="game-center-label" style="margin-top: 0.55rem;">Latest Play</div>
                <div class="game-center-value">{html.escape(latest_play)}</div>
            </div>
        </div>
        {current_matchup_html(snapshot)}
        """
    )


def render_today_player_stat_rows(players: list[dict], game_pk: str, section_key: str) -> None:
    """Render current-game box-score rows with stat buttons."""
    sorted_players = sort_today_players(players)

    if not sorted_players:
        st.info("Player game stats will appear once the MLB box score is available.")
        return

    for index, player in enumerate(sorted_players):
        player_id = str(player.get("PLAYER_ID", "") or "")
        player_key = player_id or f"row_{index}"
        player_name = str(player.get("PLAYER_NAME", "") or "Player")
        position = str(player.get("POSITION", "") or "")
        today_line = player_today_stat_line(player)
        cols = st.columns([4, 1])

        with cols[0]:
            st.html(
                f"""
                <div class="game-today-row">
                    <div class="mlb-row-title">{html.escape(player_name)}{html.escape(f" / {position}" if position else "")}</div>
                    <div class="game-today-stats">{html.escape(today_line)}</div>
                </div>
                """
            )

        with cols[1]:
            if st.button(
                "Stats",
                key=f"mlb_today_stats_{game_pk}_{section_key}_{player_key}",
                width="stretch",
            ):
                st.session_state["mlb_selected_player_stats"] = player


def render_player_rows(players: list[dict], game_pk: str, section_key: str) -> None:
    """Render player rows with stat buttons."""
    if not players:
        st.info("Player data is not available yet from the MLB live feed.")
        return

    for index, player in enumerate(players):
        player_id = str(player.get("PLAYER_ID", "") or "")
        player_key = player_id or f"row_{index}"
        player_name = str(player.get("PLAYER_NAME", "") or "Player")
        position = str(player.get("POSITION", "") or "")
        title_bits = [player_name]
        today_line = player_today_stat_line(player)
        today_html = (
            f'<div class="mlb-row-meta">Today: {html.escape(today_line)}</div>'
            if today_line != "No game stats yet"
            else ""
        )

        if position:
            title_bits.append(position)

        cols = st.columns([4, 1])

        with cols[0]:
            st.html(
                f"""
                <div class="mlb-player-row">
                    <div class="mlb-row-title">{html.escape(" / ".join(title_bits))}</div>
                    <div class="mlb-row-meta">{html.escape(player_stat_line(player))}</div>
                    {today_html}
                </div>
                """
            )

        with cols[1]:
            if st.button(
                "View stats",
                key=f"mlb_player_stats_{game_pk}_{section_key}_{player_key}",
                width="stretch",
            ):
                st.session_state["mlb_selected_player_stats"] = player


def render_lineup_rows(players: list[dict], game_pk: str, section_key: str) -> None:
    """Render lineup rows with stat buttons."""
    if not players:
        st.info("Projected lineup data is not available yet from the MLB live feed.")
        return

    for index, player in enumerate(players, start=1):
        player_id = str(player.get("PLAYER_ID", "") or "")
        player_key = player_id or f"row_{index}"
        slot = player.get("BATTING_SLOT") or index
        lineup_label = str(player.get("LINEUP_LABEL", "") or "").strip()
        player_name = str(player.get("PLAYER_NAME", "") or "Player")
        position = str(player.get("POSITION", "") or "")
        today_line = player_today_stat_line(player)
        today_html = (
            f'<div class="mlb-row-meta">Today: {html.escape(today_line)}</div>'
            if today_line != "No game stats yet"
            else ""
        )
        cols = st.columns([4, 1])
        row_prefix = lineup_label or f"{int(slot)}."

        with cols[0]:
            st.html(
                f"""
                <div class="mlb-lineup-row">
                    <div class="mlb-row-title">{html.escape(row_prefix)} {html.escape(player_name)}</div>
                    <div class="mlb-row-meta">{html.escape(position)} / {html.escape(player_stat_line(player))}</div>
                    {today_html}
                </div>
                """
            )

        with cols[1]:
            if st.button(
                "View stats",
                key=f"mlb_lineup_stats_{game_pk}_{section_key}_{player_key}",
                width="stretch",
            ):
                st.session_state["mlb_selected_player_stats"] = player


def render_selected_player_stats() -> None:
    """Render selected player stat details."""
    player = st.session_state.get("mlb_selected_player_stats")

    if not player:
        st.info("Choose a player with the Stats button.")
        return

    st.markdown(f"**{player.get('PLAYER_NAME', 'Player')}**")
    st.caption(str(player.get("POSITION", "") or ""))
    stat_rows = [
        {"Stat": key, "Value": display_stat(player.get(key))}
        for key in PLAYER_STATS_KEYS
        if display_stat(player.get(key)) != "-"
    ]
    game_rows = [
        {"Stat": "Game AB", "Value": display_stat(player.get("GAME_AB"))},
        {"Stat": "Game R", "Value": display_stat(player.get("GAME_R"))},
        {"Stat": "Game H", "Value": display_stat(player.get("GAME_H"))},
        {"Stat": "Game RBI", "Value": display_stat(player.get("GAME_RBI"))},
        {"Stat": "Game BB", "Value": display_stat(player.get("GAME_BB"))},
        {"Stat": "Game HR", "Value": display_stat(player.get("GAME_HR"))},
        {"Stat": "Game SB", "Value": display_stat(player.get("GAME_SB"))},
        {"Stat": "Game IP", "Value": display_stat(player.get("GAME_IP"))},
        {"Stat": "Game H Allowed", "Value": display_stat(player.get("GAME_P_H"))},
        {"Stat": "Game ER", "Value": display_stat(player.get("GAME_ER"))},
        {"Stat": "Game BB Allowed", "Value": display_stat(player.get("GAME_P_BB"))},
        {"Stat": "Game SO", "Value": display_stat(player.get("GAME_SO"))},
    ]
    game_rows = [row for row in game_rows if row["Value"] != "-"]

    if stat_rows:
        st.dataframe(pd.DataFrame(stat_rows), width="stretch", hide_index=True)
    else:
        st.info("Season stats are not available yet for this player.")

    if game_rows:
        st.caption("Current game")
        st.dataframe(pd.DataFrame(game_rows), width="stretch", hide_index=True)


def render_game_detail_content(game: pd.Series) -> None:
    """Render game details inside the popup."""
    snapshot = build_live_game_snapshot(game, include_pregame_feed=True)
    feed = snapshot.get("FEED", {}) or {}
    selected_game_pk = str(game.get("GAME_PK", "") or "")
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    away_players = extract_team_players(feed, "away")
    home_players = extract_team_players(feed, "home")
    away_stars = get_star_players(feed, "away")
    home_stars = get_star_players(feed, "home")
    away_lineup, away_lineup_label = get_lineup_players(feed, "away")
    home_lineup, home_lineup_label = get_lineup_players(feed, "home")
    away_pitcher = build_probable_pitcher_profile(game, feed, "away")
    home_pitcher = build_probable_pitcher_profile(game, feed, "home")
    away_lineup = add_starting_pitcher_to_lineup(away_lineup, away_pitcher)
    home_lineup = add_starting_pitcher_to_lineup(home_lineup, home_pitcher)
    team_stats = build_team_stats_table(game, snapshot)
    inning = str(snapshot.get("INNING", "") or "")
    count_summary = str(snapshot.get("COUNT_SUMMARY", "") or "").strip()
    status = str(snapshot.get("STATUS", "") or "Scheduled")
    countdown = format_game_start_countdown(game, status_override=status)
    status_bits = [status, inning]

    if countdown:
        status_bits.append(countdown)

    if count_summary:
        status_bits.append(count_summary)

    status_line = " / ".join([bit for bit in status_bits if bit])

    st.html(
        f"""
        <div class="mlb-detail-title">{html.escape(away_team)} at {html.escape(home_team)}</div>
        <div class="mlb-row-meta">{html.escape(status_line)}</div>
        """
    )
    render_game_center_panel(game, snapshot)

    tabs = st.tabs(
        [
            "Play by Play",
            "Pitch by Pitch",
            "Today Stats",
            "Star Players",
            "Lineups",
            "Pitchers",
            "Team Stats",
        ]
    )

    with tabs[0]:
        render_play_by_play(snapshot)

    with tabs[1]:
        render_pitch_by_pitch(snapshot)

    with tabs[2]:
        away_col, home_col = st.columns(2)

        with away_col:
            st.markdown(f"**{away_team} Today**")
            render_today_player_stat_rows(away_players, selected_game_pk, "away_today")

        with home_col:
            st.markdown(f"**{home_team} Today**")
            render_today_player_stat_rows(home_players, selected_game_pk, "home_today")

        st.divider()
        render_selected_player_stats()

    with tabs[3]:
        away_col, home_col = st.columns(2)

        with away_col:
            st.markdown(f"**{away_team}**")
            render_player_rows(away_stars, selected_game_pk, "away_stars")

        with home_col:
            st.markdown(f"**{home_team}**")
            render_player_rows(home_stars, selected_game_pk, "home_stars")

        st.divider()
        render_selected_player_stats()

    with tabs[4]:
        away_col, home_col = st.columns(2)

        with away_col:
            st.markdown(f"**{away_team} {away_lineup_label}**")
            render_lineup_rows(away_lineup, selected_game_pk, "away_lineup")

        with home_col:
            st.markdown(f"**{home_team} {home_lineup_label}**")
            render_lineup_rows(home_lineup, selected_game_pk, "home_lineup")

        st.divider()
        render_selected_player_stats()

    with tabs[5]:
        render_probable_pitcher_matchup(game, feed)

    with tabs[6]:
        st.dataframe(team_stats, width="stretch", hide_index=True)


def render_game_detail_dialog(game_row: dict) -> None:
    """Open a modal with game details."""
    game = pd.Series(game_row)
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")

    def clear_game_dialog() -> None:
        st.session_state.pop("mlb_selected_player_stats", None)
        st.session_state.pop(GAME_DETAIL_STATE_KEY, None)

    @st.dialog(
        f"{away_team} at {home_team}",
        width="large",
        dismissible=True,
        on_dismiss=clear_game_dialog,
    )
    def _dialog() -> None:
        render_game_detail_content(game)

    _dialog()


def build_stable_matchup_seed(home_team: str, away_team: str, offset: int = 0) -> int:
    """Build a deterministic simulation seed for one MLB matchup."""
    key = f"{home_team}|{away_team}"
    seed = sum((index + 1) * ord(character) for index, character in enumerate(key))
    return (seed + offset) % 999999 or 1


def estimate_mlb_expected_scores(home_team: str, away_team: str, home_probability: float) -> dict:
    """Estimate baseball runs from team strength and model probability."""
    home = get_optional_team_strength(home_team)
    away = get_optional_team_strength(away_team)

    def stat(row: pd.Series | None, column: str, default: float) -> float:
        if row is None or column not in row.index:
            return default

        parsed = safe_float(row[column], default=None)
        return default if parsed is None else float(parsed)

    home_offense = (stat(home, "SEASON_AVG_RUNS_FOR", 4.4) * 0.62) + (
        stat(home, "ROLLING_RUNS_FOR_10", 4.4) * 0.38
    )
    home_defense = (stat(home, "SEASON_AVG_RUNS_AGAINST", 4.4) * 0.62) + (
        stat(home, "ROLLING_RUNS_AGAINST_10", 4.4) * 0.38
    )
    away_offense = (stat(away, "SEASON_AVG_RUNS_FOR", 4.4) * 0.62) + (
        stat(away, "ROLLING_RUNS_FOR_10", 4.4) * 0.38
    )
    away_defense = (stat(away, "SEASON_AVG_RUNS_AGAINST", 4.4) * 0.62) + (
        stat(away, "ROLLING_RUNS_AGAINST_10", 4.4) * 0.38
    )
    raw_home = (home_offense + away_defense) / 2
    raw_away = (away_offense + home_defense) / 2
    probability = min(max(home_probability, 0.01), 0.99)
    probability_margin = math.log(probability / (1 - probability)) * 1.15
    rating_margin = raw_home - raw_away
    total = min(max(raw_home + raw_away, 5.5), 13.5)
    margin = (probability_margin * 0.58) + (rating_margin * 0.42)
    home_expected = min(max((total + margin) / 2, 1.7), 9.5)
    away_expected = min(max((total - margin) / 2, 1.7), 9.5)

    return {
        "home_expected": home_expected,
        "away_expected": away_expected,
        "expected_total": home_expected + away_expected,
        "expected_margin": home_expected - away_expected,
    }


def sample_team_runs(expected_runs: float, rng: random.Random) -> int:
    """Sample plausible MLB team runs with a lightweight distribution."""
    std_dev = max(1.25, math.sqrt(max(expected_runs, 0.5)) * 1.05)
    return max(0, int(round(rng.gauss(expected_runs, std_dev))))


def sample_projected_score(
    home_team: str,
    away_team: str,
    home_probability: float,
    rng: random.Random,
) -> tuple[int, int]:
    """Sample one plausible MLB final score."""
    expected = estimate_mlb_expected_scores(home_team, away_team, home_probability)
    home_score = sample_team_runs(float(expected["home_expected"]), rng)
    away_score = sample_team_runs(float(expected["away_expected"]), rng)

    if home_score == away_score:
        if rng.random() < home_probability:
            home_score += 1
        else:
            away_score += 1

    return home_score, away_score


def split_runs_by_inning(total_runs: int, rng: random.Random) -> list[int]:
    """Split projected runs into nine innings."""
    innings = [0] * 9

    for _ in range(max(total_runs, 0)):
        inning = rng.choices(
            population=list(range(9)),
            weights=[1.0, 1.0, 1.05, 1.1, 1.05, 1.0, 1.05, 1.08, 0.95],
            k=1,
        )[0]
        innings[inning] += 1

    return innings


def build_game_score_projection(
    home_team: str,
    away_team: str,
    home_probability: float,
    seed: int,
    simulations: int = SCORE_SIMULATION_COUNT,
) -> dict:
    """Build final score, range, and inning-by-inning projections."""
    rng = random.Random(seed)
    samples = []

    for _ in range(simulations):
        home_score, away_score = sample_projected_score(home_team, away_team, home_probability, rng)
        samples.append(
            {
                "Home Score": home_score,
                "Away Score": away_score,
                "Total": home_score + away_score,
                "Home Margin": home_score - away_score,
                "Home Win": home_score > away_score,
            }
        )

    sample_df = pd.DataFrame(samples)
    projected_home_score = int(round(sample_df["Home Score"].mean()))
    projected_away_score = int(round(sample_df["Away Score"].mean()))

    if projected_home_score == projected_away_score:
        if home_probability >= 0.5:
            projected_home_score += 1
        else:
            projected_away_score += 1

    flow_rng = random.Random(seed + 17)
    home_innings = split_runs_by_inning(projected_home_score, flow_rng)
    away_innings = split_runs_by_inning(projected_away_score, flow_rng)
    flow_rows = []
    away_total = 0
    home_total = 0

    for inning, (away_runs, home_runs) in enumerate(zip(away_innings, home_innings), start=1):
        away_total += away_runs
        home_total += home_runs
        flow_rows.append(
            {
                "Inning": inning,
                away_team: away_runs,
                home_team: home_runs,
                f"{away_team} Total": away_total,
                f"{home_team} Total": home_total,
                "Home Win %": min(max(home_probability + ((home_total - away_total) * 0.035), 0.01), 0.99),
            }
        )

    expected = estimate_mlb_expected_scores(home_team, away_team, home_probability)
    return {
        "projected_home_score": projected_home_score,
        "projected_away_score": projected_away_score,
        "projected_total": projected_home_score + projected_away_score,
        "projected_winner": home_team if projected_home_score > projected_away_score else away_team,
        "home_win_rate": float(sample_df["Home Win"].mean()),
        "total_range": (
            int(round(sample_df["Total"].quantile(0.1))),
            int(round(sample_df["Total"].quantile(0.9))),
        ),
        "margin_range": (
            int(round(sample_df["Home Margin"].quantile(0.1))),
            int(round(sample_df["Home Margin"].quantile(0.9))),
        ),
        "expected": expected,
        "flow": pd.DataFrame(flow_rows),
        "samples": sample_df,
    }


def render_score_projection_card(home_team: str, away_team: str, projection: dict) -> None:
    """Render projected MLB final score."""
    home_score = int(projection["projected_home_score"])
    away_score = int(projection["projected_away_score"])
    winner = str(projection["projected_winner"])
    total_low, total_high = projection["total_range"]
    margin_low, margin_high = projection["margin_range"]
    signals = [
        f"Total {total_low}-{total_high}",
        f"Home margin {margin_low:+d} to {margin_high:+d}",
        f"{SCORE_SIMULATION_COUNT:,} sims",
    ]
    signal_html = "".join(
        f'<span class="signal-pill">{html.escape(signal)}</span>'
        for signal in signals
    )
    away_logo = team_logo_url(get_team_id_from_strength(away_team))
    home_logo = team_logo_url(get_team_id_from_strength(home_team))

    st.html(
        f"""
        <div class="score-sim-card" style="--team-color: {html.escape(get_team_color(winner))};">
            <div class="score-topline">
                <div class="dashboard-label">Projected Score</div>
                <span class="score-chip">{html.escape(winner)}</span>
            </div>
            <div class="score-sim-board">
                <div class="score-sim-team">
                    <img class="score-sim-logo" src="{html.escape(away_logo)}" alt="{html.escape(away_team)} logo">
                    <div>
                        <div class="score-sim-name">{html.escape(away_team)}</div>
                        <div class="score-sim-role">Away</div>
                    </div>
                    <div class="score-sim-points">{away_score}</div>
                </div>
                <div class="score-sim-center">AT</div>
                <div class="score-sim-team">
                    <img class="score-sim-logo" src="{html.escape(home_logo)}" alt="{html.escape(home_team)} logo">
                    <div>
                        <div class="score-sim-name">{html.escape(home_team)}</div>
                        <div class="score-sim-role">Home</div>
                    </div>
                    <div class="score-sim-points">{home_score}</div>
                </div>
            </div>
            <div class="signal-grid">{signal_html}</div>
        </div>
        """
    )


def render_game_score_simulation(
    home_team: str,
    away_team: str,
    home_probability: float,
    seed: int,
) -> dict:
    """Render projected final score and inning table."""
    projection = build_game_score_projection(home_team, away_team, home_probability, seed)
    render_score_projection_card(home_team, away_team, projection)

    with st.expander("Inning-by-inning projection", expanded=False):
        flow = projection["flow"].copy()
        flow["Home Win %"] = flow["Home Win %"].map("{:.1%}".format)
        st.dataframe(flow, width="stretch", hide_index=True)

    return projection


def get_matchup_probabilities(
    home_team: str,
    away_team: str,
    probability_cache: dict[tuple[str, str], dict] | None = None,
) -> dict:
    """Return cached MLB prediction details for a home/away matchup."""
    key = (home_team, away_team)

    if probability_cache is not None and key in probability_cache:
        return probability_cache[key]

    details = predict_game_details(home_team, away_team)

    if probability_cache is not None:
        probability_cache[key] = details

    return details


def series_home_team(higher_seed_team: str, lower_seed_team: str, game_number: int, best_of: int) -> str:
    """Return home team for MLB-style postseason formats."""
    if best_of == 3:
        return higher_seed_team

    if best_of == 5:
        pattern = [higher_seed_team, higher_seed_team, lower_seed_team, lower_seed_team, higher_seed_team]
        return pattern[game_number - 1]

    pattern = [
        higher_seed_team,
        higher_seed_team,
        lower_seed_team,
        lower_seed_team,
        higher_seed_team,
        lower_seed_team,
        higher_seed_team,
    ]
    return pattern[game_number - 1]


def simulate_series_once(
    higher_seed_team: str,
    lower_seed_team: str,
    best_of: int,
    rng: random.Random,
    probability_cache: dict[tuple[str, str], dict],
) -> dict:
    """Simulate one MLB series."""
    wins_needed = (best_of // 2) + 1
    wins = {higher_seed_team: 0, lower_seed_team: 0}
    game_rows = []

    for game_number in range(1, best_of + 1):
        home_team = series_home_team(higher_seed_team, lower_seed_team, game_number, best_of)
        away_team = lower_seed_team if home_team == higher_seed_team else higher_seed_team
        details = get_matchup_probabilities(home_team, away_team, probability_cache)
        home_probability = float(details["home_probability"])
        winner = home_team if rng.random() < home_probability else away_team
        wins[winner] += 1
        game_rows.append(
            {
                "Game": game_number,
                "Home": home_team,
                "Away": away_team,
                "Winner": winner,
                "Home Win %": home_probability,
            }
        )

        if wins[winner] >= wins_needed:
            break

    series_winner = higher_seed_team if wins[higher_seed_team] > wins[lower_seed_team] else lower_seed_team
    return {
        "winner": series_winner,
        "games": len(game_rows),
        "wins": wins,
        "game_rows": game_rows,
        "result": f"{series_winner} in {len(game_rows)}",
    }


def simulate_series(
    higher_seed_team: str,
    lower_seed_team: str,
    simulations: int,
    seed: int,
    best_of: int = 7,
) -> dict:
    """Simulate a best-of MLB series."""
    rng = random.Random(seed)
    probability_cache: dict[tuple[str, str], dict] = {}
    winner_counts = {higher_seed_team: 0, lower_seed_team: 0}
    result_counts: dict[str, int] = {}

    for _ in range(simulations):
        result = simulate_series_once(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            best_of=best_of,
            rng=rng,
            probability_cache=probability_cache,
        )
        winner_counts[result["winner"]] += 1
        result_counts[result["result"]] = result_counts.get(result["result"], 0) + 1

    result_table = pd.DataFrame(
        [
            {"Result": result, "Probability": count / simulations, "Count": count}
            for result, count in result_counts.items()
        ]
    ).sort_values("Probability", ascending=False)
    favorite = max(winner_counts, key=winner_counts.get)

    return {
        "winner_counts": winner_counts,
        "series_probabilities": {
            team: count / simulations for team, count in winner_counts.items()
        },
        "predicted_winner": favorite,
        "favorite_probability": winner_counts[favorite] / simulations,
        "result_table": result_table,
        "probability_cache": probability_cache,
    }


def render_series_score_card(
    team_a: str,
    team_b: str,
    probability_a: float,
    probability_b: float,
    best_of: int,
    predicted_winner: str,
) -> None:
    """Render compact series score/probability card."""
    st.html(
        f"""
        <div class="score-card" style="--team-color: {html.escape(get_team_color(predicted_winner))};">
            <div class="score-topline">
                <span class="dashboard-label">Best of {best_of}</span>
                <span class="score-chip">{html.escape(predicted_winner)}</span>
            </div>
            <div class="score-row">
                <span class="score-team-wrap">
                    <img class="score-logo" src="{html.escape(team_logo_url(get_team_id_from_strength(team_a)))}" alt="{html.escape(team_a)} logo">
                    <span class="score-team">{html.escape(team_a)}</span>
                </span>
                <span class="score-wins">{probability_a:.0%}</span>
            </div>
            <div class="score-row">
                <span class="score-team-wrap">
                    <img class="score-logo" src="{html.escape(team_logo_url(get_team_id_from_strength(team_b)))}" alt="{html.escape(team_b)} logo">
                    <span class="score-team">{html.escape(team_b)}</span>
                </span>
                <span class="score-wins">{probability_b:.0%}</span>
            </div>
        </div>
        """
    )


def simulate_bracket_once(
    seeds: list[str],
    rng: random.Random,
    probability_cache: dict[tuple[str, str], dict],
) -> tuple[str, list[dict]]:
    """Simulate one simplified 12-team MLB playoff bracket."""
    path_rows = []

    def play(label: str, higher: str, lower: str, best_of: int) -> str:
        result = simulate_series_once(
            higher_seed_team=higher,
            lower_seed_team=lower,
            best_of=best_of,
            rng=rng,
            probability_cache=probability_cache,
        )
        path_rows.append(
            {
                "Round": label,
                "Matchup": f"{higher} vs {lower}",
                "Winner": result["winner"],
                "Result": result["result"],
            }
        )
        return str(result["winner"])

    wild_1 = play("Wild Card", seeds[4], seeds[11], 3)
    wild_2 = play("Wild Card", seeds[5], seeds[10], 3)
    wild_3 = play("Wild Card", seeds[6], seeds[9], 3)
    wild_4 = play("Wild Card", seeds[7], seeds[8], 3)
    div_1 = play("Division", seeds[0], wild_4, 5)
    div_2 = play("Division", seeds[1], wild_3, 5)
    div_3 = play("Division", seeds[2], wild_2, 5)
    div_4 = play("Division", seeds[3], wild_1, 5)
    semi_1 = play("League Semifinal", div_1, div_4, 7)
    semi_2 = play("League Semifinal", div_2, div_3, 7)
    champion = play("Championship", semi_1, semi_2, 7)
    return champion, path_rows


def simulate_full_bracket(seeds: list[str], simulations: int, seed: int) -> dict:
    """Simulate a simplified MLB postseason bracket."""
    rng = random.Random(seed)
    probability_cache: dict[tuple[str, str], dict] = {}
    champion_counts = {team: 0 for team in seeds}
    finals_counts = {team: 0 for team in seeds}

    for _ in range(simulations):
        champion, path_rows = simulate_bracket_once(seeds, rng, probability_cache)
        champion_counts[champion] += 1

        for row in path_rows:
            if row["Round"] == "Championship":
                matchup = str(row["Matchup"])
                for team in seeds:
                    if team in matchup:
                        finals_counts[team] += 1

    results = pd.DataFrame(
        [
            {
                "Team": team,
                "Finals Probability": finals_counts[team] / simulations,
                "Championship Probability": champion_counts[team] / simulations,
                "Titles Won": champion_counts[team],
            }
            for team in seeds
        ]
    ).sort_values("Championship Probability", ascending=False)
    _, sample_path = simulate_bracket_once(seeds, random.Random(seed + 9000), probability_cache)
    return {
        "results": results,
        "sample_path": pd.DataFrame(sample_path),
        "probability_cache": probability_cache,
    }


def game_card_sort_key(game: pd.Series, snapshot: dict) -> tuple[int, float, str]:
    """Sort live games first, scheduled games next, completed games last."""
    status = snapshot.get("STATUS") or game.get("STATUS", "")
    status_code = game.get("STATUS_CODE", "")
    game_time = game_datetime_utc(game)
    timestamp = game_time.timestamp() if game_time is not None else 0.0

    if is_live_status_value(status, status_code, game.get("IS_LIVE")):
        return (0, timestamp, str(game.get("GAME_PK", "")))

    if is_final_status_value(status, status_code, game.get("IS_FINAL")):
        return (2, -timestamp, str(game.get("GAME_PK", "")))

    return (1, timestamp, str(game.get("GAME_PK", "")))


def build_display_game_rows(games: pd.DataFrame) -> list[tuple[pd.Series, dict]]:
    """Attach predictions and live snapshots, then return display-ordered rows."""
    if games.empty:
        return []

    predictions = build_game_predictions(games)

    if predictions.empty:
        predictions = games.copy()

    display_rows = []

    for _, game in predictions.iterrows():
        snapshot = build_live_game_snapshot(game)
        display_rows.append((game, snapshot))

    return sorted(display_rows, key=lambda row: game_card_sort_key(row[0], row[1]))


def render_game_cards(games: pd.DataFrame) -> None:
    """Render MLB game cards."""
    display_rows = build_display_game_rows(games)

    if not display_rows:
        st.info("No MLB games available from the schedule feed.")
        return

    try:
        requested_win_probability_key = st.query_params.get(WIN_PROB_QUERY_PARAM, "")
    except Exception:
        requested_win_probability_key = ""

    if isinstance(requested_win_probability_key, list):
        requested_win_probability_key = requested_win_probability_key[0] if requested_win_probability_key else ""

    requested_win_probability_key = str(requested_win_probability_key or "")
    requested_win_probability_game = None
    column_count = 1 if len(display_rows) == 1 else min(2, len(display_rows))
    columns = st.columns(column_count)

    for index, (game, snapshot) in enumerate(display_rows):
        game_pk = str(game.get("GAME_PK", "") or "")
        game_key = game_pk or f"row_{index}"
        card_key = safe_streamlit_dom_key(game_row_key(game) or game_key)
        live_wp_label = live_win_probability_label(game, snapshot)

        if requested_win_probability_key and game_row_key(game) == requested_win_probability_key:
            requested_win_probability_game = game.to_dict()

        with columns[index % column_count]:
            with st.container(key=f"mlb-game-card-wrap-{card_key}"):
                render_game_card(game, live_snapshot=snapshot)

                if live_wp_label:
                    with st.container(key=f"mlb-live-wp-wrap-{card_key}"):
                        if st.button(
                            live_wp_label,
                            key=f"mlb_live_wp_{card_key}",
                            width="stretch",
                            help="Open win probability graph",
                        ):
                            st.session_state[WIN_PROB_DIALOG_STATE_KEY] = game.to_dict()

                if st.button(
                    "Open game center",
                    key=f"mlb_game_details_{game_key}",
                    width="stretch",
                ):
                    st.session_state.pop("mlb_selected_player_stats", None)
                    st.session_state[GAME_DETAIL_STATE_KEY] = game.to_dict()

    selected_game_detail = st.session_state.get(GAME_DETAIL_STATE_KEY)

    if selected_game_detail:
        render_game_detail_dialog(selected_game_detail)

    selected_win_probability_game = requested_win_probability_game or st.session_state.get(WIN_PROB_DIALOG_STATE_KEY)

    if selected_win_probability_game:
        render_win_probability_dialog(selected_win_probability_game)


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def render_today_live_fragment() -> None:
    """Render the MLB Today tab with live score refreshes."""
    today_games = load_today_games()

    if today_games.empty:
        st.info("No MLB games are scheduled for today.")
        upcoming = load_next_upcoming_games()

        if upcoming.empty:
            st.info("No upcoming MLB games were found in the next 14 days.")
            return

        slate_date = pd.to_datetime(upcoming.iloc[0]["GAME_DATE"]).strftime("%A, %B %-d")
        render_section_kicker("Upcoming Games", slate_date)
        render_game_cards(upcoming)
        return

    render_section_kicker("Today's Games", f"Live refresh every {LIVE_REFRESH_SECONDS}s")
    render_game_cards(today_games)


def get_home_slate() -> tuple[pd.DataFrame, str, bool]:
    """Return today's slate, or the next upcoming slate if today is empty."""
    today_games = load_today_games()

    if not today_games.empty:
        return today_games, "Today", False

    upcoming = load_next_upcoming_games()

    if upcoming.empty:
        return pd.DataFrame(), "No upcoming slate", True

    slate_date = pd.to_datetime(upcoming.iloc[0]["GAME_DATE"]).strftime("%A, %B %-d")
    return upcoming, slate_date, True


def game_datetime_utc(game: pd.Series) -> pd.Timestamp | None:
    """Return one game datetime as UTC."""
    parsed = pd.to_datetime(game.get("GAME_DATETIME"), errors="coerce")

    if pd.isna(parsed):
        return None

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize("UTC")

    return parsed.tz_convert(ZoneInfo("UTC"))


def is_truthy_schedule_value(value: object) -> bool:
    """Interpret boolean-ish schedule fields."""
    if value is None or pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes"}


def is_final_status_value(status: object, status_code: object, is_final_value: object = None) -> bool:
    """Return whether schedule status values represent a final game."""
    status_text = str(status or "").strip().lower()
    code = str(status_code or "").strip().upper()
    return (
        is_truthy_schedule_value(is_final_value)
        or status_text in {"final", "game over", "completed early"}
        or "final" in status_text
        or code in {"F", "O"}
    )


def is_live_status_value(status: object, status_code: object, is_live_value: object = None) -> bool:
    """Return whether schedule status values represent a live game."""
    status_text = str(status or "").strip().lower()
    code = str(status_code or "").strip().upper()
    return (
        not is_final_status_value(status, status_code)
        and (
            is_truthy_schedule_value(is_live_value)
            or status_text in {"in progress", "live"}
            or code in {"I", "M", "N"}
        )
    )


def is_pregame_status_value(status: object, status_code: object) -> bool:
    """Return whether a status means the game is still waiting for first pitch."""
    status_text = str(status or "").strip().lower()
    code = str(status_code or "").strip().upper()

    if is_final_status_value(status, status_code) or is_live_status_value(status, status_code):
        return False

    if any(term in status_text for term in ["postponed", "cancel", "suspended"]):
        return False

    if code in {"C", "D", "DR", "DS"} and "delayed" not in status_text:
        return False

    return True


def is_final_schedule_game(game: pd.Series) -> bool:
    """Return whether a schedule row is final/completed."""
    return is_final_status_value(
        game.get("STATUS", ""),
        game.get("STATUS_CODE", ""),
        game.get("IS_FINAL"),
    )


def is_live_schedule_game(game: pd.Series) -> bool:
    """Return whether a schedule row is live."""
    return is_live_status_value(
        game.get("STATUS", ""),
        game.get("STATUS_CODE", ""),
        game.get("IS_LIVE"),
    )


def is_pregame_schedule_game(game: pd.Series, status_override: object = None) -> bool:
    """Return whether a game is scheduled/pre-game and not live or final."""
    status = game.get("STATUS", "") if status_override is None else status_override
    return is_pregame_status_value(status, game.get("STATUS_CODE", ""))


def is_future_schedule_game(game: pd.Series) -> bool:
    """Return whether a game has not started yet."""
    game_time = game_datetime_utc(game)
    is_pregame = is_pregame_schedule_game(game)

    if game_time is None:
        return is_pregame

    now = pd.Timestamp.now(tz=ZoneInfo("UTC"))

    if game_time > now:
        return is_pregame

    return is_pregame and game_time >= now - timedelta(hours=PREGAME_OVERDUE_WINDOW_HOURS)


def format_game_start_countdown(game: pd.Series, status_override: object = None) -> str:
    """Format countdown text for a game row, preserving overdue pregame games."""
    return format_start_countdown(
        game.get("GAME_DATETIME"),
        show_starting_soon=is_pregame_schedule_game(game, status_override=status_override),
    )


def split_slate_states(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a schedule frame into live, upcoming, and final rows."""
    if games.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    live_mask = games.apply(is_live_schedule_game, axis=1)
    final_mask = games.apply(is_final_schedule_game, axis=1)
    future_mask = games.apply(is_future_schedule_game, axis=1)
    live = games[live_mask].copy().sort_values(["GAME_DATETIME", "GAME_PK"])
    upcoming = games[future_mask].copy().sort_values(["GAME_DATETIME", "GAME_PK"])
    finals = games[final_mask].copy().sort_values(["GAME_DATETIME", "GAME_PK"], ascending=[False, True])
    return (
        live.reset_index(drop=True),
        upcoming.reset_index(drop=True),
        finals.reset_index(drop=True),
    )


def get_next_up_game(today_games: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Return the next future game, skipping completed games."""
    _, upcoming_today, _ = split_slate_states(today_games)

    if not upcoming_today.empty:
        return upcoming_today.iloc[0], "Today"

    upcoming = load_next_upcoming_games()

    if upcoming.empty:
        return None, ""

    _, future_games, _ = split_slate_states(upcoming)

    if future_games.empty:
        future_games = upcoming.copy().sort_values(["GAME_DATETIME", "GAME_PK"])

    if future_games.empty:
        return None, ""

    slate_label = pd.to_datetime(future_games.iloc[0]["GAME_DATE"]).strftime("%A, %B %-d")
    return future_games.iloc[0], slate_label


def format_matchup(game: pd.Series) -> str:
    """Format one matchup label."""
    return f"{game.get('AWAY_TEAM', 'Away')} at {game.get('HOME_TEAM', 'Home')}"


def build_home_game_signals(games: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Build compact home-dashboard signals from the slate."""
    predictions = build_game_predictions(games)

    if predictions.empty:
        return [], pd.DataFrame()

    predictions = predictions.copy()
    predictions["FAVORITE_EDGE"] = predictions["WINNER_PROBABILITY"].astype(float) - 0.5
    predictions["CLOSENESS"] = (predictions["WINNER_PROBABILITY"].astype(float) - 0.5).abs()
    predictions["MODEL_ELO_GAP"] = (
        predictions["MODEL_PROBABILITY"].astype(float)
        - predictions["ELO_PROBABILITY"].astype(float)
    ).abs()
    expected_totals = []

    for _, game in predictions.iterrows():
        try:
            expected = estimate_mlb_expected_scores(
                home_team=str(game["HOME_TEAM"]),
                away_team=str(game["AWAY_TEAM"]),
                home_probability=float(game["HOME_WIN_PROBABILITY"]),
            )
            expected_totals.append(float(expected["expected_total"]))
        except Exception:
            expected_totals.append(math.nan)

    predictions["EXPECTED_TOTAL"] = expected_totals
    strongest = predictions.sort_values("WINNER_PROBABILITY", ascending=False).iloc[0]
    closest = predictions.sort_values("CLOSENESS", ascending=True).iloc[0]
    disagreement = predictions.sort_values("MODEL_ELO_GAP", ascending=False).iloc[0]
    total_rows = predictions.dropna(subset=["EXPECTED_TOTAL"])
    signal_cards = [
        {
            "label": "Strongest Pick",
            "title": str(strongest["PREDICTED_WINNER"]),
            "value": f"{float(strongest['WINNER_PROBABILITY']):.0%}",
            "note": format_matchup(strongest),
            "color": "#166534",
        },
        {
            "label": "Closest Game",
            "title": format_matchup(closest),
            "value": f"{float(closest['WINNER_PROBABILITY']):.0%}",
            "note": f"Lean: {closest['PREDICTED_WINNER']}",
            "color": "#7c3aed",
        },
        {
            "label": "Price Watch",
            "title": format_matchup(disagreement),
            "value": f"{float(disagreement['MODEL_ELO_GAP']):.0%}",
            "note": "Largest pricing disagreement",
            "color": "#b91c1c",
        },
    ]

    if not total_rows.empty:
        highest_total = total_rows.sort_values("EXPECTED_TOTAL", ascending=False).iloc[0]
        signal_cards.append(
            {
                "label": "Projected Runs",
                "title": format_matchup(highest_total),
                "value": f"{float(highest_total['EXPECTED_TOTAL']):.1f}",
                "note": "Highest estimated total",
                "color": "#c2410c",
            }
        )

    return signal_cards, predictions


def format_home_game_score_label(game: pd.Series, mode: str) -> str:
    """Return compact score/countdown text for a home row."""
    if mode == "upcoming":
        return format_game_start_countdown(game) or format_game_time(game.get("GAME_DATETIME"))

    live_snapshot = build_live_game_snapshot(game) if mode == "live" else {}
    away_score = format_score(live_snapshot.get("AWAY_SCORE") if live_snapshot else game.get("AWAY_SCORE"))
    home_score = format_score(live_snapshot.get("HOME_SCORE") if live_snapshot else game.get("HOME_SCORE"))

    if away_score or home_score:
        return f"{away_score or '-'} - {home_score or '-'}"

    return str(game.get("STATUS", "") or "")


def render_home_game_rows(title: str, games: pd.DataFrame, mode: str, empty_message: str, limit: int = 4) -> None:
    """Render compact game summary rows for Home."""
    render_section_kicker(title)

    if games.empty:
        st.info(empty_message)
        return

    row_html = []

    for _, game in games.head(limit).iterrows():
        away_team = str(game.get("AWAY_TEAM", "") or "Away")
        home_team = str(game.get("HOME_TEAM", "") or "Home")
        away_logo = team_logo_url(game.get("AWAY_TEAM_ID"))
        home_logo = team_logo_url(game.get("HOME_TEAM_ID"))
        score_label = format_home_game_score_label(game, mode)
        status = str(game.get("STATUS", "") or "Scheduled")
        time_label = format_game_time(game.get("GAME_DATETIME"))
        venue = str(game.get("VENUE", "") or "")
        meta = " / ".join([value for value in [status, time_label, venue] if value])
        row_html.append(
            f"""
            <div class="home-game-row">
                <div>
                    <div class="home-game-matchup">
                        <img class="home-game-logo" src="{html.escape(away_logo)}" alt="{html.escape(away_team)} logo">
                        <span class="home-game-team">{html.escape(away_team)}</span>
                        <span class="home-game-at">at</span>
                        <img class="home-game-logo" src="{html.escape(home_logo)}" alt="{html.escape(home_team)} logo">
                        <span class="home-game-team">{html.escape(home_team)}</span>
                    </div>
                    <div class="home-game-meta">{html.escape(meta)}</div>
                </div>
                <div class="home-game-score">{html.escape(score_label)}</div>
            </div>
            """
        )

    st.html("".join(row_html))

    if len(games) > limit:
        st.caption(f"{len(games) - limit} more in the Today tab.")


def render_home_next_up(game: pd.Series | None, slate_label: str) -> None:
    """Render the true next future game."""
    render_section_kicker("Next Up")

    if game is None:
        st.info("No future MLB games were found in the schedule feed.")
        return

    countdown = format_game_start_countdown(game)
    feed = load_live_game_feed(str(game.get("GAME_PK", "") or ""))
    away_team = str(game.get("AWAY_TEAM", "") or "Away")
    home_team = str(game.get("HOME_TEAM", "") or "Home")
    away_pitcher = build_probable_pitcher_profile(game, feed, "away")
    home_pitcher = build_probable_pitcher_profile(game, feed, "home")
    meta = " / ".join(
        [
            value
            for value in [
                slate_label,
                format_game_time(game.get("GAME_DATETIME")),
                str(game.get("VENUE", "") or ""),
            ]
            if value
        ]
    )
    countdown_html = (
        f'<span class="mlb-countdown">{html.escape(countdown)}</span>'
        if countdown
        else ""
    )

    st.html(
        f"""
        <div class="next-up-card">
            <div class="next-up-head">
                <div>
                    <div class="dashboard-label">Next First Pitch</div>
                    <div class="next-up-title">{html.escape(format_matchup(game))}</div>
                    <div class="next-up-meta">{html.escape(meta)}</div>
                </div>
                {countdown_html}
            </div>
            <div class="next-up-pitchers">
                {build_pitcher_profile_card_html(away_pitcher, away_team)}
                {build_pitcher_profile_card_html(home_pitcher, home_team)}
            </div>
        </div>
        """
    )


def render_home_spotlight(games: pd.DataFrame, slate_label: str, is_upcoming: bool) -> None:
    """Render one next-game or best-matchup spotlight."""
    if games.empty:
        st.info("No upcoming MLB games were found.")
        return

    sorted_games = games.sort_values(["GAME_DATETIME", "GAME_PK"]).reset_index(drop=True)
    spotlight = sorted_games.iloc[0]
    predictions = build_game_predictions(pd.DataFrame([spotlight.to_dict()]))
    game = predictions.iloc[0] if not predictions.empty else spotlight
    matchup = format_matchup(game)
    game_time = format_game_time(game.get("GAME_DATETIME"))
    countdown = format_game_start_countdown(game)
    note = countdown or game_time
    title = "Upcoming Spotlight" if is_upcoming else "Next First Pitch"
    cards = [
        {
            "label": title,
            "title": matchup,
            "value": str(game.get("STATUS", "Scheduled") or "Scheduled"),
            "note": f"{slate_label} / {note}",
            "color": "#1d4ed8",
        }
    ]

    if not predictions.empty:
        expected = estimate_mlb_expected_scores(
            home_team=str(game["HOME_TEAM"]),
            away_team=str(game["AWAY_TEAM"]),
            home_probability=float(game["HOME_WIN_PROBABILITY"]),
        )
        cards.append(
            {
                "label": "Spotlight Pick",
                "title": str(game["PREDICTED_WINNER"]),
                "value": f"{float(game['WINNER_PROBABILITY']):.0%}",
                "note": f"Projected total {float(expected['expected_total']):.1f}",
                "color": "#166534",
            }
        )

    render_home_signal_cards(cards)


def render_home_power_and_form(strength: pd.DataFrame) -> None:
    """Render top teams and form movers."""
    if strength.empty:
        st.info("Team strength data is not available.")
        return

    ranked = strength.sort_values("ELO", ascending=False).reset_index(drop=True)
    form_best = strength.sort_values("ROLLING_RUN_DIFF_10", ascending=False).head(4)
    form_worst = strength.sort_values("ROLLING_RUN_DIFF_10", ascending=True).head(4)
    left_col, right_col = st.columns(2)

    with left_col:
        render_section_kicker("Power Rankings")
        render_team_mini_rows(
            [
                {
                    "team": row["TEAM_NAME"],
                    "meta": (
                        f"Win {float(row['SEASON_WIN_PCT']):.0%} / "
                        f"Run diff {float(row['SEASON_RUN_DIFF_PER_GAME']):+.1f}"
                    ),
                    "score": f"{float(row['ELO']):.0f}",
                }
                for _, row in ranked.head(5).iterrows()
            ]
        )

    with right_col:
        render_section_kicker("Risers and Fallers")
        rows = []

        for _, row in form_best.head(2).iterrows():
            rows.append(
                {
                    "team": row["TEAM_NAME"],
                    "meta": f"Last 10 win {float(row['ROLLING_WIN_PCT_10']):.0%}",
                    "score": f"{float(row['ROLLING_RUN_DIFF_10']):+.1f}",
                }
            )

        for _, row in form_worst.head(2).iterrows():
            rows.append(
                {
                    "team": row["TEAM_NAME"],
                    "meta": f"Last 10 win {float(row['ROLLING_WIN_PCT_10']):.0%}",
                    "score": f"{float(row['ROLLING_RUN_DIFF_10']):+.1f}",
                }
            )

        render_team_mini_rows(rows)


def render_home_quick_actions() -> None:
    """Render quick navigation buttons for common workflows."""
    def set_view(view: str) -> None:
        st.session_state["mlb_selected_view"] = view

    render_section_kicker("Quick Actions")
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    actions = [
        (action_col1, "Predict Game", "Game"),
        (action_col2, "Simulate Series", "Series"),
        (action_col3, "Simulate Bracket", "Bracket"),
        (action_col4, "Power Rankings", "Teams"),
    ]

    for column, label, view in actions:
        with column:
            st.button(
                label,
                width="stretch",
                key=f"mlb_home_action_{view}",
                on_click=set_view,
                args=(view,),
            )


def render_home_view() -> None:
    """Render MLB home dashboard."""
    strength = load_team_strength()
    today_games = load_today_games()
    live_games, upcoming_today, final_games = split_slate_states(today_games)
    next_game, next_label = get_next_up_game(today_games)
    next_value = "TBD"
    next_note = "No future games found"

    if next_game is not None:
        next_value = format_game_start_countdown(next_game) or format_game_time(next_game.get("GAME_DATETIME"))
        next_note = format_matchup(next_game)

    render_dashboard_cards(
        [
            {"label": "Live Now", "value": str(len(live_games)), "note": "Games currently in progress", "color": "#b91c1c"},
            {"label": "Upcoming Today", "value": str(len(upcoming_today)), "note": "Games still waiting for first pitch", "color": "#1d4ed8"},
            {"label": "Final Today", "value": str(len(final_games)), "note": "Completed games", "color": "#166534"},
            {"label": "Next Up", "value": next_value, "note": next_note, "color": "#7c3aed"},
        ]
    )

    render_home_next_up(next_game, next_label)

    left_col, right_col = st.columns(2)

    with left_col:
        render_home_game_rows(
            "Live Now",
            live_games,
            mode="live",
            empty_message="No MLB games are live right now.",
            limit=3,
        )

    with right_col:
        render_home_game_rows(
            "Recently Final",
            final_games,
            mode="final",
            empty_message="No MLB games have gone final today.",
            limit=3,
        )

    if not upcoming_today.empty:
        render_home_game_rows(
            "Upcoming Today",
            upcoming_today,
            mode="upcoming",
            empty_message="No upcoming games remain today.",
            limit=4,
        )

    render_home_power_and_form(strength)
    render_home_quick_actions()


def render_game_view() -> None:
    """Render manual MLB game predictor."""
    teams = get_available_teams()

    if not teams:
        st.warning("Missing MLB team strength data. Run the MLB refresh commands first.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        home_team = st.selectbox("Home", teams, index=0, key="mlb_home_team")

    with col2:
        away_team = st.selectbox("Away", teams, index=min(1, len(teams) - 1), key="mlb_away_team")

    with col3:
        st.write("")
        predict_clicked = st.button("Predict", type="primary", width="stretch")

    if home_team == away_team:
        st.warning("Choose two different teams.")
        return

    render_matchup_preview(away_team, home_team, "Away", "Home")
    prediction_state_key = "mlb_game_last_prediction"

    if predict_clicked:
        details = predict_game_details(home_team, away_team)
        projection = build_game_score_projection(
            home_team=home_team,
            away_team=away_team,
            home_probability=float(details["home_probability"]),
            seed=build_stable_matchup_seed(home_team, away_team),
        )
        report = create_prediction_report(
            title=f"MLB Game Prediction: {away_team} at {home_team}",
            rows=[
                f"Predicted winner: {details['winner']} ({details['winner_probability']:.1%})",
                f"{home_team}: {details['home_probability']:.1%}",
                f"{away_team}: {details['away_probability']:.1%}",
                f"Projected score: {away_team} {projection['projected_away_score']}, {home_team} {projection['projected_home_score']}",
                f"Win chance: {details['model_probability']:.1%}",
                f"Power rating chance: {details['elo_probability']:.1%}",
            ],
        )
        st.session_state[prediction_state_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "details": details,
            "projection_seed": build_stable_matchup_seed(home_team, away_team),
            "report": report,
        }

    prediction_state = st.session_state.get(prediction_state_key)

    if prediction_state and (
        prediction_state.get("home_team") == home_team
        and prediction_state.get("away_team") == away_team
    ):
        details = prediction_state["details"]
        winner = str(details["winner"])
        winner_probability = float(details["winner_probability"])

        render_prediction_result_card(
            label="Game prediction",
            winner=winner,
            probability=winner_probability,
            details=details,
            note=f"{away_team} at {home_team}",
        )
        render_matchup_cards(
            team_a=away_team,
            team_b=home_team,
            probability_a=float(details["away_probability"]),
            probability_b=float(details["home_probability"]),
            winner=winner,
        )
        render_game_score_simulation(
            home_team=home_team,
            away_team=away_team,
            home_probability=float(details["home_probability"]),
            seed=int(prediction_state["projection_seed"]),
        )

        with st.expander("Pricing details", expanded=False):
            st.write(f"Win chance: {float(details['model_probability']):.1%}")
            st.write(f"Power rating chance: {float(details['elo_probability']):.1%}")
            st.write(f"Blended home probability: {float(details['home_probability']):.1%}")
            render_report_download(
                report_text=str(prediction_state["report"]),
                file_name="mlb_game_prediction_report.txt",
                key="mlb_game_report",
            )


def render_series_view() -> None:
    """Render MLB best-of series simulator."""
    teams = get_available_teams()

    if not teams:
        st.warning("Missing MLB team strength data. Run the MLB refresh commands first.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        higher_seed_team = st.selectbox("Higher Seed", teams, index=0, key="mlb_series_higher")

    with col2:
        lower_seed_team = st.selectbox("Lower Seed", teams, index=min(1, len(teams) - 1), key="mlb_series_lower")

    with col3:
        best_of = st.selectbox("Format", [3, 5, 7], index=2, key="mlb_series_best_of")

    if higher_seed_team == lower_seed_team:
        st.warning("Choose two different teams.")
        return

    render_matchup_preview(higher_seed_team, lower_seed_team, "Higher Seed", "Lower Seed")

    settings_col, seed_col, action_col = st.columns([2, 1, 1])

    with settings_col:
        simulations = st.slider(
            "Series simulations",
            min_value=100,
            max_value=10000,
            value=DEFAULT_SERIES_SIMULATIONS,
            step=100,
            key="mlb_series_simulations",
        )

    with seed_col:
        seed = st.number_input(
            "Simulation ID",
            min_value=1,
            max_value=999999,
            value=42,
            step=1,
            key="mlb_series_seed",
        )

    with action_col:
        st.write("")
        simulate_clicked = st.button("Simulate", type="primary", width="stretch", key="mlb_series_run")

    state_key = "mlb_series_last_simulation"

    if simulate_clicked:
        with st.spinner("Simulating MLB series..."):
            results = simulate_series(
                higher_seed_team=higher_seed_team,
                lower_seed_team=lower_seed_team,
                simulations=int(simulations),
                seed=int(seed),
                best_of=int(best_of),
            )
        st.session_state[state_key] = {
            "higher_seed_team": higher_seed_team,
            "lower_seed_team": lower_seed_team,
            "best_of": int(best_of),
            "simulations": int(simulations),
            "seed": int(seed),
            "results": results,
        }

    simulation = st.session_state.get(state_key)

    if simulation and (
        simulation.get("higher_seed_team") == higher_seed_team
        and simulation.get("lower_seed_team") == lower_seed_team
        and int(simulation.get("best_of")) == int(best_of)
        and int(simulation.get("simulations")) == int(simulations)
        and int(simulation.get("seed")) == int(seed)
    ):
        results = simulation["results"]
        probabilities = results["series_probabilities"]
        predicted_winner = str(results["predicted_winner"])
        favorite_probability = float(results["favorite_probability"])
        render_prediction_result_card(
            label="Series prediction",
            winner=predicted_winner,
            probability=favorite_probability,
            note=f"Best of {best_of} / {simulations:,} simulations",
        )
        render_series_score_card(
            team_a=higher_seed_team,
            team_b=lower_seed_team,
            probability_a=float(probabilities[higher_seed_team]),
            probability_b=float(probabilities[lower_seed_team]),
            best_of=int(best_of),
            predicted_winner=predicted_winner,
        )

        result_table = results["result_table"].copy()
        result_table["Probability"] = result_table["Probability"].map("{:.1%}".format)
        st.dataframe(result_table, width="stretch", hide_index=True)

        report = create_prediction_report(
            title=f"MLB Series Prediction: {higher_seed_team} vs {lower_seed_team}",
            rows=[
                f"Predicted winner: {predicted_winner} ({favorite_probability:.1%})",
                f"{higher_seed_team}: {probabilities[higher_seed_team]:.1%}",
                f"{lower_seed_team}: {probabilities[lower_seed_team]:.1%}",
                f"Format: best of {best_of}",
                f"Simulations: {simulations:,}",
            ],
        )
        render_report_download(report, "mlb_series_prediction_report.txt", "mlb_series_report")


def render_bracket_view() -> None:
    """Render simplified MLB playoff bracket simulator."""
    strength = load_team_strength()

    if strength.empty:
        st.warning("Missing MLB team strength data. Run `python src/mlb_team_strength.py`.")
        return

    teams = strength.sort_values("ELO", ascending=False)["TEAM_NAME"].astype(str).tolist()
    defaults = teams[:12]

    if len(defaults) < 12:
        st.warning("Need at least 12 MLB teams for the bracket simulator.")
        return

    st.caption("Uses a simplified 12-team MLB-style bracket: four byes, four Wild Card series, Division series, semifinals, and championship.")
    seed_cols = st.columns(3)
    selected_teams = []

    for index in range(12):
        with seed_cols[index % 3]:
            selected = st.selectbox(
                f"Seed {index + 1}",
                teams,
                index=teams.index(defaults[index]),
                key=f"mlb_bracket_seed_{index + 1}",
            )
            selected_teams.append(selected)

    if len(set(selected_teams)) != len(selected_teams):
        st.warning("Each bracket seed must be a unique team.")
        return

    render_team_logo_strip(selected_teams)

    sim_col, seed_col, action_col = st.columns([2, 1, 1])

    with sim_col:
        simulations = st.slider(
            "Bracket simulations",
            min_value=100,
            max_value=5000,
            value=DEFAULT_BRACKET_SIMULATIONS,
            step=100,
            key="mlb_bracket_simulations",
        )

    with seed_col:
        seed = st.number_input(
            "Simulation ID",
            min_value=1,
            max_value=999999,
            value=123,
            step=1,
            key="mlb_bracket_seed",
        )

    with action_col:
        st.write("")
        simulate_clicked = st.button("Simulate playoffs", type="primary", width="stretch")

    state_key = "mlb_bracket_last_simulation"
    signature = (tuple(selected_teams), int(simulations), int(seed))

    if simulate_clicked:
        with st.spinner("Simulating MLB bracket..."):
            bracket = simulate_full_bracket(selected_teams, int(simulations), int(seed))
        st.session_state[state_key] = {
            "signature": signature,
            "bracket": bracket,
        }

    state = st.session_state.get(state_key)

    if state and state.get("signature") == signature:
        bracket = state["bracket"]
        results = bracket["results"].copy()
        winner_row = results.iloc[0]
        winner = str(winner_row["Team"])
        probability = float(winner_row["Championship Probability"])
        render_prediction_result_card(
            label="Championship prediction",
            winner=winner,
            probability=probability,
            note=f"{simulations:,} bracket simulations",
        )
        st.bar_chart(results.set_index("Team")[["Championship Probability"]])

        formatted = results.copy()
        formatted["Finals Probability"] = formatted["Finals Probability"].map("{:.1%}".format)
        formatted["Championship Probability"] = formatted["Championship Probability"].map("{:.1%}".format)
        st.dataframe(formatted, width="stretch", hide_index=True)

        with st.expander("Sample bracket path", expanded=True):
            st.dataframe(bracket["sample_path"], width="stretch", hide_index=True)

        st.download_button(
            "Download championship probabilities CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="mlb_championship_probabilities.csv",
            mime="text/csv",
            width="stretch",
        )

        report = create_prediction_report(
            title="MLB Bracket Prediction",
            rows=[
                f"Predicted champion: {winner} ({probability:.1%})",
                f"Simulations: {simulations:,}",
                "Top five:",
                *[
                    f"{row['Team']}: {float(row['Championship Probability']):.1%}"
                    for _, row in results.head(5).iterrows()
                ],
            ],
        )
        render_report_download(report, "mlb_bracket_prediction_report.txt", "mlb_bracket_report")


def standings_logo_html(team_id: object, team_name: object) -> str:
    """Return a compact team logo for standings rows."""
    logo_url = team_logo_url(team_id)

    if not logo_url:
        return ""

    return (
        f'<img class="standings-logo" src="{html.escape(logo_url)}" '
        f'alt="{html.escape(str(team_name or "Team"))} logo">'
    )


def render_standings_summary_cards(league_rows: pd.DataFrame, divisions: list[str]) -> None:
    """Render compact division-leader cards."""
    card_html = []
    colors = ["#1d4ed8", "#166534", "#c2410c"]

    for index, division in enumerate(divisions):
        division_rows = league_rows[league_rows["Division"].eq(division)].sort_values("Rank Sort")

        if division_rows.empty:
            continue

        leader = division_rows.iloc[0]
        card_html.append(
            f"""
            <div class="standings-summary-card" style="--card-color: {colors[index % len(colors)]};">
                <div class="standings-card-label">{html.escape(division)} leader</div>
                <div class="standings-card-team">
                    {standings_logo_html(leader.get("Team ID"), leader.get("Team"))}
                    <span>{html.escape(str(leader.get("Team", "Team")))}</span>
                </div>
                <div class="standings-card-meta">
                    {html.escape(str(leader.get("W", "-")))}-{html.escape(str(leader.get("L", "-")))}
                    / {html.escape(str(leader.get("Pct", "-")))}
                    / Run diff {html.escape(str(leader.get("Run Diff", "-")))}
                </div>
            </div>
            """
        )

    if card_html:
        st.html(f'<div class="standings-summary-grid">{"".join(card_html)}</div>')


def render_standings_table(rows: pd.DataFrame, include_division: bool = False) -> None:
    """Render a styled MLB standings table."""
    if rows.empty:
        st.info("No standings rows found.")
        return

    columns = [
        ("Rank", "#"),
        ("Team", "Team"),
        ("W", "W"),
        ("L", "L"),
        ("Pct", "Pct"),
        ("GB", "GB"),
        ("WC GB", "WC GB"),
        ("Last 10", "Last 10"),
        ("Streak", "Streak"),
        ("Run Diff", "Run Diff"),
    ]

    if include_division:
        columns.insert(2, ("Division", "Division"))

    header_html = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    row_html = []

    for _, row in rows.iterrows():
        cells = []

        for key, _ in columns:
            if key == "Rank":
                cells.append(f'<td><span class="standings-rank">{html.escape(str(row.get(key, "-")))}</span></td>')
            elif key == "Team":
                cells.append(
                    "<td>"
                    '<div class="standings-team-cell">'
                    f'{standings_logo_html(row.get("Team ID"), row.get("Team"))}'
                    f'<span>{html.escape(str(row.get("Team", "Team")))}</span>'
                    "</div>"
                    "</td>"
                )
            else:
                cells.append(f"<td>{html.escape(str(row.get(key, '-')))}</td>")

        row_html.append(f"<tr>{''.join(cells)}</tr>")

    st.html(
        f"""
        <div class="standings-table-wrap">
            <table class="standings-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
            </table>
        </div>
        """
    )


def render_standings_league(league_rows: pd.DataFrame, league_name: str) -> None:
    """Render one league with nested standings tabs."""
    ordered_divisions = [
        division
        for division in MLB_DIVISION_ORDER_BY_LEAGUE.get(league_name, [])
        if division in set(league_rows["Division"].dropna().tolist())
    ]
    extra_divisions = [
        str(division)
        for division in league_rows["Division"].dropna().unique().tolist()
        if division not in ordered_divisions
    ]
    divisions = [*ordered_divisions, *extra_divisions]

    render_standings_summary_cards(league_rows, divisions)

    tab_labels = ["Overall", *[division.replace("AL ", "").replace("NL ", "") for division in divisions]]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        overall = league_rows.copy()
        overall["_league_rank_sort"] = overall["League Rank"].map(lambda value: safe_int(value) or 99)
        overall = overall.sort_values(["_league_rank_sort", "Division Sort", "Rank Sort", "Team"])
        render_standings_table(overall, include_division=True)

    for tab, division in zip(tabs[1:], divisions, strict=False):
        with tab:
            division_rows = league_rows[league_rows["Division"].eq(division)].sort_values("Rank Sort")
            render_section_kicker(division)
            render_standings_table(division_rows, include_division=False)


def render_standings_view() -> None:
    """Render MLB standings by league and division."""
    st.caption(f"Official MLB standings feed. Cached for {STANDINGS_CACHE_SECONDS}s so updates appear shortly after MLB posts final games.")
    current_season = current_mlb_season()
    control_col, refresh_col = st.columns([2, 1])

    with control_col:
        season = st.number_input(
            "Season",
            min_value=1901,
            max_value=current_season + 1,
            value=current_season,
            step=1,
            key="mlb_standings_season",
        )

    with refresh_col:
        st.write("")

        if st.button("Refresh standings", width="stretch", key="mlb_standings_refresh"):
            load_mlb_standings.clear()

    standings, loaded_at = load_mlb_standings(int(season))

    if standings.empty:
        st.warning("Standings are not available from the MLB feed right now.")
        return

    st.html(
        f"""
        <div class="standings-header">
            <div>
                <div class="standings-header-title">MLB Standings</div>
                <div class="standings-header-note">
                    League tabs, division tabs, and official records from MLB.
                </div>
            </div>
            <div class="standings-header-note">{html.escape(f"Loaded {loaded_at}" if loaded_at else "Live MLB feed")}</div>
        </div>
        """
    )

    league_tabs = st.tabs(["American League", "National League"])

    for tab, league_name in zip(league_tabs, ["American League", "National League"], strict=False):
        with tab:
            league_rows = standings[standings["League"].eq(league_name)].copy()

            if league_rows.empty:
                st.info(f"No {league_name} standings found.")
                continue

            render_standings_league(league_rows, league_name)


def load_betting_slate_games(slate: str) -> pd.DataFrame:
    """Load a schedule frame for betting research views."""
    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()

    if slate == "Today":
        games = load_today_games()
        return games if not games.empty else load_next_upcoming_games()

    if slate == "Next 7 Days":
        games = load_mlb_schedule(today.isoformat(), (today + timedelta(days=7)).isoformat())
        return filter_known_games(games)

    return load_next_upcoming_games()


def betting_edge_rows_for_game(
    game: pd.Series,
    away_odds: object,
    home_odds: object,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Build display rows and raw values for a moneyline edge comparison."""
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    away_probability = safe_float(game.get("AWAY_WIN_PROBABILITY"), default=None)
    home_probability = safe_float(game.get("HOME_WIN_PROBABILITY"), default=None)
    away_implied = american_odds_to_implied_probability(away_odds)
    home_implied = american_odds_to_implied_probability(home_odds)
    away_market, home_market = remove_two_way_vig(away_implied, home_implied)
    raw_rows = [
        {
            "team": away_team,
            "team_id": game.get("AWAY_TEAM_ID"),
            "side": "Away",
            "odds": away_odds,
            "model_probability": away_probability,
            "market_probability": away_market,
        },
        {
            "team": home_team,
            "team_id": game.get("HOME_TEAM_ID"),
            "side": "Home",
            "odds": home_odds,
            "model_probability": home_probability,
            "market_probability": home_market,
        },
    ]
    display_rows = []
    lookup = {}

    for row in raw_rows:
        model_probability = row["model_probability"]
        market_probability = row["market_probability"]
        edge = (
            float(model_probability) - float(market_probability)
            if model_probability is not None and market_probability is not None
            else None
        )
        ev = expected_value_per_unit(model_probability, row["odds"])
        kelly = kelly_fraction(model_probability, row["odds"])
        signal, _ = edge_strength_label(edge)
        fair_odds = probability_to_american_odds(model_probability)
        team = str(row["team"])
        lookup[team] = {
            **row,
            "edge": edge,
            "ev": ev,
            "kelly": kelly,
            "fair_odds": fair_odds,
            "signal": signal,
        }
        display_rows.append(
            {
                "Team": team_abbreviation(team, row["team_id"]),
                "Side": row["side"],
                "Odds": format_american_odds(row["odds"]),
                "Model": f"{model_probability:.1%}" if model_probability is not None else "-",
                "Market": f"{market_probability:.1%}" if market_probability is not None else "-",
                "Edge": format_signed_percent(edge),
                "Fair Odds": fair_odds,
                "EV/Unit": f"{ev:+.2f}" if ev is not None else "-",
                "Kelly": f"{kelly:.1%}" if kelly is not None else "-",
                "Signal": signal,
            }
        )

    return pd.DataFrame(display_rows), lookup


def format_plain_percent(value: object) -> str:
    """Format one probability for simple betting cards."""
    parsed = safe_float(value, default=None)
    return f"{parsed:.0%}" if parsed is not None else "-"


def format_profit_on_stake(odds: object, stake: float = 10.0) -> str:
    """Format potential profit for a small stake."""
    profit_per_unit = american_odds_profit_per_unit(odds)

    if profit_per_unit is None:
        return "-"

    return f"${profit_per_unit * stake:,.2f}"


def simple_betting_signal(edge: object) -> tuple[str, str, str]:
    """Return beginner-facing signal text, note, and CSS class."""
    parsed = safe_float(edge, default=None)

    if parsed is None:
        return "Add odds", "Enter or load a sportsbook price.", "edge-none"

    if parsed >= 0.05:
        return "Good value", "Our win chance is better than the book price.", "edge-strong"

    if parsed >= 0.02:
        return "Small value", "Worth reviewing, but the gap is not huge.", "edge-small"

    if parsed <= -0.02:
        return "Price is high", "The book price is worse than our number.", "edge-none"

    return "Fair price", "Win chance and book price are close.", "edge-none"


def best_side_from_lookup(lookup: dict[str, dict]) -> str:
    """Choose the most beginner-friendly default side for a game."""
    rows = list(lookup.values())
    with_edges = [row for row in rows if safe_float(row.get("edge"), default=None) is not None]

    if with_edges:
        return str(max(with_edges, key=lambda row: float(row.get("edge") or -99)).get("team", ""))

    with_probs = [row for row in rows if safe_float(row.get("model_probability"), default=None) is not None]

    if with_probs:
        return str(max(with_probs, key=lambda row: float(row.get("model_probability") or 0)).get("team", ""))

    return str(rows[0].get("team", "")) if rows else ""


def betting_logo_html(team_id: object, team_name: object) -> str:
    """Return a compact team logo for betting cards."""
    logo_url = team_logo_url(team_id)

    if not logo_url:
        return ""

    return (
        f'<img class="betting-matchup-logo" src="{html.escape(logo_url)}" '
        f'alt="{html.escape(str(team_name or "Team"))} logo">'
    )


def render_betting_hero(edge_rows: list[dict], odds_source: str) -> None:
    """Render a simple top summary for the betting tab."""
    valid_rows = [
        row
        for row in edge_rows
        if safe_float(row.get("Edge"), default=None) is not None
    ]
    best_row = max(valid_rows, key=lambda row: float(row.get("Edge") or -99), default=None)
    best_value = f"{best_row['Team']} {float(best_row['Edge']):+.0%}" if best_row else "No picks yet"
    best_note = str(best_row.get("Game", "Add odds to compare games.")) if best_row else "Add odds to compare games."
    st.html(
        f"""
        <div class="betting-hero">
            <div>
                <div class="betting-hero-title">Betting</div>
                <div class="betting-hero-note">
                    Pick a team to win. Live sportsbook odds refresh every {ODDS_REFRESH_SECONDS} seconds.
                </div>
            </div>
            <div class="betting-hero-stat">
                <div class="betting-hero-stat-label">{html.escape(odds_source)}</div>
                <div class="betting-hero-stat-value">{html.escape(best_value)}</div>
                <div class="betting-hero-note">{html.escape(best_note)}</div>
            </div>
        </div>
        """
    )


def render_beginner_pick_card(
    game: pd.Series,
    selection: str,
    selected_data: dict,
    handoff_data: dict,
) -> None:
    """Render a beginner-friendly game card."""
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    model_probability = selected_data.get("model_probability")
    edge = handoff_data.get("edge") if handoff_data.get("edge") is not None else selected_data.get("edge")
    odds = handoff_data.get("odds") or selected_data.get("odds")
    sportsbook = handoff_data.get("sportsbook") or "Sportsbook"
    signal, signal_note, signal_class = simple_betting_signal(edge)
    meter_width = int(round((safe_float(model_probability, default=0.5) or 0.5) * 100))
    st.html(
        f"""
        <div class="betting-card">
            <div class="betting-card-head">
                <div>
                    <div class="betting-game-title">{html.escape(format_matchup(game))}</div>
                    <div class="betting-game-meta">
                        {html.escape(str(game.get("STATUS", "Scheduled") or "Scheduled"))}
                        / {html.escape(format_game_time(game.get("GAME_DATETIME")))}
                    </div>
                </div>
                <div class="betting-matchup-logos">
                    {betting_logo_html(game.get("AWAY_TEAM_ID"), away_team)}
                    {betting_logo_html(game.get("HOME_TEAM_ID"), home_team)}
                </div>
            </div>
            <div class="betting-pick-body">
                <div class="betting-pick-main">
                    <div class="betting-pick-primary">
                        <div class="betting-pick-label">Suggested pick</div>
                        <div class="betting-pick-team">{html.escape(selection)}</div>
                        <div class="betting-pick-sub">
                            {html.escape(format_plain_percent(model_probability))} chance to win
                        </div>
                        <div class="betting-meter">
                            <div class="betting-meter-fill" style="--meter-width: {meter_width}%;"></div>
                        </div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Book price</div>
                        <div class="betting-simple-value">{html.escape(format_american_odds(odds))}</div>
                        <div class="betting-simple-note">{html.escape(str(sportsbook))}</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Price gap</div>
                        <div class="betting-simple-value">
                            <span class="edge-pill {signal_class}">{html.escape(signal)}</span>
                        </div>
                        <div class="betting-simple-note">{html.escape(signal_note)}</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">$10 win profit</div>
                        <div class="betting-simple-value">{html.escape(format_profit_on_stake(odds, 10.0))}</div>
                        <div class="betting-simple-note">Stake is not included.</div>
                    </div>
                </div>
            </div>
        </div>
        """
    )


def render_prop_beginner_card(
    player: dict,
    prop_type: str,
    side: str,
    line: float,
    projection: float,
    model_probability: float,
    market_probability: float | None,
    odds: object,
    edge: object,
    note: str,
) -> None:
    """Render beginner-friendly player prop result."""
    signal, signal_note, signal_class = simple_betting_signal(edge)
    market_text = format_plain_percent(market_probability) if market_probability is not None else "-"
    meter_width = int(round(clamp_float(float(model_probability), 0.0, 1.0) * 100))
    st.html(
        f"""
        <div class="betting-card">
            <div class="betting-card-head">
                <div>
                    <div class="betting-game-title">
                        {html.escape(str(player.get("PLAYER_NAME", "Player")))}
                    </div>
                    <div class="betting-game-meta">
                        {html.escape(side)} {line:g} {html.escape(prop_type)}
                    </div>
                </div>
                <span class="edge-pill {signal_class}">{html.escape(signal)}</span>
            </div>
            <div class="betting-pick-body">
                <div class="betting-pick-main">
                    <div class="betting-pick-primary">
                        <div class="betting-pick-label">Win chance</div>
                        <div class="betting-pick-team">{model_probability:.0%}</div>
                        <div class="betting-pick-sub">{html.escape(signal_note)}</div>
                        <div class="betting-meter">
                            <div class="betting-meter-fill" style="--meter-width: {meter_width}%;"></div>
                        </div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Projection</div>
                        <div class="betting-simple-value">{projection:.2f}</div>
                        <div class="betting-simple-note">{html.escape(note)}</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Book chance</div>
                        <div class="betting-simple-value">{html.escape(market_text)}</div>
                        <div class="betting-simple-note">Based on the entered odds.</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">$10 win profit</div>
                        <div class="betting-simple-value">{html.escape(format_profit_on_stake(odds, 10.0))}</div>
                        <div class="betting-simple-note">{html.escape(format_american_odds(odds))}</div>
                    </div>
                </div>
            </div>
        </div>
        """
    )


def render_betting_summary(edge_rows: list[dict]) -> None:
    """Render top betting slate summary cards."""
    if not edge_rows:
        render_dashboard_cards(
            [
                {"label": "Priced Picks", "value": "0", "note": "Add odds to compare games.", "color": "#64748b"},
                {"label": "Good Value", "value": "0", "note": "No clear price gap yet.", "color": "#16a34a"},
                {"label": "Top Pick", "value": "-", "note": "No prices loaded yet.", "color": "#2563eb"},
            ]
        )
        return

    edges = pd.DataFrame(edge_rows)
    valid_edges = edges[pd.to_numeric(edges["Edge"], errors="coerce").notna()].copy()

    if valid_edges.empty:
        render_dashboard_cards(
            [
                {"label": "Priced Picks", "value": "0", "note": "Add odds to compare games.", "color": "#64748b"},
                {"label": "Good Value", "value": "0", "note": "No clear price gap yet.", "color": "#16a34a"},
                {"label": "Top Pick", "value": "-", "note": "No prices loaded yet.", "color": "#2563eb"},
            ]
        )
        return

    valid_edges = valid_edges.sort_values(["EV/Unit", "Edge"], ascending=False)
    best = valid_edges.iloc[0]
    strong_edges = int((valid_edges["Edge"] >= 0.05).sum())
    small_edges = int(((valid_edges["Edge"] >= 0.02) & (valid_edges["Edge"] < 0.05)).sum())
    render_dashboard_cards(
        [
            {
                "label": "Top Value",
                "value": f"{best['Team']} {best['Edge']:+.0%}",
                "note": str(best["Game"]),
                "color": "#16a34a" if best["Edge"] >= 0.05 else "#ca8a04",
            },
            {
                "label": "Good Value",
                "value": str(strong_edges),
                "note": f"{small_edges} smaller gaps also found.",
                "color": "#2563eb",
            },
            {
                "label": "$10 Win Profit",
                "value": format_profit_on_stake(best["Odds"], 10.0),
                "note": f"{format_american_odds(best['Odds'])} on {best['Team']}",
                "color": "#0f766e",
            },
        ]
    )

    display = valid_edges[
        ["Game", "Team", "Sportsbook", "Odds", "Model", "Market", "Edge", "EV/Unit", "Kelly"]
    ].assign(
        Odds=lambda frame: frame["Odds"].map(format_american_odds),
        Model=lambda frame: frame["Model"].map(lambda value: f"{value:.1%}"),
        Market=lambda frame: frame["Market"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "-"),
        Edge=lambda frame: frame["Edge"].map(lambda value: f"{value:+.1%}"),
        Kelly=lambda frame: frame["Kelly"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "-"),
        **{"EV/Unit": lambda frame: frame["EV/Unit"].map(lambda value: f"{value:+.2f}" if pd.notna(value) else "-")},
    ).rename(
        columns={
            "Odds": "Book Price",
            "Model": "Win Chance",
            "Market": "Book Chance",
            "Edge": "Price Gap",
        }
    )
    st.html(
        """
        <div class="betting-line-table">
            <div class="betting-game-title">Line comparison</div>
            <div class="betting-game-meta">
                Sorted by expected value, then price gap.
            </div>
        </div>
        """
    )
    st.dataframe(display.head(24), hide_index=True, width="stretch")


def moneyline_selection_payload(
    game: pd.Series,
    selection: str,
    selected_data: dict,
    book_rows: list[dict],
) -> dict:
    """Return best available price and edge values for one moneyline side."""
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    best_book = best_book_row_for_selection(book_rows, selection, home_team, away_team)
    price = (
        sportsbook_price_for_selection(best_book, selection, home_team, away_team)
        if best_book
        else selected_data.get("odds")
    )
    market_probability = (
        sportsbook_market_probability_for_selection(best_book, selection, home_team, away_team)
        if best_book
        else selected_data.get("market_probability")
    )
    model_probability = selected_data.get("model_probability")
    edge = (
        float(model_probability) - float(market_probability)
        if model_probability is not None and market_probability is not None
        else selected_data.get("edge")
    )
    ev = expected_value_per_unit(model_probability, price)
    kelly = kelly_fraction(model_probability, price)
    sportsbook = best_book.get("sportsbook") if best_book else selected_data.get("sportsbook", "")

    return {
        **selected_data,
        "selection": selection,
        "sportsbook": sportsbook or "Manual",
        "sportsbook_key": best_book.get("sportsbook_key", "") if best_book else "",
        "book_row": best_book,
        "odds": price,
        "market_probability": market_probability,
        "edge": edge,
        "ev": ev,
        "kelly": kelly,
        "external_url": sportsbook_external_url(best_book.get("sportsbook_key", "")) if best_book else sportsbook_external_url("draftkings"),
    }


def moneyline_payload_sort_value(payload: dict) -> tuple[float, float]:
    """Sort value sides by edge first, then model probability."""
    edge = safe_float(payload.get("edge"), default=None)
    model_probability = safe_float(payload.get("model_probability"), default=0.0) or 0.0
    return (
        float(edge) if edge is not None else -9.0,
        float(model_probability),
    )


def best_moneyline_selection_payload(
    game: pd.Series,
    lookup: dict[str, dict],
    book_rows: list[dict],
) -> dict:
    """Return the selected side for the board card."""
    payloads = [
        moneyline_selection_payload(game, team, data, book_rows)
        for team, data in lookup.items()
    ]

    if not payloads:
        return {}

    return max(payloads, key=moneyline_payload_sort_value)


def format_betting_edge_value(value: object) -> str:
    """Format an edge in beginner-facing percentage-point terms."""
    parsed = safe_float(value, default=None)

    if parsed is None:
        return "-"

    return f"{parsed * 100:+.1f} pts"


def render_betting_side_tile(
    team: str,
    team_id: object,
    payload: dict,
    is_selected: bool,
) -> str:
    """Return HTML for one team side inside a betting card."""
    signal, _, signal_class = simple_betting_signal(payload.get("edge"))
    model_probability = safe_float(payload.get("model_probability"), default=None)
    edge = payload.get("edge")
    odds = payload.get("odds")
    sportsbook = payload.get("sportsbook") or "Manual"
    selected_class = " betting-side-selected" if is_selected else ""
    logo = betting_logo_html(team_id, team)
    return f"""
        <div class="betting-side-tile{selected_class}">
            <div class="betting-side-top">
                <div>
                    <div class="betting-side-name">{html.escape(team_abbreviation(team, team_id))}</div>
                    <div class="betting-game-meta">{html.escape(team)}</div>
                </div>
                {logo}
            </div>
            <div class="betting-side-odds">{html.escape(format_american_odds(odds))}</div>
            <div class="betting-game-meta">{html.escape(str(sportsbook))}</div>
            <div class="betting-side-meta-grid">
                <div class="betting-side-meta">
                    <div class="betting-side-meta-label">Win</div>
                    <div class="betting-side-meta-value">{html.escape(format_plain_percent(model_probability))}</div>
                </div>
                <div class="betting-side-meta">
                    <div class="betting-side-meta-label">Edge</div>
                    <div class="betting-side-meta-value">{html.escape(format_betting_edge_value(edge))}</div>
                </div>
                <div class="betting-side-meta">
                    <div class="betting-side-meta-label">Signal</div>
                    <div class="betting-side-meta-value">
                        <span class="edge-pill {signal_class}">{html.escape(signal)}</span>
                    </div>
                </div>
            </div>
        </div>
    """


def set_bet_slip(
    game: pd.Series,
    selection: str,
    payload: dict,
    book_rows: list[dict],
    stake: float,
) -> None:
    """Store a selected pick for the slip panel."""
    locked_odds = safe_float(payload.get("odds"), default=None)

    if locked_odds is None:
        st.warning("Live odds are not available for that pick yet.")
        return

    st.session_state[BET_SLIP_STATE_KEY] = {
        "game": game.to_dict(),
        "selection": selection,
        "payload": {
            **payload,
            "locked_odds": locked_odds,
            "locked_sportsbook": payload.get("sportsbook") or "Sportsbook",
            "locked_sportsbook_key": payload.get("sportsbook_key") or "",
        },
        "book_rows": book_rows,
        "stake": stake,
        "game_label": format_matchup(game),
    }


def render_moneyline_game_card(
    game: pd.Series,
    lookup: dict[str, dict],
    book_rows: list[dict],
    stake: float,
    rank: int,
) -> list[dict]:
    """Render one non-expander betting-board card and return edge rows."""
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    away_payload = moneyline_selection_payload(
        game,
        away_team,
        lookup.get(away_team, {}),
        book_rows,
    )
    home_payload = moneyline_selection_payload(
        game,
        home_team,
        lookup.get(home_team, {}),
        book_rows,
    )
    selected_payload = max(
        [away_payload, home_payload],
        key=moneyline_payload_sort_value,
    )
    selection = str(selected_payload.get("selection") or best_side_from_lookup(lookup))
    signal, signal_note, signal_class = simple_betting_signal(selected_payload.get("edge"))
    status = str(game.get("STATUS", "Scheduled") or "Scheduled")
    start_time = format_game_time(game.get("GAME_DATETIME"))
    selected_team_id = game.get("HOME_TEAM_ID") if selection == home_team else game.get("AWAY_TEAM_ID")
    selected_abbr = team_abbreviation(
        selection,
        selected_team_id,
    )
    other_team = away_team if selection == home_team else home_team
    other_payload = away_payload if other_team == away_team else home_payload
    profit_preview = format_profit_on_stake(selected_payload.get("odds"), stake)
    card_key = safe_streamlit_dom_key(str(game.get("GAME_PK", "")) or format_matchup(game))
    odds_text = format_american_odds(selected_payload.get("odds"))
    odds_note = "Waiting for live odds" if odds_text == "-" else str(selected_payload.get("sportsbook") or "Sportsbook")
    model_probability = selected_payload.get("model_probability")
    logo = betting_logo_html(selected_team_id, selection)

    st.html(
        f"""
        <div class="betting-card-shell">
            <div class="betting-card-head">
                <div>
                    <div class="betting-game-title">{html.escape(format_matchup(game))}</div>
                    <div class="betting-game-meta">
                        {html.escape(status)} / {html.escape(start_time)} / Moneyline
                    </div>
                </div>
                <div class="betting-rank-pill">#{rank} {html.escape(signal)}</div>
            </div>
            <div class="betting-pick-body">
                <div class="betting-pick-simple">
                    <div class="betting-pick-logo">{logo}</div>
                    <div>
                        <div class="betting-pick-label">Suggested pick</div>
                        <div class="betting-pick-team">{html.escape(selected_abbr)}</div>
                        <div class="betting-pick-sub">{html.escape(selection)} to win</div>
                    </div>
                </div>
                <div class="betting-pick-row">
                    <div class="betting-pick-stat">
                        <div class="betting-simple-label">Win chance</div>
                        <div class="betting-simple-value">{html.escape(format_plain_percent(model_probability))}</div>
                    </div>
                    <div class="betting-pick-stat">
                        <div class="betting-simple-label">Odds</div>
                        <div class="betting-simple-value">{html.escape(odds_text)}</div>
                        <div class="betting-simple-note">{html.escape(odds_note)}</div>
                    </div>
                    <div class="betting-pick-stat">
                        <div class="betting-simple-label">Profit on ${stake:g}</div>
                        <div class="betting-simple-value">{html.escape(profit_preview)}</div>
                    </div>
                </div>
                <div class="betting-note">
                    <span class="edge-pill {signal_class}">{html.escape(signal)}</span>
                    {html.escape(signal_note)}
                </div>
            </div>
        </div>
        """
    )

    action_cols = st.columns(2)
    with action_cols[0]:
        selected_has_odds = safe_float(selected_payload.get("odds"), default=None) is not None
        if st.button(
            f"Add {selected_abbr}",
            key=f"mlb_add_suggested_slip_{card_key}",
            width="stretch",
            type="primary",
            disabled=not selected_has_odds,
        ):
            set_bet_slip(game, selection, selected_payload, book_rows, stake)

    with action_cols[1]:
        other_has_odds = safe_float(other_payload.get("odds"), default=None) is not None
        if st.button(
            f"Add {team_abbreviation(other_team, game.get('AWAY_TEAM_ID') if other_team == away_team else game.get('HOME_TEAM_ID'))}",
            key=f"mlb_add_other_slip_{card_key}",
            width="stretch",
            disabled=not other_has_odds,
        ):
            set_bet_slip(game, other_team, other_payload, book_rows, stake)

    edge_rows = []
    for payload in [away_payload, home_payload]:
        if payload.get("edge") is None and payload.get("odds") is None:
            continue

        edge_rows.append(
            {
                "Game": format_matchup(game),
                "Team": team_abbreviation(
                    payload.get("selection"),
                    game.get("HOME_TEAM_ID") if payload.get("selection") == home_team else game.get("AWAY_TEAM_ID"),
                ),
                "Sportsbook": payload.get("sportsbook", "Manual"),
                "Odds": payload.get("odds"),
                "Model": payload.get("model_probability"),
                "Market": payload.get("market_probability"),
                "Edge": payload.get("edge"),
                "EV/Unit": payload.get("ev"),
                "Kelly": payload.get("kelly"),
            }
        )

    return edge_rows


def render_empty_bet_slip() -> None:
    """Render an empty slip placeholder."""
    st.html(
        """
        <div class="bet-slip">
            <div class="bet-slip-title">Bet slip</div>
            <div class="bet-slip-empty">
                Add a pick from the board, then choose the sportsbook in the slip.
                Odds are read-only and come from the live feed.
            </div>
            <div class="bet-slip-warning">
                Verify the team, market, and price inside the sportsbook before wagering.
            </div>
        </div>
        """
    )


def render_bet_slip_panel(current_book_rows: list[dict] | None = None) -> None:
    """Render the selected pick, stake, sportsbook handoff, and tracker logging."""
    slip = st.session_state.get(BET_SLIP_STATE_KEY)

    if not slip:
        render_empty_bet_slip()
        return

    game = pd.Series(slip.get("game", {}))
    selection = str(slip.get("selection", ""))
    payload = dict(slip.get("payload", {}))
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    game_key = safe_streamlit_dom_key(str(game.get("GAME_PK", "")) or format_matchup(game))
    saved_book_rows = list(slip.get("book_rows", []))
    book_rows = list(current_book_rows) if current_book_rows is not None else saved_book_rows
    option_rows = [
        option
        for option in sportsbook_option_rows(book_rows, selection, home_team, away_team)
        if safe_float(option.get("price"), default=None) is not None
    ]

    if not option_rows:
        st.warning("Live odds for this pick are not available right now. Keep waiting or clear the slip.")
        if st.button("Clear", key=f"mlb_slip_clear_missing_odds_{game_key}", width="stretch"):
            st.session_state.pop(BET_SLIP_STATE_KEY, None)
            st.rerun()
        return

    preferred_key = sportsbook_key_from_bookmaker(
        payload.get("locked_sportsbook_key")
        or payload.get("sportsbook_key")
        or option_rows[0].get("sportsbook_key")
    )
    option_keys = [
        str(option.get("sportsbook_key") or sportsbook_key_from_bookmaker(option.get("sportsbook")))
        for option in option_rows
    ]
    labels_by_key = {
        str(
            option.get("sportsbook_key")
            or sportsbook_key_from_bookmaker(option.get("sportsbook"))
        ): str(option.get("sportsbook") or "Sportsbook")
        for option in option_rows
    }
    default_index = next(
        (
            index
            for index, sportsbook_key in enumerate(option_keys)
            if sportsbook_key_from_bookmaker(sportsbook_key) == preferred_key
        ),
        0,
    )
    selector_key = f"mlb_slip_sportsbook_{game_key}_{safe_streamlit_dom_key(selection)}"

    if st.session_state.get(selector_key) not in option_keys:
        st.session_state.pop(selector_key, None)

    st.html(
        f"""
        <div class="bet-slip">
            <div class="bet-slip-title">{html.escape(selection)}</div>
            <div class="bet-slip-market">
                {html.escape(format_matchup(game))} / Moneyline
            </div>
        </div>
        """
    )
    selected_key = st.selectbox(
        "Sportsbook",
        option_keys,
        index=default_index,
        format_func=lambda sportsbook_key: labels_by_key.get(str(sportsbook_key), str(sportsbook_key)),
        key=selector_key,
    )
    selected_index = option_keys.index(selected_key) if selected_key in option_keys else default_index
    selected_option = option_rows[selected_index]
    selected_row = selected_option.get("row") or {}
    live_odds = safe_float(selected_option.get("price"), default=None)
    sportsbook_key = str(selected_option.get("sportsbook_key") or selected_key or "draftkings")
    sportsbook_name = str(selected_option.get("sportsbook") or labels_by_key.get(sportsbook_key) or "Sportsbook")

    stake = st.number_input(
        "Stake",
        min_value=0.0,
        max_value=100000.0,
        value=float(slip.get("stake", 10.0) or 10.0),
        step=1.0,
        key=f"mlb_slip_stake_{game_key}_{safe_streamlit_dom_key(selection)}",
    )
    model_probability = safe_float(payload.get("model_probability"), default=None)
    market_probability = sportsbook_market_probability_for_selection(
        selected_row,
        selection,
        home_team,
        away_team,
    ) if selected_row else None

    if market_probability is None:
        market_probability = american_odds_to_implied_probability(live_odds)

    edge = (
        float(model_probability) - float(market_probability)
        if model_probability is not None and market_probability is not None
        else None
    )
    ev = expected_value_per_unit(model_probability, live_odds)
    kelly = kelly_fraction(model_probability, live_odds)
    signal, signal_note, signal_class = simple_betting_signal(edge)
    profit = format_profit_on_stake(live_odds, stake)
    external_url = sportsbook_external_url(sportsbook_key)
    last_update = selected_row.get("last_update")
    update_note = f"Updated {format_game_time(last_update)}." if last_update else "Updated by the live odds feed."

    st.html(
        f"""
        <div class="bet-slip">
            <div class="bet-slip-stat-grid">
                <div class="bet-slip-stat">
                    <div class="betting-simple-label">Live odds</div>
                    <div class="betting-simple-value">{html.escape(format_american_odds(live_odds))}</div>
                </div>
                <div class="bet-slip-stat">
                    <div class="betting-simple-label">Win chance</div>
                    <div class="betting-simple-value">{html.escape(format_plain_percent(model_probability))}</div>
                </div>
                <div class="bet-slip-stat">
                    <div class="betting-simple-label">If it wins</div>
                    <div class="betting-simple-value">{html.escape(profit)}</div>
                </div>
            </div>
            <div class="betting-simple-value">
                <span class="edge-pill {signal_class}">{html.escape(signal)}</span>
            </div>
            <div class="betting-simple-note">{html.escape(signal_note)}</div>
            <div class="bet-slip-warning">
                Odds update with the selected sportsbook. This app does not submit wagers.
                Confirm the team and price inside the sportsbook. {html.escape(update_note)}
            </div>
        </div>
        """
    )
    render_sportsbook_open_link(sportsbook_name, external_url)
    save_col, clear_col = st.columns(2)

    with save_col:
        if st.button("Save pick", key=f"mlb_slip_save_{game_key}_{safe_streamlit_dom_key(selection)}", width="stretch"):
            game_date = pd.to_datetime(game.get("GAME_DATE"), errors="coerce")
            append_bet_tracker_row(
                {
                    "Date Logged": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "Game Date": game_date.strftime("%Y-%m-%d") if not pd.isna(game_date) else "",
                    "Matchup": format_matchup(game),
                    "Market": "Moneyline",
                    "Selection": selection,
                    "Odds": live_odds,
                    "Stake": stake,
                    "Model Probability": model_probability,
                    "Market Probability": market_probability,
                    "Edge": edge,
                    "EV/Unit": ev,
                    "Kelly": kelly,
                    "Result": "Open",
                    "Notes": f"{sportsbook_name}: live odds",
                }
            )
            st.success("Pick saved to the Saved Picks tab.")

    with clear_col:
        if st.button("Clear", key=f"mlb_slip_clear_{game_key}_{safe_streamlit_dom_key(selection)}", width="stretch"):
            st.session_state.pop(BET_SLIP_STATE_KEY, None)
            st.rerun()


def render_sportsbook_handoff(
    game: pd.Series,
    selection: str,
    selected_data: dict,
    book_rows: list[dict],
) -> dict:
    """Render external sportsbook handoff controls and return selected price data."""
    away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
    home_team = str(game.get("HOME_TEAM", "Home") or "Home")
    game_pk = str(game.get("GAME_PK", "") or safe_streamlit_dom_key(format_matchup(game)))
    options = sportsbook_option_rows(book_rows, selection, home_team, away_team)
    option_labels = []

    for option in options:
        price = option.get("price")
        price_text = format_american_odds(price) if price is not None else "open app"
        option_labels.append(f"{option['sportsbook']} / {price_text}")

    if not options:
        options = [
            {
                "sportsbook_key": "draftkings",
                "sportsbook": "DraftKings",
                "price": None,
                "row": {},
            }
        ]
        option_labels = ["DraftKings / open app"]

    selected_label = st.selectbox(
        "Sportsbook handoff",
        option_labels,
        key=f"mlb_sportsbook_handoff_{game_pk}",
    )
    selected_index = option_labels.index(selected_label)
    selected_option = options[selected_index]
    selected_row = selected_option.get("row") or {}
    selected_price = selected_option.get("price")
    market_probability = sportsbook_market_probability_for_selection(
        selected_row,
        selection,
        home_team,
        away_team,
    ) if selected_row else None

    if selected_price is None:
        selected_price = selected_data.get("odds")

    if market_probability is None:
        market_probability = selected_data.get("market_probability")

    model_probability = selected_data.get("model_probability")
    edge = (
        float(model_probability) - float(market_probability)
        if model_probability is not None and market_probability is not None
        else selected_data.get("edge")
    )
    ev = expected_value_per_unit(model_probability, selected_price)
    kelly = kelly_fraction(model_probability, selected_price)
    sportsbook_key = selected_option.get("sportsbook_key")
    sportsbook_name = selected_option.get("sportsbook")
    external_url = sportsbook_external_url(sportsbook_key)
    handoff_col, price_col, link_col = st.columns([1, 1, 1])

    with handoff_col:
        st.html(
            f"""
            <div class="betting-handoff-card">
                <div class="betting-handoff-label">Sportsbook</div>
                <div class="betting-handoff-value">{html.escape(str(sportsbook_name))}</div>
                <div class="betting-handoff-note">Selected for this side.</div>
            </div>
            """
        )

    with price_col:
        st.html(
            f"""
            <div class="betting-handoff-card">
                <div class="betting-handoff-label">Current price</div>
                <div class="betting-handoff-value">{html.escape(format_american_odds(selected_price))}</div>
                <div class="betting-handoff-note">{html.escape(format_profit_on_stake(selected_price, 10.0))} profit on $10.</div>
            </div>
            """
        )

    with link_col:
        render_sportsbook_open_link(sportsbook_name, external_url)

        st.caption("Review and confirm externally.")

    return {
        "sportsbook": sportsbook_name,
        "sportsbook_key": sportsbook_key,
        "odds": selected_price,
        "market_probability": market_probability,
        "edge": edge,
        "ev": ev,
        "kelly": kelly,
        "external_url": external_url,
    }


def format_tracker_money(value: object, signed: bool = False) -> str:
    """Format money values for saved-pick cards."""
    parsed = safe_float(value, default=None)

    if parsed is None:
        return "-"

    prefix = "+" if signed and parsed > 0 else ""
    return f"{prefix}${parsed:,.2f}"


def tracker_result_state(row: pd.Series, settled_view: bool = False) -> tuple[str, str]:
    """Return display text and class for a tracker row result."""
    result = str(row.get("Result", "") or "Open").strip().lower()

    if result in {"win", "loss", "push"}:
        return result.title(), result

    if settled_view:
        return "Needs result", "review"

    return "Open", "open"


def is_tracker_row_active(row: pd.Series) -> bool:
    """Return whether a saved pick belongs in Active Picks."""
    result = str(row.get("Result", "") or "Open").strip().lower()

    if result in {"win", "loss", "push"}:
        return False

    game_date = parse_tracker_game_date(row.get("Game Date"))

    if game_date is None:
        return True

    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    return game_date >= today


def is_tracker_row_settled(row: pd.Series) -> bool:
    """Return whether a saved pick belongs in Settled Bets."""
    result = str(row.get("Result", "") or "Open").strip().lower()

    if result in {"win", "loss", "push"}:
        return True

    game_date = parse_tracker_game_date(row.get("Game Date"))

    if game_date is None:
        return False

    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    return game_date < today


def sort_tracker_cards(tracker: pd.DataFrame) -> pd.DataFrame:
    """Sort tracker cards by game date and save time, newest first."""
    if tracker.empty:
        return tracker

    display = tracker.copy()
    display["_game_date_sort"] = pd.to_datetime(display["Game Date"], errors="coerce")
    display["_saved_sort"] = pd.to_datetime(display["Date Logged"], errors="coerce")
    display["_bet_id_sort"] = pd.to_numeric(display["Bet ID"], errors="coerce")
    return display.sort_values(
        ["_game_date_sort", "_saved_sort", "_bet_id_sort"],
        ascending=[False, False, False],
    )


def render_tracker_pick_card(row: pd.Series, key_prefix: str, settled_view: bool = False) -> None:
    """Render one saved pick as a card with a delete action."""
    result_label, result_class = tracker_result_state(row, settled_view=settled_view)
    accent = {
        "win": "#16a34a",
        "loss": "#dc2626",
        "push": "#d97706",
        "review": "#d97706",
        "open": "#2563eb",
    }.get(result_class, "#0f766e")
    selection = str(row.get("Selection", "") or "Pick")
    market = str(row.get("Market", "") or "Market")
    matchup = str(row.get("Matchup", "") or "Game")
    game_date = str(row.get("Game Date", "") or "Date TBD")
    saved_at = str(row.get("Date Logged", "") or "")
    notes = str(row.get("Notes", "") or "").strip()
    profit = format_tracker_money(row.get("Profit"), signed=True)
    edge = format_betting_edge_value(row.get("Edge"))
    detail_items = [
        ("Odds", format_american_odds(row.get("Odds"))),
        ("Stake", format_tracker_money(row.get("Stake"))),
        ("Profit", profit),
        ("Edge", edge),
    ]
    detail_html = "".join(
        f"""
        <div class="saved-pick-detail">
            <div class="betting-simple-label">{html.escape(label)}</div>
            <div class="betting-simple-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in detail_items
    )
    note_html = (
        f'<div class="saved-pick-note">{html.escape(notes)}</div>'
        if notes
        else ""
    )

    card_col, action_col = st.columns([0.82, 0.18], gap="small")

    with card_col:
        st.html(
            f"""
            <div class="saved-pick-card" style="--pick-accent: {accent};">
                <div class="saved-pick-top">
                    <div>
                        <div class="saved-pick-title">{html.escape(selection)}</div>
                        <div class="saved-pick-meta">
                            {html.escape(market)} / {html.escape(matchup)}
                        </div>
                        <div class="saved-pick-meta">
                            Game {html.escape(game_date)}
                            {html.escape(f" / Saved {saved_at}" if saved_at else "")}
                        </div>
                    </div>
                    <span class="saved-pick-result {html.escape(result_class)}">
                        {html.escape(result_label)}
                    </span>
                </div>
                <div class="saved-pick-detail-grid">{detail_html}</div>
                {note_html}
            </div>
            """
        )

    with action_col:
        st.write("")
        st.write("")

        if st.button(
            "Delete",
            key=f"{key_prefix}_delete_{safe_streamlit_dom_key(row.get('Bet ID'))}",
            width="stretch",
        ):
            delete_bet_tracker_row(row.get("Bet ID"))
            st.rerun()


def render_tracker_card_list(
    tracker: pd.DataFrame,
    empty_message: str,
    key_prefix: str,
    settled_view: bool = False,
    limit: int = 24,
) -> None:
    """Render tracker rows as responsive saved-pick cards."""
    if tracker.empty:
        st.info(empty_message)
        return

    display = sort_tracker_cards(tracker).head(limit)

    for _, row in display.iterrows():
        render_tracker_pick_card(row, key_prefix=key_prefix, settled_view=settled_view)


def render_saved_betting_picks(limit: int = 24) -> None:
    """Render active saved picks as cards."""
    tracker = load_bet_tracker_with_auto_settlement()
    active = tracker[tracker.apply(is_tracker_row_active, axis=1)].copy() if not tracker.empty else tracker
    render_section_kicker("Active picks", "Finished games move to Settled Bets automatically.")
    render_tracker_card_list(
        active,
        "No active saved picks. Add one from Place Bets.",
        "mlb_active_pick",
        settled_view=False,
        limit=limit,
    )


def render_settled_betting_picks(limit: int = 48) -> None:
    """Render settled and expired saved picks as cards."""
    tracker = load_bet_tracker_with_auto_settlement()
    settled = tracker[tracker.apply(is_tracker_row_settled, axis=1)].copy() if not tracker.empty else tracker
    render_bet_tracker_summary(tracker)
    render_section_kicker("Settled bets", "Moneyline picks settle from final MLB scores.")
    render_tracker_card_list(
        settled,
        "No settled bets yet. Finished games will appear here.",
        "mlb_settled_pick",
        settled_view=True,
        limit=limit,
    )


@st.fragment(run_every=ODDS_REFRESH_SECONDS)
def render_odds_board() -> None:
    """Render moneyline odds and price-gap comparisons."""
    st.caption(
        "Choose a team to win. Pick a sportsbook in the slip; odds update automatically and cannot be typed in."
    )
    preview_stake = 10.0
    games = load_betting_slate_games("Today")

    if games.empty:
        st.info("No MLB games are available right now.")
        return

    predictions = build_game_predictions(games)

    if predictions.empty:
        st.warning("Game picks are not available yet. Run the MLB data refresh first.")
        return

    api_key = get_configured_odds_api_key()
    odds_error = ""
    odds_events = []

    if api_key:
        odds_events, odds_error = load_mlb_odds_with_status(api_key, markets="h2h,spreads,totals")

    odds_map = extract_moneyline_odds_map(odds_events)
    sportsbook_rows_map = extract_moneyline_sportsbook_rows(odds_events)

    if not api_key:
        st.info("Live odds are not connected yet, so picks cannot be added.")

    if odds_error:
        st.warning(f"{odds_error} Picks cannot be added until live odds are available.")

    if api_key and not odds_events and not odds_error:
        st.info("No live sportsbook odds came back from the feed right now. Picks will unlock when odds are available.")

    entries = []
    edge_rows = []

    for _, game in predictions.iterrows():
        away_team = str(game.get("AWAY_TEAM", "Away") or "Away")
        home_team = str(game.get("HOME_TEAM", "Home") or "Home")
        game_key = betting_game_key(away_team, home_team)
        api_odds = odds_map.get(game_key, {})
        book_rows = sportsbook_rows_map.get(game_key, [])
        edge_table, lookup = betting_edge_rows_for_game(
            game,
            api_odds.get("away_moneyline"),
            api_odds.get("home_moneyline"),
        )
        selected_payload = best_moneyline_selection_payload(game, lookup, book_rows)
        selected_edge = safe_float(selected_payload.get("edge"), default=None)
        selected_model = safe_float(selected_payload.get("model_probability"), default=0.0) or 0.0
        has_price = any(
            safe_float(value, default=None) is not None
            for value in [api_odds.get("away_moneyline"), api_odds.get("home_moneyline")]
        )
        entries.append(
            {
                "game": game,
                "lookup": lookup,
                "book_rows": book_rows,
                "selected_payload": selected_payload,
                "edge": selected_edge,
                "model_probability": selected_model,
                "has_price": has_price,
            }
        )

        for team, data in lookup.items():
            payload = moneyline_selection_payload(game, team, data, book_rows)
            if payload.get("edge") is None and payload.get("odds") is None:
                continue

            edge_rows.append(
                {
                    "Game": format_matchup(game),
                    "Team": team_abbreviation(team, data.get("team_id")),
                    "Sportsbook": payload.get("sportsbook", "Manual"),
                    "Odds": payload.get("odds"),
                    "Model": payload.get("model_probability"),
                    "Market": payload.get("market_probability"),
                    "Edge": payload.get("edge"),
                    "EV/Unit": payload.get("ev"),
                    "Kelly": payload.get("kelly"),
                }
            )

    render_betting_hero(
        edge_rows,
        "Live odds" if odds_events else "Manual mode",
    )

    entries = sorted(
        entries,
        key=lambda row: (
            safe_float(row["edge"], default=-9.0) or -9.0,
            safe_float(row["model_probability"], default=0.0) or 0.0,
        ),
        reverse=True,
    )

    board_col, slip_col = st.columns([1.0, 0.42], gap="large")

    with board_col:
        if not entries:
            st.info("No games are available for betting right now.")
        else:
            column_count = 1 if len(entries) == 1 else 2
            columns = st.columns(column_count, gap="medium")

            for index, entry in enumerate(entries, start=1):
                with columns[(index - 1) % column_count]:
                    render_moneyline_game_card(
                        game=entry["game"],
                        lookup=entry["lookup"],
                        book_rows=entry["book_rows"],
                        stake=float(preview_stake),
                        rank=index,
                    )

    slip_book_rows = None
    slip = st.session_state.get(BET_SLIP_STATE_KEY)

    if slip:
        slip_game = pd.Series(slip.get("game", {}))
        slip_away = str(slip_game.get("AWAY_TEAM", "Away") or "Away")
        slip_home = str(slip_game.get("HOME_TEAM", "Home") or "Home")
        slip_game_key = betting_game_key(slip_away, slip_home)
        slip_book_rows = sportsbook_rows_map.get(slip_game_key)

        if slip_book_rows is None:
            slip_book_rows = slip.get("book_rows", [])

    with slip_col:
        render_bet_slip_panel(slip_book_rows)


def get_prop_players_for_game(game: pd.Series) -> list[dict]:
    """Return players available for the props lab."""
    game_pk = str(game.get("GAME_PK", "") or "")
    feed = load_live_game_feed(game_pk) if game_pk else {}
    players = extract_team_players(feed, "away") + extract_team_players(feed, "home")

    if players:
        return sorted(players, key=lambda player: str(player.get("PLAYER_NAME", "")))

    fallback = [
        build_probable_pitcher_profile(game, feed, "away"),
        build_probable_pitcher_profile(game, feed, "home"),
    ]
    return [
        player
        for player in fallback
        if str(player.get("PLAYER_NAME", "") or "") != "Probable pitcher TBD"
    ]


def render_player_props_lab() -> None:
    """Render simple player prop projection tools."""
    st.info("Automatic player prop odds are not connected yet. Moneyline odds are automatic in the Moneyline tab.")
    return
    st.caption("Quick prop projections use current season player stats from the MLB feed. Treat them as a research starting point, not a pick generator.")
    games = load_betting_slate_games("Today")

    if games.empty:
        games = load_betting_slate_games("Next Upcoming")

    if games.empty:
        st.info("No games are available for player props.")
        return

    game_options = {
        f"{format_matchup(game)} / {format_game_time(game.get('GAME_DATETIME'))}": game
        for _, game in games.iterrows()
    }
    game_label = st.selectbox("Game", list(game_options.keys()), key="mlb_prop_game")
    game = game_options[game_label]
    players = get_prop_players_for_game(game)

    if not players:
        st.info("Player data is not available for this game yet.")
        return

    player_options = {
        f"{player.get('PLAYER_NAME')} / {player.get('POSITION', '-') or '-'}": player
        for player in players
    }
    player_label = st.selectbox("Player", list(player_options.keys()), key="mlb_prop_player")
    player = player_options[player_label]
    prop_col, line_col, side_col, odds_col = st.columns([1.4, 0.8, 0.8, 0.8])

    with prop_col:
        prop_type = st.selectbox(
            "Prop",
            ["Batter hits", "Total bases", "RBIs", "Home runs", "Pitcher strikeouts"],
            key="mlb_prop_type",
        )

    with line_col:
        line = st.number_input("Line", min_value=0.5, value=0.5, step=0.5, key="mlb_prop_line")

    with side_col:
        side = st.selectbox("Side", ["Over", "Under"], key="mlb_prop_side")

    with odds_col:
        odds = st.number_input("Odds", min_value=-10000, max_value=10000, value=-110, step=5, key="mlb_prop_odds")

    projection, note = project_player_prop(player, prop_type)

    if projection is None:
        st.warning(note)
        return

    over_probability = probability_over_poisson(float(line), projection)
    model_probability = over_probability if side == "Over" else 1 - over_probability
    market_probability = american_odds_to_implied_probability(odds)
    edge = (
        model_probability - market_probability
        if market_probability is not None
        else None
    )
    ev = expected_value_per_unit(model_probability, odds)
    kelly = kelly_fraction(model_probability, odds)
    render_prop_beginner_card(
        player=player,
        prop_type=prop_type,
        side=side,
        line=float(line),
        projection=float(projection),
        model_probability=float(model_probability),
        market_probability=market_probability,
        odds=odds,
        edge=edge,
        note=note,
    )
    metrics = [
        {"label": "Projection", "value": f"{projection:.2f}", "note": prop_type, "color": "#2563eb"},
        {"label": "Win Chance", "value": f"{model_probability:.1%}", "note": f"{side} {line:g}", "color": "#16a34a"},
        {"label": "Advanced Edge", "value": f"{ev:+.2f}" if ev is not None else "-", "note": format_signed_percent(edge), "color": "#7c3aed"},
    ]

    with st.expander("Advanced numbers"):
        render_dashboard_cards(metrics)
        advanced = pd.DataFrame(
            [
                {
                    "Book Price": format_american_odds(odds),
                    "Book Chance": f"{market_probability:.1%}" if market_probability is not None else "-",
                    "Win Chance": f"{model_probability:.1%}",
                    "Price Gap": format_signed_percent(edge),
                    "EV/Unit": f"{ev:+.2f}" if ev is not None else "-",
                    "Kelly": f"{kelly:.1%}" if kelly is not None else "-",
                }
            ]
        )
        st.dataframe(advanced, hide_index=True, width="stretch")

    notes = st.text_input("Tracker notes", value="", key="mlb_prop_notes")

    if st.button("Log prop bet", width="stretch", key="mlb_log_prop_bet"):
        game_date = pd.to_datetime(game.get("GAME_DATE"), errors="coerce")
        append_bet_tracker_row(
            {
                "Date Logged": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Game Date": game_date.strftime("%Y-%m-%d") if not pd.isna(game_date) else "",
                "Matchup": format_matchup(game),
                "Market": prop_type,
                "Selection": f"{player.get('PLAYER_NAME')} {side} {line:g}",
                "Odds": odds,
                "Stake": 1.0,
                "Model Probability": model_probability,
                "Market Probability": market_probability,
                "Edge": edge,
                "EV/Unit": ev,
                "Kelly": kelly,
                "Result": "Open",
                "Notes": notes,
            }
        )
        st.success("Prop added to tracker.")


def api_list_items(payload: object) -> list[dict]:
    """Return the first list of dictionary items from a provider payload."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ["data", "objects", "results", "betSlips", "items"]:
        value = payload.get(key)

        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    return []


def first_nested_value(row: dict, paths: list[list[str]]) -> object:
    """Return the first present nested provider value."""
    for path in paths:
        value = get_nested(row, path, default=None)

        if value not in [None, ""]:
            return value

    return None


def summarize_sharpsports_selection(row: dict) -> str:
    """Return a compact synced bet selection label."""
    bets = row.get("bets") or row.get("legs") or []

    if isinstance(bets, list) and bets:
        labels = []

        for bet in bets[:3]:
            if not isinstance(bet, dict):
                continue

            label = (
                first_nested_value(
                    bet,
                    [
                        ["marketSelection", "name"],
                        ["marketSelection", "position"],
                        ["outcome", "name"],
                        ["selection"],
                        ["name"],
                    ],
                )
                or first_nested_value(
                    bet,
                    [
                        ["proposition", "name"],
                        ["market", "name"],
                    ],
                )
            )

            if label:
                labels.append(str(label))

        if labels:
            suffix = f" +{len(bets) - 3}" if len(bets) > 3 else ""
            return " / ".join(labels) + suffix

    return str(
        first_nested_value(
            row,
            [
                ["selection"],
                ["name"],
                ["description"],
                ["outcome", "name"],
            ],
        )
        or "Synced bet"
    )


def flatten_synced_bet_slips(payload: object) -> pd.DataFrame:
    """Flatten synced SharpSports bet slips into a display table."""
    rows = []

    for row in api_list_items(payload):
        book = first_nested_value(
            row,
            [
                ["book", "name"],
                ["book", "abbr"],
                ["bookAbbr"],
                ["sportsbook"],
            ],
        )
        placed_at = first_nested_value(row, [["timePlaced"], ["placedAt"], ["createdAt"]])
        at_risk = first_nested_value(row, [["atRisk"], ["stake"], ["adjusted", "atRisk"]])
        to_win = first_nested_value(row, [["toWin"], ["potentialProfit"], ["adjusted", "toWin"]])
        profit = first_nested_value(row, [["netProfit"], ["profit"], ["adjusted", "netProfit"]])
        odds = first_nested_value(row, [["oddsAmerican"], ["odds"], ["price"]])
        status = first_nested_value(row, [["status"], ["outcome"], ["result"]])
        rows.append(
            {
                "Placed": format_game_time(placed_at) if placed_at else "-",
                "Sportsbook": sportsbook_label(book),
                "Selection": summarize_sharpsports_selection(row),
                "Odds": format_american_odds(odds),
                "At Risk": safe_float(at_risk, default=None),
                "To Win": safe_float(to_win, default=None),
                "Profit": safe_float(profit, default=None),
                "Status": str(status or "-"),
            }
        )

    return pd.DataFrame(rows)


def render_synced_bet_summary(bets: pd.DataFrame) -> None:
    """Render synced-bet account summary cards."""
    if bets.empty:
        render_dashboard_cards(
            [
                {"label": "Synced Bets", "value": "0", "note": "No provider rows returned.", "color": "#64748b"},
                {"label": "At Risk", "value": "$0.00", "note": "Requires linked accounts.", "color": "#2563eb"},
                {"label": "Profit", "value": "$0.00", "note": "Provider settled rows.", "color": "#16a34a"},
            ]
        )
        return

    at_risk = pd.to_numeric(bets.get("At Risk"), errors="coerce").fillna(0).sum()
    profit = pd.to_numeric(bets.get("Profit"), errors="coerce").fillna(0).sum()
    settled = bets[bets["Profit"].notna()].copy()
    roi = profit / at_risk if at_risk else None
    render_dashboard_cards(
        [
            {"label": "Synced Bets", "value": str(len(bets)), "note": f"{len(settled)} settled with profit data.", "color": "#2563eb"},
            {"label": "At Risk", "value": f"${at_risk:,.2f}", "note": "From synced sportsbooks.", "color": "#7c3aed"},
            {"label": "Profit / ROI", "value": f"${profit:,.2f}", "note": f"ROI {roi:.1%}" if roi is not None else "ROI unavailable.", "color": "#16a34a" if profit >= 0 else "#dc2626"},
        ]
    )


def render_sportsbook_connections_view() -> None:
    """Render provider account-linking and synced-bet controls."""
    st.caption("Connect sportsbook accounts to see placed bets, open bets, and profit history in one place.")
    odds_key = bool(get_configured_odds_api_key())
    public_key = get_configured_sharpsports_public_key()
    private_key = get_configured_sharpsports_private_key()
    render_dashboard_cards(
        [
            {"label": "Live Prices", "value": "On" if odds_key else "Manual", "note": "Odds can still be typed in.", "color": "#2563eb"},
            {"label": "Account Sync", "value": "Ready" if public_key else "Setup Needed", "note": "Connect sportsbook accounts.", "color": "#16a34a" if public_key else "#64748b"},
            {"label": "Bet History", "value": "Ready" if private_key else "Setup Needed", "note": "Placed bets and profit/loss.", "color": "#16a34a" if private_key else "#64748b"},
        ]
    )

    render_section_kicker("Connect sportsbook account")
    st.html(
        """
        <div class="betting-panel">
            <div class="betting-game-title">Secure account connection</div>
            <div class="betting-game-meta">
                Users should connect through the provider page. This app should not collect or store sportsbook passwords.
            </div>
        </div>
        """
    )
    user_id = "local_user"
    webhook_url = get_configured_secret_or_env("SHARPSPORTS_WEBHOOK_URL")

    with st.expander("Developer setup"):
        user_id = st.text_input(
            "Internal user ID",
            value=user_id,
            key="mlb_sportsbook_internal_user",
            help="Use your app's user id in a deployed version. Local testing can keep this default.",
        )
        webhook_url = st.text_input(
            "Webhook URL",
            value=webhook_url,
            key="mlb_sportsbook_webhook",
            help="A deployed backend webhook is required for a production account-linking flow.",
        )

    if public_key:
        if st.button("Connect sportsbook account", width="stretch", key="mlb_create_account_link"):
            payload = {
                "internalId": user_id.strip() or "local_user",
            }

            if webhook_url.strip():
                payload["webhookUrl"] = webhook_url.strip()

            try:
                context = create_sharpsports_context(public_key, payload)
                link_url = extract_sharpsports_context_url(context, mode="link")
            except Exception as exc:
                link_url = ""
                st.error(f"Could not create account-link session: {exc}")

            if link_url:
                st.session_state["mlb_account_link_url"] = link_url

        link_url = st.session_state.get("mlb_account_link_url", "")

        if link_url:
            st.link_button("Open secure connection page", link_url, width="stretch")
    else:
        st.info("Account syncing is not connected for this app yet.")

    render_section_kicker("Synced sportsbook bets")

    if not private_key:
        st.info("Bet history syncing is not connected for this app yet.")
        return

    sync_col_1, sync_col_2 = st.columns([1, 1])

    with sync_col_1:
        status = st.selectbox(
            "Status filter",
            ["", "pending", "completed", "cancelled"],
            format_func=lambda value: "All" if not value else value.title(),
            key="mlb_synced_bet_status",
        )

    with sync_col_2:
        limit = st.number_input("Limit", min_value=10, max_value=250, value=50, step=10, key="mlb_synced_bet_limit")

    if st.button("Refresh synced bets", width="stretch", key="mlb_refresh_synced_bets"):
        load_sharpsports_bet_slips.clear()

    payload = load_sharpsports_bet_slips(private_key, status=status, limit=int(limit))
    bets = flatten_synced_bet_slips(payload)
    render_synced_bet_summary(bets)

    if bets.empty:
        st.info("No synced bets were returned for the selected filter.")
        return

    display = bets.copy()

    for column in ["At Risk", "To Win", "Profit"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: f"${value:,.2f}" if pd.notna(value) else "-"
        )

    st.dataframe(display, hide_index=True, width="stretch")


def render_bet_tracker_summary(tracker: pd.DataFrame) -> None:
    """Render tracker results cards."""
    if tracker.empty:
        render_dashboard_cards(
            [
                {"label": "Open Bets", "value": "0", "note": "No saved tracker rows.", "color": "#64748b"},
                {"label": "Profit", "value": "$0.00", "note": "Set results after games finish.", "color": "#16a34a"},
                {"label": "ROI", "value": "-", "note": "Settled bets only.", "color": "#7c3aed"},
            ]
        )
        return

    results = tracker["Result"].astype(str).str.lower()
    settled = tracker[results.isin(["win", "loss", "push"])].copy()
    open_count = int((results == "open").sum())
    profit = pd.to_numeric(settled.get("Profit"), errors="coerce").fillna(0).sum() if not settled.empty else 0.0
    stake = pd.to_numeric(settled.get("Stake"), errors="coerce").fillna(0).sum() if not settled.empty else 0.0
    wins = int((settled["Result"].astype(str).str.lower() == "win").sum()) if not settled.empty else 0
    losses = int((settled["Result"].astype(str).str.lower() == "loss").sum()) if not settled.empty else 0
    decisions = wins + losses
    roi = profit / stake if stake else None
    win_rate = wins / decisions if decisions else None
    render_dashboard_cards(
        [
            {"label": "Open Bets", "value": str(open_count), "note": f"{len(settled)} settled rows.", "color": "#2563eb"},
            {"label": "Profit", "value": f"${profit:,.2f}", "note": f"Stake ${stake:,.2f}", "color": "#16a34a" if profit >= 0 else "#dc2626"},
            {"label": "ROI / Win Rate", "value": f"{roi:.1%}" if roi is not None else "-", "note": f"Win rate {win_rate:.1%}" if win_rate is not None else "No decisions.", "color": "#7c3aed"},
        ]
    )


def render_bet_tracker_view() -> None:
    """Render editable saved bet tracker."""
    tracker = recalculate_tracker_results(load_bet_tracker())
    tracker["Result"] = tracker["Result"].fillna("Open").astype(str)
    tracker["Notes"] = tracker["Notes"].fillna("").astype(str)
    render_bet_tracker_summary(tracker)

    if tracker.empty:
        st.info("Tracked bets will appear here after you log one from the odds board or props lab.")
        return

    simple = tracker[
        ["Game Date", "Matchup", "Selection", "Odds", "Stake", "Result", "Profit", "Notes"]
    ].copy()
    simple["Odds"] = simple["Odds"].map(format_american_odds)
    simple["Stake"] = pd.to_numeric(simple["Stake"], errors="coerce").map(
        lambda value: f"${value:,.2f}" if pd.notna(value) else "-"
    )
    simple["Profit"] = pd.to_numeric(simple["Profit"], errors="coerce").map(
        lambda value: f"${value:,.2f}" if pd.notna(value) else "-"
    )
    simple = simple.rename(
        columns={
            "Game Date": "Date",
            "Matchup": "Game",
            "Selection": "Pick",
        }
    )
    st.dataframe(simple, hide_index=True, width="stretch")

    with st.expander("Edit tracker"):
        edited = st.data_editor(
            tracker,
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Result": st.column_config.SelectboxColumn(
                    "Result",
                    options=["Open", "Win", "Loss", "Push"],
                ),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
            key="mlb_bet_tracker_editor",
        )

        if st.button("Save tracker changes", width="stretch", key="mlb_save_bet_tracker"):
            save_bet_tracker(recalculate_tracker_results(edited))
            st.success("Bet tracker saved.")


def render_betting_backtest() -> None:
    """Render tracker and pricing-history breakdowns."""
    tracker = recalculate_tracker_results(load_bet_tracker())
    results = tracker["Result"].astype(str).str.lower() if not tracker.empty else pd.Series(dtype=str)
    settled = tracker[results.isin(["win", "loss", "push"])].copy() if not tracker.empty else pd.DataFrame()

    if not settled.empty:
        settled["Edge"] = pd.to_numeric(settled["Edge"], errors="coerce")
        settled["Stake"] = pd.to_numeric(settled["Stake"], errors="coerce").fillna(0)
        settled["Profit"] = pd.to_numeric(settled["Profit"], errors="coerce").fillna(0)
        settled["Edge Bucket"] = pd.cut(
            settled["Edge"],
            bins=[-1, 0, 0.02, 0.05, 1],
            labels=["No Edge", "0-2%", "2-5%", "5%+"],
            include_lowest=True,
        )
        grouped = settled.groupby("Edge Bucket", observed=False).agg(
            Bets=("Bet ID", "count"),
            Stake=("Stake", "sum"),
            Profit=("Profit", "sum"),
        ).reset_index()
        grouped["ROI"] = grouped.apply(
            lambda row: row["Profit"] / row["Stake"] if row["Stake"] else None,
            axis=1,
        )
        grouped["ROI"] = grouped["ROI"].map(lambda value: f"{value:.1%}" if value is not None else "-")
        grouped["Stake"] = grouped["Stake"].map(lambda value: f"${value:,.2f}")
        grouped["Profit"] = grouped["Profit"].map(lambda value: f"${value:,.2f}")
        st.dataframe(grouped, hide_index=True, width="stretch")
    else:
        st.info("Set bet results to Win/Loss/Push in the tracker to unlock edge-bucket results.")

    pricing_summary = build_team_backtest_summary(get_model_mtime())

    if not pricing_summary.empty:
        with st.expander("Pricing-history check"):
            st.caption("This shows historical pick results by team bucket. It is context, not a guarantee.")
            display = pricing_summary.copy()

            for column in ["Accuracy", "Avg_Confidence", "Predicted", "Actual"]:
                if column in display.columns:
                    display[column] = pd.to_numeric(display[column], errors="coerce").map(
                        lambda value: f"{value:.1%}" if pd.notna(value) else "-"
                    )

            st.dataframe(display, hide_index=True, width="stretch")


def render_betting_view() -> None:
    """Render betting research tools."""
    st.caption("A simple moneyline screen for choosing a team, checking odds, and tracking results.")
    place_tab, saved_tab, settled_tab = st.tabs(["Place Bets", "Saved Picks", "Settled Bets"])

    with place_tab:
        render_odds_board()

    with saved_tab:
        render_saved_betting_picks()

    with settled_tab:
        render_settled_betting_picks()


def render_teams_view() -> None:
    """Render MLB team profiles and strength table."""
    strength = load_team_strength()

    if strength.empty:
        st.warning("Missing MLB team strength data. Run `python src/mlb_team_strength.py`.")
        return

    strength = strength.sort_values("ELO", ascending=False).reset_index(drop=True)
    teams = strength["TEAM_NAME"].astype(str).tolist()
    selected_team = st.selectbox("Team", teams, index=0, key="mlb_profile_team")
    row = get_team_strength_row(selected_team, strength)
    league_rank = teams.index(selected_team) + 1
    team_id = row.get("TEAM_ID")
    logo_url = team_logo_url(team_id)
    logo_html = (
        f'<img class="profile-logo" src="{html.escape(logo_url)}" alt="{html.escape(selected_team)} logo">'
        if logo_url
        else ""
    )
    profile_metrics = [
        ("League rank", f"#{league_rank}"),
        ("Win %", f"{float(row['SEASON_WIN_PCT']):.0%}"),
        ("Run diff/G", f"{float(row['SEASON_RUN_DIFF_PER_GAME']):+.1f}"),
        ("Power rating", f"{float(row['ELO']):.0f}"),
    ]
    metric_html = "".join(
        f"""
        <div class="profile-stat">
            <div class="profile-label">{html.escape(label)}</div>
            <div class="profile-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in profile_metrics
    )
    st.html(
        f"""
        <div class="profile-hero" style="--team-color: {html.escape(get_team_color(selected_team))};">
            {logo_html}
            <div>
                <div class="profile-label">MLB Team Profile</div>
                <div class="profile-name">{html.escape(selected_team)}</div>
                <div class="profile-metrics">{metric_html}</div>
            </div>
        </div>
        """
    )

    left_col, right_col = st.columns(2)

    with left_col:
        render_section_kicker("Form")
        render_dashboard_cards(
            [
                {
                    "label": "Last 10",
                    "value": f"{float(row['ROLLING_WIN_PCT_10']):.0%}",
                    "note": f"Run diff {float(row['ROLLING_RUN_DIFF_10']):+.1f}",
                    "color": "#1d4ed8",
                },
                {
                    "label": "Runs/G",
                    "value": f"{float(row['SEASON_AVG_RUNS_FOR']):.1f}",
                    "note": "Season offense",
                    "color": "#166534",
                },
                {
                    "label": "Lineup",
                    "value": f"{numeric_row_value(row, 'PROJECTED_LINEUP_STRENGTH'):.1f}",
                    "note": "Run creation proxy",
                    "color": "#c2410c",
                },
            ]
        )

    with right_col:
        render_section_kicker("Schedule Load")
        render_dashboard_cards(
            [
                {
                    "label": "Rest",
                    "value": f"{float(row['DAYS_REST']):.0f} days",
                    "note": "Days since last game",
                    "color": "#7c3aed",
                },
                {
                    "label": "Allowed/G",
                    "value": f"{float(row['SEASON_AVG_RUNS_AGAINST']):.1f}",
                    "note": "Season prevention",
                    "color": "#b91c1c",
                },
                {
                    "label": "Bullpen",
                    "value": f"{numeric_row_value(row, 'BULLPEN_FATIGUE_PROXY'):.1f}",
                    "note": "Recent workload proxy",
                    "color": "#7c3aed",
                },
            ]
        )

    display = strength[
        [
            "TEAM_NAME",
            "ELO",
            "SEASON_WIN_PCT",
            "SEASON_RUN_DIFF_PER_GAME",
            "SEASON_AVG_RUNS_FOR",
            "SEASON_AVG_RUNS_AGAINST",
            "ROLLING_WIN_PCT_10",
            "ROLLING_RUN_DIFF_10",
            "ROLLING_WIN_PCT_5",
            "ROLLING_RUN_DIFF_5",
            "ROLLING_WIN_PCT_20",
            "ROLLING_RUN_DIFF_20",
            "BULLPEN_FATIGUE_PROXY",
            "PROJECTED_LINEUP_STRENGTH",
            "DAYS_REST",
        ]
    ].rename(
        columns={
            "TEAM_NAME": "Team",
            "SEASON_WIN_PCT": "Season Win %",
            "SEASON_RUN_DIFF_PER_GAME": "Run Diff/G",
            "SEASON_AVG_RUNS_FOR": "Runs/G",
            "SEASON_AVG_RUNS_AGAINST": "Allowed/G",
            "ROLLING_WIN_PCT_10": "Last 10 Win %",
            "ROLLING_RUN_DIFF_10": "Last 10 Run Diff",
            "ROLLING_WIN_PCT_5": "Last 5 Win %",
            "ROLLING_RUN_DIFF_5": "Last 5 Run Diff",
            "ROLLING_WIN_PCT_20": "Last 20 Win %",
            "ROLLING_RUN_DIFF_20": "Last 20 Run Diff",
            "BULLPEN_FATIGUE_PROXY": "Bullpen Fatigue",
            "PROJECTED_LINEUP_STRENGTH": "Lineup Strength",
            "DAYS_REST": "Rest",
        }
    )
    display.insert(0, "Rank", range(1, len(display) + 1))

    with st.expander("Power rankings table", expanded=True):
        st.dataframe(display, width="stretch", hide_index=True)
        st.download_button(
            "Download MLB team strength CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="mlb_current_team_strength.csv",
            mime="text/csv",
            width="stretch",
        )


def render_model_view() -> None:
    """Render MLB model status, metrics, and reliability."""
    st.header("Model")
    metrics = load_model_metrics()
    features = load_model_features()
    raw_games = load_raw_games()
    model_name = "Missing"
    feature_count = 0

    if not MODEL_PATH.exists():
        st.warning("Missing MLB model. Run `python src/mlb_train_model.py` after building features.")
    else:
        bundle = load_model_bundle(get_model_mtime())
        model_name = str(bundle.get("model_name", "Unknown"))
        feature_count = len(bundle.get("feature_columns", []))

    model_accuracy = "TBD"

    if not metrics.empty and "Accuracy" in metrics.columns:
        model_accuracy = f"{float(metrics.sort_values('Accuracy', ascending=False).iloc[0]['Accuracy']):.1%}"

    latest_game = "TBD"

    if not raw_games.empty and "GAME_DATE" in raw_games.columns:
        latest_game = pd.to_datetime(raw_games["GAME_DATE"].max()).strftime("%b %-d, %Y")

    render_status_grid(
        [
            {"label": "Active Model", "value": model_name, "note": "Saved joblib bundle", "color": "#1d4ed8"},
            {"label": "Features", "value": str(feature_count), "note": f"{len(features):,} training rows", "color": "#166534"},
            {"label": "Accuracy", "value": model_accuracy, "note": "Best saved model", "color": "#7c3aed"},
            {"label": "Data Through", "value": latest_game, "note": "Raw game file", "color": "#b91c1c"},
        ]
    )

    calibration = build_probability_calibration(get_model_mtime())
    backtest = build_team_backtest_summary(get_model_mtime())

    chart_col, reliability_col = st.columns(2)

    with chart_col:
        render_section_kicker("Calibration")

        if calibration.empty:
            st.caption("Calibration data is unavailable.")
        else:
            chart = calibration.set_index("Bucket")[["Predicted", "Actual"]]
            st.line_chart(chart)
            with st.expander("Calibration table", expanded=False):
                formatted = calibration.copy()
                formatted["Predicted"] = formatted["Predicted"].map("{:.1%}".format)
                formatted["Actual"] = formatted["Actual"].map("{:.1%}".format)
                st.dataframe(formatted, width="stretch", hide_index=True)

    with reliability_col:
        render_section_kicker("Team Reliability")

        if backtest.empty:
            st.caption("Team reliability is unavailable.")
        else:
            teams = sorted(backtest["Team"].dropna().unique())
            selected_team = st.selectbox("Backtest Team", teams, key="mlb_backtest_team")
            team_rows = backtest[backtest["Team"].eq(selected_team)].sort_values("Season")
            st.line_chart(team_rows.set_index("Season")[["Accuracy", "Avg_Confidence"]])
            latest = team_rows.iloc[-1]
            st.caption(
                f"Latest: {float(latest['Accuracy']):.1%} pick accuracy across "
                f"{int(latest['Games'])} team-games."
            )

    if not metrics.empty:
        formatted = metrics.copy()

        for column in ["Accuracy", "ROC_AUC", "Brier_Score", "Log_Loss"]:
            if column in formatted.columns:
                formatted[column] = formatted[column].map("{:.3f}".format)

        with st.expander("Model comparison", expanded=True):
            st.dataframe(formatted, width="stretch", hide_index=True)

            if {"Model", "Accuracy", "ROC_AUC"}.issubset(metrics.columns):
                st.bar_chart(metrics.set_index("Model")[["Accuracy", "ROC_AUC"]])

    st.subheader("Refresh Commands")
    st.code(
        "\n".join(
            [
                "python src/mlb_collect_data.py",
                "python src/mlb_features.py",
                "python src/mlb_team_strength.py",
                "python src/mlb_train_model.py",
            ]
        ),
        language="bash",
    )


def main() -> None:
    """Render MLB Streamlit app."""
    st.set_page_config(
        page_title="MLB Betting Dashboard",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.title("MLB Predictor")
    st.caption("Separate MLB app using the same project structure as the NBA predictor.")

    views = [
        {"label": "Home", "icon": ":material/home:"},
        {"label": "Today", "icon": ":material/calendar_today:"},
        {"label": "Game", "icon": ":material/sports_baseball:"},
        {"label": "Series", "icon": ":material/format_list_numbered:"},
        {"label": "Bracket", "icon": ":material/account_tree:"},
        {"label": "Standings", "icon": ":material/leaderboard:"},
        {"label": "Betting", "icon": ":material/attach_money:"},
        {"label": "Teams", "icon": ":material/groups:"},
    ]
    view_key = "mlb_selected_view"
    sidebar_compact_key = "mlb_sidebar_compact"
    valid_views = {view["label"] for view in views}

    if view_key not in st.session_state or st.session_state[view_key] not in valid_views:
        st.session_state[view_key] = "Home"

    try:
        has_win_probability_deep_link = bool(st.query_params.get(WIN_PROB_QUERY_PARAM, ""))
    except Exception:
        has_win_probability_deep_link = False

    if has_win_probability_deep_link:
        st.session_state[view_key] = "Today"

    if sidebar_compact_key not in st.session_state:
        st.session_state[sidebar_compact_key] = False

    if st.session_state[sidebar_compact_key]:
        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"],
                section[data-testid="stSidebar"] > div:first-child {
                    width: 5.2rem;
                    min-width: 5.2rem;
                    max-width: 5.2rem;
                }

                section[data-testid="stSidebar"] .sidebar-brand-text,
                section[data-testid="stSidebar"] .sidebar-nav-title {
                    display: none;
                }

                section[data-testid="stSidebar"] div.stButton > button {
                    justify-content: center;
                    padding-left: 0.4rem;
                    padding-right: 0.4rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"],
                section[data-testid="stSidebar"] > div:first-child {
                    width: 12rem;
                    min-width: 12rem;
                    max-width: 12rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">MLB</div>
                <div class="sidebar-brand-text">
                    <div class="sidebar-brand-name">MLB Predictor</div>
                    <div class="sidebar-brand-sub">Betting dashboard</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-toggle-wrap">', unsafe_allow_html=True)
        toggle_label = "Open" if st.session_state[sidebar_compact_key] else "Close"
        toggle_icon = (
            ":material/keyboard_double_arrow_right:"
            if st.session_state[sidebar_compact_key]
            else ":material/keyboard_double_arrow_left:"
        )

        if st.button(
            toggle_label if not st.session_state[sidebar_compact_key] else " ",
            key="mlb_sidebar_toggle",
            icon=toggle_icon,
            width="stretch",
            help="Open sidebar" if st.session_state[sidebar_compact_key] else "Collapse sidebar",
        ):
            st.session_state[sidebar_compact_key] = not st.session_state[sidebar_compact_key]
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-nav-title">View</div>', unsafe_allow_html=True)

        for view in views:
            selected = st.session_state[view_key] == view["label"]
            nav_label = view["label"] if not st.session_state[sidebar_compact_key] else " "

            if st.button(
                nav_label,
                key=f"mlb_sidebar_nav_{view['label'].lower()}",
                icon=view["icon"],
                type="primary" if selected else "secondary",
                width="stretch",
                help=view["label"],
            ):
                st.session_state[view_key] = view["label"]
                st.rerun()

    selected_view = st.session_state[view_key]

    if selected_view == "Home":
        render_home_view()
    elif selected_view == "Today":
        render_today_live_fragment()
    elif selected_view == "Game":
        render_game_view()
    elif selected_view == "Series":
        render_series_view()
    elif selected_view == "Bracket":
        render_bracket_view()
    elif selected_view == "Standings":
        render_standings_view()
    elif selected_view == "Betting":
        render_betting_view()
    elif selected_view == "Teams":
        render_teams_view()


if __name__ == "__main__":
    main()
