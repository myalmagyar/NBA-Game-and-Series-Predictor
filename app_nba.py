# app.py

from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import html
import math
import random
import re
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import subprocess
import sys
import time
import unicodedata

import joblib
import pandas as pd
import streamlit as st
from nba_api.stats.static import players as nba_players


DATA_DIR = Path("data")
MODELS_DIR = Path("models")

FEATURES_PATH = DATA_DIR / "model_features.csv"
RAW_GAMES_PATH = DATA_DIR / "raw_games.csv"
TEAM_STRENGTH_PATH = DATA_DIR / "current_team_strength.csv"
PLAYER_IMPACT_PATH = DATA_DIR / "current_player_impact.csv"
CURRENT_INJURIES_PATH = DATA_DIR / "current_injuries.csv"
MODEL_PATH = MODELS_DIR / "game_winner_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.csv"
BACKTEST_METRICS_PATH = DATA_DIR / "backtest_metrics.csv"
CALIBRATION_PATH = DATA_DIR / "calibration_metrics.csv"

CURRENT_INJURIES_COLUMNS = [
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

HOME_ELO_ADVANTAGE = 65.0

MODEL_PROBABILITY_WEIGHT = 0.70
ELO_PROBABILITY_WEIGHT = 0.30
PROBABILITY_SHRINKAGE = 0.88
MIN_SINGLE_GAME_PROBABILITY = 0.18
MAX_SINGLE_GAME_PROBABILITY = 0.82
TEAM_ADJUSTMENT_LOGIT_WEIGHT = 0.08
DEFAULT_UNMATCHED_PLAYER_IMPACT = 1.0
MAX_TEAM_AVAILABILITY_PENALTY = 15.0
LIVE_SCOREBOARD_TIMEOUT_SECONDS = 3
DEFAULT_SERIES_PROJECTION_SCALE = 1000
SCORE_SIMULATION_COUNT = 1000
SCORE_MARGIN_POINTS_PER_LOGIT = 6.0
SCORE_TOTAL_STD_DEV = 13.0
SCORE_MARGIN_STD_DEV = 12.0
QUARTER_SCORE_WEIGHTS = [0.245, 0.255, 0.25, 0.25]
FULL_GAME_SIMULATION_MINUTES = 48
GAME_SIMULATION_FRAME_DELAY_SECONDS = 0.2

INJURY_MODEL_FEATURES = {
    "DIFF_INJURY_WEIGHTED_IMPACT",
    "DIFF_OUT_PLAYER_COUNT",
    "DIFF_QUESTIONABLE_PLAYER_COUNT",
}

PLAYOFF_CONTEXT_MODEL_FEATURES = {
    "IS_PLAYOFF_GAME",
    "PLAYOFF_SERIES_GAME_NUMBER",
    "DIFF_SERIES_WINS_ENTERING",
    "HOME_SERIES_WINS_ENTERING",
    "AWAY_SERIES_WINS_ENTERING",
    "HOME_FACING_ELIMINATION",
    "AWAY_FACING_ELIMINATION",
    "HOME_CAN_CLINCH_SERIES",
    "AWAY_CAN_CLINCH_SERIES",
}

LOWER_IS_BETTER_FEATURE_PARTS = [
    "DEF_RATING",
    "TOV",
    "TOV_RATE",
]

HOME_COURT_SCHEDULE = [
    "higher",
    "higher",
    "lower",
    "lower",
    "higher",
    "lower",
    "higher",
]

DEFAULT_EAST_TEAMS = [
    "Boston Celtics",
    "New York Knicks",
    "Milwaukee Bucks",
    "Cleveland Cavaliers",
    "Orlando Magic",
    "Indiana Pacers",
    "Philadelphia 76ers",
    "Miami Heat",
]

DEFAULT_WEST_TEAMS = [
    "Denver Nuggets",
    "Minnesota Timberwolves",
    "Oklahoma City Thunder",
    "LA Clippers",
    "Dallas Mavericks",
    "Phoenix Suns",
    "Los Angeles Lakers",
    "New Orleans Pelicans",
]

TEAM_COLORS = {
    "Atlanta Hawks": "#E03A3E",
    "Boston Celtics": "#007A33",
    "Brooklyn Nets": "#000000",
    "Charlotte Hornets": "#1D1160",
    "Chicago Bulls": "#CE1141",
    "Cleveland Cavaliers": "#860038",
    "Dallas Mavericks": "#00538C",
    "Denver Nuggets": "#0E2240",
    "Detroit Pistons": "#C8102E",
    "Golden State Warriors": "#1D428A",
    "Houston Rockets": "#CE1141",
    "Indiana Pacers": "#002D62",
    "LA Clippers": "#C8102E",
    "Los Angeles Clippers": "#C8102E",
    "Los Angeles Lakers": "#552583",
    "Memphis Grizzlies": "#5D76A9",
    "Miami Heat": "#98002E",
    "Milwaukee Bucks": "#00471B",
    "Minnesota Timberwolves": "#0C2340",
    "New Orleans Pelicans": "#0C2340",
    "New York Knicks": "#006BB6",
    "Oklahoma City Thunder": "#007AC1",
    "Orlando Magic": "#0077C0",
    "Philadelphia 76ers": "#006BB6",
    "Phoenix Suns": "#1D1160",
    "Portland Trail Blazers": "#E03A3E",
    "Sacramento Kings": "#5A2D81",
    "San Antonio Spurs": "#8A8D8F",
    "Toronto Raptors": "#CE1141",
    "Utah Jazz": "#002B5C",
    "Washington Wizards": "#002B5C",
}

APP_CSS = """
<style>
    :root {
        --page: #f6f7fb;
        --surface: #ffffff;
        --surface-soft: #f9fafb;
        --ink: #172033;
        --muted: #667085;
        --line: #dfe4ea;
        --line-strong: #cbd5e1;
        --accent: #0f766e;
        --accent-blue: #2563eb;
        --accent-orange: #f97316;
        --accent-pink: #db2777;
        --accent-purple: #7c3aed;
        --accent-gold: #f59e0b;
        --success: #16a34a;
        --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.06);
        --shadow-md: 0 10px 28px rgba(16, 24, 40, 0.08);
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 6%, rgba(249, 115, 22, 0.14), transparent 28%),
            radial-gradient(circle at 92% 12%, rgba(37, 99, 235, 0.14), transparent 28%),
            radial-gradient(circle at 50% 0%, rgba(219, 39, 119, 0.08), transparent 30%),
            linear-gradient(180deg, #fffdf8 0%, #f6f7fb 320px, #f6f7fb 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(251, 252, 255, 0.88);
        backdrop-filter: blur(10px);
    }

    .block-container {
        max-width: 1320px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        padding-top: 0.9rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }

    h1 {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-pink), var(--accent-orange));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        font-size: 1.55rem !important;
        margin: 0 0 0.5rem !important;
    }

    h2 {
        font-size: 1.25rem !important;
        margin-top: 0.4rem !important;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.15rem;
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        color: #475467;
        font-weight: 700;
        padding: 0.45rem 0.9rem;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #eff6ff, #fdf2f8);
        border: 1px solid #bfdbfe;
        color: var(--accent-blue);
        box-shadow: var(--shadow-sm);
    }

    div[data-testid="stTabs"] button p {
        color: inherit;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        width: var(--sidebar-width, 10.25rem);
        min-width: var(--sidebar-width, 10.25rem);
        max-width: var(--sidebar-width, 10.25rem);
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.98));
        border-right: 1px solid var(--line);
        overflow-x: hidden;
    }

    [data-testid="stSidebar"][data-compact="true"],
    [data-testid="stSidebar"][data-compact="true"] > div:first-child {
        width: 4.9rem;
        min-width: 4.9rem;
        max-width: 4.9rem;
    }

    [data-testid="stSidebar"] .sidebar-brand {
        align-items: center;
        display: flex;
        gap: 0.55rem;
        margin: 0.55rem 0 0.7rem;
        padding: 0 0.2rem;
    }

    [data-testid="stSidebar"][data-compact="true"] .sidebar-brand {
        justify-content: center;
        margin: 0.55rem 0 0.45rem;
        padding: 0;
    }

    [data-testid="stSidebar"] .sidebar-brand-mark {
        align-items: center;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-pink));
        border-radius: 10px;
        color: #ffffff;
        display: flex;
        flex: 0 0 auto;
        font-size: 0.95rem;
        font-weight: 950;
        height: 2rem;
        justify-content: center;
        width: 2rem;
    }

    [data-testid="stSidebar"] .sidebar-brand-text {
        min-width: 0;
    }

    [data-testid="stSidebar"] .sidebar-brand-name {
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 950;
        line-height: 1.05;
    }

    [data-testid="stSidebar"] .sidebar-brand-sub {
        color: var(--muted);
        font-size: 0.65rem;
        font-weight: 700;
        line-height: 1.1;
    }

    [data-testid="stSidebar"][data-compact="true"] .sidebar-brand-text,
    [data-testid="stSidebar"][data-compact="true"] .sidebar-nav-title {
        display: none;
    }

    [data-testid="stSidebar"][data-compact="true"] .sidebar-brand-mark {
        width: 2.2rem;
        height: 2.2rem;
    }

    [data-testid="stSidebar"] .sidebar-header-row {
        align-items: center;
        display: grid;
        gap: 0.35rem;
        grid-template-columns: minmax(0, 1fr) auto;
        margin: 0.15rem 0 0.55rem;
    }

    [data-testid="stSidebar"][data-compact="true"] .sidebar-header-row {
        grid-template-columns: auto auto;
        justify-content: center;
    }

    [data-testid="stSidebar"] .sidebar-toggle-wrap .stButton > button {
        border-radius: 999px;
        min-height: 2rem;
        min-width: 2rem;
        padding: 0;
        width: 2rem;
    }

    [data-testid="stSidebar"] .sidebar-toggle-wrap .stButton > button span {
        font-size: 0.95rem;
        line-height: 1;
    }

    [data-testid="stSidebar"] .sidebar-nav-title {
        color: var(--accent-blue);
        font-size: 0.7rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        margin: 0.25rem 0 0.4rem;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        background: transparent;
        border: 0;
        box-shadow: none;
        display: grid;
        gap: 0.35rem;
        padding: 0;
    }

    [data-testid="stSidebar"][data-compact="true"] div[role="radiogroup"] {
        gap: 0.25rem;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #dbe4ff;
        border-radius: 9px;
        box-shadow: var(--shadow-sm);
        margin: 0;
        padding: 0.4rem 0.5rem;
        transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
    }

    [data-testid="stSidebar"][data-compact="true"] div[role="radiogroup"] label {
        align-items: center;
        display: flex;
        justify-content: center;
        min-height: 2.7rem;
        padding: 0.35rem 0.25rem;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(239, 246, 255, 0.98);
        border-color: #bfdbfe;
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.10);
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #475467;
        font-size: 0.76rem;
        font-weight: 850;
        margin: 0;
    }

    [data-testid="stSidebar"][data-compact="true"] div[role="radiogroup"] label p {
        font-size: 0.9rem;
        text-align: center;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-pink));
        border-color: transparent;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff;
    }

    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[title="Close sidebar"],
    button[title="Open sidebar"],
    button[aria-label*="sidebar"],
    button[title*="sidebar"] {
        display: none !important;
        pointer-events: none !important;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.55rem 0.65rem;
        box-shadow: var(--shadow-sm);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: var(--surface);
        border-color: var(--line);
        border-radius: 8px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid var(--line-strong);
        font-weight: 800;
        box-shadow: var(--shadow-sm);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        border-color: var(--accent-blue);
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #6d28d9);
        border-color: #1d4ed8;
    }

    .compact-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(239, 246, 255, 0.96) 40%, rgba(253, 242, 248, 0.96) 72%, rgba(255, 247, 237, 0.96) 100%);
        border: 1px solid var(--line);
        border-top: 4px solid var(--accent-orange);
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        padding: 1.4rem 1.1rem 1rem;
        margin-top: 0.35rem;
        margin-bottom: 0.85rem;
    }

    .app-name {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-pink), var(--accent-orange));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 1.8rem;
        line-height: 1.2;
        font-weight: 900;
        letter-spacing: 0;
        padding-top: 0.15rem;
    }

    .header-meta {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(203, 213, 225, 0.78);
        border-radius: 999px;
        color: #475467;
        font-size: 0.78rem;
        line-height: 1.45;
        padding: 0.32rem 0.7rem;
        text-align: right;
        overflow-wrap: anywhere;
    }

    .section-kicker {
        background: linear-gradient(90deg, var(--accent-orange), var(--accent-pink), var(--accent-blue));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0;
        text-transform: uppercase;
        margin: 0.65rem 0 0.35rem;
    }

    .compact-note {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
        margin: 0.1rem 0 0.45rem;
    }

    .result-callout {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 52%, #fdf2f8 100%);
        border: 1px solid #bfdbfe;
        border-left: 5px solid var(--accent-blue);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        padding: 0.75rem 0.9rem;
        margin: 0.55rem 0 0.7rem;
    }

    .result-label {
        color: var(--accent-blue);
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .result-title {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 1.45rem;
        line-height: 1.2;
        font-weight: 900;
        margin-top: 0.18rem;
    }

    .result-meta {
        color: var(--muted);
        font-size: 0.84rem;
        margin-top: 0.2rem;
    }

    .matchup-board {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        gap: 0.55rem;
        align-items: stretch;
        margin: 0.55rem 0 0.75rem;
    }

    .matchup-panel {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 14%, #ffffff), #ffffff 58%);
        border: 1px solid var(--line);
        border-top: 5px solid var(--team-color);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        min-height: 96px;
        padding: 0.75rem;
    }

    .matchup-panel-winner {
        background: linear-gradient(135deg, #ecfdf5, #eff6ff);
        border-color: #99f6e4;
    }

    .matchup-logo {
        width: 56px;
        height: 56px;
        object-fit: contain;
        flex: 0 0 auto;
    }

    .matchup-team {
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.2;
        font-weight: 900;
        overflow-wrap: anywhere;
    }

    .matchup-sub {
        color: var(--accent-blue);
        font-size: 0.75rem;
        margin-top: 0.12rem;
    }

    .matchup-prob {
        color: var(--accent-purple);
        font-size: 1.65rem;
        line-height: 1;
        font-weight: 900;
        margin-left: auto;
        white-space: nowrap;
    }

    .matchup-vs {
        background: linear-gradient(135deg, var(--accent-orange), var(--accent-pink));
        border-radius: 999px;
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 900;
        align-self: center;
        padding: 0.35rem 0.55rem;
    }

    .matchup-preview {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        gap: 0.75rem;
        align-items: stretch;
        margin: 0.15rem 0 0.75rem;
    }

    .preview-team-card {
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 16%, #ffffff), #ffffff 60%);
        border: 1px solid var(--line);
        border-top: 6px solid var(--team-color);
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        min-height: 180px;
        padding: 0.85rem;
        position: relative;
        overflow: hidden;
    }

    .preview-team-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 88% 18%, color-mix(in srgb, var(--team-color) 26%, transparent), transparent 30%),
            linear-gradient(135deg, rgba(37, 99, 235, 0.08), transparent 42%);
        pointer-events: none;
    }

    .preview-logo-row {
        align-items: center;
        display: flex;
        gap: 0.8rem;
        min-width: 0;
        position: relative;
        z-index: 2;
    }

    .preview-logo {
        width: 88px;
        height: 88px;
        object-fit: contain;
        flex: 0 0 auto;
    }

    .preview-role {
        color: var(--accent-blue);
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .preview-name {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 900;
        line-height: 1.1;
        margin-top: 0.25rem;
        overflow-wrap: anywhere;
    }

    .preview-pill {
        background: linear-gradient(135deg, #eef2ff, #fff7ed);
        border: 1px solid #fed7aa;
        border-radius: 999px;
        color: #9a3412;
        display: inline-flex;
        font-size: 0.78rem;
        font-weight: 900;
        margin-top: 0.55rem;
        padding: 0.25rem 0.65rem;
        position: relative;
        z-index: 2;
    }

    .preview-center {
        align-items: center;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-pink));
        border-radius: 999px;
        color: #ffffff;
        display: flex;
        font-size: 1rem;
        font-weight: 900;
        justify-content: center;
        min-height: 2.6rem;
        min-width: 2.6rem;
    }

    .logo-strip {
        display: grid;
        grid-template-columns: repeat(8, minmax(0, 1fr));
        gap: 0.5rem;
        margin: 0.45rem 0 0.85rem;
    }

    .logo-tile {
        align-items: center;
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 12%, #ffffff), #ffffff 65%);
        border: 1px solid var(--line);
        border-top: 4px solid var(--team-color);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        display: flex;
        gap: 0.45rem;
        min-height: 58px;
        min-width: 0;
        padding: 0.45rem;
    }

    .logo-tile img {
        width: 34px;
        height: 34px;
        object-fit: contain;
        flex: 0 0 auto;
    }

    .logo-tile span {
        color: var(--ink);
        font-size: 0.76rem;
        font-weight: 800;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .score-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        padding: 0.6rem 0.7rem;
        margin-bottom: 0.45rem;
    }

    .score-card-complete {
        border-color: #99f6e4;
    }

    .score-topline {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        color: var(--accent-blue);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .score-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 0.55rem;
        align-items: center;
        padding: 0.25rem 0;
    }

    .score-team-wrap {
        align-items: center;
        display: flex;
        gap: 0.45rem;
        min-width: 0;
    }

    .score-logo {
        height: 24px;
        width: 24px;
        object-fit: contain;
        flex: 0 0 auto;
    }

    .score-team {
        color: var(--ink);
        font-weight: 900;
        overflow-wrap: anywhere;
    }

    .score-wins {
        color: #1e3a8a;
        background: #dbeafe;
        border-radius: 6px;
        min-width: 2.15rem;
        text-align: center;
        padding: 0.18rem 0.45rem;
        font-weight: 900;
    }

    .score-chip {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 999px;
        color: #9a3412;
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.55rem;
        white-space: nowrap;
    }

    .score-meta {
        background: linear-gradient(135deg, #eff6ff, #fdf2f8);
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        color: #334155;
        font-size: 0.8rem;
        line-height: 1.35;
        margin-top: 0.45rem;
        padding: 0.5rem 0.6rem;
        display: grid;
        gap: 0.45rem;
    }

    .score-meta-section {
        display: grid;
        gap: 0.3rem;
    }

    .score-meta-section-title {
        color: #1e3a8a;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .score-meta-line {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #dbe4ff;
        border-radius: 8px;
        color: #334155;
        display: block;
        padding: 0.42rem 0.55rem;
    }

    .score-meta-line.score-meta-summary {
        background: #eef2ff;
        border-color: #c7d2fe;
        color: #1e3a8a;
        font-weight: 900;
    }

    .score-meta-line.score-meta-game {
        background: #ffffff;
    }

    .series-section-box {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border: 1px solid #dbe4ff;
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        margin-top: 0.5rem;
        padding: 0.55rem;
    }

    .series-section-title {
        color: var(--accent-blue);
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    .series-game-list {
        background: #ffffff;
        border: 1px solid #dbe4ff;
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }

    .series-game-row {
        padding: 0.4rem 0.55rem;
    }

    .series-game-row + .series-game-row {
        border-top: 1px solid #e2e8f0;
    }

    .series-game-label {
        color: #1e3a8a;
        display: block;
        font-weight: 900;
    }

    .series-game-detail {
        color: #475569;
        display: block;
        margin-top: 0.12rem;
    }

    .team-card {
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 14%, #ffffff), #ffffff 60%);
        border: 1px solid var(--line);
        border-top: 4px solid var(--team-color);
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        padding: 0.85rem;
        text-align: center;
        min-height: 180px;
        position: relative;
        overflow: hidden;
    }

    .team-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 50% 5%, color-mix(in srgb, var(--team-color) 24%, transparent), transparent 32%),
            linear-gradient(180deg, rgba(15, 118, 110, 0.06), transparent 42%);
        pointer-events: none;
    }

    .team-logo {
        width: 82px;
        height: 82px;
        object-fit: contain;
        margin-bottom: 0.45rem;
        position: relative;
        z-index: 2;
    }

    .team-name {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        position: relative;
        z-index: 2;
        overflow-wrap: anywhere;
    }

    .team-probability {
        color: var(--accent-purple);
        font-size: 2rem;
        font-weight: 900;
        margin-top: 0.25rem;
        position: relative;
        z-index: 2;
    }

    .winner-card {
        border-color: #86efac;
        border-top-color: var(--success);
    }

    .winner-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        background: #dcfce7;
        color: #166534;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 900;
        z-index: 3;
    }

    .vs-text {
        font-size: 0.85rem;
        font-weight: 900;
        text-align: center;
        padding-top: 4.2rem;
        color: #98a2b3;
        letter-spacing: 0;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.84rem;
        position: relative;
        z-index: 2;
    }

    .summary-box {
        background: linear-gradient(135deg, #ffffff, #eff6ff);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent-blue);
        padding: 0.7rem 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: var(--ink);
    }

    .bracket-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        padding: 0.7rem;
        margin-bottom: 0.55rem;
        min-height: 106px;
    }

    .bracket-round-title {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-pink));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 1rem;
        font-weight: 900;
        margin: 0.6rem 0 0.8rem 0;
    }

    .bracket-team-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.35rem 0;
        color: #344054;
        font-weight: 700;
    }

    .bracket-team-winner {
        color: var(--accent-pink);
    }

    .bracket-logo {
        width: 28px;
        height: 28px;
        object-fit: contain;
    }

    div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-radius: 999px;
        box-shadow: var(--shadow-sm);
        display: inline-flex;
        gap: 0.15rem;
        padding: 0.25rem;
        margin-bottom: 0.7rem;
    }

    div[role="radiogroup"] label {
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        font-weight: 800;
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.45rem 0 0.8rem;
    }

    .dashboard-card,
    .today-card,
    .profile-panel,
    .model-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        padding: 0.82rem;
    }

    .dashboard-card {
        border-top: 4px solid var(--card-color, var(--accent-blue));
        min-height: 104px;
    }

    .dashboard-label,
    .today-label,
    .profile-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .dashboard-value {
        color: var(--ink);
        font-size: 1.5rem;
        font-weight: 950;
        line-height: 1.1;
        margin-top: 0.25rem;
        overflow-wrap: anywhere;
    }

    .dashboard-note {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.35;
        margin-top: 0.25rem;
    }

    .team-mini-row,
    .player-row,
    .impact-row {
        align-items: center;
        border-bottom: 1px solid #edf2f7;
        display: grid;
        gap: 0.6rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        padding: 0.45rem 0;
    }

    .team-mini-row:last-child,
    .player-row:last-child,
    .impact-row:last-child {
        border-bottom: 0;
    }

    .team-mini-logo,
    .impact-logo {
        height: 34px;
        object-fit: contain;
        width: 34px;
    }

    .team-mini-name,
    .player-name,
    .impact-name {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 900;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .team-mini-meta,
    .player-meta,
    .impact-meta {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.25;
    }

    .team-mini-score,
    .impact-score {
        background: #eef2ff;
        border-radius: 999px;
        color: #3730a3;
        font-size: 0.8rem;
        font-weight: 900;
        padding: 0.22rem 0.55rem;
        white-space: nowrap;
    }

    .today-card {
        border-top: 5px solid var(--team-color, var(--accent-blue));
        margin-bottom: 0.55rem;
        overflow: hidden;
        position: relative;
    }

    .today-topline,
    .prediction-topline {
        align-items: center;
        display: flex;
        gap: 0.6rem;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .status-chip {
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        border-radius: 999px;
        color: #155e75;
        font-size: 0.74rem;
        font-weight: 900;
        padding: 0.18rem 0.55rem;
        white-space: nowrap;
    }

    .today-live-state {
        align-items: center;
        display: flex;
        gap: 0.45rem;
        margin-top: 0.25rem;
    }

    .today-state-badge {
        background: #dcfce7;
        border: 1px solid #86efac;
        border-radius: 999px;
        color: #166534;
        font-size: 0.7rem;
        font-weight: 950;
        padding: 0.16rem 0.5rem;
        white-space: nowrap;
    }

    .today-state-badge.is-countdown {
        background: #fff7ed;
        border-color: #fdba74;
        color: #9a3412;
    }

    .today-state-detail {
        color: #0f172a;
        font-size: 0.8rem;
        font-weight: 900;
        line-height: 1.2;
    }

    .today-matchup {
        align-items: center;
        display: grid;
        gap: 0.45rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    }

    .today-team {
        align-items: center;
        display: flex;
        gap: 0.55rem;
        min-width: 0;
    }

    .today-team:last-child {
        justify-content: flex-end;
        text-align: right;
    }

    .today-logo {
        height: 40px;
        object-fit: contain;
        width: 40px;
    }

    .today-name {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 950;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .today-team-score {
        color: var(--accent-purple);
        font-size: 1.55rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 0.18rem;
    }

    .today-score {
        color: var(--accent-purple);
        font-size: 0.98rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 0.12rem;
    }

    .today-play-line {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--accent-orange);
        border-radius: 8px;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 800;
        line-height: 1.3;
        margin: 0.55rem 0 0.45rem;
        padding: 0.42rem 0.55rem;
        overflow-wrap: anywhere;
    }

    .today-breakdown-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        margin: 0.55rem 0 0.75rem;
        padding: 0.8rem;
    }

    .today-breakdown-top {
        align-items: center;
        display: flex;
        gap: 0.6rem;
        justify-content: space-between;
        margin-bottom: 0.55rem;
    }

    .today-breakdown-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.15;
    }

    .today-breakdown-sub {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.3;
        margin-top: 0.12rem;
    }

    .today-breakdown-grid {
        display: grid;
        gap: 0.65rem;
        grid-template-columns: 0.95fr 1.05fr;
        align-items: stretch;
    }

    .today-breakdown-panel {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.65rem;
    }

    .today-breakdown-panel-title {
        color: var(--accent-blue);
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }

    .today-comparison-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbe4ff;
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        margin-top: 0.65rem;
        padding: 0.75rem;
    }

    .today-comparison-title {
        color: var(--ink);
        font-size: 0.88rem;
        font-weight: 950;
        letter-spacing: 0.04em;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }

    .today-star-stack {
        display: grid;
        gap: 0.5rem;
    }

    .today-player-card {
        background: linear-gradient(135deg, #ffffff, #f8fbff);
        border: 1px solid #dbe4ff;
        border-radius: 8px;
        display: grid;
        gap: 0.55rem;
        grid-template-columns: auto minmax(0, 1fr);
        min-height: 128px;
        padding: 0.6rem;
    }

    .today-player-card-head {
        align-items: center;
        display: grid;
        gap: 0.55rem;
        grid-template-columns: auto minmax(0, 1fr);
    }

    .today-player-photo {
        width: 42px;
        height: 42px;
        object-fit: cover;
        border-radius: 50%;
        background: #eef2ff;
        border: 1px solid #dbe4ff;
    }

    .today-player-logo {
        width: 30px;
        height: 30px;
        object-fit: contain;
        flex: 0 0 auto;
    }

    .today-player-name {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 950;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .today-player-meta {
        color: var(--muted);
        font-size: 0.75rem;
        line-height: 1.25;
    }

    .today-player-actions {
        align-self: end;
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
    }

    .today-impact-legend {
        border-top: 1px solid #e5e7eb;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
    }

    .today-impact-legend-title {
        color: var(--accent-blue);
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }

    .today-impact-legend-grid {
        display: grid;
        gap: 0.4rem;
        grid-template-columns: repeat(5, minmax(0, 1fr));
    }

    .today-impact-legend-item {
        background: linear-gradient(135deg, #ffffff, #f8fbff);
        border: 1px solid #dbe4ff;
        border-radius: 8px;
        padding: 0.45rem 0.5rem;
    }

    .today-impact-legend-score {
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 900;
        line-height: 1.1;
    }

    .today-impact-legend-label {
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1.2;
        margin-top: 0.15rem;
    }

    .today-impact-disclaimer {
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1.35;
        margin-top: 0.55rem;
    }

    @media (max-width: 980px) {
        .today-impact-legend-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    .today-lineup-grid {
        display: grid;
        gap: 0.5rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .today-lineup-team {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.55rem;
    }

    .today-lineup-team-name {
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 950;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }

    .today-lineup-row {
        align-items: center;
        display: grid;
        gap: 0.45rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        padding: 0.26rem 0;
    }

    .today-lineup-number {
        background: #eef2ff;
        border-radius: 999px;
        color: #3730a3;
        font-size: 0.72rem;
        font-weight: 950;
        min-width: 1.65rem;
        padding: 0.1rem 0.35rem;
        text-align: center;
    }

    .today-lineup-name {
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 900;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .today-lineup-meta {
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.2;
        white-space: nowrap;
    }

    .today-vs {
        background: linear-gradient(135deg, var(--accent-orange), var(--accent-pink));
        border-radius: 999px;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 950;
        padding: 0.22rem 0.45rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 48%, #fff7ed 100%);
        border: 1px solid #ddd6fe;
        border-left: 5px solid var(--team-color, var(--accent-purple));
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        margin: 0.55rem 0 0.7rem;
        padding: 0.8rem;
    }

    .prediction-main {
        align-items: center;
        display: grid;
        gap: 0.7rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
    }

    .prediction-logo {
        height: 64px;
        object-fit: contain;
        width: 64px;
    }

    .prediction-winner {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 950;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .prediction-sub {
        color: var(--muted);
        font-size: 0.83rem;
        line-height: 1.35;
        margin-top: 0.18rem;
    }

    .prediction-prob {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        color: var(--accent-purple);
        font-size: 1.6rem;
        font-weight: 950;
        line-height: 1;
        min-width: 5.8rem;
        padding: 0.45rem 0.55rem;
        text-align: center;
    }

    .probability-meter {
        background: #e5e7eb;
        border-radius: 999px;
        height: 0.55rem;
        margin-top: 0.6rem;
        overflow: hidden;
    }

    .probability-fill {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-pink), var(--accent-orange));
        border-radius: inherit;
        height: 100%;
        width: var(--probability-width);
    }

    .signal-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.55rem;
    }

    .signal-pill {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: #475467;
        font-size: 0.76rem;
        font-weight: 850;
        padding: 0.22rem 0.55rem;
    }

    .today-prediction-grid {
        display: grid;
        gap: 0.45rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: 0.58rem;
    }

    .today-prediction-cell {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #dbe4ee;
        border-top: 3px solid var(--team-color, var(--accent-blue));
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        min-width: 0;
        padding: 0.48rem 0.55rem;
    }

    .today-prediction-label {
        color: var(--muted);
        font-size: 0.66rem;
        font-weight: 950;
        letter-spacing: 0.04em;
        line-height: 1.05;
        text-transform: uppercase;
    }

    .today-prediction-value {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 950;
        line-height: 1.1;
        margin-top: 0.22rem;
        overflow-wrap: anywhere;
    }

    .today-prediction-note {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 800;
        line-height: 1.15;
        margin-top: 0.16rem;
    }

    .profile-hero {
        align-items: center;
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 18%, #ffffff), #ffffff 62%);
        border: 1px solid var(--line);
        border-top: 6px solid var(--team-color);
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        display: grid;
        gap: 1rem;
        grid-template-columns: auto minmax(0, 1fr);
        margin: 0.45rem 0 0.85rem;
        padding: 1rem;
    }

    .profile-logo {
        height: 112px;
        object-fit: contain;
        width: 112px;
    }

    .profile-name {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 950;
        line-height: 1.05;
        overflow-wrap: anywhere;
    }

    .profile-metrics {
        display: grid;
        gap: 0.55rem;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-top: 0.75rem;
    }

    .profile-stat {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(203, 213, 225, 0.72);
        border-radius: 8px;
        padding: 0.55rem;
    }

    .profile-value {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 950;
        margin-top: 0.18rem;
    }

    .profile-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
        margin-top: 0.6rem;
    }

    .model-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        margin-top: 0.75rem;
    }

    .score-sim-card {
        background: linear-gradient(135deg, #ffffff 0%, #ecfeff 46%, #fff7ed 100%);
        border: 1px solid #bae6fd;
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        margin: 0.7rem 0 0.85rem;
        overflow: hidden;
        padding: 0.95rem;
    }

    .score-sim-title {
        color: var(--accent-blue);
        font-size: 0.76rem;
        font-weight: 950;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .score-sim-note {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 700;
        line-height: 1.35;
        margin-top: 0.2rem;
        max-width: 58rem;
    }

    .game-sim-panel {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 48%, #fff7ed 100%);
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        box-shadow: var(--shadow-md);
        margin: 0.8rem 0 0.9rem;
        overflow: hidden;
        padding: 0.9rem;
    }

    .game-sim-controls {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .game-sim-stage {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem;
    }

    .game-sim-stage-head {
        align-items: baseline;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem 0.85rem;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }

    .game-sim-stage-title {
        color: var(--accent-blue);
        font-size: 0.76rem;
        font-weight: 950;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .game-sim-stage-minute {
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 850;
    }

    .game-sim-stage-note {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .game-sim-board {
        align-items: stretch;
        display: grid;
        gap: 0.7rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin-top: 0.35rem;
    }

    .game-sim-team {
        align-items: center;
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 14%, #ffffff), #ffffff 68%);
        border: 1px solid var(--line);
        border-top: 5px solid var(--team-color);
        border-radius: 8px;
        display: grid;
        gap: 0.65rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        min-width: 0;
        padding: 0.7rem;
    }

    .game-sim-logo {
        height: 48px;
        object-fit: contain;
        width: 48px;
    }

    .game-sim-name {
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 950;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .game-sim-meta {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        margin-top: 0.12rem;
    }

    .game-sim-score {
        color: var(--accent-purple);
        font-size: 2.1rem;
        font-weight: 950;
        line-height: 1;
        min-width: 3.1rem;
        text-align: right;
    }

    .game-sim-center {
        align-items: center;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-pink));
        border-radius: 999px;
        color: #ffffff;
        display: flex;
        flex-direction: column;
        font-size: 0.8rem;
        font-weight: 950;
        justify-content: center;
        min-width: 4.2rem;
        padding: 0.45rem 0.75rem;
        text-align: center;
    }

    .game-sim-center span {
        font-size: 1.05rem;
        line-height: 1.1;
    }

    .game-sim-progress {
        background: #e5e7eb;
        border-radius: 999px;
        height: 0.55rem;
        margin-top: 0.7rem;
        overflow: hidden;
    }

    .game-sim-progress-fill {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-pink), var(--accent-orange));
        border-radius: inherit;
        height: 100%;
        width: var(--progress-width);
    }

    .score-sim-board {
        align-items: stretch;
        display: grid;
        gap: 0.75rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin-top: 0.65rem;
    }

    .score-sim-team {
        align-items: center;
        background: linear-gradient(135deg, color-mix(in srgb, var(--team-color) 14%, #ffffff), #ffffff 66%);
        border: 1px solid var(--line);
        border-top: 5px solid var(--team-color);
        border-radius: 8px;
        display: grid;
        gap: 0.7rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        min-width: 0;
        padding: 0.72rem;
    }

    .score-sim-logo {
        height: 54px;
        object-fit: contain;
        width: 54px;
    }

    .score-sim-name {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }

    .score-sim-role {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 800;
        margin-top: 0.12rem;
    }

    .score-sim-points {
        color: var(--accent-purple);
        font-size: 2.2rem;
        font-weight: 950;
        line-height: 1;
        min-width: 3.2rem;
        text-align: right;
    }

    .score-sim-center {
        align-items: center;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-pink));
        border-radius: 999px;
        color: #ffffff;
        display: flex;
        font-size: 0.8rem;
        font-weight: 950;
        justify-content: center;
        min-width: 3.1rem;
        padding: 0 0.55rem;
    }

    .series-score-row {
        align-items: center;
        border-bottom: 1px solid #edf2f7;
        display: grid;
        gap: 0.5rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        padding: 0.38rem 0;
    }

    .series-score-row:last-child {
        border-bottom: 0;
    }

    .series-score-game {
        background: #fef3c7;
        border-radius: 999px;
        color: #92400e;
        font-size: 0.72rem;
        font-weight: 950;
        padding: 0.2rem 0.5rem;
        white-space: nowrap;
    }

    .series-score-matchup {
        color: var(--ink);
        font-size: 0.88rem;
        font-weight: 900;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .series-score-meta {
        color: var(--muted);
        font-size: 0.74rem;
        line-height: 1.25;
    }

    div[data-testid="column"] .dashboard-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    @media (max-width: 1000px) {
        .dashboard-grid,
        .profile-metrics {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 640px) {
        .compact-header {
            display: block;
        }

        .header-meta {
            margin-top: 0.35rem;
            text-align: left;
        }

        .matchup-board {
            grid-template-columns: 1fr;
        }

        .matchup-preview {
            grid-template-columns: 1fr;
        }

        .preview-center {
            min-height: 1.5rem;
        }

        .preview-logo {
            width: 82px;
            height: 82px;
        }

        .logo-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .dashboard-grid,
        .profile-metrics,
        .profile-grid,
        .model-grid {
            grid-template-columns: 1fr;
        }

        .matchup-vs {
            text-align: center;
        }

        .today-matchup,
        .prediction-main,
        .profile-hero,
        .score-sim-board {
            grid-template-columns: 1fr;
            text-align: center;
        }

        .today-team,
        .today-team:last-child {
            justify-content: center;
            text-align: center;
        }

        .score-sim-team {
            grid-template-columns: 1fr;
            text-align: center;
        }

        .today-breakdown-grid,
        .today-lineup-grid {
            grid-template-columns: 1fr;
        }

        .today-player-card {
            min-height: 0;
        }

        .score-sim-points {
            text-align: center;
        }

        .team-card {
            min-height: 170px;
        }

        .vs-text {
            padding: 0.25rem 0;
        }
    }
</style>
"""


@st.cache_resource
def load_model_bundle() -> dict:
    """Load trained model bundle."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_team_strength() -> pd.DataFrame:
    """Load current team strength data."""
    if not TEAM_STRENGTH_PATH.exists():
        raise FileNotFoundError(
            "Missing data/current_team_strength.csv. Run: python src/team_strength.py"
        )

    strength = pd.read_csv(TEAM_STRENGTH_PATH)
    strength["GAME_DATE"] = pd.to_datetime(strength["GAME_DATE"])
    return strength


@st.cache_data
def load_raw_games() -> pd.DataFrame:
    """Load raw NBA game logs."""
    if not RAW_GAMES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/raw_games.csv. Run: python src/collect_data.py"
        )

    games = pd.read_csv(RAW_GAMES_PATH)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    return games


@st.cache_data
def load_model_features() -> pd.DataFrame:
    """Load model feature rows for diagnostics."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Missing data/model_features.csv. Run: python src/features.py"
        )

    features = pd.read_csv(FEATURES_PATH)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])
    return features


@st.cache_data
def load_player_impact(file_mtime: float | None = None) -> pd.DataFrame:
    """Load automatic player impact ratings."""
    if not PLAYER_IMPACT_PATH.exists():
        raise FileNotFoundError(
            "Missing data/current_player_impact.csv. Run: python src/player_impact.py"
        )

    return pd.read_csv(PLAYER_IMPACT_PATH)


@st.cache_data
def load_current_injuries() -> pd.DataFrame:
    """Load latest saved official NBA injury report."""
    if not CURRENT_INJURIES_PATH.exists():
        return pd.DataFrame(columns=CURRENT_INJURIES_COLUMNS)

    try:
        return pd.read_csv(CURRENT_INJURIES_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CURRENT_INJURIES_COLUMNS)


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Load model comparison metrics."""
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_backtest_metrics() -> pd.DataFrame:
    """Load rolling backtest metrics."""
    if not BACKTEST_METRICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(BACKTEST_METRICS_PATH)


@st.cache_data
def load_calibration_metrics() -> pd.DataFrame:
    """Load probability calibration metrics."""
    if not CALIBRATION_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(CALIBRATION_PATH)


def clear_app_caches() -> None:
    """Clear cached app data after refreshes."""
    for loader in [
        load_model_bundle,
        load_team_strength,
        load_raw_games,
        load_model_features,
        load_player_impact,
        load_current_injuries,
        load_model_metrics,
        load_backtest_metrics,
        load_calibration_metrics,
        load_live_scoreboard_games,
        load_espn_scoreboard_games,
        load_espn_latest_play,
        load_scoreboard_v3_games,
        load_live_playbyplay_state,
        load_injury_report_schedule_games,
        load_today_games,
        load_next_upcoming_games,
        load_nba_schedule_games,
    ]:
        loader.clear()


def run_data_refresh_pipeline(retrain_model: bool = False) -> list[str]:
    """Run the free data refresh scripts used by the app."""
    scripts = [
        "src/collect_data.py",
        "src/injuries.py",
        "src/player_impact.py",
        "src/features.py",
        "src/team_strength.py",
    ]

    if retrain_model:
        scripts.append("src/train_model.py")

    messages = []

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise RuntimeError(f"{script} failed: {error_text}")

        messages.append(f"Ran {script}")

    clear_app_caches()
    return messages


def inject_custom_css() -> None:
    """Apply custom app styling."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def get_teams_from_strength(strength: pd.DataFrame) -> list[str]:
    """Return sorted NBA team names."""
    return sorted(strength["TEAM_NAME"].unique())


def get_known_team_names() -> set[str]:
    """Return normalized team names available to the model."""
    try:
        strength = load_team_strength()
    except Exception:
        return set()

    if strength.empty or "TEAM_NAME" not in strength.columns:
        return set()

    return {
        str(team).strip().lower()
        for team in strength["TEAM_NAME"].dropna().unique()
    }


def is_known_team_name(team_name: object) -> bool:
    """Return whether a name maps to a team in the current strength table."""
    normalized = str(team_name).strip().lower()
    return bool(normalized and normalized in get_known_team_names())


def filter_games_to_known_teams(games: pd.DataFrame) -> pd.DataFrame:
    """Drop schedule rows with non-NBA placeholder teams."""
    if games.empty or not {"Home Team", "Away Team"}.issubset(games.columns):
        return games

    known_teams = get_known_team_names()

    if not known_teams:
        return games

    keep_mask = (
        games["Home Team"].astype(str).str.strip().str.lower().isin(known_teams)
        & games["Away Team"].astype(str).str.strip().str.lower().isin(known_teams)
    )
    return games[keep_mask].copy().reset_index(drop=True)


def safe_team_index(teams: list[str], team_name: str, fallback: int = 0) -> int:
    """Return team index for Streamlit dropdown defaults."""
    return teams.index(team_name) if team_name in teams else fallback


def get_team_color(team_name: str) -> str:
    """Return team primary color."""
    return TEAM_COLORS.get(team_name, "#1f2937")


def get_team_strength_row(team_name: str, strength: pd.DataFrame) -> pd.Series:
    """Get one team's current strength row."""
    rows = strength[strength["TEAM_NAME"].str.lower() == team_name.lower()]

    if rows.empty:
        raise ValueError(f"Team not found in current strength table: {team_name}")

    return rows.iloc[0]


def get_logo_url(team_name: str) -> str:
    """Return NBA CDN logo URL for a team."""
    strength = load_team_strength()
    row = get_team_strength_row(team_name, strength)
    team_id = int(row["TEAM_ID"])
    return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"


def get_team_abbreviation(team_name: str) -> str:
    """Return team abbreviation."""
    strength = load_team_strength()
    row = get_team_strength_row(team_name, strength)
    return str(row["TEAM_ABBREVIATION"])


def get_team_abbreviation_from_strength(team_name: str) -> str:
    """Return team abbreviation from current team strength file."""
    strength = load_team_strength()
    row = get_team_strength_row(team_name, strength)
    return str(row["TEAM_ABBREVIATION"])


def get_team_name_by_id(team_id: int | str) -> str | None:
    """Look up a team name from the current strength file by NBA team id."""
    strength = load_team_strength()
    rows = strength[strength["TEAM_ID"].astype(str).eq(str(team_id))]

    if rows.empty:
        return None

    return str(rows.iloc[0]["TEAM_NAME"])


def get_team_name_by_abbreviation(abbreviation: str) -> str | None:
    """Look up a team name from the current strength file by abbreviation."""
    strength = load_team_strength()
    rows = strength[
        strength["TEAM_ABBREVIATION"].str.upper().eq(str(abbreviation).upper())
    ]

    if rows.empty:
        return None

    return str(rows.iloc[0]["TEAM_NAME"])


def format_team_record(team_name: str) -> str:
    """Return the latest season record for one team when available."""
    try:
        games = load_raw_games()
    except FileNotFoundError:
        return "Record unavailable"

    if games.empty or "SEASON" not in games.columns:
        return "Record unavailable"

    latest_season = sorted(games["SEASON"].dropna().unique())[-1]
    team_games = games[
        games["SEASON"].eq(latest_season)
        & games["TEAM_NAME"].eq(team_name)
    ]

    if team_games.empty:
        return "Record unavailable"

    wins = int(team_games["WL"].eq("W").sum())
    losses = int(team_games["WL"].eq("L").sum())
    return f"{wins}-{losses}"


def normalize_live_team_name(team_data: dict) -> str:
    """Build a readable team name from a live scoreboard team payload."""
    team_id = team_data.get("teamId")

    if team_id:
        mapped_team = get_team_name_by_id(team_id)

        if mapped_team:
            return mapped_team

    city = str(team_data.get("teamCity", "")).strip()
    name = str(team_data.get("teamName", "")).strip()
    combined_name = f"{city} {name}".strip()

    return combined_name or str(team_data.get("teamTricode", "Unknown"))


def normalize_score_value(value: object) -> object:
    """Return a score value unless the feed is sending an empty/NaN placeholder."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else value


def coerce_game_status_code(value: object) -> int:
    """Coerce NBA status codes where 1=scheduled, 2=live, 3=final."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def normalize_score_pair(
    home_score: object,
    away_score: object,
    status_code: object,
) -> tuple[object, object]:
    """Normalize score values, hiding placeholder 0-0 before tipoff."""
    home_score = normalize_score_value(home_score)
    away_score = normalize_score_value(away_score)

    if coerce_game_status_code(status_code) <= 1:
        home_text = str(home_score).strip()
        away_text = str(away_score).strip()

        if home_text in {"", "0", "0.0"} and away_text in {"", "0", "0.0"}:
            return "", ""

    return home_score, away_score


def get_espn_competitor_team_name(competitor: dict) -> str | None:
    """Map an ESPN scoreboard competitor into a local team name."""
    team = competitor.get("team", {}) or {}
    abbreviation = str(team.get("abbreviation", "")).strip()

    if abbreviation:
        mapped_team = get_team_name_by_abbreviation(abbreviation)

        if mapped_team:
            return mapped_team

    return str(team.get("displayName") or team.get("name") or "").strip() or None


def coerce_espn_game_status_code(status_payload: dict) -> int:
    """Map ESPN game status into NBA-style status codes."""
    status_type = status_payload.get("type", {}) or {}
    state = str(status_type.get("state", "")).strip().lower()

    if bool(status_type.get("completed")) or state == "post":
        return 3
    if state == "in":
        return 2
    return 1


@st.cache_data(ttl=1)
def load_espn_scoreboard_games(game_date: str | None = None) -> pd.DataFrame:
    """Fetch ESPN's public scoreboard as the fastest live score source."""
    if game_date is None:
        game_date = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y%m%d")

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?"
        + urlencode({"dates": game_date})
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )

    try:
        with urlopen(request, timeout=LIVE_SCOREBOARD_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for event in data.get("events", []) or []:
        competitions = event.get("competitions", []) or []
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", []) or []
        home_competitor = next(
            (
                competitor
                for competitor in competitors
                if str(competitor.get("homeAway", "")).lower() == "home"
            ),
            None,
        )
        away_competitor = next(
            (
                competitor
                for competitor in competitors
                if str(competitor.get("homeAway", "")).lower() == "away"
            ),
            None,
        )

        if home_competitor is None or away_competitor is None:
            continue

        home_team = get_espn_competitor_team_name(home_competitor)
        away_team = get_espn_competitor_team_name(away_competitor)

        if not home_team or not away_team:
            continue

        status_payload = competition.get("status", {}) or {}
        status_type = status_payload.get("type", {}) or {}
        status_code = coerce_espn_game_status_code(status_payload)
        status_text = (
            str(status_type.get("shortDetail", "")).strip()
            or str(status_type.get("detail", "")).strip()
            or str(status_type.get("description", "")).strip()
        )
        home_score, away_score = normalize_score_pair(
            home_competitor.get("score", ""),
            away_competitor.get("score", ""),
            status_code,
        )
        game_datetime = pd.to_datetime(
            competition.get("date") or event.get("date"),
            errors="coerce",
            utc=True,
        )

        rows.append(
            {
                "Game ID": f"ESPN_{event.get('id', '')}",
                "ESPN Game ID": str(event.get("id", "")),
                "Game Date": game_date,
                "Game DateTime": game_datetime,
                "Game Time": status_text,
                "Status": "Scheduled" if status_code <= 1 else status_text,
                "Game Status Code": status_code,
                "Period": status_payload.get("period", 0),
                "Game Clock": status_payload.get("displayClock", ""),
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": home_score,
                "Away Score": away_score,
                "Source": "ESPN scoreboard",
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=1)
def load_espn_latest_play(espn_game_id: str) -> dict[str, object] | None:
    """Fetch the most recent ESPN play for one game."""
    espn_game_id = str(espn_game_id).strip()

    if not espn_game_id:
        return None

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?"
        + urlencode({"event": espn_game_id})
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )

    try:
        with urlopen(request, timeout=LIVE_SCOREBOARD_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    plays = [
        play
        for play in data.get("plays", []) or []
        if str(play.get("text", "")).strip()
    ]

    if not plays:
        return None

    def sort_key(play: dict) -> int:
        try:
            return int(play.get("sequenceNumber") or play.get("id") or 0)
        except (TypeError, ValueError):
            return 0

    latest_play = sorted(plays, key=sort_key)[-1]
    period = latest_play.get("period", {}) or {}
    clock = latest_play.get("clock", {}) or {}

    return {
        "text": str(latest_play.get("text", "")).strip(),
        "period": period.get("number", ""),
        "period_label": str(period.get("displayValue", "")).strip(),
        "clock": str(clock.get("displayValue", "")).strip(),
        "home_score": latest_play.get("homeScore", ""),
        "away_score": latest_play.get("awayScore", ""),
    }


def format_latest_play_line(play: dict[str, object] | None) -> str:
    """Format the latest live play into one compact card line."""
    if not play:
        return ""

    text = str(play.get("text", "")).strip()

    if not text:
        return ""

    period_label = format_live_period_label(play.get("period"))
    clock = str(play.get("clock", "")).strip()
    prefix = " ".join(part for part in [period_label, clock] if part)

    return f"{prefix} - {text}" if prefix else text


@st.cache_data(ttl=1)
def load_live_scoreboard_games() -> pd.DataFrame:
    """Fetch free live NBA scoreboard games when internet access is available."""
    try:
        from nba_api.live.nba.endpoints.scoreboard import ScoreBoard

        board = ScoreBoard(timeout=LIVE_SCOREBOARD_TIMEOUT_SECONDS)
        games = board.get_dict().get("scoreboard", {}).get("games", [])
    except Exception:
        return pd.DataFrame()

    rows = []

    for game in games:
        home_team_data = game.get("homeTeam", {}) or {}
        away_team_data = game.get("awayTeam", {}) or {}
        home_team = normalize_live_team_name(home_team_data)
        away_team = normalize_live_team_name(away_team_data)

        if not home_team or not away_team:
            continue

        status_code = game.get("gameStatus", 0)
        home_score, away_score = normalize_score_pair(
            home_team_data.get("score", ""),
            away_team_data.get("score", ""),
            status_code,
        )

        rows.append(
            {
                "Game ID": str(game.get("gameId", "")),
                "Game Date": game.get("gameDateEst")
                or game.get("gameTimeUTC")
                or game.get("gameEt")
                or "",
                "Game DateTime": pd.to_datetime(
                    game.get("gameTimeUTC"),
                    errors="coerce",
                    utc=True,
                ),
                "Game Time": game.get("gameTimeLTZ")
                or game.get("gameEt")
                or game.get("gameStatusText", ""),
                "Status": game.get("gameStatusText", "Scheduled"),
                "Game Status Code": status_code,
                "Period": game.get("period", 0),
                "Game Clock": game.get("gameClock", ""),
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": home_score,
                "Away Score": away_score,
                "Source": "NBA live scoreboard",
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=1)
def load_scoreboard_v3_games(game_date: str | None = None) -> pd.DataFrame:
    """Fetch the daily scoreboard v3 feed as a fallback live source."""
    if game_date is None:
        game_date = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    try:
        from nba_api.stats.endpoints import ScoreboardV3

        board = ScoreboardV3(game_date=game_date, timeout=LIVE_SCOREBOARD_TIMEOUT_SECONDS)
        games = board.get_dict().get("scoreboard", {}).get("games", []) or []
    except Exception:
        return pd.DataFrame()

    if not games:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for game in games:
        game_id = str(game.get("gameId", "")).strip()
        if not game_id:
            continue

        home_team_data = game.get("homeTeam", {}) or {}
        away_team_data = game.get("awayTeam", {}) or {}
        home_team = get_schedule_team_name(
            home_team_data.get("teamId"),
            home_team_data.get("teamTricode"),
            home_team_data.get("teamCity"),
            home_team_data.get("teamName"),
        )
        away_team = get_schedule_team_name(
            away_team_data.get("teamId"),
            away_team_data.get("teamTricode"),
            away_team_data.get("teamCity"),
            away_team_data.get("teamName"),
        )

        if not home_team or not away_team:
            continue

        status_code = game.get("gameStatus", 0)
        home_score, away_score = normalize_score_pair(
            home_team_data.get("score", ""),
            away_team_data.get("score", ""),
            status_code,
        )

        rows.append(
            {
                "Game ID": game_id,
                "Game Date": game_date,
                "Game DateTime": pd.to_datetime(game.get("gameTimeUTC"), errors="coerce", utc=True),
                "Game Time": str(game.get("gameStatusText", "")).strip(),
                "Status": str(game.get("gameStatusText", "")).strip(),
                "Game Status Code": status_code,
                "Period": game.get("period", 0),
                "Game Clock": game.get("gameClock", ""),
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": home_score,
                "Away Score": away_score,
                "Source": "NBA scoreboard v3",
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=1)
def load_live_playbyplay_state(game_id: str) -> dict[str, object] | None:
    """Fetch the latest live play-by-play snapshot for one game."""
    game_id = str(game_id).strip()
    if not game_id:
        return None

    try:
        from nba_api.live.nba.endpoints.playbyplay import PlayByPlay

        board = PlayByPlay(game_id, timeout=LIVE_SCOREBOARD_TIMEOUT_SECONDS)
        actions = board.get_dict().get("game", {}).get("actions", []) or []
    except Exception:
        return None

    if not actions:
        return None

    actions_df = pd.DataFrame(actions)
    if actions_df.empty:
        return None

    if "actionNumber" in actions_df.columns:
        actions_df = actions_df.sort_values("actionNumber")
    elif "orderNumber" in actions_df.columns:
        actions_df = actions_df.sort_values("orderNumber")

    latest = actions_df.iloc[-1]
    home_score = latest.get("scoreHome", "")
    away_score = latest.get("scoreAway", "")
    period = latest.get("period", 0)
    clock = latest.get("clock", "")

    status = ""
    if clock and format_live_clock(clock):
        try:
            period_num = int(float(period))
        except (TypeError, ValueError):
            period_num = 0
        if period_num > 0:
            status = build_live_detail_label("", period_num, clock, fallback="")

    return {
        "Game ID": game_id,
        "Status": status,
        "Period": period,
        "Game Clock": clock,
        "Home Score": home_score,
        "Away Score": away_score,
        "Source": "NBA live play-by-play",
    }


@st.cache_data(ttl=1)
def resolve_live_today_snapshot(game_id: str, home_team: str, away_team: str) -> dict[str, object] | None:
    """Return the freshest live row for a matchup if one exists."""
    game_id = str(game_id).strip()
    home_team = str(home_team).strip().lower()
    away_team = str(away_team).strip().lower()

    def iter_candidate_rows() -> list[pd.Series]:
        frames = [
            load_espn_scoreboard_games(),
            load_scoreboard_v3_games(),
            load_live_scoreboard_games(),
        ]
        candidates: list[pd.Series] = []

        for frame in frames:
            if frame.empty:
                continue

            by_game_id = pd.DataFrame()
            if game_id and "Game ID" in frame.columns:
                by_game_id = frame[frame["Game ID"].astype(str).eq(game_id)]
                if not by_game_id.empty:
                    candidates.extend(row for _, row in by_game_id.iterrows())

            if candidates:
                continue

            if {"Home Team", "Away Team"}.issubset(frame.columns) and home_team and away_team:
                matchup_rows = frame[
                    frame["Home Team"].astype(str).str.lower().eq(home_team)
                    & frame["Away Team"].astype(str).str.lower().eq(away_team)
                ]
                if not matchup_rows.empty:
                    candidates.extend(row for _, row in matchup_rows.iterrows())

        return candidates

    for candidate in iter_candidate_rows():
        status = str(candidate.get("Status", "")).strip().lower()
        status_code = coerce_game_status_code(candidate.get("Game Status Code", 0))
        score_present = (
            bool(str(candidate.get("Home Score", "")).strip())
            or bool(str(candidate.get("Away Score", "")).strip())
        )
        clock_present = bool(format_live_clock(candidate.get("Game Clock")))
        period_present = bool(format_live_period_label(candidate.get("Period")))

        if (
            status_code > 1
            or "final" in status
            or (score_present and status_code != 1)
            or ((clock_present or period_present) and status_code != 1)
        ):
            return candidate.to_dict()

    return None


@st.cache_data
def load_injury_report_schedule_games() -> pd.DataFrame:
    """Build a free current-game schedule fallback from the injury report file."""
    injuries = load_current_injuries()

    if injuries.empty or "MATCHUP" not in injuries.columns:
        return pd.DataFrame()

    rows = []
    schedule = injuries[
        ["GAME_DATE", "GAME_TIME", "MATCHUP"]
    ].drop_duplicates().reset_index(drop=True)

    for _, game in schedule.iterrows():
        matchup = str(game["MATCHUP"])

        if "@" not in matchup:
            continue

        away_abbr, home_abbr = matchup.split("@", 1)
        away_team = get_team_name_by_abbreviation(away_abbr.strip())
        home_team = get_team_name_by_abbreviation(home_abbr.strip())

        if not home_team or not away_team:
            continue

        rows.append(
            {
                "Game ID": f"{away_abbr.strip()}_{home_abbr.strip()}_{game['GAME_DATE']}",
                "Game Date": str(game["GAME_DATE"]),
                "Game Time": str(game["GAME_TIME"]),
                "Game DateTime": pd.to_datetime(
                    f"{game['GAME_DATE']} {game['GAME_TIME']}",
                    errors="coerce",
                ),
                "Status": "Scheduled",
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": "",
                "Away Score": "",
                "Source": "Injury report schedule",
            }
        )

    return pd.DataFrame(rows)


def filter_games_to_today(games: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose game date is today in Eastern time."""
    if games.empty or "Game Date" not in games.columns:
        return games

    today_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    game_dates = pd.to_datetime(games["Game Date"], errors="coerce")
    return games[game_dates.dt.date.eq(today_et)].copy().reset_index(drop=True)


def build_playoff_series_key(home_team: object, away_team: object) -> str:
    """Build the same unordered matchup key used by playoff series results."""
    teams = sorted([str(home_team).strip(), str(away_team).strip()])
    return " vs ".join(teams)


def load_completed_playoff_series_lookup(
    season: str | None = None,
) -> dict[str, dict[str, object]]:
    """Return completed playoff series keyed by unordered team matchup."""
    try:
        results = build_completed_playoff_game_results(season)
    except Exception:
        return {}

    if results.empty or "Series Key" not in results.columns:
        return {}

    completed_series: dict[str, dict[str, object]] = {}

    for series_key, games in results.groupby("Series Key", sort=False):
        wins = games["Winner"].astype(str).value_counts()

        if wins.empty or int(wins.iloc[0]) < 4:
            continue

        last_game_date = pd.to_datetime(games["GAME_DATE"], errors="coerce").max()
        completed_series[str(series_key)] = {
            "winner": str(wins.index[0]),
            "wins": int(wins.iloc[0]),
            "games_played": int(len(games)),
            "last_game_date": last_game_date,
        }

    return completed_series


def filter_schedule_playoff_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that look like playoff schedule games."""
    if games.empty:
        return games

    games = filter_games_to_known_teams(games)

    if games.empty:
        return games

    masks = []

    if "Series Game Number" in games.columns:
        masks.append(games["Series Game Number"].notna())

    if "Game ID" in games.columns:
        game_ids = games["Game ID"].astype(str).str.strip()
        masks.append(game_ids.str.startswith("004") | game_ids.str.startswith("4"))

    label_columns = [
        column
        for column in ["Game Label", "Game SubLabel"]
        if column in games.columns
    ]

    for column in label_columns:
        masks.append(
            games[column]
            .astype(str)
            .str.contains(r"\bGame\s+\d+\b", case=False, na=False)
        )

    if not masks:
        return games

    combined_mask = masks[0].copy()

    for mask in masks[1:]:
        combined_mask = combined_mask | mask

    return games[combined_mask].copy().reset_index(drop=True)


def build_completed_series_lookup_from_schedule(games: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Infer completed playoff series from final score rows in a schedule frame."""
    required_columns = {"Home Team", "Away Team", "Home Score", "Away Score"}

    if games.empty or not required_columns.issubset(games.columns):
        return {}

    games = filter_schedule_playoff_rows(games)

    if games.empty:
        return {}

    rows = []

    for _, row in games.iterrows():
        status = str(row.get("Status", "")).strip().lower()
        status_code = coerce_game_status_code(row.get("Game Status Code", 0))

        if status_code < 3 and "final" not in status:
            continue

        home_team = str(row.get("Home Team", "")).strip()
        away_team = str(row.get("Away Team", "")).strip()
        home_score = normalize_score_value(row.get("Home Score", ""))
        away_score = normalize_score_value(row.get("Away Score", ""))

        if not home_team or not away_team or home_score == "" or away_score == "":
            continue

        if not is_known_team_name(home_team) or not is_known_team_name(away_team):
            continue

        try:
            home_points = int(float(home_score))
            away_points = int(float(away_score))
        except (TypeError, ValueError):
            continue

        if home_points == away_points:
            continue

        rows.append(
            {
                "Series Key": build_playoff_series_key(home_team, away_team),
                "Winner": home_team if home_points > away_points else away_team,
                "GAME_DATE": row.get("Game Date"),
            }
        )

    if not rows:
        return {}

    results = pd.DataFrame(rows)
    completed_series: dict[str, dict[str, object]] = {}

    for series_key, series_games in results.groupby("Series Key", sort=False):
        wins = series_games["Winner"].astype(str).value_counts()

        if wins.empty or int(wins.iloc[0]) < 4:
            continue

        completed_series[str(series_key)] = {
            "winner": str(wins.index[0]),
            "wins": int(wins.iloc[0]),
            "games_played": int(len(series_games)),
            "last_game_date": pd.to_datetime(
                series_games["GAME_DATE"],
                errors="coerce",
            ).max(),
        }

    return completed_series


def is_scheduled_completed_series_row(
    row: pd.Series,
    completed_series: dict[str, dict[str, object]],
) -> bool:
    """Return whether a scheduled row belongs to a series already won."""
    if not completed_series:
        return False

    series_key = build_playoff_series_key(row.get("Home Team", ""), row.get("Away Team", ""))
    series_state = completed_series.get(series_key)

    if not series_state:
        return False

    status = str(row.get("Status", "")).strip().lower()
    status_code = coerce_game_status_code(row.get("Game Status Code", 0))

    if status_code > 1 or "final" in status or "live" in status:
        return False

    score_present = (
        bool(str(normalize_score_value(row.get("Home Score", ""))).strip())
        or bool(str(normalize_score_value(row.get("Away Score", ""))).strip())
    )

    if score_present:
        return False

    return True


def filter_completed_series_scheduled_games(
    games: pd.DataFrame,
    season: str | None = None,
) -> pd.DataFrame:
    """Drop scheduled-only games for playoff series that are already complete."""
    if games.empty or not {"Home Team", "Away Team"}.issubset(games.columns):
        return games

    completed_series = load_completed_playoff_series_lookup(season)
    completed_series.update(build_completed_series_lookup_from_schedule(games))

    if not completed_series:
        return games

    keep_mask = ~games.apply(
        lambda row: is_scheduled_completed_series_row(row, completed_series),
        axis=1,
    )
    return games[keep_mask].copy().reset_index(drop=True)


def normalize_schedule_games_for_display(games: pd.DataFrame) -> pd.DataFrame:
    """Normalize schedule rows into the same display schema as live games."""
    if games.empty:
        return games

    games = games.copy()
    games = filter_games_to_known_teams(games)

    if games.empty:
        return games

    games["Game Date"] = pd.to_datetime(games["Game Date"], errors="coerce")

    if "Game DateTime" in games.columns:
        games["Game DateTime"] = games["Game DateTime"].apply(coerce_game_datetime)
    else:
        games["Game DateTime"] = pd.NaT

    games["Source"] = "NBA schedule"

    def normalize_schedule_status(row: pd.Series) -> pd.Series:
        status = str(row.get("Status", "")).strip()
        tipoff = format_tipoff_label(row.get("Game DateTime"))
        status_code = coerce_game_status_code(row.get("Game Status Code"))

        if status_code <= 1:
            row["Game Time"] = tipoff or str(row.get("Game Time", "")).strip()
            row["Status"] = "Scheduled"
        else:
            row["Game Time"] = status or tipoff
            row["Status"] = status or ("Final" if status_code >= 3 else "Live")

        return row

    return games.apply(normalize_schedule_status, axis=1)


def select_next_upcoming_slate(games: pd.DataFrame) -> pd.DataFrame:
    """Return all games on the nearest future date with scheduled games."""
    if games.empty or "Game Date" not in games.columns:
        return pd.DataFrame()

    games = games.copy()
    game_dates = pd.to_datetime(games["Game Date"], errors="coerce")
    today_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    future_mask = game_dates.dt.date.gt(today_et)

    if not future_mask.any():
        return pd.DataFrame()

    next_game_date = game_dates[future_mask].dt.date.min()
    return games[game_dates.dt.date.eq(next_game_date)].copy().reset_index(drop=True)


@st.cache_data(ttl=60)
def load_next_upcoming_games() -> pd.DataFrame:
    """Return the next future NBA slate when there are no games today."""
    try:
        season = get_latest_playoff_season(load_raw_games())
        schedule_games = load_nba_schedule_games(season)
    except Exception:
        return pd.DataFrame()

    if schedule_games.empty:
        return pd.DataFrame()

    schedule_games = normalize_schedule_games_for_display(schedule_games)
    schedule_games = filter_completed_series_scheduled_games(
        schedule_games,
        season=season,
    )
    upcoming_games = select_next_upcoming_slate(schedule_games)

    return upcoming_games.sort_values(
        ["Game Date", "Game DateTime", "Game ID"],
        na_position="last",
    ).reset_index(drop=True)


def format_game_slate_label(games: pd.DataFrame) -> str:
    """Format the date label for a slate of games."""
    if games.empty or "Game Date" not in games.columns:
        return ""

    parsed = pd.to_datetime(games.iloc[0].get("Game Date"), errors="coerce")

    if pd.isna(parsed):
        return ""

    return f"{parsed.strftime('%A, %B')} {parsed.day}"


@st.cache_data(ttl=1)
def load_today_games() -> pd.DataFrame:
    """Return current NBA games using the live scoreboard with a local fallback."""
    espn_games = load_espn_scoreboard_games()
    live_games = load_live_scoreboard_games()
    v3_games = load_scoreboard_v3_games()

    try:
        schedule_games = load_nba_schedule_games()
    except Exception:
        schedule_games = pd.DataFrame()

    if not schedule_games.empty:
        today_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).normalize()
        schedule_games = normalize_schedule_games_for_display(schedule_games)

        schedule_games = schedule_games[
            schedule_games["Game Date"].dt.tz_localize(ZoneInfo("America/New_York"))
            .dt.normalize()
            .eq(today_et)
        ].copy()

        if not schedule_games.empty:
            schedule_games = filter_completed_series_scheduled_games(schedule_games)
    if schedule_games.empty and espn_games.empty and live_games.empty and v3_games.empty:
        fallback_games = filter_games_to_today(load_injury_report_schedule_games())
        return filter_completed_series_scheduled_games(fallback_games)

    if schedule_games.empty:
        if not espn_games.empty:
            return filter_completed_series_scheduled_games(espn_games)
        live_fallback = v3_games if not v3_games.empty else live_games
        return filter_completed_series_scheduled_games(live_fallback)

    def normalize_name(value: object) -> str:
        return str(value).strip().lower()

    def build_game_key(home_team: object, away_team: object) -> tuple[str, str]:
        return (normalize_name(home_team), normalize_name(away_team))

    merged_rows = []
    live_lookup = {
        build_game_key(row.get("Home Team"), row.get("Away Team")): row
        for _, row in live_games.iterrows()
    }
    espn_lookup = {
        build_game_key(row.get("Home Team"), row.get("Away Team")): row
        for _, row in espn_games.iterrows()
    }
    v3_lookup = {
        build_game_key(row.get("Home Team"), row.get("Away Team")): row
        for _, row in v3_games.iterrows()
    }

    for _, row in schedule_games.iterrows():
        merged = row.to_dict()
        key = build_game_key(row.get("Home Team"), row.get("Away Team"))
        live_row = espn_lookup.pop(key, None)
        if live_row is None:
            live_row = v3_lookup.pop(key, None)
            if live_row is None:
                live_row = live_lookup.pop(key, None)
            else:
                live_lookup.pop(key, None)
        else:
            v3_lookup.pop(key, None)
            live_lookup.pop(key, None)
        if live_row is not None:
            for column in live_row.index:
                if (
                    column == "Game ID"
                    and str(merged.get("Game ID", "")).strip()
                    and str(live_row.get("Source", "")) == "ESPN scoreboard"
                ):
                    continue
                merged[column] = live_row[column]

        if pd.isna(merged.get("Game DateTime")):
            merged["Game DateTime"] = row.get("Game DateTime")

        if live_row is not None:
            live_game_id = str(merged.get("Game ID", "")).strip()
            live_state = load_live_playbyplay_state(live_game_id)
            if live_state is not None:
                live_status = str(live_state.get("Status", "")).strip().lower()
                if live_status or live_state.get("Home Score") != "" or live_state.get("Away Score") != "":
                    existing_break_label = format_live_break_label(merged.get("Status", ""))
                    for column in ["Status", "Period", "Game Clock", "Home Score", "Away Score"]:
                        if column == "Status" and existing_break_label:
                            continue
                        value = live_state.get(column)
                        if value is not None and str(value).strip() != "":
                            merged[column] = value
                    merged["Source"] = str(live_state.get("Source", merged.get("Source", "NBA live scoreboard")))

        merged_rows.append(merged)

    for live_row in espn_lookup.values():
        merged_rows.append(live_row.to_dict())

    for live_row in live_lookup.values():
        merged_rows.append(live_row.to_dict())

    for live_row in v3_lookup.values():
        merged_rows.append(live_row.to_dict())

    return filter_completed_series_scheduled_games(pd.DataFrame(merged_rows))


def format_live_period_label(period: object) -> str:
    """Format a live game period into a compact basketball label."""
    try:
        value = int(float(period))
    except (TypeError, ValueError):
        return ""

    if value <= 0:
        return ""
    if value <= 4:
        return f"Q{value}"
    if value == 5:
        return "OT"
    return f"{value - 4}OT"


def format_live_clock(clock: object) -> str:
    """Format a live scoreboard clock into a readable label."""
    text = str(clock).strip()

    if not text:
        return ""

    match = re.fullmatch(r"PT(\d+)M(\d+(?:\.\d+)?)S", text)

    if not match:
        return text

    minutes = int(match.group(1))
    seconds = int(round(float(match.group(2))))
    return f"{minutes}:{seconds:02d}"


def format_live_break_label(status: object) -> str:
    """Return status labels that should outrank a zero game clock."""
    text = str(status).strip()
    status_lower = text.lower()

    if "halftime" in status_lower or "half time" in status_lower:
        return "Halftime"

    if status_lower in {"half", "the half"}:
        return "Halftime"

    if status_lower.startswith("end of"):
        return text

    return ""


def build_live_detail_label(
    status: object,
    period: object,
    clock: object,
    fallback: str = "Live",
) -> str:
    """Build the card's live detail text from status, period, and clock."""
    break_label = format_live_break_label(status)

    if break_label:
        return break_label

    period_label = format_live_period_label(period)
    clock_label = format_live_clock(clock)

    if period_label and clock_label:
        return f"{period_label} {clock_label}"
    if period_label:
        return period_label
    if clock_label:
        return clock_label

    return str(status).strip() or fallback


def coerce_game_datetime(value: object) -> pd.Timestamp:
    """Coerce a schedule timestamp to a timezone-aware timestamp when possible."""
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return pd.NaT

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize(ZoneInfo("America/New_York"))

    return parsed


def format_countdown_delta(delta: pd.Timedelta | object) -> str:
    """Format the remaining time until tipoff."""
    if delta is None or pd.isna(delta):
        return ""

    total_seconds = max(0, int(getattr(delta, "total_seconds", lambda: 0)()))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60

    parts = []

    if days > 0:
        parts.append(f"{days}d")

    if hours > 0:
        parts.append(f"{hours}h")

    if days == 0 and minutes > 0:
        parts.append(f"{minutes}m")

    if not parts:
        parts.append("0m")

    return " ".join(parts[:2])


def format_tipoff_label(game_dt: pd.Timestamp | object) -> str:
    """Format a game datetime as a compact Eastern Time tipoff label."""
    if game_dt is None or pd.isna(game_dt):
        return ""

    parsed = pd.to_datetime(game_dt, errors="coerce")

    if pd.isna(parsed):
        return ""

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize(ZoneInfo("America/New_York"))

    eastern = parsed.tz_convert(ZoneInfo("America/New_York"))
    return eastern.strftime("%b %d, %I:%M %p ET").replace(" 0", " ")


def build_today_game_timing_state(game: pd.Series) -> dict[str, str | bool]:
    """Build a compact timing state for a current game card."""
    status = str(game.get("Status", "")).strip()
    source = str(game.get("Source", ""))
    game_dt = coerce_game_datetime(game.get("Game DateTime"))
    game_time = str(game.get("Game Time", "")).strip()
    home_score = str(game.get("Home Score", "")).strip()
    away_score = str(game.get("Away Score", "")).strip()
    game_clock = format_live_clock(game.get("Game Clock"))
    period_label = format_live_period_label(game.get("Period"))
    live_state, is_live = build_live_game_state(game)
    game_id = str(game.get("Game ID", "")).strip()
    home_team = str(game.get("Home Team", "")).strip()
    away_team = str(game.get("Away Team", "")).strip()

    if is_live:
        return {
            "mode": "live",
            "badge": "LIVE",
            "detail": live_state,
            "status": status or "Live",
            "refresh": "15",
        }

    if "final" in status.lower():
        return {
            "mode": "final",
            "badge": "FINAL",
            "detail": "",
            "status": "Final",
            "refresh": "0",
        }

    should_probe_live = False
    if not pd.isna(game_dt):
        now = pd.Timestamp.now(tz=game_dt.tzinfo)
        should_probe_live = now >= (game_dt - pd.Timedelta(minutes=2))
    if (
        source in {"ESPN scoreboard", "NBA live scoreboard", "NBA live boxscore", "NBA live play-by-play", "NBA scoreboard v3"}
        and coerce_game_status_code(game.get("Game Status Code", 0)) > 1
    ):
        should_probe_live = True

    if should_probe_live:
        live_snapshot = resolve_live_today_snapshot(game_id, home_team, away_team)
        if live_snapshot is not None:
            snapshot_status = str(live_snapshot.get("Status", "")).strip()
            snapshot_detail = build_live_detail_label(
                snapshot_status,
                live_snapshot.get("Period"),
                live_snapshot.get("Game Clock"),
            )

            if "final" in snapshot_status.lower():
                return {
                    "mode": "final",
                    "badge": "FINAL",
                    "detail": "",
                    "status": "Final",
                    "refresh": "0",
                }

            if snapshot_detail or str(live_snapshot.get("Home Score", "")).strip() or str(live_snapshot.get("Away Score", "")).strip():
                return {
                    "mode": "live",
                    "badge": "LIVE",
                    "detail": snapshot_detail,
                    "status": snapshot_status or "Live",
                    "refresh": "15",
                }

    if not pd.isna(game_dt):
        now = pd.Timestamp.now(tz=game_dt.tzinfo)
        remaining = game_dt - now

        if remaining.total_seconds() > 0:
            return {
                "mode": "countdown",
                "badge": f"Starts in {format_countdown_delta(remaining)}",
                "detail": f"Tipoff {format_tipoff_label(game_dt)}",
                "status": status or "Scheduled",
                "refresh": "0",
                "target_iso": game_dt.isoformat(),
            }

        if coerce_game_status_code(game.get("Game Status Code", 0)) <= 1:
            return {
                "mode": "starting",
                "badge": "Starting soon",
                "detail": f"Tipoff {format_tipoff_label(game_dt)}",
                "status": "Starting soon",
                "refresh": "15",
                "target_iso": game_dt.isoformat(),
            }

    if game_time and source not in {"ESPN scoreboard", "NBA live scoreboard", "NBA live boxscore", "NBA live play-by-play", "NBA scoreboard v3"}:
        return {
            "mode": "starting",
            "badge": "Starting soon",
            "detail": f"Tipoff {game_time}",
            "status": status or "Starting",
            "refresh": "0",
        }

    return {
        "mode": "scheduled",
        "badge": status or "Scheduled",
        "detail": "",
        "status": status or "Scheduled",
        "refresh": "0",
    }


def build_live_game_state(game: pd.Series) -> tuple[str, bool]:
    """Return a compact live state label and whether the game is live."""
    status = str(game.get("Status", "")).strip()
    source = str(game.get("Source", ""))
    game_clock = format_live_clock(game.get("Game Clock"))
    period_label = format_live_period_label(game.get("Period"))
    live_detail = build_live_detail_label(
        status,
        game.get("Period"),
        game.get("Game Clock"),
    )
    status_code_value = coerce_game_status_code(game.get("Game Status Code", 0))
    home_score = str(game.get("Home Score", "")).strip()
    away_score = str(game.get("Away Score", "")).strip()
    status_lower = status.lower()
    in_progress_hint = any(
        token in status_lower
        for token in [
            "live",
            "in progress",
            "q1",
            "q2",
            "q3",
            "q4",
            "1st qtr",
            "2nd qtr",
            "3rd qtr",
            "4th qtr",
            "qtr",
            "end of",
            "halftime",
            "half",
            "ot",
        ]
    )

    if status_code_value >= 3 or "final" in status_lower:
        return "Final", False

    if status_code_value == 2:
        return live_detail, True

    if source == "NBA scoreboard v3":
        if status_code_value <= 1 and not period_label and not game_clock:
            return status or "Scheduled", False

        if live_detail:
            return live_detail, True

        if status_code_value > 1:
            return status or "Live", True
        return status or "Scheduled", False

    if "scheduled" in status_lower or "pregame" in status_lower:
        return status or "Scheduled", False

    if not (period_label or game_clock or in_progress_hint):
        return status or "Scheduled", False

    if live_detail:
        return live_detail, True
    if in_progress_hint:
        return status or "Live", True

    return status or "Scheduled", False


def render_live_refresh_script(enabled: bool, interval_ms: int = 15000) -> None:
    """Auto-refresh the page while a live scoreboard is available."""
    if not enabled:
        return

    st.components.v1.html(
        f"""
        <script>
            setTimeout(() => window.parent.location.reload(), {interval_ms});
        </script>
        """,
        height=0,
    )


def get_today_refresh_interval(games: pd.DataFrame) -> int | None:
    """Return a refresh cadence for live current games only."""
    if games.empty:
        return None

    has_live = False

    for _, game in games.iterrows():
        timing = build_today_game_timing_state(game)

        if timing["mode"] == "live":
            has_live = True

    if has_live:
        return 15000

    return None


@st.fragment(run_every=1)
def render_today_games_live_fragment(compact: bool = False, show_schedule_rows: bool = False) -> None:
    """Render today games in a fragment that updates countdowns without rerunning the page."""
    games = load_today_games()
    is_upcoming_slate = False

    if games.empty:
        games = load_next_upcoming_games()
        is_upcoming_slate = True

        if games.empty:
            st.info(
                "No current or upcoming games were found in the free schedule feeds."
            )
            return

    predictions = build_today_game_predictions(games)
    source = str(games.iloc[0].get("Source", "Current games"))

    if is_upcoming_slate:
        slate_label = format_game_slate_label(games)
        st.info("No NBA games are scheduled for today.")
        render_section_kicker("Upcoming Games", slate_label or None)
        st.caption(f"{len(games)} upcoming game(s) loaded from {source}.")
    else:
        st.caption(f"{len(games)} game(s) loaded from {source}.")

    if not compact:
        render_upset_alert_board(predictions)

    render_today_games_cards(predictions, compact=compact)

    if show_schedule_rows:
        expander_label = "Upcoming schedule rows" if is_upcoming_slate else "Schedule rows"
        with st.expander(expander_label, expanded=False):
            display_columns = [
                "Game Date",
                "Game Time",
                "Status",
                "Away Team",
                "Away Score",
                "Home Team",
                "Home Score",
                "Predicted Winner",
                "Winner Probability",
                "Home Win Probability",
                "Away Win Probability",
                "Source",
            ]
            display = predictions[
                [column for column in display_columns if column in predictions.columns]
            ].copy()

            for column in [
                "Winner Probability",
                "Home Win Probability",
                "Away Win Probability",
            ]:
                if column in display.columns:
                    display[column] = display[column].map("{:.1%}".format)

            st.dataframe(display, width="stretch", hide_index=True)


def parse_series_game_number(value: object) -> int | None:
    """Parse NBA schedule labels like 'Game 6' into an integer."""
    match = re.search(r"Game\s+(\d+)", str(value), flags=re.IGNORECASE)

    if not match:
        return None

    return int(match.group(1))


def parse_bool_value(value: object) -> bool:
    """Parse NBA schedule boolean-like values."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_schedule_team_name(team_id: object, abbreviation: object, city: object, name: object) -> str | None:
    """Map NBA schedule team fields into local team names."""
    if team_id is not None and not pd.isna(team_id):
        mapped_team = get_team_name_by_id(str(int(float(team_id))))

        if mapped_team:
            return mapped_team

    if abbreviation is not None and not pd.isna(abbreviation):
        mapped_team = get_team_name_by_abbreviation(str(abbreviation))

        if mapped_team:
            return mapped_team

    if city is not None and name is not None and not pd.isna(city) and not pd.isna(name):
        combined_name = f"{str(city).strip()} {str(name).strip()}".strip()

        if combined_name and combined_name.lower() != "nan nan":
            return combined_name

    return None


@st.cache_data(ttl=60)
def load_nba_schedule_games(season: str | None = None) -> pd.DataFrame:
    """Fetch free NBA schedule rows with a graceful offline fallback."""
    if season is None:
        try:
            season = get_latest_playoff_season(load_raw_games())
        except Exception:
            season = None

    if season is None:
        return pd.DataFrame()

    try:
        from nba_api.stats.endpoints.scheduleleaguev2 import ScheduleLeagueV2

        schedule = ScheduleLeagueV2(
            season=season,
            timeout=LIVE_SCOREBOARD_TIMEOUT_SECONDS,
        ).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

    rows = []

    for _, game in schedule.iterrows():
        home_team = get_schedule_team_name(
            game.get("homeTeam_teamId"),
            game.get("homeTeam_teamTricode"),
            game.get("homeTeam_teamCity"),
            game.get("homeTeam_teamName"),
        )
        away_team = get_schedule_team_name(
            game.get("awayTeam_teamId"),
            game.get("awayTeam_teamTricode"),
            game.get("awayTeam_teamCity"),
            game.get("awayTeam_teamName"),
        )

        if not home_team or not away_team:
            continue

        if not is_known_team_name(home_team) or not is_known_team_name(away_team):
            continue

        game_date = pd.to_datetime(game.get("gameDate"), errors="coerce")
        game_datetime = pd.to_datetime(
            game.get("gameDateTimeUTC") or game.get("gameDateTimeEst"),
            errors="coerce",
            utc=True,
        )
        status_code = game.get("gameStatus", 0)
        home_score, away_score = normalize_score_pair(
            game.get("homeTeam_score", ""),
            game.get("awayTeam_score", ""),
            status_code,
        )

        rows.append(
            {
                "Game ID": str(game.get("gameId", "")),
                "Game Date": game_date,
                "Game DateTime": game_datetime,
                "Game Time": str(game.get("gameStatusText", "")).strip(),
                "Status": str(game.get("gameStatusText", "")).strip(),
                "Game Status Code": status_code,
                "Period": game.get("period", 0),
                "Game Clock": game.get("gameClock", ""),
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": home_score,
                "Away Score": away_score,
                "Series Game Number": parse_series_game_number(
                    game.get("seriesGameNumber")
                )
                or parse_series_game_number(game.get("gameSubLabel")),
                "Game Label": str(game.get("gameLabel", "")).strip(),
                "Game SubLabel": str(game.get("gameSubLabel", "")).strip(),
                "If Necessary": parse_bool_value(game.get("ifNecessary")),
                "Source": "NBA schedule",
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Game Date", "Game DateTime", "Game ID"],
        na_position="last",
    ).reset_index(drop=True)


def format_status_timestamp(value: object) -> str:
    """Format a timestamp for compact status tiles."""
    if value is None or pd.isna(value):
        return "Unknown"

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return str(value)

    if parsed.hour == 0 and parsed.minute == 0:
        return parsed.date().isoformat()

    return parsed.strftime("%Y-%m-%d %H:%M")


def shorten_model_name(model_name: str) -> str:
    """Keep the model label readable in the app header."""
    return (
        model_name.replace("Calibrated ", "Cal. ")
        .replace("Random Forest", "RF")
        .replace("Logistic Regression", "Logit")
    )


def get_header_status_items() -> list[dict[str, str]]:
    """Collect compact app status values without failing the page."""
    latest_game_date = "Unknown"
    injury_report = "No report"
    active_series = "0"

    if TEAM_STRENGTH_PATH.exists():
        try:
            strength = load_team_strength()
            latest_game_date = format_status_timestamp(strength["GAME_DATE"].max())
        except Exception:
            latest_game_date = "Unavailable"

    if CURRENT_INJURIES_PATH.exists():
        try:
            injuries = load_current_injuries()

            if not injuries.empty and "REPORT_TIMESTAMP" in injuries.columns:
                injury_report = format_status_timestamp(
                    injuries.iloc[0].get("REPORT_TIMESTAMP")
                )
        except Exception:
            injury_report = "Unavailable"

    if RAW_GAMES_PATH.exists():
        try:
            series_states = build_current_playoff_series_states()
            active_series = str(
                sum(1 for state in series_states if not state["completed"])
            )
        except Exception:
            active_series = "Unavailable"

    return [
        {"label": "Data", "value": latest_game_date},
        {"label": "Injuries", "value": injury_report},
        {"label": "Series", "value": active_series},
    ]


def render_status_grid(items: list[dict[str, str]]) -> None:
    """Render compact native status metrics."""
    columns = st.columns(len(items), gap="small")

    for column, item in zip(columns, items):
        with column:
            st.metric(item["label"], item["value"])


def render_dashboard_cards(cards: list[dict[str, str]]) -> None:
    """Render compact dashboard stat cards."""
    card_html = []

    for card in cards:
        color = html.escape(card.get("color", "#2563eb"))
        card_html.append(
            f"""
            <div class="dashboard-card" style="--card-color: {color};">
                <div class="dashboard-label">{html.escape(card["label"])}</div>
                <div class="dashboard-value">{html.escape(card["value"])}</div>
                <div class="dashboard-note">{html.escape(card.get("note", ""))}</div>
            </div>
            """
        )

    st.html(f'<div class="dashboard-grid">{"".join(card_html)}</div>')


def render_team_mini_rows(rows: list[dict]) -> None:
    """Render small logo rows for rankings and summaries."""
    row_html = []

    for row in rows:
        team = str(row["team"])
        row_html.append(
            f"""
            <div class="team-mini-row">
                <img class="team-mini-logo" src="{html.escape(get_logo_url(team))}" alt="{html.escape(team)} logo">
                <div>
                    <div class="team-mini-name">{html.escape(team)}</div>
                    <div class="team-mini-meta">{html.escape(str(row.get("meta", "")))}</div>
                </div>
                <div class="team-mini-score">{html.escape(str(row.get("score", "")))}</div>
            </div>
            """
        )

    st.html("".join(row_html))


def render_section_kicker(label: str, note: str | None = None) -> None:
    """Render a compact section label with optional short note."""
    st.html(f'<div class="section-kicker">{html.escape(label)}</div>')

    if note:
        st.html(f'<div class="compact-note">{html.escape(note)}</div>')


def render_result_callout(label: str, title: str, meta: str) -> None:
    """Render the main prediction result as a compact callout."""
    st.html(
        f"""
        <div class="result-callout">
            <div class="result-label">{html.escape(label)}</div>
            <div class="result-title">{html.escape(title)}</div>
            <div class="result-meta">{html.escape(meta)}</div>
        </div>
        """
    )


def estimate_prediction_interval(
    probability: float,
    context: str = "game",
) -> tuple[float, float, float]:
    """Return a practical uncertainty interval around a displayed probability."""
    favorite_probability = max(probability, 1 - probability)
    closeness = 1 - min(abs(favorite_probability - 0.5) * 2, 1)
    base_margin = 0.045 if context == "game" else 0.055
    margin = base_margin + (0.045 * closeness)
    return (
        max(0.01, probability - margin),
        min(0.99, probability + margin),
        margin,
    )


def render_prediction_result_card(
    label: str,
    winner: str,
    probability: float,
    confidence: str,
    details: dict | None = None,
    context: str = "game",
    note: str | None = None,
) -> None:
    """Render a more visual prediction card with pricing signals."""
    lower, upper, _ = estimate_prediction_interval(probability, context=context)
    team_color = get_team_color(winner)
    logo_url = get_logo_url(winner)
    signals = [
        f"Range {lower:.0%}-{upper:.0%}",
        confidence,
    ]

    if details:
        model_probability = details.get("model_probability")
        elo_probability = details.get("elo_probability")
        blended_probability = details.get("blended_probability")

        if model_probability is not None:
            signals.append(f"Win {float(model_probability):.0%}")
        if elo_probability is not None:
            signals.append(f"Power {float(elo_probability):.0%}")
        if blended_probability is not None:
            signals.append(f"Pick {float(blended_probability):.0%}")

    signal_html = "".join(
        f'<span class="signal-pill">{html.escape(signal)}</span>'
        for signal in signals
    )
    note_html = ""

    if note:
        note_html = f'<div class="prediction-sub">{html.escape(note)}</div>'

    st.html(
        f"""
        <div class="prediction-card" style="--team-color: {html.escape(team_color)};">
            <div class="prediction-topline">
                <div class="dashboard-label">{html.escape(label)}</div>
                <span class="status-chip">{html.escape(confidence)}</span>
            </div>
            <div class="prediction-main">
                <img class="prediction-logo" src="{html.escape(logo_url)}" alt="{html.escape(winner)} logo">
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
        </div>
        """
    )


def render_team_logo_strip(teams: list[str]) -> None:
    """Render a compact strip of team logos."""
    tiles = []

    for team in teams:
        safe_team = html.escape(team)
        safe_logo = html.escape(get_logo_url(team))
        team_color = html.escape(get_team_color(team))
        tiles.append(
            f"""
            <div class="logo-tile" style="--team-color: {team_color};">
                <img src="{safe_logo}" alt="{safe_team} logo">
                <span>{safe_team}</span>
            </div>
            """
        )

    st.html(f'<div class="logo-strip">{"".join(tiles)}</div>')


def render_matchup_preview(
    team_a: str,
    team_b: str,
    label_a: str,
    label_b: str,
    center_label: str = "VS",
) -> None:
    """Render selected teams before a prediction is run."""
    team_a_logo = get_logo_url(team_a)
    team_b_logo = get_logo_url(team_b)
    team_a_color = get_team_color(team_a)
    team_b_color = get_team_color(team_b)
    team_a_abbr = get_team_abbreviation(team_a)
    team_b_abbr = get_team_abbreviation(team_b)

    st.html(
        f"""
        <div class="matchup-preview">
            <div class="preview-team-card" style="--team-color: {html.escape(team_a_color)};">
                <div class="preview-logo-row">
                    <img class="preview-logo" src="{html.escape(team_a_logo)}" alt="{html.escape(team_a)} logo">
                    <div>
                        <div class="preview-role">{html.escape(label_a)}</div>
                        <div class="preview-name">{html.escape(team_a)}</div>
                    </div>
                </div>
                <div class="preview-pill">{html.escape(team_a_abbr)}</div>
            </div>
            <div class="preview-center">{html.escape(center_label)}</div>
            <div class="preview-team-card" style="--team-color: {html.escape(team_b_color)};">
                <div class="preview-logo-row">
                    <img class="preview-logo" src="{html.escape(team_b_logo)}" alt="{html.escape(team_b)} logo">
                    <div>
                        <div class="preview-role">{html.escape(label_b)}</div>
                        <div class="preview-name">{html.escape(team_b)}</div>
                    </div>
                </div>
                <div class="preview-pill">{html.escape(team_b_abbr)}</div>
            </div>
        </div>
        """
    )


def render_app_header() -> None:
    """Render polished app header."""
    status = get_header_status_items()
    meta_text = " / ".join(
        f"{item['label']} {item['value']}" for item in status
    )
    st.html(
        f"""
        <div class="app-header">
            <div class="compact-header">
                <div class="app-name">NBA Predictor</div>
                <div class="header-meta">{html.escape(meta_text)}</div>
            </div>
        </div>
        """
    )


def render_series_score_card(
    state: dict,
    projection: str | None = None,
    next_game: str | None = None,
    remaining_games: list[dict[str, str]] | None = None,
) -> None:
    """Render a compact scorecard for a playoff series."""
    team_a = state["home_court_team"]
    team_b = state["other_team"]
    team_a_logo = get_logo_url(team_a)
    team_b_logo = get_logo_url(team_b)
    wins = state["wins"]
    completed_class = " score-card-complete" if state["completed"] else ""
    status = "Complete" if state["completed"] else f"Game {state['next_game_number']} next"
    summary_rows: list[str] = []
    remaining_rows = remaining_games or []
    momentum_rows: list[str] = []

    if projection:
        summary_rows.append(html.escape(projection))

    if next_game:
        summary_rows.append(html.escape(next_game))

    if state.get("current_streak", 0):
        streak_team = state.get("last_winner", "")
        if streak_team:
            momentum_rows.append(
                html.escape(
                    f"{streak_team} on a {int(state['current_streak'])}-game run"
                )
            )

    meta_sections = []

    if summary_rows:
        meta_sections.append(
            """
            <div class="score-meta-section">
                <div class="score-meta-section-title">Percentages</div>
                {summary_rows}
            </div>
            """.format(
                summary_rows="".join(
                    f'<span class="score-meta-line score-meta-summary">{line}</span>'
                    for line in summary_rows
                )
            )
        )

    if remaining_rows:
        meta_sections.append(
            """
            <div class="score-meta-section">
                <div class="score-meta-section-title">Remaining games</div>
                <div class="series-game-list">{game_rows}</div>
            </div>
            """.format(
                game_rows="".join(
                    """
                    <div class="series-game-row">
                        <span class="series-game-label">{label}</span>
                        <span class="series-game-detail">{detail}</span>
                    </div>
                    """.format(
                        label=html.escape(game["label"]),
                        detail=html.escape(game["detail"]),
                    )
                    for game in remaining_rows
                )
            )
        )

    if momentum_rows:
        meta_sections.append(
            """
            <div class="score-meta-section">
                <div class="score-meta-section-title">Momentum</div>
                {momentum_rows}
            </div>
            """.format(
                momentum_rows="".join(
                    f'<span class="score-meta-line score-meta-summary">{line}</span>'
                    for line in momentum_rows
                )
            )
        )

    meta_html = ""

    if meta_sections:
        meta_html = '<div class="score-meta">' + "".join(meta_sections) + "</div>"

    st.html(
        f"""
        <div class="score-card{completed_class}">
            <div class="score-topline">
                <span>{html.escape(state["series_key"])}</span>
                <span class="score-chip">{html.escape(status)}</span>
            </div>
            <div class="score-row">
                <span class="score-team-wrap">
                    <img class="score-logo" src="{html.escape(team_a_logo)}" alt="{html.escape(team_a)} logo">
                    <span class="score-team">{html.escape(team_a)}</span>
                </span>
                <span class="score-wins">{wins.get(team_a, 0)}</span>
            </div>
            <div class="score-row">
                <span class="score-team-wrap">
                    <img class="score-logo" src="{html.escape(team_b_logo)}" alt="{html.escape(team_b)} logo">
                    <span class="score-team">{html.escape(team_b)}</span>
                </span>
                <span class="score-wins">{wins.get(team_b, 0)}</span>
            </div>
            {meta_html}
        </div>
        """
    )



def render_team_card(
    team_name: str,
    probability: float,
    is_winner: bool = False,
    label: str = "Win probability",
) -> None:
    """Render a logo card for one team."""
    logo_url = get_logo_url(team_name)
    abbreviation = get_team_abbreviation(team_name)
    winner_class = "winner-card" if is_winner else ""
    badge_html = '<div class="winner-badge">Projected Winner</div>' if is_winner else ""
    team_color = get_team_color(team_name)

    safe_name = html.escape(team_name)
    safe_abbreviation = html.escape(abbreviation)
    safe_label = html.escape(label)
    safe_logo_url = html.escape(logo_url)

    card_html = f"""
<div class="team-card {winner_class}" style="--team-color: {team_color};">
    {badge_html}
    <img src="{safe_logo_url}" class="team-logo" alt="{safe_name} logo">
    <div class="team-name">{safe_name}</div>
    <div class="small-muted">{safe_abbreviation}</div>
    <div class="small-muted">{safe_label}</div>
    <div class="team-probability">{probability:.1%}</div>
</div>
"""
    st.html(card_html)


def render_matchup_cards(
    team_a: str,
    team_b: str,
    probability_a: float,
    probability_b: float,
    winner: str,
    label: str = "Win probability",
) -> None:
    """Render a compact matchup probability strip."""
    team_a_logo = get_logo_url(team_a)
    team_b_logo = get_logo_url(team_b)
    team_a_color = get_team_color(team_a)
    team_b_color = get_team_color(team_b)
    team_a_abbr = get_team_abbreviation(team_a)
    team_b_abbr = get_team_abbreviation(team_b)
    team_a_winner_class = " matchup-panel-winner" if winner == team_a else ""
    team_b_winner_class = " matchup-panel-winner" if winner == team_b else ""

    st.html(
        f"""
        <div class="matchup-board">
            <div class="matchup-panel{team_a_winner_class}" style="--team-color: {html.escape(team_a_color)};">
                <img class="matchup-logo" src="{html.escape(team_a_logo)}" alt="{html.escape(team_a)} logo">
                <div>
                    <div class="matchup-team">{html.escape(team_a)}</div>
                    <div class="matchup-sub">{html.escape(team_a_abbr)} / {html.escape(label)}</div>
                </div>
                <div class="matchup-prob">{probability_a:.0%}</div>
            </div>
            <div class="matchup-vs">VS</div>
            <div class="matchup-panel{team_b_winner_class}" style="--team-color: {html.escape(team_b_color)};">
                <img class="matchup-logo" src="{html.escape(team_b_logo)}" alt="{html.escape(team_b)} logo">
                <div>
                    <div class="matchup-team">{html.escape(team_b)}</div>
                    <div class="matchup-sub">{html.escape(team_b_abbr)} / {html.escape(label)}</div>
                </div>
                <div class="matchup-prob">{probability_b:.0%}</div>
            </div>
        </div>
        """
    )


def get_players_for_team(team_name: str) -> pd.DataFrame:
    """Return player impact rows for one NBA team."""
    file_mtime = PLAYER_IMPACT_PATH.stat().st_mtime if PLAYER_IMPACT_PATH.exists() else None
    player_impact = load_player_impact(file_mtime)
    abbreviation = get_team_abbreviation_from_strength(team_name)

    players = player_impact[
        player_impact["TEAM_ABBREVIATION"] == abbreviation
    ].copy()

    return players.sort_values(
        ["IMPACT_SCORE", "STAR_BONUS", "PPG", "APG", "MPG", "PLUS_MINUS_PER_GAME"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


@st.cache_data
def get_player_headshot_url(player_name: str) -> str:
    """Return an NBA headshot URL for one player, with a fallback image."""
    matches = nba_players.find_players_by_full_name(player_name)

    if matches:
        player_id = matches[0]["id"]
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

    initials = "".join(
        part[0].upper()
        for part in normalize_name_for_matching(player_name).split()
        if part
    )[:2] or "?"
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="104" height="104" viewBox="0 0 104 104">
        <rect width="104" height="104" rx="52" fill="#eef2ff"/>
        <circle cx="52" cy="38" r="17" fill="#c7d2fe"/>
        <path d="M18 90c7-18 21-26 34-26s27 8 34 26" fill="#c7d2fe"/>
        <text x="52" y="63" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1e3a8a">{initials}</text>
    </svg>
    """
    return f"data:image/svg+xml;charset=UTF-8,{quote(svg)}"


def normalize_name_for_matching(name: str) -> str:
    """Normalize player names for injury-report to stats matching."""
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"[^a-z0-9 ]", " ", ascii_name)
    ascii_name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", ascii_name)
    return re.sub(r"\s+", " ", ascii_name).strip()


def build_player_impact_lookup(players: pd.DataFrame) -> dict[str, pd.Series]:
    """Build normalized player-name to impact-row lookup."""
    return {
        normalize_name_for_matching(row["PLAYER_NAME"]): row
        for _, row in players.iterrows()
    }


def get_injury_report_status_weight(row: pd.Series) -> float:
    """Return parsed status weight for one official injury row."""
    if "STATUS_WEIGHT" in row and not pd.isna(row["STATUS_WEIGHT"]):
        return float(row["STATUS_WEIGHT"])

    fallback_weights = {
        "Out": 1.00,
        "Doubtful": 0.75,
        "Questionable": 0.50,
        "Probable": 0.10,
        "Available": 0.00,
    }
    return fallback_weights.get(str(row.get("CURRENT_STATUS", "")), 0.0)


def build_official_availability_adjustments(
    selected_teams: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate team penalties from the latest official NBA injury report."""
    team_adjustments = {team: 0.0 for team in selected_teams}
    injuries = load_current_injuries()

    if injuries.empty:
        return team_adjustments, pd.DataFrame()

    selected_injuries = injuries[injuries["TEAM"].isin(selected_teams)].copy()

    if selected_injuries.empty:
        return team_adjustments, pd.DataFrame()

    display_rows = []

    for team in selected_teams:
        team_injuries = selected_injuries[selected_injuries["TEAM"] == team]

        if team_injuries.empty:
            continue

        try:
            players = get_players_for_team(team)
            impact_lookup = build_player_impact_lookup(players)
        except FileNotFoundError:
            impact_lookup = {}

        total_weighted_impact = 0.0

        for _, injury in team_injuries.iterrows():
            player_name = str(injury["PLAYER_NAME"])
            normalized_name = normalize_name_for_matching(player_name)
            impact_row = impact_lookup.get(normalized_name)
            status_weight = get_injury_report_status_weight(injury)

            if impact_row is None:
                impact_score = DEFAULT_UNMATCHED_PLAYER_IMPACT
                impact_source = "Default fallback"
                player_detail = "No current stat match"
            else:
                impact_score = float(impact_row["IMPACT_SCORE"])
                impact_source = "Current player stats"
                player_detail = (
                    f"{float(impact_row['MPG']):.1f} MPG, "
                    f"{float(impact_row['PPG']):.1f} PPG"
                )

            weighted_impact = impact_score * status_weight
            total_weighted_impact += weighted_impact

            display_rows.append(
                {
                    "Team": team,
                    "Player": player_name,
                    "Status": injury["CURRENT_STATUS"],
                    "Availability Weight": status_weight,
                    "Impact Score": impact_score,
                    "Weighted Impact": weighted_impact,
                    "Impact Source": impact_source,
                    "Player Detail": player_detail,
                    "Reason": injury.get("REASON", ""),
                }
            )

        team_adjustments[team] = -min(
            total_weighted_impact,
            MAX_TEAM_AVAILABILITY_PENALTY,
        )

    display = pd.DataFrame(display_rows)

    if display.empty:
        return team_adjustments, display

    return team_adjustments, display.sort_values(
        ["Team", "Weighted Impact"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_current_injury_feature_values(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build current injury-related feature values for one matchup."""
    injuries = load_current_injuries()
    feature_values = {
        "DIFF_INJURY_WEIGHTED_IMPACT": 0.0,
        "DIFF_OUT_PLAYER_COUNT": 0.0,
        "DIFF_QUESTIONABLE_PLAYER_COUNT": 0.0,
    }

    if injuries.empty:
        return feature_values

    selected = injuries[injuries["TEAM"].isin([home_team, away_team])].copy()

    if selected.empty:
        return feature_values

    if team_adjustments is None:
        team_adjustments, _ = build_official_availability_adjustments(
            [home_team, away_team]
        )

    def count_status(team_rows: pd.DataFrame, keyword: str) -> int:
        return int(
            team_rows["CURRENT_STATUS"]
            .astype(str)
            .str.contains(keyword, case=False, na=False)
            .sum()
        )

    home_rows = selected[selected["TEAM"].eq(home_team)]
    away_rows = selected[selected["TEAM"].eq(away_team)]
    home_missing_impact = max(0.0, -float(team_adjustments.get(home_team, 0.0)))
    away_missing_impact = max(0.0, -float(team_adjustments.get(away_team, 0.0)))

    feature_values["DIFF_INJURY_WEIGHTED_IMPACT"] = (
        home_missing_impact - away_missing_impact
    )
    feature_values["DIFF_OUT_PLAYER_COUNT"] = (
        count_status(home_rows, r"\bout\b") - count_status(away_rows, r"\bout\b")
    )
    feature_values["DIFF_QUESTIONABLE_PLAYER_COUNT"] = (
        count_status(home_rows, r"questionable|doubtful")
        - count_status(away_rows, r"questionable|doubtful")
    )

    return feature_values


@st.cache_data
def load_current_player_advanced_stats(season: str = "2025-26") -> pd.DataFrame:
    """Load current-season advanced player stats when the NBA stats feed is available."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except Exception:
        return pd.DataFrame()

    try:
        response = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            timeout=5,
        )
        table = response.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

    return table if not table.empty else pd.DataFrame()


def estimate_player_position(player_row: pd.Series) -> str:
    """Estimate a player's position from local stats when no roster position is available."""
    mpg = float(player_row.get("MPG", 0.0) or 0.0)
    ppg = float(player_row.get("PPG", 0.0) or 0.0)
    apg = float(player_row.get("APG", 0.0) or 0.0)
    rpg = float(player_row.get("RPG", 0.0) or 0.0)

    if apg >= 5 and ppg >= 12:
        return "G"
    if rpg >= 8 and mpg >= 28 and apg <= 4:
        return "C/F"
    if rpg >= 6 and ppg >= 10:
        return "F"
    if ppg >= 15 or apg >= 4:
        return "W"
    return "UTIL"


@st.cache_data
def get_player_profile(
    team_name: str,
    player_name: str,
    file_mtime: float | None = None,
) -> dict[str, object]:
    """Build a player profile with local stats, position, and optional advanced metrics."""
    players = get_players_for_team(team_name)
    base_row = players[players["PLAYER_NAME"].eq(player_name)]
    if base_row.empty:
        base = pd.Series({"PLAYER_NAME": player_name})
    else:
        base = base_row.iloc[0]

    team_rank = None
    if not players.empty:
        rank_rows = players.index[players["PLAYER_NAME"].eq(player_name)].tolist()
        if rank_rows:
            team_rank = int(rank_rows[0] + 1)

    player_id = None
    matches = nba_players.find_players_by_full_name(player_name)
    if matches:
        player_id = matches[0]["id"]

    position = ""
    height = ""
    weight = ""
    experience = ""
    advanced: dict[str, object] = {}

    if player_id is not None:
        try:
            from nba_api.stats.endpoints import commonplayerinfo

            response = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=5)
            info_frames = response.get_data_frames()
            if info_frames:
                info = info_frames[0]
                if not info.empty:
                    info_row = info.iloc[0].to_dict()
                    position = str(info_row.get("POSITION", "") or "").strip()
                    height = str(info_row.get("HEIGHT", "") or "").strip()
                    weight = str(info_row.get("WEIGHT", "") or "").strip()
                    experience = str(info_row.get("SEASON_EXP", "") or "").strip()
        except Exception:
            pass

    advanced_df = load_current_player_advanced_stats()
    if not advanced_df.empty and "PLAYER_NAME" in advanced_df.columns:
        advanced_row = advanced_df[advanced_df["PLAYER_NAME"].eq(player_name)]
        if advanced_row.empty and "PLAYER_NAME" in advanced_df.columns:
            advanced_row = advanced_df[
                advanced_df["PLAYER_NAME"].astype(str).str.contains(
                    normalize_name_for_matching(player_name).split()[0],
                    case=False,
                    na=False,
                )
            ]
        if not advanced_row.empty:
            advanced = advanced_row.iloc[0].to_dict()

    if not position:
        position = estimate_player_position(base)

    stats = {
        "MPG": float(base.get("MPG", 0.0) or 0.0),
        "PPG": float(base.get("PPG", 0.0) or 0.0),
        "RPG": float(base.get("RPG", 0.0) or 0.0),
        "APG": float(base.get("APG", 0.0) or 0.0),
        "PLUS_MINUS_PER_GAME": float(base.get("PLUS_MINUS_PER_GAME", 0.0) or 0.0),
        "IMPACT_SCORE": float(base.get("IMPACT_SCORE", 0.0) or 0.0),
        "INJURY_TIER": str(base.get("INJURY_TIER", "")),
    }

    return {
        "name": player_name,
        "team": team_name,
        "player_id": player_id,
        "position": position,
        "height": height,
        "weight": weight,
        "experience": experience,
        "team_rank": team_rank,
        "stats": stats,
        "advanced": advanced,
        "headshot_url": get_player_headshot_url(player_name),
    }
    feature_values["DIFF_OUT_PLAYER_COUNT"] = (
        count_status(home_rows, r"\bout\b") - count_status(away_rows, r"\bout\b")
    )
    feature_values["DIFF_QUESTIONABLE_PLAYER_COUNT"] = (
        count_status(home_rows, r"questionable|doubtful")
        - count_status(away_rows, r"questionable|doubtful")
    )

    return feature_values


def build_matchup_feature_snapshot(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build a compact set of matchup signals for narrative UI and tooltips."""
    strength = load_team_strength()
    home_row = get_team_strength_row(home_team, strength)
    away_row = get_team_strength_row(away_team, strength)
    team_adjustments = team_adjustments or {}

    def safe_float(value: object, default: float = 0.0) -> float:
        return default if pd.isna(value) else float(value)

    injury_features = build_current_injury_feature_values(
        home_team=home_team,
        away_team=away_team,
        team_adjustments=team_adjustments,
    )
    home_rest = safe_float(home_row.get("DAYS_REST", 0.0))
    away_rest = safe_float(away_row.get("DAYS_REST", 0.0))
    home_net = safe_float(home_row.get("ROLLING_NET_RATING_10", 0.0))
    away_net = safe_float(away_row.get("ROLLING_NET_RATING_10", 0.0))
    home_season_net = safe_float(home_row.get("SEASON_AVG_NET_RATING", 0.0))
    away_season_net = safe_float(away_row.get("SEASON_AVG_NET_RATING", 0.0))
    home_pace = safe_float(
        home_row.get("SEASON_AVG_PACE", home_row.get("ROLLING_PACE_10", 0.0))
    )
    away_pace = safe_float(
        away_row.get("SEASON_AVG_PACE", away_row.get("ROLLING_PACE_10", 0.0))
    )
    home_top8 = safe_float(home_row.get("PLAYER_TOP_8", 0.0))
    away_top8 = safe_float(away_row.get("PLAYER_TOP_8", 0.0))
    home_star_count = int(safe_float(home_row.get("STAR_COUNT", 0.0)))
    away_star_count = int(safe_float(away_row.get("STAR_COUNT", 0.0)))

    signals = [
        {
            "label": "Rest",
            "value": f"{home_rest:.0f} vs {away_rest:.0f} days",
            "note": (
                home_team
                if home_rest != away_rest
                else "Even rest"
            ),
            "color": "#2563eb",
        },
        {
            "label": "Recent form",
            "value": f"{home_net - away_net:+.1f} net",
            "note": (
                home_team
                if home_net != away_net
                else "Recent form even"
            ),
            "color": "#db2777",
        },
        {
            "label": "Season strength",
            "value": f"{home_season_net - away_season_net:+.1f} net",
            "note": (
                home_team
                if home_season_net != away_season_net
                else "Season profile even"
            ),
            "color": "#0f766e",
        },
        {
            "label": "Player impact",
            "value": f"{home_top8 - away_top8:+.1f}",
            "note": f"Stars {home_star_count} vs {away_star_count}",
            "color": "#f97316",
        },
        {
            "label": "Pace",
            "value": f"{(home_pace + away_pace) / 2:.1f}",
            "note": "Projected possessions",
            "color": "#7c3aed",
        },
        {
            "label": "Availability",
            "value": f"{injury_features['DIFF_INJURY_WEIGHTED_IMPACT']:+.1f}",
            "note": "Positive means home has more missing impact",
            "color": "#ea580c",
        },
    ]

    reasons = []
    if abs(home_rest - away_rest) >= 1:
        reasons.append(
            f"Rest edge: {home_team if home_rest > away_rest else away_team} "
            f"has more rest."
        )
    if abs(injury_features["DIFF_INJURY_WEIGHTED_IMPACT"]) >= 1:
        reasons.append(
            f"Availability edge: "
            f"{home_team if injury_features['DIFF_INJURY_WEIGHTED_IMPACT'] < 0 else away_team} "
            f"is healthier."
        )
    if abs(home_net - away_net) >= 2:
        reasons.append(
            f"Recent form edge: "
            f"{home_team if home_net > away_net else away_team} is hotter."
        )
    if abs(home_top8 - away_top8) >= 2:
        reasons.append(
            f"Rotation edge: "
            f"{home_team if home_top8 > away_top8 else away_team} has more top-end depth."
        )
    if playoff_context:
        reasons.append("Playoff context is included in the model input.")

    top_reasons = reasons[:3]

    return {
        "signals": signals,
        "top_reasons": top_reasons,
        "injury_features": injury_features,
    }


def build_active_roster_summary(
    selected_teams: list[str],
    availability_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize active roster strength after official availability adjustments."""
    rows = []

    for team in selected_teams:
        try:
            players = get_players_for_team(team)
        except FileNotFoundError:
            continue

        if players.empty:
            continue

        baseline_top_5 = float(players.head(5)["IMPACT_SCORE"].sum())
        baseline_top_8 = float(players.head(8)["IMPACT_SCORE"].sum())

        team_availability = availability_rows[
            availability_rows["Team"].eq(team)
        ] if not availability_rows.empty else pd.DataFrame()

        if team_availability.empty:
            weighted_missing_impact = 0.0
            missing_star_count = 0
        else:
            weighted_missing_impact = float(team_availability["Weighted Impact"].sum())
            missing_star_count = int(
                (
                    (team_availability["Impact Score"] >= 6.5)
                    & (team_availability["Availability Weight"] > 0)
                ).sum()
            )

        rows.append(
            {
                "Team": team,
                "Baseline Top 5": baseline_top_5,
                "Baseline Top 8": baseline_top_8,
                "Weighted Missing Impact": weighted_missing_impact,
                "Estimated Available Top 8": max(
                    0.0,
                    baseline_top_8 - weighted_missing_impact,
                ),
                "Missing Star Count": missing_star_count,
            }
        )

    return pd.DataFrame(rows)


def refresh_official_injury_report() -> pd.DataFrame:
    """Fetch the latest official NBA injury report into the local CSV."""
    from src.injuries import build_current_injuries_file

    injuries = build_current_injuries_file()
    load_current_injuries.clear()
    return injuries


def render_official_availability_adjustments(
    selected_teams: list[str],
    key_prefix: str,
) -> dict[str, float]:
    """Render and return official injury-report team adjustments."""
    injuries = load_current_injuries()

    if injuries.empty:
        with st.expander("Availability: no report", expanded=False):
            if st.button("Refresh report", key=f"{key_prefix}_refresh_injuries"):
                try:
                    with st.spinner("Fetching latest official NBA injury report..."):
                        refresh_official_injury_report()
                    st.success("Injury report refreshed.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not refresh injury report: {error}")
        return {team: 0.0 for team in selected_teams}

    report_timestamp = injuries.iloc[0].get("REPORT_TIMESTAMP", "Unknown")
    source_url = injuries.iloc[0].get("SOURCE_URL", "")
    team_adjustments, availability_rows = build_official_availability_adjustments(
        selected_teams,
    )
    impacted_rows = pd.DataFrame()

    if not availability_rows.empty:
        impacted_rows = availability_rows[
            availability_rows["Weighted Impact"].fillna(0) > 0
        ].copy()

    adjusted_team_count = sum(
        1
        for adjustment in team_adjustments.values()
        if adjustment != 0
    )
    availability_label = (
        f"Availability: {len(impacted_rows)} listed impact"
        if not impacted_rows.empty
        else "Availability: clear"
    )
    with st.expander(availability_label, expanded=False):
        control_col, report_col, source_col = st.columns([1.1, 2.0, 1.1])

        with control_col:
            if st.button("Refresh", key=f"{key_prefix}_refresh_injuries"):
                try:
                    with st.spinner("Fetching latest official NBA injury report..."):
                        refresh_official_injury_report()
                    st.success("Refreshed.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Refresh failed: {error}")

        with report_col:
            st.caption(
                f"Report {format_status_timestamp(report_timestamp)} / "
                f"{adjusted_team_count} team adjustment(s)"
            )

        if source_url:
            with source_col:
                st.link_button("PDF", source_url)

        if availability_rows.empty:
            st.caption("No listed impact for the selected teams.")
        else:
            display = availability_rows.copy()

            for column in ["Availability Weight", "Impact Score", "Weighted Impact"]:
                display[column] = display[column].map("{:.2f}".format)

            display_columns = [
                "Team",
                "Player",
                "Status",
                "Impact Score",
                "Weighted Impact",
                "Reason",
            ]
            st.dataframe(
                display[
                    [column for column in display_columns if column in display.columns]
                ],
                width="stretch",
                hide_index=True,
            )

    return team_adjustments


def get_game_confidence_label(probability: float) -> str:
    """Convert single-game probability into a readable confidence label."""
    favorite_probability = max(probability, 1 - probability)

    if favorite_probability < 0.55:
        return "Toss-up"
    if favorite_probability < 0.62:
        return "Slight edge"
    if favorite_probability < 0.72:
        return "Strong edge"

    return "Heavy favorite"


def get_series_confidence_label(probability: float) -> str:
    """Convert series probability into a readable confidence label."""
    favorite_probability = max(probability, 1 - probability)

    if favorite_probability < 0.55:
        return "Toss-up"
    if favorite_probability < 0.65:
        return "Slight edge"
    if favorite_probability < 0.80:
        return "Strong edge"

    return "Heavy favorite"


def get_bracket_confidence_label(championship_probability: float) -> str:
    """Convert championship probability into a readable confidence label."""
    if championship_probability < 0.12:
        return "Wide-open race"
    if championship_probability < 0.20:
        return "Contender"
    if championship_probability < 0.30:
        return "Strong contender"

    return "Championship favorite"


def expected_score(team_elo: float, opponent_elo: float) -> float:
    """Calculate Elo expected win probability."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def clamp_probability(probability: float) -> float:
    """Avoid unrealistically extreme single-game probabilities."""
    return min(
        max(probability, MIN_SINGLE_GAME_PROBABILITY),
        MAX_SINGLE_GAME_PROBABILITY,
    )


def shrink_probability(probability: float) -> float:
    """Pull extreme probabilities slightly toward 50%."""
    return 0.5 + ((probability - 0.5) * PROBABILITY_SHRINKAGE)


def apply_team_adjustments(
    probability: float,
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None,
) -> float:
    """Apply stat-based player availability adjustments to a home win probability."""
    if not team_adjustments:
        return probability

    safe_probability = min(max(probability, 0.001), 0.999)
    log_odds = math.log(safe_probability / (1 - safe_probability))

    adjustment_delta = team_adjustments.get(home_team, 0.0) - team_adjustments.get(
        away_team,
        0.0,
    )

    adjusted_log_odds = log_odds + (
        adjustment_delta * TEAM_ADJUSTMENT_LOGIT_WEIGHT
    )
    adjusted_probability = 1 / (1 + math.exp(-adjusted_log_odds))

    return clamp_probability(adjusted_probability)


def blend_model_and_elo_probability(
    model_probability: float,
    elo_probability: float,
    model_bundle: dict | None = None,
) -> float:
    """Blend calibrated model probability with Elo probability."""
    blend_settings = {}

    if model_bundle:
        blend_settings = model_bundle.get("blend_settings", {}) or {}

    model_weight = float(
        blend_settings.get("model_probability_weight", MODEL_PROBABILITY_WEIGHT)
    )
    elo_weight = float(
        blend_settings.get("elo_probability_weight", ELO_PROBABILITY_WEIGHT)
    )
    probability_shrinkage = float(
        blend_settings.get("probability_shrinkage", PROBABILITY_SHRINKAGE)
    )

    blended_probability = (
        model_weight * model_probability
        + elo_weight * elo_probability
    )
    shrunk_probability = 0.5 + (
        (blended_probability - 0.5) * probability_shrinkage
    )
    return clamp_probability(shrunk_probability)


def model_trained_with_historical_injuries(model_bundle: dict) -> bool:
    """Return whether the saved model saw non-zero historical injury features."""
    feature_availability = model_bundle.get("feature_availability", {}) or {}
    return bool(feature_availability.get("historical_injuries_used", False))


def convert_model_feature_to_strength_column(feature_column: str) -> str | None:
    """Map trained model feature names to current team strength columns."""
    if feature_column == "DIFF_ELO":
        return "ELO"

    if feature_column == "HOME_ELO_WIN_PROB":
        return None

    if feature_column in INJURY_MODEL_FEATURES:
        return None

    if feature_column == "DIFF_ROAD_TRIP_GAME_NUMBER":
        return None

    if not feature_column.startswith("DIFF_"):
        return None

    strength_column = feature_column.replace("DIFF_", "")

    if strength_column.startswith("PREV_SEASON_PLAYER_"):
        return strength_column.replace("PREV_SEASON_PLAYER_", "PLAYER_")

    if strength_column == "PREV_SEASON_STAR_COUNT":
        return "STAR_COUNT"

    return strength_column


def build_prediction_row_from_strength(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
    feature_columns: list[str],
    playoff_context: dict[str, float] | None = None,
    team_adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build one model input row from current team strength."""
    home_strength = get_team_strength_row(home_team, strength)
    away_strength = get_team_strength_row(away_team, strength)
    injury_features = build_current_injury_feature_values(
        home_team=home_team,
        away_team=away_team,
        team_adjustments=team_adjustments,
    )

    prediction_values = {}
    playoff_context = playoff_context or {}

    for feature_column in feature_columns:
        if feature_column == "HOME_ELO_WIN_PROB":
            prediction_values[feature_column] = expected_score(
                float(home_strength["ELO"]) + HOME_ELO_ADVANTAGE,
                float(away_strength["ELO"]),
            )
            continue

        if feature_column in PLAYOFF_CONTEXT_MODEL_FEATURES:
            prediction_values[feature_column] = float(
                playoff_context.get(feature_column, 0.0)
            )
            continue

        if feature_column in INJURY_MODEL_FEATURES:
            prediction_values[feature_column] = float(
                injury_features.get(feature_column, 0.0)
            )
            continue

        if feature_column == "DIFF_ROAD_TRIP_GAME_NUMBER":
            home_road_trip_game_number = 0.0
            away_road_trip_game_number = float(
                away_strength.get("CURRENT_ROAD_STREAK", 0.0)
            ) + 1.0
            prediction_values[feature_column] = (
                home_road_trip_game_number - away_road_trip_game_number
            )
            continue

        strength_column = convert_model_feature_to_strength_column(feature_column)

        if strength_column is None:
            raise ValueError(f"Could not map feature column: {feature_column}")

        prediction_values[feature_column] = (
            float(home_strength[strength_column])
            - float(away_strength[strength_column])
        )

    return pd.DataFrame([prediction_values], columns=feature_columns)


def get_elo_home_probability(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
) -> float:
    """Get Elo-only home win probability."""
    home_strength = get_team_strength_row(home_team, strength)
    away_strength = get_team_strength_row(away_team, strength)

    return expected_score(
        float(home_strength["ELO"]) + HOME_ELO_ADVANTAGE,
        float(away_strength["ELO"]),
    )


def predict_game_probability_details(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
    model_bundle: dict,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> dict:
    """Return model, Elo, adjusted, and blended home-team win probabilities."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    prediction_row = build_prediction_row_from_strength(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        feature_columns=feature_columns,
        playoff_context=playoff_context,
        team_adjustments=team_adjustments,
    )

    model_probability = float(model.predict_proba(prediction_row)[0][1])
    elo_probability = get_elo_home_probability(home_team, away_team, strength)
    blended_probability = blend_model_and_elo_probability(
        model_probability=model_probability,
        elo_probability=elo_probability,
        model_bundle=model_bundle,
    )

    if model_trained_with_historical_injuries(model_bundle):
        final_probability = blended_probability
    else:
        final_probability = apply_team_adjustments(
            probability=blended_probability,
            home_team=home_team,
            away_team=away_team,
            team_adjustments=team_adjustments,
        )

    return {
        "model_probability": model_probability,
        "elo_probability": elo_probability,
        "blended_probability": blended_probability,
        "final_probability": final_probability,
    }


def predict_game_probability(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
    model_bundle: dict,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> float:
    """Return final home team win probability."""
    return predict_game_probability_details(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        model_bundle=model_bundle,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )["final_probability"]


def build_prediction_rows_for_matchups(
    matchups: list[tuple[str, str]],
    strength: pd.DataFrame,
    feature_columns: list[str],
    team_adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build model input rows for many home-away matchups at once."""
    strength_by_team = {
        str(row["TEAM_NAME"]).lower(): row
        for _, row in strength.iterrows()
    }
    rows = []

    for home_team, away_team in matchups:
        home_strength = strength_by_team[home_team.lower()]
        away_strength = strength_by_team[away_team.lower()]
        injury_features = build_current_injury_feature_values(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=team_adjustments,
        )
        prediction_values = {}

        for feature_column in feature_columns:
            if feature_column == "HOME_ELO_WIN_PROB":
                prediction_values[feature_column] = expected_score(
                    float(home_strength["ELO"]) + HOME_ELO_ADVANTAGE,
                    float(away_strength["ELO"]),
                )
                continue

            if feature_column in PLAYOFF_CONTEXT_MODEL_FEATURES:
                prediction_values[feature_column] = 0.0
                continue

            if feature_column in INJURY_MODEL_FEATURES:
                prediction_values[feature_column] = float(
                    injury_features.get(feature_column, 0.0)
                )
                continue

            if feature_column == "DIFF_ROAD_TRIP_GAME_NUMBER":
                prediction_values[feature_column] = -(
                    float(away_strength.get("CURRENT_ROAD_STREAK", 0.0)) + 1.0
                )
                continue

            strength_column = convert_model_feature_to_strength_column(feature_column)

            if strength_column is None:
                raise ValueError(f"Could not map feature column: {feature_column}")

            prediction_values[feature_column] = (
                float(home_strength[strength_column])
                - float(away_strength[strength_column])
            )

        rows.append(prediction_values)

    return pd.DataFrame(rows, columns=feature_columns)


def build_matchup_probability_cache(
    teams: list[str],
    strength: pd.DataFrame,
    model_bundle: dict,
    team_adjustments: dict[str, float] | None = None,
) -> dict[tuple[str, str], float]:
    """Precompute all home-away probabilities for selected teams."""
    matchups = [
        (home_team, away_team)
        for home_team in teams
        for away_team in teams
        if home_team != away_team
    ]

    if not matchups:
        return {}

    feature_columns = model_bundle["feature_columns"]
    prediction_rows = build_prediction_rows_for_matchups(
        matchups=matchups,
        strength=strength,
        feature_columns=feature_columns,
        team_adjustments=team_adjustments,
    )
    model = model_bundle["model"]
    model_probabilities = model.predict_proba(prediction_rows)[:, 1]
    strength_by_team = {
        str(row["TEAM_NAME"]).lower(): row
        for _, row in strength.iterrows()
    }
    probability_cache = {}
    uses_historical_injuries = model_trained_with_historical_injuries(model_bundle)

    for (home_team, away_team), model_probability in zip(matchups, model_probabilities):
        home_strength = strength_by_team[home_team.lower()]
        away_strength = strength_by_team[away_team.lower()]
        elo_probability = expected_score(
            float(home_strength["ELO"]) + HOME_ELO_ADVANTAGE,
            float(away_strength["ELO"]),
        )
        blended_probability = blend_model_and_elo_probability(
            model_probability=float(model_probability),
            elo_probability=elo_probability,
            model_bundle=model_bundle,
        )

        if uses_historical_injuries:
            final_probability = blended_probability
        else:
            final_probability = apply_team_adjustments(
                probability=blended_probability,
                home_team=home_team,
                away_team=away_team,
                team_adjustments=team_adjustments,
            )

        probability_cache[(home_team, away_team)] = final_probability

    return probability_cache


def get_cached_home_win_probability(
    home_team: str,
    away_team: str,
    probability_cache: dict[tuple[str, str], float],
) -> float:
    """Return cached home-team win probability."""
    key = (home_team, away_team)

    if key not in probability_cache:
        raise KeyError(f"Missing cached probability for {home_team} vs {away_team}")

    return probability_cache[key]


def predict_game(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> tuple[str, float, float, dict]:
    """Predict winner and win probabilities."""
    model_bundle = load_model_bundle()
    strength = load_team_strength()

    probability_details = predict_game_probability_details(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        model_bundle=model_bundle,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )

    home_probability = probability_details["final_probability"]
    away_probability = 1 - home_probability
    winner = home_team if home_probability >= 0.5 else away_team

    return winner, home_probability, away_probability, probability_details


def probability_to_logit(probability: float) -> float:
    """Convert a probability into log-odds with numerical guards."""
    safe_probability = min(max(float(probability), 0.001), 0.999)
    return math.log(safe_probability / (1 - safe_probability))


def logit_to_probability(logit: float) -> float:
    """Convert log-odds back into a probability."""
    return 1 / (1 + math.exp(-float(logit)))


def format_probability_shift(shift: float) -> str:
    """Format a probability delta as percentage points."""
    return f"{shift * 100:+.1f} pts"


def predict_game_probability_details_with_feature_overrides(
    home_team: str,
    away_team: str,
    strength: pd.DataFrame,
    model_bundle: dict,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
    feature_overrides: dict[str, float] | None = None,
    home_elo_advantage: float = HOME_ELO_ADVANTAGE,
) -> dict:
    """Predict a matchup after controlled feature overrides for scenario analysis."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    prediction_row = build_prediction_row_from_strength(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        feature_columns=feature_columns,
        playoff_context=playoff_context,
        team_adjustments=team_adjustments,
    )

    for feature_column, value in (feature_overrides or {}).items():
        if feature_column in prediction_row.columns:
            prediction_row.loc[0, feature_column] = float(value)

    model_probability = float(model.predict_proba(prediction_row)[0][1])
    home_strength = get_team_strength_row(home_team, strength)
    away_strength = get_team_strength_row(away_team, strength)
    elo_probability = expected_score(
        float(home_strength["ELO"]) + home_elo_advantage,
        float(away_strength["ELO"]),
    )
    blended_probability = blend_model_and_elo_probability(
        model_probability=model_probability,
        elo_probability=elo_probability,
        model_bundle=model_bundle,
    )

    if model_trained_with_historical_injuries(model_bundle):
        final_probability = blended_probability
    else:
        final_probability = apply_team_adjustments(
            probability=blended_probability,
            home_team=home_team,
            away_team=away_team,
            team_adjustments=team_adjustments,
        )

    return {
        "model_probability": model_probability,
        "elo_probability": elo_probability,
        "blended_probability": blended_probability,
        "final_probability": final_probability,
    }


def build_scenario_feature_overrides(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None,
    playoff_context: dict[str, float] | None,
    home_rest_delta: int = 0,
    away_rest_delta: int = 0,
    home_back_to_back: bool = False,
    away_back_to_back: bool = False,
    neutral_site: bool = False,
) -> dict[str, float]:
    """Build model feature overrides from Prediction Lab controls."""
    model_bundle = load_model_bundle()
    strength = load_team_strength()
    feature_columns = model_bundle["feature_columns"]
    base_row = build_prediction_row_from_strength(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        feature_columns=feature_columns,
        playoff_context=playoff_context,
        team_adjustments=team_adjustments,
    )
    overrides: dict[str, float] = {}

    if "DIFF_DAYS_REST" in base_row.columns:
        overrides["DIFF_DAYS_REST"] = (
            float(base_row.loc[0, "DIFF_DAYS_REST"])
            + float(home_rest_delta - away_rest_delta)
        )

    if "DIFF_IS_BACK_TO_BACK" in base_row.columns and (home_back_to_back or away_back_to_back):
        overrides["DIFF_IS_BACK_TO_BACK"] = float(
            int(home_back_to_back) - int(away_back_to_back)
        )

    if "DIFF_IS_THIRD_IN_FOUR_DAYS" in base_row.columns and (home_back_to_back or away_back_to_back):
        overrides["DIFF_IS_THIRD_IN_FOUR_DAYS"] = float(
            int(home_back_to_back and home_rest_delta < 0)
            - int(away_back_to_back and away_rest_delta < 0)
        )

    if "DIFF_GAMES_LAST_7_DAYS" in base_row.columns:
        overrides["DIFF_GAMES_LAST_7_DAYS"] = (
            float(base_row.loc[0, "DIFF_GAMES_LAST_7_DAYS"])
            + float(int(home_back_to_back) - int(away_back_to_back))
        )

    if neutral_site and "HOME_ELO_WIN_PROB" in base_row.columns:
        home_strength = get_team_strength_row(home_team, strength)
        away_strength = get_team_strength_row(away_team, strength)
        overrides["HOME_ELO_WIN_PROB"] = expected_score(
            float(home_strength["ELO"]),
            float(away_strength["ELO"]),
        )

    return overrides


def get_selected_player_impacts(team_name: str, player_names: list[str]) -> float:
    """Return summed impact score for selected players."""
    if not player_names:
        return 0.0

    players = get_players_for_team(team_name)
    selected = players[players["PLAYER_NAME"].isin(player_names)]
    if selected.empty:
        return 0.0

    return float(selected["IMPACT_SCORE"].sum())


def render_prediction_lab(
    home_team: str,
    away_team: str,
    base_home_probability: float,
    team_adjustments: dict[str, float],
    playoff_context: dict[str, float] | None,
    key_prefix: str,
) -> None:
    """Render what-if controls and scenario probability shifts."""
    with st.expander("Prediction Lab", expanded=True):
        control_cols = st.columns(2, gap="medium")
        home_players = get_players_for_team(home_team).head(8)
        away_players = get_players_for_team(away_team).head(8)

        with control_cols[0]:
            neutral_site = st.checkbox(
                "Neutral site",
                key=f"{key_prefix}_lab_neutral_site",
            )
            home_rest_delta = st.slider(
                f"{home_team} rest adjustment",
                min_value=-3,
                max_value=3,
                value=0,
                step=1,
                key=f"{key_prefix}_lab_home_rest",
            )
            home_back_to_back = st.checkbox(
                f"{home_team} on back-to-back",
                key=f"{key_prefix}_lab_home_b2b",
            )
            home_unavailable = st.multiselect(
                f"{home_team} additional unavailable",
                home_players["PLAYER_NAME"].tolist(),
                key=f"{key_prefix}_lab_home_unavailable",
            )

        with control_cols[1]:
            away_rest_delta = st.slider(
                f"{away_team} rest adjustment",
                min_value=-3,
                max_value=3,
                value=0,
                step=1,
                key=f"{key_prefix}_lab_away_rest",
            )
            away_back_to_back = st.checkbox(
                f"{away_team} on back-to-back",
                key=f"{key_prefix}_lab_away_b2b",
            )
            away_unavailable = st.multiselect(
                f"{away_team} additional unavailable",
                away_players["PLAYER_NAME"].tolist(),
                key=f"{key_prefix}_lab_away_unavailable",
            )

        scenario_adjustments = dict(team_adjustments or {})
        scenario_adjustments[home_team] = scenario_adjustments.get(home_team, 0.0) - get_selected_player_impacts(
            home_team,
            home_unavailable,
        )
        scenario_adjustments[away_team] = scenario_adjustments.get(away_team, 0.0) - get_selected_player_impacts(
            away_team,
            away_unavailable,
        )
        feature_overrides = build_scenario_feature_overrides(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=scenario_adjustments,
            playoff_context=playoff_context,
            home_rest_delta=int(home_rest_delta),
            away_rest_delta=int(away_rest_delta),
            home_back_to_back=bool(home_back_to_back),
            away_back_to_back=bool(away_back_to_back),
            neutral_site=bool(neutral_site),
        )
        strength = load_team_strength()
        model_bundle = load_model_bundle()
        scenario_details = predict_game_probability_details_with_feature_overrides(
            home_team=home_team,
            away_team=away_team,
            strength=strength,
            model_bundle=model_bundle,
            team_adjustments=scenario_adjustments,
            playoff_context=playoff_context,
            feature_overrides=feature_overrides,
            home_elo_advantage=0.0 if neutral_site else HOME_ELO_ADVANTAGE,
        )
        scenario_home_probability = float(scenario_details["final_probability"])
        scenario_away_probability = 1 - scenario_home_probability
        scenario_winner = (
            home_team if scenario_home_probability >= scenario_away_probability else away_team
        )
        scenario_winner_probability = max(
            scenario_home_probability,
            scenario_away_probability,
        )
        shift = scenario_home_probability - float(base_home_probability)

        render_prediction_result_card(
            label="Scenario result",
            winner=scenario_winner,
            probability=scenario_winner_probability,
            confidence=get_game_confidence_label(scenario_winner_probability),
            details=scenario_details,
            context="game",
            note=f"{format_probability_shift(shift)} home-team shift from baseline",
        )
        render_dashboard_cards(
            [
                {
                    "label": f"{home_team} baseline",
                    "value": f"{base_home_probability:.0%}",
                    "note": "Original home win chance",
                    "color": get_team_color(home_team),
                },
                {
                    "label": f"{home_team} scenario",
                    "value": f"{scenario_home_probability:.0%}",
                    "note": format_probability_shift(shift),
                    "color": get_team_color(home_team),
                },
                {
                    "label": f"{away_team} scenario",
                    "value": f"{scenario_away_probability:.0%}",
                    "note": format_probability_shift(-shift),
                    "color": get_team_color(away_team),
                },
                {
                    "label": "Changed inputs",
                    "value": str(
                        sum(
                            [
                                bool(neutral_site),
                                home_rest_delta != 0,
                                away_rest_delta != 0,
                                bool(home_back_to_back),
                                bool(away_back_to_back),
                                bool(home_unavailable),
                                bool(away_unavailable),
                            ]
                        )
                    ),
                    "note": "Scenario controls active",
                    "color": "#7c3aed",
                },
            ]
        )


def get_team_signal_values(team_name: str) -> dict[str, float]:
    """Return compact team metrics used by narrative features."""
    strength = load_team_strength()
    row = get_team_strength_row(team_name, strength)

    def safe_float(column: str, default: float = 0.0) -> float:
        value = row.get(column, default)
        return default if pd.isna(value) else float(value)

    return {
        "rest": safe_float("DAYS_REST"),
        "recent_net": safe_float("ROLLING_NET_RATING_10"),
        "season_net": safe_float("SEASON_AVG_NET_RATING"),
        "offense": safe_float("SEASON_AVG_OFF_RATING"),
        "defense": safe_float("SEASON_AVG_DEF_RATING"),
        "top8": safe_float("PLAYER_TOP_8"),
        "pace": safe_float("SEASON_AVG_PACE", safe_float("ROLLING_PACE_10")),
    }


def build_underdog_path_rows(
    home_team: str,
    away_team: str,
    home_probability: float,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> list[dict[str, str]]:
    """Build a plain-English path for the underdog to win."""
    away_probability = 1 - home_probability
    underdog = home_team if home_probability < away_probability else away_team
    favorite = away_team if underdog == home_team else home_team
    underdog_probability = min(home_probability, away_probability)
    underdog_values = get_team_signal_values(underdog)
    favorite_values = get_team_signal_values(favorite)
    snapshot = build_matchup_feature_snapshot(
        home_team=home_team,
        away_team=away_team,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )
    injury_diff = float(snapshot["injury_features"]["DIFF_INJURY_WEIGHTED_IMPACT"])
    underdog_missing_more = (
        injury_diff > 0 if underdog == home_team else injury_diff < 0
    )
    rows = []

    if underdog_values["recent_net"] < favorite_values["recent_net"]:
        rows.append(
            {
                "Lever": "Flip the recent-form gap",
                "Need": (
                    f"Outplay the favorite's last-10 net rating edge "
                    f"({favorite_values['recent_net'] - underdog_values['recent_net']:.1f})."
                ),
            }
        )

    if underdog_values["top8"] < favorite_values["top8"]:
        rows.append(
            {
                "Lever": "Top-end minutes have to win",
                "Need": (
                    f"Close a {favorite_values['top8'] - underdog_values['top8']:.1f} "
                    "rotation-impact gap with star scoring or bench minutes."
                ),
            }
        )

    if underdog_values["rest"] < favorite_values["rest"]:
        rows.append(
            {
                "Lever": "Protect the tired stretches",
                "Need": (
                    f"Handle a {favorite_values['rest'] - underdog_values['rest']:.0f}-day "
                    "rest disadvantage without losing the middle quarters."
                ),
            }
        )

    if underdog_missing_more:
        rows.append(
            {
                "Lever": "Cover availability loss",
                "Need": "Replacement minutes need to keep the game close enough for variance.",
            }
        )

    if underdog_values["defense"] > favorite_values["defense"]:
        rows.append(
            {
                "Lever": "Win the shot-quality battle",
                "Need": "The underdog defense grades worse, so turnovers and transition defense matter more.",
            }
        )

    rows.append(
        {
            "Lever": "Keep it within two possessions late",
            "Need": (
                f"The betting board still gives {underdog} a {underdog_probability:.0%} path; "
                "late-game variance is the cleanest route."
            ),
        }
    )

    return rows[:4]


def render_underdog_path(
    home_team: str,
    away_team: str,
    home_probability: float,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> None:
    """Render the underdog's best route to an upset."""
    away_probability = 1 - home_probability
    underdog = home_team if home_probability < away_probability else away_team
    underdog_probability = min(home_probability, away_probability)
    rows = build_underdog_path_rows(
        home_team=home_team,
        away_team=away_team,
        home_probability=home_probability,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )
    row_html = "".join(
        f"""
        <div class="player-row">
            <div class="team-mini-score">{index}</div>
            <div>
                <div class="player-name">{html.escape(row["Lever"])}</div>
                <div class="player-meta">{html.escape(row["Need"])}</div>
            </div>
            <div class="impact-score">{underdog_probability:.0%}</div>
        </div>
        """
        for index, row in enumerate(rows, start=1)
    )
    render_section_kicker("How the underdog wins", f"{underdog} upset path")
    st.html(row_html)


def find_calibration_row(home_probability: float) -> pd.Series | None:
    """Find the nearest saved calibration bucket for a home probability."""
    calibration = load_calibration_metrics()
    if calibration.empty or "Average_Predicted_Probability" not in calibration.columns:
        return None

    rows = calibration.copy()
    rows["Distance"] = (
        rows["Average_Predicted_Probability"].astype(float) - float(home_probability)
    ).abs()
    return rows.sort_values("Distance").iloc[0]


def render_prediction_trust_meter(
    home_probability: float,
    winner_probability: float,
) -> None:
    """Render a local bet grade using pricing spread and similar-game context."""
    calibration_row = find_calibration_row(home_probability)
    calibration_gap = None
    bucket_label = "No bucket"
    bucket_games = "0"

    if calibration_row is not None:
        predicted = float(calibration_row["Average_Predicted_Probability"])
        actual = float(calibration_row["Actual_Home_Win_Rate"])
        calibration_gap = abs(predicted - actual)
        bucket_label = str(calibration_row.get("Bucket", "Nearest bucket"))
        bucket_games = str(int(calibration_row.get("Games", 0)))

    if calibration_gap is None:
        grade_label = "Needs price"
        grade_note = "Similar-game context unavailable"
    elif calibration_gap <= 0.035:
        grade_label = "Strong"
        grade_note = "Clean similar-game profile"
    elif calibration_gap <= 0.065:
        grade_label = "Playable"
        grade_note = "Review the sportsbook price"
    else:
        grade_label = "Pass"
        grade_note = "Pricing profile is volatile"

    lower, upper, margin = estimate_prediction_interval(
        winner_probability,
        context="game",
    )
    render_section_kicker("Bet grade")
    render_dashboard_cards(
        [
            {
                "label": "Grade",
                "value": grade_label,
                "note": grade_note,
                "color": "#0f766e",
            },
            {
                "label": "Similar Games",
                "value": bucket_games,
                "note": bucket_label,
                "color": "#2563eb",
            },
            {
                "label": "Win Range",
                "value": f"{lower:.0%}-{upper:.0%}",
                "note": f"About {margin:.0%} swing",
                "color": "#f97316",
            },
        ]
    )


def build_similar_historical_games(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
    top_n: int = 6,
) -> pd.DataFrame:
    """Find historical games with similar model-feature profiles."""
    try:
        historical = load_model_features()
        model_bundle = load_model_bundle()
        strength = load_team_strength()
    except Exception:
        return pd.DataFrame()

    feature_columns = [
        column
        for column in model_bundle.get("feature_columns", [])
        if column in historical.columns
    ]
    if not feature_columns:
        return pd.DataFrame()

    current_row = build_prediction_row_from_strength(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        feature_columns=feature_columns,
        playoff_context=playoff_context,
        team_adjustments=team_adjustments,
    )
    historical_values = historical[feature_columns].astype(float).fillna(0.0)
    current_values = current_row.iloc[0][feature_columns].astype(float).fillna(0.0)
    stds = historical_values.std().replace(0, 1).fillna(1)
    distances = (((historical_values - current_values) / stds) ** 2).mean(axis=1) ** 0.5
    selected = historical.copy()
    selected["Similarity Score"] = 1 / (1 + distances)
    selected = selected.sort_values("Similarity Score", ascending=False).head(top_n)

    rows = []
    for _, row in selected.iterrows():
        home_win = int(row.get("HOME_WIN", 0)) == 1
        actual_winner = row["HOME_TEAM"] if home_win else row["AWAY_TEAM"]
        rows.append(
            {
                "Date": format_status_timestamp(row.get("GAME_DATE")),
                "Season": row.get("SEASON", ""),
                "Matchup": f"{row['AWAY_TEAM']} at {row['HOME_TEAM']}",
                "Actual Winner": actual_winner,
                "Similarity": float(row["Similarity Score"]),
            }
        )

    return pd.DataFrame(rows)


def render_similar_historical_games(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> None:
    """Render similar historical games for evidence context."""
    similar_games = build_similar_historical_games(
        home_team=home_team,
        away_team=away_team,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )
    render_section_kicker("Similar historical games")

    if similar_games.empty:
        st.caption("No historical feature matches are available.")
        return

    display = similar_games.copy()
    display["Similarity"] = display["Similarity"].map("{:.0%}".format)
    st.dataframe(display, width="stretch", hide_index=True)


def get_current_player_status_lookup(teams: list[str]) -> dict[tuple[str, str], str]:
    """Return current injury-report status by normalized team/player."""
    injuries = load_current_injuries()
    if injuries.empty:
        return {}

    selected = injuries[injuries["TEAM"].isin(teams)].copy()
    return {
        (str(row["TEAM"]), normalize_name_for_matching(str(row["PLAYER_NAME"]))): str(
            row.get("CURRENT_STATUS", "")
        )
        for _, row in selected.iterrows()
    }


def build_player_swing_rankings(
    home_team: str,
    away_team: str,
    base_home_probability: float,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
    limit_per_team: int = 6,
) -> pd.DataFrame:
    """Estimate which player availability changes move the matchup most."""
    strength = load_team_strength()
    model_bundle = load_model_bundle()
    status_lookup = get_current_player_status_lookup([home_team, away_team])
    rows = []

    for team in [home_team, away_team]:
        players = get_players_for_team(team).head(limit_per_team)

        for _, player in players.iterrows():
            player_name = str(player["PLAYER_NAME"])
            impact = float(player["IMPACT_SCORE"])
            scenario_adjustments = dict(team_adjustments or {})
            scenario_adjustments[team] = scenario_adjustments.get(team, 0.0) - impact
            scenario_details = predict_game_probability_details(
                home_team=home_team,
                away_team=away_team,
                strength=strength,
                model_bundle=model_bundle,
                team_adjustments=scenario_adjustments,
                playoff_context=playoff_context,
            )
            scenario_home_probability = float(scenario_details["final_probability"])
            if team == home_team:
                team_shift = scenario_home_probability - float(base_home_probability)
            else:
                team_shift = (
                    (1 - scenario_home_probability)
                    - (1 - float(base_home_probability))
                )

            rows.append(
                {
                    "Team": team,
                    "Player": player_name,
                    "Status": status_lookup.get(
                        (team, normalize_name_for_matching(player_name)),
                        "Active/Unlisted",
                    ),
                    "Impact": impact,
                    "Team Win Shift If Unavailable": team_shift,
                    "Game Swing": abs(scenario_home_probability - float(base_home_probability)),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("Game Swing", ascending=False)
        .reset_index(drop=True)
    )


def render_player_swing_rankings(
    home_team: str,
    away_team: str,
    base_home_probability: float,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> None:
    """Render player availability swing rankings."""
    swings = build_player_swing_rankings(
        home_team=home_team,
        away_team=away_team,
        base_home_probability=base_home_probability,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )
    render_section_kicker("Player swing rankings")

    if swings.empty:
        st.caption("Player swing rankings are unavailable.")
        return

    display = swings.head(8).copy()
    display["Impact"] = display["Impact"].map("{:.1f}".format)
    display["Team Win Shift If Unavailable"] = display[
        "Team Win Shift If Unavailable"
    ].map(format_probability_shift)
    display["Game Swing"] = display["Game Swing"].map(lambda value: f"{value * 100:.1f} pts")
    st.dataframe(display, width="stretch", hide_index=True)


def parse_live_clock_remaining_minutes(clock: object) -> float | None:
    """Parse a live clock value into minutes remaining in the period."""
    formatted = format_live_clock(clock)
    match = re.search(r"(\d+):(\d{2})", formatted)
    if not match:
        return None

    return int(match.group(1)) + (int(match.group(2)) / 60)


def estimate_live_home_probability(
    pregame_home_probability: float,
    home_score: object,
    away_score: object,
    period: object,
    clock: object,
) -> float | None:
    """Estimate a lightweight live probability from score margin and game progress."""
    try:
        home_points = float(home_score)
        away_points = float(away_score)
    except (TypeError, ValueError):
        return None

    if home_points == 0 and away_points == 0:
        return None

    try:
        period_number = int(float(period))
    except (TypeError, ValueError):
        period_number = 2

    remaining = parse_live_clock_remaining_minutes(clock)
    if remaining is None:
        remaining = 0.0 if period_number >= 4 else 6.0

    elapsed = ((max(period_number, 1) - 1) * 12) + max(0.0, 12 - remaining)
    progress = min(max(elapsed / FULL_GAME_SIMULATION_MINUTES, 0.05), 1.0)
    margin = home_points - away_points
    score_weight = 0.035 + (0.105 * progress)
    live_logit = probability_to_logit(pregame_home_probability) + (margin * score_weight)
    return clamp_probability(logit_to_probability(live_logit))


def build_live_probability_movement_html(game: pd.Series) -> str:
    """Build a live probability movement note for current-game cards."""
    home_score = normalize_score_value(game.get("Home Score", ""))
    away_score = normalize_score_value(game.get("Away Score", ""))
    if home_score == "" or away_score == "":
        return ""

    home_team = str(game["Home Team"])
    away_team = str(game["Away Team"])
    pregame_home_probability = float(game["Home Win Probability"])
    live_home_probability = estimate_live_home_probability(
        pregame_home_probability=pregame_home_probability,
        home_score=home_score,
        away_score=away_score,
        period=game.get("Period", 0),
        clock=game.get("Game Clock", ""),
    )
    if live_home_probability is None:
        return ""

    live_winner = home_team if live_home_probability >= 0.5 else away_team
    shift = live_home_probability - pregame_home_probability
    score_margin = float(home_score) - float(away_score)
    margin_team = home_team if score_margin >= 0 else away_team
    reason = (
        f"{margin_team} leads by {abs(score_margin):.0f}"
        if score_margin != 0
        else "The score is tied"
    )

    return f"""
        <div class="summary-box">
            <div class="dashboard-label">Live probability movement</div>
            <div class="result-meta">
                Pregame home chance {pregame_home_probability:.0%} to live estimate
                {live_home_probability:.0%} ({format_probability_shift(shift)}).
                Current lean: {html.escape(live_winner)} because {html.escape(reason.lower())}.
            </div>
        </div>
    """


def build_upset_alert_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank current games by underdog upset potential."""
    if predictions.empty:
        return pd.DataFrame()

    rows = []
    for _, game in predictions.iterrows():
        home_team = str(game["Home Team"])
        away_team = str(game["Away Team"])
        home_probability = float(game["Home Win Probability"])
        away_probability = 1 - home_probability
        favorite = home_team if home_probability >= away_probability else away_team
        underdog = away_team if favorite == home_team else home_team
        underdog_probability = min(home_probability, away_probability)
        home_values = get_team_signal_values(home_team)
        away_values = get_team_signal_values(away_team)
        snapshot = build_matchup_feature_snapshot(
            home_team=home_team,
            away_team=away_team,
        )
        injury_diff = float(snapshot["injury_features"]["DIFF_INJURY_WEIGHTED_IMPACT"])
        vulnerabilities = []

        if favorite == home_team:
            if home_values["recent_net"] + 2 < away_values["recent_net"]:
                vulnerabilities.append("favorite recent-form drag")
            if home_values["rest"] < away_values["rest"]:
                vulnerabilities.append("favorite rest disadvantage")
            if injury_diff > 1:
                vulnerabilities.append("favorite availability risk")
        else:
            if away_values["recent_net"] + 2 < home_values["recent_net"]:
                vulnerabilities.append("favorite recent-form drag")
            if away_values["rest"] < home_values["rest"]:
                vulnerabilities.append("favorite rest disadvantage")
            if injury_diff < -1:
                vulnerabilities.append("favorite availability risk")

        alert_score = underdog_probability + (0.035 * len(vulnerabilities))
        rows.append(
            {
                "Game": f"{away_team} at {home_team}",
                "Favorite": favorite,
                "Underdog": underdog,
                "Upset Chance": underdog_probability,
                "Alert Score": alert_score,
                "Why": ", ".join(vulnerabilities) or "price is close enough to monitor",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Alert Score", ascending=False)
        .reset_index(drop=True)
    )


def render_upset_alert_board(predictions: pd.DataFrame) -> None:
    """Render the Today page's upset board."""
    alerts = build_upset_alert_rows(predictions)
    render_section_kicker("Upset alert board")

    if alerts.empty:
        st.caption("No upset alerts are available for this slate.")
        return

    display = alerts.head(5).copy()
    display["Upset Chance"] = display["Upset Chance"].map("{:.0%}".format)
    display["Alert Score"] = display["Alert Score"].map("{:.2f}".format)
    st.dataframe(display, width="stretch", hide_index=True)


def build_today_game_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Predict all currently scheduled games that can be mapped to local teams."""
    if games.empty:
        return pd.DataFrame()

    available_teams = set(get_teams_from_strength(load_team_strength()))
    selected_teams = sorted(
        {
            str(team)
            for team in pd.concat([games["Home Team"], games["Away Team"]])
            if str(team) in available_teams
        }
    )
    team_adjustments, _ = build_official_availability_adjustments(selected_teams)
    rows = []

    for _, game in games.iterrows():
        home_team = str(game["Home Team"])
        away_team = str(game["Away Team"])

        if home_team not in available_teams or away_team not in available_teams:
            continue

        playoff_context = infer_current_playoff_context_for_matchup(
            home_team=home_team,
            away_team=away_team,
        )
        winner, home_probability, away_probability, details = predict_game(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=team_adjustments,
            playoff_context=playoff_context,
        )
        winner_probability = (
            home_probability if winner == home_team else away_probability
        )
        matchup_snapshot = build_matchup_feature_snapshot(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=team_adjustments,
            playoff_context=playoff_context,
        )
        top_reason = (
            matchup_snapshot["top_reasons"][0]
            if matchup_snapshot["top_reasons"]
            else ""
        )
        rows.append(
            {
                **game.to_dict(),
                "Predicted Winner": winner,
                "Winner Probability": winner_probability,
                "Home Win Probability": home_probability,
                "Away Win Probability": away_probability,
                "Confidence": get_game_confidence_label(winner_probability),
                "Model Probability": details["model_probability"],
                "Elo Probability": details["elo_probability"],
                "Blended Probability": details["blended_probability"],
                "Final Probability": details["final_probability"],
                "Playoff Context": bool(playoff_context),
                "Prediction Note": top_reason,
            }
        )

    return pd.DataFrame(rows)


def build_today_prediction_summary_html(
    winner: str,
    winner_probability: float,
    confidence: object,
    home_team: str,
    home_probability: float,
) -> str:
    """Build a consistent prediction summary for current-game cards."""
    return f"""
        <div class="today-prediction-grid">
            <div class="today-prediction-cell">
                <div class="today-prediction-label">Pick</div>
                <div class="today-prediction-value">{html.escape(winner)}</div>
                <div class="today-prediction-note">{winner_probability:.0%} win chance</div>
            </div>
            <div class="today-prediction-cell">
                <div class="today-prediction-label">Confidence</div>
                <div class="today-prediction-value">{html.escape(str(confidence))}</div>
                <div class="today-prediction-note">Pick signal</div>
            </div>
            <div class="today-prediction-cell">
                <div class="today-prediction-label">Home team chance</div>
                <div class="today-prediction-value">{home_probability:.0%}</div>
                <div class="today-prediction-note">Chance {html.escape(home_team)} wins</div>
            </div>
        </div>
    """


def render_today_game_card(game: pd.Series, compact: bool = False) -> None:
    """Render one current-game card with score and prediction."""
    home_team = str(game["Home Team"])
    away_team = str(game["Away Team"])
    winner = str(game["Predicted Winner"])
    winner_probability = float(game["Winner Probability"])
    status = str(game.get("Status", "Scheduled"))
    game_time = str(game.get("Game Time", ""))
    source = str(game.get("Source", ""))
    prediction_note = str(game.get("Prediction Note", ""))
    home_score = game.get("Home Score", "")
    away_score = game.get("Away Score", "")
    team_color = get_team_color(winner)
    live_game_id = str(game.get("Game ID", "")).strip()

    def format_score(value: object) -> str:
        if value is None or pd.isna(value) or str(value) == "":
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

        if numeric.is_integer():
            return str(int(numeric))

        return str(numeric)

    home_score_text = format_score(home_score)
    away_score_text = format_score(away_score)
    timing = build_today_game_timing_state(game)
    live_state, _ = build_live_game_state(game)
    movement_game = game.copy()

    live_snapshot = resolve_live_today_snapshot(live_game_id, home_team, away_team)
    if live_snapshot is not None:
        live_home = format_score(live_snapshot.get("Home Score", home_score_text))
        live_away = format_score(live_snapshot.get("Away Score", away_score_text))
        live_label = str(live_snapshot.get("Status", "")).strip()
        live_detail = build_live_detail_label(
            live_label,
            live_snapshot.get("Period"),
            live_snapshot.get("Game Clock"),
        )
        live_score_present = bool(live_home or live_away)
        live_time_present = bool(live_detail)
        for column in ["Home Score", "Away Score", "Period", "Game Clock", "Status"]:
            if column in live_snapshot:
                movement_game[column] = live_snapshot.get(column)

        if live_score_present:
            home_score_text = live_home or home_score_text
            away_score_text = live_away or away_score_text

        if "final" in live_label.lower():
            timing = {
                "mode": "final",
                "badge": "FINAL",
                "detail": "",
                "status": "Final",
                "refresh": "0",
            }
        elif live_score_present or live_time_present:
            live_state = live_detail or live_state or "Live"
            timing = {
                "mode": "live",
                "badge": "LIVE",
                "detail": live_state,
                "status": "Live",
                "refresh": "15",
            }

    away_score_html = (
        f'<div class="today-team-score">{html.escape(away_score_text)}</div>'
        if away_score_text
        else ""
    )
    home_score_html = (
        f'<div class="today-team-score">{html.escape(home_score_text)}</div>'
        if home_score_text
        else ""
    )

    context_note = "Playoff context included" if bool(game.get("Playoff Context")) else source
    reason_html = ""
    timing_html = ""
    play_html = ""
    prediction_summary_html = build_today_prediction_summary_html(
        winner=winner,
        winner_probability=winner_probability,
        confidence=game["Confidence"],
        home_team=home_team,
        home_probability=float(game["Home Win Probability"]),
    )

    if prediction_note and prediction_note.lower() != "nan":
        reason_html = f'<div class="dashboard-note">{html.escape(prediction_note)}</div>'

    if timing["mode"] == "live":
        timing_html = (
            '<div class="today-live-state">'
            '<span class="today-state-badge">LIVE</span>'
            f'<span class="today-state-detail">{html.escape(live_state)}</span>'
            '</div>'
        )
    elif timing["mode"] in {"countdown", "starting"}:
        badge_class = "today-state-badge is-countdown"
        timing_html = (
            '<div class="today-live-state">'
            f'<span class="{badge_class}">{html.escape(str(timing["badge"]))}</span>'
            f'<span class="today-state-detail">{html.escape(str(timing["detail"]))}</span>'
            '</div>'
        )

    espn_game_id_value = game.get("ESPN Game ID", "")
    espn_game_id = (
        ""
        if espn_game_id_value is None or pd.isna(espn_game_id_value)
        else str(espn_game_id_value).strip()
    )

    if timing["mode"] == "live" and espn_game_id:
        latest_play_line = format_latest_play_line(load_espn_latest_play(espn_game_id))

        if latest_play_line:
            play_html = (
                f'<div class="today-play-line">{html.escape(latest_play_line)}</div>'
            )

    live_probability_html = build_live_probability_movement_html(movement_game)

    st.html(
        f"""
        <div class="today-card" style="--team-color: {html.escape(team_color)};">
            <div class="today-topline">
                <div>
                    <div class="today-label">{html.escape(game_time or source)}</div>
                    <div class="dashboard-note">{html.escape(context_note)}</div>
                    {timing_html}
                </div>
                <span class="status-chip">{html.escape(str(timing["status"]))}</span>
            </div>
            {reason_html}
            <div class="today-matchup">
                <div class="today-team">
                    <img class="today-logo" src="{html.escape(get_logo_url(away_team))}" alt="{html.escape(away_team)} logo">
                    <div>
                        <div class="today-name">{html.escape(away_team)}</div>
                        {away_score_html}
                        <div class="team-mini-meta">Away</div>
                    </div>
                </div>
                <div>
                    <div class="today-vs">AT</div>
                </div>
                <div class="today-team">
                    <div>
                        <div class="today-name">{html.escape(home_team)}</div>
                        {home_score_html}
                        <div class="team-mini-meta">Home</div>
                    </div>
                    <img class="today-logo" src="{html.escape(get_logo_url(home_team))}" alt="{html.escape(home_team)} logo">
                </div>
            </div>
            {play_html}
            {prediction_summary_html}
            {live_probability_html}
        </div>
        """
    )


def render_today_player_summary_card(team_name: str, player_row: pd.Series, key: str) -> None:
    """Render one clickable star-player summary card."""
    file_mtime = PLAYER_IMPACT_PATH.stat().st_mtime if PLAYER_IMPACT_PATH.exists() else None
    profile = get_player_profile(team_name, str(player_row["PLAYER_NAME"]), file_mtime)
    stats = profile["stats"]
    position = str(profile["position"] or "EST")
    photo_url = str(profile["headshot_url"])
    team_rank = profile.get("team_rank")
    team_rank_label = f"Team #{team_rank}" if team_rank else ""
    meta_parts = [team_name, position]
    if team_rank_label:
        meta_parts.append(team_rank_label)
    summary_meta = " / ".join(meta_parts)

    st.html(
        f"""
        <div class="today-player-card">
            <div class="today-player-card-head">
                <img class="today-player-photo" src="{html.escape(photo_url)}" alt="{html.escape(str(player_row['PLAYER_NAME']))} headshot">
                <div>
                    <div class="today-player-name">{html.escape(str(player_row["PLAYER_NAME"]))}</div>
                    <div class="today-player-meta">{html.escape(summary_meta)}</div>
                </div>
            </div>
            <div class="today-player-meta">
                {float(stats["MPG"]):.1f} MPG / {float(stats["PPG"]):.1f} PPG / {float(stats["RPG"]):.1f} RPG / {float(stats["APG"]):.1f} APG
            </div>
            <div class="today-player-actions">
                <span class="team-mini-score">{float(stats["IMPACT_SCORE"]):.2f}</span>
                <span class="score-chip">{html.escape(str(stats["INJURY_TIER"]))}</span>
            </div>
        </div>
        """
    )

    if st.button("View stats", key=key, width="stretch"):
        st.session_state["today_player_profile"] = profile


def render_matchup_comparison_panel(
    team_a: str,
    team_b: str,
    title: str = "Star comparison",
    key_prefix: str = "matchup",
) -> None:
    """Render star comparison and projected starting lineups for two teams."""
    team_a_players = get_players_for_team(team_a).head(1)
    team_b_players = get_players_for_team(team_b).head(1)

    with st.container(border=True):
        st.html(
            f"""
            <div class="today-comparison-title" style="margin-top: 0.1rem;">{html.escape(title)}</div>
            """
        )
        comparison_cols = st.columns(2, gap="medium")

        with comparison_cols[0]:
            if team_a_players.empty:
                st.caption("No player data found.")
            else:
                render_today_player_summary_card(
                    team_a,
                    team_a_players.iloc[0],
                    f"{key_prefix}_star_{team_a}_btn",
                )

        with comparison_cols[1]:
            if team_b_players.empty:
                st.caption("No player data found.")
            else:
                render_today_player_summary_card(
                    team_b,
                    team_b_players.iloc[0],
                    f"{key_prefix}_star_{team_b}_btn",
                )

        st.html(
            """
            <div class="today-impact-legend">
                <div class="today-impact-legend-title">Impact levels</div>
                <div class="today-impact-legend-grid">
                    <div class="today-impact-legend-item">
                        <div class="today-impact-legend-score">0.0 - 2.0</div>
                        <div class="today-impact-legend-label">Low rotation impact</div>
                    </div>
                    <div class="today-impact-legend-item">
                        <div class="today-impact-legend-score">2.0 - 4.0</div>
                        <div class="today-impact-legend-label">Rotation player</div>
                    </div>
                    <div class="today-impact-legend-item">
                        <div class="today-impact-legend-score">4.0 - 6.5</div>
                        <div class="today-impact-legend-label">Important player</div>
                    </div>
                    <div class="today-impact-legend-item">
                        <div class="today-impact-legend-score">6.5 - 8.5</div>
                        <div class="today-impact-legend-label">Star-level impact</div>
                    </div>
                    <div class="today-impact-legend-item">
                        <div class="today-impact-legend-score">8.5 - 10.0</div>
                        <div class="today-impact-legend-label">Superstar-level impact</div>
                    </div>
                </div>
                <div class="today-impact-disclaimer">
                    Player scores are calculated from current-season box-score production and impact signals:
                    minutes, points, rebounds, assists, steals, blocks, plus/minus, and turnovers.
                    Scores are normalized to a 0-10 scale, then grouped into impact tiers.
                </div>
            </div>
            """
        )

        lineup_cols = st.columns(2, gap="medium")
        with lineup_cols[0]:
            st.html(
                f"""
                <div class="today-breakdown-panel">
                    <div class="today-breakdown-panel-title">{html.escape(team_a)} projected starting lineup</div>
                    {render_projected_lineup_rows(team_a)}
                </div>
                """
            )
        with lineup_cols[1]:
            st.html(
                f"""
                <div class="today-breakdown-panel">
                    <div class="today-breakdown-panel-title">{html.escape(team_b)} projected starting lineup</div>
                    {render_projected_lineup_rows(team_b)}
                </div>
                """
            )


def render_projected_lineup_rows(team_name: str) -> str:
    """Return HTML for a projected starting lineup based on player impact."""
    players = get_players_for_team(team_name).head(5)
    file_mtime = PLAYER_IMPACT_PATH.stat().st_mtime if PLAYER_IMPACT_PATH.exists() else None

    if players.empty:
        return """
        <div class="today-lineup-team">
            <div class="today-lineup-team-name">Projected starting lineup unavailable</div>
        </div>
        """

    rows = []
    for slot, (_, player) in enumerate(players.iterrows(), start=1):
        profile = get_player_profile(team_name, str(player["PLAYER_NAME"]), file_mtime)
        position = str(profile["position"] or estimate_player_position(player))
        rows.append(
            f"""
            <div class="today-lineup-row">
                <div class="today-lineup-number">{html.escape(position)}</div>
                <div class="today-lineup-name">{html.escape(str(player["PLAYER_NAME"]))}</div>
                <div class="today-lineup-meta">{float(player["IMPACT_SCORE"]):.1f}</div>
            </div>
            """
        )

    return (
        f"""
        <div class="today-lineup-team">
            <div class="today-lineup-team-name">{html.escape(team_name)}</div>
            {''.join(rows)}
        </div>
        """
    )


def render_today_player_profile_dialog() -> None:
    """Render a modal with player details when a star player is selected."""
    profile = st.session_state.get("today_player_profile")
    if not profile:
        return

    def clear_profile() -> None:
        st.session_state.pop("today_player_profile", None)
        st.session_state.pop("today_player_profile_open", None)

    @st.dialog(
        f"{profile['name']} profile",
        width="large",
        dismissible=True,
        on_dismiss=clear_profile,
    )
    def _dialog() -> None:
        stats = profile["stats"]
        advanced = profile["advanced"] or {}

        left_col, right_col = st.columns([0.8, 1.2], gap="medium")

        with left_col:
            st.html(
                f"""
                <div class="today-player-card">
                    <div class="today-player-card-head">
                        <img class="today-player-photo" src="{html.escape(str(profile['headshot_url']))}" alt="{html.escape(str(profile['name']))} headshot">
                        <div>
                            <div class="today-player-name">{html.escape(str(profile['name']))}</div>
                    <div class="today-player-meta">
                        {html.escape(str(profile['team']))} / {html.escape(str(profile['position']))}
                        {f" / Team #{int(profile['team_rank'])}" if profile.get('team_rank') else ""}
                    </div>
                        </div>
                    </div>
                    <div class="today-player-meta">
                        {html.escape(str(profile['height'] or ''))}
                        {(" / " + html.escape(str(profile['weight']))) if profile['weight'] else ""}
                        {(" / " + html.escape(str(profile['experience'])) + " seasons") if profile['experience'] else ""}
                    </div>
                </div>
                """
            )

            metric_rows = [
                ("MPG", f"{stats['MPG']:.1f}"),
                ("PPG", f"{stats['PPG']:.1f}"),
                ("RPG", f"{stats['RPG']:.1f}"),
                ("APG", f"{stats['APG']:.1f}"),
                ("Plus/Minus", f"{stats['PLUS_MINUS_PER_GAME']:+.1f}"),
                ("Impact", f"{stats['IMPACT_SCORE']:.2f}"),
            ]
            if profile.get("team_rank"):
                metric_rows.insert(0, ("Team rank", f"#{int(profile['team_rank'])}"))
            metric_cols = st.columns(3, gap="small")
            for index, (label, value) in enumerate(metric_rows):
                with metric_cols[index % 3]:
                    st.metric(label, value)

        with right_col:
            st.subheader("Advanced stats")
            advanced_fields = [
                ("USG_PCT", "Usage %"),
                ("TS_PCT", "True Shooting %"),
                ("AST_PCT", "Assist %"),
                ("REB_PCT", "Rebound %"),
                ("EFG_PCT", "eFG %"),
                ("TOV_PCT", "TOV %"),
                ("PIE", "PIE"),
                ("OFF_RATING", "Off Rating"),
                ("DEF_RATING", "Def Rating"),
                ("NET_RATING", "Net Rating"),
                ("PACE", "Pace"),
            ]

            advanced_rows = []
            for field, label in advanced_fields:
                if field not in advanced:
                    continue
                value = advanced[field]
                if pd.isna(value):
                    continue
                if field.endswith("_PCT") or field in {"PIE"}:
                    formatted = f"{float(value):.1%}" if float(value) <= 1.0 else f"{float(value):.1f}"
                else:
                    formatted = f"{float(value):.1f}"
                advanced_rows.append({"Metric": label, "Value": formatted})

            if advanced_rows:
                st.dataframe(pd.DataFrame(advanced_rows), width="stretch", hide_index=True)
            else:
                st.caption("Advanced NBA stats were not available locally for this player.")

        if st.button("Close", key="today_player_profile_close"):
            clear_profile()
            st.rerun()

    _dialog()


def render_today_game_breakdown_card(game: pd.Series) -> None:
    """Render one game card with star players and projected lineups."""
    away_team = str(game["Away Team"])
    home_team = str(game["Home Team"])
    timing = build_today_game_timing_state(game)
    status = str(timing.get("status", "Scheduled"))
    winner = str(game["Predicted Winner"])
    winner_probability = float(game["Winner Probability"])
    prediction_summary_html = build_today_prediction_summary_html(
        winner=winner,
        winner_probability=winner_probability,
        confidence=game["Confidence"],
        home_team=home_team,
        home_probability=float(game["Home Win Probability"]),
    )

    st.html(
        f"""
        <div class="today-breakdown-card">
            <div class="today-breakdown-top">
                <div>
                    <div class="today-breakdown-title">{html.escape(away_team)} at {html.escape(home_team)}</div>
                    <div class="today-breakdown-sub">{html.escape(str(timing.get("detail", "")) or str(game.get("Game Time", "")))}</div>
                </div>
                <span class="status-chip">{html.escape(status)}</span>
            </div>
        </div>
        """
    )

    render_matchup_comparison_panel(
        team_a=away_team,
        team_b=home_team,
        title="Star comparison",
        key_prefix=f"today_{away_team}_{home_team}",
    )

    st.html(
        f"""
        {prediction_summary_html}
        """
    )
    movement_html = build_live_probability_movement_html(game)
    if movement_html:
        st.html(movement_html)

    team_adjustments, _ = build_official_availability_adjustments([home_team, away_team])
    playoff_context = infer_current_playoff_context_for_matchup(
        home_team=home_team,
        away_team=away_team,
    )
    render_prediction_trust_meter(
        home_probability=float(game["Home Win Probability"]),
        winner_probability=winner_probability,
    )
    render_section_kicker("Prediction factors")
    render_matchup_signal_cards(
        home_team=home_team,
        away_team=away_team,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )
    render_underdog_path(
        home_team=home_team,
        away_team=away_team,
        home_probability=float(game["Home Win Probability"]),
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )


def render_today_game_breakdown_selector(games: pd.DataFrame) -> None:
    """Render a selectable game breakdown for the current slate."""
    if games.empty:
        st.caption("No games are available for breakdown right now.")
        return

    def build_label(row: pd.Series) -> str:
        return f"{row['Away Team']} at {row['Home Team']}"

    labels = [build_label(row) for _, row in games.iterrows()]
    selected_label = st.selectbox(
        "Game",
        labels,
        key="today_breakdown_game",
    )

    selected_index = labels.index(selected_label)
    selected_game = games.iloc[selected_index]
    timing = build_today_game_timing_state(selected_game)
    st.caption(str(timing.get("detail", "")) or str(selected_game.get("Game Time", "")))
    render_today_game_breakdown_card(selected_game)


@st.fragment(run_every=1)
def render_today_game_breakdown_live_fragment() -> None:
    """Render the current-game breakdown from the same live source as the cards."""
    games = load_today_games()
    is_upcoming_slate = False

    if games.empty:
        games = load_next_upcoming_games()
        is_upcoming_slate = True

        if games.empty:
            st.info("No current or upcoming games are available from the free feeds.")
            return

    predictions = build_today_game_predictions(games)

    if predictions.empty:
        st.info("No current or upcoming games are available for prediction.")
        return

    if is_upcoming_slate:
        render_section_kicker("Upcoming breakdown", format_game_slate_label(games) or None)
    else:
        render_section_kicker("Today breakdown")

    render_today_game_breakdown_selector(predictions)


def render_today_games_cards(games: pd.DataFrame, compact: bool = False) -> None:
    """Render current-game prediction cards."""
    if games.empty:
        st.info("No current NBA games were found in the live feed or injury report.")
        return

    column_count = 1 if len(games) == 1 else min(2, len(games))
    columns = st.columns(column_count)

    for index, (_, game) in enumerate(games.iterrows()):
        with columns[index % column_count]:
            render_today_game_card(game, compact=compact)


def render_today_game_summary(games: pd.DataFrame) -> None:
    """Render a compact summary of today's game situation."""
    if games.empty:
        st.caption("No games are available right now.")
        return

    live_games = []
    upcoming_games = []
    final_games = []

    for _, game in games.iterrows():
        timing = build_today_game_timing_state(game)
        if timing["mode"] == "live":
            live_games.append(game)
        elif timing["mode"] in {"countdown", "starting"}:
            upcoming_games.append((game, timing))
        elif timing["mode"] == "final":
            final_games.append(game)

    summary_rows = []

    if live_games:
        summary_rows.append(
            f"""
            <div class="summary-box">
                <div class="dashboard-label">Live now</div>
                <div class="result-title">{len(live_games)} game(s) in progress</div>
                <div class="result-meta">The live scoreboard is updating with score and clock.</div>
            </div>
            """
        )

    if upcoming_games:
        next_game, timing = min(
            upcoming_games,
            key=lambda item: (
                coerce_game_datetime(item[0].get("Game DateTime"))
                if not pd.isna(coerce_game_datetime(item[0].get("Game DateTime")))
                else pd.Timestamp.max
            ),
        )
        summary_rows.append(
            f"""
            <div class="summary-box">
                <div class="dashboard-label">Next tipoff</div>
                <div class="result-title">{html.escape(str(next_game["Away Team"]))} at {html.escape(str(next_game["Home Team"]))}</div>
                <div class="result-meta">{html.escape(str(timing["badge"]))}</div>
                <div class="compact-note">{html.escape(str(timing["detail"]))}</div>
            </div>
            """
        )

    if final_games:
        summary_rows.append(
            f"""
            <div class="summary-box">
                <div class="dashboard-label">Already final</div>
                <div class="result-title">{len(final_games)} game(s)</div>
                <div class="result-meta">Results are already locked in for today.</div>
            </div>
            """
        )

    st.html("".join(summary_rows))


def get_weighted_strength_value(
    row: pd.Series,
    rolling_column: str,
    season_column: str,
    rolling_weight: float = 0.58,
) -> float:
    """Blend recent and season-long values for score projection."""
    season_value = row.get(season_column)
    rolling_value = row.get(rolling_column)

    if pd.isna(season_value):
        season_value = rolling_value

    if pd.isna(rolling_value):
        rolling_value = season_value

    return (
        float(rolling_value) * rolling_weight
        + float(season_value) * (1 - rolling_weight)
    )


def estimate_matchup_expected_scores(
    home_team: str,
    away_team: str,
    home_probability: float,
    team_adjustments: dict[str, float] | None = None,
) -> dict[str, float]:
    """Estimate expected final scores from pace, ratings, probability, and injuries."""
    strength = load_team_strength()
    home_row = get_team_strength_row(home_team, strength)
    away_row = get_team_strength_row(away_team, strength)
    league_def_rating = float(strength["SEASON_AVG_DEF_RATING"].mean())

    home_offense = get_weighted_strength_value(
        home_row,
        "ROLLING_OFF_RATING_10",
        "SEASON_AVG_OFF_RATING",
    )
    away_offense = get_weighted_strength_value(
        away_row,
        "ROLLING_OFF_RATING_10",
        "SEASON_AVG_OFF_RATING",
    )
    home_defense = get_weighted_strength_value(
        home_row,
        "ROLLING_DEF_RATING_10",
        "SEASON_AVG_DEF_RATING",
    )
    away_defense = get_weighted_strength_value(
        away_row,
        "ROLLING_DEF_RATING_10",
        "SEASON_AVG_DEF_RATING",
    )
    home_pace = get_weighted_strength_value(
        home_row,
        "ROLLING_PACE_10",
        "SEASON_AVG_PACE",
    )
    away_pace = get_weighted_strength_value(
        away_row,
        "ROLLING_PACE_10",
        "SEASON_AVG_PACE",
    )
    expected_pace = (home_pace + away_pace) / 2

    matchup_home_rating = home_offense + (away_defense - league_def_rating)
    matchup_away_rating = away_offense + (home_defense - league_def_rating)
    raw_home_score = matchup_home_rating * expected_pace / 100
    raw_away_score = matchup_away_rating * expected_pace / 100

    probability = min(max(home_probability, 0.01), 0.99)
    probability_margin = (
        math.log(probability / (1 - probability)) * SCORE_MARGIN_POINTS_PER_LOGIT
    )
    rating_margin = raw_home_score - raw_away_score

    home_availability_delta = 0.0
    away_availability_delta = 0.0

    if team_adjustments:
        home_availability_delta = team_adjustments.get(home_team, 0.0) * 0.35
        away_availability_delta = team_adjustments.get(away_team, 0.0) * 0.35

    expected_total = raw_home_score + raw_away_score
    expected_total += home_availability_delta + away_availability_delta
    expected_total = min(max(expected_total, 188.0), 265.0)

    expected_margin = (probability_margin * 0.68) + (rating_margin * 0.32)
    expected_margin += home_availability_delta - away_availability_delta

    home_expected = min(max((expected_total + expected_margin) / 2, 82.0), 155.0)
    away_expected = min(max((expected_total - expected_margin) / 2, 82.0), 155.0)

    return {
        "home_expected": home_expected,
        "away_expected": away_expected,
        "expected_total": home_expected + away_expected,
        "expected_margin": home_expected - away_expected,
        "expected_pace": expected_pace,
    }


def sample_projected_score(
    home_team: str,
    away_team: str,
    home_probability: float,
    team_adjustments: dict[str, float] | None,
    rng: random.Random,
) -> tuple[int, int]:
    """Sample one plausible final score for a matchup."""
    expected = estimate_matchup_expected_scores(
        home_team=home_team,
        away_team=away_team,
        home_probability=home_probability,
        team_adjustments=team_adjustments,
    )
    total = rng.gauss(expected["expected_total"], SCORE_TOTAL_STD_DEV)
    margin = rng.gauss(expected["expected_margin"], SCORE_MARGIN_STD_DEV)
    home_score = max(72, int(round((total + margin) / 2)))
    away_score = max(72, int(round((total - margin) / 2)))

    if home_score == away_score:
        if rng.random() < home_probability:
            home_score += 1
        else:
            away_score += 1

    return home_score, away_score


def force_projected_score_winner(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    target_winner: str,
    rng: random.Random,
) -> tuple[int, int]:
    """Adjust a sampled score so it matches a target winner."""
    current_winner = home_team if home_score > away_score else away_team

    if current_winner == target_winner:
        return home_score, away_score

    total = home_score + away_score
    margin = min(
        18,
        max(1, int(round(abs(home_score - away_score) * 0.55)) + rng.randint(1, 5)),
    )
    winner_score = int(round((total + margin) / 2))
    loser_score = max(72, total - winner_score)

    if winner_score <= loser_score:
        winner_score = loser_score + 1

    if target_winner == home_team:
        return winner_score, loser_score

    if target_winner == away_team:
        return loser_score, winner_score

    return home_score, away_score


def sample_projected_score_for_winner(
    home_team: str,
    away_team: str,
    home_probability: float,
    team_adjustments: dict[str, float] | None,
    target_winner: str,
    rng: random.Random,
) -> tuple[int, int]:
    """Sample a plausible score while matching a planned game winner."""
    home_score, away_score = sample_projected_score(
        home_team=home_team,
        away_team=away_team,
        home_probability=home_probability,
        team_adjustments=team_adjustments,
        rng=rng,
    )
    return force_projected_score_winner(
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        target_winner=target_winner,
        rng=rng,
    )


def split_score_by_quarter(total_points: int, rng: random.Random) -> list[int]:
    """Split a final score into four realistic quarter scores."""
    adjusted_weights = [
        max(0.18, weight + rng.uniform(-0.018, 0.018))
        for weight in QUARTER_SCORE_WEIGHTS
    ]
    weight_total = sum(adjusted_weights)
    normalized_weights = [weight / weight_total for weight in adjusted_weights]
    first_three = [
        max(12, int(round(total_points * weight)))
        for weight in normalized_weights[:3]
    ]
    fourth = max(12, total_points - sum(first_three))
    quarter_scores = first_three + [fourth]

    difference = total_points - sum(quarter_scores)
    quarter_scores[-1] += difference
    return quarter_scores


def split_points_by_segments(
    total_points: int,
    segments: int,
    rng: random.Random,
) -> list[int]:
    """Split a score into evenly distributed segment totals."""
    if segments <= 0:
        return []

    if total_points <= 0:
        return [0] * segments

    weights = [max(0.08, 1.0 + rng.uniform(-0.35, 0.35)) for _ in range(segments)]
    weight_total = sum(weights)
    scaled = [total_points * weight / weight_total for weight in weights]
    points = [int(math.floor(value)) for value in scaled]
    remainder = total_points - sum(points)

    if remainder > 0:
        fractional_order = sorted(
            range(segments),
            key=lambda index: scaled[index] - points[index],
            reverse=True,
        )

        for index in range(remainder):
            points[fractional_order[index % segments]] += 1

    return points


def estimate_home_win_probability_from_margin(
    home_margin: int,
    remaining_minutes: int,
    projected_total: int,
) -> float:
    """Estimate live home win probability from score margin and time left."""
    remaining_fraction = max(remaining_minutes / FULL_GAME_SIMULATION_MINUTES, 0.0)
    scale = max(
        3.5,
        (max(projected_total, 1) / 16.0) * math.sqrt(remaining_fraction + 0.08),
    )
    return 1 / (1 + math.exp(-(home_margin / scale)))


def build_game_timeline_flow(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    seed: int,
) -> pd.DataFrame:
    """Build minute-by-minute cumulative score flow."""
    rng = random.Random(seed)
    home_quarters = split_score_by_quarter(home_score, rng)
    away_quarters = split_score_by_quarter(away_score, rng)
    rows = []
    home_cumulative = 0
    away_cumulative = 0
    projected_total = home_score + away_score

    for quarter_index, (home_points, away_points) in enumerate(
        zip(home_quarters, away_quarters),
        start=1,
    ):
        home_segments = split_points_by_segments(home_points, 12, rng)
        away_segments = split_points_by_segments(away_points, 12, rng)

        for minute_in_quarter, (home_segment, away_segment) in enumerate(
            zip(home_segments, away_segments),
            start=1,
        ):
            home_cumulative += home_segment
            away_cumulative += away_segment
            minute = ((quarter_index - 1) * 12) + minute_in_quarter
            remaining_minutes = FULL_GAME_SIMULATION_MINUTES - minute
            rows.append(
                {
                    "Minute": minute,
                    "Quarter": f"Q{quarter_index}",
                    "Minute in Quarter": minute_in_quarter,
                    home_team: home_cumulative,
                    away_team: away_cumulative,
                    f"{home_team} Minute": home_segment,
                    f"{away_team} Minute": away_segment,
                    "Home Margin": home_cumulative - away_cumulative,
                    "Home Win %": estimate_home_win_probability_from_margin(
                        home_margin=home_cumulative - away_cumulative,
                        remaining_minutes=remaining_minutes,
                        projected_total=projected_total,
                    )
                    * 100,
                }
            )

    return pd.DataFrame(rows)


def build_stable_matchup_seed(home_team: str, away_team: str, offset: int = 0) -> int:
    """Build a deterministic simulation seed for one matchup."""
    key = f"{home_team}|{away_team}"
    seed = sum((index + 1) * ord(character) for index, character in enumerate(key))
    return (seed + offset) % 999999 or 1


def build_game_simulation_state_key(home_team: str, away_team: str, seed: int) -> str:
    """Build a session-state key for one game simulation view."""
    key = f"{home_team}_{away_team}_{seed}".lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return f"game_sim_{key}"


def render_game_simulation_frame(
    flow: pd.DataFrame,
    home_team: str,
    away_team: str,
    minute_index: int,
) -> str:
    """Render one scoreboard frame for the game simulator."""
    total_minutes = len(flow)
    if minute_index <= 0:
        current_row = flow.iloc[0]
        minute_label = "Pre-game"
        quarter_label = "Tipoff"
        away_score = 0
        home_score = 0
        margin = 0
        progress_width = 0
    else:
        row_index = min(minute_index, total_minutes) - 1
        current_row = flow.iloc[row_index]
        minute_label = f"Minute {int(current_row['Minute'])} of {FULL_GAME_SIMULATION_MINUTES}"
        away_score = int(current_row[away_team])
        home_score = int(current_row[home_team])
        margin = home_score - away_score
        progress_width = int(round((int(current_row["Minute"]) / FULL_GAME_SIMULATION_MINUTES) * 100))

    quarter_label = str(current_row["Quarter"])
    away_color = html.escape(get_team_color(away_team))
    home_color = html.escape(get_team_color(home_team))

    leader = home_team if margin > 0 else away_team if margin < 0 else "Tied"
    leader_text = f"{leader} lead" if leader != "Tied" else "Tied game"

    return f"""
        <div class="game-sim-panel">
            <div class="game-sim-stage">
                <div class="game-sim-stage-head">
                    <div>
                        <div class="game-sim-stage-title">Game simulation</div>
                        <div class="game-sim-stage-note">{html.escape(minute_label)} / {html.escape(quarter_label)}</div>
                    </div>
                    <div class="game-sim-stage-minute">{html.escape(leader_text)}</div>
                </div>
                <div class="game-sim-board">
                    <div class="game-sim-team" style="--team-color: {away_color};">
                        <img class="game-sim-logo" src="{html.escape(get_logo_url(away_team))}" alt="{html.escape(away_team)} logo">
                        <div>
                            <div class="game-sim-name">{html.escape(away_team)}</div>
                            <div class="game-sim-meta">Away</div>
                        </div>
                        <div class="game-sim-score">{away_score}</div>
                    </div>
                    <div class="game-sim-center">
                        <span>{home_score - away_score:+d}</span>
                        margin
                    </div>
                    <div class="game-sim-team" style="--team-color: {home_color};">
                        <img class="game-sim-logo" src="{html.escape(get_logo_url(home_team))}" alt="{html.escape(home_team)} logo">
                        <div>
                            <div class="game-sim-name">{html.escape(home_team)}</div>
                            <div class="game-sim-meta">Home</div>
                        </div>
                        <div class="game-sim-score">{home_score}</div>
                    </div>
                </div>
                <div class="game-sim-progress">
                    <div class="game-sim-progress-fill" style="--progress-width: {progress_width}%"></div>
                </div>
            </div>
        </div>
    """


def build_game_score_projection(
    home_team: str,
    away_team: str,
    home_probability: float,
    team_adjustments: dict[str, float] | None = None,
    seed: int = 42,
    simulations: int = SCORE_SIMULATION_COUNT,
) -> dict:
    """Build final score, score range, and quarter flow projections."""
    rng = random.Random(seed)
    samples = []

    for _ in range(simulations):
        home_score, away_score = sample_projected_score(
            home_team=home_team,
            away_team=away_team,
            home_probability=home_probability,
            team_adjustments=team_adjustments,
            rng=rng,
        )
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

    flow = build_game_timeline_flow(
        home_team=home_team,
        away_team=away_team,
        home_score=projected_home_score,
        away_score=projected_away_score,
        seed=seed + 17,
    )
    expected = estimate_matchup_expected_scores(
        home_team=home_team,
        away_team=away_team,
        home_probability=home_probability,
        team_adjustments=team_adjustments,
    )

    return {
        "projected_home_score": projected_home_score,
        "projected_away_score": projected_away_score,
        "projected_total": projected_home_score + projected_away_score,
        "projected_winner": home_team
        if projected_home_score > projected_away_score
        else away_team,
        "home_win_rate": float(sample_df["Home Win"].mean()),
        "total_range": (
            int(round(sample_df["Total"].quantile(0.1))),
            int(round(sample_df["Total"].quantile(0.9))),
        ),
        "margin_range": (
            int(round(sample_df["Home Margin"].quantile(0.1))),
            int(round(sample_df["Home Margin"].quantile(0.9))),
        ),
        "expected_pace": expected["expected_pace"],
        "flow": flow,
        "samples": sample_df,
    }


def render_score_projection_card(
    home_team: str,
    away_team: str,
    projection: dict,
    label: str = "Score simulation",
) -> None:
    """Render a compact projected scoreboard."""
    home_score = int(projection["projected_home_score"])
    away_score = int(projection["projected_away_score"])
    winner = str(projection["projected_winner"])
    total_low, total_high = projection["total_range"]
    margin_low, margin_high = projection["margin_range"]
    signals = [
        f"Total {total_low}-{total_high}",
        f"Projected home margin {margin_low:+d} to {margin_high:+d}",
        f"Pace {float(projection['expected_pace']):.1f}",
        f"{SCORE_SIMULATION_COUNT:,} sims",
    ]
    signal_html = "".join(
        f'<span class="signal-pill">{html.escape(signal)}</span>'
        for signal in signals
    )

    st.html(
        f"""
        <div class="score-sim-card">
            <div class="score-sim-title">{html.escape(label)}</div>
            <div class="score-sim-note">Home score is the projected points for the team listed on the right. Home margin means home points minus away points, so a positive number favors the home team.</div>
            <div class="score-sim-board">
                <div class="score-sim-team" style="--team-color: {html.escape(get_team_color(away_team))};">
                    <img class="score-sim-logo" src="{html.escape(get_logo_url(away_team))}" alt="{html.escape(away_team)} logo">
                    <div>
                        <div class="score-sim-name">{html.escape(away_team)}</div>
                        <div class="score-sim-role">Away</div>
                    </div>
                    <div class="score-sim-points">{away_score}</div>
                </div>
                <div class="score-sim-center">AT</div>
                <div class="score-sim-team" style="--team-color: {html.escape(get_team_color(home_team))};">
                    <img class="score-sim-logo" src="{html.escape(get_logo_url(home_team))}" alt="{html.escape(home_team)} logo">
                    <div>
                        <div class="score-sim-name">{html.escape(home_team)}</div>
                        <div class="score-sim-role">Home team / projected winner: {html.escape(winner)}</div>
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
    team_adjustments: dict[str, float] | None,
    seed: int,
    label: str = "Score simulation",
) -> dict:
    """Render final score and full-game playback for one matchup."""
    projection = build_game_score_projection(
        home_team=home_team,
        away_team=away_team,
        home_probability=home_probability,
        team_adjustments=team_adjustments,
        seed=seed,
    )
    render_score_projection_card(
        home_team=home_team,
        away_team=away_team,
        projection=projection,
        label=label,
    )

    render_matchup_comparison_panel(
        team_a=away_team,
        team_b=home_team,
        title="Star comparison",
        key_prefix=f"game_{away_team}_{home_team}_{seed}",
    )

    flow = projection["flow"]
    state_key = build_game_simulation_state_key(home_team, away_team, seed)
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "minute_index": 0,
            "is_playing": False,
        }

    state = st.session_state[state_key]

    control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
    with control_col1:
        play_clicked = st.button(
            "Play",
            type="primary",
            width="stretch",
            key=f"{state_key}_play",
        )
    with control_col2:
        reset_clicked = st.button(
            "Reset",
            width="stretch",
            key=f"{state_key}_reset",
        )
    with control_col3:
        st.caption("Press Play to run the projected game minute by minute.")

    if reset_clicked:
        state["minute_index"] = 0
        state["is_playing"] = False

    if play_clicked:
        state["minute_index"] = 0
        state["is_playing"] = True

    frame_placeholder = st.empty()
    if state["is_playing"]:
        start_index = max(1, int(state["minute_index"]) + 1)
        progress_placeholder = st.empty()

        for minute_index in range(start_index, len(flow) + 1):
            state["minute_index"] = minute_index
            frame_placeholder.markdown(
                render_game_simulation_frame(
                    flow=flow,
                    home_team=home_team,
                    away_team=away_team,
                    minute_index=minute_index,
                ),
                unsafe_allow_html=True,
            )
            progress_placeholder.progress(int(round(minute_index / len(flow) * 100)))
            time.sleep(GAME_SIMULATION_FRAME_DELAY_SECONDS)

        state["is_playing"] = False
    else:
        frame_placeholder.markdown(
            render_game_simulation_frame(
                flow=flow,
                home_team=home_team,
                away_team=away_team,
                minute_index=int(state["minute_index"]),
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Minute-by-minute score table", expanded=False):
        st.dataframe(flow, width="stretch", hide_index=True)

    return projection


def parse_series_result_label(
    result_label: str | None,
    teams: list[str],
) -> tuple[str, int] | None:
    """Parse labels like 'Boston Celtics in 7'."""
    if not result_label:
        return None

    for team in teams:
        match = re.match(
            rf"^{re.escape(team)}\s+in\s+([4-7])$",
            str(result_label).strip(),
        )

        if match:
            return team, int(match.group(1))

    return None


def get_scheduled_matchup_for_series_game(
    higher_seed_team: str,
    lower_seed_team: str,
    game_number: int,
) -> tuple[str, str]:
    """Return home and away teams for one standard best-of-7 game."""
    home_team = (
        higher_seed_team
        if HOME_COURT_SCHEDULE[game_number - 1] == "higher"
        else lower_seed_team
    )
    away_team = lower_seed_team if home_team == higher_seed_team else higher_seed_team
    return home_team, away_team


def get_team_win_probability_for_scheduled_game(
    target_team: str,
    home_team: str,
    away_team: str,
    probability_cache: dict[tuple[str, str], float],
) -> float:
    """Return target team's win probability for a scheduled series game."""
    home_probability = get_cached_home_win_probability(
        home_team=home_team,
        away_team=away_team,
        probability_cache=probability_cache,
    )

    return home_probability if target_team == home_team else 1 - home_probability


def build_targeted_series_game_winners(
    higher_seed_team: str,
    lower_seed_team: str,
    probability_cache: dict[tuple[str, str], float],
    target_result: str | None,
) -> list[str] | None:
    """Build game winners that match the most-likely series result."""
    parsed_result = parse_series_result_label(
        target_result,
        [higher_seed_team, lower_seed_team],
    )

    if parsed_result is None:
        return None

    target_winner, series_length = parsed_result
    target_loser = (
        lower_seed_team if target_winner == higher_seed_team else higher_seed_team
    )
    target_losses = series_length - 4

    if target_losses == 0:
        return [target_winner] * 4

    candidate_losses = []

    for game_number in range(1, series_length):
        home_team, away_team = get_scheduled_matchup_for_series_game(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            game_number=game_number,
        )
        target_win_probability = get_team_win_probability_for_scheduled_game(
            target_team=target_winner,
            home_team=home_team,
            away_team=away_team,
            probability_cache=probability_cache,
        )
        candidate_losses.append((target_win_probability, game_number))

    loss_games = {
        game_number
        for _, game_number in sorted(candidate_losses)[:target_losses]
    }
    winners = []

    for game_number in range(1, series_length + 1):
        if game_number == series_length:
            winners.append(target_winner)
        elif game_number in loss_games:
            winners.append(target_loser)
        else:
            winners.append(target_winner)

    return winners


def build_series_score_simulation_path(
    higher_seed_team: str,
    lower_seed_team: str,
    probability_cache: dict[tuple[str, str], float],
    team_adjustments: dict[str, float] | None,
    seed: int,
    target_result: str | None = None,
) -> pd.DataFrame:
    """Build a score path through a best-of-7 series."""
    rng = random.Random(seed)
    wins = {higher_seed_team: 0, lower_seed_team: 0}
    rows = []
    targeted_winners = build_targeted_series_game_winners(
        higher_seed_team=higher_seed_team,
        lower_seed_team=lower_seed_team,
        probability_cache=probability_cache,
        target_result=target_result,
    )

    for game_number, home_court in enumerate(HOME_COURT_SCHEDULE, start=1):
        home_team, away_team = get_scheduled_matchup_for_series_game(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            game_number=game_number,
        )
        home_probability = get_cached_home_win_probability(
            home_team=home_team,
            away_team=away_team,
            probability_cache=probability_cache,
        )

        if targeted_winners is None:
            home_score, away_score = sample_projected_score(
                home_team=home_team,
                away_team=away_team,
                home_probability=home_probability,
                team_adjustments=team_adjustments,
                rng=rng,
            )
        else:
            target_winner = targeted_winners[game_number - 1]
            home_score, away_score = sample_projected_score_for_winner(
                home_team=home_team,
                away_team=away_team,
                home_probability=home_probability,
                team_adjustments=team_adjustments,
                target_winner=target_winner,
                rng=rng,
            )

        winner = home_team if home_score > away_score else away_team
        wins[winner] += 1
        higher_score = home_score if home_team == higher_seed_team else away_score
        lower_score = home_score if home_team == lower_seed_team else away_score

        rows.append(
            {
                "Game": game_number,
                "Home": home_team,
                "Away": away_team,
                "Home Score": home_score,
                "Away Score": away_score,
                "Winner": winner,
                f"{higher_seed_team} Score": higher_score,
                f"{lower_seed_team} Score": lower_score,
                f"{higher_seed_team} Wins": wins[higher_seed_team],
                f"{lower_seed_team} Wins": wins[lower_seed_team],
                "Series": (
                    f"{higher_seed_team} {wins[higher_seed_team]} - "
                    f"{lower_seed_team} {wins[lower_seed_team]}"
                ),
            }
        )

        if wins[higher_seed_team] >= 4 or wins[lower_seed_team] >= 4:
            break

    path = pd.DataFrame(rows)
    path.attrs["target_result"] = target_result
    path.attrs["is_targeted"] = targeted_winners is not None
    return path


def render_series_score_simulation_path(
    path: pd.DataFrame,
    higher_seed_team: str,
    lower_seed_team: str,
) -> None:
    """Render one score path through a playoff series."""
    if path.empty:
        return

    final_row = path.iloc[-1]
    series_winner = (
        higher_seed_team
        if int(final_row[f"{higher_seed_team} Wins"]) >= 4
        else lower_seed_team
    )
    is_targeted = bool(path.attrs.get("is_targeted"))
    target_result = path.attrs.get("target_result")
    title = "Most-likely score path" if is_targeted else "Score simulation path"
    note = (
        f"Projected scores aligned to {target_result}."
        if is_targeted and target_result
        else f"One simulated scoring path: {series_winner} in {len(path)}."
    )
    render_section_kicker(title, note)

    row_html = []

    for _, row in path.iterrows():
        away_team = str(row["Away"])
        home_team = str(row["Home"])
        score_line = (
            f"{away_team} {int(row['Away Score'])} at "
            f"{home_team} {int(row['Home Score'])}"
        )
        row_html.append(
            f"""
            <div class="series-score-row">
                <div class="series-score-game">Game {int(row["Game"])}</div>
                <div>
                    <div class="series-score-matchup">{html.escape(score_line)}</div>
                    <div class="series-score-meta">{html.escape(str(row["Series"]))}</div>
                </div>
                <div class="impact-score">{html.escape(str(row["Winner"]))}</div>
            </div>
            """
        )

    st.html("".join(row_html))

    score_columns = [f"{higher_seed_team} Score", f"{lower_seed_team} Score"]
    st.line_chart(path.set_index("Game")[score_columns])

    with st.expander("Series score table", expanded=False):
        st.dataframe(path, width="stretch", hide_index=True)


def build_matchup_explanation(
    team_a: str,
    team_b: str,
    strength: pd.DataFrame,
) -> pd.DataFrame:
    """Build a weighted comparison table for two teams."""
    row_a = get_team_strength_row(team_a, strength)
    row_b = get_team_strength_row(team_b, strength)

    comparison_rows = [
        {
            "Metric": "Power Rating",
            "Category": "Overall strength",
            "Weight": 3,
            team_a: float(row_a["ELO"]),
            team_b: float(row_b["ELO"]),
            "Edge": team_a if float(row_a["ELO"]) > float(row_b["ELO"]) else team_b,
        },
        {
            "Metric": "Season Win %",
            "Category": "Season results",
            "Weight": 3,
            team_a: float(row_a["SEASON_WIN_PCT"]),
            team_b: float(row_b["SEASON_WIN_PCT"]),
            "Edge": team_a
            if float(row_a["SEASON_WIN_PCT"]) > float(row_b["SEASON_WIN_PCT"])
            else team_b,
        },
        {
            "Metric": "Season Avg Plus/Minus",
            "Category": "Season results",
            "Weight": 3,
            team_a: float(row_a["SEASON_AVG_PLUS_MINUS"]),
            team_b: float(row_b["SEASON_AVG_PLUS_MINUS"]),
            "Edge": team_a
            if float(row_a["SEASON_AVG_PLUS_MINUS"])
            > float(row_b["SEASON_AVG_PLUS_MINUS"])
            else team_b,
        },
        {
            "Metric": "Last 10 Win %",
            "Category": "Recent form",
            "Weight": 2,
            team_a: float(row_a["ROLLING_WIN_PCT_10"]),
            team_b: float(row_b["ROLLING_WIN_PCT_10"]),
            "Edge": team_a
            if float(row_a["ROLLING_WIN_PCT_10"]) > float(row_b["ROLLING_WIN_PCT_10"])
            else team_b,
        },
        {
            "Metric": "Last 10 Plus/Minus",
            "Category": "Recent form",
            "Weight": 2,
            team_a: float(row_a["ROLLING_PLUS_MINUS_10"]),
            team_b: float(row_b["ROLLING_PLUS_MINUS_10"]),
            "Edge": team_a
            if float(row_a["ROLLING_PLUS_MINUS_10"])
            > float(row_b["ROLLING_PLUS_MINUS_10"])
            else team_b,
        },
        {
            "Metric": "Last 5 Win %",
            "Category": "Short-term form",
            "Weight": 1,
            team_a: float(row_a["ROLLING_WIN_PCT_5"]),
            team_b: float(row_b["ROLLING_WIN_PCT_5"]),
            "Edge": team_a
            if float(row_a["ROLLING_WIN_PCT_5"]) > float(row_b["ROLLING_WIN_PCT_5"])
            else team_b,
        },
        {
            "Metric": "Last 5 Plus/Minus",
            "Category": "Short-term form",
            "Weight": 1,
            team_a: float(row_a["ROLLING_PLUS_MINUS_5"]),
            team_b: float(row_b["ROLLING_PLUS_MINUS_5"]),
            "Edge": team_a
            if float(row_a["ROLLING_PLUS_MINUS_5"])
            > float(row_b["ROLLING_PLUS_MINUS_5"])
            else team_b,
        },
    ]

    explanation = pd.DataFrame(comparison_rows)
    explanation["Weighted Edge Points"] = explanation["Weight"]
    return explanation


def get_weighted_profile_summary(
    team_a: str,
    team_b: str,
    explanation: pd.DataFrame,
) -> dict:
    """Summarize weighted matchup edges."""
    team_a_points = int(
        explanation.loc[explanation["Edge"] == team_a, "Weighted Edge Points"].sum()
    )
    team_b_points = int(
        explanation.loc[explanation["Edge"] == team_b, "Weighted Edge Points"].sum()
    )

    team_a_edges = int((explanation["Edge"] == team_a).sum())
    team_b_edges = int((explanation["Edge"] == team_b).sum())

    if team_a_points > team_b_points:
        weighted_leader = team_a
        weighted_points = team_a_points
        weighted_edges = team_a_edges
    elif team_b_points > team_a_points:
        weighted_leader = team_b
        weighted_points = team_b_points
        weighted_edges = team_b_edges
    else:
        weighted_leader = "Even"
        weighted_points = team_a_points
        weighted_edges = team_a_edges

    return {
        "team_a_points": team_a_points,
        "team_b_points": team_b_points,
        "team_a_edges": team_a_edges,
        "team_b_edges": team_b_edges,
        "weighted_leader": weighted_leader,
        "weighted_points": weighted_points,
        "weighted_edges": weighted_edges,
    }


def render_prediction_summary(
    predicted_winner: str,
    winner_probability: float,
    team_a: str,
    team_b: str,
    explanation: pd.DataFrame,
) -> None:
    """Render a clear explanation summary."""
    summary = get_weighted_profile_summary(team_a, team_b, explanation)
    weighted_leader = summary["weighted_leader"]

    render_section_kicker("Prediction summary")

    if weighted_leader == "Even":
        summary_text = (
            f"The simulation gives <b>{html.escape(predicted_winner)}</b> a "
            f"<b>{winner_probability:.1%}</b> chance to win, while the weighted "
            "team-strength profile is almost even."
        )
    elif weighted_leader == predicted_winner:
        summary_text = (
            f"The simulation favors <b>{html.escape(predicted_winner)}</b> at "
            f"<b>{winner_probability:.1%}</b>. The weighted strength table agrees: "
            f"<b>{html.escape(weighted_leader)}</b> leads "
            f"<b>{summary['weighted_edges']} of {len(explanation)}</b> categories, "
            f"worth <b>{summary['weighted_points']} weighted points</b>."
        )
    else:
        summary_text = (
            f"The simulation favors <b>{html.escape(predicted_winner)}</b> at "
            f"<b>{winner_probability:.1%}</b>, but the weighted strength table leans "
            f"toward <b>{html.escape(weighted_leader)}</b>. That can happen because "
            "the final prediction uses calibrated model probability, Elo, home-court "
            "order, simulation outcomes, and stat-based player availability adjustments, "
            "not only category counts."
        )

    st.html(
        f"""
        <div class="summary-box">
            {summary_text}
        </div>
        """
    )


def render_matchup_signal_cards(
    home_team: str,
    away_team: str,
    team_adjustments: dict[str, float] | None = None,
    playoff_context: dict[str, float] | None = None,
) -> None:
    """Render a compact set of matchup signals."""
    snapshot = build_matchup_feature_snapshot(
        home_team=home_team,
        away_team=away_team,
        team_adjustments=team_adjustments,
        playoff_context=playoff_context,
    )

    render_dashboard_cards(snapshot["signals"])

    if snapshot["top_reasons"]:
        reason_lines = "".join(
            f'<div class="compact-note">{html.escape(reason)}</div>'
            for reason in snapshot["top_reasons"]
        )
        st.html(
            f"""
            <div class="summary-box">
                <div class="dashboard-label">Top reasons</div>
                {reason_lines}
            </div>
            """
        )


def render_matchup_explanation(
    team_a: str,
    team_b: str,
    predicted_winner: str,
    winner_probability: float,
    playoff_context: dict[str, float] | None = None,
    team_adjustments: dict[str, float] | None = None,
) -> None:
    """Render matchup explanation for two teams."""
    strength = load_team_strength()
    explanation = build_matchup_explanation(team_a, team_b, strength)
    summary = get_weighted_profile_summary(team_a, team_b, explanation)

    with st.expander("Why this prediction?", expanded=False):
        render_prediction_summary(
            predicted_winner=predicted_winner,
            winner_probability=winner_probability,
            team_a=team_a,
            team_b=team_b,
            explanation=explanation,
        )
        render_matchup_signal_cards(
            home_team=team_a,
            away_team=team_b,
            team_adjustments=team_adjustments,
            playoff_context=playoff_context,
        )

        render_section_kicker("Team profile")

        if summary["weighted_leader"] == "Even":
            st.html('<div class="compact-note">The weighted profile is nearly even.</div>')
        else:
            st.html(
                f"""
                <div class="compact-note">
                    <b>{html.escape(summary["weighted_leader"])}</b> leads by
                    <b>{summary["weighted_points"]} weighted points</b>.
                </div>
                """
            )

        formatted = explanation.copy()

        for column in [team_a, team_b]:
            formatted[column] = formatted.apply(
                lambda row: f"{row[column]:.3f}"
                if "Win %" in row["Metric"]
                else f"{row[column]:.1f}",
                axis=1,
            )

        formatted = formatted[
            [
                "Metric",
                "Category",
                "Weight",
                team_a,
                team_b,
                "Edge",
                "Weighted Edge Points",
            ]
        ]

        st.dataframe(formatted, width="stretch")

        render_key_model_edges(
            team_a,
            team_b,
            playoff_context=playoff_context,
            team_adjustments=team_adjustments,
        )


def format_feature_label(feature_column: str) -> str:
    """Format a model feature name for display."""
    label = feature_column

    if label.startswith("DIFF_"):
        label = label.replace("DIFF_", "")

    label = label.replace("ROLLING_", "Last ")
    label = label.replace("SEASON_AVG_", "Season Avg ")
    label = label.replace("PREV_SEASON_", "Previous Season ")
    label = label.replace("HOME_ELO_WIN_PROB", "Home Elo Win Probability")
    label = label.replace("_", " ").title()
    label = label.replace("Pts", "PTS")
    label = label.replace("Fg", "FG")
    label = label.replace("Ft", "FT")
    label = label.replace("Elo", "Elo")
    label = label.replace("Mpg", "MPG")
    return label


def lower_is_better_feature(feature_column: str) -> bool:
    """Return whether lower values are better for a model feature."""
    return any(part in feature_column for part in LOWER_IS_BETTER_FEATURE_PARTS)


def build_key_model_edges(
    home_team: str,
    away_team: str,
    playoff_context: dict[str, float] | None = None,
    team_adjustments: dict[str, float] | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    """Build largest standardized model feature edges for a matchup."""
    model_bundle = load_model_bundle()
    strength = load_team_strength()
    feature_columns = model_bundle["feature_columns"]
    prediction_row = build_prediction_row_from_strength(
        home_team=home_team,
        away_team=away_team,
        strength=strength,
        feature_columns=feature_columns,
        playoff_context=playoff_context,
        team_adjustments=team_adjustments,
    )

    try:
        historical_features = load_model_features()
    except FileNotFoundError:
        historical_features = pd.DataFrame()

    rows = []

    for feature_column in feature_columns:
        value = float(prediction_row.iloc[0][feature_column])

        if value == 0:
            continue

        if not historical_features.empty and feature_column in historical_features:
            std = float(historical_features[feature_column].std())
        else:
            std = 1.0

        if std == 0 or pd.isna(std):
            std = 1.0

        standardized_size = abs(value / std)

        if feature_column in PLAYOFF_CONTEXT_MODEL_FEATURES:
            edge = "Series context"
        elif feature_column == "HOME_ELO_WIN_PROB":
            edge = home_team if value >= 0.5 else away_team
        elif lower_is_better_feature(feature_column):
            edge = home_team if value < 0 else away_team
        else:
            edge = home_team if value > 0 else away_team

        rows.append(
            {
                "Feature": format_feature_label(feature_column),
                "Raw Value": value,
                "Relative Size": standardized_size,
                "Edge": edge,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("Relative Size", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def render_key_model_edges(
    home_team: str,
    away_team: str,
    playoff_context: dict[str, float] | None = None,
    team_adjustments: dict[str, float] | None = None,
) -> None:
    """Render the most meaningful model-feature edges for a matchup."""
    edges = build_key_model_edges(
        home_team=home_team,
        away_team=away_team,
        playoff_context=playoff_context,
        team_adjustments=team_adjustments,
    )

    if edges.empty:
        return

    render_section_kicker("Key betting edges")

    display = edges.copy()
    display["Raw Value"] = display["Raw Value"].map("{:.3f}".format)
    display["Relative Size"] = display["Relative Size"].map("{:.2f}".format)
    st.dataframe(display, width="stretch")


def create_prediction_report(
    title: str,
    predicted_winner: str,
    probabilities: dict[str, float],
    confidence: str,
    most_likely_result: str | None = None,
) -> str:
    """Create a downloadable text report."""
    lines = [
        title,
        "=" * len(title),
        "",
        f"Predicted winner: {predicted_winner}",
        f"Confidence: {confidence}",
        "",
        "Probabilities:",
    ]

    for team, probability in probabilities.items():
        lines.append(f"- {team}: {probability:.1%}")

    if most_likely_result:
        lines.extend(["", f"Most likely result: {most_likely_result}"])

    lines.extend(
        [
            "",
            "Pricing note:",
            "This educational pick engine uses current team strength, recent form, power ratings, "
            "calibrated probabilities, simulation logic, and the latest saved official "
            "NBA injury report. It does not include betting odds, travel, matchup-specific "
            "defense, or injury-report changes made after the report was refreshed.",
        ]
    )

    return "\n".join(lines)


def render_report_download(
    report_text: str,
    file_name: str,
    key: str,
) -> None:
    """Render download button for a text report."""
    st.download_button(
        label="Download prediction report",
        data=report_text.encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        key=key,
    )


def simulate_single_series(
    higher_seed_team: str,
    lower_seed_team: str,
    probability_cache: dict[tuple[str, str], float],
    rng: random.Random,
) -> tuple[str, int, int]:
    """Simulate one best-of-7 playoff series."""
    higher_seed_wins = 0
    lower_seed_wins = 0

    for home_court in HOME_COURT_SCHEDULE:
        if home_court == "higher":
            home_team = higher_seed_team
            away_team = lower_seed_team
        else:
            home_team = lower_seed_team
            away_team = higher_seed_team

        home_win_probability = get_cached_home_win_probability(
            home_team=home_team,
            away_team=away_team,
            probability_cache=probability_cache,
        )

        home_team_wins = rng.random() < home_win_probability

        if home_team_wins and home_team == higher_seed_team:
            higher_seed_wins += 1
        elif home_team_wins and home_team == lower_seed_team:
            lower_seed_wins += 1
        elif not home_team_wins and away_team == higher_seed_team:
            higher_seed_wins += 1
        else:
            lower_seed_wins += 1

        if higher_seed_wins == 4:
            return higher_seed_team, higher_seed_wins, lower_seed_wins

        if lower_seed_wins == 4:
            return lower_seed_team, higher_seed_wins, lower_seed_wins

    raise RuntimeError("Series simulation ended without a winner.")


def simulate_series(
    higher_seed_team: str,
    lower_seed_team: str,
    simulations: int,
    seed: int,
    team_adjustments: dict[str, float] | None = None,
) -> dict:
    """Run many best-of-7 simulations."""
    strength = load_team_strength()
    model_bundle = load_model_bundle()
    rng = random.Random(seed)

    probability_cache = build_matchup_probability_cache(
        teams=[higher_seed_team, lower_seed_team],
        strength=strength,
        model_bundle=model_bundle,
        team_adjustments=team_adjustments,
    )

    winners = []
    result_counts = Counter()

    for _ in range(simulations):
        winner, higher_wins, lower_wins = simulate_single_series(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            probability_cache=probability_cache,
            rng=rng,
        )

        winners.append(winner)
        result_counts[f"{winner} in {higher_wins + lower_wins}"] += 1

    winner_counts = Counter(winners)

    higher_seed_probability = winner_counts[higher_seed_team] / simulations
    lower_seed_probability = winner_counts[lower_seed_team] / simulations
    most_likely_result, most_likely_count = result_counts.most_common(1)[0]

    result_table = pd.DataFrame(
        [
            {
                "Result": result,
                "Count": count,
                "Probability": count / simulations,
            }
            for result, count in result_counts.most_common()
        ]
    )

    return {
        "higher_seed_probability": higher_seed_probability,
        "lower_seed_probability": lower_seed_probability,
        "most_likely_result": most_likely_result,
        "most_likely_probability": most_likely_count / simulations,
        "result_table": result_table,
        "probability_cache": probability_cache,
    }


def build_playoff_context(
    home_team: str,
    away_team: str,
    game_number: int,
    home_series_wins: int,
    away_series_wins: int,
) -> dict[str, float]:
    """Build model playoff context for one future playoff game."""
    return {
        "IS_PLAYOFF_GAME": 1.0,
        "PLAYOFF_SERIES_GAME_NUMBER": float(game_number),
        "DIFF_SERIES_WINS_ENTERING": float(home_series_wins - away_series_wins),
        "HOME_SERIES_WINS_ENTERING": float(home_series_wins),
        "AWAY_SERIES_WINS_ENTERING": float(away_series_wins),
        "HOME_FACING_ELIMINATION": float(away_series_wins == 3),
        "AWAY_FACING_ELIMINATION": float(home_series_wins == 3),
        "HOME_CAN_CLINCH_SERIES": float(home_series_wins == 3),
        "AWAY_CAN_CLINCH_SERIES": float(away_series_wins == 3),
    }


def get_latest_playoff_season(games: pd.DataFrame) -> str | None:
    """Return the latest season that has playoff game rows."""
    if "SEASON_TYPE" not in games.columns:
        return None

    playoff_games = games[games["SEASON_TYPE"].eq("Playoffs")]

    if playoff_games.empty:
        return None

    return str(sorted(playoff_games["SEASON"].unique())[-1])


def build_completed_playoff_game_results(season: str | None = None) -> pd.DataFrame:
    """Return one row per completed playoff game."""
    games = load_raw_games()

    if "SEASON_TYPE" not in games.columns:
        return pd.DataFrame()

    if season is None:
        season = get_latest_playoff_season(games)

    if season is None:
        return pd.DataFrame()

    playoff_games = games[
        games["SEASON"].eq(season) & games["SEASON_TYPE"].eq("Playoffs")
    ].copy()

    if playoff_games.empty:
        return pd.DataFrame()

    home_games = playoff_games[playoff_games["MATCHUP"].str.contains("vs.")][
        ["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS", "WL"]
    ].rename(
        columns={
            "TEAM_NAME": "Home Team",
            "PTS": "Home Points",
            "WL": "Home Result",
        }
    )
    away_games = playoff_games[playoff_games["MATCHUP"].str.contains("@")][
        ["GAME_ID", "TEAM_NAME", "PTS"]
    ].rename(
        columns={
            "TEAM_NAME": "Away Team",
            "PTS": "Away Points",
        }
    )

    results = home_games.merge(away_games, on="GAME_ID", how="inner")
    results["Winner"] = results.apply(
        lambda row: row["Home Team"]
        if row["Home Result"] == "W"
        else row["Away Team"],
        axis=1,
    )
    results["Loser"] = results.apply(
        lambda row: row["Away Team"]
        if row["Home Result"] == "W"
        else row["Home Team"],
        axis=1,
    )
    results["Series Key"] = results.apply(
        lambda row: " vs ".join(sorted([row["Home Team"], row["Away Team"]])),
        axis=1,
    )

    return results.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def build_final_game_results_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """Build completed playoff result rows from normalized game rows."""
    if games.empty:
        return pd.DataFrame()

    rows = []

    for _, game in games.iterrows():
        status = str(game.get("Status", "")).strip().lower()
        status_code = coerce_game_status_code(game.get("Game Status Code", 0))

        if status_code < 3 and "final" not in status:
            continue

        home_team = str(game.get("Home Team", "")).strip()
        away_team = str(game.get("Away Team", "")).strip()
        home_points = normalize_score_value(game.get("Home Score", ""))
        away_points = normalize_score_value(game.get("Away Score", ""))

        if not home_team or not away_team or home_points == "" or away_points == "":
            continue

        if not is_known_team_name(home_team) or not is_known_team_name(away_team):
            continue

        try:
            home_points = int(float(home_points))
            away_points = int(float(away_points))
        except (TypeError, ValueError):
            continue

        if home_points == away_points:
            continue

        game_date_value = game.get("Game Date")

        if game_date_value is None or pd.isna(game_date_value):
            game_date_value = game.get("Game DateTime")

        game_date = pd.to_datetime(game_date_value, errors="coerce")

        if pd.isna(game_date):
            game_date = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).normalize()

        winner = home_team if home_points > away_points else away_team
        loser = away_team if winner == home_team else home_team

        rows.append(
            {
                "GAME_ID": str(game.get("Game ID", "")).strip()
                or str(game.get("ESPN Game ID", "")).strip(),
                "GAME_DATE": game_date,
                "Home Team": home_team,
                "Home Points": home_points,
                "Home Result": "W" if winner == home_team else "L",
                "Away Team": away_team,
                "Away Points": away_points,
                "Winner": winner,
                "Loser": loser,
                "Series Key": " vs ".join(sorted([home_team, away_team])),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_live_final_game_results() -> pd.DataFrame:
    """Build completed game result rows from today's live feed before local refresh."""
    try:
        games = load_today_games()
    except Exception:
        return pd.DataFrame()

    return build_final_game_results_from_games(games)


def build_schedule_final_game_results(season: str | None = None) -> pd.DataFrame:
    """Build completed game result rows from the NBA schedule feed."""
    try:
        games = load_nba_schedule_games(season)
    except Exception:
        return pd.DataFrame()

    games = filter_schedule_playoff_rows(games)
    return build_final_game_results_from_games(games)


def normalize_game_id_for_merge(value: object) -> str:
    """Normalize NBA game IDs across feeds that disagree on leading zeroes."""
    text = str(value).strip()
    normalized = text.lstrip("0")
    return normalized or text


def build_completed_playoff_game_results_with_live(
    season: str | None = None,
) -> pd.DataFrame:
    """Return completed playoff results plus schedule/live finals not yet collected."""
    results = build_completed_playoff_game_results(season)
    schedule_results = build_schedule_final_game_results(season)
    live_results = build_live_final_game_results()
    frames = [
        frame
        for frame in [results, schedule_results, live_results]
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["GAME_ID_MERGE_KEY"] = combined["GAME_ID"].map(normalize_game_id_for_merge)
    combined = combined.drop_duplicates(subset=["GAME_ID_MERGE_KEY"], keep="first")
    combined = combined.drop(columns=["GAME_ID_MERGE_KEY"])
    return combined.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def build_current_playoff_series_states(season: str | None = None) -> list[dict]:
    """Build current playoff series states from completed game logs."""
    results = build_completed_playoff_game_results_with_live(season)

    if results.empty:
        return []

    states = []

    for series_key, games in results.groupby("Series Key", sort=False):
        games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        first_game = games.iloc[0]
        home_court_team = str(first_game["Home Team"])
        other_team = (
            str(first_game["Away Team"])
            if str(first_game["Away Team"]) != home_court_team
            else str(first_game["Home Team"])
        )
        wins = {home_court_team: 0, other_team: 0}
        game_rows = []

        for game_number, (_, game) in enumerate(games.iterrows(), start=1):
            winner = str(game["Winner"])
            wins[winner] = wins.get(winner, 0) + 1
            game_rows.append(
                {
                    "Game": game_number,
                    "Date": pd.to_datetime(game["GAME_DATE"]).date().isoformat(),
                    "Home": game["Home Team"],
                    "Away": game["Away Team"],
                    "Score": f"{int(game['Away Points'])}-{int(game['Home Points'])}",
                    "Winner": winner,
                }
            )

        teams = list(wins.keys())
        leader = max(teams, key=lambda team: wins[team])
        completed = wins[leader] >= 4
        current_streak = 0
        last_winner = None

        for row in reversed(game_rows):
            if last_winner is None:
                last_winner = row["Winner"]
                current_streak = 1
            elif row["Winner"] == last_winner:
                current_streak += 1
            else:
                break

        states.append(
            {
                "series_key": series_key,
                "home_court_team": home_court_team,
                "other_team": other_team,
                "teams": teams,
                "wins": wins,
                "leader": leader,
                "games_played": len(games),
                "next_game_number": len(games) + 1,
                "completed": completed,
                "game_rows": game_rows,
                "last_winner": last_winner,
                "current_streak": current_streak,
            }
        )

    return sorted(
        states,
        key=lambda state: (state["completed"], -state["games_played"], state["series_key"]),
    )


def normalize_schedule_fallback_games(games: pd.DataFrame) -> pd.DataFrame:
    """Normalize injury-report schedule rows into the schedule schema."""
    if games.empty:
        return pd.DataFrame()

    rows = []

    for _, game in games.iterrows():
        game_date = pd.to_datetime(game.get("Game Date"), errors="coerce")
        rows.append(
            {
                "Game ID": str(game.get("Game ID", "")),
                "Game Date": game_date,
                "Game DateTime": pd.NaT,
                "Game Time": str(game.get("Game Time", "")).strip(),
                "Status": str(game.get("Status", "Scheduled")).strip(),
                "Home Team": str(game.get("Home Team", "")),
                "Away Team": str(game.get("Away Team", "")),
                "Series Game Number": None,
                "Game Label": "",
                "Game SubLabel": "",
                "If Necessary": False,
                "Source": str(game.get("Source", "Injury report schedule")),
            }
        )

    return pd.DataFrame(rows)


def get_schedule_for_series(state: dict) -> pd.DataFrame:
    """Return remaining scheduled games for a current playoff series."""
    if state["completed"]:
        return pd.DataFrame()

    try:
        season = get_latest_playoff_season(load_raw_games())
    except Exception:
        season = None

    schedule = load_nba_schedule_games(season)

    if schedule.empty:
        schedule = normalize_schedule_fallback_games(load_injury_report_schedule_games())

    if schedule.empty:
        return pd.DataFrame()

    teams = set(state["teams"])
    selected = schedule[
        schedule.apply(
            lambda row: {row["Home Team"], row["Away Team"]} == teams,
            axis=1,
        )
    ].copy()

    if selected.empty:
        return pd.DataFrame()

    selected["Series Game Number"] = selected["Series Game Number"].apply(
        lambda value: int(value) if value is not None and not pd.isna(value) else None
    )
    selected = selected.sort_values(
        ["Game Date", "Game DateTime", "Game ID"],
        na_position="last",
    ).reset_index(drop=True)

    next_game_number = int(state["next_game_number"])

    if selected["Series Game Number"].notna().any():
        selected = selected[
            selected["Series Game Number"].fillna(0).astype(int) >= next_game_number
        ].copy()
    else:
        last_completed_date = None

        if state["game_rows"]:
            last_completed_date = pd.to_datetime(
                state["game_rows"][-1]["Date"],
                errors="coerce",
            )

        if last_completed_date is not None and not pd.isna(last_completed_date):
            selected = selected[selected["Game Date"] > last_completed_date].copy()

        selected["Series Game Number"] = range(
            next_game_number,
            next_game_number + len(selected),
        )

    selected = selected[selected["Series Game Number"].astype(int) <= 7].copy()
    return selected.reset_index(drop=True)


def format_schedule_date(value: object) -> str:
    """Format a schedule date compactly."""
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return "Date TBD"

    return f"{parsed.strftime('%b')} {parsed.day}"


def format_schedule_time(value: object) -> str:
    """Format a schedule time/status compactly."""
    text = str(value).strip()

    if not text or text.lower() in {"nan", "none"}:
        return "TBD"

    return text


def format_schedule_game_label(row: pd.Series) -> str:
    """Format one remaining series schedule row."""
    game_number = int(row["Series Game Number"])
    date_label = format_schedule_date(row["Game Date"])
    time_label = format_schedule_time(row.get("Game Time", ""))
    if_necessary = " if necessary" if bool(row.get("If Necessary", False)) else ""
    return f"Game {game_number}: {date_label}, {time_label}{if_necessary}"


def summarize_remaining_series_schedule(schedule: pd.DataFrame) -> str | None:
    """Build a compact remaining-games schedule summary."""
    if schedule.empty:
        return None

    return " / ".join(
        format_schedule_game_label(row)
        for _, row in schedule.head(3).iterrows()
    )


def build_remaining_series_lines(
    next_winner: str,
    next_probability: float,
    schedule: pd.DataFrame,
) -> tuple[str, list[dict[str, str]]]:
    """Build separate lines for the next game and remaining series games."""
    next_line = f"Next: {next_winner} {next_probability:.0%}"
    games = []

    if schedule.empty:
        return next_line, games

    for _, row in schedule.iterrows():
        games.append(
            {
                "label": f"Game {int(row['Series Game Number'])}",
                "detail": format_schedule_game_label(row),
            }
        )

    return next_line, games


def render_remaining_game_boxes(games: list[dict[str, str]]) -> None:
    """Render all remaining series games in one grouped block."""
    if not games:
        st.caption("Remaining game dates are unavailable from the free schedule feed.")
        return

    row_html = []

    for game in games:
        row_html.append(
            f"""
            <div class="series-game-row">
                <span class="series-game-label">{html.escape(game["label"])}</span>
                <span class="series-game-detail">{html.escape(game["detail"])}</span>
            </div>
            """
        )

    st.html(
        '<div class="series-section-box">'
        '<div class="series-section-title">Remaining games</div>'
        '<div class="series-game-list">'
        + "".join(row_html)
        + "</div></div>"
    )


def render_remaining_series_schedule(schedule: pd.DataFrame) -> None:
    """Render remaining scheduled games for a live playoff series."""
    if schedule.empty:
        st.caption("Remaining game dates are unavailable from the free schedule feed.")
        return

    row_html = []

    for _, row in schedule.iterrows():
        home_team = str(row["Home Team"])
        away_team = str(row["Away Team"])
        matchup = f"{away_team} at {home_team}"
        label = format_schedule_game_label(row)
        status = str(row.get("Source", "Schedule"))
        row_html.append(
            f"""
            <div class="series-score-row">
                <div class="series-score-game">Game {int(row["Series Game Number"])}</div>
                <div>
                    <div class="series-score-matchup">{html.escape(matchup)}</div>
                    <div class="series-score-meta">{html.escape(label)}</div>
                </div>
                <div class="impact-score">{html.escape(status)}</div>
            </div>
            """
        )

    st.html("".join(row_html))


def get_scheduled_home_team_for_series_game(
    home_court_team: str,
    other_team: str,
    game_number: int,
) -> str:
    """Return home team from standard 2-2-1-1-1 series format."""
    if game_number < 1 or game_number > len(HOME_COURT_SCHEDULE):
        raise ValueError(f"Invalid series game number: {game_number}")

    return (
        home_court_team
        if HOME_COURT_SCHEDULE[game_number - 1] == "higher"
        else other_team
    )


def simulate_remaining_current_series(
    series_state: dict,
    simulations: int,
    seed: int,
    team_adjustments: dict[str, float] | None = None,
) -> dict:
    """Calculate exact rest-of-series probabilities from the actual series state."""
    strength = load_team_strength()
    model_bundle = load_model_bundle()
    home_court_team = series_state["home_court_team"]
    other_team = series_state["other_team"]
    teams = [home_court_team, other_team]
    initial_home_court_wins = int(series_state["wins"].get(home_court_team, 0))
    initial_other_wins = int(series_state["wins"].get(other_team, 0))
    probability_cache = {}
    memo = {}

    def get_game_home_probability(
        game_number: int,
        home_court_wins: int,
        other_wins: int,
    ) -> tuple[str, str, float]:
        """Return cached home-team win probability for one possible series state."""
        scheduled_home_team = get_scheduled_home_team_for_series_game(
            home_court_team,
            other_team,
            game_number,
        )
        scheduled_away_team = (
            other_team
            if scheduled_home_team == home_court_team
            else home_court_team
        )
        state_wins = {
            home_court_team: home_court_wins,
            other_team: other_wins,
        }
        cache_key = (
            scheduled_home_team,
            scheduled_away_team,
            game_number,
            state_wins[scheduled_home_team],
            state_wins[scheduled_away_team],
        )

        if cache_key not in probability_cache:
            playoff_context = build_playoff_context(
                home_team=scheduled_home_team,
                away_team=scheduled_away_team,
                game_number=game_number,
                home_series_wins=state_wins[scheduled_home_team],
                away_series_wins=state_wins[scheduled_away_team],
            )
            probability_cache[cache_key] = predict_game_probability(
                home_team=scheduled_home_team,
                away_team=scheduled_away_team,
                strength=strength,
                model_bundle=model_bundle,
                team_adjustments=team_adjustments,
                playoff_context=playoff_context,
            )

        return scheduled_home_team, scheduled_away_team, probability_cache[cache_key]

    def combine_probabilities(
        base: dict[str, float],
        addition: dict[str, float],
        weight: float,
    ) -> dict[str, float]:
        """Add weighted probabilities into a result dictionary."""
        combined = dict(base)

        for key, value in addition.items():
            combined[key] = combined.get(key, 0.0) + (value * weight)

        return combined

    def evaluate_state(
        game_number: int,
        home_court_wins: int,
        other_wins: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Recursively calculate exact series winner and result probabilities."""
        memo_key = (game_number, home_court_wins, other_wins)

        if memo_key in memo:
            return memo[memo_key]

        games_played = home_court_wins + other_wins

        if home_court_wins >= 4:
            result = (
                {home_court_team: 1.0, other_team: 0.0},
                {f"{home_court_team} in {games_played}": 1.0},
            )
            memo[memo_key] = result
            return result

        if other_wins >= 4:
            result = (
                {home_court_team: 0.0, other_team: 1.0},
                {f"{other_team} in {games_played}": 1.0},
            )
            memo[memo_key] = result
            return result

        scheduled_home_team, scheduled_away_team, home_probability = (
            get_game_home_probability(
                game_number,
                home_court_wins,
                other_wins,
            )
        )

        if scheduled_home_team == home_court_team:
            home_win_state = (game_number + 1, home_court_wins + 1, other_wins)
            away_win_state = (game_number + 1, home_court_wins, other_wins + 1)
        else:
            home_win_state = (game_number + 1, home_court_wins, other_wins + 1)
            away_win_state = (game_number + 1, home_court_wins + 1, other_wins)

        home_winner_probs, home_result_probs = evaluate_state(*home_win_state)
        away_winner_probs, away_result_probs = evaluate_state(*away_win_state)

        winner_probs = combine_probabilities({}, home_winner_probs, home_probability)
        winner_probs = combine_probabilities(
            winner_probs,
            away_winner_probs,
            1 - home_probability,
        )
        result_probs = combine_probabilities({}, home_result_probs, home_probability)
        result_probs = combine_probabilities(
            result_probs,
            away_result_probs,
            1 - home_probability,
        )

        result = (winner_probs, result_probs)
        memo[memo_key] = result
        return result

    winner_probabilities, result_probabilities = evaluate_state(
        series_state["next_game_number"],
        initial_home_court_wins,
        initial_other_wins,
    )
    result_table = pd.DataFrame(
        [
            {
                "Result": result,
                "Count": round(probability * simulations),
                "Probability": probability,
            }
            for result, probability in sorted(
                result_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
    )

    return {
        "series_probabilities": winner_probabilities,
        "result_table": result_table,
    }


def build_series_state_for_matchup(
    higher_seed_team: str,
    lower_seed_team: str,
) -> dict:
    """Create a clean 0-0 series state for the momentum map."""
    return {
        "series_key": " vs ".join(sorted([higher_seed_team, lower_seed_team])),
        "home_court_team": higher_seed_team,
        "other_team": lower_seed_team,
        "teams": [higher_seed_team, lower_seed_team],
        "wins": {higher_seed_team: 0, lower_seed_team: 0},
        "leader": higher_seed_team,
        "games_played": 0,
        "next_game_number": 1,
        "completed": False,
        "game_rows": [],
        "last_winner": None,
        "current_streak": 0,
    }


def build_hypothetical_series_state_after_next_game(
    series_state: dict,
    winner: str,
) -> dict:
    """Advance a series state by one hypothetical next-game result."""
    wins = dict(series_state["wins"])
    wins[winner] = wins.get(winner, 0) + 1
    teams = list(series_state["teams"])
    leader = max(teams, key=lambda team: wins.get(team, 0))
    next_game_number = int(series_state["next_game_number"]) + 1
    completed = wins[winner] >= 4

    return {
        **series_state,
        "wins": wins,
        "leader": leader,
        "games_played": int(series_state["games_played"]) + 1,
        "next_game_number": next_game_number,
        "completed": completed,
        "last_winner": winner,
        "current_streak": (
            int(series_state.get("current_streak", 0)) + 1
            if series_state.get("last_winner") == winner
            else 1
        ),
    }


def build_series_next_game_swing_rows(
    series_state: dict,
    team_adjustments: dict[str, float] | None,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Show how the series probabilities change after each possible next result."""
    if series_state.get("completed") or int(series_state["next_game_number"]) > 7:
        return pd.DataFrame()

    teams = list(series_state["teams"])
    base_results = simulate_remaining_current_series(
        series_state=series_state,
        simulations=simulations,
        seed=seed,
        team_adjustments=team_adjustments,
    )
    base_probabilities = base_results["series_probabilities"]
    next_game_number = int(series_state["next_game_number"])
    next_home_team = get_scheduled_home_team_for_series_game(
        series_state["home_court_team"],
        series_state["other_team"],
        next_game_number,
    )
    next_away_team = (
        series_state["other_team"]
        if next_home_team == series_state["home_court_team"]
        else series_state["home_court_team"]
    )
    rows = []

    for hypothetical_winner in [next_home_team, next_away_team]:
        hypothetical_state = build_hypothetical_series_state_after_next_game(
            series_state,
            hypothetical_winner,
        )

        if hypothetical_state["completed"]:
            probabilities = {
                team: 1.0 if team == hypothetical_winner else 0.0
                for team in teams
            }
        else:
            results = simulate_remaining_current_series(
                series_state=hypothetical_state,
                simulations=simulations,
                seed=seed + next_game_number,
                team_adjustments=team_adjustments,
            )
            probabilities = results["series_probabilities"]

        favorite = max(probabilities, key=probabilities.get)
        swing_team = hypothetical_winner
        rows.append(
            {
                "Scenario": f"If {hypothetical_winner} wins Game {next_game_number}",
                teams[0]: probabilities.get(teams[0], 0.0),
                teams[1]: probabilities.get(teams[1], 0.0),
                "Projected Leader": favorite,
                "Swing": probabilities.get(swing_team, 0.0)
                - base_probabilities.get(swing_team, 0.0),
            }
        )

    return pd.DataFrame(rows)


def render_series_momentum_map(
    series_state: dict,
    team_adjustments: dict[str, float] | None,
    simulations: int,
    seed: int,
) -> None:
    """Render next-game series leverage."""
    swing_rows = build_series_next_game_swing_rows(
        series_state=series_state,
        team_adjustments=team_adjustments,
        simulations=simulations,
        seed=seed,
    )
    render_section_kicker("Series momentum map")

    if swing_rows.empty:
        st.caption("No next-game series swing is available.")
        return

    display = swing_rows.copy()
    for team in series_state["teams"]:
        if team in display.columns:
            display[team] = display[team].map("{:.0%}".format)
    display["Swing"] = display["Swing"].map(format_probability_shift)
    st.dataframe(display, width="stretch", hide_index=True)


def infer_current_playoff_context_for_matchup(
    home_team: str,
    away_team: str,
) -> dict[str, float] | None:
    """Infer current playoff context when a selected matchup is an active series."""
    selected_teams = {home_team, away_team}

    for state in build_current_playoff_series_states():
        if state["completed"]:
            continue

        if set(state["teams"]) != selected_teams:
            continue

        return build_playoff_context(
            home_team=home_team,
            away_team=away_team,
            game_number=state["next_game_number"],
            home_series_wins=state["wins"].get(home_team, 0),
            away_series_wins=state["wins"].get(away_team, 0),
        )

    return None


def validate_unique_teams(teams: list[str]) -> bool:
    """Validate that all selected teams are unique."""
    return len(teams) == len(set(teams))


def get_higher_seed_team(team_a: str, team_b: str, seeded_teams: list[str]) -> str:
    """Return higher-seeded team based on list order."""
    return team_a if seeded_teams.index(team_a) < seeded_teams.index(team_b) else team_b


def simulate_conference_bracket(
    seeded_teams: list[str],
    probability_cache: dict[tuple[str, str], float],
    rng: random.Random,
) -> str:
    """Simulate one 8-team conference bracket."""
    champion, _ = simulate_conference_bracket_with_path(
        conference_name="Conference",
        seeded_teams=seeded_teams,
        probability_cache=probability_cache,
        rng=rng,
    )
    return champion


def simulate_conference_bracket_with_path(
    conference_name: str,
    seeded_teams: list[str],
    probability_cache: dict[tuple[str, str], float],
    rng: random.Random,
) -> tuple[str, list[dict]]:
    """Simulate one conference bracket and return the path."""
    path_rows = []

    first_round_pairs = [
        (seeded_teams[0], seeded_teams[7], "1 vs 8"),
        (seeded_teams[3], seeded_teams[4], "4 vs 5"),
        (seeded_teams[2], seeded_teams[5], "3 vs 6"),
        (seeded_teams[1], seeded_teams[6], "2 vs 7"),
    ]

    first_round_winners = []

    for higher_seed_team, lower_seed_team, matchup_label in first_round_pairs:
        winner, _, _ = simulate_single_series(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            probability_cache=probability_cache,
            rng=rng,
        )
        first_round_winners.append(winner)
        path_rows.append(
            {
                "Round": f"{conference_name} First Round",
                "Matchup": matchup_label,
                "Team A": higher_seed_team,
                "Team B": lower_seed_team,
                "Winner": winner,
            }
        )

    semifinal_pairs = [
        (first_round_winners[0], first_round_winners[1], "Semifinal 1"),
        (first_round_winners[2], first_round_winners[3], "Semifinal 2"),
    ]

    semifinal_winners = []

    for team_a, team_b, matchup_label in semifinal_pairs:
        higher_seed_team = get_higher_seed_team(team_a, team_b, seeded_teams)
        lower_seed_team = team_b if higher_seed_team == team_a else team_a

        winner, _, _ = simulate_single_series(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            probability_cache=probability_cache,
            rng=rng,
        )
        semifinal_winners.append(winner)
        path_rows.append(
            {
                "Round": f"{conference_name} Semifinals",
                "Matchup": matchup_label,
                "Team A": higher_seed_team,
                "Team B": lower_seed_team,
                "Winner": winner,
            }
        )

    higher_seed_team = get_higher_seed_team(
        semifinal_winners[0],
        semifinal_winners[1],
        seeded_teams,
    )

    lower_seed_team = (
        semifinal_winners[1]
        if higher_seed_team == semifinal_winners[0]
        else semifinal_winners[0]
    )

    conference_champion, _, _ = simulate_single_series(
        higher_seed_team=higher_seed_team,
        lower_seed_team=lower_seed_team,
        probability_cache=probability_cache,
        rng=rng,
    )

    path_rows.append(
        {
            "Round": f"{conference_name} Finals",
            "Matchup": "Conference Finals",
            "Team A": higher_seed_team,
            "Team B": lower_seed_team,
            "Winner": conference_champion,
        }
    )

    return conference_champion, path_rows


def simulate_single_full_bracket_path(
    east_teams: list[str],
    west_teams: list[str],
    seed: int,
    team_adjustments: dict[str, float] | None = None,
    probability_cache: dict[tuple[str, str], float] | None = None,
) -> list[dict]:
    """Simulate and return one visual playoff bracket path."""
    all_teams = east_teams + west_teams
    rng = random.Random(seed)

    if probability_cache is None:
        strength = load_team_strength()
        model_bundle = load_model_bundle()
        probability_cache = build_matchup_probability_cache(
            teams=all_teams,
            strength=strength,
            model_bundle=model_bundle,
            team_adjustments=team_adjustments,
        )

    east_champion, east_path = simulate_conference_bracket_with_path(
        conference_name="East",
        seeded_teams=east_teams,
        probability_cache=probability_cache,
        rng=rng,
    )

    west_champion, west_path = simulate_conference_bracket_with_path(
        conference_name="West",
        seeded_teams=west_teams,
        probability_cache=probability_cache,
        rng=rng,
    )

    east_has_higher_seed = east_teams.index(east_champion) <= west_teams.index(
        west_champion
    )

    if east_has_higher_seed:
        higher_seed_team = east_champion
        lower_seed_team = west_champion
    else:
        higher_seed_team = west_champion
        lower_seed_team = east_champion

    champion, _, _ = simulate_single_series(
        higher_seed_team=higher_seed_team,
        lower_seed_team=lower_seed_team,
        probability_cache=probability_cache,
        rng=rng,
    )

    finals_path = [
        {
            "Round": "NBA Finals",
            "Matchup": "NBA Finals",
            "Team A": higher_seed_team,
            "Team B": lower_seed_team,
            "Winner": champion,
        }
    ]

    return east_path + west_path + finals_path


def render_bracket_game_card(row: dict) -> None:
    """Render one bracket matchup card."""
    team_a = row["Team A"]
    team_b = row["Team B"]
    winner = row["Winner"]

    team_a_logo = get_logo_url(team_a)
    team_b_logo = get_logo_url(team_b)

    team_a_class = "bracket-team-winner" if team_a == winner else ""
    team_b_class = "bracket-team-winner" if team_b == winner else ""

    st.html(
        f"""
        <div class="bracket-card">
            <div class="small-muted">{html.escape(row["Matchup"])}</div>
            <div class="bracket-team-row {team_a_class}">
                <img class="bracket-logo" src="{html.escape(team_a_logo)}">
                <span>{html.escape(team_a)}</span>
            </div>
            <div class="bracket-team-row {team_b_class}">
                <img class="bracket-logo" src="{html.escape(team_b_logo)}">
                <span>{html.escape(team_b)}</span>
            </div>
            <div class="small-muted">Winner: <b>{html.escape(winner)}</b></div>
        </div>
        """
    )


def render_bracket_visualization(path_rows: list[dict]) -> None:
    """Render a visual playoff bracket path."""
    render_section_kicker("Example bracket path")

    columns = st.columns(4)

    column_rounds = [
        ["East First Round", "West First Round"],
        ["East Semifinals", "West Semifinals"],
        ["East Finals", "West Finals"],
        ["NBA Finals"],
    ]

    for column, rounds in zip(columns, column_rounds):
        with column:
            for round_name in rounds:
                rows = [row for row in path_rows if row["Round"] == round_name]

                if not rows:
                    continue

                st.html(
                    f'<div class="bracket-round-title">{html.escape(round_name)}</div>'
                )

                for row in rows:
                    render_bracket_game_card(row)


def simulate_full_playoff_bracket(
    east_teams: list[str],
    west_teams: list[str],
    simulations: int,
    seed: int,
    progress_bar,
    status_text,
    team_adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate full NBA playoffs many times."""
    strength = load_team_strength()
    model_bundle = load_model_bundle()
    rng = random.Random(seed)

    all_teams = east_teams + west_teams
    start_time = time.perf_counter()

    status_text.text("Preparing matchup probabilities...")
    probability_cache = build_matchup_probability_cache(
        teams=all_teams,
        strength=strength,
        model_bundle=model_bundle,
        team_adjustments=team_adjustments,
    )

    champion_counts = Counter()
    finals_counts = Counter()
    progress_update_interval = max(1, simulations // 4)

    for simulation_number in range(1, simulations + 1):
        east_champion = simulate_conference_bracket(
            seeded_teams=east_teams,
            probability_cache=probability_cache,
            rng=rng,
        )

        west_champion = simulate_conference_bracket(
            seeded_teams=west_teams,
            probability_cache=probability_cache,
            rng=rng,
        )

        east_has_higher_seed = (
            east_teams.index(east_champion) <= west_teams.index(west_champion)
        )

        if east_has_higher_seed:
            higher_seed_team = east_champion
            lower_seed_team = west_champion
        else:
            higher_seed_team = west_champion
            lower_seed_team = east_champion

        champion, _, _ = simulate_single_series(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            probability_cache=probability_cache,
            rng=rng,
        )

        champion_counts[champion] += 1
        finals_counts[east_champion] += 1
        finals_counts[west_champion] += 1

        if (
            simulation_number % progress_update_interval == 0
            or simulation_number == simulations
        ):
            progress_bar.progress(simulation_number / simulations)
            status_text.text(
                f"Completed {simulation_number:,} of {simulations:,} simulations..."
            )

    elapsed_seconds = time.perf_counter() - start_time

    rows = []

    for team in all_teams:
        rows.append(
            {
                "Team": team,
                "Finals Probability": finals_counts[team] / simulations,
                "Championship Probability": champion_counts[team] / simulations,
                "Titles Won": champion_counts[team],
                "Player Availability Adjustment": 0.0
                if not team_adjustments
                else team_adjustments.get(team, 0.0),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "Championship Probability",
        ascending=False,
    )
    results.attrs["probability_cache"] = probability_cache

    status_text.text(
        f"Simulation complete. Cached {len(probability_cache):,} matchup probabilities. "
        f"Runtime: {elapsed_seconds:.2f} seconds."
    )

    return results


def build_top_availability_impacts(limit: int = 6) -> pd.DataFrame:
    """Return the largest current player availability impacts."""
    teams = get_teams_from_strength(load_team_strength())
    _, availability_rows = build_official_availability_adjustments(teams)

    if availability_rows.empty:
        return pd.DataFrame()

    impacted = availability_rows[
        availability_rows["Weighted Impact"].fillna(0) > 0
    ].copy()

    if impacted.empty:
        return pd.DataFrame()

    return impacted.sort_values("Weighted Impact", ascending=False).head(limit)


def render_availability_impact_rows(availability_rows: pd.DataFrame) -> None:
    """Render current availability impacts with team logos."""
    if availability_rows.empty:
        st.caption("No current player availability impacts in the saved report.")
        return

    rows = []

    for _, row in availability_rows.iterrows():
        team = str(row["Team"])
        player = str(row["Player"])
        status = str(row["Status"])
        detail = str(row.get("Player Detail", ""))
        impact = float(row["Weighted Impact"])
        rows.append(
            f"""
            <div class="impact-row">
                <img class="impact-logo" src="{html.escape(get_logo_url(team))}" alt="{html.escape(team)} logo">
                <div>
                    <div class="impact-name">{html.escape(player)}</div>
                    <div class="impact-meta">{html.escape(team)} / {html.escape(status)} / {html.escape(detail)}</div>
                </div>
                <div class="impact-score">{impact:.1f}</div>
            </div>
            """
        )

    st.html("".join(rows))


def build_active_series_projection_rows(
    simulations: int = DEFAULT_SERIES_PROJECTION_SCALE,
    seed: int = 77,
) -> list[dict]:
    """Build compact current-series projections for dashboard cards."""
    series_states = build_current_playoff_series_states()
    active_states = [
        state
        for state in series_states
        if not state["completed"]
        and all(is_known_team_name(team) for team in state["teams"])
    ]

    if not active_states:
        return []

    active_teams = sorted(
        {
            team
            for state in active_states
            for team in state["teams"]
        }
    )
    team_adjustments, _ = build_official_availability_adjustments(active_teams)
    rows = []

    for index, state in enumerate(active_states):
        wins = state["wins"]
        schedule = get_schedule_for_series(state)
        next_game_number = state["next_game_number"]
        next_home_team = get_scheduled_home_team_for_series_game(
            state["home_court_team"],
            state["other_team"],
            next_game_number,
        )
        next_away_team = (
            state["other_team"]
            if next_home_team == state["home_court_team"]
            else state["home_court_team"]
        )
        playoff_context = build_playoff_context(
            home_team=next_home_team,
            away_team=next_away_team,
            game_number=next_game_number,
            home_series_wins=wins.get(next_home_team, 0),
            away_series_wins=wins.get(next_away_team, 0),
        )
        next_winner, next_home_probability, next_away_probability, _ = predict_game(
            home_team=next_home_team,
            away_team=next_away_team,
            team_adjustments=team_adjustments,
            playoff_context=playoff_context,
        )
        rest_of_series = simulate_remaining_current_series(
            series_state=state,
            simulations=simulations,
            seed=seed + index,
            team_adjustments=team_adjustments,
        )
        probabilities = rest_of_series["series_probabilities"]
        projected_winner = max(probabilities, key=probabilities.get)
        next_probability = (
            next_home_probability
            if next_winner == next_home_team
            else next_away_probability
        )
        rows.append(
            {
                "state": state,
                "projected_winner": projected_winner,
                "series_probability": probabilities[projected_winner],
                "next_winner": next_winner,
                "next_probability": next_probability,
                "next_home_team": next_home_team,
                "next_away_team": next_away_team,
                "schedule": schedule,
                "schedule_summary": summarize_remaining_series_schedule(schedule),
            }
        )

    return rows


def render_current_series_cards() -> None:
    """Render compact current-series cards."""
    series_projections = build_active_series_projection_rows()

    if not series_projections:
        return

    render_section_kicker("Current Series")
    columns = st.columns(min(2, len(series_projections)))

    for index, projection in enumerate(series_projections):
        next_game_line, remaining_games = build_remaining_series_lines(
            next_winner=projection["next_winner"],
            next_probability=projection["next_probability"],
            schedule=projection["schedule"],
        )

        with columns[index % len(columns)]:
            render_series_score_card(
                projection["state"],
                projection=(
                    f"Series: {projection['projected_winner']} "
                    f"{projection['series_probability']:.0%}"
                ),
                next_game=next_game_line,
                remaining_games=remaining_games,
            )


@st.fragment(run_every=15)
def render_current_series_live_fragment() -> None:
    """Refresh current-series state after live games go final."""
    render_current_series_cards()


def render_power_snapshot(strength: pd.DataFrame, limit: int = 5) -> None:
    """Render a compact power ranking snapshot."""
    top_rows = strength.sort_values("ELO", ascending=False).head(limit)
    render_team_mini_rows(
        [
            {
                "team": row["TEAM_NAME"],
                "meta": (
                    f"Win {float(row['SEASON_WIN_PCT']):.0%} / "
                    f"Last 10 {float(row['ROLLING_WIN_PCT_10']):.0%}"
                ),
                "score": f"{float(row['ELO']):.0f}",
            }
            for _, row in top_rows.iterrows()
        ]
    )


def render_home_dashboard(teams: list[str]) -> None:
    """Render the first-screen dashboard."""
    st.header("Home")

    render_section_kicker("Today's Games")
    render_today_games_live_fragment(compact=True)

    render_today_game_breakdown_live_fragment()

    render_current_series_live_fragment()


def render_today_games_section(teams: list[str]) -> None:
    """Render a dedicated current-games view."""
    st.header("Today")
    render_today_games_live_fragment(compact=False, show_schedule_rows=True)


def format_team_rest_display(row: pd.Series) -> tuple[str, str]:
    """Format the schedule-load rest label for display."""
    last_game_date = pd.to_datetime(row.get("GAME_DATE"), errors="coerce")
    if pd.isna(last_game_date):
        return "TBD", "Recent game date unavailable"

    days_since_last_game = max((date.today() - last_game_date.date()).days, 0)
    return f"{days_since_last_game} days", "Actual days since last game"


def render_team_profile_hero(team_name: str, row: pd.Series) -> None:
    """Render a colorful profile header for one team."""
    team_color = get_team_color(team_name)
    strength = load_team_strength()
    league_rank = get_team_league_rank(team_name, strength)
    metrics = [
        ("League rank", f"#{league_rank}" if league_rank else "TBD"),
        ("Record", format_team_record(team_name)),
        ("Win %", f"{float(row['SEASON_WIN_PCT']):.0%}"),
        ("Point differential", f"{float(row['SEASON_AVG_PLUS_MINUS']):+.1f}"),
    ]
    metric_html = "".join(
        f"""
        <div class="profile-stat">
            <div class="profile-label">{html.escape(label)}</div>
            <div class="profile-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in metrics
    )

    st.html(
        f"""
        <div class="profile-hero" style="--team-color: {html.escape(team_color)};">
            <img class="profile-logo" src="{html.escape(get_logo_url(team_name))}" alt="{html.escape(team_name)} logo">
            <div>
                <div class="profile-label">{html.escape(get_team_abbreviation(team_name))}</div>
                <div class="profile-name">{html.escape(team_name)}</div>
                <div class="profile-metrics">{metric_html}</div>
            </div>
        </div>
        """
    )


def get_team_league_rank(team_name: str, strength: pd.DataFrame) -> int:
    """Return a 1-based league rank for one team by Elo."""
    ranked = strength.sort_values("ELO", ascending=False).reset_index(drop=True)
    matches = ranked.index[ranked["TEAM_NAME"].eq(team_name)].tolist()
    return int(matches[0] + 1) if matches else 0


def render_player_impact_rows(team_name: str, limit: int = 8) -> None:
    """Render the highest-impact players for one team."""
    players = get_players_for_team(team_name).head(limit)

    if players.empty:
        st.caption("No player impact rows found for this team.")
        return

    rows = []

    for _, player in players.iterrows():
        rows.append(
            f"""
            <div class="player-row">
                <div class="team-mini-score">{float(player["IMPACT_SCORE"]):.1f}</div>
                <div>
                    <div class="player-name">{html.escape(str(player["PLAYER_NAME"]))}</div>
                    <div class="player-meta">
                        {float(player["MPG"]):.1f} MPG / {float(player["PPG"]):.1f} PPG /
                        {float(player["RPG"]):.1f} RPG / {float(player["APG"]):.1f} APG
                    </div>
                </div>
                <div class="impact-score">{html.escape(str(player["INJURY_TIER"]))}</div>
            </div>
            """
        )

    st.html("".join(rows))


def render_team_availability_panel(team_name: str) -> None:
    """Render one team's current availability impact."""
    _, availability_rows = build_official_availability_adjustments([team_name])

    if availability_rows.empty:
        st.caption("No listed impact for this team in the saved injury report.")
        return

    render_availability_impact_rows(
        availability_rows[
            availability_rows["Weighted Impact"].fillna(0) > 0
        ].sort_values("Weighted Impact", ascending=False)
    )


def render_team_profiles_section(teams: list[str]) -> None:
    """Render team profile, rankings, roster impact, and schedule-load signals."""
    st.header("Teams")

    selected_team = st.selectbox(
        "Team",
        teams,
        index=safe_team_index(teams, "Oklahoma City Thunder"),
        key="profile_team",
    )
    strength = load_team_strength().sort_values("ELO", ascending=False).reset_index(drop=True)
    row = get_team_strength_row(selected_team, strength)
    render_team_profile_hero(selected_team, row)

    left_col, right_col = st.columns([1.1, 0.9], gap="medium")

    with left_col:
        render_section_kicker("Rotation Impact")
        render_player_impact_rows(selected_team)

    with right_col:
        render_section_kicker("Schedule Load")
        rest_value, rest_note = format_team_rest_display(row)
        render_dashboard_cards(
            [
                {
                    "label": "Rest",
                    "value": rest_value,
                    "note": rest_note,
                    "color": "#2563eb",
                },
                {
                    "label": "Last 7",
                    "value": f"{int(row['GAMES_LAST_7_DAYS'])} games",
                    "note": "Recent workload",
                    "color": "#f97316",
                },
                {
                    "label": "Road",
                    "value": f"{int(row['CURRENT_ROAD_STREAK'])}",
                    "note": "Current road streak",
                    "color": "#7c3aed",
                },
                {
                    "label": "Stars",
                    "value": f"{int(row['STAR_COUNT'])}",
                    "note": "Stat-based star count",
                    "color": "#db2777",
                },
            ]
        )

        render_section_kicker("Availability")
        render_team_availability_panel(selected_team)

    with st.expander("Power rankings table", expanded=False):
        rankings = strength[
            [
                "TEAM_NAME",
                "ELO",
                "SEASON_WIN_PCT",
                "ROLLING_NET_RATING_10",
                "PLAYER_TOP_8",
                "STAR_COUNT",
            ]
        ].copy()
        rankings["League Rank"] = range(1, len(rankings) + 1)
        rankings = rankings.rename(
            columns={
                "TEAM_NAME": "Team",
                "ELO": "Elo",
                "SEASON_WIN_PCT": "Win %",
                "ROLLING_NET_RATING_10": "Recent Form",
                "PLAYER_TOP_8": "Top 8 Player Strength",
                "STAR_COUNT": "Stars",
            }
        )
        rankings = rankings[
            [
                "League Rank",
                "Team",
                "Elo",
                "Win %",
                "Recent Form",
                "Top 8 Player Strength",
                "Stars",
            ]
        ]
        st.dataframe(rankings, width="stretch", hide_index=True)


@st.cache_data
def build_team_backtest_summary() -> pd.DataFrame:
    """Calculate historical model-pick accuracy by team from saved feature rows."""
    try:
        features = load_model_features()
        model_bundle = load_model_bundle()
    except Exception:
        return pd.DataFrame()

    feature_columns = model_bundle.get("feature_columns", [])

    if not feature_columns or any(column not in features.columns for column in feature_columns):
        return pd.DataFrame()

    model = model_bundle["model"]
    probabilities = model.predict_proba(features[feature_columns])[:, 1]
    rows = []

    for (_, game), home_probability in zip(features.iterrows(), probabilities):
        home_team = str(game["HOME_TEAM"])
        away_team = str(game["AWAY_TEAM"])
        home_win = int(game["HOME_WIN"]) == 1
        predicted_home = home_probability >= 0.5
        predicted_winner = home_team if predicted_home else away_team
        actual_winner = home_team if home_win else away_team
        correct = predicted_winner == actual_winner
        confidence = max(float(home_probability), 1 - float(home_probability))

        for team in [home_team, away_team]:
            rows.append(
                {
                    "Team": team,
                    "Season": game["SEASON"],
                    "Games": 1,
                    "Correct": int(correct),
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


def render_model_calibration_chart(calibration_metrics: pd.DataFrame) -> None:
    """Render calibration as predicted probability vs actual home win rate."""
    if calibration_metrics.empty:
        st.caption("No calibration metrics found.")
        return

    chart = calibration_metrics.copy()
    chart = chart.rename(
        columns={
            "Average_Predicted_Probability": "Predicted",
            "Actual_Home_Win_Rate": "Actual",
        }
    )
    chart["Bucket"] = chart["Bucket"].astype(str)
    st.line_chart(chart.set_index("Bucket")[["Predicted", "Actual"]])


def render_backtest_chart(backtest_metrics: pd.DataFrame) -> None:
    """Render rolling backtest accuracy and ROC-AUC."""
    if backtest_metrics.empty:
        st.caption("No backtest metrics found.")
        return

    metric_columns = [
        column
        for column in ["Accuracy", "Blend_ROC_AUC", "ROC_AUC"]
        if column in backtest_metrics.columns
    ]

    if not metric_columns:
        st.caption("Backtest file does not include chartable metrics.")
        return

    chart = backtest_metrics.copy()
    st.line_chart(chart.set_index("Test_Season")[metric_columns])


def render_team_backtest_panel(teams: list[str]) -> None:
    """Render team-level historical model reliability."""
    summary = build_team_backtest_summary()

    if summary.empty:
        st.caption("Team backtest summary is unavailable.")
        return

    selected_team = st.selectbox(
        "Backtest Team",
        teams,
        index=safe_team_index(teams, "Boston Celtics"),
        key="model_backtest_team",
    )
    team_rows = summary[summary["Team"].eq(selected_team)].copy()

    if team_rows.empty:
        st.caption("No historical rows found for this team.")
        return

    st.line_chart(team_rows.set_index("Season")[["Accuracy", "Avg_Confidence"]])
    latest = team_rows.sort_values("Season").iloc[-1]
    st.caption(
        f"Latest: {float(latest['Accuracy']):.1%} pick accuracy across "
        f"{int(latest['Games'])} games."
    )


def render_single_game_section(teams: list[str]) -> None:
    """Render single-game predictor."""
    st.header("Game")
    prediction_state_key = "single_game_last_prediction"

    home_col, away_col, action_col = st.columns([2, 2, 1])

    with home_col:
        home_team = st.selectbox(
            "Home",
            teams,
            index=safe_team_index(teams, "Boston Celtics"),
            key="single_home_team",
        )

    with away_col:
        away_team = st.selectbox(
            "Away",
            teams,
            index=safe_team_index(teams, "Denver Nuggets"),
            key="single_away_team",
        )

    with action_col:
        st.write("")
        predict_clicked = st.button(
            "Predict",
            type="primary",
            width="stretch",
        )

    if home_team == away_team:
        st.warning("Choose two different teams for the single-game predictor.")
        return

    render_matchup_preview(
        team_a=home_team,
        team_b=away_team,
        label_a="Home",
        label_b="Away",
    )

    team_adjustments = render_official_availability_adjustments(
        selected_teams=[home_team, away_team],
        key_prefix="single_game",
    )

    prediction_state = st.session_state.get(prediction_state_key)

    if predict_clicked:
        playoff_context = infer_current_playoff_context_for_matchup(
            home_team,
            away_team,
        )

        if playoff_context:
            st.html(
                '<div class="compact-note">Active playoff series context included.</div>'
            )

        winner, home_probability, away_probability, details = predict_game(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=team_adjustments,
            playoff_context=playoff_context,
        )
        winner_probability = max(home_probability, away_probability)
        confidence = get_game_confidence_label(winner_probability)

        report = create_prediction_report(
            title=f"NBA Game Prediction: {home_team} vs {away_team}",
            predicted_winner=winner,
            probabilities={
                home_team: home_probability,
                away_team: away_probability,
            },
            confidence=confidence,
        )

        prediction_state = {
            "home_team": home_team,
            "away_team": away_team,
            "home_probability": home_probability,
            "away_probability": away_probability,
            "winner": winner,
            "winner_probability": winner_probability,
            "confidence": confidence,
            "details": details,
            "team_adjustments": team_adjustments,
            "playoff_context": playoff_context,
            "seed": build_stable_matchup_seed(home_team, away_team),
            "report": report,
        }
        st.session_state[prediction_state_key] = prediction_state

    if prediction_state and (
        prediction_state.get("home_team") == home_team
        and prediction_state.get("away_team") == away_team
    ):
        playoff_context = prediction_state.get("playoff_context")

        if playoff_context:
            st.html(
                '<div class="compact-note">Active playoff series context included.</div>'
            )

        render_prediction_result_card(
            label="Game prediction",
            winner=str(prediction_state["winner"]),
            probability=float(prediction_state["winner_probability"]),
            confidence=str(prediction_state["confidence"]),
            details=dict(prediction_state["details"]),
            context="game",
            note=f"{home_team} home vs {away_team}",
        )

        render_matchup_cards(
            team_a=home_team,
            team_b=away_team,
            probability_a=float(prediction_state["home_probability"]),
            probability_b=float(prediction_state["away_probability"]),
            winner=str(prediction_state["winner"]),
            label="Win probability",
        )

        render_prediction_trust_meter(
            home_probability=float(prediction_state["home_probability"]),
            winner_probability=float(prediction_state["winner_probability"]),
        )

        render_section_kicker("Prediction factors")
        render_matchup_signal_cards(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=dict(prediction_state["team_adjustments"]),
            playoff_context=playoff_context,
        )

        render_game_score_simulation(
            home_team=home_team,
            away_team=away_team,
            home_probability=float(prediction_state["home_probability"]),
            team_adjustments=dict(prediction_state["team_adjustments"]),
            seed=int(prediction_state["seed"]),
            label="Projected score",
        )

        render_prediction_lab(
            home_team=home_team,
            away_team=away_team,
            base_home_probability=float(prediction_state["home_probability"]),
            team_adjustments=dict(prediction_state["team_adjustments"]),
            playoff_context=playoff_context,
            key_prefix="single_game",
        )

        render_underdog_path(
            home_team=home_team,
            away_team=away_team,
            home_probability=float(prediction_state["home_probability"]),
            team_adjustments=dict(prediction_state["team_adjustments"]),
            playoff_context=playoff_context,
        )

        render_player_swing_rankings(
            home_team=home_team,
            away_team=away_team,
            base_home_probability=float(prediction_state["home_probability"]),
            team_adjustments=dict(prediction_state["team_adjustments"]),
            playoff_context=playoff_context,
        )

        render_similar_historical_games(
            home_team=home_team,
            away_team=away_team,
            team_adjustments=dict(prediction_state["team_adjustments"]),
            playoff_context=playoff_context,
        )

        render_matchup_explanation(
            team_a=home_team,
            team_b=away_team,
            predicted_winner=str(prediction_state["winner"]),
            winner_probability=float(prediction_state["winner_probability"]),
            playoff_context=playoff_context,
            team_adjustments=dict(prediction_state["team_adjustments"]),
        )

        with st.expander("Details", expanded=False):
            details = prediction_state["details"]
            st.write(f"Win chance: {float(details['model_probability']):.1%}")
            st.write(f"Power rating chance: {float(details['elo_probability']):.1%}")
            st.write(
                "Blended probability before player availability adjustment: "
                f"{float(details['blended_probability']):.1%}"
            )
            st.write(f"Final probability: {float(details['final_probability']):.1%}")
            render_report_download(
                report_text=str(prediction_state["report"]),
                file_name="nba_game_prediction_report.txt",
                key="single_game_report_saved",
            )


def render_series_section(teams: list[str]) -> None:
    """Render playoff series simulator."""
    st.header("Series")

    higher_col, lower_col, action_col = st.columns([2, 2, 1])

    with higher_col:
        higher_seed_team = st.selectbox(
            "Higher Seed",
            teams,
            index=safe_team_index(teams, "Boston Celtics"),
            key="series_higher_seed_team",
        )

    with lower_col:
        lower_seed_team = st.selectbox(
            "Lower Seed",
            teams,
            index=safe_team_index(teams, "Denver Nuggets"),
            key="series_lower_seed_team",
        )

    with action_col:
        st.write("")
        simulate_clicked = st.button(
            "Simulate",
            type="primary",
            width="stretch",
        )

    if higher_seed_team == lower_seed_team:
        st.warning("Choose two different teams for the series simulator.")
        return

    render_matchup_preview(
        team_a=higher_seed_team,
        team_b=lower_seed_team,
        label_a="Higher Seed",
        label_b="Lower Seed",
    )

    team_adjustments = render_official_availability_adjustments(
        selected_teams=[higher_seed_team, lower_seed_team],
        key_prefix="series",
    )

    with st.expander("Settings", expanded=False):
        sim_col, seed_col = st.columns(2)

        with sim_col:
            simulations = st.slider(
                "Series simulations",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100,
            )

        with seed_col:
            seed = st.number_input(
                "Simulation ID",
                min_value=1,
                max_value=999999,
                value=42,
                step=1,
                key="series_seed",
            )

    series_simulation = st.session_state.get("series_last_simulation")

    if simulate_clicked:
        with st.spinner("Simulating playoff series..."):
            results = simulate_series(
                higher_seed_team=higher_seed_team,
                lower_seed_team=lower_seed_team,
                simulations=simulations,
                seed=int(seed),
                team_adjustments=team_adjustments,
            )

        higher_probability = results["higher_seed_probability"]
        lower_probability = results["lower_seed_probability"]
        favorite_probability = max(higher_probability, lower_probability)
        confidence = get_series_confidence_label(favorite_probability)

        predicted_winner = (
            higher_seed_team
            if higher_probability >= lower_probability
            else lower_seed_team
        )

        most_likely = (
            f"{results['most_likely_result']} "
            f"({results['most_likely_probability']:.1%})"
        )

        series_score_path = build_series_score_simulation_path(
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
            probability_cache=results["probability_cache"],
            team_adjustments=team_adjustments,
            seed=int(seed) + 9000,
            target_result=results["most_likely_result"],
        )

        st.session_state["series_last_simulation"] = {
            "higher_seed_team": higher_seed_team,
            "lower_seed_team": lower_seed_team,
            "simulations": int(simulations),
            "seed": int(seed),
            "team_adjustments": team_adjustments,
            "results": results,
            "predicted_winner": predicted_winner,
            "higher_probability": higher_probability,
            "lower_probability": lower_probability,
            "favorite_probability": favorite_probability,
            "confidence": confidence,
            "most_likely": most_likely,
            "series_score_path": series_score_path,
        }
        series_simulation = st.session_state["series_last_simulation"]

    current_signature = (
        higher_seed_team,
        lower_seed_team,
        int(simulations),
        int(seed),
    )
    stored_signature = None
    if series_simulation:
        stored_signature = (
            series_simulation.get("higher_seed_team"),
            series_simulation.get("lower_seed_team"),
            int(series_simulation.get("simulations", 0)),
            int(series_simulation.get("seed", 0)),
        )

    if series_simulation and stored_signature == current_signature:
        results = series_simulation["results"]
        predicted_winner = series_simulation["predicted_winner"]
        higher_probability = series_simulation["higher_probability"]
        lower_probability = series_simulation["lower_probability"]
        favorite_probability = series_simulation["favorite_probability"]
        confidence = series_simulation["confidence"]
        most_likely = series_simulation["most_likely"]
        series_score_path = series_simulation["series_score_path"]

        render_prediction_result_card(
            label="Series prediction",
            winner=predicted_winner,
            probability=favorite_probability,
            confidence=confidence,
            context="series",
            note=f"Most likely: {most_likely}",
        )

        render_matchup_cards(
            team_a=higher_seed_team,
            team_b=lower_seed_team,
            probability_a=higher_probability,
            probability_b=lower_probability,
            winner=predicted_winner,
            label="Series win probability",
        )

        render_matchup_comparison_panel(
            team_a=higher_seed_team,
            team_b=lower_seed_team,
            title="Star comparison",
            key_prefix=f"series_{higher_seed_team}_{lower_seed_team}_{seed}",
        )

        render_series_score_simulation_path(
            path=series_score_path,
            higher_seed_team=higher_seed_team,
            lower_seed_team=lower_seed_team,
        )

        render_series_momentum_map(
            series_state=build_series_state_for_matchup(
                higher_seed_team=higher_seed_team,
                lower_seed_team=lower_seed_team,
            ),
            team_adjustments=team_adjustments,
            simulations=int(simulations),
            seed=int(seed),
        )

        render_matchup_explanation(
            team_a=higher_seed_team,
            team_b=lower_seed_team,
            predicted_winner=predicted_winner,
            winner_probability=favorite_probability,
            team_adjustments=team_adjustments,
        )

        report = create_prediction_report(
            title=f"NBA Best-of-7 Series Prediction: {higher_seed_team} vs {lower_seed_team}",
            predicted_winner=predicted_winner,
            probabilities={
                higher_seed_team: higher_probability,
                lower_seed_team: lower_probability,
            },
            confidence=confidence,
            most_likely_result=most_likely,
        )

        with st.expander("Details", expanded=False):
            chart_data = pd.DataFrame(
                {
                    "Team": [higher_seed_team, lower_seed_team],
                    "Series Win Probability": [higher_probability, lower_probability],
                }
            ).set_index("Team")
            st.bar_chart(chart_data)

            display_table = results["result_table"].copy()
            display_table["Probability"] = display_table["Probability"].map(
                "{:.1%}".format
            )
            st.dataframe(display_table, width="stretch")

            probability_cache = results["probability_cache"]
            rows = []

            for (home_team, away_team), probability in probability_cache.items():
                rows.append(
                    {
                        "Home Team": home_team,
                        "Away Team": away_team,
                        "Home Win Probability": probability,
                    }
                )

            cache_df = pd.DataFrame(rows)
            cache_df["Home Win Probability"] = cache_df[
                "Home Win Probability"
            ].map("{:.1%}".format)
            st.dataframe(cache_df, width="stretch")
            render_report_download(
                report_text=report,
                file_name="nba_series_prediction_report.txt",
                key="series_report",
            )


def render_current_playoff_series_section() -> None:
    """Render current playoff series tracker and rest-of-series predictions."""
    st.header("Current")

    series_states = build_current_playoff_series_states()

    if not series_states:
        st.warning("No playoff series data found. Run `python src/collect_data.py` first.")
        return

    valid_series_states = [
        state
        for state in series_states
        if all(is_known_team_name(team) for team in state["teams"])
    ]
    active_states = [state for state in valid_series_states if not state["completed"]]
    completed_states = [state for state in valid_series_states if state["completed"]]
    active_teams = sorted(
        {
            team
            for state in active_states
            for team in state["teams"]
        }
    )

    total_games = sum(int(state["games_played"]) for state in valid_series_states)
    st.caption(
        f"{len(active_states)} active series / "
        f"{len(completed_states)} complete / {total_games} games logged"
    )

    if active_teams:
        team_adjustments = render_official_availability_adjustments(
            selected_teams=active_teams,
            key_prefix="current_series",
        )
    else:
        team_adjustments = {}

    with st.expander("Settings", expanded=False):
        sim_col, seed_col = st.columns(2)

        with sim_col:
            simulations = st.slider(
                "Result scale",
                min_value=100,
                max_value=10000,
                value=3000,
                step=100,
                key="current_series_simulations",
            )

        with seed_col:
            seed = st.number_input(
                "Calculation ID",
                min_value=1,
                max_value=999999,
                value=77,
                step=1,
                key="current_series_seed",
            )

    if not active_states:
        st.info("No active playoff series in the current data.")

    projected_states = []

    with st.spinner("Updating current series..."):
        for index, state in enumerate(active_states):
            wins = state["wins"]
            schedule = get_schedule_for_series(state)

            next_game_number = state["next_game_number"]
            next_home_team = get_scheduled_home_team_for_series_game(
                state["home_court_team"],
                state["other_team"],
                next_game_number,
            )
            next_away_team = (
                state["other_team"]
                if next_home_team == state["home_court_team"]
                else state["home_court_team"]
            )

            playoff_context = build_playoff_context(
                home_team=next_home_team,
                away_team=next_away_team,
                game_number=next_game_number,
                home_series_wins=wins.get(next_home_team, 0),
                away_series_wins=wins.get(next_away_team, 0),
            )
            next_winner, next_home_probability, next_away_probability, _ = predict_game(
                home_team=next_home_team,
                away_team=next_away_team,
                team_adjustments=team_adjustments,
                playoff_context=playoff_context,
            )

            results = simulate_remaining_current_series(
                series_state=state,
                simulations=simulations,
                seed=int(seed) + index,
                team_adjustments=team_adjustments,
            )
            probabilities = results["series_probabilities"]
            predicted_series_winner = max(probabilities, key=probabilities.get)
            next_winner_probability = (
                next_home_probability
                if next_winner == next_home_team
                else next_away_probability
            )

            projected_states.append(
                {
                    "state": state,
                    "next_game_number": next_game_number,
                    "next_home_team": next_home_team,
                    "next_away_team": next_away_team,
                    "next_winner": next_winner,
                    "next_home_probability": next_home_probability,
                    "next_away_probability": next_away_probability,
                    "next_winner_probability": next_winner_probability,
                    "results": results,
                    "probabilities": probabilities,
                    "predicted_series_winner": predicted_series_winner,
                    "schedule": schedule,
                    "schedule_summary": summarize_remaining_series_schedule(schedule),
                }
            )

    if projected_states:
        render_section_kicker("Live series")
        column_count = min(2, len(projected_states))
        columns = st.columns(column_count)

        for index, projection in enumerate(projected_states):
            state = projection["state"]
            probabilities = projection["probabilities"]
            predicted_series_winner = projection["predicted_series_winner"]
            series_probability = probabilities[predicted_series_winner]
            next_game_line, remaining_games = build_remaining_series_lines(
                next_winner=projection["next_winner"],
                next_probability=projection["next_winner_probability"],
                schedule=projection["schedule"],
            )

            with columns[index % column_count]:
                render_series_score_card(
                    state,
                    projection=(
                        f"Series: {predicted_series_winner} "
                        f"{series_probability:.0%}"
                    ),
                    next_game=next_game_line,
                    remaining_games=remaining_games,
                )

        render_section_kicker("Details")

        for projection in projected_states:
            state = projection["state"]

            with st.expander(state["series_key"], expanded=False):
                detail_left, detail_right = st.columns([1.1, 0.9], gap="medium")

                with detail_left:
                    render_section_kicker("Remaining schedule")
                    render_remaining_series_schedule(projection["schedule"])

                    render_section_kicker("Completed games")
                    st.dataframe(
                        pd.DataFrame(state["game_rows"]),
                        width="stretch",
                        hide_index=True,
                    )

                with detail_right:
                    render_prediction_result_card(
                        label=f"Game {projection['next_game_number']}",
                        winner=projection["next_winner"],
                        probability=projection["next_winner_probability"],
                        confidence=get_game_confidence_label(
                            projection["next_winner_probability"]
                        ),
                        context="game",
                        note=(
                            f"{projection['next_home_team']} home vs "
                            f"{projection['next_away_team']}"
                        ),
                    )
                    render_matchup_cards(
                        team_a=projection["next_home_team"],
                        team_b=projection["next_away_team"],
                        probability_a=projection["next_home_probability"],
                        probability_b=projection["next_away_probability"],
                        winner=projection["next_winner"],
                        label=f"Game {projection['next_game_number']}",
                    )

                    render_matchup_comparison_panel(
                        team_a=projection["next_away_team"],
                        team_b=projection["next_home_team"],
                        title="Star comparison",
                        key_prefix=(
                            f"current_series_{state['series_key']}"
                            f"_{projection['next_game_number']}"
                        ),
                    )

                    render_series_momentum_map(
                        series_state=state,
                        team_adjustments=team_adjustments,
                        simulations=int(simulations),
                        seed=int(seed),
                    )

                    result_table = projection["results"]["result_table"].copy()
                    result_table["Probability"] = result_table["Probability"].map(
                        "{:.1%}".format
                    )
                    render_section_kicker("Series results")
                    st.dataframe(result_table, width="stretch", hide_index=True)

    if completed_states:
        with st.expander("Completed series", expanded=False):
            for state in completed_states:
                render_series_score_card(
                    state,
                    projection=f"Winner: {state['leader']}",
                )


def render_team_seed_selectors(
    conference_name: str,
    teams: list[str],
    defaults: list[str],
) -> list[str]:
    """Render 1-8 seed dropdowns for one conference."""
    render_section_kicker(conference_name)

    selected_teams = []

    for seed_number in range(1, 9):
        default_team = defaults[seed_number - 1]
        selected_team = st.selectbox(
            f"{conference_name} Seed {seed_number}",
            teams,
            index=safe_team_index(teams, default_team),
            key=f"{conference_name.lower().replace(' ', '_')}_seed_{seed_number}",
        )
        selected_teams.append(selected_team)

    return selected_teams


def render_bracket_section(teams: list[str]) -> None:
    """Render full playoff bracket simulator."""
    st.header("Full Playoff Bracket")

    col1, col2 = st.columns(2)

    with col1:
        east_teams = render_team_seed_selectors(
            conference_name="Eastern Conference",
            teams=teams,
            defaults=DEFAULT_EAST_TEAMS,
        )

    with col2:
        west_teams = render_team_seed_selectors(
            conference_name="Western Conference",
            teams=teams,
            defaults=DEFAULT_WEST_TEAMS,
        )

    all_selected_teams = east_teams + west_teams

    if not validate_unique_teams(all_selected_teams):
        st.warning("Each playoff seed must be a unique team.")
        return

    render_team_logo_strip(all_selected_teams)

    team_adjustments = render_official_availability_adjustments(
        selected_teams=all_selected_teams,
        key_prefix="bracket",
    )

    sim_col, seed_col = st.columns(2)

    with sim_col:
        bracket_simulations = st.slider(
            "Bracket simulations",
            min_value=100,
            max_value=10000,
            value=3000,
            step=100,
        )

    with seed_col:
        bracket_seed = st.number_input(
            "Simulation ID",
            min_value=1,
            max_value=999999,
            value=123,
            step=1,
            key="bracket_seed",
        )

    if st.button("Simulate playoffs", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = simulate_full_playoff_bracket(
            east_teams=east_teams,
            west_teams=west_teams,
            simulations=bracket_simulations,
            seed=int(bracket_seed),
            progress_bar=progress_bar,
            status_text=status_text,
            team_adjustments=team_adjustments,
        )

        winner = results.iloc[0]["Team"]
        winner_probability = results.iloc[0]["Championship Probability"]
        confidence = get_bracket_confidence_label(winner_probability)

        render_prediction_result_card(
            label="Championship prediction",
            winner=winner,
            probability=winner_probability,
            confidence=confidence,
            context="series",
            note="Full bracket simulation",
        )

        render_team_card(
            team_name=winner,
            probability=winner_probability,
            is_winner=True,
            label="Championship probability",
        )

        chart_data = results.set_index("Team")[["Championship Probability"]]
        st.bar_chart(chart_data)

        path_rows = simulate_single_full_bracket_path(
            east_teams=east_teams,
            west_teams=west_teams,
            seed=int(bracket_seed),
            team_adjustments=team_adjustments,
            probability_cache=results.attrs.get("probability_cache"),
        )
        render_bracket_visualization(path_rows)

        display_results = results.copy()
        display_results["Finals Probability"] = display_results[
            "Finals Probability"
        ].map("{:.1%}".format)
        display_results["Championship Probability"] = display_results[
            "Championship Probability"
        ].map("{:.1%}".format)

        with st.expander("Champion probability table", expanded=False):
            st.dataframe(display_results, width="stretch")

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download championship probabilities as CSV",
            data=csv_data,
            file_name="nba_championship_probabilities.csv",
            mime="text/csv",
        )

        report = create_prediction_report(
            title="NBA Full Playoff Bracket Prediction",
            predicted_winner=winner,
            probabilities={
                row["Team"]: row["Championship Probability"]
                for _, row in results.head(8).iterrows()
            },
            confidence=confidence,
        )

        render_report_download(
            report_text=report,
            file_name="nba_playoff_bracket_prediction_report.txt",
            key="bracket_report",
        )


def render_elo_rankings_section() -> None:
    """Render team Elo rankings."""
    st.header("Elo Rankings")

    strength = load_team_strength()
    rankings = strength[["TEAM_NAME", "ELO"]].copy()
    rankings = rankings.sort_values("ELO", ascending=False).reset_index(drop=True)
    rankings["Rank"] = range(1, len(rankings) + 1)
    rankings = rankings[["Rank", "TEAM_NAME", "ELO"]].rename(
        columns={"TEAM_NAME": "Team"}
    )

    top_team = rankings.iloc[0]
    render_result_callout(
        label="Top Elo",
        title=str(top_team["Team"]),
        meta=f"{float(top_team['ELO']):.0f}",
    )

    st.bar_chart(rankings.set_index("Team")[["ELO"]])
    st.dataframe(rankings, width="stretch")

    csv_data = rankings.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Elo rankings as CSV",
        data=csv_data,
        file_name="nba_elo_rankings.csv",
        mime="text/csv",
    )


def render_team_strength_section() -> None:
    """Render current team strength diagnostics."""
    st.header("Team Strength")

    strength = load_team_strength().copy()
    strength = strength.sort_values("ELO", ascending=False)

    display_columns = [
        "TEAM_NAME",
        "ELO",
        "SEASON_WIN_PCT",
        "SEASON_AVG_PLUS_MINUS",
        "ROLLING_WIN_PCT_10",
        "ROLLING_PLUS_MINUS_10",
        "ROLLING_NET_RATING_10",
        "ROLLING_OFF_RATING_10",
        "ROLLING_DEF_RATING_10",
        "ROLLING_EFG_PCT_10",
        "ROLLING_TOV_RATE_10",
        "ROLLING_OREB_PCT_10",
        "ROLLING_WIN_PCT_5",
        "ROLLING_PLUS_MINUS_5",
        "DAYS_REST",
        "IS_BACK_TO_BACK",
        "GAMES_LAST_7_DAYS",
        "CURRENT_ROAD_STREAK",
        "PLAYER_TOP_8",
        "PLAYER_DEPTH",
        "STAR_COUNT",
        "GAME_DATE",
    ]

    display = strength[display_columns].rename(
        columns={
            "TEAM_NAME": "Team",
            "ELO": "Elo",
            "SEASON_WIN_PCT": "Season Win %",
            "SEASON_AVG_PLUS_MINUS": "Season Avg Plus/Minus",
            "ROLLING_WIN_PCT_10": "Last 10 Win %",
            "ROLLING_PLUS_MINUS_10": "Last 10 Plus/Minus",
            "ROLLING_NET_RATING_10": "Last 10 Net Rating",
            "ROLLING_OFF_RATING_10": "Last 10 Off Rating",
            "ROLLING_DEF_RATING_10": "Last 10 Def Rating",
            "ROLLING_EFG_PCT_10": "Last 10 eFG%",
            "ROLLING_TOV_RATE_10": "Last 10 TOV Rate",
            "ROLLING_OREB_PCT_10": "Last 10 OREB%",
            "ROLLING_WIN_PCT_5": "Last 5 Win %",
            "ROLLING_PLUS_MINUS_5": "Last 5 Plus/Minus",
            "DAYS_REST": "Current Rest Days",
            "IS_BACK_TO_BACK": "Back-to-Back Flag",
            "GAMES_LAST_7_DAYS": "Games Last 7 Days",
            "CURRENT_ROAD_STREAK": "Current Road Streak",
            "PLAYER_TOP_8": "Top 8 Player Strength",
            "PLAYER_DEPTH": "Rotation Depth Strength",
            "STAR_COUNT": "Star Count",
            "GAME_DATE": "Last Game Date",
        }
    )

    latest_date = format_status_timestamp(display["Last Game Date"].max())
    top_strength = display.iloc[0]
    render_result_callout(
        label="Current top team",
        title=str(top_strength["Team"]),
        meta=f"Elo {float(top_strength['Elo']):.0f} - data through {latest_date}",
    )

    st.dataframe(display, width="stretch")

    csv_data = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current team strength as CSV",
        data=csv_data,
        file_name="current_team_strength.csv",
        mime="text/csv",
    )


def render_model_info_section(teams: list[str]) -> None:
    """Render model metrics and limitations."""
    st.header("Model Info")

    bundle = load_model_bundle()
    model_name = str(bundle.get("model_name", "Unknown model"))
    feature_count = len(bundle.get("feature_columns", []))
    blend_settings = bundle.get("blend_settings", {}) or {}
    metrics = load_model_metrics()
    backtest_metrics = load_backtest_metrics()
    calibration_metrics = load_calibration_metrics()
    model_accuracy = "Unknown"

    if not metrics.empty and "Accuracy" in metrics.columns:
        if "Model" in metrics.columns:
            model_rows = metrics[metrics["Model"].eq(model_name)]
        else:
            model_rows = pd.DataFrame()

        if model_rows.empty:
            model_rows = metrics.sort_values("Accuracy", ascending=False).head(1)

        if not model_rows.empty:
            model_accuracy = f"{float(model_rows.iloc[0]['Accuracy']):.1%}"

    if blend_settings:
        blend_label = (
            f"{float(blend_settings.get('model_probability_weight', MODEL_PROBABILITY_WEIGHT)):.0%} "
            "model / "
            f"{float(blend_settings.get('elo_probability_weight', ELO_PROBABILITY_WEIGHT)):.0%} Elo"
        )
    else:
        blend_label = (
            f"{MODEL_PROBABILITY_WEIGHT:.0%} model / {ELO_PROBABILITY_WEIGHT:.0%} Elo"
        )

    render_status_grid(
        [
            {"label": "Active Model", "value": shorten_model_name(model_name)},
            {"label": "Features", "value": str(feature_count)},
            {"label": "Accuracy", "value": model_accuracy},
            {"label": "Blend", "value": blend_label},
        ]
    )

    calibration_col, backtest_col = st.columns(2, gap="medium")

    with calibration_col:
        render_section_kicker("Calibration")
        render_model_calibration_chart(calibration_metrics)

    with backtest_col:
        render_section_kicker("Season Backtests")
        render_backtest_chart(backtest_metrics)

    render_section_kicker("Team Reliability")
    render_team_backtest_panel(teams)

    if not metrics.empty:
        formatted_metrics = metrics.copy()

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "Brier_Score",
            "Log_Loss",
        ]

        for column in metric_columns:
            if column in formatted_metrics.columns:
                formatted_metrics[column] = formatted_metrics[column].map("{:.3f}".format)

        with st.expander("Model comparison", expanded=False):
            st.dataframe(formatted_metrics, width="stretch")

            chart_metrics = metrics.set_index("Model")[["Accuracy", "ROC_AUC"]]
            st.bar_chart(chart_metrics)
    else:
        st.warning("No model metrics found. Run `python src/train_model.py` again.")

    if not backtest_metrics.empty:
        formatted_backtests = backtest_metrics.copy()

        for column in [
            "Accuracy",
            "ROC_AUC",
            "Brier_Score",
            "Log_Loss",
            "Blend_Brier_Score",
            "Blend_Log_Loss",
            "Blend_ROC_AUC",
        ]:
            if column in formatted_backtests.columns:
                formatted_backtests[column] = formatted_backtests[column].map(
                    "{:.3f}".format
                )

        with st.expander("Rolling season backtests", expanded=False):
            st.dataframe(formatted_backtests, width="stretch")

    if not calibration_metrics.empty:
        formatted_calibration = calibration_metrics.copy()

        for column in [
            "Average_Predicted_Probability",
            "Actual_Home_Win_Rate",
        ]:
            if column in formatted_calibration.columns:
                formatted_calibration[column] = formatted_calibration[column].map(
                    "{:.1%}".format
                )

        with st.expander("Probability calibration", expanded=False):
            st.dataframe(formatted_calibration, width="stretch")

    with st.expander("Availability and limits", expanded=False):
        render_section_kicker("Availability")
        st.html(
            """
            <div class="compact-note">
                Official injury-report statuses are combined with each player's
                stat-based impact score before predictions are made.
            </div>
            """
        )
        render_section_kicker("Limits")
        st.html(
            """
            <div class="compact-note">
                The model does not include betting odds, travel detail,
                matchup-specific defensive assignments, or injury changes made after
                the report was refreshed.
            </div>
            """
        )


def main() -> None:
    """Render Streamlit app."""
    st.set_page_config(
        page_title="NBA Betting Dashboard",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_app_header()

    if not MODEL_PATH.exists():
        st.error("Missing model file. Run `python src/train_model.py` first.")
        return

    if not FEATURES_PATH.exists():
        st.error("Missing features file. Run `python src/features.py` first.")
        return

    if not TEAM_STRENGTH_PATH.exists():
        st.error("Missing current team strength file. Run `python src/team_strength.py` first.")
        return

    strength = load_team_strength()
    teams = get_teams_from_strength(strength)

    views = [
        {"label": "Home", "icon": "🏠"},
        {"label": "Today", "icon": "📅"},
        {"label": "Game", "icon": "🎮"},
        {"label": "Series", "icon": "🔁"},
        {"label": "Bracket", "icon": "🏆"},
        {"label": "Teams", "icon": "👥"},
    ]
    view_key = "view_switcher"
    sidebar_compact_key = "sidebar_compact"
    if view_key not in st.session_state:
        st.session_state[view_key] = "Home"
    if sidebar_compact_key not in st.session_state:
        st.session_state[sidebar_compact_key] = True
    valid_views = {view["label"] for view in views}
    if st.session_state[view_key] not in valid_views:
        st.session_state[view_key] = "Home"

    with st.sidebar:
        if st.session_state[sidebar_compact_key]:
            st.markdown(
                """
                <style>
                    section[data-testid="stSidebar"],
                    section[data-testid="stSidebar"] > div:first-child {
                        width: 4.9rem;
                        min-width: 4.9rem;
                        max-width: 4.9rem;
                    }
                    section[data-testid="stSidebar"] .sidebar-brand-text,
                    section[data-testid="stSidebar"] .sidebar-nav-title {
                        display: none;
                    }
                    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
                        display: none;
                    }
                    section[data-testid="stSidebar"] div[role="radiogroup"] label {
                        justify-content: center;
                        padding: 0.35rem 0.25rem;
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
                        width: 10.25rem;
                        min-width: 10.25rem;
                        max-width: 10.25rem;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">🏀</div>
                <div class="sidebar-brand-text">
                    <div class="sidebar-brand-name">NBA Predictor</div>
                    <div class="sidebar-brand-sub">Betting dashboard</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-toggle-wrap">', unsafe_allow_html=True)
        toggle_label = "»" if st.session_state[sidebar_compact_key] else "«"
        if st.button(
            toggle_label,
            key="sidebar_toggle",
            use_container_width=True,
        ):
            st.session_state[sidebar_compact_key] = not st.session_state[sidebar_compact_key]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="sidebar-nav-title">View</div>',
            unsafe_allow_html=True,
        )
        for view in views:
            selected = st.session_state[view_key] == view["label"]
            label = view["icon"] if st.session_state[sidebar_compact_key] else f"{view['icon']} {view['label']}"
            if st.button(
                label,
                key=f"sidebar_nav_{view['label'].lower()}",
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                st.session_state[view_key] = view["label"]
                st.rerun()

    selected_view = st.session_state[view_key]

    if selected_view == "Home":
        render_home_dashboard(teams)
    elif selected_view == "Today":
        render_today_games_section(teams)
    elif selected_view == "Game":
        render_single_game_section(teams)
    elif selected_view == "Series":
        render_series_section(teams)
    elif selected_view == "Bracket":
        render_bracket_section(teams)
    elif selected_view == "Teams":
        render_team_profiles_section(teams)
    else:
        render_home_dashboard(teams)

    render_today_player_profile_dialog()


if __name__ == "__main__":
    main()
