# app.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import app_mlb as mlb
import app_nba as nba
import app_nhl as nhl


DATA_DIR = Path("data")
NBA_BET_TRACKER_PATH = DATA_DIR / "nba_bet_tracker.csv"
FAVORITES_PATH = DATA_DIR / "favorite_teams.csv"
PREDICTION_TIMELINE_PATH = DATA_DIR / "prediction_timeline.csv"

SPORT_OPTIONS = ["Betting Hub", "NBA", "MLB", "NHL"]
HUB_PAGES = [
    "Overview",
    "Games",
    "Betting",
    "Futures",
    "Tracker",
]
NBA_PAGES = [
    "Home",
    "Games",
    "Playoffs",
    "Betting",
    "Standings",
]
MLB_PAGES = [
    "Home",
    "Games",
    "Playoffs",
    "Betting",
    "Standings",
]
NHL_PAGES = [
    "Home",
    "Games",
    "Playoffs",
    "Betting",
    "Standings",
]
NBA_BET_TRACKER_COLUMNS = [
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
    "Notes",
]
PREDICTION_TIMELINE_COLUMNS = [
    "Snapshot Time",
    "Sport",
    "Game Date",
    "Matchup",
    "Away Team",
    "Home Team",
    "Predicted Winner",
    "Winner Probability",
    "Home Probability",
    "Away Probability",
    "Model Probability",
    "Elo Probability",
    "Confidence",
    "Watchability",
    "Prediction Note",
]
FAVORITE_COLUMNS = ["Sport", "Team", "Added At"]
NBA_CONFERENCES = {
    "Boston Celtics": "East",
    "Brooklyn Nets": "East",
    "New York Knicks": "East",
    "Philadelphia 76ers": "East",
    "Toronto Raptors": "East",
    "Chicago Bulls": "East",
    "Cleveland Cavaliers": "East",
    "Detroit Pistons": "East",
    "Indiana Pacers": "East",
    "Milwaukee Bucks": "East",
    "Atlanta Hawks": "East",
    "Charlotte Hornets": "East",
    "Miami Heat": "East",
    "Orlando Magic": "East",
    "Washington Wizards": "East",
    "Denver Nuggets": "West",
    "Minnesota Timberwolves": "West",
    "Oklahoma City Thunder": "West",
    "Portland Trail Blazers": "West",
    "Utah Jazz": "West",
    "Golden State Warriors": "West",
    "LA Clippers": "West",
    "Los Angeles Clippers": "West",
    "Los Angeles Lakers": "West",
    "Phoenix Suns": "West",
    "Sacramento Kings": "West",
    "Dallas Mavericks": "West",
    "Houston Rockets": "West",
    "Memphis Grizzlies": "West",
    "New Orleans Pelicans": "West",
    "San Antonio Spurs": "West",
}


@dataclass(frozen=True)
class SportConfig:
    """Registry entry used by the unified shell and future league templates."""

    key: str
    label: str
    accent: str
    standalone_app: str
    model_path: Path
    feature_path: Path
    strength_path: Path
    tracker_path: Path | None
    default_home: str
    default_away: str
    pages: tuple[str, ...]


SPORT_REGISTRY = {
    "NBA": SportConfig(
        key="NBA",
        label="NBA",
        accent="#c1121f",
        standalone_app="app_nba.py",
        model_path=nba.MODEL_PATH,
        feature_path=nba.FEATURES_PATH,
        strength_path=nba.TEAM_STRENGTH_PATH,
        tracker_path=NBA_BET_TRACKER_PATH,
        default_home="Boston Celtics",
        default_away="Denver Nuggets",
        pages=tuple(NBA_PAGES),
    ),
    "MLB": SportConfig(
        key="MLB",
        label="MLB",
        accent="#1d4ed8",
        standalone_app="app_mlb.py",
        model_path=mlb.MODEL_PATH,
        feature_path=mlb.FEATURES_PATH,
        strength_path=mlb.TEAM_STRENGTH_PATH,
        tracker_path=mlb.BET_TRACKER_PATH,
        default_home="Los Angeles Dodgers",
        default_away="New York Yankees",
        pages=tuple(MLB_PAGES),
    ),
    "NHL": SportConfig(
        key="NHL",
        label="NHL",
        accent="#0f766e",
        standalone_app="app_nhl.py",
        model_path=nhl.MODEL_PATH,
        feature_path=nhl.FEATURES_PATH,
        strength_path=nhl.TEAM_STRENGTH_PATH,
        tracker_path=nhl.BET_TRACKER_PATH,
        default_home="Carolina Hurricanes",
        default_away="Vegas Golden Knights",
        pages=tuple(NHL_PAGES),
    ),
}


def render_unified_css() -> None:
    """Render the shared app shell styling."""
    st.markdown(
        """
        <style>
            :root {
                --league-page: #f6f9fc;
                --league-ink: #132033;
                --league-muted: #617083;
                --league-line: #d5e0ea;
                --league-surface: #ffffff;
                --league-surface-soft: #edf6fb;
                --league-red: #d92d20;
                --league-blue: #1570ef;
                --league-cyan: #0891b2;
                --league-green: #079455;
                --league-gold: #dc8a00;
                --league-purple: #7a5af8;
                --league-lime: #65a30d;
                --league-shadow: 0 18px 42px rgba(19, 32, 51, 0.12);
                --league-radius: 8px;
            }

            .stApp,
            .stApp *,
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] * {
                font-family: "Avenir Next", "Sohne", "Segoe UI", sans-serif;
                letter-spacing: 0;
            }

            .stApp,
            [data-testid="stAppViewContainer"] {
                background:
                    linear-gradient(118deg, rgba(21, 112, 239, 0.13) 0%, rgba(21, 112, 239, 0) 31%),
                    linear-gradient(245deg, rgba(7, 148, 85, 0.14) 0%, rgba(7, 148, 85, 0) 29%),
                    linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(246, 249, 252, 0.98) 340px),
                    repeating-linear-gradient(135deg, rgba(19, 32, 51, 0.035) 0 1px, transparent 1px 16px),
                    var(--league-page);
                color: var(--league-ink);
            }

            [data-testid="stHeader"] {
                background: rgba(246, 249, 252, 0.76);
                border-bottom: 1px solid rgba(213, 224, 234, 0.72);
                backdrop-filter: blur(14px);
            }

            .block-container {
                max-width: 1420px;
                padding-top: 0.85rem;
                padding-bottom: 2.4rem;
            }

            h1, h2, h3 {
                color: var(--league-ink);
                letter-spacing: 0;
            }

            h1 {
                font-size: 1.72rem !important;
                font-weight: 920 !important;
                margin-bottom: 0.55rem !important;
            }

            h2, h3 {
                font-weight: 880 !important;
            }

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div:first-child {
                width: 14rem;
                min-width: 14rem;
                max-width: 14rem;
                background:
                    linear-gradient(180deg, rgba(19, 32, 51, 0.98) 0 5.35rem, rgba(255, 255, 255, 0.97) 5.35rem 100%),
                    #ffffff;
                border-right: 1px solid var(--league-line);
                box-shadow: 10px 0 34px rgba(19, 32, 51, 0.08);
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
                color: #667085;
                font-size: 0.74rem;
                font-weight: 900;
                letter-spacing: 0.08em;
                margin: 0.65rem 0 0.45rem;
                text-transform: uppercase;
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:first-child h3 {
                color: #ffffff;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 0.35rem;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid #d8e2ec;
                border-radius: var(--league-radius);
                box-shadow: 0 3px 10px rgba(19, 32, 51, 0.05);
                margin-bottom: 0.35rem;
                min-height: 2.35rem;
                padding: 0.35rem 0.55rem;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, #132033, #1570ef 76%, #079455);
                border-color: transparent;
                box-shadow: 0 12px 24px rgba(21, 112, 239, 0.22);
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
                color: #ffffff;
                font-weight: 850;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label p {
                color: #344054;
                font-size: 0.86rem;
                font-weight: 760;
            }

            div[data-testid="stSegmentedControl"] label {
                border-radius: var(--league-radius) !important;
            }

            div[data-testid="stTabs"] {
                margin: 0.4rem 0 0.85rem;
            }

            div[data-testid="stTabs"] div[role="tablist"] {
                gap: 0.45rem;
                overflow-x: auto;
            }

            div[data-testid="stTabs"] button[role="tab"] {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid #d8e2ec;
                border-radius: var(--league-radius);
                box-shadow: 0 5px 14px rgba(19, 32, 51, 0.05);
                color: #334155;
                min-height: 2.45rem;
                padding: 0.42rem 0.8rem;
            }

            div[data-testid="stTabs"] button[role="tab"] p {
                color: inherit;
                font-size: 0.84rem;
                font-weight: 850;
                letter-spacing: 0;
                white-space: nowrap;
            }

            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                background: linear-gradient(135deg, #132033, #1570ef 64%, #079455);
                border-color: transparent;
                box-shadow: 0 12px 24px rgba(21, 112, 239, 0.19);
                color: #ffffff;
            }

            div[data-testid="stTabs"] button[role="tab"]:hover {
                border-color: #98a2b3;
                color: #101828;
            }

            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]:hover {
                color: #ffffff;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid #d8e2ec;
                border-radius: var(--league-radius);
                box-shadow: 0 8px 22px rgba(19, 32, 51, 0.07);
                padding: 0.7rem 0.8rem;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                border: 1px solid rgba(213, 224, 234, 0.95);
                border-radius: var(--league-radius);
                box-shadow: 0 12px 28px rgba(19, 32, 51, 0.07);
                overflow: hidden;
            }

            div.stButton > button,
            div[data-testid="stDownloadButton"] button,
            a[data-testid="stLinkButton"] {
                border-radius: var(--league-radius) !important;
                font-weight: 840 !important;
            }

            div.stButton > button[kind="primary"],
            div.stButton > button:hover,
            div[data-testid="stDownloadButton"] button:hover,
            a[data-testid="stLinkButton"]:hover {
                border-color: transparent !important;
                box-shadow: 0 10px 22px rgba(21, 112, 239, 0.17) !important;
            }

            .league-topbar {
                align-items: center;
                background:
                    linear-gradient(120deg, rgba(255, 255, 255, 0.09) 0 18%, transparent 18% 32%, rgba(255, 255, 255, 0.07) 32% 44%, transparent 44%),
                    linear-gradient(135deg, #132033 0%, #18427a 37%, #1570ef 66%, #079455 100%);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: var(--league-radius);
                box-shadow: 0 22px 52px rgba(19, 32, 51, 0.22);
                display: flex;
                gap: 1rem;
                justify-content: space-between;
                margin-bottom: 0.85rem;
                overflow: hidden;
                padding: 1.05rem 1.15rem;
                position: relative;
            }

            .league-brand {
                align-items: center;
                display: flex;
                gap: 0.75rem;
                min-width: 0;
                position: relative;
                z-index: 1;
            }

            .league-mark {
                align-items: center;
                background:
                    linear-gradient(135deg, #ffffff, #dff7ea);
                color: #ffffff;
                display: flex;
                font-size: 0;
                font-weight: 950;
                height: 2.55rem;
                justify-content: center;
                letter-spacing: 0.04em;
                position: relative;
                width: 2.55rem;
                border-radius: var(--league-radius);
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.58), 0 10px 24px rgba(0, 0, 0, 0.16);
            }

            .league-mark::before {
                color: #132033;
                content: "SP";
                font-size: 0.78rem;
                letter-spacing: 0.02em;
            }

            .league-title {
                color: #ffffff;
                font-size: 1.48rem;
                font-weight: 930;
                letter-spacing: 0;
                line-height: 1.1;
            }

            .league-subtitle {
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.86rem;
                margin-top: 0.12rem;
            }

            .league-tabs {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                justify-content: flex-end;
                position: relative;
                z-index: 1;
            }

            .league-pill {
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.22);
                border-radius: 999px;
                color: rgba(255, 255, 255, 0.82);
                font-size: 0.8rem;
                font-weight: 800;
                padding: 0.44rem 0.78rem;
            }

            .league-pill.active {
                background: #ffffff;
                border-color: #ffffff;
                color: #132033;
                box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
            }

            .league-banner {
                align-items: center;
                background:
                    linear-gradient(100deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.86)),
                    linear-gradient(135deg, var(--league-accent, var(--league-blue)), rgba(7, 148, 85, 0.52));
                border: 1px solid rgba(213, 224, 234, 0.9);
                border-left: 7px solid var(--league-accent, var(--league-blue));
                border-radius: var(--league-radius);
                box-shadow: 0 14px 32px rgba(19, 32, 51, 0.09);
                display: flex;
                justify-content: space-between;
                margin: 0.55rem 0 1rem;
                padding: 0.92rem 1rem;
            }

            .league-banner-title {
                font-size: 1.08rem;
                font-weight: 920;
            }

            .league-banner-meta {
                color: var(--league-muted);
                font-size: 0.82rem;
                margin-top: 0.1rem;
            }

            .league-badge {
                background: #132033;
                border: 1px solid rgba(19, 32, 51, 0.08);
                border-radius: var(--league-radius);
                color: #ffffff;
                display: grid;
                gap: 0.1rem;
                justify-items: end;
                min-width: 8.7rem;
                padding: 0.48rem 0.62rem;
                white-space: nowrap;
            }

            .league-badge-label {
                color: rgba(255, 255, 255, 0.62);
                font-size: 0.62rem;
                font-weight: 860;
                text-transform: uppercase;
            }

            .league-badge-value {
                color: #ffffff;
                font-size: 0.86rem;
                font-weight: 900;
            }

            .hub-grid {
                display: grid;
                gap: 0.95rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin: 0.85rem 0 1rem;
            }

            .sport-card {
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.9)),
                    linear-gradient(135deg, rgba(21, 112, 239, 0.16), rgba(7, 148, 85, 0.13));
                border: 1px solid rgba(213, 224, 234, 0.9);
                border-radius: var(--league-radius);
                box-shadow: var(--league-shadow);
                overflow: hidden;
                padding: 1.08rem;
                position: relative;
            }

            .sport-card::before {
                content: "";
                inset: 0 0 auto 0;
                height: 0.34rem;
                position: absolute;
            }

            .sport-card.nba {
                border-top: 0;
            }

            .sport-card.nba::before {
                background: linear-gradient(90deg, var(--league-red), var(--league-gold), var(--league-blue));
            }

            .sport-card.mlb {
                border-top: 0;
            }

            .sport-card.mlb::before {
                background: linear-gradient(90deg, var(--league-blue), var(--league-cyan), var(--league-green));
            }

            .sport-card.nhl {
                border-top: 0;
            }

            .sport-card.nhl::before {
                background: linear-gradient(90deg, #0f766e, #38bdf8, #f8fafc);
            }

            .sport-card-head {
                align-items: flex-start;
                display: flex;
                gap: 0.7rem;
                justify-content: space-between;
            }

            .sport-name {
                font-size: 1.26rem;
                font-weight: 940;
                letter-spacing: 0;
            }

            .sport-note {
                color: var(--league-muted);
                font-size: 0.82rem;
                margin-top: 0.2rem;
            }

            .sport-chip {
                background: #132033;
                border: 1px solid rgba(19, 32, 51, 0.08);
                border-radius: 999px;
                color: #ffffff;
                font-size: 0.76rem;
                font-weight: 850;
                padding: 0.32rem 0.56rem;
            }

            .sport-metrics {
                display: grid;
                gap: 0.55rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-top: 0.8rem;
            }

            .metric-tile {
                background:
                    linear-gradient(180deg, #ffffff, #f7fbff);
                border: 1px solid #dbe4ee;
                border-radius: var(--league-radius);
                min-height: 4.65rem;
                padding: 0.65rem;
            }

            .metric-label {
                color: var(--league-muted);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
            }

            .metric-value {
                color: var(--league-ink);
                font-size: 1rem;
                font-weight: 880;
                line-height: 1.15;
                margin-top: 0.25rem;
                overflow-wrap: anywhere;
            }

            .metric-note {
                color: var(--league-muted);
                font-size: 0.74rem;
                margin-top: 0.15rem;
            }

            .cross-table-wrap {
                margin-top: 0.8rem;
            }

            .edge-positive {
                color: #047857;
                font-weight: 850;
            }

            .edge-negative {
                color: #b42318;
                font-weight: 850;
            }

            .feature-grid {
                display: grid;
                gap: 0.8rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin: 0.75rem 0 1rem;
            }

            .feature-card,
            .prediction-card,
            .trust-card,
            .template-card,
            .dashboard-card,
            .status-card,
            .game-card,
            .score-card,
            .score-sim-card,
            .profile-hero,
            .result-callout,
            .team-card,
            .series-section-box,
            .series-score-row,
            .player-row,
            .game-center-panel,
            .betting-panel,
            .betting-card,
            .betting-card-shell,
            .bet-slip,
            .saved-pick-card {
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 252, 255, 0.98));
                border: 1px solid rgba(213, 224, 234, 0.95);
                border-radius: var(--league-radius);
                box-shadow: 0 12px 30px rgba(19, 32, 51, 0.08);
                padding: 0.95rem;
            }

            .betting-hero,
            .hero-card,
            .dashboard-hero {
                border-radius: var(--league-radius);
                box-shadow: 0 18px 42px rgba(19, 32, 51, 0.12);
            }

            .betting-hero,
            .game-center-panel {
                background:
                    linear-gradient(135deg, rgba(19, 32, 51, 0.96), rgba(21, 112, 239, 0.9) 58%, rgba(7, 148, 85, 0.9)) !important;
                border-color: rgba(255, 255, 255, 0.16) !important;
            }

            .betting-hero *,
            .game-center-panel * {
                color: #ffffff;
            }

            .betting-side-tile,
            .betting-simple-stat,
            .bet-slip-stat,
            .saved-pick-detail {
                background: #f7fbff;
                border: 1px solid #dbe8f4;
                border-radius: var(--league-radius);
            }

            .betting-side-selected {
                background: #ecfdf3 !important;
                border-color: #73d6a2 !important;
                box-shadow: inset 0 0 0 1px rgba(7, 148, 85, 0.16);
            }

            .betting-hero {
                align-items: center;
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                margin: 0.35rem 0 1rem;
                min-height: 118px;
                padding: 1.15rem 1.25rem;
            }

            .betting-hero-title {
                font-size: 1.45rem;
                font-weight: 950;
                letter-spacing: 0;
                line-height: 1.08;
            }

            .betting-hero-note {
                font-size: 0.83rem;
                font-weight: 650;
                line-height: 1.35;
                margin-top: 0.25rem;
                opacity: 0.88;
            }

            .betting-hero-stat {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: var(--league-radius);
                min-width: 230px;
                padding: 0.85rem;
            }

            .betting-hero-stat-label {
                font-size: 0.72rem;
                font-weight: 850;
                opacity: 0.82;
                text-transform: uppercase;
            }

            .betting-hero-stat-value {
                font-size: 1.2rem;
                font-weight: 950;
                line-height: 1.15;
                margin-top: 0.2rem;
            }

            .betting-card-shell {
                border-left: 6px solid var(--card-accent, #1570ef);
                margin-bottom: 1rem;
                padding: 1rem;
            }

            .betting-card-head,
            .betting-side-top {
                align-items: flex-start;
                display: flex;
                gap: 0.75rem;
                justify-content: space-between;
            }

            .betting-game-title {
                color: #132033;
                font-size: 1rem;
                font-weight: 950;
                line-height: 1.18;
            }

            .betting-game-meta {
                color: #5d728a;
                font-size: 0.76rem;
                font-weight: 700;
                line-height: 1.35;
                margin-top: 0.18rem;
            }

            .betting-rank-pill,
            .edge-pill {
                align-items: center;
                border-radius: 999px;
                display: inline-flex;
                font-size: 0.72rem;
                font-weight: 900;
                justify-content: center;
                line-height: 1;
                min-height: 28px;
                padding: 0.4rem 0.62rem;
                white-space: nowrap;
            }

            .betting-rank-pill {
                background: #eef6ff;
                border: 1px solid #c9dff7;
                color: #1454a8;
            }

            .edge-pill.edge-strong,
            .edge-pill.win {
                background: #dcfce7;
                color: #166534;
            }

            .edge-pill.edge-small,
            .edge-pill.review,
            .edge-pill.push {
                background: #fff7ed;
                color: #c2410c;
            }

            .edge-pill.edge-none,
            .edge-pill.open {
                background: #eef2f7;
                color: #475569;
            }

            .edge-pill.loss {
                background: #fee2e2;
                color: #b42318;
            }

            .betting-pick-body {
                display: grid;
                gap: 0.9rem;
                margin-top: 0.95rem;
            }

            .betting-pick-primary {
                background:
                    linear-gradient(135deg, rgba(239, 246, 255, 0.98), rgba(236, 253, 243, 0.98));
                border: 1px solid #cfe4f7;
                border-radius: var(--league-radius);
                padding: 0.9rem;
            }

            .betting-pick-label,
            .betting-simple-label,
            .betting-side-meta-label {
                color: #66778b;
                font-size: 0.7rem;
                font-weight: 850;
                text-transform: uppercase;
            }

            .betting-pick-team {
                color: #132033;
                font-size: 1.35rem;
                font-weight: 950;
                letter-spacing: 0;
                line-height: 1.12;
                margin-top: 0.18rem;
                overflow-wrap: anywhere;
            }

            .betting-pick-sub,
            .betting-simple-note {
                color: #5d728a;
                font-size: 0.78rem;
                font-weight: 650;
                line-height: 1.35;
                margin-top: 0.2rem;
            }

            .betting-meter {
                background: #dbe7f3;
                border-radius: 999px;
                height: 9px;
                margin-top: 0.72rem;
                overflow: hidden;
            }

            .betting-meter-fill {
                background: linear-gradient(90deg, #1570ef, #16a34a);
                border-radius: inherit;
                height: 100%;
                width: var(--meter-width, 50%);
            }

            .betting-pick-row,
            .betting-side-grid {
                display: grid;
                gap: 0.7rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .betting-simple-stat,
            .betting-side-tile {
                padding: 0.75rem;
            }

            .betting-simple-value,
            .betting-side-odds,
            .betting-side-name,
            .betting-side-meta-value {
                color: #132033;
                font-weight: 950;
                line-height: 1.12;
            }

            .betting-simple-value {
                font-size: 1.02rem;
                margin-top: 0.2rem;
            }

            .betting-side-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .betting-side-name {
                font-size: 0.95rem;
                overflow-wrap: anywhere;
            }

            .betting-side-odds {
                font-size: 1.38rem;
                margin-top: 0.7rem;
            }

            .betting-side-meta-grid {
                display: grid;
                gap: 0.45rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-top: 0.65rem;
            }

            .betting-side-meta {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(213, 224, 234, 0.9);
                border-radius: var(--league-radius);
                padding: 0.5rem;
            }

            .feature-title {
                color: var(--league-ink);
                font-size: 1rem;
                font-weight: 900;
                line-height: 1.2;
            }

            .feature-meta {
                color: var(--league-muted);
                font-size: 0.78rem;
                margin-top: 0.2rem;
            }

            .feature-value {
                color: #0f2544;
                font-size: 1.42rem;
                font-weight: 940;
                line-height: 1.1;
                margin-top: 0.45rem;
                overflow-wrap: anywhere;
            }

            .prediction-card {
                border-left: 6px solid var(--card-accent, var(--league-blue));
            }

            .prediction-teams {
                color: var(--league-muted);
                font-size: 0.84rem;
                font-weight: 750;
            }

            .prediction-winner {
                color: var(--league-ink);
                font-size: 1.55rem;
                font-weight: 920;
                line-height: 1.12;
                margin-top: 0.28rem;
            }

            .prediction-signals {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin-top: 0.65rem;
            }

            .signal-chip {
                background: #eef6ff;
                border: 1px solid #cfe3ff;
                border-radius: 999px;
                color: #344054;
                font-size: 0.74rem;
                font-weight: 820;
                padding: 0.28rem 0.44rem;
            }

            .trust-bar {
                background: #e1ebf5;
                border-radius: 999px;
                height: 0.6rem;
                margin-top: 0.55rem;
                overflow: hidden;
            }

            .trust-fill {
                background: linear-gradient(90deg, #b42318, #b54708, #047857);
                height: 100%;
                width: var(--trust-width, 50%);
            }

            .alert-row {
                align-items: center;
                background: linear-gradient(135deg, #fff7ed, #fff1f2);
                border: 1px solid #fed7aa;
                border-radius: var(--league-radius);
                display: flex;
                gap: 0.75rem;
                justify-content: space-between;
                margin: 0.5rem 0;
                padding: 0.7rem 0.8rem;
            }

            .alert-row strong {
                color: #9a3412;
            }

            .share-card {
                background:
                    linear-gradient(135deg, #132033, #18427a 58%, #079455);
                border-color: rgba(255, 255, 255, 0.18);
                color: #ffffff;
                padding: 1rem;
            }

            .share-card .prediction-teams,
            .share-card .feature-meta {
                color: #cbd5e1;
            }

            .share-card .prediction-winner,
            .share-card .feature-title {
                color: #ffffff;
            }

            @media (max-width: 900px) {
                .league-topbar,
                .league-banner {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .league-tabs {
                    justify-content: flex-start;
                }

                .hub-grid,
                .feature-grid,
                .sport-metrics,
                .betting-pick-row,
                .betting-side-grid {
                    grid-template-columns: 1fr;
                }

                .betting-hero {
                    align-items: stretch;
                    flex-direction: column;
                }

                .betting-hero-stat {
                    min-width: 0;
                    width: 100%;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value: object, digits: int = 1) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "TBD"
    return f"{numeric:.{digits}%}"


def format_small_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "TBD"
    return parsed.strftime("%b %-d, %Y")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_nba_summary() -> dict[str, str]:
    strength = read_csv_if_exists(nba.TEAM_STRENGTH_PATH)
    features = read_csv_if_exists(nba.FEATURES_PATH)
    raw = read_csv_if_exists(nba.RAW_GAMES_PATH)

    top_team = "TBD"
    top_note = "Current power leader"

    if not strength.empty and {"TEAM_NAME", "ELO"}.issubset(strength.columns):
        title_field = build_championship_proxy("NBA")
        if not title_field.empty and {"Team", "Title Probability"}.issubset(title_field.columns):
            top = title_field.iloc[0]
            top_team = str(top["Team"])
            top_note = "Current title field"
        else:
            top = strength.sort_values("ELO", ascending=False).iloc[0]
            top_team = str(top["TEAM_NAME"])

    latest = "TBD"
    if not raw.empty and "GAME_DATE" in raw.columns:
        completed = raw.copy()
        if {"HOME_SCORE", "AWAY_SCORE"}.issubset(completed.columns):
            completed["HOME_SCORE"] = pd.to_numeric(completed["HOME_SCORE"], errors="coerce")
            completed["AWAY_SCORE"] = pd.to_numeric(completed["AWAY_SCORE"], errors="coerce")
            completed = completed[completed["HOME_SCORE"].notna() & completed["AWAY_SCORE"].notna()]
        latest = format_small_date(completed["GAME_DATE"].max()) if not completed.empty else format_small_date(raw["GAME_DATE"].max())

    return {
        "Sport": "NBA",
        "Top Team": top_team,
        "Top Note": top_note,
        "Rows": f"{len(features):,} feature rows",
        "Data Through": latest,
        "Saved Picks": str(len(load_nba_bet_tracker())),
    }


def get_mlb_summary() -> dict[str, str]:
    strength = read_csv_if_exists(mlb.TEAM_STRENGTH_PATH)
    features = read_csv_if_exists(mlb.FEATURES_PATH)
    raw = read_csv_if_exists(mlb.RAW_GAMES_PATH)

    top_team = "TBD"

    if not strength.empty and {"TEAM_NAME", "ELO"}.issubset(strength.columns):
        top = strength.sort_values("ELO", ascending=False).iloc[0]
        top_team = str(top["TEAM_NAME"])

    latest = "TBD"
    if not raw.empty and "GAME_DATE" in raw.columns:
        completed = raw.copy()
        if {"HOME_SCORE", "AWAY_SCORE"}.issubset(completed.columns):
            completed["HOME_SCORE"] = pd.to_numeric(completed["HOME_SCORE"], errors="coerce")
            completed["AWAY_SCORE"] = pd.to_numeric(completed["AWAY_SCORE"], errors="coerce")
            completed = completed[completed["HOME_SCORE"].notna() & completed["AWAY_SCORE"].notna()]
        latest = format_small_date(completed["GAME_DATE"].max()) if not completed.empty else format_small_date(raw["GAME_DATE"].max())

    return {
        "Sport": "MLB",
        "Top Team": top_team,
        "Top Note": "Current power leader",
        "Rows": f"{len(features):,} feature rows",
        "Data Through": latest,
        "Saved Picks": str(len(read_csv_if_exists(mlb.BET_TRACKER_PATH))),
    }


def get_nhl_summary() -> dict[str, str]:
    strength = read_csv_if_exists(nhl.TEAM_STRENGTH_PATH)
    features = read_csv_if_exists(nhl.FEATURES_PATH)
    raw = read_csv_if_exists(nhl.RAW_GAMES_PATH)

    top_team = "TBD"

    if not strength.empty and {"TEAM_NAME", "ELO"}.issubset(strength.columns):
        top = strength.sort_values("ELO", ascending=False).iloc[0]
        top_team = str(top["TEAM_NAME"])

    latest = "TBD"
    if not raw.empty and "GAME_DATE" in raw.columns:
        completed = raw.copy()
        if {"HOME_SCORE", "AWAY_SCORE"}.issubset(completed.columns):
            completed["HOME_SCORE"] = pd.to_numeric(completed["HOME_SCORE"], errors="coerce")
            completed["AWAY_SCORE"] = pd.to_numeric(completed["AWAY_SCORE"], errors="coerce")
            completed = completed[completed["HOME_SCORE"].notna() & completed["AWAY_SCORE"].notna()]
        latest = format_small_date(completed["GAME_DATE"].max()) if not completed.empty else format_small_date(raw["GAME_DATE"].max())

    return {
        "Sport": "NHL",
        "Top Team": top_team,
        "Top Note": "Current power leader",
        "Rows": f"{len(features):,} feature rows",
        "Data Through": latest,
        "Saved Picks": str(len(read_csv_if_exists(nhl.BET_TRACKER_PATH))),
    }


def get_sport_summary(sport: str) -> dict[str, str]:
    if sport == "NBA":
        return get_nba_summary()
    if sport == "MLB":
        return get_mlb_summary()
    if sport == "NHL":
        return get_nhl_summary()
    return {
        "Sport": sport,
        "Top Team": "TBD",
        "Top Note": "Current power leader",
        "Rows": "0 feature rows",
        "Data Through": "TBD",
        "Saved Picks": "0",
    }


def render_topbar(selected_sport: str) -> None:
    pills = []
    for sport in SPORT_OPTIONS:
        active = " active" if sport == selected_sport else ""
        pills.append(f'<span class="league-pill{active}">{html.escape(sport)}</span>')

    st.markdown(
        f"""
        <div class="league-topbar">
            <div class="league-brand">
                <div class="league-mark">SP</div>
                <div>
                    <div class="league-title">Sportsbook Edge</div>
                    <div class="league-subtitle">Odds, picks, bankroll, and game-day betting board</div>
                </div>
            </div>
            <div class="league-tabs">{"".join(pills)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sport_banner(sport: str, page: str, summary: dict[str, str]) -> None:
    accent = SPORT_REGISTRY.get(sport, SPORT_REGISTRY["NBA"]).accent
    badge_label = "Title lean" if summary["Top Note"] == "Current title field" else "Power team"
    st.markdown(
        f"""
        <div class="league-banner" style="--league-accent: {accent};">
            <div>
                <div class="league-banner-title">{html.escape(sport)} / {html.escape(page)}</div>
                <div class="league-banner-meta">
                    {html.escape(summary["Saved Picks"])} saved picks / data through {html.escape(summary["Data Through"])}
                </div>
            </div>
            <div class="league-badge">
                <span class="league-badge-label">{html.escape(badge_label)}</span>
                <span class="league-badge-value">{html.escape(summary["Top Team"])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-tile">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-note">{html.escape(note)}</div>
    </div>
    """


def render_sport_card(summary: dict[str, str], css_class: str) -> None:
    lead_label = "Title Lean" if summary["Top Note"] == "Current title field" else "Power Team"
    metrics = [
        metric_tile(lead_label, summary["Top Team"], summary["Top Note"]),
        metric_tile("Saved Picks", summary["Saved Picks"], "Open and settled bet slips"),
        metric_tile("Data Through", summary["Data Through"], "Latest local sport data"),
        metric_tile("Board Size", summary["Rows"], "Current prediction dataset"),
    ]
    st.markdown(
        f"""
        <div class="sport-card {css_class}">
            <div class="sport-card-head">
                <div>
                    <div class="sport-name">{html.escape(summary["Sport"])}</div>
                    <div class="sport-note">Odds board, picks, trends, and bet tracking</div>
                </div>
                <div class="sport-chip">{html.escape(summary["Saved Picks"])} picks</div>
            </div>
            <div class="sport-metrics">{"".join(metrics)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_selected_sport(sport: str) -> None:
    st.session_state["selected_sport"] = sport


def render_league_hub() -> None:
    summaries = [get_sport_summary(sport) for sport in ["NBA", "MLB", "NHL"]]
    columns = st.columns(len(summaries), gap="medium")
    for column, summary in zip(columns, summaries):
        sport = summary["Sport"]
        with column:
            render_sport_card(summary, sport.lower())
            st.button(f"Open {sport}", width="stretch", on_click=set_selected_sport, args=(sport,))

    comparison = pd.DataFrame(
        [
            {
                "Sport": summary["Sport"],
                "Featured Team": summary["Top Team"],
                "Data Through": summary["Data Through"],
                "Saved Picks": summary["Saved Picks"],
            }
            for summary in summaries
        ]
    )
    st.markdown('<div class="cross-table-wrap">', unsafe_allow_html=True)
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def get_teams_for_sport(sport: str) -> list[str]:
    """Return known team names for one registered sport."""
    try:
        if sport == "NBA":
            return nba.get_teams_from_strength(nba.load_team_strength())
        if sport == "MLB":
            return mlb.get_available_teams()
        if sport == "NHL":
            return nhl.get_available_teams()
    except Exception:
        return []
    return []


def logit(probability: float) -> float:
    probability = min(max(float(probability), 0.001), 0.999)
    return math.log(probability / (1 - probability))


def inv_logit(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def apply_probability_shift(probability: float, shift_points: float) -> float:
    """Apply an intuitive percentage-point shift on a logit scale."""
    return min(max(inv_logit(logit(probability) + (shift_points / 12.0)), 0.01), 0.99)


def prediction_confidence_label(probability: float) -> str:
    if probability >= 0.72:
        return "High"
    if probability >= 0.60:
        return "Medium"
    return "Thin"


def calculate_watchability(home_probability: float, status: object = "") -> float:
    closeness = 1 - min(abs(float(home_probability) - 0.5) * 2, 1)
    live_bonus = 0.12 if str(status).lower() not in {"scheduled", "", "tbd"} else 0.0
    score = (0.78 * closeness) + live_bonus + 0.10
    return round(min(max(score, 0), 1) * 100, 1)


def trust_score_from_details(
    winner_probability: float,
    model_probability: object = None,
    elo_probability: object = None,
) -> float:
    confidence = min(max((float(winner_probability) - 0.5) * 2, 0), 1)
    model_value = safe_float(model_probability)
    elo_value = safe_float(elo_probability)
    agreement = 0.65

    if model_value is not None and elo_value is not None:
        agreement = 1 - min(abs(model_value - elo_value) / 0.22, 1)

    return round(((0.58 * confidence) + (0.42 * agreement)) * 100, 1)


def standard_prediction_row(
    sport: str,
    game_date: object,
    away_team: str,
    home_team: str,
    winner: str,
    home_probability: float,
    away_probability: float,
    model_probability: object = None,
    elo_probability: object = None,
    confidence: object = "",
    note: object = "",
    status: object = "Scheduled",
    away_score: object = "",
    home_score: object = "",
) -> dict:
    winner_probability = home_probability if winner == home_team else away_probability
    favorite = home_team if home_probability >= away_probability else away_team
    underdog_pick = winner != favorite or winner_probability < 0.58
    watchability = calculate_watchability(home_probability, status)
    return {
        "Sport": sport,
        "Game Date": game_date,
        "Matchup": f"{away_team} at {home_team}",
        "Away Team": away_team,
        "Home Team": home_team,
        "Predicted Winner": winner,
        "Winner Probability": float(winner_probability),
        "Home Probability": float(home_probability),
        "Away Probability": float(away_probability),
        "Model Probability": safe_float(model_probability),
        "Elo Probability": safe_float(elo_probability),
        "Confidence": str(confidence or prediction_confidence_label(float(winner_probability))),
        "Prediction Note": str(note or ""),
        "Status": str(status or "Scheduled"),
        "Away Score": away_score,
        "Home Score": home_score,
        "Watchability": watchability,
        "Upset Radar": round((1 - float(winner_probability)) * 100 + (12 if underdog_pick else 0), 1),
        "Trust Score": trust_score_from_details(winner_probability, model_probability, elo_probability),
        "Fair Odds": probability_to_american_odds(winner_probability),
    }


def predict_matchup_for_sport(
    sport: str,
    home_team: str,
    away_team: str,
) -> dict:
    """Return one normalized prediction row for a manual matchup."""
    if sport == "NBA":
        winner, home_probability, away_probability, details = nba.predict_game(home_team, away_team)
        return standard_prediction_row(
            sport="NBA",
            game_date="Manual",
            away_team=away_team,
            home_team=home_team,
            winner=winner,
            home_probability=float(home_probability),
            away_probability=float(away_probability),
            model_probability=details.get("model_probability"),
            elo_probability=details.get("elo_probability"),
            confidence=nba.get_game_confidence_label(max(home_probability, away_probability)),
            note="Manual matchup",
        )

    if sport == "MLB":
        details = mlb.predict_game_details(home_team, away_team)
        return standard_prediction_row(
            sport="MLB",
            game_date="Manual",
            away_team=away_team,
            home_team=home_team,
            winner=str(details["winner"]),
            home_probability=float(details["home_probability"]),
            away_probability=float(details["away_probability"]),
            model_probability=details.get("model_probability"),
            elo_probability=details.get("elo_probability"),
            confidence=prediction_confidence_label(float(details["winner_probability"])),
            note=" / ".join(details.get("explanation", [])),
        )

    details = nhl.predict_game_details(home_team, away_team)
    return standard_prediction_row(
        sport="NHL",
        game_date="Manual",
        away_team=away_team,
        home_team=home_team,
        winner=str(details["winner"]),
        home_probability=float(details["home_probability"]),
        away_probability=float(details["away_probability"]),
        model_probability=details.get("model_probability"),
        elo_probability=details.get("elo_probability"),
        confidence=prediction_confidence_label(float(details["winner_probability"])),
        note=" / ".join(details.get("explanation", [])),
    )


@st.cache_data(ttl=45, show_spinner=False)
def build_nba_prediction_board() -> pd.DataFrame:
    """Build normalized NBA prediction rows for the current or next slate."""
    try:
        games = nba.load_today_games()
        if games.empty:
            games = nba.load_next_upcoming_games()
        predictions = nba.build_today_game_predictions(games)
    except Exception:
        return pd.DataFrame()

    if predictions.empty:
        return pd.DataFrame()

    rows = []
    for _, row in predictions.iterrows():
        home_team = str(row.get("Home Team", ""))
        away_team = str(row.get("Away Team", ""))
        home_probability = safe_float(row.get("Home Win Probability"), 0.5) or 0.5
        away_probability = safe_float(row.get("Away Win Probability"), 0.5) or 0.5
        rows.append(
            standard_prediction_row(
                sport="NBA",
                game_date=row.get("Game Date") or row.get("GAME_DATE") or "",
                away_team=away_team,
                home_team=home_team,
                winner=str(row.get("Predicted Winner", home_team if home_probability >= 0.5 else away_team)),
                home_probability=home_probability,
                away_probability=away_probability,
                model_probability=row.get("Model Probability"),
                elo_probability=row.get("Elo Probability"),
                confidence=row.get("Confidence", ""),
                note=row.get("Prediction Note", ""),
                status=row.get("Status", "Scheduled"),
                away_score=row.get("Away Score", ""),
                home_score=row.get("Home Score", ""),
            )
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=45, show_spinner=False)
def build_mlb_prediction_board() -> pd.DataFrame:
    """Build normalized MLB prediction rows for the current or next slate."""
    try:
        games = mlb.load_betting_slate_games("Today")
        predictions = mlb.build_game_predictions(games)
    except Exception:
        return pd.DataFrame()

    if predictions.empty:
        return pd.DataFrame()

    rows = []
    for _, row in predictions.iterrows():
        home_team = str(row.get("HOME_TEAM", ""))
        away_team = str(row.get("AWAY_TEAM", ""))
        rows.append(
            standard_prediction_row(
                sport="MLB",
                game_date=row.get("GAME_DATE") or row.get("GAME_DATETIME") or "",
                away_team=away_team,
                home_team=home_team,
                winner=str(row.get("PREDICTED_WINNER", "")),
                home_probability=float(row.get("HOME_WIN_PROBABILITY", 0.5)),
                away_probability=float(row.get("AWAY_WIN_PROBABILITY", 0.5)),
                model_probability=row.get("MODEL_PROBABILITY"),
                elo_probability=row.get("ELO_PROBABILITY"),
                confidence=prediction_confidence_label(float(row.get("WINNER_PROBABILITY", 0.5))),
                note=row.get("PREDICTION_EXPLANATION", ""),
                status=row.get("STATUS", row.get("DETAILED_STATE", "Scheduled")),
                away_score=row.get("AWAY_SCORE", ""),
                home_score=row.get("HOME_SCORE", ""),
            )
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=45, show_spinner=False)
def build_nhl_prediction_board() -> pd.DataFrame:
    """Build normalized NHL prediction rows for the current or next slate."""
    try:
        games = nhl.load_betting_slate_games("Today")
        predictions = nhl.build_game_predictions(games)
    except Exception:
        return pd.DataFrame()

    if predictions.empty:
        return pd.DataFrame()

    rows = []
    for _, row in predictions.iterrows():
        home_team = str(row.get("HOME_TEAM", ""))
        away_team = str(row.get("AWAY_TEAM", ""))
        rows.append(
            standard_prediction_row(
                sport="NHL",
                game_date=row.get("GAME_DATE") or row.get("GAME_DATETIME") or "",
                away_team=away_team,
                home_team=home_team,
                winner=str(row.get("PREDICTED_WINNER", "")),
                home_probability=float(row.get("HOME_WIN_PROBABILITY", 0.5)),
                away_probability=float(row.get("AWAY_WIN_PROBABILITY", 0.5)),
                model_probability=row.get("MODEL_PROBABILITY"),
                elo_probability=row.get("ELO_PROBABILITY"),
                confidence=prediction_confidence_label(float(row.get("WINNER_PROBABILITY", 0.5))),
                note=row.get("PREDICTION_EXPLANATION", ""),
                status=row.get("STATUS", "Scheduled"),
                away_score=row.get("AWAY_SCORE", ""),
                home_score=row.get("HOME_SCORE", ""),
            )
        )

    return pd.DataFrame(rows)


def build_prediction_board_for_sport(sport: str) -> pd.DataFrame:
    if sport == "NBA":
        return build_nba_prediction_board()
    if sport == "MLB":
        return build_mlb_prediction_board()
    if sport == "NHL":
        return build_nhl_prediction_board()
    return pd.DataFrame()


def build_all_prediction_rows() -> pd.DataFrame:
    frames = [build_prediction_board_for_sport(sport) for sport in SPORT_REGISTRY]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def format_prediction_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    display = rows.copy()
    for column in ["Winner Probability", "Home Probability", "Away Probability", "Model Probability", "Elo Probability"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "-"
            )
    for column in ["Watchability", "Upset Radar", "Trust Score", "Bet Grade"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:.1f}" if pd.notna(value) else "-"
            )
    return display


def render_prediction_card(row: pd.Series, share_mode: bool = False) -> None:
    accent = SPORT_REGISTRY.get(str(row.get("Sport")), SPORT_REGISTRY["NBA"]).accent
    signals = [
        f"Win {float(row.get('Winner Probability', 0.5)):.1%}",
        f"Grade {float(row.get('Trust Score', 0)):.0f}",
        f"Watch {float(row.get('Watchability', 0)):.0f}",
        f"Fair {row.get('Fair Odds', '-')}",
    ]
    chip_html = "".join(f'<span class="signal-chip">{html.escape(signal)}</span>' for signal in signals)
    css_class = "prediction-card share-card" if share_mode else "prediction-card"
    st.markdown(
        f"""
        <div class="{css_class}" style="--card-accent: {html.escape(accent)};">
            <div class="prediction-teams">{html.escape(str(row.get("Sport", "")))} / {html.escape(str(row.get("Matchup", "")))}</div>
            <div class="prediction-winner">{html.escape(str(row.get("Predicted Winner", "")))}</div>
            <div class="feature-meta">{html.escape(str(row.get("Prediction Note", "")))}</div>
            <div class="prediction-signals">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bet_grade_card(row: pd.Series) -> None:
    grade = float(row.get("Trust Score", 0) or 0)
    warnings = []
    if float(row.get("Winner Probability", 0.5)) < 0.58:
        warnings.append("thin price")
    if float(row.get("Upset Radar", 0.0)) > 55:
        warnings.append("upset-prone")
    if not str(row.get("Prediction Note", "")).strip():
        warnings.append("limited slate detail")
    if not warnings:
        warnings.append("clean betting profile")

    st.markdown(
        f"""
        <div class="trust-card">
            <div class="feature-title">Bet Grade</div>
            <div class="feature-meta">Pricing note: {html.escape(", ".join(warnings))}</div>
            <div class="trust-bar"><div class="trust-fill" style="--trust-width: {grade:.1f}%;"></div></div>
            <div class="prediction-signals">
                <span class="signal-chip">Grade {grade:.1f}</span>
                <span class="signal-chip">Fair {html.escape(str(row.get("Fair Odds", "-")))}</span>
                <span class="signal-chip">Win {float(row.get("Winner Probability", 0.5)):.1%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_nba_regular_season_rows() -> pd.DataFrame:
    games = read_csv_if_exists(nba.RAW_GAMES_PATH)
    if games.empty:
        return pd.DataFrame()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    regular = games[games["SEASON_TYPE"].astype(str).eq("Regular Season")].copy()
    if regular.empty:
        regular = games.copy()

    latest_season = str(regular.sort_values("GAME_DATE")["SEASON"].dropna().iloc[-1])
    return regular[regular["SEASON"].astype(str).eq(latest_season)].copy()


@st.cache_data(show_spinner=False)
def build_nba_standings() -> pd.DataFrame:
    games = current_nba_regular_season_rows()
    if games.empty:
        return pd.DataFrame()

    rows = []
    for team, team_rows in games.groupby("TEAM_NAME"):
        team_rows = team_rows.sort_values("GAME_DATE")
        wins = int(team_rows["WL"].astype(str).eq("W").sum())
        losses = int(team_rows["WL"].astype(str).eq("L").sum())
        games_played = wins + losses
        last_10 = team_rows.tail(10)
        last_10_wins = int(last_10["WL"].astype(str).eq("W").sum())
        rows.append(
            {
                "Conference": NBA_CONFERENCES.get(str(team), "Other"),
                "Team": str(team),
                "W": wins,
                "L": losses,
                "Pct": wins / games_played if games_played else 0.0,
                "Last 10": f"{last_10_wins}-{len(last_10) - last_10_wins}",
                "Point Diff/G": pd.to_numeric(team_rows["PLUS_MINUS"], errors="coerce").mean(),
                "Season": str(team_rows["SEASON"].iloc[-1]),
            }
        )

    standings = pd.DataFrame(rows)
    standings = standings.sort_values(["Conference", "Pct", "Point Diff/G"], ascending=[True, False, False])
    standings["Rank"] = standings.groupby("Conference").cumcount() + 1
    return standings[
        ["Conference", "Rank", "Team", "W", "L", "Pct", "Last 10", "Point Diff/G", "Season"]
    ]


def format_standings_table(rows: pd.DataFrame) -> pd.DataFrame:
    display = rows.copy()
    display["Pct"] = pd.to_numeric(display["Pct"], errors="coerce").map(lambda value: f"{value:.3f}")
    display["Point Diff/G"] = pd.to_numeric(display["Point Diff/G"], errors="coerce").map(lambda value: f"{value:+.1f}")
    return display


def render_nba_standings_page() -> None:
    st.header("Standings")
    standings = build_nba_standings()

    if standings.empty:
        st.warning("No NBA standings data found. Refresh `data/raw_games.csv` first.")
        return

    latest_season = str(standings["Season"].iloc[0])
    st.caption(f"Built from saved NBA regular-season game logs for {latest_season}.")
    overall_tab, east_tab, west_tab = st.tabs(["Overall", "East", "West"])

    with overall_tab:
        overall = standings.sort_values(["Pct", "Point Diff/G"], ascending=False).copy()
        overall["Rank"] = range(1, len(overall) + 1)
        st.dataframe(format_standings_table(overall), hide_index=True, width="stretch")

    with east_tab:
        east = standings[standings["Conference"].eq("East")]
        st.dataframe(format_standings_table(east), hide_index=True, width="stretch")

    with west_tab:
        west = standings[standings["Conference"].eq("West")]
        st.dataframe(format_standings_table(west), hide_index=True, width="stretch")


def american_odds_to_probability(odds: object) -> float | None:
    value = safe_float(odds)
    if value is None or value == 0:
        return None
    if value > 0:
        return 100 / (value + 100)
    return abs(value) / (abs(value) + 100)


def american_odds_profit_per_unit(odds: object) -> float | None:
    value = safe_float(odds)
    if value is None or value == 0:
        return None
    if value > 0:
        return value / 100
    return 100 / abs(value)


def probability_to_american_odds(probability: object) -> str:
    value = safe_float(probability)
    if value is None or value <= 0 or value >= 1:
        return "-"
    if value >= 0.5:
        return f"{round(-100 * value / (1 - value)):+.0f}"
    return f"{round(100 * (1 - value) / value):+.0f}"


def expected_value_per_unit(model_probability: object, odds: object) -> float | None:
    probability = safe_float(model_probability)
    payout = american_odds_profit_per_unit(odds)
    if probability is None or payout is None:
        return None
    return (probability * payout) - (1 - probability)


def kelly_fraction(model_probability: object, odds: object) -> float | None:
    probability = safe_float(model_probability)
    payout = american_odds_profit_per_unit(odds)
    if probability is None or payout is None or payout <= 0:
        return None
    fraction = ((payout * probability) - (1 - probability)) / payout
    return max(0.0, fraction)


def load_nba_bet_tracker() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if not NBA_BET_TRACKER_PATH.exists():
        return pd.DataFrame(columns=NBA_BET_TRACKER_COLUMNS)

    tracker = read_csv_if_exists(NBA_BET_TRACKER_PATH)
    for column in NBA_BET_TRACKER_COLUMNS:
        if column not in tracker.columns:
            tracker[column] = None
    tracker["Market"] = tracker["Market"].fillna("").astype(str)
    tracker.loc[tracker["Market"].str.strip().eq(""), "Market"] = "Moneyline"
    return tracker[NBA_BET_TRACKER_COLUMNS]


def save_nba_bet_tracker(tracker: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tracker.to_csv(NBA_BET_TRACKER_PATH, index=False)


def next_nba_bet_id(tracker: pd.DataFrame) -> int:
    if tracker.empty or "Bet ID" not in tracker.columns:
        return 1
    ids = pd.to_numeric(tracker["Bet ID"], errors="coerce").dropna()
    return int(ids.max()) + 1 if not ids.empty else 1


def calculate_bet_profit(result: object, odds: object, stake: object) -> float | None:
    result_text = str(result or "").strip().lower()
    stake_value = safe_float(stake)
    payout = american_odds_profit_per_unit(odds)

    if stake_value is None or payout is None:
        return None
    if result_text == "win":
        return stake_value * payout
    if result_text == "loss":
        return -stake_value
    if result_text == "push":
        return 0.0
    return None


def format_american_odds_value(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None or numeric == 0:
        return "-"
    return f"{int(round(numeric)):+d}"


def format_money_value(value: object, signed: bool = False) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    prefix = "+" if signed and numeric > 0 else ""
    return f"{prefix}${numeric:,.2f}"


def format_probability_value(value: object, digits: int = 1) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}%}"


def format_edge_points(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric * 100:+.1f} pts"


def format_profit_on_stake(odds: object, stake: object) -> str:
    payout = american_odds_profit_per_unit(odds)
    stake_value = safe_float(stake)
    if payout is None or stake_value is None:
        return "-"
    return format_money_value(payout * stake_value)


def simple_nba_betting_signal(edge: object) -> tuple[str, str, str]:
    numeric = safe_float(edge)
    if numeric is None:
        return "Add odds", "Enter the sportsbook price to compare it to our number.", "edge-none"
    if numeric >= 0.05:
        return "Good value", "Our win chance is meaningfully better than the book price.", "edge-strong"
    if numeric >= 0.02:
        return "Small value", "There is a positive gap, but it is not huge.", "edge-small"
    if numeric <= -0.02:
        return "Price is high", "The sportsbook price is worse than our number.", "edge-none"
    return "Fair price", "Our number and the book price are close.", "edge-none"


def recalculate_nba_tracker(tracker: pd.DataFrame) -> pd.DataFrame:
    if tracker.empty:
        return tracker

    tracker = tracker.copy()
    tracker["Profit"] = tracker.apply(
        lambda row: calculate_bet_profit(row.get("Result"), row.get("Odds"), row.get("Stake")),
        axis=1,
    )
    return tracker


def parse_nba_tracker_game_date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def parse_nba_tracker_matchup(matchup: object) -> tuple[str, str] | None:
    parts = str(matchup or "").split(" at ", 1)
    if len(parts) != 2:
        return None
    away_team = parts[0].strip()
    home_team = parts[1].strip()
    if not away_team or not home_team:
        return None
    return away_team, home_team


def normalize_betting_team_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def build_nba_final_score_lookup() -> dict[tuple[str, str], dict[str, object]]:
    games = read_csv_if_exists(nba.RAW_GAMES_PATH)
    if games.empty or not {"GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"}.issubset(games.columns):
        return {}

    home_rows = games[games["MATCHUP"].astype(str).str.contains("vs.", na=False)].copy()
    away_rows = games[games["MATCHUP"].astype(str).str.contains("@", na=False)].copy()
    if home_rows.empty or away_rows.empty:
        return {}

    home_rows = home_rows[["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS"]].rename(
        columns={"TEAM_NAME": "Home Team", "PTS": "Home Score"}
    )
    away_rows = away_rows[["GAME_ID", "TEAM_NAME", "PTS"]].rename(
        columns={"TEAM_NAME": "Away Team", "PTS": "Away Score"}
    )
    finals = home_rows.merge(away_rows, on="GAME_ID", how="inner")
    lookup: dict[tuple[str, str], dict[str, object]] = {}

    for _, game in finals.iterrows():
        game_date = parse_nba_tracker_game_date(game.get("GAME_DATE"))
        if game_date is None:
            continue

        away_team = str(game.get("Away Team", ""))
        home_team = str(game.get("Home Team", ""))
        matchup_key = normalize_betting_team_key(f"{away_team} at {home_team}")
        lookup[(game_date.date().isoformat(), matchup_key)] = game.to_dict()

    return lookup


def settle_nba_moneyline_pick(row: pd.Series, game: dict[str, object]) -> tuple[str | None, str]:
    home_score = safe_int(game.get("Home Score"), default=-1)
    away_score = safe_int(game.get("Away Score"), default=-1)
    if home_score < 0 or away_score < 0:
        return None, ""

    home_team = str(game.get("Home Team", ""))
    away_team = str(game.get("Away Team", ""))
    final_note = f"Final: {away_team} {away_score}, {home_team} {home_score}"

    if home_score == away_score:
        return "Push", final_note

    winner = home_team if home_score > away_score else away_team
    selection_key = normalize_betting_team_key(row.get("Selection"))
    return ("Win" if selection_key == normalize_betting_team_key(winner) else "Loss"), final_note


def append_tracker_note(existing: object, note: str) -> str:
    current = str(existing or "").strip()
    if not note or note in current:
        return current
    return f"{current}; {note}" if current else note


def auto_settle_nba_bet_tracker(tracker: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    tracker = tracker.copy()
    changed = False
    if tracker.empty:
        return recalculate_nba_tracker(tracker), changed

    score_lookup = build_nba_final_score_lookup()
    if not score_lookup:
        return recalculate_nba_tracker(tracker), changed

    for index, row in tracker.iterrows():
        result = str(row.get("Result", "") or "Open").strip().lower()
        if result in {"win", "loss", "push"}:
            continue
        if str(row.get("Market", "Moneyline") or "Moneyline").strip().lower() != "moneyline":
            continue

        game_date = parse_nba_tracker_game_date(row.get("Game Date"))
        matchup = parse_nba_tracker_matchup(row.get("Matchup"))
        if game_date is None or matchup is None:
            continue

        matchup_key = normalize_betting_team_key(f"{matchup[0]} at {matchup[1]}")
        game = score_lookup.get((game_date.date().isoformat(), matchup_key))
        if game is None:
            continue

        settled_result, final_note = settle_nba_moneyline_pick(row, game)
        if not settled_result:
            continue

        tracker.at[index, "Result"] = settled_result
        tracker.at[index, "Notes"] = append_tracker_note(row.get("Notes"), final_note)
        changed = True

    return recalculate_nba_tracker(tracker), changed


def load_nba_bet_tracker_with_auto_settlement() -> pd.DataFrame:
    tracker, changed = auto_settle_nba_bet_tracker(load_nba_bet_tracker())
    if changed:
        save_nba_bet_tracker(tracker)
    return tracker


def append_nba_bet_tracker_row(row: dict[str, object]) -> None:
    tracker = load_nba_bet_tracker()
    row = {"Bet ID": next_nba_bet_id(tracker), **row}
    tracker = pd.concat([tracker, pd.DataFrame([row])], ignore_index=True)
    save_nba_bet_tracker(recalculate_nba_tracker(tracker))


def delete_nba_bet_tracker_row(bet_id: object) -> None:
    tracker = load_nba_bet_tracker()
    if tracker.empty:
        return

    bet_id_text = str(bet_id or "").strip()
    keep = tracker["Bet ID"].astype(str).str.strip().ne(bet_id_text)
    save_nba_bet_tracker(tracker[keep].reset_index(drop=True))


def build_nba_active_series_games() -> pd.DataFrame:
    try:
        states = nba.build_current_playoff_series_states()
    except Exception:
        return pd.DataFrame()

    rows = []
    home_court_schedule = getattr(
        nba,
        "HOME_COURT_SCHEDULE",
        ["higher", "higher", "lower", "lower", "higher", "lower", "higher"],
    )

    for state in states:
        if state.get("completed"):
            continue

        teams = [str(team) for team in state.get("teams", []) if str(team).strip()]
        if len(teams) < 2:
            continue

        game_number = safe_int(state.get("next_game_number"), default=1)
        if game_number < 1 or game_number > 7:
            continue

        home_court_team = str(state.get("home_court_team") or teams[0])
        other_team = str(state.get("other_team") or next((team for team in teams if team != home_court_team), teams[-1]))
        home_side = home_court_schedule[min(game_number - 1, len(home_court_schedule) - 1)]
        home_team = home_court_team if home_side == "higher" else other_team
        away_team = other_team if home_team == home_court_team else home_court_team
        game_label = f"Game {game_number}"

        rows.append(
            {
                "Game ID": f"nba_active_{normalize_betting_team_key(state.get('series_key'))}_{game_number}",
                "Game Date": game_label,
                "Game DateTime": pd.NaT,
                "Game Time": "TBD",
                "Status": "Scheduled",
                "Game Status Code": 1,
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": "",
                "Away Score": "",
                "Series Game Number": game_number,
                "Game Label": game_label,
                "Game SubLabel": str(state.get("series_key", "")),
                "If Necessary": False,
                "Source": "Active playoff series",
            }
        )

    return pd.DataFrame(rows)


def load_nba_betting_slate_games() -> tuple[pd.DataFrame, str]:
    try:
        games = nba.load_today_games()
    except Exception:
        games = pd.DataFrame()
    if not games.empty:
        return games, "Today"

    try:
        games = nba.load_next_upcoming_games()
    except Exception:
        games = pd.DataFrame()
    if not games.empty:
        return games, "Next Upcoming"

    games = build_nba_active_series_games()
    if not games.empty:
        return games, "Active Series"

    return pd.DataFrame(), ""


def build_nba_betting_board() -> pd.DataFrame:
    games, slate_label = load_nba_betting_slate_games()
    if games.empty:
        return pd.DataFrame()

    try:
        predictions = nba.build_today_game_predictions(games)
    except Exception:
        return pd.DataFrame()

    if predictions.empty:
        return pd.DataFrame()

    rows = []
    for _, row in predictions.iterrows():
        home_team = str(row.get("Home Team", ""))
        away_team = str(row.get("Away Team", ""))
        home_probability = safe_float(row.get("Home Win Probability"), 0.5) or 0.5
        away_probability = safe_float(row.get("Away Win Probability"), 0.5) or 0.5

        rows.append(
            {
                "Game Date": row.get("Game Date") or row.get("GAME_DATE") or "",
                "Game Time": row.get("Game Time") or row.get("Status") or "",
                "Game Key": row.get("Game ID") or f"{away_team}_{home_team}_{row.get('Game Date', '')}",
                "Slate": slate_label,
                "Matchup": f"{away_team} at {home_team}",
                "Home Team": home_team,
                "Away Team": away_team,
                "Side": "Home",
                "Selection": home_team,
                "Model Probability": home_probability,
                "Fair Odds": probability_to_american_odds(home_probability),
                "Confidence": row.get("Confidence", ""),
                "Prediction Note": row.get("Prediction Note", ""),
                "Status": row.get("Status", "Scheduled"),
            }
        )
        rows.append(
            {
                "Game Date": row.get("Game Date") or row.get("GAME_DATE") or "",
                "Game Time": row.get("Game Time") or row.get("Status") or "",
                "Game Key": row.get("Game ID") or f"{away_team}_{home_team}_{row.get('Game Date', '')}",
                "Slate": slate_label,
                "Matchup": f"{away_team} at {home_team}",
                "Home Team": home_team,
                "Away Team": away_team,
                "Side": "Away",
                "Selection": away_team,
                "Model Probability": away_probability,
                "Fair Odds": probability_to_american_odds(away_probability),
                "Confidence": row.get("Confidence", ""),
                "Prediction Note": row.get("Prediction Note", ""),
                "Status": row.get("Status", "Scheduled"),
            }
        )

    return pd.DataFrame(rows)


def is_nba_tracker_row_active(row: pd.Series) -> bool:
    result = str(row.get("Result", "") or "Open").strip().lower()
    if result in {"win", "loss", "push"}:
        return False

    game_date = parse_nba_tracker_game_date(row.get("Game Date"))
    if game_date is None:
        return True

    today = pd.Timestamp.now().normalize()
    return game_date >= today


def is_nba_tracker_row_settled(row: pd.Series) -> bool:
    result = str(row.get("Result", "") or "Open").strip().lower()
    if result in {"win", "loss", "push"}:
        return True

    game_date = parse_nba_tracker_game_date(row.get("Game Date"))
    if game_date is None:
        return False

    today = pd.Timestamp.now().normalize()
    return game_date < today


def sort_nba_tracker_cards(tracker: pd.DataFrame) -> pd.DataFrame:
    if tracker.empty:
        return tracker
    display = tracker.copy()
    display["_game_date_sort"] = pd.to_datetime(display["Game Date"], errors="coerce")
    display["_saved_sort"] = pd.to_datetime(display["Date Logged"], errors="coerce")
    display["_bet_id_sort"] = pd.to_numeric(display["Bet ID"], errors="coerce")
    return display.sort_values(
        ["_game_date_sort", "_saved_sort", "_bet_id_sort"],
        ascending=[False, False, False],
        na_position="last",
    )


def render_nba_betting_summary(tracker: pd.DataFrame) -> None:
    if tracker.empty:
        metrics = [
            metric_tile("Open Bets", "0", "No saved NBA picks."),
            metric_tile("Profit", "$0.00", "Set results after games finish."),
            metric_tile("ROI", "-", "Settled bets only."),
        ]
    else:
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
        metrics = [
            metric_tile("Open Bets", str(open_count), f"{len(settled)} settled rows."),
            metric_tile("Profit", format_money_value(profit, signed=True), f"Stake {format_money_value(stake)}."),
            metric_tile(
                "ROI / Win Rate",
                f"{roi:.1%}" if roi is not None else "-",
                f"Win rate {win_rate:.1%}" if win_rate is not None else "No decisions.",
            ),
        ]

    st.html(f'<div class="metric-grid">{"".join(metrics)}</div>')


def render_nba_tracker_pick_card(row: pd.Series, key_prefix: str, settled_view: bool = False) -> None:
    result = str(row.get("Result", "") or "Open").strip()
    result_key = result.lower()
    if result_key not in {"win", "loss", "push"}:
        result = "Needs result" if settled_view else "Open"
        result_key = "review" if settled_view else "open"

    accent = {
        "win": "#16a34a",
        "loss": "#dc2626",
        "push": "#d97706",
        "review": "#d97706",
        "open": "#2563eb",
    }.get(result_key, "#2563eb")
    notes = str(row.get("Notes", "") or "").strip()
    note_html = f'<div class="betting-simple-note">{html.escape(notes)}</div>' if notes else ""
    bet_id = row.get("Bet ID")
    detail_html = "".join(
        f"""
        <div class="betting-simple-stat">
            <div class="betting-simple-label">{html.escape(label)}</div>
            <div class="betting-simple-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in [
            ("Odds", format_american_odds_value(row.get("Odds"))),
            ("Stake", format_money_value(row.get("Stake"))),
            ("Profit", format_money_value(row.get("Profit"), signed=True)),
            ("Edge", format_edge_points(row.get("Edge"))),
        ]
    )

    card_col, action_col = st.columns([0.84, 0.16], gap="small")
    with card_col:
        st.html(
            f"""
            <div class="betting-card-shell" style="--card-accent: {accent};">
                <div class="betting-card-head">
                    <div>
                        <div class="betting-game-title">{html.escape(str(row.get("Selection", "Pick")))}</div>
                        <div class="betting-game-meta">
                            {html.escape(str(row.get("Market", "Moneyline")))}
                            / {html.escape(str(row.get("Matchup", "Game")))}
                        </div>
                        <div class="betting-game-meta">
                            Game {html.escape(str(row.get("Game Date", "Date TBD")))}
                            / Saved {html.escape(str(row.get("Date Logged", "")))}
                        </div>
                    </div>
                    <span class="edge-pill {html.escape(result_key)}">{html.escape(result)}</span>
                </div>
                <div class="betting-pick-row">{detail_html}</div>
                {note_html}
            </div>
            """
        )

    with action_col:
        st.write("")
        st.write("")
        if st.button("Delete", key=f"{key_prefix}_delete_{bet_id}", width="stretch"):
            delete_nba_bet_tracker_row(bet_id)
            st.rerun()


def render_nba_tracker_card_list(
    tracker: pd.DataFrame,
    empty_message: str,
    key_prefix: str,
    settled_view: bool = False,
    limit: int = 24,
) -> None:
    if tracker.empty:
        st.info(empty_message)
        return

    for _, row in sort_nba_tracker_cards(tracker).head(limit).iterrows():
        render_nba_tracker_pick_card(row, key_prefix=key_prefix, settled_view=settled_view)


def render_nba_betting_hero(board: pd.DataFrame) -> None:
    slate = str(board["Slate"].dropna().iloc[0]) if not board.empty and "Slate" in board.columns else "Manual"
    top = board.sort_values("Model Probability", ascending=False).iloc[0] if not board.empty else None
    top_value = (
        f"{top['Selection']} {float(top['Model Probability']):.0%}"
        if top is not None
        else "No board"
    )
    top_note = str(top.get("Matchup", "Choose a manual matchup.")) if top is not None else "Choose a manual matchup."
    st.html(
        f"""
        <div class="betting-hero">
            <div>
                <div class="betting-hero-title">NBA Betting</div>
                <div class="betting-hero-note">
                    Pick a team to win, enter the sportsbook price, and save the pick to the tracker.
                </div>
            </div>
            <div class="betting-hero-stat">
                <div class="betting-hero-stat-label">{html.escape(slate)}</div>
                <div class="betting-hero-stat-value">{html.escape(top_value)}</div>
                <div class="betting-hero-note">{html.escape(top_note)}</div>
            </div>
        </div>
        """
    )


def render_nba_moneyline_card(game_rows: pd.DataFrame, rank: int) -> None:
    if game_rows.empty:
        return

    game_rows = game_rows.sort_values("Model Probability", ascending=False).reset_index(drop=True)
    best = game_rows.iloc[0]
    matchup = str(best.get("Matchup", "NBA game"))
    game_key = normalize_betting_team_key(best.get("Game Key") or matchup)
    options = game_rows["Selection"].astype(str).tolist()
    selection = st.selectbox(
        f"Pick for {matchup}",
        options,
        key=f"nba_bet_selection_{game_key}_{rank}",
        label_visibility="collapsed",
    )
    selected = game_rows[game_rows["Selection"].astype(str).eq(selection)].iloc[0]
    model_probability = float(selected.get("Model Probability", 0.5))

    odds_col, stake_col = st.columns(2)
    with odds_col:
        odds = st.number_input(
            "Sportsbook odds",
            min_value=-10000,
            max_value=10000,
            value=-110,
            step=5,
            key=f"nba_bet_odds_{game_key}_{rank}",
        )
    with stake_col:
        stake = st.number_input(
            "Stake",
            min_value=0.0,
            max_value=100000.0,
            value=10.0,
            step=1.0,
            key=f"nba_bet_stake_{game_key}_{rank}",
        )

    market_probability = american_odds_to_probability(odds)
    edge = model_probability - market_probability if market_probability is not None else None
    ev = expected_value_per_unit(model_probability, odds)
    kelly = kelly_fraction(model_probability, odds)
    signal, signal_note, signal_class = simple_nba_betting_signal(edge)
    meter_width = int(round(model_probability * 100))
    sides_html = ""

    for _, row in game_rows.iterrows():
        is_selected = str(row["Selection"]) == selection
        selected_class = " betting-side-selected" if is_selected else ""
        sides_html += f"""
            <div class="betting-side-tile{selected_class}">
                <div class="betting-side-top">
                    <div>
                        <div class="betting-side-name">{html.escape(str(row["Selection"]))}</div>
                        <div class="betting-game-meta">{html.escape(str(row.get("Side", "")))}</div>
                    </div>
                    <span class="edge-pill {signal_class if is_selected else "edge-none"}">
                        {html.escape(signal if is_selected else "Compare")}
                    </span>
                </div>
                <div class="betting-side-odds">{html.escape(str(row.get("Fair Odds", "-")))}</div>
                <div class="betting-game-meta">Fair price</div>
                <div class="betting-side-meta-grid">
                    <div class="betting-side-meta">
                        <div class="betting-side-meta-label">Win</div>
                        <div class="betting-side-meta-value">{float(row["Model Probability"]):.0%}</div>
                    </div>
                    <div class="betting-side-meta">
                        <div class="betting-side-meta-label">Conf</div>
                        <div class="betting-side-meta-value">{html.escape(str(row.get("Confidence", "-")))}</div>
                    </div>
                </div>
            </div>
        """

    st.html(
        f"""
        <div class="betting-card-shell">
            <div class="betting-card-head">
                <div>
                    <div class="betting-game-title">{html.escape(matchup)}</div>
                    <div class="betting-game-meta">
                        {html.escape(str(best.get("Status", "Scheduled")))}
                        / {html.escape(str(best.get("Game Date", "")))}
                        {html.escape(str(" / " + str(best.get("Game Time", "")) if str(best.get("Game Time", "")).strip() else ""))}
                    </div>
                </div>
                <div class="betting-rank-pill">#{rank} {html.escape(signal)}</div>
            </div>
            <div class="betting-pick-body">
                <div class="betting-pick-primary">
                    <div class="betting-pick-label">Selected pick</div>
                    <div class="betting-pick-team">{html.escape(selection)}</div>
                    <div class="betting-pick-sub">{html.escape(signal_note)}</div>
                    <div class="betting-meter">
                        <div class="betting-meter-fill" style="--meter-width: {meter_width}%;"></div>
                    </div>
                </div>
                <div class="betting-pick-row">
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Book price</div>
                        <div class="betting-simple-value">{html.escape(format_american_odds_value(odds))}</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Win chance</div>
                        <div class="betting-simple-value">{model_probability:.0%}</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Profit on stake</div>
                        <div class="betting-simple-value">{html.escape(format_profit_on_stake(odds, stake))}</div>
                    </div>
                    <div class="betting-simple-stat">
                        <div class="betting-simple-label">Edge</div>
                        <div class="betting-simple-value">{html.escape(format_edge_points(edge))}</div>
                    </div>
                </div>
                <div class="betting-side-grid">{sides_html}</div>
            </div>
        </div>
        """
    )

    notes = st.text_input("Notes", value="", key=f"nba_bet_notes_{game_key}_{rank}")
    if st.button("Save pick", type="primary", width="stretch", key=f"nba_save_pick_{game_key}_{rank}"):
        append_nba_bet_tracker_row(
            {
                "Date Logged": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Game Date": selected.get("Game Date", ""),
                "Matchup": selected["Matchup"],
                "Market": "Moneyline",
                "Selection": selected["Selection"],
                "Odds": odds,
                "Stake": stake,
                "Model Probability": model_probability,
                "Market Probability": market_probability,
                "Edge": edge,
                "EV/Unit": ev,
                "Kelly": kelly,
                "Result": "Open",
                "Profit": None,
                "Notes": notes,
            }
        )
        st.success("NBA pick saved.")
        st.rerun()


def render_nba_place_bets() -> None:
    board = build_nba_betting_board()
    render_nba_betting_hero(board)

    if board.empty:
        st.info("No NBA slate was found. Use the Games page for manual matchup research until the schedule feed returns games.")
        return

    grouped = list(board.groupby("Game Key", sort=False))
    columns = st.columns(1 if len(grouped) == 1 else 2, gap="medium")
    for index, (_, game_rows) in enumerate(grouped, start=1):
        with columns[(index - 1) % len(columns)]:
            render_nba_moneyline_card(game_rows, rank=index)


def render_nba_saved_betting_picks(limit: int = 24) -> None:
    tracker = load_nba_bet_tracker_with_auto_settlement()
    active = tracker[tracker.apply(is_nba_tracker_row_active, axis=1)].copy() if not tracker.empty else tracker
    render_nba_tracker_card_list(
        active,
        "No active saved NBA picks. Add one from Place Bets.",
        "nba_active_pick",
        settled_view=False,
        limit=limit,
    )


def render_nba_settled_betting_picks(limit: int = 48) -> None:
    tracker = load_nba_bet_tracker_with_auto_settlement()
    settled = tracker[tracker.apply(is_nba_tracker_row_settled, axis=1)].copy() if not tracker.empty else tracker
    render_nba_betting_summary(tracker)
    render_nba_tracker_card_list(
        settled,
        "No settled NBA bets yet. Finished games will appear here after results are set or matched to final scores.",
        "nba_settled_pick",
        settled_view=True,
        limit=limit,
    )

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
                "Market": st.column_config.SelectboxColumn(
                    "Market",
                    options=["Moneyline"],
                ),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
            key="nba_bet_tracker_editor",
        )
        if st.button("Save tracker changes", width="stretch", key="nba_save_bet_tracker"):
            save_nba_bet_tracker(recalculate_nba_tracker(edited))
            st.success("NBA tracker saved.")
            st.rerun()


def render_nba_betting_page() -> None:
    st.caption("A simple NBA moneyline screen for choosing a team, checking the sportsbook price, and tracking results.")
    place_tab, saved_tab, settled_tab = st.tabs(["Place Bets", "Saved Picks", "Settled Bets"])

    with place_tab:
        render_nba_place_bets()

    with saved_tab:
        render_nba_saved_betting_picks()

    with settled_tab:
        render_nba_settled_betting_picks()


def render_legacy_nba_tracker_table() -> None:
    tracker = load_nba_bet_tracker_with_auto_settlement()
    if tracker.empty:
        st.caption("Saved NBA picks will appear here.")
        return

    simple = tracker.copy()
    for column in ["Model Probability", "Market Probability", "Edge", "Kelly"]:
        simple[column] = pd.to_numeric(simple[column], errors="coerce").map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "-"
        )
    simple["EV/Unit"] = pd.to_numeric(simple["EV/Unit"], errors="coerce").map(
        lambda value: f"{value:+.2f}" if pd.notna(value) else "-"
    )
    simple["Profit"] = pd.to_numeric(simple["Profit"], errors="coerce").map(
        lambda value: f"${value:,.2f}" if pd.notna(value) else "-"
    )
    simple = simple.rename(columns={"Model Probability": "Win Chance", "Market Probability": "Book Probability"})
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
                )
            },
            key="nba_bet_tracker_editor",
        )
        if st.button("Save Tracker Changes", width="stretch"):
            save_nba_bet_tracker(recalculate_nba_tracker(edited))
            st.success("NBA tracker saved.")
            st.rerun()


def render_feature_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-title">{html.escape(title)}</div>
            <div class="feature-value">{html.escape(value)}</div>
            <div class="feature-meta">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_prediction_row(rows: pd.DataFrame, key: str) -> pd.Series | None:
    if rows.empty:
        return None
    labels = [f"{row['Sport']} / {row['Matchup']}" for _, row in rows.iterrows()]
    label = st.selectbox("Game", labels, key=key)
    return rows.iloc[labels.index(label)]


def render_daily_brief_page() -> None:
    st.header("Daily Sports Brief")
    rows = build_all_prediction_rows()

    if rows.empty:
        st.info("No current slate predictions are available right now.")
        return

    high_conf = rows.sort_values("Winner Probability", ascending=False).iloc[0]
    closest = rows.assign(_gap=(rows["Home Probability"] - 0.5).abs()).sort_values("_gap").iloc[0]
    upset = rows.sort_values("Upset Radar", ascending=False).iloc[0]
    watch = rows.sort_values("Watchability", ascending=False).iloc[0]

    cols = st.columns(4)
    with cols[0]:
        render_feature_card("Highest Confidence", str(high_conf["Predicted Winner"]), f"{high_conf['Matchup']} / {float(high_conf['Winner Probability']):.1%}")
    with cols[1]:
        render_feature_card("Closest Game", str(closest["Matchup"]), f"{float(closest['Home Probability']):.1%} home win")
    with cols[2]:
        render_feature_card("Upset Radar", str(upset["Predicted Winner"]), f"{upset['Sport']} / score {float(upset['Upset Radar']):.1f}")
    with cols[3]:
        render_feature_card("Best Watch", str(watch["Matchup"]), f"Watchability {float(watch['Watchability']):.1f}")

    render_pick_movement_feed()
    render_live_upset_alerts(rows)

    st.subheader("Watchability Board")
    columns = [
        "Sport",
        "Matchup",
        "Predicted Winner",
        "Winner Probability",
        "Watchability",
        "Upset Radar",
        "Bet Grade",
        "Prediction Note",
    ]
    display_rows = rows.rename(columns={"Trust Score": "Bet Grade"})
    st.dataframe(
        format_prediction_table(display_rows.sort_values("Watchability", ascending=False)[columns]),
        hide_index=True,
        width="stretch",
    )


def render_game_center_page(default_sport: str | None = None) -> None:
    st.header("Unified Game Center")
    sport_options = list(SPORT_REGISTRY.keys()) if default_sport is None else [default_sport]
    sport = st.radio("Sport", sport_options, horizontal=True, key=f"game_center_sport_{default_sport or 'hub'}")
    current_rows = build_prediction_board_for_sport(sport)
    mode = st.segmented_control(
        "Mode",
        ["Current Slate", "Manual Matchup"],
        default="Current Slate" if not current_rows.empty else "Manual Matchup",
        key=f"game_center_mode_{sport}_{default_sport or 'hub'}",
    )

    if mode == "Current Slate" and not current_rows.empty:
        row = selected_prediction_row(current_rows, key=f"game_center_game_{sport}_{default_sport or 'hub'}")
    else:
        teams = get_teams_for_sport(sport)
        config = SPORT_REGISTRY[sport]
        if len(teams) < 2:
            st.warning(f"No {sport} teams are available from the saved strength file.")
            return
        home_col, away_col = st.columns(2)
        with home_col:
            home_team = st.selectbox("Home", teams, index=teams.index(config.default_home) if config.default_home in teams else 0, key=f"gc_home_{sport}_{default_sport or 'hub'}")
        with away_col:
            away_team = st.selectbox("Away", teams, index=teams.index(config.default_away) if config.default_away in teams else min(1, len(teams) - 1), key=f"gc_away_{sport}_{default_sport or 'hub'}")
        if home_team == away_team:
            st.warning("Choose two different teams.")
            return
        row = pd.Series(predict_matchup_for_sport(sport, home_team, away_team))

    if row is None:
        st.info("No game selected.")
        return

    left, right = st.columns([1.0, 0.8], gap="medium")
    with left:
        render_prediction_card(row)
    with right:
        render_bet_grade_card(row)

    tabs = st.tabs(["What-If", "Odds", "Pick Card"])
    with tabs[0]:
        render_scenario_controls(row, key_prefix=f"gc_scenario_{sport}_{default_sport or 'hub'}")
    with tabs[1]:
        render_odds_comparison_for_row(row, key_prefix=f"gc_odds_{sport}_{default_sport or 'hub'}")
    with tabs[2]:
        render_share_card_for_row(row)


def render_upset_radar_page() -> None:
    st.header("Upset Radar")
    rows = build_all_prediction_rows()
    if rows.empty:
        st.info("No slate predictions are available for upset ranking.")
        return

    radar = rows.sort_values(["Upset Radar", "Watchability"], ascending=False)
    top = radar.head(8)
    for _, row in top.iterrows():
        st.markdown(
            f"""
            <div class="alert-row">
                <div><strong>{html.escape(str(row["Predicted Winner"]))}</strong> / {html.escape(str(row["Matchup"]))}</div>
                <div>{float(row["Upset Radar"]):.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    display_radar = radar.rename(columns={"Trust Score": "Bet Grade"})
    st.dataframe(
        format_prediction_table(
            display_radar[["Sport", "Matchup", "Predicted Winner", "Winner Probability", "Upset Radar", "Bet Grade", "Prediction Note"]]
        ),
        hide_index=True,
        width="stretch",
    )


def build_championship_proxy(sport: str) -> pd.DataFrame:
    strength = read_csv_if_exists(SPORT_REGISTRY[sport].strength_path)
    if strength.empty or not {"TEAM_NAME", "ELO"}.issubset(strength.columns):
        return pd.DataFrame()

    rows = strength[["TEAM_NAME", "ELO"]].copy()
    rows["ELO"] = pd.to_numeric(rows["ELO"], errors="coerce")
    rows = rows.dropna(subset=["ELO"])
    if rows.empty:
        return pd.DataFrame()

    if sport == "NBA":
        active_teams: set[str] = set()
        try:
            series_states = nba.build_current_playoff_series_states()
            for state in series_states:
                if state.get("completed"):
                    continue
                active_teams.update(
                    str(team)
                    for team in state.get("teams", [])
                    if str(team).strip()
                )
        except Exception:
            series_states = []
            active_teams = set()

        if active_teams:
            rows = rows[rows["TEAM_NAME"].isin(active_teams)].copy()
            rows = rows.dropna(subset=["ELO"])
            if rows.empty:
                return pd.DataFrame()
        else:
            completed_states = [
                state
                for state in series_states
                if state.get("completed") and state.get("leader")
            ]
            if not completed_states:
                return pd.DataFrame()

            recent_completed = sorted(
                completed_states,
                key=lambda state: (
                    pd.to_datetime(state["game_rows"][-1]["Date"], errors="coerce")
                    if state.get("game_rows")
                    else pd.Timestamp.min,
                    int(state.get("games_played", 0)),
                ),
                reverse=True,
            )[:2]
            fallback_teams = {
                str(state["leader"])
                for state in recent_completed
                if str(state["leader"]).strip()
            }
            if not fallback_teams:
                return pd.DataFrame()

            rows = rows[rows["TEAM_NAME"].isin(fallback_teams)].copy()
            rows = rows.dropna(subset=["ELO"])
            if rows.empty:
                return pd.DataFrame()

    if sport == "NHL":
        active_teams: set[str] = set()
        try:
            series_states = nhl.build_playoff_series_states()
            for state in series_states:
                if state.get("completed"):
                    continue
                active_teams.update(
                    str(team)
                    for team in state.get("wins", {}).keys()
                    if str(team).strip()
                )
        except Exception:
            series_states = []
            active_teams = set()

        if active_teams:
            rows = rows[rows["TEAM_NAME"].isin(active_teams)].copy()
            rows = rows.dropna(subset=["ELO"])
            if rows.empty:
                return pd.DataFrame()
        else:
            completed_states = [
                state
                for state in series_states
                if state.get("completed") and state.get("leader")
            ]
            if completed_states:
                recent_completed = sorted(
                    completed_states,
                    key=lambda state: (
                        pd.to_datetime(state["game_rows"][-1]["Date"], errors="coerce")
                        if state.get("game_rows")
                        else pd.Timestamp.min,
                        int(state.get("games_played", 0)),
                    ),
                    reverse=True,
                )[:2]
                fallback_teams = {
                    str(state["leader"])
                    for state in recent_completed
                    if str(state["leader"]).strip()
                }
                if fallback_teams:
                    rows = rows[rows["TEAM_NAME"].isin(fallback_teams)].copy()
                    rows = rows.dropna(subset=["ELO"])
                    if rows.empty:
                        return pd.DataFrame()

    max_elo = rows["ELO"].max()
    rows["Title Probability"] = ((rows["ELO"] - max_elo) / 95).map(math.exp)
    rows["Title Probability"] = rows["Title Probability"] / rows["Title Probability"].sum()
    rows["Sport"] = sport
    rows = rows.sort_values("Title Probability", ascending=False)
    return rows.rename(columns={"TEAM_NAME": "Team"})


def render_futures_board_page() -> None:
    st.header("Futures")
    frames = [build_championship_proxy(sport) for sport in SPORT_REGISTRY]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        st.info("No active title field is available right now.")
        return

    rows = pd.concat(frames, ignore_index=True)
    sports = list(SPORT_REGISTRY.keys())
    cols = st.columns(len(sports))
    for index, sport in enumerate(sports):
        sport_rows = rows[rows["Sport"].eq(sport)].head(10).copy()
        with cols[index]:
            st.subheader(sport)
            if sport_rows.empty:
                st.caption("No rows.")
            else:
                st.bar_chart(sport_rows.set_index("Team")[["Title Probability"]])
                display = sport_rows[["Team", "Title Probability"]].copy()
                display["Fair Odds"] = display["Title Probability"].map(probability_to_american_odds)
                display["Title Probability"] = display["Title Probability"].map("{:.1%}".format)
                st.dataframe(display, hide_index=True, width="stretch")


def build_line_shopping_rows(manual_odds: int = -110) -> pd.DataFrame:
    rows = build_all_prediction_rows()
    if rows.empty:
        return pd.DataFrame()

    side_rows = []
    for _, row in rows.iterrows():
        for side, probability_column in [("Home", "Home Probability"), ("Away", "Away Probability")]:
            team = row[f"{side} Team"]
            win_chance = float(row[probability_column])
            market_probability = american_odds_to_probability(manual_odds)
            edge = win_chance - market_probability if market_probability is not None else None
            side_rows.append(
                {
                    "Sport": row["Sport"],
                    "Matchup": row["Matchup"],
                    "Selection": team,
                    "Side": side,
                    "Market Odds": manual_odds,
                    "Fair Odds": probability_to_american_odds(win_chance),
                    "Win Chance": win_chance,
                    "Market Probability": market_probability,
                    "Edge": edge,
                    "EV/Unit": expected_value_per_unit(win_chance, manual_odds),
                    "Kelly": kelly_fraction(win_chance, manual_odds),
                }
            )
    return pd.DataFrame(side_rows).sort_values(["Edge", "Win Chance"], ascending=False)


def render_odds_board_page() -> None:
    st.header("Odds Board")
    manual_odds = st.number_input("Manual market odds applied to each side", min_value=-10000, max_value=10000, value=-110, step=5)
    rows = build_line_shopping_rows(int(manual_odds))
    if rows.empty:
        st.info("No current slate is available for odds comparison.")
        return

    display = rows.copy()
    for column in ["Win Chance", "Market Probability", "Edge", "Kelly"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: f"{value:.1%}" if pd.notna(value) else "-")
    display["EV/Unit"] = pd.to_numeric(display["EV/Unit"], errors="coerce").map(lambda value: f"{value:+.2f}" if pd.notna(value) else "-")
    st.dataframe(display, hide_index=True, width="stretch")


def load_combined_bet_tracker() -> pd.DataFrame:
    frames = []
    nba_tracker = recalculate_nba_tracker(load_nba_bet_tracker())
    if not nba_tracker.empty:
        nba_tracker = nba_tracker.copy()
        nba_tracker["Sport"] = "NBA"
        frames.append(nba_tracker)

    try:
        mlb_tracker = mlb.recalculate_tracker_results(mlb.load_bet_tracker())
        if not mlb_tracker.empty:
            mlb_tracker = mlb_tracker.copy()
            mlb_tracker["Sport"] = "MLB"
            frames.append(mlb_tracker)
    except Exception:
        pass

    try:
        nhl_tracker = nhl.recalculate_tracker_results(nhl.load_bet_tracker())
        if not nhl_tracker.empty:
            nhl_tracker = nhl_tracker.copy()
            nhl_tracker["Sport"] = "NHL"
            frames.append(nhl_tracker)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def render_bet_tracker_page() -> None:
    st.header("Bet Tracker")
    tracker = load_combined_bet_tracker()
    if tracker.empty:
        st.info("No saved picks yet. NBA, MLB, and NHL tracked picks will roll up here.")
        return

    tracker = tracker.copy()
    tracker["Stake"] = pd.to_numeric(tracker.get("Stake"), errors="coerce").fillna(0)
    tracker["Profit"] = pd.to_numeric(tracker.get("Profit"), errors="coerce").fillna(0)
    result = tracker.get("Result", pd.Series(dtype=str)).astype(str).str.lower()
    settled = tracker[result.isin(["win", "loss", "push"])].copy()
    stake = float(settled["Stake"].sum()) if not settled.empty else 0.0
    profit = float(settled["Profit"].sum()) if not settled.empty else 0.0
    roi = profit / stake if stake else None
    wins = int(result.eq("win").sum())
    losses = int(result.eq("loss").sum())
    decisions = wins + losses

    cols = st.columns(4)
    with cols[0]:
        render_feature_card("Open Picks", str(int(result.eq("open").sum())), "Across NBA, MLB, and NHL")
    with cols[1]:
        render_feature_card("Profit", f"${profit:,.2f}", f"Stake ${stake:,.2f}")
    with cols[2]:
        render_feature_card("ROI", f"{roi:.1%}" if roi is not None else "-", "Settled picks")
    with cols[3]:
        render_feature_card("Win Rate", f"{wins / decisions:.1%}" if decisions else "-", f"{wins}-{losses}")

    by_sport = tracker.groupby("Sport", as_index=False).agg(Bets=("Sport", "count"), Stake=("Stake", "sum"), Profit=("Profit", "sum"))
    by_sport["ROI"] = by_sport.apply(lambda row: row["Profit"] / row["Stake"] if row["Stake"] else None, axis=1)
    display = by_sport.copy()
    display["Stake"] = display["Stake"].map("${:,.2f}".format)
    display["Profit"] = display["Profit"].map("${:,.2f}".format)
    display["ROI"] = display["ROI"].map(lambda value: f"{value:.1%}" if value is not None else "-")
    st.dataframe(display, hide_index=True, width="stretch")
    tracker_display = tracker.copy().rename(
        columns={"Model Probability": "Win Chance", "Market Probability": "Book Probability"}
    )
    st.dataframe(tracker_display, hide_index=True, width="stretch")


def load_prediction_timeline() -> pd.DataFrame:
    timeline = read_csv_if_exists(PREDICTION_TIMELINE_PATH)
    for column in PREDICTION_TIMELINE_COLUMNS:
        if column not in timeline.columns:
            timeline[column] = None
    return timeline[PREDICTION_TIMELINE_COLUMNS]


def save_prediction_timeline(timeline: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    timeline.to_csv(PREDICTION_TIMELINE_PATH, index=False)


def append_prediction_snapshot(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    snapshot = rows[PREDICTION_TIMELINE_COLUMNS[1:]].copy()
    snapshot.insert(0, "Snapshot Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    timeline = load_prediction_timeline()
    save_prediction_timeline(pd.concat([timeline, snapshot], ignore_index=True))


def build_pick_movement_rows() -> pd.DataFrame:
    timeline = load_prediction_timeline()
    if timeline.empty:
        return pd.DataFrame()

    timeline["Snapshot Time"] = pd.to_datetime(timeline["Snapshot Time"], errors="coerce")
    rows = []
    for (sport, matchup), group in timeline.dropna(subset=["Snapshot Time"]).groupby(["Sport", "Matchup"]):
        group = group.sort_values("Snapshot Time")
        if len(group) < 2:
            continue
        previous = group.iloc[-2]
        latest = group.iloc[-1]
        previous_probability = safe_float(previous.get("Winner Probability"), 0.0) or 0.0
        latest_probability = safe_float(latest.get("Winner Probability"), 0.0) or 0.0
        rows.append(
            {
                "Sport": sport,
                "Matchup": matchup,
                "Previous Winner": previous.get("Predicted Winner"),
                "Latest Winner": latest.get("Predicted Winner"),
                "Probability Shift": latest_probability - previous_probability,
                "Latest Snapshot": latest.get("Snapshot Time"),
                "Reason": "winner changed" if previous.get("Predicted Winner") != latest.get("Predicted Winner") else "probability moved",
            }
        )
    return pd.DataFrame(rows).sort_values("Probability Shift", key=lambda series: series.abs(), ascending=False) if rows else pd.DataFrame()


def render_pick_movement_feed() -> None:
    changes = build_pick_movement_rows()
    st.subheader("Pick Movement")
    if changes.empty:
        st.caption("Save at least two snapshots in Pick History to populate this feed.")
        return

    display = changes.head(6).copy()
    display["Probability Shift"] = display["Probability Shift"].map(lambda value: f"{value:+.1%}")
    st.dataframe(display, hide_index=True, width="stretch")


def render_pick_history_page() -> None:
    st.header("Pick History")
    rows = build_all_prediction_rows()
    if st.button("Save Current Snapshot", type="primary"):
        append_prediction_snapshot(rows)
        st.success("Pick snapshot saved.")
        st.rerun()

    render_pick_movement_feed()
    timeline = load_prediction_timeline()
    if timeline.empty:
        st.info("No pick snapshots saved yet.")
        return

    display = timeline.sort_values("Snapshot Time", ascending=False).head(200).copy()
    display = display.drop(columns=[column for column in ["Model Probability", "Elo Probability"] if column in display.columns])
    for column in ["Winner Probability", "Home Probability", "Away Probability"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: f"{value:.1%}" if pd.notna(value) else "-")
    st.dataframe(display, hide_index=True, width="stretch")


def render_live_upset_alerts(rows: pd.DataFrame) -> None:
    st.subheader("Live Upset Alerts")
    if rows.empty:
        st.caption("No games to monitor.")
        return

    alerts = []
    for _, row in rows.iterrows():
        away_score = safe_float(row.get("Away Score"))
        home_score = safe_float(row.get("Home Score"))
        if away_score is None or home_score is None:
            continue
        predicted = str(row.get("Predicted Winner", ""))
        current_leader = str(row.get("Home Team")) if home_score > away_score else str(row.get("Away Team")) if away_score > home_score else ""
        if current_leader and current_leader != predicted:
            alerts.append({**row.to_dict(), "Current Leader": current_leader})

    if not alerts:
        st.caption("No pregame-pick danger spots from games with scores.")
        return

    for alert in alerts[:5]:
        st.markdown(
            f"""
            <div class="alert-row">
                <div><strong>{html.escape(alert["Current Leader"])}</strong> leading {html.escape(str(alert["Matchup"]))}</div>
                <div>Pregame pick: {html.escape(str(alert["Predicted Winner"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def load_favorites() -> pd.DataFrame:
    favorites = read_csv_if_exists(FAVORITES_PATH)
    for column in FAVORITE_COLUMNS:
        if column not in favorites.columns:
            favorites[column] = None
    return favorites[FAVORITE_COLUMNS]


def save_favorites(favorites: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    favorites.drop_duplicates(["Sport", "Team"]).to_csv(FAVORITES_PATH, index=False)


def render_favorites_page() -> None:
    st.header("Favorite Teams")
    favorites = load_favorites()
    sport = st.radio("Sport", list(SPORT_REGISTRY.keys()), horizontal=True, key="favorites_sport")
    teams = get_teams_for_sport(sport)
    current = sorted(favorites[favorites["Sport"].eq(sport)]["Team"].dropna().astype(str).tolist())
    selected = st.multiselect("Teams", teams, default=[team for team in current if team in teams])
    if st.button("Save Favorites", type="primary"):
        remaining = favorites[~favorites["Sport"].eq(sport)].copy()
        new_rows = pd.DataFrame(
            [{"Sport": sport, "Team": team, "Added At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for team in selected]
        )
        save_favorites(pd.concat([remaining, new_rows], ignore_index=True))
        st.success("Favorites saved.")
        st.rerun()

    all_rows = build_all_prediction_rows()
    favorite_teams = set(load_favorites()["Team"].dropna().astype(str))
    favorite_games = all_rows[
        all_rows["Home Team"].isin(favorite_teams) | all_rows["Away Team"].isin(favorite_teams)
    ] if not all_rows.empty else pd.DataFrame()
    if favorite_games.empty:
        st.caption("No current slate games for your favorites.")
    else:
        display = favorite_games.rename(columns={"Trust Score": "Bet Grade"})
        columns = [
            "Sport",
            "Matchup",
            "Predicted Winner",
            "Winner Probability",
            "Fair Odds",
            "Bet Grade",
            "Watchability",
            "Prediction Note",
        ]
        st.dataframe(format_prediction_table(display[columns]), hide_index=True, width="stretch")


def render_share_card_for_row(row: pd.Series) -> None:
    render_prediction_card(row, share_mode=True)
    copy = (
        f"{row['Sport']} pick: {row['Predicted Winner']} in "
        f"{row['Matchup']} ({float(row['Winner Probability']):.1%}). "
        f"Bet grade {float(row['Trust Score']):.0f}/100. Fair odds {row['Fair Odds']}."
    )
    st.text_area("Copy-ready text", copy, height=90)
    st.download_button(
        "Download Card Text",
        data=copy.encode("utf-8"),
        file_name=f"{str(row['Sport']).lower()}_prediction_card.txt",
        mime="text/plain",
    )


def render_pick_cards_page() -> None:
    st.header("Pick Cards")
    rows = build_all_prediction_rows()
    if rows.empty:
        st.info("No current predictions are available for cards.")
        return
    row = selected_prediction_row(rows, key="share_card_game")
    if row is not None:
        render_share_card_for_row(row)


def render_scenario_controls(row: pd.Series, key_prefix: str) -> None:
    st.subheader("What-If")
    base_home = float(row.get("Home Probability", 0.5))
    sport = str(row.get("Sport"))
    home_team = str(row.get("Home Team"))
    away_team = str(row.get("Away Team"))
    shift = 0.0

    if sport == "NBA":
        home_star = st.toggle(f"{home_team} key player active", value=True, key=f"{key_prefix}_home_star")
        away_star = st.toggle(f"{away_team} key player active", value=True, key=f"{key_prefix}_away_star")
        rest_shift = st.slider("Home rest / travel edge", -8.0, 8.0, 0.0, 0.5, key=f"{key_prefix}_rest")
        shift += rest_shift
        shift += 4.0 if home_star else -4.0
        shift += -4.0 if away_star else 4.0
    elif sport == "MLB":
        starter_shift = st.slider("Home starter advantage", -10.0, 10.0, 0.0, 0.5, key=f"{key_prefix}_starter")
        weather_shift = st.slider("Weather / park run-environment edge", -6.0, 6.0, 0.0, 0.5, key=f"{key_prefix}_weather")
        bullpen_shift = st.slider("Bullpen fatigue edge", -6.0, 6.0, 0.0, 0.5, key=f"{key_prefix}_bullpen")
        shift += starter_shift + weather_shift + bullpen_shift
    else:
        goalie_shift = st.slider("Home goalie edge", -10.0, 10.0, 0.0, 0.5, key=f"{key_prefix}_goalie")
        special_teams_shift = st.slider("Power play / penalty kill edge", -6.0, 6.0, 0.0, 0.5, key=f"{key_prefix}_special_teams")
        travel_shift = st.slider("Home rest / travel edge", -6.0, 6.0, 0.0, 0.5, key=f"{key_prefix}_nhl_rest")
        shift += goalie_shift + special_teams_shift + travel_shift

    adjusted_home = apply_probability_shift(base_home, shift)
    adjusted_away = 1 - adjusted_home
    adjusted_winner = home_team if adjusted_home >= adjusted_away else away_team
    adjusted_probability = max(adjusted_home, adjusted_away)
    cols = st.columns(3)
    cols[0].metric("Base Home", f"{base_home:.1%}")
    cols[1].metric("Adjusted Home", f"{adjusted_home:.1%}", f"{adjusted_home - base_home:+.1%}")
    cols[2].metric("Scenario Winner", f"{adjusted_winner}", f"{adjusted_probability:.1%}")


def render_what_if_page(sport: str) -> None:
    rows = build_prediction_board_for_sport(sport)
    if rows.empty:
        teams = get_teams_for_sport(sport)
        config = SPORT_REGISTRY[sport]
        home = st.selectbox("Home", teams, index=teams.index(config.default_home) if config.default_home in teams else 0, key=f"scenario_home_{sport}")
        away = st.selectbox("Away", teams, index=teams.index(config.default_away) if config.default_away in teams else min(1, len(teams) - 1), key=f"scenario_away_{sport}")
        if home == away:
            st.warning("Choose two different teams.")
            return
        row = pd.Series(predict_matchup_for_sport(sport, home, away))
    else:
        row = selected_prediction_row(rows, key=f"scenario_game_{sport}")

    if row is not None:
        render_prediction_card(row)
        render_scenario_controls(row, key_prefix=f"scenario_{sport}")


def render_odds_comparison_for_row(row: pd.Series, key_prefix: str) -> None:
    st.subheader("Odds")
    market_odds = st.number_input("Market odds for home team", min_value=-10000, max_value=10000, value=-110, step=5, key=f"{key_prefix}_market_odds")
    user_pick = st.selectbox("Your pick", [str(row["Home Team"]), str(row["Away Team"])], key=f"{key_prefix}_user_pick")
    market_probability = american_odds_to_probability(market_odds) or 0.5
    home_probability = float(row.get("Home Probability", 0.5))
    away_probability = float(row.get("Away Probability", 0.5))
    odds_rows = pd.DataFrame(
        [
            {
                "Side": "Home",
                "Team": row["Home Team"],
                "Win Chance": home_probability,
                "Market Probability": market_probability,
                "Edge": home_probability - market_probability,
                "Fair Odds": probability_to_american_odds(home_probability),
            },
            {
                "Side": "Away",
                "Team": row["Away Team"],
                "Win Chance": away_probability,
                "Market Probability": 1 - market_probability,
                "Edge": away_probability - (1 - market_probability),
                "Fair Odds": probability_to_american_odds(away_probability),
            },
        ]
    )
    st.bar_chart(odds_rows.set_index("Team")[["Win Chance", "Market Probability"]])
    display = odds_rows.copy()
    for column in ["Win Chance", "Market Probability", "Edge"]:
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(f"Your pick: {user_pick}. Home edge vs market: {home_probability - market_probability:+.1%}.")


def render_hub_overview_page() -> None:
    summary_tab, brief_tab = st.tabs(["Summary", "Daily Brief"])

    with summary_tab:
        render_league_hub()

    with brief_tab:
        render_daily_brief_page()


def render_hub_games_page() -> None:
    center_tab, upset_tab = st.tabs(["Game Center", "Upset Radar"])

    with center_tab:
        render_game_center_page()

    with upset_tab:
        render_upset_radar_page()


def render_hub_betting_page() -> None:
    board_tab, cards_tab, favorites_tab = st.tabs(["Odds Board", "Pick Cards", "Favorites"])

    with board_tab:
        render_odds_board_page()

    with cards_tab:
        render_pick_cards_page()

    with favorites_tab:
        render_favorites_page()


def render_hub_tracker_page() -> None:
    tracker_tab, history_tab = st.tabs(["Bet Tracker", "Pick History"])

    with tracker_tab:
        render_bet_tracker_page()

    with history_tab:
        render_pick_history_page()


def render_nba_games_page(teams: list[str]) -> None:
    today_tab, center_tab, what_if_tab = st.tabs(["Today", "Game Center", "What-If"])

    with today_tab:
        nba.render_today_games_section(teams)

    with center_tab:
        render_game_center_page("NBA")

    with what_if_tab:
        render_what_if_page("NBA")


def render_mlb_games_page() -> None:
    today_tab, center_tab, what_if_tab = st.tabs(["Today", "Game Center", "What-If"])

    with today_tab:
        mlb.render_today_live_fragment()

    with center_tab:
        render_game_center_page("MLB")

    with what_if_tab:
        render_what_if_page("MLB")


def render_nhl_games_page() -> None:
    today_tab, center_tab, what_if_tab = st.tabs(["Today", "Game Center", "What-If"])

    with today_tab:
        nhl.render_today_live_fragment()

    with center_tab:
        render_game_center_page("NHL")

    with what_if_tab:
        render_what_if_page("NHL")


def render_nba_playoffs_page(teams: list[str]) -> None:
    series_tab, bracket_tab = st.tabs(["Series", "Bracket"])

    with series_tab:
        nba.render_series_section(teams)

    with bracket_tab:
        nba.render_bracket_section(teams)


def render_mlb_playoffs_page() -> None:
    series_tab, bracket_tab = st.tabs(["Series", "Bracket"])

    with series_tab:
        mlb.render_series_view()

    with bracket_tab:
        mlb.render_bracket_view()


def render_nhl_playoffs_page() -> None:
    series_tab, bracket_tab = st.tabs(["Series", "Bracket"])

    with series_tab:
        nhl.render_series_view()

    with bracket_tab:
        nhl.render_bracket_view()


def render_hub_page(page: str) -> None:
    if page == "Overview":
        render_hub_overview_page()
    elif page == "Games":
        render_hub_games_page()
    elif page == "Betting":
        render_hub_betting_page()
    elif page == "Futures":
        render_futures_board_page()
    elif page == "Tracker":
        render_hub_tracker_page()


def render_nba_page(page: str) -> None:
    nba.inject_custom_css()
    render_unified_css()
    summary = get_nba_summary()
    render_sport_banner("NBA", page, summary)

    if not nba.MODEL_PATH.exists():
        st.error("Missing NBA pricing file. Run the NBA data refresh commands first.")
        return
    if not nba.FEATURES_PATH.exists():
        st.error("Missing NBA features file. Run `python src/features.py` first.")
        return
    if not nba.TEAM_STRENGTH_PATH.exists():
        st.error("Missing NBA team strength file. Run `python src/team_strength.py` first.")
        return

    strength = nba.load_team_strength()
    teams = nba.get_teams_from_strength(strength)

    if page == "Home":
        nba.render_home_dashboard(teams)
    elif page == "Games":
        render_nba_games_page(teams)
    elif page == "Playoffs":
        render_nba_playoffs_page(teams)
    elif page == "Betting":
        render_nba_betting_page()
    elif page == "Standings":
        render_nba_standings_page()

    nba.render_today_player_profile_dialog()


def render_mlb_page(page: str) -> None:
    st.markdown(mlb.APP_CSS, unsafe_allow_html=True)
    render_unified_css()
    summary = get_mlb_summary()
    render_sport_banner("MLB", page, summary)

    if page == "Home":
        mlb.render_home_view()
    elif page == "Games":
        render_mlb_games_page()
    elif page == "Playoffs":
        render_mlb_playoffs_page()
    elif page == "Betting":
        mlb.render_betting_view()
    elif page == "Standings":
        mlb.render_standings_view()


def render_nhl_page(page: str) -> None:
    st.markdown(nhl.APP_CSS, unsafe_allow_html=True)
    render_unified_css()
    summary = get_nhl_summary()
    render_sport_banner("NHL", page, summary)

    if page == "Home":
        nhl.render_home_view()
    elif page == "Games":
        render_nhl_games_page()
    elif page == "Playoffs":
        render_nhl_playoffs_page()
    elif page == "Betting":
        nhl.render_betting_view()
    elif page == "Standings":
        nhl.render_standings_view()


def ensure_valid_navigation() -> None:
    if "selected_sport" not in st.session_state:
        st.session_state["selected_sport"] = "Betting Hub"

    if st.session_state["selected_sport"] not in SPORT_OPTIONS:
        st.session_state["selected_sport"] = "Betting Hub"

    if "hub_page" not in st.session_state or st.session_state["hub_page"] not in HUB_PAGES:
        st.session_state["hub_page"] = "Overview"

    if "nba_page" not in st.session_state or st.session_state["nba_page"] not in NBA_PAGES:
        st.session_state["nba_page"] = "Home"

    if "mlb_page" not in st.session_state or st.session_state["mlb_page"] not in MLB_PAGES:
        st.session_state["mlb_page"] = "Home"

    if "nhl_page" not in st.session_state or st.session_state["nhl_page"] not in NHL_PAGES:
        st.session_state["nhl_page"] = "Home"


def render_sidebar() -> tuple[str, str | None]:
    with st.sidebar:
        st.markdown("### Sports")
        selected_sport = st.radio(
            "Sport",
            SPORT_OPTIONS,
            key="selected_sport",
            label_visibility="collapsed",
        )

        selected_page = None
        if selected_sport == "Betting Hub":
            st.markdown("### Betting Pages")
            selected_page = st.radio(
                "Hub Page",
                HUB_PAGES,
                key="hub_page",
                label_visibility="collapsed",
            )
        elif selected_sport == "NBA":
            st.markdown("### NBA Pages")
            selected_page = st.radio(
                "NBA Page",
                NBA_PAGES,
                key="nba_page",
                label_visibility="collapsed",
            )
        elif selected_sport == "MLB":
            st.markdown("### MLB Pages")
            selected_page = st.radio(
                "MLB Page",
                MLB_PAGES,
                key="mlb_page",
                label_visibility="collapsed",
            )
        elif selected_sport == "NHL":
            st.markdown("### NHL Pages")
            selected_page = st.radio(
                "NHL Page",
                NHL_PAGES,
                key="nhl_page",
                label_visibility="collapsed",
            )

        st.divider()
        if st.button("Clear Caches", width="stretch"):
            try:
                nba.clear_app_caches()
            except Exception:
                pass
            try:
                mlb.clear_mlb_caches()
            except Exception:
                pass
            try:
                nhl.clear_nhl_caches()
            except Exception:
                pass
            st.rerun()

    return selected_sport, selected_page


def main() -> None:
    st.set_page_config(
        page_title="Sportsbook Edge",
        page_icon="SP",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_valid_navigation()
    selected_sport, selected_page = render_sidebar()
    render_unified_css()
    render_topbar(selected_sport)

    if selected_sport == "Betting Hub":
        render_hub_page(selected_page or "Overview")
    elif selected_sport == "NBA":
        render_nba_page(selected_page or "Home")
    elif selected_sport == "MLB":
        render_mlb_page(selected_page or "Home")
    elif selected_sport == "NHL":
        render_nhl_page(selected_page or "Home")


if __name__ == "__main__":
    main()
