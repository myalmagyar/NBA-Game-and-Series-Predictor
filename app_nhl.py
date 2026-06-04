# app_nhl.py

from __future__ import annotations

from datetime import date, datetime
import html
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import streamlit as st

from src.nhl_api import iter_score_games, load_score
from src.nhl_features import FEATURE_COLUMNS, HOME_ELO_ADVANTAGE, expected_score


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RAW_GAMES_PATH = DATA_DIR / "nhl_raw_games.csv"
FEATURES_PATH = DATA_DIR / "nhl_model_features.csv"
TEAM_STRENGTH_PATH = DATA_DIR / "nhl_current_team_strength.csv"
MODEL_PATH = MODELS_DIR / "nhl_game_winner_model.joblib"
METRICS_PATH = DATA_DIR / "nhl_model_metrics.csv"
BET_TRACKER_PATH = DATA_DIR / "nhl_bet_tracker.csv"

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
    "Notes",
]

APP_CSS = """
<style>
    .nhl-small-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0.75rem 0 1rem;
    }
    .nhl-rink-card {
        background: linear-gradient(135deg, #f8fbff, #eef8fb);
        border: 1px solid #d5e5f1;
        border-radius: 8px;
        box-shadow: 0 12px 30px rgba(15, 33, 55, 0.08);
        padding: 0.9rem;
    }
    .nhl-rink-title {
        color: #12263f;
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.2;
    }
    .nhl-rink-meta {
        color: #5c7188;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .nhl-rink-value {
        color: #0b3b75;
        font-size: 1.35rem;
        font-weight: 950;
        line-height: 1.1;
        margin-top: 0.5rem;
    }
    @media (max-width: 900px) {
        .nhl-small-grid { grid-template-columns: 1fr; }
    }
</style>
"""


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_raw_games() -> pd.DataFrame:
    games = read_csv_if_exists(RAW_GAMES_PATH)
    if not games.empty and "GAME_DATE" in games.columns:
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    return games


@st.cache_data(show_spinner=False)
def load_team_strength() -> pd.DataFrame:
    return read_csv_if_exists(TEAM_STRENGTH_PATH)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        return {}
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return {}


def clear_nhl_caches() -> None:
    load_raw_games.clear()
    load_team_strength.clear()
    load_model_bundle.clear()
    load_today_games.clear()
    load_next_upcoming_games.clear()
    load_local_slate_games.clear()


def get_available_teams() -> list[str]:
    strength = load_team_strength()
    if strength.empty or "TEAM_NAME" not in strength.columns:
        return []
    return strength.sort_values("TEAM_NAME")["TEAM_NAME"].dropna().astype(str).tolist()


def format_game_time(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "TBD"
    return parsed.tz_convert(ZoneInfo("America/New_York")).strftime("%b %-d, %-I:%M %p ET")


def format_percent(value: object, digits: int = 1) -> str:
    parsed = safe_float(value)
    return f"{parsed:.{digits}%}" if parsed is not None else "-"


def american_odds_to_probability(odds: object) -> float | None:
    value = safe_float(str(odds).replace("+", ""))
    if value is None or value == 0:
        return None
    if value > 0:
        return 100 / (value + 100)
    return abs(value) / (abs(value) + 100)


def american_odds_profit_per_unit(odds: object) -> float | None:
    value = safe_float(str(odds).replace("+", ""))
    if value is None or value == 0:
        return None
    return value / 100 if value > 0 else 100 / abs(value)


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
    return max(0.0, ((payout * probability) - (1 - probability)) / payout)


def format_american_odds(value: object) -> str:
    parsed = safe_float(str(value).replace("+", ""))
    return f"{int(round(parsed)):+d}" if parsed is not None and parsed != 0 else "-"


def format_money(value: object, signed: bool = False) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "-"
    prefix = "+" if signed and parsed > 0 else ""
    return f"{prefix}${parsed:,.2f}"


def team_strength_row(team: str) -> pd.Series:
    strength = load_team_strength()
    if strength.empty:
        raise FileNotFoundError("Missing NHL team strength data.")
    rows = strength[strength["TEAM_NAME"].astype(str).eq(team)]
    if rows.empty:
        raise ValueError(f"Unknown NHL team: {team}")
    return rows.iloc[0]


def context_value(game_context: pd.Series | dict | None, key: str, default: object = None) -> object:
    if game_context is None:
        return default
    if isinstance(game_context, pd.Series):
        return game_context.get(key, default)
    return game_context.get(key, default)


def feature_row_for_matchup(home_team: str, away_team: str, game_context: pd.Series | dict | None = None) -> pd.DataFrame:
    home = team_strength_row(home_team)
    away = team_strength_row(away_team)
    home_elo = safe_float(home.get("ELO"), 1500.0) or 1500.0
    away_elo = safe_float(away.get("ELO"), 1500.0) or 1500.0
    values = {
        "HOME_ELO_WIN_PROB": expected_score(home_elo + HOME_ELO_ADVANTAGE, away_elo),
        "DIFF_ELO": home_elo - away_elo,
        "IS_PLAYOFF_GAME": 1.0 if safe_int(context_value(game_context, "GAME_TYPE"), default=0) == 3 else 0.0,
    }
    mappings = {
        "DIFF_SEASON_WIN_PCT": "SEASON_WIN_PCT",
        "DIFF_SEASON_GOAL_DIFF_PER_GAME": "SEASON_GOAL_DIFF_PER_GAME",
        "DIFF_SEASON_AVG_GOALS_FOR": "SEASON_AVG_GOALS_FOR",
        "DIFF_SEASON_AVG_GOALS_AGAINST": "SEASON_AVG_GOALS_AGAINST",
        "DIFF_ROLLING_WIN_PCT_5": "ROLLING_WIN_PCT_5",
        "DIFF_ROLLING_GOAL_DIFF_5": "ROLLING_GOAL_DIFF_5",
        "DIFF_ROLLING_GOALS_FOR_5": "ROLLING_GOALS_FOR_5",
        "DIFF_ROLLING_GOALS_AGAINST_5": "ROLLING_GOALS_AGAINST_5",
        "DIFF_ROLLING_WIN_PCT_10": "ROLLING_WIN_PCT_10",
        "DIFF_ROLLING_GOAL_DIFF_10": "ROLLING_GOAL_DIFF_10",
        "DIFF_ROLLING_GOALS_FOR_10": "ROLLING_GOALS_FOR_10",
        "DIFF_ROLLING_GOALS_AGAINST_10": "ROLLING_GOALS_AGAINST_10",
        "DIFF_DAYS_REST": "DAYS_REST",
    }
    for feature, column in mappings.items():
        values[feature] = (safe_float(home.get(column), 0.0) or 0.0) - (safe_float(away.get(column), 0.0) or 0.0)
    return pd.DataFrame([{column: values.get(column, 0.0) for column in FEATURE_COLUMNS}])


def predict_game_details(home_team: str, away_team: str, game_context: pd.Series | dict | None = None) -> dict:
    row = feature_row_for_matchup(home_team, away_team, game_context)
    elo_probability = float(row["HOME_ELO_WIN_PROB"].iloc[0])
    bundle = load_model_bundle()
    model_probability = elo_probability
    if bundle and "model" in bundle:
        feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        model_probability = float(bundle["model"].predict_proba(row[feature_columns])[0][1])
    home_probability = min(max((0.78 * model_probability) + (0.22 * elo_probability), 0.03), 0.97)
    away_probability = 1 - home_probability
    winner = home_team if home_probability >= away_probability else away_team
    winner_probability = max(home_probability, away_probability)
    home = team_strength_row(home_team)
    away = team_strength_row(away_team)
    explanation = [
        f"Power rating {home_team} {safe_float(home.get('ELO'), 1500):.0f} vs {away_team} {safe_float(away.get('ELO'), 1500):.0f}",
        f"Recent form {format_percent(home.get('ROLLING_WIN_PCT_10'), 0)} vs {format_percent(away.get('ROLLING_WIN_PCT_10'), 0)}",
        f"Goal diff/G {safe_float(home.get('SEASON_GOAL_DIFF_PER_GAME'), 0):+.2f} vs {safe_float(away.get('SEASON_GOAL_DIFF_PER_GAME'), 0):+.2f}",
    ]
    return {
        "winner": winner,
        "winner_probability": winner_probability,
        "home_probability": home_probability,
        "away_probability": away_probability,
        "model_probability": model_probability,
        "elo_probability": elo_probability,
        "explanation": explanation,
    }


def normalize_status(row: pd.Series) -> str:
    state = str(row.get("STATUS", "") or "")
    if state == "FINAL":
        return "Final"
    if state in {"LIVE", "CRIT"}:
        period = row.get("PERIOD")
        clock = str(row.get("CLOCK", "") or "")
        return f"P{safe_int(period)} {clock}".strip()
    if state in {"FUT", "PRE"}:
        return "Scheduled"
    return state or "Scheduled"


@st.cache_data(ttl=60, show_spinner=False)
def load_local_slate_games(require_today: bool = False) -> pd.DataFrame:
    """Use the refreshed local NHL schedule when the live score feed is unavailable."""
    games = load_raw_games()
    if games.empty or "GAME_DATE" not in games.columns:
        return pd.DataFrame()

    games = games.copy()
    games["_GAME_DAY"] = pd.to_datetime(games["GAME_DATE"], errors="coerce").dt.normalize()
    today = pd.Timestamp(datetime.now(tz=ZoneInfo("America/New_York")).date())
    upcoming = games[games["_GAME_DAY"].ge(today)].copy()
    if upcoming.empty:
        return pd.DataFrame()

    slate_day = today if require_today else upcoming["_GAME_DAY"].min()
    slate = upcoming[upcoming["_GAME_DAY"].eq(slate_day)].copy()
    if slate.empty:
        return pd.DataFrame()

    for column in ["AWAY_ODDS", "HOME_ODDS", "ODDS_PROVIDER", "CLOCK", "PERIOD"]:
        if column not in slate.columns:
            slate[column] = None
    slate["STATUS"] = slate.apply(normalize_status, axis=1)
    return slate.drop(columns=["_GAME_DAY"], errors="ignore").reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def load_today_games() -> pd.DataFrame:
    today = datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat()
    try:
        payload = load_score(today)
    except Exception:
        return load_local_slate_games(require_today=True)
    games = pd.DataFrame(iter_score_games(payload))
    if games.empty:
        return load_local_slate_games(require_today=True)
    games["STATUS"] = games.apply(normalize_status, axis=1)
    return games


@st.cache_data(ttl=60, show_spinner=False)
def load_next_upcoming_games() -> pd.DataFrame:
    today = datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat()
    try:
        payload = load_score(today)
    except Exception:
        return load_local_slate_games(require_today=False)
    next_date = payload.get("nextDate")
    if not next_date:
        return load_local_slate_games(require_today=False)
    try:
        payload = load_score(str(next_date))
    except Exception:
        return load_local_slate_games(require_today=False)
    games = pd.DataFrame(iter_score_games(payload))
    if games.empty:
        return load_local_slate_games(require_today=False)
    games["STATUS"] = games.apply(normalize_status, axis=1)
    return games


def load_betting_slate_games(slate: str = "Today") -> pd.DataFrame:
    games = load_today_games()
    return games if not games.empty else load_next_upcoming_games()


def build_game_predictions(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    rows = []
    available = set(get_available_teams())
    for _, game in games.iterrows():
        home_team = str(game.get("HOME_TEAM", ""))
        away_team = str(game.get("AWAY_TEAM", ""))
        if home_team not in available or away_team not in available:
            continue
        details = predict_game_details(home_team, away_team, game)
        rows.append(
            {
                **game.to_dict(),
                "PREDICTED_WINNER": details["winner"],
                "WINNER_PROBABILITY": details["winner_probability"],
                "HOME_WIN_PROBABILITY": details["home_probability"],
                "AWAY_WIN_PROBABILITY": details["away_probability"],
                "MODEL_PROBABILITY": details["model_probability"],
                "ELO_PROBABILITY": details["elo_probability"],
                "PREDICTION_EXPLANATION": " / ".join(details["explanation"]),
            }
        )
    return pd.DataFrame(rows)


def render_card_grid(cards: list[dict[str, str]]) -> None:
    card_html = []
    for card in cards:
        card_html.append(
            f"""
            <div class="nhl-rink-card">
                <div class="nhl-rink-title">{html.escape(card["title"])}</div>
                <div class="nhl-rink-value">{html.escape(card["value"])}</div>
                <div class="nhl-rink-meta">{html.escape(card["note"])}</div>
            </div>
            """
        )
    st.html(f'<div class="nhl-small-grid">{"".join(card_html)}</div>')


def render_home_view() -> None:
    strength = load_team_strength()
    features = read_csv_if_exists(FEATURES_PATH)
    raw = load_raw_games()
    top = strength.iloc[0] if not strength.empty else pd.Series(dtype=object)
    completed = raw.copy()
    if not completed.empty and {"HOME_SCORE", "AWAY_SCORE"}.issubset(completed.columns):
        completed["HOME_SCORE"] = pd.to_numeric(completed["HOME_SCORE"], errors="coerce")
        completed["AWAY_SCORE"] = pd.to_numeric(completed["AWAY_SCORE"], errors="coerce")
        completed = completed[completed["HOME_SCORE"].notna() & completed["AWAY_SCORE"].notna()]
    latest = pd.to_datetime(completed["GAME_DATE"], errors="coerce").max() if not completed.empty and "GAME_DATE" in completed.columns else pd.NaT
    render_card_grid(
        [
            {
                "title": "Power Team",
                "value": str(top.get("TEAM_NAME", "TBD")),
                "note": f"ELO {safe_float(top.get('ELO'), 0):.0f}" if not top.empty else "Run NHL refresh",
            },
            {
                "title": "Data Through",
                "value": latest.strftime("%b %-d, %Y") if not pd.isna(latest) else "TBD",
                "note": f"{len(raw):,} raw games",
            },
            {
                "title": "Model Rows",
                "value": f"{len(features):,}",
                "note": "NHL matchup training rows",
            },
        ]
    )
    render_today_live_fragment()


def render_today_live_fragment() -> None:
    games = load_betting_slate_games("Today")
    predictions = build_game_predictions(games)
    if predictions.empty:
        st.info("No NHL games are available from the current or next slate.")
        return
    for _, row in predictions.iterrows():
        away = str(row.get("AWAY_TEAM", ""))
        home = str(row.get("HOME_TEAM", ""))
        st.html(
            f"""
            <div class="nhl-rink-card">
                <div class="nhl-rink-title">{html.escape(away)} at {html.escape(home)}</div>
                <div class="nhl-rink-meta">{html.escape(str(row.get("STATUS", "Scheduled")))} / {html.escape(format_game_time(row.get("GAME_DATETIME")))}</div>
                <div class="nhl-rink-value">{html.escape(str(row.get("PREDICTED_WINNER", "")))}</div>
                <div class="nhl-rink-meta">Win chance {float(row.get("WINNER_PROBABILITY", 0.5)):.1%} / Fair {probability_to_american_odds(row.get("WINNER_PROBABILITY"))}</div>
            </div>
            """
        )


def build_playoff_series_states() -> list[dict]:
    games = load_raw_games()
    if games.empty:
        return []
    games = games[games["GAME_TYPE"].eq(3)].copy()
    if games.empty:
        return []
    latest_season = games.sort_values("GAME_DATE")["SEASON"].iloc[-1]
    games = games[games["SEASON"].eq(latest_season)].copy()
    states = []
    for series_key, series_games in games.groupby(
        games.apply(lambda row: " vs ".join(sorted([str(row["HOME_TEAM"]), str(row["AWAY_TEAM"])])), axis=1)
    ):
        wins: dict[str, int] = {}
        rows = []
        for _, game in series_games.sort_values(["GAME_DATE", "GAME_ID"]).iterrows():
            home_score = safe_float(game.get("HOME_SCORE"), 0) or 0
            away_score = safe_float(game.get("AWAY_SCORE"), 0) or 0
            winner = str(game["HOME_TEAM"]) if home_score > away_score else str(game["AWAY_TEAM"])
            wins[winner] = wins.get(winner, 0) + 1
            rows.append(
                {
                    "Date": str(game.get("GAME_DATE"))[:10],
                    "Winner": winner,
                    "Score": f"{safe_int(game.get('AWAY_SCORE'))}-{safe_int(game.get('HOME_SCORE'))}",
                }
            )
        leader = max(wins, key=wins.get) if wins else ""
        states.append(
            {
                "series_key": str(series_key),
                "wins": wins,
                "leader": leader,
                "completed": wins.get(leader, 0) >= 4 if leader else False,
                "games_played": len(series_games),
                "game_rows": rows,
            }
        )
    return sorted(states, key=lambda row: (row["completed"], -row["games_played"], row["series_key"]))


def render_series_view() -> None:
    states = build_playoff_series_states()
    if not states:
        st.info("No NHL playoff series data is available yet.")
        return
    for state in states:
        status = "Complete" if state["completed"] else "Active"
        wins = ", ".join(f"{team} {count}" for team, count in state["wins"].items())
        with st.expander(f"{status}: {state['series_key']} ({wins})", expanded=not state["completed"]):
            st.dataframe(pd.DataFrame(state["game_rows"]), hide_index=True, width="stretch")


def render_bracket_view() -> None:
    strength = load_team_strength()
    if strength.empty:
        st.info("Run the NHL refresh before using bracket futures.")
        return
    field = strength.sort_values("ELO", ascending=False).head(16)[["TEAM_NAME", "ELO"]].copy()
    max_elo = pd.to_numeric(field["ELO"], errors="coerce").max()
    field["Championship Probability"] = ((field["ELO"] - max_elo) / 95).map(math.exp)
    field["Championship Probability"] = field["Championship Probability"] / field["Championship Probability"].sum()
    st.bar_chart(field.set_index("TEAM_NAME")[["Championship Probability"]])
    display = field.rename(columns={"TEAM_NAME": "Team"}).copy()
    display["Championship Probability"] = display["Championship Probability"].map("{:.1%}".format)
    st.dataframe(display, hide_index=True, width="stretch")


def render_standings_view() -> None:
    strength = load_team_strength()
    if strength.empty:
        st.warning("Missing NHL standings data. Run the NHL refresh first.")
        return
    display = strength[
        [
            "TEAM_NAME",
            "WINS",
            "LOSSES",
            "SEASON_WIN_PCT",
            "SEASON_GOAL_DIFF_PER_GAME",
            "ROLLING_WIN_PCT_10",
            "ELO",
        ]
    ].copy()
    display = display.rename(
        columns={
            "TEAM_NAME": "Team",
            "WINS": "W",
            "LOSSES": "L",
            "SEASON_WIN_PCT": "Pct",
            "SEASON_GOAL_DIFF_PER_GAME": "Goal Diff/G",
            "ROLLING_WIN_PCT_10": "Last 10",
        }
    )
    display["Pct"] = pd.to_numeric(display["Pct"], errors="coerce").map(lambda value: f"{value:.3f}")
    display["Goal Diff/G"] = pd.to_numeric(display["Goal Diff/G"], errors="coerce").map(lambda value: f"{value:+.2f}")
    display["Last 10"] = pd.to_numeric(display["Last 10"], errors="coerce").map(lambda value: f"{value:.0%}")
    st.dataframe(display.sort_values("ELO", ascending=False), hide_index=True, width="stretch")


def load_bet_tracker() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    tracker = read_csv_if_exists(BET_TRACKER_PATH)
    for column in BET_TRACKER_COLUMNS:
        if column not in tracker.columns:
            tracker[column] = None
    return tracker[BET_TRACKER_COLUMNS]


def save_bet_tracker(tracker: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tracker.to_csv(BET_TRACKER_PATH, index=False)


def next_bet_id(tracker: pd.DataFrame) -> int:
    ids = pd.to_numeric(tracker.get("Bet ID"), errors="coerce").dropna() if not tracker.empty else pd.Series(dtype=float)
    return int(ids.max()) + 1 if not ids.empty else 1


def calculate_bet_profit(result: object, odds: object, stake: object) -> float | None:
    result_text = str(result or "").lower()
    stake_value = safe_float(stake)
    payout = american_odds_profit_per_unit(odds)
    if stake_value is None:
        return None
    if result_text == "win" and payout is not None:
        return stake_value * payout
    if result_text == "loss":
        return -stake_value
    if result_text == "push":
        return 0.0
    return None


def recalculate_tracker_results(tracker: pd.DataFrame) -> pd.DataFrame:
    tracker = tracker.copy()
    if tracker.empty:
        return tracker
    tracker["Profit"] = tracker.apply(lambda row: calculate_bet_profit(row.get("Result"), row.get("Odds"), row.get("Stake")), axis=1)
    return tracker


def betting_edge_signal(edge: object) -> tuple[str, str]:
    value = safe_float(edge)
    if value is None:
        return "Add odds", "edge-none"
    if value >= 0.05:
        return "Good value", "edge-strong"
    if value >= 0.02:
        return "Small value", "edge-small"
    if value <= -0.02:
        return "Price is high", "edge-none"
    return "Fair price", "edge-none"


def selection_odds(row: pd.Series, selection: str) -> object:
    if selection == str(row.get("HOME_TEAM")):
        return row.get("HOME_ODDS")
    return row.get("AWAY_ODDS")


def render_odds_board() -> None:
    games = build_game_predictions(load_betting_slate_games("Today"))
    if games.empty:
        st.info("No NHL games are available for betting right now.")
        return
    for index, (_, row) in enumerate(games.iterrows(), start=1):
        home = str(row.get("HOME_TEAM"))
        away = str(row.get("AWAY_TEAM"))
        game_key = str(row.get("GAME_ID") or f"{away}_{home}")
        options = [home, away]
        selected = st.selectbox(f"{away} at {home}", options, key=f"nhl_bet_selection_{game_key}")
        model_probability = float(row.get("HOME_WIN_PROBABILITY") if selected == home else row.get("AWAY_WIN_PROBABILITY"))
        feed_odds = selection_odds(row, selected)
        default_odds = safe_int(str(feed_odds).replace("+", ""), default=-110) if feed_odds not in [None, ""] else -110
        odds_col, stake_col = st.columns(2)
        with odds_col:
            odds = st.number_input("Sportsbook odds", min_value=-10000, max_value=10000, value=default_odds, step=5, key=f"nhl_odds_{game_key}")
        with stake_col:
            stake = st.number_input("Stake", min_value=0.0, value=10.0, step=1.0, key=f"nhl_stake_{game_key}")
        market_probability = american_odds_to_probability(odds)
        edge = model_probability - market_probability if market_probability is not None else None
        ev = expected_value_per_unit(model_probability, odds)
        kelly = kelly_fraction(model_probability, odds)
        signal, signal_class = betting_edge_signal(edge)
        edge_text = f"{edge * 100:+.1f} pts" if edge is not None else "-"
        st.html(
            f"""
            <div class="betting-card-shell">
                <div class="betting-card-head">
                    <div>
                        <div class="betting-game-title">{html.escape(away)} at {html.escape(home)}</div>
                        <div class="betting-game-meta">{html.escape(str(row.get("STATUS", "Scheduled")))} / {html.escape(format_game_time(row.get("GAME_DATETIME")))}</div>
                    </div>
                    <span class="edge-pill {signal_class}">{html.escape(signal)}</span>
                </div>
                <div class="betting-pick-row">
                    <div class="betting-simple-stat"><div class="betting-simple-label">Pick</div><div class="betting-simple-value">{html.escape(selected)}</div></div>
                    <div class="betting-simple-stat"><div class="betting-simple-label">Win chance</div><div class="betting-simple-value">{model_probability:.0%}</div></div>
                    <div class="betting-simple-stat"><div class="betting-simple-label">Odds</div><div class="betting-simple-value">{html.escape(format_american_odds(odds))}</div><div class="betting-simple-note">{html.escape(str(row.get("ODDS_PROVIDER") or "Manual price"))}</div></div>
                    <div class="betting-simple-stat"><div class="betting-simple-label">Edge</div><div class="betting-simple-value">{html.escape(edge_text)}</div></div>
                </div>
            </div>
            """
        )
        notes = st.text_input("Notes", value="", key=f"nhl_notes_{game_key}")
        if st.button("Save pick", type="primary", width="stretch", key=f"nhl_save_pick_{game_key}"):
            tracker = load_bet_tracker()
            new_row = {
                "Bet ID": next_bet_id(tracker),
                "Date Logged": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Game Date": row.get("GAME_DATE", ""),
                "Matchup": f"{away} at {home}",
                "Market": "Moneyline",
                "Selection": selected,
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
            save_bet_tracker(pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True))
            st.success("NHL pick saved.")
            st.rerun()


def render_tracker_cards(tracker: pd.DataFrame, settled: bool = False) -> None:
    if tracker.empty:
        st.info("No saved NHL picks yet.")
        return
    for _, row in tracker.sort_values("Date Logged", ascending=False).iterrows():
        result = str(row.get("Result") or "Open")
        st.html(
            f"""
            <div class="betting-card-shell">
                <div class="betting-card-head">
                    <div>
                        <div class="betting-game-title">{html.escape(str(row.get("Selection", "Pick")))}</div>
                        <div class="betting-game-meta">{html.escape(str(row.get("Market", "Moneyline")))} / {html.escape(str(row.get("Matchup", "")))}</div>
                    </div>
                    <span class="edge-pill {'open' if result.lower() == 'open' else result.lower()}">{html.escape(result)}</span>
                </div>
                <div class="betting-pick-row">
                    <div class="betting-simple-stat"><div class="betting-simple-label">Odds</div><div class="betting-simple-value">{html.escape(format_american_odds(row.get("Odds")))}</div></div>
                    <div class="betting-simple-stat"><div class="betting-simple-label">Stake</div><div class="betting-simple-value">{html.escape(format_money(row.get("Stake")))}</div></div>
                    <div class="betting-simple-stat"><div class="betting-simple-label">Profit</div><div class="betting-simple-value">{html.escape(format_money(row.get("Profit"), signed=True))}</div></div>
                    <div class="betting-simple-stat"><div class="betting-simple-label">Edge</div><div class="betting-simple-value">{(safe_float(row.get("Edge"), 0) or 0) * 100:+.1f} pts</div></div>
                </div>
            </div>
            """
        )


def render_saved_betting_picks() -> None:
    tracker = recalculate_tracker_results(load_bet_tracker())
    active = tracker[~tracker["Result"].astype(str).str.lower().isin(["win", "loss", "push"])] if not tracker.empty else tracker
    render_tracker_cards(active)


def render_settled_betting_picks() -> None:
    tracker = recalculate_tracker_results(load_bet_tracker())
    settled = tracker[tracker["Result"].astype(str).str.lower().isin(["win", "loss", "push"])] if not tracker.empty else tracker
    render_tracker_cards(settled, settled=True)
    with st.expander("Edit tracker"):
        edited = st.data_editor(
            tracker,
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Result": st.column_config.SelectboxColumn("Result", options=["Open", "Win", "Loss", "Push"]),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
            key="nhl_bet_tracker_editor",
        )
        if st.button("Save tracker changes", width="stretch", key="nhl_save_tracker"):
            save_bet_tracker(recalculate_tracker_results(edited))
            st.success("NHL tracker saved.")


def render_betting_view() -> None:
    st.caption("NHL moneyline picks with embedded NHL feed odds when available.")
    place_tab, saved_tab, settled_tab = st.tabs(["Place Bets", "Saved Picks", "Settled Bets"])
    with place_tab:
        render_odds_board()
    with saved_tab:
        render_saved_betting_picks()
    with settled_tab:
        render_settled_betting_picks()
