# src/player_impact.py

from pathlib import Path
from time import sleep

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


DATA_DIR = Path("data")
PLAYER_IMPACT_PATH = DATA_DIR / "current_player_impact.csv"

SEASON = "2025-26"


def fetch_player_stats(season: str = SEASON) -> pd.DataFrame:
    """Fetch current regular-season NBA player stats."""
    totals_response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        timeout=60,
    )
    advanced_response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=60,
    )

    stats = totals_response.get_data_frames()[0]
    advanced_stats = advanced_response.get_data_frames()[0]

    if stats.empty:
        raise ValueError(f"No player stats returned for season {season}")

    if not advanced_stats.empty:
        merge_keys = [
            column
            for column in ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
            if column in stats.columns and column in advanced_stats.columns
        ]
        advanced_columns = [
            column
            for column in [
                "PLAYER_ID",
                "PLAYER_NAME",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "USG_PCT",
                "TS_PCT",
                "EFG_PCT",
                "PIE",
                "AST_PCT",
                "REB_PCT",
                "OREB_PCT",
                "DREB_PCT",
                "TOV_PCT",
                "OFF_RATING",
                "DEF_RATING",
                "NET_RATING",
            ]
            if column in advanced_stats.columns
        ]
        advanced_stats = advanced_stats[advanced_columns].copy()
        stats = stats.merge(
            advanced_stats,
            on=merge_keys,
            how="left",
            suffixes=("", "_ADV"),
        )

    stats["SEASON"] = season
    return stats


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two pandas Series."""
    return numerator.div(denominator.replace(0, pd.NA)).fillna(0)


def calculate_player_impact(stats: pd.DataFrame) -> pd.DataFrame:
    """Calculate automatic injury impact scores from player production."""
    required_columns = [
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PLUS_MINUS",
    ]
    optional_columns = [
        "USG_PCT",
        "TS_PCT",
        "EFG_PCT",
        "PIE",
        "AST_PCT",
        "REB_PCT",
        "OREB_PCT",
        "DREB_PCT",
        "TOV_PCT",
        "MINUS_PLUS",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
    ]

    missing_columns = [
        column for column in required_columns if column not in stats.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required player stat columns: {missing_columns}")

    selected_columns = required_columns + [
        column for column in optional_columns if column in stats.columns
    ] + ["SEASON"]
    player_stats = stats[selected_columns].copy()

    player_stats["MPG"] = safe_divide(player_stats["MIN"], player_stats["GP"])
    player_stats["PPG"] = safe_divide(player_stats["PTS"], player_stats["GP"])
    player_stats["RPG"] = safe_divide(player_stats["REB"], player_stats["GP"])
    player_stats["APG"] = safe_divide(player_stats["AST"], player_stats["GP"])
    player_stats["SPG"] = safe_divide(player_stats["STL"], player_stats["GP"])
    player_stats["BPG"] = safe_divide(player_stats["BLK"], player_stats["GP"])
    player_stats["TOPG"] = safe_divide(player_stats["TOV"], player_stats["GP"])
    player_stats["PLUS_MINUS_PER_GAME"] = safe_divide(
        player_stats["PLUS_MINUS"],
        player_stats["GP"],
    )

    if "USG_PCT" in player_stats.columns:
        player_stats["USG_PCT"] = pd.to_numeric(
            player_stats["USG_PCT"],
            errors="coerce",
        ).fillna(0.0)
    if "TS_PCT" in player_stats.columns:
        player_stats["TS_PCT"] = pd.to_numeric(
            player_stats["TS_PCT"],
            errors="coerce",
        ).fillna(0.0)
    if "EFG_PCT" in player_stats.columns:
        player_stats["EFG_PCT"] = pd.to_numeric(
            player_stats["EFG_PCT"],
            errors="coerce",
        ).fillna(0.0)
    if "PIE" in player_stats.columns:
        player_stats["PIE"] = pd.to_numeric(player_stats["PIE"], errors="coerce").fillna(0.0)
    if "AST_PCT" in player_stats.columns:
        player_stats["AST_PCT"] = pd.to_numeric(
            player_stats["AST_PCT"],
            errors="coerce",
        ).fillna(0.0)
    if "REB_PCT" in player_stats.columns:
        player_stats["REB_PCT"] = pd.to_numeric(
            player_stats["REB_PCT"],
            errors="coerce",
        ).fillna(0.0)
    if "TOV_PCT" in player_stats.columns:
        player_stats["TOV_PCT"] = pd.to_numeric(
            player_stats["TOV_PCT"],
            errors="coerce",
        ).fillna(0.0)

    # Emphasize true star usage and two-way impact, while still rewarding high volume.
    player_stats["SCORING_LOAD"] = (player_stats["PPG"] / 26).clip(0, 1.5)
    player_stats["CREATION_LOAD"] = (player_stats["APG"] / 8).clip(0, 1.2)
    player_stats["WORKLOAD_LOAD"] = (player_stats["MPG"] / 34).clip(0, 1.2)
    player_stats["REBOUND_LOAD"] = (player_stats["RPG"] / 10).clip(0, 0.9)
    player_stats["IMPACT_PLUS_MINUS"] = (
        player_stats["PLUS_MINUS_PER_GAME"] / 8
    ).clip(-0.8, 0.8)
    player_stats["BALL_SECURITY"] = (1.0 - (player_stats["TOPG"] / 4)).clip(0, 1.0)

    if "USG_PCT" in player_stats.columns:
        player_stats["USAGE_LOAD"] = (player_stats["USG_PCT"] / 0.28).clip(0, 1.3)
    else:
        player_stats["USAGE_LOAD"] = 1.0

    if "TS_PCT" in player_stats.columns:
        player_stats["EFFICIENCY_LOAD"] = (player_stats["TS_PCT"] / 0.60).clip(0, 1.25)
    elif "EFG_PCT" in player_stats.columns:
        player_stats["EFFICIENCY_LOAD"] = (player_stats["EFG_PCT"] / 0.56).clip(0, 1.25)
    else:
        player_stats["EFFICIENCY_LOAD"] = 1.0

    if "PIE" in player_stats.columns:
        player_stats["PIE_LOAD"] = (player_stats["PIE"] / 0.15).clip(0, 1.3)
    else:
        player_stats["PIE_LOAD"] = 1.0

    star_bonus = pd.Series(0, index=player_stats.index, dtype=float)
    star_bonus = star_bonus + (player_stats["PPG"] >= 20).astype(float)
    star_bonus = star_bonus + (player_stats["PPG"] >= 25).astype(float)
    star_bonus = star_bonus + (player_stats["MPG"] >= 30).astype(float)
    star_bonus = star_bonus + (player_stats["APG"] >= 5).astype(float)
    if "USG_PCT" in player_stats.columns:
        star_bonus = star_bonus + (player_stats["USG_PCT"] >= 0.24).astype(float)
        star_bonus = star_bonus + (player_stats["USG_PCT"] >= 0.28).astype(float)

    player_stats["STAR_BONUS"] = star_bonus.astype(int)

    scoring_pct = player_stats["PPG"].rank(pct=True, method="average")
    usage_pct = player_stats["USAGE_LOAD"].rank(pct=True, method="average")
    efficiency_pct = player_stats["EFFICIENCY_LOAD"].rank(pct=True, method="average")
    pie_pct = player_stats["PIE_LOAD"].rank(pct=True, method="average")
    creation_pct = player_stats["APG"].rank(pct=True, method="average")
    workload_pct = player_stats["MPG"].rank(pct=True, method="average")
    rebound_pct = player_stats["RPG"].rank(pct=True, method="average")
    plus_minus_pct = player_stats["PLUS_MINUS_PER_GAME"].rank(pct=True, method="average")
    ball_security_pct = player_stats["BALL_SECURITY"].rank(pct=True, method="average")
    stock_pct = ((player_stats["SPG"] + player_stats["BPG"]) / 4).rank(
        pct=True,
        method="average",
    )

    net_value = pd.Series(0.5, index=player_stats.index, dtype=float)
    if "NET_RATING" in player_stats.columns:
        net_value = pd.to_numeric(player_stats["NET_RATING"], errors="coerce").fillna(0.0)
    net_pct = net_value.rank(pct=True, method="average")

    defense_value = pd.Series(100.0, index=player_stats.index, dtype=float)
    if "DEF_RATING" in player_stats.columns:
        defense_value = pd.to_numeric(player_stats["DEF_RATING"], errors="coerce").fillna(100.0)
    defense_pct = 1.0 - defense_value.rank(pct=True, method="average")

    raw_impact_score = (
        scoring_pct * 0.18
        + usage_pct * 0.10
        + efficiency_pct * 0.16
        + pie_pct * 0.14
        + creation_pct * 0.08
        + workload_pct * 0.06
        + rebound_pct * 0.08
        + plus_minus_pct * 0.04
        + ball_security_pct * 0.02
        + stock_pct * 0.04
        + net_pct * 0.07
        + defense_pct * 0.07
        + player_stats["STAR_BONUS"] / 25.0
    )

    raw_min = float(raw_impact_score.min())
    raw_max = float(raw_impact_score.max())
    raw_range = raw_max - raw_min
    if raw_range > 0:
        raw_component = (raw_impact_score - raw_min) / raw_range
    else:
        raw_component = pd.Series(0.5, index=player_stats.index)

    league_component = raw_impact_score.rank(pct=True, method="average")
    player_stats["IMPACT_SCORE"] = (
        (league_component * 0.7 + raw_component * 0.3) * 9.9
    ).clip(lower=0, upper=9.9)
    player_stats["IMPACT_SCORE"] = player_stats["IMPACT_SCORE"].round(2)

    player_stats["INJURY_TIER"] = pd.cut(
        player_stats["IMPACT_SCORE"],
        bins=[-0.1, 2.5, 4.8, 6.8, 8.8, 10.0],
        labels=[
            "Low rotation impact",
            "Rotation player",
            "Important player",
            "Star-level impact",
            "Superstar-level impact",
        ],
    )

    output_columns = [
        "SEASON",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GP",
        "MPG",
        "PPG",
        "RPG",
        "APG",
        "PLUS_MINUS_PER_GAME",
        "STAR_BONUS",
        "IMPACT_SCORE",
        "INJURY_TIER",
    ]

    return player_stats[output_columns].sort_values(
        ["TEAM_ABBREVIATION", "IMPACT_SCORE"],
        ascending=[True, False],
    )


def build_player_impact_file() -> pd.DataFrame:
    """Build player impact CSV used by the Streamlit app."""
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Fetching player stats for {SEASON}...")
    stats = fetch_player_stats(SEASON)
    sleep(1)

    player_impact = calculate_player_impact(stats)
    player_impact.to_csv(PLAYER_IMPACT_PATH, index=False)

    print(f"Saved player impact ratings to {PLAYER_IMPACT_PATH}")
    print()
    print(player_impact.head(25).to_string(index=False))

    return player_impact


if __name__ == "__main__":
    build_player_impact_file()
