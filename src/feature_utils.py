# src/feature_utils.py

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
CURRENT_PLAYER_IMPACT_PATH = DATA_DIR / "current_player_impact.csv"
HISTORICAL_INJURIES_PATH = DATA_DIR / "historical_injuries.csv"
TEAM_PLAYER_STRENGTH_BY_SEASON_PATH = DATA_DIR / "team_player_strength_by_season.csv"

SCHEDULE_FEATURE_COLUMNS = [
    "DAYS_REST",
    "IS_BACK_TO_BACK",
    "IS_THIRD_IN_FOUR_DAYS",
    "GAMES_LAST_7_DAYS",
    "ROAD_TRIP_GAME_NUMBER",
]

ADVANCED_STAT_COLUMNS = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "PACE",
    "EFG_PCT",
    "TOV_RATE",
    "OREB_PCT",
    "FT_RATE",
]

PLAYOFF_CONTEXT_FEATURE_COLUMNS = [
    "IS_PLAYOFF_GAME",
    "PLAYOFF_SERIES_GAME_NUMBER",
    "DIFF_SERIES_WINS_ENTERING",
    "HOME_SERIES_WINS_ENTERING",
    "AWAY_SERIES_WINS_ENTERING",
    "HOME_FACING_ELIMINATION",
    "AWAY_FACING_ELIMINATION",
    "HOME_CAN_CLINCH_SERIES",
    "AWAY_CAN_CLINCH_SERIES",
]

INJURY_FEATURE_COLUMNS = [
    "INJURY_WEIGHTED_IMPACT",
    "OUT_PLAYER_COUNT",
    "QUESTIONABLE_PLAYER_COUNT",
]

PLAYER_STRENGTH_FEATURE_COLUMNS = [
    "PREV_SEASON_PLAYER_TOP_5",
    "PREV_SEASON_PLAYER_TOP_8",
    "PREV_SEASON_PLAYER_DEPTH",
    "PREV_SEASON_STAR_COUNT",
]

CURRENT_PLAYER_STRENGTH_COLUMNS = [
    "PLAYER_TOP_5",
    "PLAYER_TOP_8",
    "PLAYER_DEPTH",
    "STAR_COUNT",
]

STATUS_WEIGHTS = {
    "Out": 1.00,
    "Doubtful": 0.75,
    "Questionable": 0.50,
    "Probable": 0.10,
    "Available": 0.00,
}


def safe_divide(numerator, denominator):
    """Divide and replace invalid results with zero."""
    return pd.Series(numerator).div(pd.Series(denominator).replace(0, pd.NA)).fillna(0)


def cap_rest_days(days: float | int | None) -> float:
    """Cap rest days so long breaks do not dominate the model."""
    if days is None or pd.isna(days):
        return 7.0

    return float(min(max(days, 0), 7))


def previous_season(season: str) -> str | None:
    """Return the previous NBA season string."""
    try:
        start_year = int(str(season).split("-")[0])
    except (TypeError, ValueError):
        return None

    if start_year <= 1900:
        return None

    previous_start = start_year - 1
    return f"{previous_start}-{str(start_year)[-2:]}"


def add_advanced_game_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Add box-score efficiency and Four Factors columns to team-game rows."""
    games = games.copy()
    games["GAME_ID"] = games["GAME_ID"].astype(str)

    opponent_columns = [
        "GAME_ID",
        "TEAM_ID",
        "DREB",
        "FGA",
        "FTA",
        "OREB",
        "TOV",
        "PTS",
    ]
    opponent = games[opponent_columns].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "DREB": "OPP_DREB",
            "FGA": "OPP_FGA",
            "FTA": "OPP_FTA",
            "OREB": "OPP_OREB",
            "TOV": "OPP_TOV",
            "PTS": "OPP_PTS",
        }
    )

    games = games.merge(opponent, on="GAME_ID", how="left")
    games = games[games["TEAM_ID"] != games["OPP_TEAM_ID"]].copy()

    possessions = games["FGA"] + (0.44 * games["FTA"]) - games["OREB"] + games["TOV"]
    opponent_possessions = (
        games["OPP_FGA"]
        + (0.44 * games["OPP_FTA"])
        - games["OPP_OREB"]
        + games["OPP_TOV"]
    )

    games["OFF_RATING"] = safe_divide(100 * games["PTS"], possessions)
    games["DEF_RATING"] = safe_divide(100 * games["OPP_PTS"], opponent_possessions)
    games["NET_RATING"] = games["OFF_RATING"] - games["DEF_RATING"]
    games["PACE"] = (possessions + opponent_possessions) / 2
    games["EFG_PCT"] = safe_divide(games["FGM"] + (0.5 * games["FG3M"]), games["FGA"])
    games["TOV_RATE"] = safe_divide(games["TOV"], possessions)
    games["OREB_PCT"] = safe_divide(games["OREB"], games["OREB"] + games["OPP_DREB"])
    games["FT_RATE"] = safe_divide(games["FTA"], games["FGA"])

    return games.drop(
        columns=[
            "OPP_TEAM_ID",
            "OPP_DREB",
            "OPP_FGA",
            "OPP_FTA",
            "OPP_OREB",
            "OPP_TOV",
            "OPP_PTS",
        ]
    )


def add_pregame_schedule_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add schedule features that are known before each game."""
    games = games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.").astype(int)
    games["IS_ROAD"] = 1 - games["IS_HOME"]

    previous_dates = games.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    day_gaps = (games["GAME_DATE"] - previous_dates).dt.days
    games["DAYS_REST"] = (day_gaps - 1).map(cap_rest_days)
    games["IS_BACK_TO_BACK"] = day_gaps.eq(1).fillna(False).astype(int)

    games["IS_THIRD_IN_FOUR_DAYS"] = 0
    games["GAMES_LAST_7_DAYS"] = 0
    games["ROAD_TRIP_GAME_NUMBER"] = 0

    for _, team_indices in games.groupby("TEAM_ID", sort=False).groups.items():
        indices = list(team_indices)
        team_dates = games.loc[indices, "GAME_DATE"].tolist()
        road_streak = 0

        for position, index in enumerate(indices):
            current_date = team_dates[position]
            previous_team_dates = team_dates[:position]

            games_last_4_days = sum(
                0 < (current_date - previous_date).days <= 3
                for previous_date in previous_team_dates
            )
            games_last_7_days = sum(
                0 < (current_date - previous_date).days <= 6
                for previous_date in previous_team_dates
            )

            games.at[index, "IS_THIRD_IN_FOUR_DAYS"] = int(games_last_4_days >= 2)
            games.at[index, "GAMES_LAST_7_DAYS"] = games_last_7_days

            if int(games.at[index, "IS_ROAD"]) == 1:
                road_streak += 1
                games.at[index, "ROAD_TRIP_GAME_NUMBER"] = road_streak
            else:
                road_streak = 0

    return games


def add_playoff_context_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """Add pregame playoff series context without leaking future games."""
    matchups = matchups.sort_values(["GAME_DATE", "GAME_ID"]).copy()

    for column in PLAYOFF_CONTEXT_FEATURE_COLUMNS:
        matchups[column] = 0

    if "SEASON_TYPE" not in matchups.columns:
        return matchups

    matchups["IS_PLAYOFF_GAME"] = matchups["SEASON_TYPE"].eq("Playoffs").astype(int)
    playoff_mask = matchups["IS_PLAYOFF_GAME"].eq(1)

    if not playoff_mask.any():
        return matchups

    playoff_rows = matchups[playoff_mask].copy()
    playoff_rows["SERIES_KEY"] = playoff_rows.apply(
        lambda row: tuple(sorted([int(row["HOME_TEAM_ID"]), int(row["AWAY_TEAM_ID"])])),
        axis=1,
    )

    for _, group in playoff_rows.groupby(["SEASON", "SERIES_KEY"], sort=False):
        series_wins: dict[int, int] = {}

        for game_number, (index, row) in enumerate(group.iterrows(), start=1):
            home_team_id = int(row["HOME_TEAM_ID"])
            away_team_id = int(row["AWAY_TEAM_ID"])
            home_wins = series_wins.get(home_team_id, 0)
            away_wins = series_wins.get(away_team_id, 0)

            matchups.at[index, "PLAYOFF_SERIES_GAME_NUMBER"] = game_number
            matchups.at[index, "HOME_SERIES_WINS_ENTERING"] = home_wins
            matchups.at[index, "AWAY_SERIES_WINS_ENTERING"] = away_wins
            matchups.at[index, "DIFF_SERIES_WINS_ENTERING"] = home_wins - away_wins
            matchups.at[index, "HOME_FACING_ELIMINATION"] = int(away_wins == 3)
            matchups.at[index, "AWAY_FACING_ELIMINATION"] = int(home_wins == 3)
            matchups.at[index, "HOME_CAN_CLINCH_SERIES"] = int(home_wins == 3)
            matchups.at[index, "AWAY_CAN_CLINCH_SERIES"] = int(away_wins == 3)

            if int(row["HOME_WIN"]) == 1:
                series_wins[home_team_id] = home_wins + 1
            else:
                series_wins[away_team_id] = away_wins + 1

    return matchups


def calculate_current_schedule_state(
    games: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate current rest and travel state for every team."""
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()
    else:
        as_of_date = pd.to_datetime(as_of_date).normalize()

    games = add_pregame_schedule_features(games)
    games = games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()

    rows = []

    for team_id, team_games in games.groupby("TEAM_ID", sort=False):
        past_games = team_games[team_games["GAME_DATE"] <= as_of_date]

        if past_games.empty:
            continue

        latest = past_games.iloc[-1]
        last_game_date = pd.to_datetime(latest["GAME_DATE"])
        day_gap = (as_of_date - last_game_date).days

        recent_dates = pd.to_datetime(past_games["GAME_DATE"])
        games_last_4_days = int(
            ((as_of_date - recent_dates).dt.days.between(1, 3)).sum()
        )
        games_last_7_days = int(
            ((as_of_date - recent_dates).dt.days.between(1, 6)).sum()
        )

        trailing_road_streak = 0

        for _, game in past_games.iloc[::-1].iterrows():
            if int(game["IS_ROAD"]) != 1:
                break

            trailing_road_streak += 1

        rows.append(
            {
                "TEAM_ID": team_id,
                "DAYS_REST": cap_rest_days(day_gap - 1),
                "IS_BACK_TO_BACK": int(day_gap == 1),
                "IS_THIRD_IN_FOUR_DAYS": int(games_last_4_days >= 2),
                "GAMES_LAST_7_DAYS": games_last_7_days,
                "CURRENT_ROAD_STREAK": trailing_road_streak,
            }
        )

    return pd.DataFrame(rows)


def load_team_player_strength_by_season() -> pd.DataFrame:
    """Load previous-season player-strength data if it exists."""
    if not TEAM_PLAYER_STRENGTH_BY_SEASON_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(TEAM_PLAYER_STRENGTH_BY_SEASON_PATH)


def add_previous_season_player_strength(matchups: pd.DataFrame) -> pd.DataFrame:
    """Join previous-season team player-strength features onto matchup rows."""
    matchups = matchups.copy()
    strength = load_team_player_strength_by_season()

    if strength.empty:
        for prefix in ["HOME", "AWAY"]:
            for column in PLAYER_STRENGTH_FEATURE_COLUMNS:
                matchups[f"{prefix}_{column}"] = 0.0
                matchups[f"DIFF_{column}"] = 0.0

        return matchups

    strength = strength.rename(
        columns={
            "SEASON": "PLAYER_STRENGTH_SEASON",
            "PLAYER_TOP_5": "PREV_SEASON_PLAYER_TOP_5",
            "PLAYER_TOP_8": "PREV_SEASON_PLAYER_TOP_8",
            "PLAYER_DEPTH": "PREV_SEASON_PLAYER_DEPTH",
            "STAR_COUNT": "PREV_SEASON_STAR_COUNT",
        }
    )
    matchups["PLAYER_STRENGTH_SEASON"] = matchups["SEASON"].map(previous_season)

    home_strength = strength[
        ["PLAYER_STRENGTH_SEASON", "TEAM_ID", *PLAYER_STRENGTH_FEATURE_COLUMNS]
    ].rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            **{
                column: f"HOME_{column}"
                for column in PLAYER_STRENGTH_FEATURE_COLUMNS
            },
        }
    )
    away_strength = strength[
        ["PLAYER_STRENGTH_SEASON", "TEAM_ID", *PLAYER_STRENGTH_FEATURE_COLUMNS]
    ].rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            **{
                column: f"AWAY_{column}"
                for column in PLAYER_STRENGTH_FEATURE_COLUMNS
            },
        }
    )

    matchups = matchups.merge(
        home_strength,
        on=["PLAYER_STRENGTH_SEASON", "HOME_TEAM_ID"],
        how="left",
    )
    matchups = matchups.merge(
        away_strength,
        on=["PLAYER_STRENGTH_SEASON", "AWAY_TEAM_ID"],
        how="left",
    )

    for column in PLAYER_STRENGTH_FEATURE_COLUMNS:
        home_column = f"HOME_{column}"
        away_column = f"AWAY_{column}"
        matchups[home_column] = matchups[home_column].fillna(0.0)
        matchups[away_column] = matchups[away_column].fillna(0.0)
        matchups[f"DIFF_{column}"] = matchups[home_column] - matchups[away_column]

    return matchups.drop(columns=["PLAYER_STRENGTH_SEASON"])


def aggregate_player_strength(player_impact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player impact rows into team strength features."""
    rows = []

    for (team_id, abbreviation), players in player_impact.groupby(
        ["TEAM_ID", "TEAM_ABBREVIATION"],
    ):
        sorted_players = players.sort_values("IMPACT_SCORE", ascending=False)
        top_5 = float(sorted_players.head(5)["IMPACT_SCORE"].sum())
        top_8 = float(sorted_players.head(8)["IMPACT_SCORE"].sum())
        depth = float(sorted_players.iloc[5:10]["IMPACT_SCORE"].sum())
        star_count = int((sorted_players["IMPACT_SCORE"] >= 6.5).sum())

        rows.append(
            {
                "TEAM_ID": int(team_id),
                "TEAM_ABBREVIATION": abbreviation,
                "PLAYER_TOP_5": top_5,
                "PLAYER_TOP_8": top_8,
                "PLAYER_DEPTH": depth,
                "STAR_COUNT": star_count,
            }
        )

    return pd.DataFrame(rows)


def load_current_player_strength() -> pd.DataFrame:
    """Load current player-impact data and aggregate it by team."""
    if not CURRENT_PLAYER_IMPACT_PATH.exists():
        return pd.DataFrame()

    player_impact = pd.read_csv(CURRENT_PLAYER_IMPACT_PATH)
    return aggregate_player_strength(player_impact)


def load_historical_injury_features() -> pd.DataFrame:
    """Load optional historical injury features."""
    if not HISTORICAL_INJURIES_PATH.exists():
        return pd.DataFrame()

    injuries = pd.read_csv(HISTORICAL_INJURIES_PATH)

    if injuries.empty:
        return pd.DataFrame()

    injuries["GAME_DATE"] = pd.to_datetime(injuries["GAME_DATE"])

    if "STATUS_WEIGHT" not in injuries.columns:
        injuries["STATUS_WEIGHT"] = injuries["CURRENT_STATUS"].map(STATUS_WEIGHTS).fillna(0)

    if "IMPACT_SCORE" not in injuries.columns:
        injuries["IMPACT_SCORE"] = 1.0

    injuries["WEIGHTED_IMPACT"] = injuries["STATUS_WEIGHT"] * injuries["IMPACT_SCORE"]
    injuries["OUT_FLAG"] = injuries["CURRENT_STATUS"].eq("Out").astype(int)
    injuries["QUESTIONABLE_FLAG"] = injuries["CURRENT_STATUS"].eq("Questionable").astype(int)

    return (
        injuries.groupby(["GAME_DATE", "TEAM"], as_index=False)
        .agg(
            INJURY_WEIGHTED_IMPACT=("WEIGHTED_IMPACT", "sum"),
            OUT_PLAYER_COUNT=("OUT_FLAG", "sum"),
            QUESTIONABLE_PLAYER_COUNT=("QUESTIONABLE_FLAG", "sum"),
        )
    )


def add_historical_injury_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """Join optional historical injury features onto matchup rows."""
    matchups = matchups.copy()
    injury_features = load_historical_injury_features()

    if injury_features.empty:
        for prefix in ["HOME", "AWAY"]:
            for column in INJURY_FEATURE_COLUMNS:
                matchups[f"{prefix}_{column}"] = 0.0
                matchups[f"DIFF_{column}"] = 0.0

        return matchups

    home_injuries = injury_features.rename(
        columns={
            "TEAM": "HOME_TEAM",
            **{column: f"HOME_{column}" for column in INJURY_FEATURE_COLUMNS},
        }
    )
    away_injuries = injury_features.rename(
        columns={
            "TEAM": "AWAY_TEAM",
            **{column: f"AWAY_{column}" for column in INJURY_FEATURE_COLUMNS},
        }
    )

    matchups = matchups.merge(
        home_injuries,
        on=["GAME_DATE", "HOME_TEAM"],
        how="left",
    )
    matchups = matchups.merge(
        away_injuries,
        on=["GAME_DATE", "AWAY_TEAM"],
        how="left",
    )

    for column in INJURY_FEATURE_COLUMNS:
        home_column = f"HOME_{column}"
        away_column = f"AWAY_{column}"
        matchups[home_column] = matchups[home_column].fillna(0.0)
        matchups[away_column] = matchups[away_column].fillna(0.0)
        matchups[f"DIFF_{column}"] = matchups[home_column] - matchups[away_column]

    return matchups
