# Sports Betting Dashboard

Unified Streamlit app for NBA and MLB betting research, odds comparison, saved
picks, futures, and bankroll views.

Run the combined app:

```bash
streamlit run app.py
```

The unified app keeps separate sport pages so more leagues can be added later.
The original sport-specific dashboards are also available as standalone entry
points:

```bash
streamlit run app_nba.py
streamlit run app_mlb.py
```

## Unified App Features

- Betting Hub with NBA/MLB overview cards
- Daily sports brief with best-bet, closest-game, upset, watchability, and pick-movement panels
- Unified Game Center for current slate games or manual matchups
- What-If controls for NBA availability/rest swings and MLB starter/weather/bullpen swings
- Odds board comparing win chance, book probability, fair odds, edge, EV/unit, and Kelly
- Upset Radar and live upset-alert checks
- Futures board with fair odds
- Combined bet tracker across NBA and MLB picks
- Pick-history snapshots and movement feed
- Favorite-team tracking
- Copy-ready pick cards

## Data Refresh Commands

These commands rebuild the internal pricing files used by the betting screens.

```bash
python src/injuries.py
python src/player_impact.py
python src/player_strength.py
python src/features.py
python src/team_strength.py
python src/train_model.py
```

## MLB App

Run the standalone MLB app separately:

```bash
streamlit run app_mlb.py
```

Build or refresh the MLB pricing files:

```bash
python src/mlb_collect_data.py
python src/mlb_features.py
python src/mlb_team_strength.py
python src/mlb_train_model.py
```

The MLB version currently uses the public MLB Stats API, team form, rest, run
differential, starter context, and power-rating inputs.

`src/injuries.py` fetches the latest official NBA injury-report PDF for free and
saves `data/current_injuries.csv`.

For a historical injury-report backfill, run:

```bash
python src/injuries.py --historical
```

That can download many official PDFs. To test it on only the latest report for a
season:

```bash
python src/injuries.py --historical --season 2025-26 --max-reports-per-season 1
```

## Betting Workflow Features

The app now includes:

- rest and schedule features
- road-trip game number
- previous-season player-strength features
- optional historical injury features
- playoff context features such as series game number, series score, elimination flags, and closeout flags
- advanced efficiency and Four Factors features
- current playoff series tracker with rest-of-series simulations
- win-chance, book-chance, edge, fair-odds, EV/unit, and Kelly views
- key betting-edge explanations
- saved-pick tracking with active and settled bet views
- pick-history snapshots and movement tracking

Current official injuries are still applied live in the NBA screens when a full
historical injury backfill is not available.

The historical injury backfill command is available, but the free official NBA
pages may expose only limited historical rows. If `data/historical_injuries.csv`
does not cover older games, injury inputs stay neutral and the app applies the
latest official report live.
