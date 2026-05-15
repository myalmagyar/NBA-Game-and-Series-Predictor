# NBA Prediction Model

Streamlit app for NBA single-game, series, and bracket predictions.

## Free Data Refresh Commands

```bash
python src/injuries.py
python src/player_impact.py
python src/player_strength.py
python src/features.py
python src/team_strength.py
python src/train_model.py
```

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

## Accuracy Features

The model now includes:

- rest and schedule features
- road-trip game number
- previous-season player-strength features
- optional historical injury features
- playoff context features such as series game number, series score, elimination flags, and closeout flags
- advanced efficiency and Four Factors features
- current playoff series tracker with rest-of-series simulations
- probability calibration table
- key model-edge explanations
- rolling season backtests
- trained model/Elo blend settings

Current official injuries are still applied live in the app when the model has
not been trained on a full historical injury backfill.

The historical injury backfill command is available, but the free official NBA
pages may expose only limited historical rows. If `data/historical_injuries.csv`
does not cover the historical games being trained, the model keeps injury
features neutral and the app applies the latest official report live.
