# CourtVision

ML-powered tennis match prediction and tournament simulation, built entirely for the terminal.

## What it does

CourtVision uses historical ATP match data (1968-2026) to predict match outcomes and simulate full tournament brackets. It runs Monte Carlo simulations to produce statistically robust predictions with realistic point-by-point simulated scores.

### Features

- **Predict H2H Winner** - Predict head-to-head matchups between any two ATP players using an ensemble of 3 ML models + a standalone ELO predictor
- **Tournament Simulation** - Two modes:
  - **Custom** - Build your own bracket with any players and seeds
  - **Live/Upcoming ATP** - Automatically fetches current or upcoming tournaments from the ATP calendar, scrapes the real draw from Wikipedia, and simulates the full bracket
- **Monte Carlo Engine** - Runs 1000+ probabilistic simulations per tournament, scores each outcome against aggregate probabilities, and displays the most statistically representative result
- **Point-by-Point Score Simulation** - Scores are generated using each player's career serve probabilities and surface-specific return pressure, not hand-coded heuristics
- **Historical Era Matching** - Predict hypothetical matchups from any year (e.g. 2006 Federer vs 2006 Nadal)
- **Model Performance** - View accuracy metrics, feature importance, and run a walk-forward backtest showing accuracy per year

## Models

Three models trained on 200K+ historical matches with a temporal train/test split to prevent data leakage, plus a standalone ELO system. All models are probability-calibrated using isotonic regression so predicted win percentages reflect real historical accuracy:

| Model | Type | ROC-AUC |
| --- | --- | --- |
| Logistic Regression | Baseline linear model | 0.788 |
| XGBoost | Gradient-boosted trees | 0.861 |
| Neural Network | 3-layer network (128-64-32) | 0.855 |
| ELO | Surface-weighted rating system | - |

Predictions are averaged across all models for the final consensus probability.

## Training Features (26)

Features are computed using symmetric A/B player assignment to prevent directional bias. All features are difference-based (Player A minus Player B):

**Ranking & Profile**
`rank_diff`, `pts_diff`, `age_diff`, `ht_surface_adv`, `seed_diff`

**Performance History**
`win_rate_diff`, `surface_wr_diff`, `form_diff`, `experience_diff`, `h2h_adv`

**Context**
`surface_enc`, `level_enc`, `round_enc`, `hand_cross`, `lefty_adv`

**ELO Ratings**
`elo_diff`, `surface_elo_diff`

**Career Serve Averages**
`ace_rate_diff`, `df_rate_diff`, `first_serve_pct_diff`, `first_serve_win_diff`, `second_serve_win_diff`, `bp_save_rate_diff`

**Fatigue**
`days_since_last_diff`, `matches_last_30d_diff`

**Surface Return Pressure**
`surface_return_diff`

## ELO System

Standalone ELO rating system with:
- Tournament-level K-factors (Grand Slams: 40, Masters: 32, ATP 500: 24, ATP 250: 16)
- Surface-specific ratings tracked independently from overall ratings
- JSON caching so ratings persist across sessions without recomputation
- Integrated into the model ensemble as an additional predictor

## Score Simulation

Scores are simulated point-by-point using each player's career serve stats:

```
P(win serve point) = (1st_serve_pct x 1st_serve_win_pct) + ((1 - 1st_serve_pct) x 2nd_serve_win_pct)
```

Each player's effective serve probability is then adjusted by their opponent's surface-specific return win rate, so a strong clay-court returner like Nadal will genuinely suppress opponents' serve effectiveness on clay.

## Project Structure

```
CourtVision/
    main.py                     # Terminal UI - match predict, tournament, model stats
    predictor.py                # Match prediction, score simulation, display
    model_trainer.py            # Training pipeline with calibration + walk-forward backtest
    feature_engineering.py      # Symmetric temporal feature computation
    elo_calculator.py           # ELO rating system with surface-specific ratings
    tournament_simulator.py     # Bracket simulation + Monte Carlo engine
    atp_api.py                  # ATP calendar API + Wikipedia draw scraper
    data_loader.py              # CSV loading with parquet cache for fast startup
    tests/                      # Test suite (pytest)
    models/                     # Saved model weights, scalers, encoders, ELO cache
    tennis_atp-master/          # Jeff Sackmann's ATP match dataset (1968-2026)
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Ball Don't Lie API key:
```
ATP_API_KEY=your_key_here
```

**Getting your API key:**
1. Go to [balldontlie.io](https://balldontlie.io) and create a free account
2. After signing in, go to your dashboard and copy your API key
3. Paste it into your `.env` file as shown above

> The API key is only needed for the **Live/Upcoming ATP Tournament** feature which fetches the current tournament calendar. All other features (H2H prediction, custom tournament simulation, model training) work without it.

Then train the models and run:
```bash
python model_trainer.py     # train models (required on first run)
python main.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Data

Uses [Jeff Sackmann's tennis_atp dataset](https://github.com/JeffSackmann/tennis_atp) containing ATP match results from 1968 to present.

## Requirements

- Python 3.10+
- TensorFlow, scikit-learn, XGBoost, pandas, numpy
- requests, beautifulsoup4 (for ATP API + Wikipedia scraping)
- python-dotenv (for API key management)
- pyarrow (for parquet caching)
