"""
Backtesting module for CourtVision.
Runs the trained models against historical match data and reports accuracy.
"""
import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def run_backtest(df_feats, models_dir='models', test_years=None):
    """
    Run backtest on the feature-engineered dataset.

    Args:
        df_feats: DataFrame from prepare_features_for_training (already has target column)
        models_dir: Path to trained models
        test_years: List of years to test on (e.g. [2022, 2023, 2024]).
                    If None, uses the last 20% of data (temporal split).

    Returns:
        dict with backtest results
    """
    # Load models and feature names
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    if not os.path.exists(feature_names_path):
        print("[ERR] No trained models found. Run model_trainer.py first.")
        return None

    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)

    # Load models
    models = {}
    for name, fname in [('logistic_regression', 'logistic_regression.pkl'),
                         ('xgboost', 'xgboost.pkl')]:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    nn_path = os.path.join(models_dir, 'neural_network.h5')
    nn_scaler_path = os.path.join(models_dir, 'neural_network_scaler.pkl')
    nn_model = None
    nn_scaler = None
    if os.path.exists(nn_path) and os.path.exists(nn_scaler_path):
        try:
            import tensorflow as tf
            nn_model = tf.keras.models.load_model(nn_path)
            with open(nn_scaler_path, 'rb') as f:
                nn_scaler = pickle.load(f)
            models['neural_network'] = nn_model
        except Exception as e:
            print(f"  [WARN] Could not load neural network: {e}")

    if not models:
        print("[ERR] No models could be loaded.")
        return None

    # Select test set
    df_feats = df_feats.copy()
    df_feats['tourney_date'] = pd.to_datetime(df_feats['tourney_date'], errors='coerce')
    df_feats['year'] = df_feats['tourney_date'].dt.year

    if test_years:
        test_df = df_feats[df_feats['year'].isin(test_years)]
    else:
        split_idx = int(len(df_feats) * 0.8)
        test_df = df_feats.iloc[split_idx:]

    if len(test_df) == 0:
        print("[ERR] No test data found for the specified years.")
        return None

    print(f"\n  Backtesting on {len(test_df):,} matches "
          f"({test_df['tourney_date'].min().year}–{test_df['tourney_date'].max().year})")

    # Build feature matrix
    available = [f for f in feature_names if f in test_df.columns]
    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        print(f"  [WARN] Missing features (will be 0): {missing}")
        for f in missing:
            test_df[f] = 0.0

    X_test = test_df[feature_names].values.astype(float)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = test_df['target'].values

    # Predict with each model
    all_probs = []
    model_results = {}

    for name, model in models.items():
        try:
            if name == 'neural_network':
                X_scaled = nn_scaler.transform(X_test)
                probs = model.predict(X_scaled, verbose=0).flatten()
            else:
                probs = model.predict_proba(X_test)[:, 1]

            preds = (probs >= 0.5).astype(int)
            acc = (preds == y_test).mean()
            model_results[name] = {'accuracy': acc, 'probs': probs}
            all_probs.append(probs)
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")

    if not all_probs:
        print("[ERR] No models produced predictions.")
        return None

    # Consensus
    consensus_probs = np.mean(all_probs, axis=0)
    consensus_preds = (consensus_probs >= 0.5).astype(int)
    consensus_acc = (consensus_preds == y_test).mean()

    # --- Overall results ---
    print("\n" + "="*60)
    print("  BACKTEST RESULTS")
    print("="*60)
    print(f"\n  {'Model':<25} {'Accuracy':>10}")
    print("  " + "-"*37)
    for name, res in model_results.items():
        print(f"  {name.replace('_',' ').title():<25} {res['accuracy']:>9.1%}")
    print("  " + "-"*37)
    print(f"  {'Consensus':<25} {consensus_acc:>9.1%}")

    # --- By year ---
    print("\n  BY YEAR")
    print(f"  {'Year':<8} {'Matches':>8} {'Consensus Acc':>14}")
    print("  " + "-"*33)
    test_df = test_df.copy()
    test_df['consensus_correct'] = (consensus_preds == y_test)
    for year in sorted(test_df['year'].dropna().unique()):
        yr_mask = test_df['year'] == year
        yr_acc = test_df.loc[yr_mask, 'consensus_correct'].mean()
        yr_n = yr_mask.sum()
        print(f"  {int(year):<8} {yr_n:>8,} {yr_acc:>13.1%}")

    # --- By surface ---
    print("\n  BY SURFACE")
    print(f"  {'Surface':<10} {'Matches':>8} {'Consensus Acc':>14}")
    print("  " + "-"*35)
    if 'surface' in test_df.columns:
        for surf in ['Hard', 'Clay', 'Grass']:
            mask = test_df['surface'] == surf
            if mask.sum() > 0:
                s_acc = test_df.loc[mask, 'consensus_correct'].mean()
                print(f"  {surf:<10} {mask.sum():>8,} {s_acc:>13.1%}")

    # --- By tournament level ---
    print("\n  BY TOURNAMENT LEVEL")
    print(f"  {'Level':<12} {'Matches':>8} {'Consensus Acc':>14}")
    print("  " + "-"*37)
    level_labels = {'G': 'Grand Slam', 'M': 'Masters', 'A': 'ATP 250/500',
                    'F': 'Finals', 'D': 'Davis Cup', 'C': 'Challenger'}
    if 'tourney_level' in test_df.columns:
        for lvl, label in level_labels.items():
            mask = test_df['tourney_level'] == lvl
            if mask.sum() > 0:
                l_acc = test_df.loc[mask, 'consensus_correct'].mean()
                print(f"  {label:<12} {mask.sum():>8,} {l_acc:>13.1%}")

    # --- Calibration: model confidence vs actual accuracy ---
    print("\n  CALIBRATION (Confidence vs Actual Win Rate)")
    print(f"  {'Confidence':<15} {'Matches':>8} {'Actual Win Rate':>16}")
    print("  " + "-"*42)
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        # Consensus prob for the predicted winner
        win_probs = np.where(consensus_probs >= 0.5, consensus_probs, 1 - consensus_probs)
        mask = (win_probs >= lo) & (win_probs < hi)
        if mask.sum() > 0:
            actual_rate = consensus_preds[mask] == y_test[mask]
            print(f"  {lo:.0%}–{hi:.0%}          {mask.sum():>8,} {actual_rate.mean():>15.1%}")

    print("="*60)

    return {
        'model_results': model_results,
        'consensus_accuracy': consensus_acc,
        'n_matches': len(test_df),
    }


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import prepare_features_for_training

    print("Loading data...")
    df = load_data()
    _, _, feature_names, encoders, df_feats = prepare_features_for_training(df)
    run_backtest(df_feats, test_years=[2022, 2023, 2024])
