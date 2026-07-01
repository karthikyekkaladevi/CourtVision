"""
Feature engineering module for tennis tournament predictor.
REFACTORED: Symmetric Player A vs Player B approach to eliminate data leakage.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def handle_missing_values(df):
    """Handle missing values in the raw dataset."""
    df_processed = df.copy()
    
    # Fill missing ranks with high value (worse than any real rank)
    df_processed['winner_rank'] = df_processed['winner_rank'].fillna(999)
    df_processed['loser_rank'] = df_processed['loser_rank'].fillna(999)
    
    # Fill missing rank points with 0
    df_processed['winner_rank_points'] = df_processed['winner_rank_points'].fillna(0)
    df_processed['loser_rank_points'] = df_processed['loser_rank_points'].fillna(0)
    
    # Fill missing seeds with 99
    df_processed['winner_seed'] = df_processed['winner_seed'].fillna(99)
    df_processed['loser_seed'] = df_processed['loser_seed'].fillna(99)
    
    # Fill missing ages/heights with median
    df_processed['winner_age'] = df_processed['winner_age'].fillna(df_processed['winner_age'].median())
    df_processed['loser_age'] = df_processed['loser_age'].fillna(df_processed['loser_age'].median())
    df_processed['winner_ht'] = df_processed['winner_ht'].fillna(df_processed['winner_ht'].median())
    df_processed['loser_ht'] = df_processed['loser_ht'].fillna(df_processed['loser_ht'].median())
    
    return df_processed

def compute_symmetric_features(df):
    """
    Process matches in date order and compute features from the perspective of Player A and Player B.
    Player A and Player B are assigned randomly per row to ensure symmetry.
    """
    from elo_calculator import K_FACTORS, INITIAL_RATING, SURFACE_K_MULTIPLIER

    print("  Processing symmetric features (temporal sequential pass)...")

    # Sort by date
    df = df.copy()
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # Tracking dictionaries for historical stats
    history = {
        'wins': {},            # player -> total wins
        'matches': {},         # player -> total matches
        'surface_wins': {},    # (player, surface) -> wins
        'surface_matches': {}, # (player, surface) -> total matches
        'recent': {},          # player -> list of last 20 results (1=win, 0=loss)
        'h2h': {},             # (player1, player2) -> wins for player1
        # Serve stat cumulative accumulators (for career averages)
        'serve_ace': {},       # player -> cumulative aces
        'serve_df': {},        # player -> cumulative double faults
        'serve_svpt': {},      # player -> cumulative serve points
        'serve_1stIn': {},     # player -> cumulative 1st serves in
        'serve_1stWon': {},    # player -> cumulative 1st serve points won
        'serve_2ndWon': {},    # player -> cumulative 2nd serve points won
        'serve_bpSaved': {},   # player -> cumulative break points saved
        'serve_bpFaced': {},   # player -> cumulative break points faced
        # Fatigue tracking
        'last_match_date': {}, # player -> datetime of last match
        'match_dates': {},     # player -> list of recent match datetimes
        # Return pressure accumulators (derived from opponent's serve stats)
        'return_svpt': {},             # player -> cumulative return points faced
        'return_won': {},              # player -> cumulative return points won
        'surface_return_svpt': {},     # (player, surface) -> cumulative surface return pts faced
        'surface_return_won': {},      # (player, surface) -> cumulative surface return pts won
    }

    # ELO state — tracked inline to maintain temporal integrity (no lookahead)
    elo_ratings = {}          # player -> float
    surface_elo_ratings = {}  # (player, surface) -> float

    # Helper: compute career serve rate from cumulative accumulators
    def _serve_rate(player, num_key, denom_key, default=0.0):
        num = history[num_key].get(player, 0)
        denom = history[denom_key].get(player, 0)
        return num / denom if denom > 0 else default

    # Helper: count matches within a time window
    def _matches_in_window(player, current_date, window_days=30):
        dates = history['match_dates'].get(player, [])
        if not dates or pd.isna(current_date):
            return 0
        cutoff = current_date - pd.Timedelta(days=window_days)
        return sum(1 for d in dates if d >= cutoff)

    processed_rows = []

    # Local RNG for reproducible A/B assignment without polluting global numpy state
    rng = np.random.default_rng(42)

    for i, row in df.iterrows():
        surface = row.get('surface', 'Hard')
        w_name = row['winner_name']
        l_name = row['loser_name']

        if pd.isna(w_name) or pd.isna(l_name):
            continue

        # Randomly assign slots A and B
        if rng.random() > 0.5:
            p1, p2 = w_name, l_name
            target = 1  # Player 1 won
            p1_rank, p2_rank = row['winner_rank'], row['loser_rank']
            p1_pts, p2_pts = row['winner_rank_points'], row['loser_rank_points']
            p1_age, p2_age = row['winner_age'], row['loser_age']
            p1_ht, p2_ht = row['winner_ht'], row['loser_ht']
            p1_seed, p2_seed = row['winner_seed'], row['loser_seed']
            p1_hand, p2_hand = row['winner_hand'], row['loser_hand']
        else:
            p1, p2 = l_name, w_name
            target = 0  # Player 1 lost
            p1_rank, p2_rank = row['loser_rank'], row['winner_rank']
            p1_pts, p2_pts = row['loser_rank_points'], row['winner_rank_points']
            p1_age, p2_age = row['loser_age'], row['winner_age']
            p1_ht, p2_ht = row['loser_ht'], row['winner_ht']
            p1_seed, p2_seed = row['loser_seed'], row['winner_seed']
            p1_hand, p2_hand = row['loser_hand'], row['winner_hand']
            
        # --- 1. Get PRE-MATCH stats for P1 ---
        p1_w = history['wins'].get(p1, 0)
        p1_m = history['matches'].get(p1, 0)
        p1_wr = p1_w / p1_m if p1_m > 0 else 0.5
        
        p1_sw = history['surface_wins'].get((p1, surface), 0)
        p1_sm = history['surface_matches'].get((p1, surface), 0)
        p1_swr = p1_sw / p1_sm if p1_sm > 0 else 0.5
        
        p1_recent = history['recent'].get(p1, [])
        p1_form = sum(p1_recent) / len(p1_recent) if p1_recent else 0.5
        
        # --- 2. Get PRE-MATCH stats for P2 ---
        p2_w = history['wins'].get(p2, 0)
        p2_m = history['matches'].get(p2, 0)
        p2_wr = p2_w / p2_m if p2_m > 0 else 0.5
        
        p2_sw = history['surface_wins'].get((p2, surface), 0)
        p2_sm = history['surface_matches'].get((p2, surface), 0)
        p2_swr = p2_sw / p2_sm if p2_sm > 0 else 0.5
        
        p2_recent = history['recent'].get(p2, [])
        p2_form = sum(p2_recent) / len(p2_recent) if p2_recent else 0.5
        
        # --- 3. Head-to-Head ---
        h2h_1 = history['h2h'].get((p1, p2), 0)
        h2h_2 = history['h2h'].get((p2, p1), 0)
        h2h_total = h2h_1 + h2h_2
        h2h_adv = h2h_1 / h2h_total if h2h_total > 0 else 0.5

        # --- 3b. ELO ratings (pre-match) ---
        p1_elo = elo_ratings.get(p1, INITIAL_RATING)
        p2_elo = elo_ratings.get(p2, INITIAL_RATING)
        p1_surf_elo = surface_elo_ratings.get((p1, surface), INITIAL_RATING)
        p2_surf_elo = surface_elo_ratings.get((p2, surface), INITIAL_RATING)

        # --- 3c. Career serve averages (PRE-MATCH) ---
        p1_ace_rate = _serve_rate(p1, 'serve_ace', 'serve_svpt', 0.0)
        p2_ace_rate = _serve_rate(p2, 'serve_ace', 'serve_svpt', 0.0)
        p1_df_rate = _serve_rate(p1, 'serve_df', 'serve_svpt', 0.0)
        p2_df_rate = _serve_rate(p2, 'serve_df', 'serve_svpt', 0.0)
        p1_1st_pct = _serve_rate(p1, 'serve_1stIn', 'serve_svpt', 0.5)
        p2_1st_pct = _serve_rate(p2, 'serve_1stIn', 'serve_svpt', 0.5)
        p1_1st_won = _serve_rate(p1, 'serve_1stWon', 'serve_1stIn', 0.5)
        p2_1st_won = _serve_rate(p2, 'serve_1stWon', 'serve_1stIn', 0.5)
        # 2nd serve won: denominator is (svpt - 1stIn)
        p1_svpt_cum = history['serve_svpt'].get(p1, 0)
        p1_1stIn_cum = history['serve_1stIn'].get(p1, 0)
        p1_2nd_denom = p1_svpt_cum - p1_1stIn_cum
        p1_2nd_won = history['serve_2ndWon'].get(p1, 0) / p1_2nd_denom if p1_2nd_denom > 0 else 0.5
        p2_svpt_cum = history['serve_svpt'].get(p2, 0)
        p2_1stIn_cum = history['serve_1stIn'].get(p2, 0)
        p2_2nd_denom = p2_svpt_cum - p2_1stIn_cum
        p2_2nd_won = history['serve_2ndWon'].get(p2, 0) / p2_2nd_denom if p2_2nd_denom > 0 else 0.5
        p1_bp_save = _serve_rate(p1, 'serve_bpSaved', 'serve_bpFaced', 0.5)
        p2_bp_save = _serve_rate(p2, 'serve_bpSaved', 'serve_bpFaced', 0.5)

        # --- 3e. Surface-specific return pressure (PRE-MATCH) ---
        # Return win % = points won on return / total return points faced
        # Use surface-specific if available, fall back to overall
        def _return_pct(player, surf):
            s_svpt = history['surface_return_svpt'].get((player, surf), 0)
            s_won = history['surface_return_won'].get((player, surf), 0)
            if s_svpt >= 50:  # Enough surface return data
                return s_won / s_svpt
            # Fall back to overall return rate
            o_svpt = history['return_svpt'].get(player, 0)
            o_won = history['return_won'].get(player, 0)
            return o_won / o_svpt if o_svpt > 0 else 0.35  # Tour avg ~35%

        p1_surface_return = _return_pct(p1, surface)
        p2_surface_return = _return_pct(p2, surface)

        # --- 3d. Fatigue metrics (PRE-MATCH) ---
        match_date = row['tourney_date']
        p1_last = history['last_match_date'].get(p1)
        p2_last = history['last_match_date'].get(p2)
        p1_days_rest = min((match_date - p1_last).days, 180) if p1_last and pd.notna(match_date) else 30
        p2_days_rest = min((match_date - p2_last).days, 180) if p2_last and pd.notna(match_date) else 30
        p1_matches_30d = _matches_in_window(p1, match_date)
        p2_matches_30d = _matches_in_window(p2, match_date)

        # --- 4. Handedness features ---
        p1_is_lefty = 1 if p1_hand == 'L' else 0
        p2_is_lefty = 1 if p2_hand == 'L' else 0
        # Cross-hand matchup: 1 if one lefty vs one righty, 0 if same-hand
        hand_cross = 1 if p1_is_lefty != p2_is_lefty else 0
        # Lefty advantage: +1 if P1 is the lefty in a cross-hand matchup,
        # -1 if P2 is the lefty, 0 if same-hand
        lefty_adv = (p1_is_lefty - p2_is_lefty) * hand_cross

        # --- 5. Height-surface interaction ---
        # Raw height diff
        raw_ht_diff = p1_ht - p2_ht
        # Height matters more on fast surfaces (grass/hard) where serve is king,
        # less on clay where rallies dominate
        surface_ht_multiplier = {'Grass': 1.5, 'Hard': 1.0, 'Clay': 0.5, 'Carpet': 1.2}
        ht_surface_adv = raw_ht_diff * surface_ht_multiplier.get(surface, 1.0)

        # --- 6. Assemble Feature Row ---
        feat_row = {
            'target': target,
            # Rank (Lower is better, so P2 - P1 makes more sense mentally)
            'rank_diff': p2_rank - p1_rank,
            'pts_diff': p1_pts - p2_pts,
            'age_diff': p1_age - p2_age,
            'ht_surface_adv': ht_surface_adv,
            'seed_diff': p2_seed - p1_seed,
            # Performance diffs
            'win_rate_diff': p1_wr - p2_wr,
            'surface_wr_diff': p1_swr - p2_swr,
            'form_diff': p1_form - p2_form,
            'experience_diff': np.log1p(p1_m) - np.log1p(p2_m),
            'h2h_adv': h2h_adv,
            # Categorical
            'surface': surface,
            'tourney_level': row['tourney_level'],
            'round': row['round'],
            # Handedness (re-engineered)
            'hand_cross': hand_cross,
            'lefty_adv': lefty_adv,
            # ELO features
            'elo_diff': p1_elo - p2_elo,
            'surface_elo_diff': p1_surf_elo - p2_surf_elo,
            # Serve career average diffs
            'ace_rate_diff': p1_ace_rate - p2_ace_rate,
            'df_rate_diff': p1_df_rate - p2_df_rate,
            'first_serve_pct_diff': p1_1st_pct - p2_1st_pct,
            'first_serve_win_diff': p1_1st_won - p2_1st_won,
            'second_serve_win_diff': p1_2nd_won - p2_2nd_won,
            'bp_save_rate_diff': p1_bp_save - p2_bp_save,
            # Fatigue diffs
            'days_since_last_diff': p1_days_rest - p2_days_rest,
            'matches_last_30d_diff': p1_matches_30d - p2_matches_30d,
            # Surface return pressure diff
            'surface_return_diff': p1_surface_return - p2_surface_return,
            # Metadata for splitting
            'tourney_date': row['tourney_date'],
            # Filter criteria
            'min_m': min(p1_m, p2_m)
        }
        processed_rows.append(feat_row)
        
        # --- 7. Update History with THIS match ---
        history['matches'][w_name] = history['matches'].get(w_name, 0) + 1
        history['wins'][w_name] = history['wins'].get(w_name, 0) + 1
        history['matches'][l_name] = history['matches'].get(l_name, 0) + 1
        
        history['surface_matches'][(w_name, surface)] = history['surface_matches'].get((w_name, surface), 0) + 1
        history['surface_wins'][(w_name, surface)] = history['surface_wins'].get((w_name, surface), 0) + 1
        history['surface_matches'][(l_name, surface)] = history['surface_matches'].get((l_name, surface), 0) + 1
        
        w_rec = history['recent'].get(w_name, [])
        w_rec.append(1)
        history['recent'][w_name] = w_rec[-20:]
        
        l_rec = history['recent'].get(l_name, [])
        l_rec.append(0)
        history['recent'][l_name] = l_rec[-20:]
        
        history['h2h'][(w_name, l_name)] = history['h2h'].get((w_name, l_name), 0) + 1

        # --- 8. Update serve stat accumulators ---
        serve_col_map = [
            ('w_ace', 'l_ace', 'serve_ace'),
            ('w_df', 'l_df', 'serve_df'),
            ('w_svpt', 'l_svpt', 'serve_svpt'),
            ('w_1stIn', 'l_1stIn', 'serve_1stIn'),
            ('w_1stWon', 'l_1stWon', 'serve_1stWon'),
            ('w_2ndWon', 'l_2ndWon', 'serve_2ndWon'),
            ('w_bpSaved', 'l_bpSaved', 'serve_bpSaved'),
            ('w_bpFaced', 'l_bpFaced', 'serve_bpFaced'),
        ]
        for w_col, l_col, hist_key in serve_col_map:
            w_val = row.get(w_col)
            l_val = row.get(l_col)
            if pd.notna(w_val):
                history[hist_key][w_name] = history[hist_key].get(w_name, 0) + float(w_val)
            if pd.notna(l_val):
                history[hist_key][l_name] = history[hist_key].get(l_name, 0) + float(l_val)

        # --- 8b. Update return pressure accumulators ---
        # Winner's return performance = loser's serve points - loser's serve points won
        # i.e., how many points winner won on return against loser's serve
        l_svpt_val = row.get('l_svpt')
        l_1stWon_val = row.get('l_1stWon')
        l_2ndWon_val = row.get('l_2ndWon')
        w_svpt_val = row.get('w_svpt')
        w_1stWon_val = row.get('w_1stWon')
        w_2ndWon_val = row.get('w_2ndWon')

        if pd.notna(l_svpt_val) and pd.notna(l_1stWon_val) and pd.notna(l_2ndWon_val):
            ret_pts = float(l_svpt_val)
            ret_won = float(l_svpt_val) - float(l_1stWon_val) - float(l_2ndWon_val)
            history['return_svpt'][w_name] = history['return_svpt'].get(w_name, 0) + ret_pts
            history['return_won'][w_name] = history['return_won'].get(w_name, 0) + ret_won
            history['surface_return_svpt'][(w_name, surface)] = history['surface_return_svpt'].get((w_name, surface), 0) + ret_pts
            history['surface_return_won'][(w_name, surface)] = history['surface_return_won'].get((w_name, surface), 0) + ret_won

        if pd.notna(w_svpt_val) and pd.notna(w_1stWon_val) and pd.notna(w_2ndWon_val):
            ret_pts = float(w_svpt_val)
            ret_won = float(w_svpt_val) - float(w_1stWon_val) - float(w_2ndWon_val)
            history['return_svpt'][l_name] = history['return_svpt'].get(l_name, 0) + ret_pts
            history['return_won'][l_name] = history['return_won'].get(l_name, 0) + ret_won
            history['surface_return_svpt'][(l_name, surface)] = history['surface_return_svpt'].get((l_name, surface), 0) + ret_pts
            history['surface_return_won'][(l_name, surface)] = history['surface_return_won'].get((l_name, surface), 0) + ret_won

        # --- 9. Update fatigue tracking ---
        match_date = row['tourney_date']
        if pd.notna(match_date):
            history['last_match_date'][w_name] = match_date
            history['last_match_date'][l_name] = match_date
            cutoff_prune = match_date - pd.Timedelta(days=90)
            for name in [w_name, l_name]:
                dates = history['match_dates'].get(name, [])
                dates.append(match_date)
                history['match_dates'][name] = [d for d in dates if d >= cutoff_prune]

        # --- 10. Update ELO state ---
        level = row.get('tourney_level', 'A')
        if pd.isna(level) or level not in K_FACTORS:
            level = 'A'

        k_overall = K_FACTORS[level]
        r_w = elo_ratings.get(w_name, INITIAL_RATING)
        r_l = elo_ratings.get(l_name, INITIAL_RATING)
        exp_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        elo_ratings[w_name] = r_w + k_overall * (1.0 - exp_w)
        elo_ratings[l_name] = r_l + k_overall * (0.0 - (1.0 - exp_w))

        k_surf = k_overall * SURFACE_K_MULTIPLIER
        r_ws = surface_elo_ratings.get((w_name, surface), INITIAL_RATING)
        r_ls = surface_elo_ratings.get((l_name, surface), INITIAL_RATING)
        exp_ws = 1.0 / (1.0 + 10.0 ** ((r_ls - r_ws) / 400.0))
        surface_elo_ratings[(w_name, surface)] = r_ws + k_surf * (1.0 - exp_ws)
        surface_elo_ratings[(l_name, surface)] = r_ls + k_surf * (0.0 - (1.0 - exp_ws))

    return pd.DataFrame(processed_rows)

def prepare_features_for_training(df, min_matches=5):
    """Refactored entry point for feature preparation."""
    df_clean = handle_missing_values(df)
    df_feats = compute_symmetric_features(df_clean)
    
    # Filter by experience
    df_feats = df_feats[df_feats['min_m'] >= min_matches].reset_index(drop=True)
    print(f"  Filtered to matches with >= {min_matches} matches: {len(df_feats):,} rows")
    
    # Encode categorical
    le_surface = LabelEncoder()
    df_feats['surface_enc'] = le_surface.fit_transform(df_feats['surface'].astype(str))
    
    le_level = LabelEncoder()
    df_feats['level_enc'] = le_level.fit_transform(df_feats['tourney_level'].astype(str))
    
    le_round = LabelEncoder()
    df_feats['round_enc'] = le_round.fit_transform(df_feats['round'].astype(str))
    
    encoders = {'surface': le_surface, 'level': le_level, 'round': le_round}
    
    feature_cols = [
        'rank_diff', 'pts_diff', 'age_diff', 'ht_surface_adv', 'seed_diff',
        'win_rate_diff', 'surface_wr_diff', 'form_diff', 'experience_diff', 'h2h_adv',
        'surface_enc', 'level_enc', 'round_enc', 'hand_cross', 'lefty_adv',
        'elo_diff', 'surface_elo_diff',
        'ace_rate_diff', 'df_rate_diff', 'first_serve_pct_diff',
        'first_serve_win_diff', 'second_serve_win_diff', 'bp_save_rate_diff',
        'days_since_last_diff', 'matches_last_30d_diff',
        'surface_return_diff',
    ]
    
    X = df_feats[feature_cols].values
    y = df_feats['target'].values
    
    # Final safety check
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y, feature_cols, encoders, df_feats

def split_data_by_date(df_feats, test_size=0.2):
    """SPLIT THE ALREADY PROCESSED FEATURES TEMPORALLY."""
    # Ensure sorted (already should be, but let's be safe)
    df_feats = df_feats.sort_values('tourney_date').reset_index(drop=True)
    
    split_idx = int(len(df_feats) * (1 - test_size))
    train_df = df_feats.iloc[:split_idx]
    test_df = df_feats.iloc[split_idx:]
    
    print(f"  Split results: Train={len(train_df):,}, Test={len(test_df):,}")
    print(f"  Test data starts from: {test_df['tourney_date'].min()}")
    
    return train_df, test_df
