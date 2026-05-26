"""
Feature engineering module for tennis tournament predictor.
REFACTORED: Symmetric Player A vs Player B approach to eliminate data leakage.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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
    }

    # ELO state — tracked inline to maintain temporal integrity (no lookahead)
    elo_ratings = {}          # player -> float
    surface_elo_ratings = {}  # (player, surface) -> float
    
    processed_rows = []
    
    # Random seed for Reproducible randomness in A/B assignment
    np.random.seed(42)
    
    for i, row in df.iterrows():
        surface = row.get('surface', 'Hard')
        w_name = row['winner_name']
        l_name = row['loser_name']
        
        if pd.isna(w_name) or pd.isna(l_name):
            continue
            
        # Randomly assign slots A and B
        if np.random.random() > 0.5:
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

        # --- 4. Assemble Feature Row ---
        feat_row = {
            'target': target,
            # Rank (Lower is better, so P2 - P1 makes more sense mentally)
            'rank_diff': p2_rank - p1_rank,
            'pts_diff': p1_pts - p2_pts,
            'age_diff': p1_age - p2_age,
            'ht_diff': p1_ht - p2_ht,
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
            'p1_hand': 1 if p1_hand == 'L' else 0,
            'p2_hand': 1 if p2_hand == 'L' else 0,
            # ELO features
            'elo_diff': p1_elo - p2_elo,
            'surface_elo_diff': p1_surf_elo - p2_surf_elo,
            # Metadata for splitting
            'tourney_date': row['tourney_date'],
            # Filter criteria
            'min_m': min(p1_m, p2_m)
        }
        processed_rows.append(feat_row)
        
        # --- 5. Update History with THIS match ---
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

        # --- 6. Update ELO state ---
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
        'rank_diff', 'pts_diff', 'age_diff', 'ht_diff', 'seed_diff',
        'win_rate_diff', 'surface_wr_diff', 'form_diff', 'experience_diff', 'h2h_adv',
        'surface_enc', 'level_enc', 'round_enc', 'p1_hand', 'p2_hand',
        'elo_diff', 'surface_elo_diff',
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


def create_basic_features(df):
    """
    Legacy helper for predictor.py, maps raw winner/loser columns to symmetric diffs.
    Computes differences as (Player 1 - Player 2) or (Player 2 - Player 1) to match 
    the symmetric training features where positive = Player 1 advantage.
    """
    df_features = df.copy()
    
    # Ranking features
    if 'winner_rank' in df_features.columns and 'loser_rank' in df_features.columns:
        # positive means P1 (winner slot in match_data) is better (lower rank)
        df_features['rank_diff'] = df_features['loser_rank'] - df_features['winner_rank']
        df_features['pts_diff'] = df_features['winner_rank_points'] - df_features['loser_rank_points']
        df_features['age_diff'] = df_features['winner_age'] - df_features['loser_age']
        df_features['ht_diff'] = df_features['winner_ht'] - df_features['loser_ht']
        df_features['seed_diff'] = df_features['loser_seed'] - df_features['winner_seed']
    
    # Historical performance features
    if 'w_win_rate' in df_features.columns and 'l_win_rate' in df_features.columns:
        df_features['win_rate_diff'] = df_features['w_win_rate'] - df_features['l_win_rate']
        df_features['surface_wr_diff'] = df_features['w_surface_win_rate'] - df_features['l_surface_win_rate']
        df_features['form_diff'] = df_features['w_recent_form'] - df_features['l_recent_form']
        df_features['experience_diff'] = np.log1p(df_features['w_total_matches']) - np.log1p(df_features['l_total_matches'])
        df_features['h2h_adv'] = df_features['h2h_w_advantage']

    # ELO diffs (present only when predictor has ELO loaded)
    if 'p1_elo' in df_features.columns and 'p2_elo' in df_features.columns:
        df_features['elo_diff'] = df_features['p1_elo'] - df_features['p2_elo']
    if 'p1_surf_elo' in df_features.columns and 'p2_surf_elo' in df_features.columns:
        df_features['surface_elo_diff'] = df_features['p1_surf_elo'] - df_features['p2_surf_elo']

    return df_features


def encode_categorical_features(df):
    """
    Legacy helper for predictor.py, handles basic encoding of fresh match data.
    """
    df_encoded = df.copy()
    
    # Hand encodings
    if 'winner_hand' in df_encoded.columns:
        df_encoded['p1_hand'] = (df_encoded['winner_hand'] == 'L').astype(int)
    if 'loser_hand' in df_encoded.columns:
        df_encoded['p2_hand'] = (df_encoded['loser_hand'] == 'L').astype(int)
        
    # Categorical mappings (simplified for the predictor)
    # MatchPredictor should ideally use fitted encoders, but these maps provide a safe default
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    if 'surface' in df_encoded.columns:
        df_encoded['surface_enc'] = df_encoded['surface'].map(surface_map).fillna(0)
        
    level_map = {'G': 0, 'M': 1, 'A': 2, 'C': 3, 'S': 4, 'F': 5}
    if 'tourney_level' in df_encoded.columns:
        df_encoded['level_enc'] = df_encoded['tourney_level'].map(level_map).fillna(0)
        
    round_map = {'R128': 0, 'R64': 1, 'R32': 2, 'R16': 3, 'QF': 4, 'SF': 5, 'F': 6, 'RR': 7}
    if 'round' in df_encoded.columns:
        df_encoded['round_enc'] = df_encoded['round'].map(round_map).fillna(0)

    return df_encoded, {}
