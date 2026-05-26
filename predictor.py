"""
Match prediction module for tennis tournament predictor.
Uses trained models to predict match outcomes.
"""
import os
import pandas as pd
import numpy as np
from model_trainer import ModelTrainer
from feature_engineering import handle_missing_values, create_basic_features, encode_categorical_features
import warnings
warnings.filterwarnings('ignore')


class MatchPredictor:
    """Class to predict match outcomes using trained models."""

    def __init__(self, models_dir='models'):
        self.trainer = ModelTrainer(models_dir)
        self.trainer.load_models()
        self.feature_names = self.trainer.feature_names
        self.encoders = {}
        from elo_calculator import ELOCalculator
        self.elo_calculator = ELOCalculator(cache_path=os.path.join(models_dir, 'elo_ratings.json'))
        self._elo_loaded = False
    
    def _build_feature_vector(self, player1_data, player2_data, surface, tourney_level):
        """
        Build the feature vector directly from player data, matching the training feature schema.
        
        This constructs the exact same features used during training:
        - Rank/points differences and ratios
        - Historical performance differences (win rate, surface win rate, recent form, h2h)
        - Categorical encodings
        """
        p1 = player1_data
        p2 = player2_data
        
        # Create a single-row DataFrame with the raw columns needed
        match_data = {
            'winner_rank': p1.get('rank', 9999),
            'loser_rank': p2.get('rank', 9999),
            'winner_rank_points': p1.get('rank_points', 0),
            'loser_rank_points': p2.get('rank_points', 0),
            'winner_seed': p1.get('seed', 999),
            'loser_seed': p2.get('seed', 999),
            'winner_age': p1.get('age', 25),
            'loser_age': p2.get('age', 25),
            'winner_ht': p1.get('height', 180),
            'loser_ht': p2.get('height', 180),
            'winner_hand': p1.get('hand', 'R'),
            'loser_hand': p2.get('hand', 'R'),
            'surface': surface or 'Hard',
            'tourney_level': tourney_level or 'M',
            'round': 'F',
            # Historical features (injected directly, not computed from match)
            'w_win_rate': p1.get('win_rate', 0.5),
            'l_win_rate': p2.get('win_rate', 0.5),
            'w_surface_win_rate': p1.get('surface_win_rate', 0.5),
            'l_surface_win_rate': p2.get('surface_win_rate', 0.5),
            'w_recent_form': p1.get('recent_form', 0.5),
            'l_recent_form': p2.get('recent_form', 0.5),
            'w_total_matches': p1.get('total_matches', 0),
            'l_total_matches': p2.get('total_matches', 0),
            'h2h_w_advantage': p1.get('h2h_advantage', 0.5),
        }
        
        # Inject ELO diffs for retrained models (ignored by old 15-feature models)
        if self._elo_loaded:
            p1_name = p1.get('name', '')
            p2_name = p2.get('name', '')
            match_data['p1_elo'] = self.elo_calculator.get_rating(p1_name)
            match_data['p2_elo'] = self.elo_calculator.get_rating(p2_name)
            if surface:
                match_data['p1_surf_elo'] = self.elo_calculator.get_surface_rating(p1_name, surface)
                match_data['p2_surf_elo'] = self.elo_calculator.get_surface_rating(p2_name, surface)

        df = pd.DataFrame([match_data])

        # Handle missing values
        df = handle_missing_values(df)

        # Create basic features (differences, ratios)
        df = create_basic_features(df)
        
        # Encode categorical features
        df, _ = encode_categorical_features(df)
        
        # Build feature vector in the same order as training
        feature_vector = np.zeros(len(self.feature_names))
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in df.columns:
                val = df[feat_name].iloc[0]
                feature_vector[i] = val if pd.notna(val) else 0.0
        
        # Safety: replace NaN/inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector.reshape(1, -1)
    
    def predict_match(self, player1_data, player2_data, surface=None, tourney_level=None,
                      player1_name="Player 1", player2_name="Player 2"):
        """
        Predict match outcome between two players.
        """
        # Build feature vector
        X = self._build_feature_vector(player1_data, player2_data, surface, tourney_level)
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.trainer.models.items():
            try:
                if model_name == 'neural_network':
                    scaler = self.trainer.scalers.get('neural_network')
                    if scaler:
                        X_scaled = scaler.transform(X)
                        proba = model.predict(X_scaled, verbose=0)[0][0]
                    else:
                        continue
                elif model_name in ['logistic_regression', 'knn', 'svm']:
                    proba = model.predict_proba(X)[0][1]
                else:
                    proba = model.predict_proba(X)[0][1]
                
                predictions[model_name] = {
                    'probability': float(proba),
                    'predicted_winner': player1_name if proba >= 0.5 else player2_name,
                    'confidence': abs(proba - 0.5) * 2
                }
            except Exception as e:
                print(f"Warning: Could not get prediction from {model_name}: {e}")
                continue
        
        # Add ELO prediction to ensemble
        if self._elo_loaded:
            predictions['elo'] = self.elo_calculator.predict_winner(
                player1_name, player2_name, surface
            )

        # Add metadata for display
        predictions['_tourney_level'] = tourney_level if tourney_level else 'M'

        return predictions
    
    def predict_from_names(self, player1_name, player2_name, df_historical=None, surface=None, tourney_level=None, 
                           player1_year=None, player2_year=None, strict_year=False):
        """
        Predict match outcome from player names using historical data.
        Optional: 
            player1_year, player2_year (int): Include stats only up to (or exactly in) this year.
            strict_year (bool): If True, only use data from the specific year(s) provided.
        """
        if df_historical is None:
            print("Error: Historical data required for prediction from names")
            return None
        
        # Helper to filter dataframe by year
        def filter_by_year(df, year, strict=False):
            if year is None:
                return df
            
            try:
                # Extract year safely
                if pd.api.types.is_numeric_dtype(df['tourney_date']):
                    # Numeric YYYYMMDD
                    years = df['tourney_date'] // 10000
                else:
                    # String YYYYMMDD or mixed
                    # Convert to string, take first 4 chars, convert to numeric
                    dates_str = df['tourney_date'].astype(str).str[:4]
                    years = pd.to_numeric(dates_str, errors='coerce')
                
                # Create mask
                if strict:
                    # Exactly this year
                    mask = (years == int(year))
                else:
                    # Up to this year
                    mask = (years.notna()) & (years <= int(year))
                
                return df[mask]
            except Exception as e:
                print(f"Warning: Could not filter by year {year}: {e}")
                return df

        # Filter data for each player if years provided
        df_p1 = filter_by_year(df_historical, player1_year, strict=strict_year)
        df_p2 = filter_by_year(df_historical, player2_year, strict=strict_year)
        
        # Find player stats from historical data (using filtered data)
        player1_data = self._get_player_stats(player1_name, df_p1, surface)
        player2_data = self._get_player_stats(player2_name, df_p2, surface)
        
        # Add year metadata
        if player1_year: player1_data['year_filter'] = player1_year
        if player2_year: player2_data['year_filter'] = player2_year

        # Compute head-to-head from historical data
        # For H2H, if years are different, we arguably should use the LATEST year to be realistic 
        # (the later player knows the history). Or separate universe? 
        # Standard approach: Use the filtered data of the "later" player (or both if same).
        # Let's use the union of the relevant timeframes or just the raw df if we want "historical truth".
        # But if I pitch 2006 Federer vs 2006 Nadal, I shouldn't see 2017 matches.
        # So H2H should also be filtered!
        # Use max year of the two inputs.
        
        h2h_year = None
        if player1_year and player2_year:
            h2h_year = max(int(player1_year), int(player2_year))
        elif player1_year:
            h2h_year = player1_year
        elif player2_year:
            h2h_year = player2_year
            
        df_h2h = filter_by_year(df_historical, h2h_year)
        
        h2h = self._get_h2h(player1_name, player2_name, df_h2h)
        player1_data['h2h_advantage'] = h2h['p1_advantage']
        player2_data['h2h_advantage'] = 1.0 - h2h['p1_advantage']
        
        # Load ELO ratings on first prediction (uses the same df already in memory)
        if not self._elo_loaded:
            self.elo_calculator.get_or_compute(df_historical)
            self._elo_loaded = True

        # Attach names so _build_feature_vector can look up ELO diffs
        player1_data['name'] = player1_name
        player2_data['name'] = player2_name

        predictions = self.predict_match(
            player1_data, player2_data, surface, tourney_level,
            player1_name, player2_name
        )
        
        # Attach stats for display
        if predictions:
            predictions['_player1_stats'] = player1_data
            predictions['_player2_stats'] = player2_data
            predictions['_h2h'] = h2h
        
        return predictions
    
    def _get_h2h(self, player1_name, player2_name, df):
        """Get head-to-head record between two players."""
        p1_wins = len(df[(df['winner_name'] == player1_name) & (df['loser_name'] == player2_name)])
        p2_wins = len(df[(df['winner_name'] == player2_name) & (df['loser_name'] == player1_name)])
        total = p1_wins + p2_wins
        
        return {
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'total': total,
            'p1_advantage': p1_wins / total if total > 0 else 0.5,
        }
    
    def _get_player_stats(self, player_name, df, surface=None):
        """Extract player statistics from historical data (averages across recent matches)."""
        winner_matches = df[df['winner_name'] == player_name].copy()
        loser_matches = df[df['loser_name'] == player_name].copy()
        
        total_wins = len(winner_matches)
        total_losses = len(loser_matches)
        total_matches = total_wins + total_losses
        
        if total_matches == 0:
            return {
                'rank': 9999, 'rank_points': 0, 'seed': 999,
                'age': 25, 'height': 180, 'hand': 'R',
                'win_rate': 0.5, 'surface_win_rate': 0.5,
                'recent_form': 0.5, 'total_matches': 0,
                'h2h_advantage': 0.5,
                'aces': 5, 'double_faults': 2, 'service_points': 60,
                'first_serve_in': 40, 'first_serve_won': 30,
                'second_serve_won': 15, 'service_games': 12,
                'break_points_saved': 2, 'break_points_faced': 3,
                'first_serve_pct': 66.7, 'first_serve_win_pct': 75.0,
                'second_serve_win_pct': 50.0, 'bp_save_pct': 66.7,
                'found_in_data': False, 'wins': 0, 'losses': 0,
                'surface_preference': 'None', 'best_surface_win_rate': 0.0 # Default
            }
        
        # Get most recent match for rank/age/height/hand
        all_matches = pd.concat([winner_matches, loser_matches])
        if 'tourney_date' in all_matches.columns:
            all_matches['tourney_date'] = pd.to_datetime(
                all_matches['tourney_date'], format='%Y%m%d', errors='coerce'
            )
            all_matches = all_matches.sort_values('tourney_date', ascending=False)
        
        latest_match = all_matches.iloc[0]
        is_winner = latest_match['winner_name'] == player_name
        
        if is_winner:
            rank = latest_match.get('winner_rank', 9999)
            rank_points = latest_match.get('winner_rank_points', 0)
            seed = latest_match.get('winner_seed', 999)
            age = latest_match.get('winner_age', 25)
            height = latest_match.get('winner_ht', 180)
            hand = latest_match.get('winner_hand', 'R')
        else:
            rank = latest_match.get('loser_rank', 9999)
            rank_points = latest_match.get('loser_rank_points', 0)
            seed = latest_match.get('loser_seed', 999)
            age = latest_match.get('loser_age', 25)
            height = latest_match.get('loser_ht', 180)
            hand = latest_match.get('loser_hand', 'R')
        
        # Overall win rate with Laplacian smoothing (pseudo-counts)
        # Add 2 wins and 2 losses to stabilize small samples
        win_rate = (total_wins + 2) / (total_matches + 4)
        
        # Surface-specific win rate
        if surface:
            surface_wins = len(winner_matches[winner_matches['surface'] == surface])
            surface_losses = len(loser_matches[loser_matches['surface'] == surface])
            surface_total = surface_wins + surface_losses
            # Blend surface win rate with overall win rate
            if surface_total > 5:
                # Use surface stats directly if decent sample
                surface_win_rate = (surface_wins + 1) / (surface_total + 2)
            else:
                # Weighted average: more overall win rate if low surface sample
                weight = surface_total / 5.0
                raw_swr = surface_wins / surface_total if surface_total > 0 else win_rate
                surface_win_rate = (weight * raw_swr) + ((1 - weight) * win_rate)
        else:
            surface_win_rate = win_rate
            
        # --- Surface Preference Logic ---
        best_surface = "None"
        best_wr = 0.0
        
        for surf in ['Hard', 'Clay', 'Grass', 'Carpet']:
            s_wins = len(winner_matches[winner_matches['surface'] == surf])
            s_losses = len(loser_matches[loser_matches['surface'] == surf])
            s_total = s_wins + s_losses
            
            if s_total >= 5: # Minimum matches to consider relevant preference
                s_wr = s_wins / s_total
                if s_wr > best_wr:
                    best_wr = s_wr
                    best_surface = surf
        
        # Fallback if no specific preference found (e.g. low matches)
        if best_surface == "None" and total_matches > 0:
             # Just take highest win rate even if low matches
             for surf in ['Hard', 'Clay', 'Grass', 'Carpet']:
                s_wins = len(winner_matches[winner_matches['surface'] == surf])
                s_losses = len(loser_matches[loser_matches['surface'] == surf])
                s_total = s_wins + s_losses
                if s_total > 0:
                    s_wr = s_wins / s_total
                    if s_wr > best_wr:
                        best_wr = s_wr
                        best_surface = surf
        
        # Recent form (last 20 matches)
        recent_n = 20
        if 'tourney_date' in all_matches.columns:
            recent_matches = all_matches.head(recent_n)
        else:
            recent_matches = all_matches.tail(recent_n)
        
        recent_wins = len(recent_matches[recent_matches['winner_name'] == player_name])
        # Smoothing for recent form too (pseudo-counts)
        recent_form = (recent_wins + 1) / (len(recent_matches) + 2) if len(recent_matches) > 0 else 0.5
        
        # Compute average match stats (for display)
        w_stats = winner_matches.tail(recent_n)
        l_stats = loser_matches.tail(recent_n)
        
        def safe_mean(series_w, series_l, default):
            all_vals = pd.concat([series_w, series_l]).dropna()
            return float(all_vals.mean()) if len(all_vals) > 0 else default
        
        aces = safe_mean(
            w_stats['w_ace'] if 'w_ace' in w_stats.columns else pd.Series(),
            l_stats['l_ace'] if 'l_ace' in l_stats.columns else pd.Series(), 5)
        double_faults = safe_mean(
            w_stats['w_df'] if 'w_df' in w_stats.columns else pd.Series(),
            l_stats['l_df'] if 'l_df' in l_stats.columns else pd.Series(), 2)
        service_points = safe_mean(
            w_stats['w_svpt'] if 'w_svpt' in w_stats.columns else pd.Series(),
            l_stats['l_svpt'] if 'l_svpt' in l_stats.columns else pd.Series(), 60)
        first_serve_in = safe_mean(
            w_stats['w_1stIn'] if 'w_1stIn' in w_stats.columns else pd.Series(),
            l_stats['l_1stIn'] if 'l_1stIn' in l_stats.columns else pd.Series(), 40)
        first_serve_won = safe_mean(
            w_stats['w_1stWon'] if 'w_1stWon' in w_stats.columns else pd.Series(),
            l_stats['l_1stWon'] if 'l_1stWon' in l_stats.columns else pd.Series(), 30)
        second_serve_won = safe_mean(
            w_stats['w_2ndWon'] if 'w_2ndWon' in w_stats.columns else pd.Series(),
            l_stats['l_2ndWon'] if 'l_2ndWon' in l_stats.columns else pd.Series(), 15)
        service_games = safe_mean(
            w_stats['w_SvGms'] if 'w_SvGms' in w_stats.columns else pd.Series(),
            l_stats['l_SvGms'] if 'l_SvGms' in l_stats.columns else pd.Series(), 12)
        bp_saved = safe_mean(
            w_stats['w_bpSaved'] if 'w_bpSaved' in w_stats.columns else pd.Series(),
            l_stats['l_bpSaved'] if 'l_bpSaved' in l_stats.columns else pd.Series(), 2)
        bp_faced = safe_mean(
            w_stats['w_bpFaced'] if 'w_bpFaced' in w_stats.columns else pd.Series(),
            l_stats['l_bpFaced'] if 'l_bpFaced' in l_stats.columns else pd.Series(), 3)
        
        first_serve_pct = (first_serve_in / service_points * 100) if service_points > 0 else 0
        first_serve_win_pct = (first_serve_won / first_serve_in * 100) if first_serve_in > 0 else 0
        second_serve_pts = service_points - first_serve_in
        second_serve_win_pct = (second_serve_won / second_serve_pts * 100) if second_serve_pts > 0 else 0
        bp_save_pct = (bp_saved / bp_faced * 100) if bp_faced > 0 else 0
        
        return {
            'rank': rank if pd.notna(rank) else 9999,
            'rank_points': rank_points if pd.notna(rank_points) else 0,
            'seed': seed if pd.notna(seed) else 999,
            'age': age if pd.notna(age) else 25,
            'height': height if pd.notna(height) else 180,
            'hand': hand if pd.notna(hand) else 'R',
            'win_rate': round(win_rate, 4),
            'surface_win_rate': round(surface_win_rate, 4),
            'recent_form': round(recent_form, 4),
            'total_matches': total_matches,
            'h2h_advantage': 0.5,  # Will be overwritten by predict_from_names
            'aces': round(aces, 1),
            'double_faults': round(double_faults, 1),
            'service_points': round(service_points, 1),
            'first_serve_in': round(first_serve_in, 1),
            'first_serve_won': round(first_serve_won, 1),
            'second_serve_won': round(second_serve_won, 1),
            'service_games': round(service_games, 1),
            'break_points_saved': round(bp_saved, 1),
            'break_points_faced': round(bp_faced, 1),
            'first_serve_pct': round(first_serve_pct, 1),
            'first_serve_win_pct': round(first_serve_win_pct, 1),
            'second_serve_win_pct': round(second_serve_win_pct, 1),
            'bp_save_pct': round(bp_save_pct, 1),
            'found_in_data': True,
            'wins': total_wins,
            'losses': total_losses,
            'surface_preference': best_surface,
            'best_surface_win_rate': round(best_wr, 4),
        }

    def simulate_score(self, winner_name, loser_name, winner_prob, sets_to_play, p1_stats, p2_stats):
        """
        Public wrapper to simulate match score.
        """
        return self._simulate_score(winner_name, loser_name, winner_prob, sets_to_play, p1_stats, p2_stats)
    
    def _simulate_score(self, winner_name, loser_name, winner_prob, sets_to_play, p1_stats, p2_stats):
        """
        Simulate a realistic match score based on winner probability and tournament rules.
        
        Args:
            winner_name: Name of predicted winner
            loser_name: Name of predicted loser
            winner_prob: Probability of winning (0.5 to 1.0)
            sets_to_play: Total sets to be played (must be odd, e.g. 3, 5, 7)
            p1_stats, p2_stats: Player statistics (for serve strength inference)
        
        Returns:
            String representing the score (e.g. "6-4 6-3" or "6-7(5) 7-6(4) 6-4")
        """
        # Determine sets needed to win
        # If sets_to_play is 3, sets_needed is 2. If 5, sets_needed is 3.
        sets_needed = (sets_to_play // 2) + 1


        
        # Adjust set win probability relative to match win probability
        # If match prob is high (e.g. 0.8), set prob should be higher to reflect dominance
        # If match prob is low (e.g. 0.55), sets should be closer
        set_margin = (winner_prob - 0.5) * 1.5  # Amplify margin for individual sets
        set_win_prob = 0.5 + set_margin
        set_win_prob = min(0.95, max(0.05, set_win_prob))
        
        # Serve strength factor (for tiebreak likelihood)
        p1_serve = p1_stats.get('first_serve_won', 70) if p1_stats else 70
        p2_serve = p2_stats.get('first_serve_won', 70) if p2_stats else 70
        avg_serve_win = (p1_serve + p2_serve) / 2
        tiebreak_factor = (avg_serve_win - 60) / 40  # Higher serve win % -> more tiebreaks
        tiebreak_factor = min(0.8, max(0.1, tiebreak_factor))
        
        winner_sets = 0
        loser_sets = 0
        score_parts = []
        
        # Simulate sets until one reaches sets_needed
        while winner_sets < sets_needed:
            # Determine who wins this set
            # Bias towards the predicted winner
            is_winner_set = np.random.random() < set_win_prob
            
            # Ensure loser doesn't exceed sets_needed - 1
            if not is_winner_set and loser_sets >= sets_needed - 1:
                is_winner_set = True
            
            if is_winner_set:
                winner_sets += 1
                # Determine score margin
                rand = np.random.random()
                if rand < 0.3 * (1 - tiebreak_factor): # Dominant set
                    score = "6-1" if np.random.random() < 0.3 else "6-2"
                elif rand < 0.7 * (1 - tiebreak_factor): # Solid set
                    score = "6-3"
                elif rand < 0.9 - (0.2 * tiebreak_factor): # Close set
                    score = "6-4"
                else: # Tiebreak or 7-5
                    if np.random.random() < 0.6 + tiebreak_factor:
                        loser_pts = np.random.randint(0, 6) if np.random.random() < 0.7 else np.random.randint(6, 9)
                        score = f"7-6({loser_pts})"
                    else:
                        score = "7-5"
                score_parts.append(score)
                
            else:
                loser_sets += 1
                # Loser wins a set (usually closer if they are the underdog)
                rand = np.random.random()
                if rand < 0.2:
                    score = "1-6" if np.random.random() < 0.3 else "2-6"
                elif rand < 0.6:
                    score = "3-6" if np.random.random() < 0.5 else "4-6"
                else:
                    if np.random.random() < 0.6 + tiebreak_factor:
                        loser_pts = np.random.randint(0, 6) if np.random.random() < 0.7 else np.random.randint(6, 9)
                        score = f"6-7({loser_pts})"
                    else:
                        score = "5-7"
                score_parts.append(score)
        
        full_score = " ".join(score_parts)
        return full_score

    def display_predictions(self, predictions, player1_name="Player 1", player2_name="Player 2", sets_to_play=None):
        """Display predictions in a formatted way with player stats."""
        # Use .get() to avoid mutating the caller's dict
        player1_stats = predictions.get('_player1_stats', None)
        player2_stats = predictions.get('_player2_stats', None)
        h2h = predictions.get('_h2h', None)
        tourney_level = predictions.get('_tourney_level', 'M')
        if sets_to_play is None:
            sets_to_play = 5 if tourney_level == 'G' else 3


        
        # Get consensus prediction for score simulation
        probs = [p['probability'] for k, p in predictions.items() if not k.startswith('_')]
        avg_prob = np.mean(probs) if probs else 0.5
        
        # Determine predicted winner based on consensus
        predicted_winner_name = player1_name if avg_prob >= 0.5 else player2_name
        predicted_loser_name = player2_name if avg_prob >= 0.5 else player1_name
        win_prob = avg_prob if avg_prob >= 0.5 else (1 - avg_prob)
        
        # Consider if we need to pass tourney_level explicitly.
        # We already extracted it above.

        
        # Simulate Score
        predicted_score = self._simulate_score(
            predicted_winner_name, predicted_loser_name, win_prob, sets_to_play,
            player1_stats, player2_stats
        )
        
        print("\n" + "="*70)
        print(f"  MATCH PREDICTION: {player1_name} vs {player2_name}")
        print("="*70)
        
        if not predictions:
            print("No predictions available.")
            return
        
        col_w = 30
        
        # --- Player Profiles ---
        if player1_stats or player2_stats:
            print("\n" + "-"*70)
            print("  PLAYER PROFILES")
            print("-"*70)
            
            header = f"  {'':20s} {player1_name:>{col_w}s}   {player2_name:>{col_w}s}"
            print(header)
            print("  " + "-" * (20 + col_w * 2 + 3))
            
            p1 = player1_stats or {}
            p2 = player2_stats or {}
            
            def row(label, key, fmt=".0f", suffix=""):
                v1 = p1.get(key, '-')
                v2 = p2.get(key, '-')
                def format_val(v):
                    if label == "Rank" and v == 9999: return "UR"
                    if isinstance(v, (int, float)): return f"{v:{fmt}}{suffix}"
                    return str(v)
                
                s1 = format_val(v1)
                s2 = format_val(v2)
                print(f"  {label:20s} {s1:>{col_w}s}   {s2:>{col_w}s}")
            
            r1 = p1.get('rank', 9999)
            r2 = p2.get('rank', 9999)
            
            # Display rank (use 'UR' for unranked)
            rank1_str = f"#{r1}" if r1 < 9999 else "UR"
            rank2_str = f"#{r2}" if r2 < 9999 else "UR"
            
            print(f"  {'Rank':20s} {rank1_str:>{col_w}s}   {rank2_str:>{col_w}s}")
            row("Ranking Points", "rank_points", ".0f")
            row("Age", "age", ".1f")
            
            # Era / Year Filter
            if p1.get('year_filter') or p2.get('year_filter'):
                row("Era (End Year)", "year_filter")
                
            row("Height (cm)", "height", ".0f")
            row("Hand", "hand")
            
            # Record
            w1, l1 = p1.get('wins', 0), p1.get('losses', 0)
            w2, l2 = p2.get('wins', 0), p2.get('losses', 0)
            rec1 = f"{w1}-{l1} ({w1/(w1+l1)*100:.1f}%)" if (w1+l1) > 0 else "N/A"
            rec2 = f"{w2}-{l2} ({w2/(w2+l2)*100:.1f}%)" if (w2+l2) > 0 else "N/A"
            print(f"  {'Record (W-L)':20s} {rec1:>{col_w}s}   {rec2:>{col_w}s}")
            
            # Win rate and form
            row("Overall Win Rate", "win_rate", ".1%")
            row("Surface Win Rate", "surface_win_rate", ".1%")
            
            # Surface Preference
            surf1 = p1.get('surface_preference', '-')
            surf2 = p2.get('surface_preference', '-')
            swr1 = p1.get('best_surface_win_rate', 0)
            swr2 = p2.get('best_surface_win_rate', 0)
            
            s1_str = f"{surf1} ({swr1*100:.1f}%)" if surf1 != 'None' and surf1 != '-' else '-'
            s2_str = f"{surf2} ({swr2*100:.1f}%)" if surf2 != 'None' and surf2 != '-' else '-'
            print(f"  {'Best Surface':20s} {s1_str:>{col_w}s}   {s2_str:>{col_w}s}")
            
            row("Recent Form (L20)", "recent_form", ".1%")
            
            # H2H
            if h2h and h2h.get('total', 0) > 0:
                h2h_str1 = f"{h2h['p1_wins']}"
                h2h_str2 = f"{h2h['p2_wins']}"
                print(f"  {'Head-to-Head':20s} {h2h_str1:>{col_w}s}   {h2h_str2:>{col_w}s}")
            else:
                print(f"  {'Head-to-Head':20s} {'No prior meetings':>{col_w}s}   {'':>{col_w}s}")
        
        # --- Predicted Stat Lines ---
        if player1_stats or player2_stats:
            print("\n" + "-"*70)
            print("  PREDICTED STAT LINES (based on recent match averages)")
            print("-"*70)
            
            col_s = col_w - 8
            header = f"  {'Stat':28s} {player1_name:>{col_s}s}   {player2_name:>{col_s}s}"
            print(header)
            print("  " + "-" * (28 + col_s * 2 + 3))
            
            def stat_row(label, key, fmt=".1f", suffix=""):
                v1 = p1.get(key, '-')
                v2 = p2.get(key, '-')
                s1 = f"{v1:{fmt}}{suffix}" if isinstance(v1, (int, float)) else str(v1)
                s2 = f"{v2:{fmt}}{suffix}" if isinstance(v2, (int, float)) else str(v2)
                print(f"  {label:28s} {s1:>{col_s}s}   {s2:>{col_s}s}")
            
            stat_row("Aces", "aces")
            stat_row("Double Faults", "double_faults")
            stat_row("Service Points", "service_points")
            stat_row("1st Serve In", "first_serve_in")
            stat_row("1st Serve %", "first_serve_pct", ".1f", "%")
            stat_row("1st Serve Won", "first_serve_won")
            stat_row("1st Serve Win %", "first_serve_win_pct", ".1f", "%")
            stat_row("2nd Serve Won", "second_serve_won")
            stat_row("2nd Serve Win %", "second_serve_win_pct", ".1f", "%")
            stat_row("Service Games", "service_games")
            stat_row("Break Points Faced", "break_points_faced")
            stat_row("Break Points Saved", "break_points_saved")
            stat_row("Break Points Saved %", "bp_save_pct", ".1f", "%")
        
        # --- Model Predictions ---
        print("\n" + "-"*90)
        print("  MODEL PREDICTIONS")
        print("-"*90)
        print(f"\n  {'Model':<20} {'Winner':<20} {'Prob':>6}   {'Conf':>6}   {'Predicted Score'}")
        print("  " + "-" * 85)
        
        for model_name, pred in predictions.items():
            # Skip metadata and non-model entries like 'Monte Carlo'
            if model_name.startswith('_') or 'predicted_winner' not in pred:
                continue
                
            model_display = model_name.replace('_', ' ').title()
            winner = pred['predicted_winner']
            loser = player2_name if winner == player1_name else player1_name
            prob_p1 = pred['probability']
            
            # (Re-fetch stats if needed or use local copy)
            p1_s = player1_stats or {}
            p2_s = player2_stats or {}
            
            # Individual Model Score
            prob_win = prob_p1 if winner == player1_name else (1.0 - prob_p1)
            model_score = self._simulate_score(winner, loser, prob_win, sets_to_play, p1_s, p2_s)
            
            conf = pred['confidence']
            print(f"  {model_display:<20} {winner:<20} {prob_win:>6.1%}   {conf:>6.1%}   {model_score}")
        
        # Average prediction (only include actual models, skip 'Monte Carlo' and metadata)
        actual_model_probs = [
            p['probability'] for k, p in predictions.items() 
            if not k.startswith('_') and 'predicted_winner' in p
        ]
        
        if not actual_model_probs:
            print("\n  [ERR] No valid model predictions found for consensus.")
            return

        avg_prob_p1 = np.mean(actual_model_probs)
        avg_winner = player1_name if avg_prob_p1 >= 0.5 else player2_name
        avg_loser = player2_name if avg_winner == player1_name else player1_name
        avg_prob_to_show = avg_prob_p1 if avg_winner == player1_name else (1.0 - avg_prob_p1)
        avg_conf = abs(avg_prob_p1 - 0.5) * 2
        
        # Consensus Score
        consensus_score = self._simulate_score(avg_winner, avg_loser, avg_prob_to_show, sets_to_play, player1_stats, player2_stats)
        
        print("  " + "-" * 85)
        print(f"  {'CONSENSUS':<20} {avg_winner:<20} {avg_prob_to_show:>6.1%}   {avg_conf:>6.1%}   {consensus_score}")
        print("  " + "-" * 85)
        print(f"  {'MATCH TYPE':<20} Best of {sets_to_play} sets (First to {(sets_to_play // 2) + 1})")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    from data_loader import load_data
    
    print("Testing match predictor...")
    df = load_data()
    
    predictor = MatchPredictor()
    predictions = predictor.predict_from_names("Roger Federer", "Rafael Nadal", df, "Clay", "G")
    predictor.display_predictions(predictions, "Roger Federer", "Rafael Nadal")
