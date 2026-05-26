"""
ELO rating system for tennis match prediction.
Computes overall and surface-specific ELO ratings from historical match data.
Integrates with the existing predictions ensemble as an additional model.
"""
import json
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

INITIAL_RATING = 1500.0

K_FACTORS = {
    'G': 40,   # Grand Slam
    'M': 32,   # Masters 1000
    'A': 24,   # ATP 500/250
    'C': 16,   # Challenger
    'S': 8,    # Satellite/ITF
    'F': 32,   # ATP Finals
    'D': 20,   # Davis Cup
}

SURFACE_K_MULTIPLIER = 0.7


class ELOCalculator:
    """
    Computes and caches ELO ratings for ATP tennis players.
    Supports overall ratings and surface-specific ratings.
    """

    def __init__(self, cache_path='models/elo_ratings.json'):
        self.cache_path = cache_path
        self.ratings = {}           # player_name -> float
        self.surface_ratings = {}   # (player_name, surface) -> float
        self._is_computed = False

    # ------------------------------------------------------------------
    # Core ELO math
    # ------------------------------------------------------------------

    def _expected_score(self, rating_a, rating_b):
        """P(A beats B) via standard ELO formula."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _get_k(self, tourney_level, surface=False):
        """Return K-factor for a tournament level, with optional surface multiplier."""
        k = K_FACTORS.get(tourney_level, K_FACTORS['A'])
        if surface:
            k *= SURFACE_K_MULTIPLIER
        return k

    def _update_ratings(self, winner, loser, k, ratings_dict):
        """
        In-place ELO update for one match result.
        Auto-initialises missing players to INITIAL_RATING.
        """
        r_w = ratings_dict.get(winner, INITIAL_RATING)
        r_l = ratings_dict.get(loser, INITIAL_RATING)

        expected_w = self._expected_score(r_w, r_l)
        expected_l = 1.0 - expected_w

        ratings_dict[winner] = r_w + k * (1.0 - expected_w)
        ratings_dict[loser] = r_l + k * (0.0 - expected_l)

    # ------------------------------------------------------------------
    # Bulk computation
    # ------------------------------------------------------------------

    def compute_from_dataframe(self, df):
        """
        Single chronological pass over historical match data to build ELO ratings.
        Updates self.ratings (overall) and self.surface_ratings (per surface).
        """
        print("  Computing ELO ratings from historical data...")

        df_sorted = df.copy()

        # Sort chronologically — same approach as feature_engineering.py
        try:
            df_sorted['tourney_date'] = pd.to_datetime(
                df_sorted['tourney_date'], format='%Y%m%d', errors='coerce'
            )
        except Exception:
            pass
        df_sorted = df_sorted.sort_values('tourney_date').reset_index(drop=True)

        ratings = {}
        surface_ratings = {}  # key: "player|surface" internally, stored as tuple key in memory
        _surface_dict = {}    # (player, surface) -> float  (used during computation)

        total = len(df_sorted)

        for i, row in df_sorted.iterrows():
            w_name = row.get('winner_name')
            l_name = row.get('loser_name')
            if pd.isna(w_name) or pd.isna(l_name):
                continue

            surface = row.get('surface', 'Hard')
            if pd.isna(surface):
                surface = 'Hard'

            level = row.get('tourney_level', 'A')
            if pd.isna(level) or level not in K_FACTORS:
                level = 'A'

            # Overall ELO update
            k_overall = self._get_k(level, surface=False)
            self._update_ratings(w_name, l_name, k_overall, ratings)

            # Surface ELO update (separate dict keyed by (name, surface))
            k_surf = self._get_k(level, surface=True)
            w_key = (w_name, surface)
            l_key = (l_name, surface)
            r_w = _surface_dict.get(w_key, INITIAL_RATING)
            r_l = _surface_dict.get(l_key, INITIAL_RATING)
            exp_w = self._expected_score(r_w, r_l)
            exp_l = 1.0 - exp_w
            _surface_dict[w_key] = r_w + k_surf * (1.0 - exp_w)
            _surface_dict[l_key] = r_l + k_surf * (0.0 - exp_l)

            if i % 100000 == 0 and i > 0:
                print(f"    Processed {i:,} / {total:,} matches...")

        self.ratings = ratings
        self.surface_ratings = _surface_dict
        self._is_computed = True
        print(f"  ELO computation complete. {len(ratings):,} players rated.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, row_count=None):
        """Serialise ratings to JSON cache file."""
        os.makedirs(os.path.dirname(self.cache_path) if os.path.dirname(self.cache_path) else '.', exist_ok=True)

        surface_serialisable = {
            f"{player}|{surface}": rating
            for (player, surface), rating in self.surface_ratings.items()
        }

        cache = {
            'metadata': {'row_count': row_count or 0},
            'overall': self.ratings,
            'surface': surface_serialisable,
        }

        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)

        print(f"  ELO ratings saved to {self.cache_path}")

    def load(self):
        """
        Load ratings from JSON cache.
        Returns (success: bool, row_count: int).
        """
        if not os.path.exists(self.cache_path):
            return False, 0

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)

            self.ratings = cache.get('overall', {})
            row_count = cache.get('metadata', {}).get('row_count', 0)

            # Deserialise surface keys from "player|surface" strings back to tuples
            self.surface_ratings = {}
            for key_str, rating in cache.get('surface', {}).items():
                parts = key_str.rsplit('|', 1)
                if len(parts) == 2:
                    self.surface_ratings[(parts[0], parts[1])] = rating

            self._is_computed = True
            return True, row_count
        except Exception as e:
            print(f"  Warning: Could not load ELO cache ({e}). Will recompute.")
            return False, 0

    def get_or_compute(self, df):
        """
        Load ratings from cache if valid; otherwise compute and save.
        Staleness check: recompute if row count has changed.
        """
        loaded, cached_row_count = self.load()

        if loaded and cached_row_count == len(df):
            print(f"  ELO ratings loaded from cache ({len(self.ratings):,} players).")
            return

        # Cache missing or stale
        self.compute_from_dataframe(df)
        self.save(row_count=len(df))

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_rating(self, player_name):
        """Return overall ELO rating, defaulting to INITIAL_RATING for unknown players."""
        return self.ratings.get(player_name, INITIAL_RATING)

    def get_surface_rating(self, player_name, surface):
        """Return surface-specific ELO, defaulting to overall ELO (or 1500) if unavailable."""
        surface_elo = self.surface_ratings.get((player_name, surface))
        if surface_elo is not None:
            return surface_elo
        # Fall back to overall ELO if player has no surface history
        return self.ratings.get(player_name, INITIAL_RATING)

    def predict_winner(self, player1_name, player2_name, surface=None):
        """
        Predict match winner using ELO ratings.
        Returns dict matching the existing predictions ensemble format:
            {'probability': float, 'predicted_winner': str, 'confidence': float}

        If surface is provided, blends overall and surface-specific ELO (50/50).
        """
        overall_p1 = self.get_rating(player1_name)
        overall_p2 = self.get_rating(player2_name)
        overall_prob = self._expected_score(overall_p1, overall_p2)

        if surface:
            surf_p1 = self.get_surface_rating(player1_name, surface)
            surf_p2 = self.get_surface_rating(player2_name, surface)
            surf_prob = self._expected_score(surf_p1, surf_p2)
            probability = 0.5 * overall_prob + 0.5 * surf_prob
        else:
            probability = overall_prob

        predicted_winner = player1_name if probability >= 0.5 else player2_name
        confidence = abs(probability - 0.5) * 2.0

        return {
            'probability': float(probability),
            'predicted_winner': predicted_winner,
            'confidence': float(confidence),
        }


if __name__ == '__main__':
    from data_loader import load_data

    print("Testing ELO calculator...")
    df = load_data()
    print(f"Loaded {len(df):,} matches.")

    calc = ELOCalculator()
    calc.get_or_compute(df)

    # Sanity checks
    players_to_check = [
        'Roger Federer', 'Rafael Nadal', 'Novak Djokovic',
        'Andy Murray', 'Pete Sampras', 'Andre Agassi'
    ]

    print("\nOverall ELO ratings:")
    for p in players_to_check:
        r = calc.get_rating(p)
        print(f"  {p}: {r:.1f}")

    print("\nClay ELO ratings:")
    for p in players_to_check:
        r = calc.get_surface_rating(p, 'Clay')
        print(f"  {p}: {r:.1f}")

    print("\nNadal vs Federer on Clay:")
    pred = calc.predict_winner('Rafael Nadal', 'Roger Federer', surface='Clay')
    print(f"  {pred}")
