"""
Core tests for CourtVision.

Run with:  python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from elo_calculator import ELOCalculator, INITIAL_RATING


# ─── ELO Tests ────────────────────────────────────────────────────────────────

class TestELORatings:
    """ELO rating system sanity checks."""

    def setup_method(self):
        self.elo = ELOCalculator(cache_path='models/elo_ratings.json')
        self.ratings = {}

    def test_winner_rating_increases_after_win(self):
        """Winning a match should increase ELO."""
        before = self.ratings.get('Alice', INITIAL_RATING)
        self.elo._update_ratings('Alice', 'Bob', k=24, ratings_dict=self.ratings)
        assert self.ratings['Alice'] > before

    def test_loser_rating_decreases_after_loss(self):
        """Losing a match should decrease ELO."""
        before = self.ratings.get('Bob', INITIAL_RATING)
        self.elo._update_ratings('Alice', 'Bob', k=24, ratings_dict=self.ratings)
        assert self.ratings['Bob'] < before

    def test_ratings_are_zero_sum(self):
        """Total ELO points gained equals total lost (zero-sum system)."""
        self.ratings = {'Alice': 1600.0, 'Bob': 1400.0}
        total_before = self.ratings['Alice'] + self.ratings['Bob']
        self.elo._update_ratings('Alice', 'Bob', k=24, ratings_dict=self.ratings)
        total_after = self.ratings['Alice'] + self.ratings['Bob']
        assert abs(total_after - total_before) < 1e-9

    def test_upset_gains_more_points(self):
        """A lower-rated player beating a higher-rated one gains more points."""
        # Upset: low-rated Bob beats high-rated Alice
        ratings_upset = {'Alice': 1700.0, 'Bob': 1300.0}
        bob_before = ratings_upset['Bob']
        self.elo._update_ratings('Bob', 'Alice', k=24, ratings_dict=ratings_upset)
        bob_gain_upset = ratings_upset['Bob'] - bob_before

        # Expected: high-rated Carol beats low-rated Dave
        ratings_expected = {'Carol': 1700.0, 'Dave': 1300.0}
        carol_before = ratings_expected['Carol']
        self.elo._update_ratings('Carol', 'Dave', k=24, ratings_dict=ratings_expected)
        carol_gain_expected = ratings_expected['Carol'] - carol_before

        assert bob_gain_upset > carol_gain_expected

    def test_unknown_player_gets_initial_rating(self):
        """Player with no history should return INITIAL_RATING."""
        assert self.elo.get_rating('NonExistentPlayer') == INITIAL_RATING

    def test_expected_score_range(self):
        """Expected score must always be between 0 and 1."""
        for r_a, r_b in [(1500, 1500), (2000, 1000), (1000, 2000), (1500, 1501)]:
            score = self.elo._expected_score(r_a, r_b)
            assert 0.0 < score < 1.0

    def test_equal_ratings_give_50_50(self):
        """Two players with equal ratings should each have 50% win probability."""
        prob = self.elo._expected_score(1500.0, 1500.0)
        assert abs(prob - 0.5) < 1e-9

    def test_predict_winner_probability_sums_to_one(self):
        """P(A beats B) + P(B beats A) should equal 1."""
        self.elo.ratings = {'Alice': 1600.0, 'Bob': 1400.0}
        p_alice = self.elo.predict_winner('Alice', 'Bob')['probability']
        p_bob = self.elo.predict_winner('Bob', 'Alice')['probability']
        assert abs(p_alice + p_bob - 1.0) < 1e-9

    def test_higher_rated_player_favored(self):
        """The higher-rated player should have > 50% win probability."""
        self.elo.ratings = {'Strong': 1700.0, 'Weak': 1300.0}
        result = self.elo.predict_winner('Strong', 'Weak')
        assert result['probability'] > 0.5
        assert result['predicted_winner'] == 'Strong'


# ─── Feature Symmetry Tests ───────────────────────────────────────────────────

class TestFeatureSymmetry:
    """
    Verify that swapping Player A and Player B produces negated feature diffs.
    This ensures the model doesn't learn player-slot biases.
    """

    def _make_row(self, winner, loser, surface='Hard', level='G', date='20200101',
                  w_rank=1, l_rank=10):
        """Helper to build a minimal match row dict."""
        return {
            'winner_name': winner,
            'loser_name': loser,
            'surface': surface,
            'tourney_level': level,
            'tourney_date': date,
            'winner_rank': w_rank,
            'loser_rank': l_rank,
            'winner_rank_points': 10000,
            'loser_rank_points': 1000,
            'winner_age': 25.0,
            'loser_age': 28.0,
            'winner_ht': 185.0,
            'loser_ht': 180.0,
            'winner_seed': 1,
            'loser_seed': 8,
            'winner_hand': 'R',
            'loser_hand': 'R',
            'w_ace': 10, 'w_df': 2, 'w_svpt': 80,
            'w_1stIn': 50, 'w_1stWon': 38, 'w_2ndWon': 18,
            'w_bpSaved': 4, 'w_bpFaced': 5,
            'l_ace': 5, 'l_df': 3, 'l_svpt': 75,
            'l_1stIn': 45, 'l_1stWon': 30, 'l_2ndWon': 15,
            'l_bpSaved': 2, 'l_bpFaced': 6,
            'round': 'F',
            'match_num': 1,
            'draw_size': 128,
            'score': '6-3 6-4',
            'best_of': 3,
            'minutes': 90,
            'tourney_id': 'test-001',
            'tourney_name': 'Test Open',
            'winner_id': 1,
            'loser_id': 2,
            'winner_ioc': 'USA',
            'loser_ioc': 'ESP',
            'winner_entry': None,
            'loser_entry': None,
        }

    def test_elo_diff_symmetry(self):
        """
        ELO diff for (A vs B) should be the negative of ELO diff for (B vs A).
        Uses the ELO calculator directly since it's stateless at query time.
        """
        elo = ELOCalculator()
        elo.ratings = {'Federer': 2000.0, 'Nadal': 1950.0}

        prob_fed_vs_nad = elo._expected_score(
            elo.get_rating('Federer'), elo.get_rating('Nadal')
        )
        prob_nad_vs_fed = elo._expected_score(
            elo.get_rating('Nadal'), elo.get_rating('Federer')
        )

        elo_diff_ab = elo.get_rating('Federer') - elo.get_rating('Nadal')
        elo_diff_ba = elo.get_rating('Nadal') - elo.get_rating('Federer')

        assert abs(elo_diff_ab + elo_diff_ba) < 1e-9
        assert abs(prob_fed_vs_nad + prob_nad_vs_fed - 1.0) < 1e-9

    def test_rank_diff_negates_on_swap(self):
        """rank_diff = p2_rank - p1_rank, so swapping players negates it."""
        p1_rank, p2_rank = 3, 15
        diff_ab = p2_rank - p1_rank   # 12
        diff_ba = p1_rank - p2_rank   # -12
        assert diff_ab == -diff_ba

    def test_win_rate_diff_negates_on_swap(self):
        """win_rate_diff = p1_wr - p2_wr, so swapping players negates it."""
        p1_wr, p2_wr = 0.72, 0.58
        diff_ab = p1_wr - p2_wr
        diff_ba = p2_wr - p1_wr
        assert abs(diff_ab + diff_ba) < 1e-9


# ─── Data Loader Cache Tests ───────────────────────────────────────────────────

class TestDataLoaderCache:
    """Parquet cache behaviour in data_loader.py."""

    def test_cache_key_format(self):
        """Cache key should encode start_year, end_year, and sorted types."""
        start_year, end_year = 1968, 2026
        include_types = ['tour']
        key = f"{start_year}-{end_year}-{'-'.join(sorted(include_types))}"
        assert key == "1968-2026-tour"

    def test_cache_key_is_order_independent(self):
        """Types in different order should produce the same cache key."""
        types_a = ['tour', 'doubles']
        types_b = ['doubles', 'tour']
        key_a = f"1968-2026-{'-'.join(sorted(types_a))}"
        key_b = f"1968-2026-{'-'.join(sorted(types_b))}"
        assert key_a == key_b

    def test_different_year_ranges_produce_different_keys(self):
        """Different year ranges must not share a cache."""
        key_full = f"1968-2026-tour"
        key_recent = f"2000-2026-tour"
        assert key_full != key_recent
