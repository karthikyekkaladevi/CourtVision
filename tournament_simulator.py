"""
Tournament simulation module for tennis tournament predictor.
Simulates tournament brackets using match predictions.
"""
import pandas as pd
import numpy as np
from predictor import MatchPredictor
from data_loader import load_data
import warnings
warnings.filterwarnings('ignore')


import difflib

class TournamentSimulator:
    """Class to simulate tennis tournaments."""
    
    def __init__(self, models_dir='models'):
        self.predictor = MatchPredictor(models_dir)
        self.df_historical = None
        self._all_players_cache = None
        self._matchup_cache = {} # Cache for Monte Carlo simulations
    
    def load_historical_data(self, data_path='tennis_atp-master', include_types=['tour']):
        """Load historical match data."""
        self.df_historical = load_data(data_path, include_types=include_types)
        # Cache unique player names for fuzzy matching
        all_winners = self.df_historical['winner_name'].dropna().unique()
        all_losers = self.df_historical['loser_name'].dropna().unique()
        self._all_players_cache = sorted(list(set(all_winners) | set(all_losers)))

    def validate_player_name(self, name):
        """
        Check if a player name exists in the dataset.
        Handles First Initial + Last Name formatting (e.g. 'C Alcaraz' or 'C. Alcaraz').
        Returns (True, "Full Name") if unique.
        Returns (False, ["Match 1", "Match 2"]) if ambiguous.
        Returns (False, []) if no matches found.
        """
        if self.df_historical is None:
            self.load_historical_data()
            
        # 1. Exact match check
        if name in self._all_players_cache:
            return True, name
            
        # 2. Check for "First Initial + Last Name" (e.g., "A Kuznetsov" or "A. Kuznetsov")
        import re
        abbrev_match = re.match(r'^([A-Za-z])\.?\s+(.+)$', name.strip())
        if abbrev_match:
            initial = abbrev_match.group(1).upper()
            target_last_name = abbrev_match.group(2).strip().lower()
            
            # Find all players that match this initial and last name
            matching_players = []
            for p in self._all_players_cache:
                parts = p.split()
                if len(parts) >= 2:
                    p_initial = parts[0][0].upper()
                    # Join the rest as last name in case of multi-word last names
                    p_last_name = ' '.join(parts[1:]).strip().lower()
                    
                    if p_initial == initial and target_last_name == p_last_name:
                        matching_players.append(p)
            
            if len(matching_players) == 1:
                return True, matching_players[0]
            elif len(matching_players) > 1:
                return False, sorted(matching_players)

        # 3. Try fuzzy matching as a fallback
        matches = difflib.get_close_matches(name, self._all_players_cache, n=3, cutoff=0.7)
        if matches:
            return False, matches
            
        return False, []

    def get_player_career_range(self, name):
        """
        Get the first and last year of a player's ATP matches in the dataset.
        Returns (first_year, last_year) or (None, None).
        """
        if self.df_historical is None:
            self.load_historical_data()
            
        p_matches = self.df_historical[(self.df_historical['winner_name'] == name) | 
                                       (self.df_historical['loser_name'] == name)]
        
        if p_matches.empty:
            return None, None
            
        dates = p_matches['tourney_date'].dropna()
        if dates.empty:
            return None, None
            
        first_date = dates.min()
        last_date = dates.max()
        
        def to_year(d):
            try:
                if isinstance(d, (int, float, np.integer, np.floating)):
                    return int(d) // 10000
                return int(str(d)[:4])
            except:
                return None
                
        return to_year(first_date), to_year(last_date)

    def get_player_debut_year(self, name):
        """Helper for Late-Pro Cutoff, returns just the first year."""
        first, _ = self.get_player_career_range(name)
        return first

    def get_top_players(self, year, count=32):
        """
        Get top ranked players for a specific year based on match data.
        Infers rankings from the latest matches played in that year.
        
        Args:
            year: Year to fetch players from (int)
            count: Number of players to return
            
        Returns:
            List of (seed, name) tuples
        """
        if self.df_historical is None:
            self.load_historical_data()
            
        df = self.df_historical.copy()
        
        # Filter for the year
        # dates are YYYYMMDD (numeric or string)
        try:
            if pd.api.types.is_numeric_dtype(df['tourney_date']):
                df_year = df[(df['tourney_date'] // 10000) == year]
            else:
                str_dates = df['tourney_date'].astype(str)
                df_year = df[str_dates.str.startswith(str(year))]
        except Exception as e:
            print(f"Error filtering by year {year}: {e}")
            return []
            
        if df_year.empty:
            print(f"No matches found for year {year}")
            return []
            
        # Extract rankings
        # We need the LATEST ranking for each player in this year
        player_ranks = {} # name -> (date, rank)
        
        def update_rank(name, rank, date):
            if pd.isna(rank) or rank == 0:
                return
            
            if name not in player_ranks:
                player_ranks[name] = (date, rank)
            else:
                # Update if date is newer
                stored_date, _ = player_ranks[name]
                if date >= stored_date:
                    player_ranks[name] = (date, rank)
        
        for _, row in df_year.iterrows():
            date = row['tourney_date']
            # Winner
            update_rank(row['winner_name'], row.get('winner_rank'), date)
            # Loser
            update_rank(row['loser_name'], row.get('loser_rank'), date)
            
        # Convert to list and sort
        ranked_players = []
        for name, (_, rank) in player_ranks.items():
            ranked_players.append((rank, name))
            
        # Sort by rank (ascending)
        ranked_players.sort(key=lambda x: x[0])
        
        # Take top N
        top_players = ranked_players[:count]
        
        # Format as (seed, name) matches the list index + 1 or actual rank?
        # Usually for a draw we want seeds 1..N.
        # Let's return (actual_rank, name) but maybe re-seed them 1..N for the draw?
        # The prompt implies "Top N Players". 
        # Let's return them as they are, main.py can handle seeding 1..N if needed.
        # Actually `create_bracket` handles sorting.
        # Let's Re-seed them 1 to N for the tournament purpose
        reseeded_players = [(i+1, p[1]) for i, p in enumerate(top_players)]
        
        return reseeded_players
    
    def simulate_match(self, p1_info, p2_info, surface=None, tourney_level=None, use_model='average', fast_mode=False):
        """
        Simulate a single match.
        
        Args:
            p1_info: Name (str) or dict with 'name' and optional 'year'
            p2_info: Name (str) or dict with 'name' and optional 'year'
            surface: Surface type
            tourney_level: Tournament level
            use_model: Which model to use ('average', 'xgboost', 'random_forest', etc.)
            
        Returns:
            Dictionary with 'winner', 'score', and 'predictions'
        """
        if self.df_historical is None:
            self.load_historical_data()

        # Normalize inputs
        if isinstance(p1_info, str): p1_info = {'name': p1_info, 'year': None}
        if isinstance(p2_info, str): p2_info = {'name': p2_info, 'year': None}
        
        p1_name = p1_info['name']
        p2_name = p2_info['name']
        p1_year = p1_info.get('year')
        p2_year = p2_info.get('year')
        
        # Check cache if in fast mode
        cache_key = tuple(sorted([
            f"{p1_name}_{p1_year}", 
            f"{p2_name}_{p2_year}"
        ]) + [surface, tourney_level, use_model])
        
        if fast_mode and cache_key in self._matchup_cache:
            predictions = self._matchup_cache[cache_key]
        else:
            predictions = self.predictor.predict_from_names(
                p1_name, p2_name, self.df_historical, surface, tourney_level,
                player1_year=p1_year, player2_year=p2_year
            )
            if fast_mode: self._matchup_cache[cache_key] = predictions
        
        if not predictions:
            # Fallback: random choice
            winner_info = p1_info if np.random.random() < 0.5 else p2_info
            return {'winner': winner_info, 'score': '0-0, 0-0', 'predictions': None}
        
        # Determine average probability for winner determination and score simulation
        avg_prob = np.mean([p['probability'] for k, p in predictions.items() if not k.startswith('_')])
        
        # Decide winner
        if use_model == 'average':
            is_p1_winner = avg_prob >= 0.5
        else:
            if use_model in predictions:
                prob = predictions[use_model]['probability']
                is_p1_winner = prob >= 0.5
            else:
                is_p1_winner = avg_prob >= 0.5
        
        winner_info = p1_info if is_p1_winner else p2_info
        loser_info = p2_info if is_p1_winner else p1_info
        
        win_prob = avg_prob if is_p1_winner else (1 - avg_prob)
        p1_stats = predictions.get('_player1_stats')
        p2_stats = predictions.get('_player2_stats')
        
        # Simulate score (skip in fast_mode to save time)
        if fast_mode:
            score = "0-0, 0-0"
        else:
            sets_to_play = 5 if tourney_level == 'G' else 3
            score = self.predictor.simulate_score(winner_info['name'], loser_info['name'], win_prob, sets_to_play, p1_stats, p2_stats)
        
        # Add "Monte Carlo" to predictions (mini-sample)
        if not fast_mode:
            predictions['Monte Carlo'] = {'probability': win_prob if is_p1_winner else (1 - win_prob)}

        return {
            'winner': winner_info,
            'score': score,
            'predictions': predictions
        }
    
    def _get_seed_distribution(self, n):
        """
        Generate standard tournament seed distribution recursively.
        Example for n=8: [1, 8, 5, 4, 3, 6, 7, 2]
        """
        if n == 1:
            return [1]
        if n == 2:
            return [1, 2]
            
        prev = self._get_seed_distribution(n // 2)
        res = []
        for i, x in enumerate(prev):
            if i % 2 == 0:
                res.extend([x, n + 1 - x])
            else:
                res.extend([n + 1 - x, x])
        return res

    def create_bracket(self, players, draw_size=None):
        """
        Create tournament bracket from list of players.
        
        Args:
            players: List of player info dicts {'name', 'year', 'seed'}
            draw_size: Size of draw (32, 64, 128). If None, uses closest power of 2 >= len(players)
            
        Returns:
            Bracket structure as nested list
        """
        if draw_size is None:
            # Find next power of 2
            draw_size = 2 ** int(np.ceil(np.log2(len(players))))
        
        # Ensure players are dicts and have seeds
        normalized_players = []
        existing_seeds = [p.get('seed') for p in players if isinstance(p, dict) and p.get('seed') is not None]
        next_seed = 1
        
        for p in players:
            if isinstance(p, dict):
                p_info = p.copy()
            elif isinstance(p, tuple):
                # (seed, name)
                p_info = {'seed': p[0], 'name': p[1], 'year': None}
            else:
                p_info = {'seed': None, 'name': p, 'year': None}
            
            if p_info.get('seed') is None:
                while next_seed in existing_seeds:
                    next_seed += 1
                p_info['seed'] = next_seed
                existing_seeds.append(next_seed)
                next_seed += 1
            
            normalized_players.append(p_info)
        
        # Add BYEs until draw_size, assigning them high seeds
        while len(normalized_players) < draw_size:
            seed = len(normalized_players) + 1
            normalized_players.append({'name': 'BYE', 'year': None, 'seed': seed})
        
        # Sort players by seed to easily map them to the distribution
        normalized_players.sort(key=lambda x: x['seed'])
        
        # Map seeds to names for quick lookup
        seed_to_player = {p['seed']: p for p in normalized_players}
        
        # Get standard seed distribution
        distribution = self._get_seed_distribution(draw_size)
        
        # Create bracket matches based on the distribution
        bracket = []
        for i in range(0, len(distribution), 2):
            p1_seed = distribution[i]
            p2_seed = distribution[i+1]
            bracket.append([seed_to_player[p1_seed], seed_to_player[p2_seed]])
        
        return bracket
    
    def simulate_round(self, bracket, round_name, surface=None, tourney_level=None, use_model='average', show_details=False, silent=False, fast_mode=False):
        """
        Simulate a single round of matches.
        
        Args:
            bracket: Current bracket (list of matches)
            round_name: Name of the round
            surface: Surface type
            tourney_level: Tournament level
            use_model: Model to use for predictions
            show_details: Whether to show full match stats
            silent: Whether to suppress output
            
        Returns:
            List of dictionaries with match results
        """
        results = []
        
        for i, match in enumerate(bracket):
            if len(match) == 2:
                p1_info, p2_info = match
                
                # Handle byes
                if p1_info['name'] == 'BYE':
                    results.append({'winner': p2_info, 'score': 'BYE', 'predictions': None, 'player1': p1_info, 'player2': p2_info})
                elif p2_info['name'] == 'BYE':
                    results.append({'winner': p1_info, 'score': 'BYE', 'predictions': None, 'player1': p1_info, 'player2': p2_info})
                else:
                    if show_details:
                        print(f"\n--- {round_name} | Match {i+1}: {p1_info['name']} vs {p2_info['name']} ---")
                    
                    match_res = self.simulate_match(p1_info, p2_info, surface, tourney_level, use_model, fast_mode=fast_mode)
                    
                    if show_details and match_res['predictions']:
                        # predictor.display_predictions takes names
                        sets_to_play = 5 if tourney_level == 'G' else 3
                        self.predictor.display_predictions(match_res['predictions'], p1_info['name'], p2_info['name'], sets_to_play=sets_to_play)
                        print(f"\nResult: {match_res['winner']['name']} wins {match_res['score']}")
                    
                    match_res.update({'player1': p1_info, 'player2': p2_info})
                    results.append(match_res)
            else:
                # Single player (bye or already determined)
                p = match[0] if isinstance(match, list) else match
                results.append({'winner': p, 'score': 'BYE', 'predictions': None, 'player1': p, 'player2': None})
        
        return results
    
    def simulate_tournament(self, players, surface=None, tourney_level=None, use_model='average', draw_size=None, show_details=False, silent=False, fast_mode=False):
        """
        Simulate entire tournament.
        
        Args:
            players: List of player names or list of tuples (seed, name)
            surface: Surface type
            tourney_level: Tournament level
            use_model: Model to use for predictions
            draw_size: Size of draw
            show_details: Whether to show full match stats
            silent: Whether to suppress output
            
        Returns:
            Dictionary with tournament results
        """
        if tourney_level == 'F' and len(players) == 8:
            return self.simulate_atp_finals(players, surface, use_model, show_details, silent, fast_mode)
            
        if self.df_historical is None:
            self.load_historical_data()
        
        # Create bracket
        bracket = self.create_bracket(players, draw_size)
        
        # Round names
        round_names = {
            2: 'Final',
            4: 'Semi-Finals',
            8: 'Quarter-Finals',
            16: 'Round of 16',
            32: 'Round of 32',
            64: 'Round of 64',
            128: 'Round of 128'
        }
        
        results = {
            'bracket_history': [],
            'champion': None,
            'runner_up': None,
            'semi_finalists': [],
            'quarter_finalists': []
        }
        
        current_round = bracket
        
        if not silent:
            print("\n" + "="*60)
            print("TOURNAMENT SIMULATION")
            print("="*60)
        
        while True:
            round_num = len(current_round)
            display_num = round_num * 2
            round_name = round_names.get(display_num, f'Round of {display_num}')
            
            if not show_details and not silent:
                print(f"\n{round_name}:")
                print("-" * 60)
                
                # Display matches
                for i, match in enumerate(current_round):
                    if len(match) == 2:
                        p1, p2 = match
                        if p1['name'] != 'BYE' and p2['name'] != 'BYE':
                            print(f"  Match {i+1}: {p1['name']} vs {p2['name']}")
            
            # Simulate round
            round_results = self.simulate_round(
                current_round, round_name, surface, tourney_level, 
                use_model, show_details, silent=silent, fast_mode=fast_mode
            )
            
            # Extract winners for next round
            winners = [r['winner'] for r in round_results]
            
            if not show_details and not silent:
                # Display winners and scores
                print(f"\nWinners:")
                for r in round_results:
                    if r['winner']['name'] != 'BYE' and r['player2'] and r['player2']['name'] != 'BYE':
                        loser_name = r['player1']['name'] if r['winner']['name'] != r['player1']['name'] else r['player2']['name']
                        print(f"  {r['winner']['name']} def. {loser_name} {r['score']}")
                    elif r['winner']['name'] != 'BYE':
                        print(f"  {r['winner']['name']} (BYE)")
            
            # Store results
            results['bracket_history'].append({
                'round': round_name,
                'matches': current_round,
                'results': round_results
            })
            
            # Track semi-finalists and quarter-finalists
            if display_num == 4:
                # Players who STARTED the semi-final round
                results['semi_finalists'] = [p for match in current_round for p in match if p['name'] != 'BYE']
            elif display_num == 8:
                results['quarter_finalists'] = [p for match in current_round for p in match if p['name'] != 'BYE']
            
            # Check if tournament is over
            if len(winners) == 1:
                results['champion'] = winners[0]['name']
                # Runner up is the person who lost in the final
                last_match = round_results[0]
                results['runner_up'] = last_match['player1']['name'] if results['champion'] != last_match['player1']['name'] else last_match['player2']['name']
                break
            
            # Prepare next round
            current_round = [[winners[i], winners[i+1]] for i in range(0, len(winners), 2)]
        
        if not silent:
            print("\n" + "="*60)
            print(f"CHAMPION: {results['champion']}")
            if results['runner_up']:
                print(f"Runner-Up: {results['runner_up']}")
            print("="*60 + "\n")
        
        return results

    def simulate_atp_finals(self, players, surface='Hard', use_model='average', show_details=False, silent=False, fast_mode=False):
        """
        Simulate ATP Finals format (8 players, 2 groups of 4, round robin -> semi finals).
        """
        if self.df_historical is None:
            self.load_historical_data()
            
        # Normalize players into dicts with seed
        normalized = []
        for i, p in enumerate(players):
            if isinstance(p, dict):
                normalized.append(p.copy())
            elif isinstance(p, tuple):
                normalized.append({'seed': p[0], 'name': p[1], 'year': None})
            else:
                normalized.append({'seed': i+1, 'name': p, 'year': None})
        
        # Ensure seeds exist and sort
        for i, p in enumerate(normalized):
            if p.get('seed') is None:
                p['seed'] = i + 1
        normalized.sort(key=lambda x: x['seed'])
            
        # Group A (seeds 1, 4, 5, 8) -> indices 0, 3, 4, 7
        group_a = [normalized[i] for i in [0, 3, 4, 7]]
        # Group B (seeds 2, 3, 6, 7) -> indices 1, 2, 5, 6
        group_b = [normalized[i] for i in [1, 2, 5, 6]]
        
        if not silent:
            print("\n" + "="*60)
            print("ATP FINALS SIMULATION")
            print("="*60)
            print("Group A:", ", ".join(p['name'] for p in group_a))
            print("Group B:", ", ".join(p['name'] for p in group_b))
        
        def simulate_group(group, group_name):
            matches = []
            # round robin pairs: (0,1), (2,3) | (0,2), (1,3) | (0,3), (1,2)
            pairs = [(0,1), (2,3), (0,2), (1,3), (0,3), (1,2)]
            
            for i1, i2 in pairs:
                p1, p2 = group[i1], group[i2]
                res = self.simulate_match(p1, p2, surface, 'F', use_model, fast_mode=fast_mode)
                res['player1'] = p1
                res['player2'] = p2
                matches.append(res)
                if show_details and not silent:
                    print(f"[{group_name}] {res['winner']['name']} def. {p1['name'] if res['winner']['name'] != p1['name'] else p2['name']} {res['score']}")
            return matches
            
        matches_a = simulate_group(group_a, "Group A")
        matches_b = simulate_group(group_b, "Group B")
        
        # Calculate standings
        def get_standings(group, matches):
            import functools
            stats = {p['name']: {'player': p, 'wins': 0, 'losses': 0} for p in group}
            h2h = {}
            for m in matches:
                w_name = m['winner']['name']
                l_name = m['player1']['name'] if w_name != m['player1']['name'] else m['player2']['name']
                stats[w_name]['wins'] += 1
                stats[l_name]['losses'] += 1
                if w_name not in h2h: h2h[w_name] = {}
                h2h[w_name][l_name] = True
                
            def compare(n1, n2):
                if stats[n1]['wins'] != stats[n2]['wins']:
                    return stats[n2]['wins'] - stats[n1]['wins']
                if n2 in h2h.get(n1, {}): return -1
                if n1 in h2h.get(n2, {}): return 1
                return 0
                
            sorted_names = sorted(stats.keys(), key=functools.cmp_to_key(compare))
            return [stats[n] for n in sorted_names]
            
        standings_a = get_standings(group_a, matches_a)
        standings_b = get_standings(group_b, matches_b)
        
        # Knockout Stage
        sf1_p1 = standings_a[0]['player'] # A1
        sf1_p2 = standings_b[1]['player'] # B2
        sf2_p1 = standings_b[0]['player'] # B1
        sf2_p2 = standings_a[1]['player'] # A2
        
        if not silent:
            print(f"\nSemi-Final 1: {sf1_p1['name']} vs {sf1_p2['name']}")
            print(f"Semi-Final 2: {sf2_p1['name']} vs {sf2_p2['name']}")
            
        sf1_res = self.simulate_match(sf1_p1, sf1_p2, surface, 'F', use_model, fast_mode=fast_mode)
        sf1_res.update({'player1': sf1_p1, 'player2': sf1_p2})
        sf2_res = self.simulate_match(sf2_p1, sf2_p2, surface, 'F', use_model, fast_mode=fast_mode)
        sf2_res.update({'player1': sf2_p1, 'player2': sf2_p2})
        
        sf_winners = [sf1_res['winner'], sf2_res['winner']]
        
        if not silent:
            print(f"Final: {sf_winners[0]['name']} vs {sf_winners[1]['name']}")
            
        f_res = self.simulate_match(sf_winners[0], sf_winners[1], surface, 'F', use_model, fast_mode=fast_mode)
        f_res.update({'player1': sf_winners[0], 'player2': sf_winners[1]})
        
        champion = f_res['winner']
        runner_up = sf_winners[0] if champion['name'] != sf_winners[0]['name'] else sf_winners[1]
        
        if not silent:
            print(f"CHAMPION: {champion['name']} def. {runner_up['name']} {f_res['score']}\n")
            
        history = [
            {'round': 'Semi-Finals', 'results': [sf1_res, sf2_res]},
            {'round': 'Final', 'results': [f_res]}
        ]
        
        return {
            'type': 'atp_finals',
            'group_a': standings_a,
            'group_b': standings_b,
            'rr_matches': matches_a + matches_b,
            'bracket_history': history,
            'champion': champion['name'],
            'runner_up': runner_up['name']
        }

    def simulate_tournament_monte_carlo(self, players, iterations=1000, surface=None, tourney_level=None, use_model='average', draw_size=None):
        """
        Run Monte Carlo simulations of the tournament.
        
        Args:
            players: List of player names or list of tuples (seed, name)
            iterations: Number of simulations to run
            surface: Surface type
            tourney_level: Tournament level
            use_model: Model to use for predictions
            draw_size: Size of draw
            
        Returns:
            Dictionary with win probabilities and iteration performance
        """
        if self.df_historical is None:
            self.load_historical_data()
            
        win_counts = {}
        runner_up_counts = {}
        semi_counts = {}
        
        print(f"\nRunning {iterations} Monte Carlo simulations...")
        
        # Pre-bracket players to ensure consistency across iterations
        # Actually, simulate_tournament creates a fresh bracket each time, 
        # but if the seeds/names are the same, the bracket structure is the same.
        # This allows for the "randomness" of each individual match outcome to play out.
        
        try:
            from tqdm import tqdm
            pbar = tqdm(range(iterations))
        except ImportError:
            # Simple fallback progress bar
            def simple_pbar(it):
                count = len(it)
                for i, val in enumerate(it):
                    if i % (max(1, count // 20)) == 0:
                        progress = (i / count) * 100
                        bar = '#' * int(progress // 5)
                        print(f"  Progress: [{bar:20s}] {progress:.0f}%", end='\r')
                    yield val
                print(f"  Progress: [{'#' * 20}] 100% ")
            pbar = simple_pbar(range(iterations))
            print("  (Simulating... please wait)")
            
        # Reset cache for this fresh MC run
        self._matchup_cache = {}
        
        for _ in pbar:
            res = self.simulate_tournament(
                players, surface=surface, tourney_level=tourney_level, 
                use_model=use_model, draw_size=draw_size, 
                show_details=False, silent=True, fast_mode=True
            )
            
            champion = res['champion']
            win_counts[champion] = win_counts.get(champion, 0) + 1
            
            runner_up = res['runner_up']
            if runner_up:
                runner_up_counts[runner_up] = runner_up_counts.get(runner_up, 0) + 1
                
            for sf in res['semi_finalists']:
                semi_counts[sf['name']] = semi_counts.get(sf['name'], 0) + 1
        
        # Calculate probabilities
        probabilities = []
        
        # Include ALL players who were in the tournament
        participating_names = []
        for p in players:
            if isinstance(p, dict): participating_names.append(p['name'])
            elif isinstance(p, tuple): participating_names.append(p[1])
            else: participating_names.append(p)

        for name in sorted(list(set(participating_names))):
            if name == 'BYE' or name is None: continue
            probabilities.append({
                'name': name,
                'win_prob': win_counts.get(name, 0) / iterations,
                'final_prob': (win_counts.get(name, 0) + runner_up_counts.get(name, 0)) / iterations,
                'semi_prob': semi_counts.get(name, 0) / iterations
            })
            
        # Sort by win probability
        return {
            'iterations': iterations,
            'probabilities': probabilities
        }

    def display_forecast_dashboard(self, mc_results):
        """DEPRECATED: Removed at user request."""
        pass

    def display_bracket(self, results):
        """DEPRECATED: Removed at user request."""
        pass


if __name__ == "__main__":
    # Test tournament simulator
    print("Testing tournament simulator...")
    
    simulator = TournamentSimulator()
    
    # Example tournament
    players = [
        (1, "Roger Federer"),
        (2, "Rafael Nadal"),
        (3, "Novak Djokovic"),
        (4, "Andy Murray"),
        (5, "Stan Wawrinka"),
        (6, "Kei Nishikori"),
        (7, "Marin Cilic"),
        (8, "Milos Raonic")
    ]
    
    results = simulator.simulate_tournament(players, surface='Hard', tourney_level='M', draw_size=8)
