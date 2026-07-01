"""
Main entry point for Tennis Tournament Predictor.
Terminal-based interface.
"""
import os
import sys
import pickle
from datetime import datetime
from data_loader import load_data
from predictor import MatchPredictor
from tournament_simulator import TournamentSimulator
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


ATP_500_TOURNAMENTS = {
    "rotterdam", "dubai", "acapulco", "barcelona", "hamburg",
    "washington", "beijing", "tokyo", "vienna", "basel",
    "halle", "queen", "marseille", "rio de janeiro",
}


def get_validated_player_name(simulator, p_name, label="Player"):
    """
    Helper to validate a player name with fuzzy matching and a retry loop.
    Returns the validated name or None if unsuccessful.
    """
    while True:
        if not p_name:
            p_name = input(f"    Enter {label} name: ").strip()
            if not p_name:
                print(f"    [ERR] {label} name is required!")
                continue

        valid, result = simulator.validate_player_name(p_name)
        if valid:
            return result
        else:
            if not result:
                print(f"    [ERR] Could not find '{p_name}' in dataset. Please check spelling.")
                p_name = ""
                continue
            else:
                print(f"    ? Could not find '{p_name}'. Did you mean:")
                for idx, m in enumerate(result):
                    print(f"      {idx+1}. {m}")
                print(f"      {len(result)+1}. Keep '{p_name}' anyway")
                print(f"      {len(result)+2}. Try again")

                choice = input("    Choice: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(result):
                    return result[int(choice)-1]
                elif choice == str(len(result)+1):
                    return p_name
                else:
                    p_name = ""
                    continue


def get_validated_year(simulator, player_name, year_input):
    """
    Validates if a given year is within the player's career range.
    Returns (validated_year, error_message or None).
    """
    if not year_input:
        return None, None

    try:
        year = int(year_input)
    except ValueError:
        return None, f"    [ERR] '{year_input}' is not a valid year."

    first, last = simulator.get_player_career_range(player_name)
    if first is None:
        return year, None

    if year < first:
        return None, f"    [ERR] Year {year} predates {player_name}'s first ATP match in {first}."
    if year > last:
        return None, f"    [ERR] Year {year} is after {player_name}'s last recorded ATP match in {last}."

    return year, None


def print_menu():
    """Print main menu."""
    print("\n" + "="*60)
    print("COURTVISION")
    print("="*60)
    print("1. Predict H2H winner")
    print("2. Simulate tournament")
    print("3. View model performance")
    print("4. Exit")
    print("="*60)


# ─── Option 1: Predict Match ───────────────────────────────

def option_1_predict():
    """Predict match winner."""
    print('\033[2J\033[3J\033[H', end='', flush=True)
    print("\n[1] Predict H2H winner")

    try:
        if not os.path.exists('models') or len(os.listdir('models')) == 0:
            print("\n  No trained models found! Train models first.")
            return

        predictor = MatchPredictor()
        simulator = TournamentSimulator()

        print("\nEnter player information:")

        final_players = []
        for i in range(1, 3):
            p_name = input(f"  Player {i} name: ").strip()
            validated_name = get_validated_player_name(simulator, p_name, label=f"Player {i}")
            final_players.append(validated_name)

        player1_name, player2_name = final_players

        # Optional year filter
        p1_y, p2_y = None, None

        while True:
            career = simulator.get_player_career_range(player1_name)
            y1_in = input(f"  End year for {player1_name} (optional, career: {career[0]}-{career[1]}): ").strip()
            p1_y, err = get_validated_year(simulator, player1_name, y1_in)
            if err:
                print(err)
                continue
            break

        while True:
            career = simulator.get_player_career_range(player2_name)
            y2_in = input(f"  End year for {player2_name} (optional, career: {career[0]}-{career[1]}): ").strip()
            p2_y, err = get_validated_year(simulator, player2_name, y2_in)
            if err:
                print(err)
                continue
            break

        surface = input("  Surface (Hard/Clay/Grass, default Hard): ").strip() or 'Hard'

        while True:
            sets_input = input("  Best of (3/5, default 3): ").strip()
            if not sets_input:
                sets_to_play = 3
                break
            try:
                sets_to_play = int(sets_input)
                if sets_to_play in (3, 5):
                    break
                else:
                    print("    Please enter 3 or 5.")
            except ValueError:
                print("    Invalid input.")

        # Era filtering
        strict_year = False
        if p1_y or p2_y:
            print(f"\n  Era filtering:")
            print(f"  1. Cumulative (career up to year) [Default]")
            print(f"  2. Strict (only that year)")
            era_choice = input("  Choice (1/2): ").strip()
            strict_year = (era_choice == '2')

        print("\nLoading data and predicting...")
        df = load_data()

        predictions = predictor.predict_from_names(
            player1_name, player2_name, df, surface, 'M',
            player1_year=p1_y, player2_year=p2_y, strict_year=strict_year
        )

        if predictions:
            predictor.display_predictions(predictions, player1_name, player2_name, sets_to_play=sets_to_play)
        else:
            print("\n  Could not generate predictions. Players may not be in dataset.")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


# ─── Option 2: Simulate Tournament ─────────────────────────

def option_2_simulate():
    """Simulate tournament."""
    print('\033[2J\033[3J\033[H', end='', flush=True)
    print("\n[2] Simulate tournament")

    try:
        if not os.path.exists('models') or len(os.listdir('models')) == 0:
            print("\n  No trained models found! Train models first.")
            return

        print("\n  a. Custom tournament")
        print("  b. Real life ATP tournament")
        mode = input("  Choice (a/b, default a): ").strip().lower() or 'a'

        if mode == 'b':
            print("\n  a. Historical tournament")
            print("  b. Current / upcoming tournament")
            sub = input("  Choice (a/b, default a): ").strip().lower() or 'a'
            if sub == 'b':
                _simulate_upcoming_tournament()
            else:
                _simulate_historical_tournament()
        else:
            _simulate_custom_tournament()

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


def _get_iterations():
    """Ask user for number of Monte Carlo iterations."""
    iterations_input = input("\n  Number of simulations (default 1000): ").strip()
    if iterations_input.isdigit() and int(iterations_input) > 0:
        return int(iterations_input)
    return 1000


def _display_mc_results(mc_results, tournament_name, surface=None, tourney_level=None, sim_players=None, draw_size=None):
    """Display Monte Carlo simulation results, then replay the best outcome with real scores."""
    best = mc_results['best_result']
    probs = sorted(mc_results['probabilities'], key=lambda p: p['win_prob'], reverse=True)
    iterations = mc_results['iterations']

    print('\033[2J\033[3J\033[H', end='', flush=True)
    print(f"\n{'='*60}")
    print(f"  {tournament_name} — Monte Carlo Results ({iterations} simulations)")
    print(f"{'='*60}")

    # Show win probabilities for top contenders
    print(f"\n  Title Probabilities:")
    for i, p in enumerate(probs[:10]):
        if p['win_prob'] > 0:
            bar = '#' * int(p['win_prob'] * 40)
            print(f"    {p['name']:<30s} {p['win_prob']*100:5.1f}%  {bar}")

    # Re-run one simulation with real scores
    print(f"\n  Generating match scores...")
    simulator = TournamentSimulator()
    replay = simulator.simulate_tournament(
        sim_players, surface=surface, tourney_level=tourney_level,
        use_model='average', draw_size=draw_size,
        show_details=False, silent=True, fast_mode=False
    )

    # Show the replayed result with real scores
    print(f"\n{'='*60}")
    print(f"  Most Likely Outcome:")
    print(f"{'='*60}")

    for round_data in replay.get('bracket_history', []):
        round_name = round_data['round']
        results = round_data['results']
        non_bye = [r for r in results if r['winner']['name'] != 'BYE'
                   and r.get('player2') and r['player2']['name'] != 'BYE']
        if len(non_bye) <= 8:
            print(f"\n  {round_name}:")
            print(f"  {'-'*50}")
            for r in non_bye:
                winner = r['winner']['name']
                loser = r['player1']['name'] if winner != r['player1']['name'] else r['player2']['name']
                score = r.get('score', '')
                print(f"    {winner} def. {loser}  {score}")

    print(f"\n{'*'*60}")
    print(f"  {replay['champion']} wins {tournament_name}!")
    if replay.get('runner_up'):
        print(f"  Runner-up: {replay['runner_up']}")
    print(f"{'*'*60}\n")


def _simulate_custom_tournament():
    """Run a custom tournament with user-specified players."""
    simulator = TournamentSimulator()

    # Draw size
    draw_size_input = input("\n  Draw size (8/16/32/64/128, default 8): ").strip()
    draw_size = int(draw_size_input) if draw_size_input.isdigit() else 8

    surface = input("  Surface (Hard/Clay/Grass, default Hard): ").strip() or 'Hard'

    tourney_level = input("  Tournament level (M=Masters / G=Grand Slam / F=ATP Finals, default M): ").strip().upper() or 'M'

    # Tournament name
    tournament_name = input("  Tournament name (optional): ").strip() or f"{surface} Tournament"

    # Player input method
    print(f"\n  Player input for {draw_size} slots:")
    print("  1. Manual entry (one by one)")
    print("  2. Bulk input (comma-separated)")
    print("  3. Auto-fill (top players from a year)")
    method = input("  Choice (1/2/3, default 1): ").strip() or '1'

    players = []

    if method == '3':
        year_input = input("  Year (e.g. 2024): ").strip()
        if year_input.isdigit():
            year = int(year_input)
            print(f"  Fetching top {draw_size} players from {year}...")
            players = simulator.get_top_players(year, count=draw_size)
            if players:
                print(f"  Loaded {len(players)} players.")
                for seed, name in players[:8]:
                    print(f"    [{seed}] {name}")
                if len(players) > 8:
                    print(f"    ... and {len(players)-8} more.")
            else:
                print("  Failed to fetch players.")

    elif method == '2':
        print(f"\n  Paste names (comma-separated):")
        raw = input("  > ").strip()
        names = [n.strip() for n in raw.split(',') if n.strip()]

        if len(names) > draw_size:
            print(f"  Truncating to {draw_size} players.")
            names = names[:draw_size]

        print(f"\n  Validating {len(names)} players...")
        for i, name in enumerate(names):
            validated_name = get_validated_player_name(simulator, name, label="player")
            players.append({'seed': i+1, 'name': validated_name, 'year': None})

    # Manual entry fallback
    if not players:
        if method != '1':
            print("\n  Falling back to manual entry...")

        for i in range(draw_size):
            p_input = input(f"  Player {i+1}: ").strip()
            parts = [p.strip() for p in p_input.split(',')]
            seed, name, year = None, None, None

            if len(parts) == 3:
                try:
                    seed, name, year = int(parts[0]), parts[1], int(parts[2])
                except ValueError:
                    name = parts[1]
            elif len(parts) == 2:
                if parts[0].isdigit() and int(parts[0]) < 200:
                    seed, name = int(parts[0]), parts[1]
                elif parts[1].isdigit() and int(parts[1]) > 1900:
                    name, year = parts[0], int(parts[1])
                else:
                    name = parts[0]
            else:
                name = p_input

            validated_name = get_validated_player_name(simulator, name, label=f"Player {i+1}")

            v_year = None
            if year:
                v_year, err = get_validated_year(simulator, validated_name, year)
                if err:
                    print(f"    {err} Skipping year.")

            players.append({'seed': seed or (i+1), 'name': validated_name, 'year': v_year})

    if len(players) < draw_size:
        print(f"\n  Not enough players! Need {draw_size}, got {len(players)}")
        return

    # Monte Carlo simulation
    iterations = _get_iterations()
    simulator = TournamentSimulator()

    mc_results = simulator.simulate_tournament_monte_carlo(
        players, iterations=iterations, surface=surface,
        tourney_level=tourney_level, use_model='average', draw_size=draw_size
    )

    _display_mc_results(mc_results, tournament_name, surface=surface,
                        tourney_level=tourney_level, sim_players=players,
                        draw_size=draw_size)


def _simulate_historical_tournament():
    """Browse and simulate any historical ATP tournament from CSV data."""
    from datetime import datetime as _dt

    # Category menu
    print("\n  Select category:")
    print("    1. Grand Slams")
    print("    2. Masters 1000")
    print("    3. ATP 500")
    print("    4. ATP 250")
    cat_choice = input("  Choice (1-4, default 1): ").strip() or '1'

    if cat_choice == '1':
        level_codes = {'G'}
        category_label = "Grand Slam"
        atp500_only = None
    elif cat_choice == '2':
        level_codes = {'M'}
        category_label = "Masters 1000"
        atp500_only = None
    elif cat_choice == '3':
        level_codes = {'A', '500'}
        category_label = "ATP 500"
        atp500_only = True
    else:
        level_codes = {'A', '250'}
        category_label = "ATP 250"
        atp500_only = False

    # Year input
    current_year = _dt.now().year
    while True:
        year_input = input(f"  Enter year (1968-{current_year}): ").strip()
        if year_input.isdigit() and 1968 <= int(year_input) <= current_year:
            year = int(year_input)
            break
        print(f"  Please enter a year between 1968 and {current_year}.")

    # Load data via simulator (reuses df_historical, tour-only is correct for draw reconstruction)
    print(f"\n  Loading {category_label} data for {year}...")
    simulator = TournamentSimulator()
    simulator.load_historical_data()
    df = simulator.df_historical

    # Filter by year
    df_year = df[df['tourney_date'].astype(str).str[:4] == str(year)]
    df_cat = df_year[df_year['tourney_level'].isin(level_codes)].copy()

    # Split ATP 500 vs 250 within 'A'-coded rows (older data uses 'A' for both)
    if atp500_only is not None:
        def _is_500(name):
            n = str(name).lower()
            return any(t in n for t in ATP_500_TOURNAMENTS)
        # Explicit '500'/'250' coded rows are already correctly separated;
        # only apply name-based split to legacy 'A'-coded rows
        legacy = df_cat['tourney_level'] == 'A'
        explicit_500 = df_cat['tourney_level'] == '500'
        explicit_250 = df_cat['tourney_level'] == '250'
        if atp500_only:
            mask = explicit_500 | (legacy & df_cat['tourney_name'].apply(_is_500))
        else:
            mask = explicit_250 | (legacy & ~df_cat['tourney_name'].apply(_is_500))
        df_cat = df_cat[mask]

    # Enumerate unique tournaments
    if df_cat.empty:
        print(f"\n  No {category_label} tournaments found for {year}.")
        return

    default_draw = 128 if 'G' in level_codes else 64 if 'M' in level_codes else 32
    tournaments = []
    for name, group in df_cat.groupby('tourney_name'):
        surface = group['surface'].dropna().iloc[0] if group['surface'].notna().any() else 'Hard'
        draw_size = int(group['draw_size'].dropna().iloc[0]) if group['draw_size'].notna().any() else default_draw
        raw_level = group['tourney_level'].iloc[0]
        tournaments.append({'name': name, 'surface': surface, 'draw_size': draw_size, 'raw_level': raw_level})
    tournaments.sort(key=lambda t: t['name'])

    # Display tournament list
    print(f"\n  {category_label} tournaments in {year}:")
    for i, t in enumerate(tournaments):
        print(f"    {i+1}. {t['name']}  ({t['surface']}, {t['draw_size']} players)")

    while True:
        pick = input(f"\n  Select tournament (1-{len(tournaments)}): ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(tournaments):
            chosen = tournaments[int(pick) - 1]
            break
        print(f"  Please enter a number between 1 and {len(tournaments)}.")

    # Reconstruct draw from CSV
    df_tourney = df_cat[df_cat['tourney_name'] == chosen['name']]
    player_map = {}  # name -> seed (real seed or NaN)

    for _, row in df_tourney.iterrows():
        for name_col, seed_col in [('winner_name', 'winner_seed'), ('loser_name', 'loser_seed')]:
            pname = row[name_col]
            seed = row[seed_col]
            if pd.isna(pname):
                continue
            if pname not in player_map:
                player_map[pname] = seed
            elif pd.notna(seed):
                existing = player_map[pname]
                if pd.isna(existing):
                    player_map[pname] = seed
                else:
                    player_map[pname] = min(existing, seed)

    seeded = sorted(
        [(int(s), n) for n, s in player_map.items() if pd.notna(s)],
        key=lambda x: x[0]
    )
    unseeded = [n for n, s in player_map.items() if pd.isna(s)]

    next_syn = (seeded[-1][0] + 1) if seeded else 1
    sim_players = [{'seed': s, 'name': n, 'year': year} for s, n in seeded]
    for n in unseeded:
        sim_players.append({'seed': next_syn, 'name': n, 'year': year})
        next_syn += 1

    if len(sim_players) < chosen['draw_size'] // 2:
        print(f"\n  Warning: Only {len(sim_players)} players found (expected {chosen['draw_size']}). Draw may be incomplete.")

    # Show draw summary
    print(f"\n  {chosen['name']} {year} — {chosen['surface']}, {chosen['draw_size']}-player draw")
    print(f"  Players found: {len(sim_players)} ({len(seeded)} seeded, {len(unseeded)} unseeded)")
    print(f"\n  Top seeds:")
    for s, n in seeded[:8]:
        print(f"    [{s}] {n}")
    if len(seeded) > 8:
        print(f"    ... and {len(seeded) - 8} more seeded players")

    # Monte Carlo simulation
    iterations = _get_iterations()

    # Normalise level_code for simulator: map explicit '500'/'250' back to 'A'
    sim_level = chosen.get('raw_level', 'A')
    if sim_level in ('500', '250'):
        sim_level = 'A'

    mc_results = simulator.simulate_tournament_monte_carlo(
        sim_players, iterations=iterations,
        surface=chosen['surface'], tourney_level=sim_level,
        use_model='average', draw_size=chosen['draw_size']
    )

    _display_mc_results(
        mc_results, f"{chosen['name']} {year}",
        surface=chosen['surface'], tourney_level=sim_level,
        sim_players=sim_players, draw_size=chosen['draw_size']
    )


def _simulate_upcoming_tournament():
    """Simulate the next upcoming ATP tournament using real draw data."""
    from atp_api import (
        fetch_tournament_calendar, find_next_tournament,
        map_category_to_level, scrape_wikipedia_draw,
        prepare_draw_for_simulator, build_fallback_draw,
        fetch_top_ranked_players
    )

    # Step 1: Find next tournament
    print("\n  Fetching ATP tournament calendar...")
    calendar = fetch_tournament_calendar()
    if not calendar:
        print("  Failed to fetch tournament calendar.")
        return

    next_tourney, upcoming = find_next_tournament(calendar)
    if not next_tourney:
        print("  No upcoming tournaments found.")
        return

    # Show upcoming tournaments
    from datetime import date as _date
    today_str = _date.today().isoformat()
    is_current = next_tourney.get('start_date', '') <= today_str

    if is_current:
        print(f"\n  Current tournament going on: {next_tourney['name']}")
    else:
        print(f"\n  Next tournament: {next_tourney['name']}")
    print(f"  Start date: {next_tourney.get('start_date', 'Unknown')}")
    print(f"  Category: {next_tourney.get('category', 'Unknown')}")
    print(f"  Surface: {next_tourney.get('surface', 'Unknown')}")

    if len(upcoming) > 1:
        # Separate current vs upcoming for display
        current = [t for t in upcoming if t.get('start_date', '') <= today_str]
        future = [t for t in upcoming if t.get('start_date', '') > today_str]

        if current:
            print(f"\n  Current tournament:")
            for i, t in enumerate(current):
                print(f"    {i+1}. {t['name']} ({t.get('start_date', '?')}) - {t.get('category', '?')}")

        if future:
            offset = len(current)
            print(f"\n  Next tournaments:")
            for i, t in enumerate(future):
                print(f"    {offset+i+1}. {t['name']} ({t.get('start_date', '?')}) - {t.get('category', '?')}")
        pick = input(f"\n  Select tournament (1-{len(upcoming)}, default 1): ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(upcoming):
            next_tourney = upcoming[int(pick) - 1]
            print(f"\n  Selected: {next_tourney['name']}")

    tournament_name = next_tourney['name']
    category = next_tourney.get('category', '')
    surface = next_tourney.get('surface', 'Hard') or 'Hard'
    tourney_level = map_category_to_level(category)
    year = int(next_tourney.get('start_date', '2025')[:4])

    # Determine draw size from category
    if category == 'Grand Slam':
        draw_size = 128
    elif category == 'Masters 1000':
        draw_size = 64
    elif category == 'ATP Finals':
        draw_size = 8
    else:
        draw_size = 32

    # Step 2: Try to get draw from Wikipedia
    print(f"\n  Looking for {tournament_name} {year} draw on Wikipedia...")
    players = scrape_wikipedia_draw(tournament_name, year, draw_size)

    if players and len(players) >= draw_size // 2:
        # Use Wikipedia draw with reverse seed mapping
        print(f"  Using Wikipedia draw ({len(players)} players)")
        players = prepare_draw_for_simulator(players, draw_size)
        source = "Wikipedia draw"
    else:
        # Fallback to rankings
        print("  Wikipedia draw not available. Falling back to current ATP rankings...")
        ranked = fetch_top_ranked_players(draw_size)
        if not ranked or len(ranked) < draw_size:
            print(f"  Could not fetch enough ranked players (got {len(ranked) if ranked else 0}).")
            return
        players = build_fallback_draw(ranked, draw_size)
        source = "ATP rankings"

    # Show draw summary using original seeds (not synthetic bracket seeds)
    seeded = sorted(
        [p for p in players if p.get('original_seed') is not None],
        key=lambda p: p['original_seed']
    )
    print(f"\n  Draw ({draw_size} players, source: {source}):")
    for p in seeded[:8]:
        print(f"    [{p['original_seed']}] {p['name']}")
    if len(seeded) > 8:
        print(f"    ... and {len(seeded) - 8} more seeded players")

    # Step 3: Monte Carlo simulation
    iterations = _get_iterations()
    simulator = TournamentSimulator()

    sim_players = [{'seed': p['seed'], 'name': p['name'], 'year': p.get('year')} for p in players]

    mc_results = simulator.simulate_tournament_monte_carlo(
        sim_players, iterations=iterations, surface=surface,
        tourney_level=tourney_level, use_model='average', draw_size=draw_size
    )

    _display_mc_results(mc_results, f"{tournament_name} {year}", surface=surface,
                        tourney_level=tourney_level, sim_players=sim_players,
                        draw_size=draw_size)


# ─── Option 3: View Model Performance ──────────────────────

def option_3_performance():
    """View model performance."""
    print('\033[2J\033[3J\033[H', end='', flush=True)
    print("\n[3] Model performance")

    try:
        if not os.path.exists('models/evaluation_results.pkl'):
            print("\n  No evaluation results found! Train models first.")
            return

        with open('models/evaluation_results.pkl', 'rb') as f:
            results = pickle.load(f)

        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)

        results_df = pd.DataFrame(results).T
        # Show all models that have numeric roc_auc
        display_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss']
        results_df = results_df[display_cols]
        results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Log Loss']
        results_df = results_df.apply(pd.to_numeric, errors='coerce')

        print("\n" + results_df.round(4).to_string())

        valid = results_df.dropna(subset=['ROC-AUC'])
        if not valid.empty:
            print("\nBest Models:")
            print(f"  Accuracy: {valid['Accuracy'].idxmax()} ({valid['Accuracy'].max():.4f})")
            print(f"  F1:       {valid['F1'].idxmax()} ({valid['F1'].max():.4f})")
            print(f"  ROC-AUC:  {valid['ROC-AUC'].idxmax()} ({valid['ROC-AUC'].max():.4f})")

        # Feature importance
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)

        for model_name in ['xgboost']:
            if model_name in results and 'feature_importance' in results[model_name]:
                print(f"\n  {model_name.replace('_', ' ').title()}:")
                importance = results[model_name]['feature_importance']
                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_imp[:10]:
                    bar = '#' * int(imp * 100)
                    print(f"    {feat:20s} {imp:.4f}  {bar}")

        print()

        # Walk-forward backtest sub-option
        print("\n  b. Run walk-forward backtest (accuracy per year)")
        run_wf = input("  Run backtest? (y/n, default n): ").strip().lower()
        if run_wf == 'y':
            from data_loader import load_data
            from feature_engineering import prepare_features_for_training
            from model_trainer import ModelTrainer

            print("\n  Loading data for backtest...")
            df = load_data()
            _, _, feature_cols, encoders, df_feats = prepare_features_for_training(df)

            trainer = ModelTrainer()
            trainer.feature_names = feature_cols
            trainer.walk_forward_backtest(df_feats, feature_cols)

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


# ─── Main Loop ─────────────────────────────────────────────

def main():
    """Main function."""
    print("\n" + "="*60)
    print("Welcome to CourtVision")
    print("="*60)
    print("ML-powered tennis match prediction and tournament simulation.")

    while True:
        print_menu()
        choice = input("\nChoice (1-4): ").strip()

        if choice == '1':
            option_1_predict()
        elif choice == '2':
            option_2_simulate()
        elif choice == '3':
            option_3_performance()
        elif choice == '4':
            print("\nGoodbye!\n")
            break
        else:
            print("\n  Invalid choice.")

        input("\nPress Enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
