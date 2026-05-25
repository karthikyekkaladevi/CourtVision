"""
Main entry point for Tennis Tournament Predictor.
Terminal-based interface for interacting with the predictor.
"""
import os
import sys
from datetime import datetime
from data_loader import load_data, explore_data
from eda import generate_all_eda_plots
from feature_engineering import prepare_features_for_training, split_data_by_date
from model_trainer import ModelTrainer
from predictor import MatchPredictor
from tournament_simulator import TournamentSimulator
from data_scrape import scrape_latest_data
from tournament_gui import launch_tournament_gui
import warnings
warnings.filterwarnings('ignore')

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
                p_name = "" # Reset to prompt again
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
                    p_name = "" # Reset to prompt again
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
        return year, None # Player not found in match data, can't validate range
    
    if year < first:
        return None, f"    [ERR] Year {year} predates {player_name}'s first ATP match in {first}."
    if year > last:
        return None, f"    [ERR] Year {year} is after {player_name}'s last recorded ATP match in {last}."
        
    return year, None

def print_menu():
    """Print main menu."""
    print("\n" + "="*60)
    print("TENNIS TOURNAMENT PREDICTOR")
    print("="*60)
    print("1. Generate EDA graphs")
    print("2. Train models")
    print("3. Predict match winner")
    print("4. Simulate tournament")
    print("5. View model performance")
    _cy = datetime.now().year
    print(f"6. Scrape latest data ({_cy-1}/{_cy})")
    print("7. Exit")
    print("="*60)


def get_data_loading_params():
    """Interactively get data loading parameters from user."""
    print("\nData Selection:")
    
    # Year range
    default_start = 1968
    default_end = datetime.now().year
    
    start_in = input(f"  Start year (default {default_start}): ").strip()
    start_year = int(start_in) if start_in.isdigit() else default_start
    
    end_in = input(f"  End year (default {default_end}): ").strip()
    year_end = int(end_in) if end_in.isdigit() else default_end
    
    # Match types
    print("\n  Match types to include:")
    print("  1. Main Tour Singles (Default)")
    print("  2. Main Tour + Qualifiers/Challengers")
    print("  3. All available (Includes Doubles, Futures, etc.)")
    print("  4. Custom Selection")
    
    type_choice = input("  Choice (1-4): ").strip()
    
    include_types = ['tour']
    if type_choice == '2':
        include_types = ['tour', 'qual_chall']
    elif type_choice == '3':
        include_types = ['tour', 'doubles', 'qual_chall', 'futures']
    elif type_choice == '4':
        print("\n    Select types (comma separated): tour, doubles, qual_chall, futures, amateur")
        custom_in = input("    Selection: ").strip()
        if custom_in:
            include_types = [t.strip() for t in custom_in.split(',')]
            
    return start_year, year_end, include_types


def option_1_eda():
    """Generate EDA graphs."""
    print("\n[Option 1] Generating EDA graphs...")
    
    try:
        start_y, end_y, types = get_data_loading_params()
        df = load_data(start_year=start_y, end_year=end_y, include_types=types)
        
        if df.empty:
            print("\n✗ No data loaded for the selected criteria. Please try different parameters.")
            return
            
        generate_all_eda_plots(df)
        print("\n[OK] EDA graphs generated successfully!")
        print("  Check the 'plots/' directory for all visualizations.")
    except Exception as e:
        print(f"\n✗ Error generating EDA graphs: {e}")
        import traceback
        traceback.print_exc()


def option_2_train():
    """Train all models."""
    print("\n[Option 2] Training models...")
    print("This may take a while, especially for SVM and Neural Network...")
    
    try:
        # Load data with user selection
        start_y, end_y, types = get_data_loading_params()
        
        print("\nLoading data...")
        df = load_data(start_year=start_y, end_year=end_y, include_types=types)
        
        if df.empty:
            print("\n✗ No data loaded for the selected criteria. Cannot train models.")
            return
        
        # Prepare features (ONE SINGLE PASS for entire dataset)
        print("Preparing features (Temporal Symmetric Pass)...")
        # New signature returns: X, y, feature_names, encoders, df_feats
        _, _, feature_names, encoders, df_feats = prepare_features_for_training(df)
        print(f"  Total processed features: {len(feature_names)}")
        
        # Split data temporally using the PROCESSED features
        print("Splitting data temporally...")
        train_df, test_df = split_data_by_date(df_feats, test_size=0.2)
        
        # Extract X and y for train and test
        X_train = train_df[feature_names].values
        y_train = train_df['target'].values
        X_test = test_df[feature_names].values
        y_test = test_df['target'].values
        
        print(f"  Training set: {X_train.shape[0]:,} samples (cutoff: {train_df['tourney_date'].max()})")
        print(f"  Test set: {X_test.shape[0]:,} samples (start: {test_df['tourney_date'].min()})")
        
        # Model Selection Menu
        print("\nSelect models to train:")
        print("  1. Logistic Regression")
        print("  2. K-Nearest Neighbors")
        print("  3. Support Vector Machine (Slow)")
        print("  4. Random Forest")
        print("  5. XGBoost")
        print("  6. Neural Network (Slow)")
        print("  7. Train All (Default)")
        
        selection = input("\nEnter choice(s) (e.g., 1,4,5 or leave blank for all): ").strip()
        
        model_map = {
            '1': 'logistic_regression',
            '2': 'knn',
            '3': 'svm',
            '4': 'random_forest',
            '5': 'xgboost',
            '6': 'neural_network'
        }
        
        selected_keys = []
        if selection and selection != '7':
            choices = [c.strip() for c in selection.replace(',', ' ').split()]
            for c in choices:
                if c in model_map:
                    selected_keys.append(model_map[c])
        
        # Train models
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, y_train, X_test, y_test, feature_names, selected_keys=selected_keys if selected_keys else None)
        
        print("\n[OK] All models trained successfully!")
        print("  Models saved to 'models/' directory.")
        
    except Exception as e:
        print(f"\n✗ Error training models: {e}")
        import traceback
        traceback.print_exc()


def option_3_predict():
    """Predict match winner."""
    print("\n[Option 3] Predict match winner")
    
    try:
        # Check if models exist
        if not os.path.exists('models') or len(os.listdir('models')) == 0:
            print("\n✗ No trained models found!")
            print("  Please train models first (Option 2).")
            return
        
        predictor = MatchPredictor()
        
        # Use simulator for fuzzy matching
        simulator = TournamentSimulator()

        # Get player names with fuzzy match validation
        print("\nEnter player information:")
        
        final_players = []
        for i in range(1, 3):
            p_name = input(f"  Player {i} name: ").strip()
            validated_name = get_validated_player_name(simulator, p_name, label=f"Player {i}")
            final_players.append(validated_name)
        
        player1_name, player2_name = final_players
        
        # Optional: Year filter (e.g. 2010 vs 2016)
        p1_y, p2_y = None, None
        
        while True:
            y1_in = input(f"  End year for {player1_name} (optional, YYYY, career: {simulator.get_player_career_range(player1_name)[0]}-{simulator.get_player_career_range(player1_name)[1]}): ").strip()
            p1_y, err = get_validated_year(simulator, player1_name, y1_in)
            if err:
                print(err)
                continue
            break

        while True:
            y2_in = input(f"  End year for {player2_name} (optional, YYYY, career: {simulator.get_player_career_range(player2_name)[0]}-{simulator.get_player_career_range(player2_name)[1]}): ").strip()
            p2_y, err = get_validated_year(simulator, player2_name, y2_in)
            if err:
                print(err)
                continue
            break
        
        # Optional: surface and tournament level
        surface = input("  Surface (Hard/Clay/Grass, press Enter for Hard): ").strip()
        if not surface:
            surface = 'Hard'
        
        tourney_level = 'M' # Internal default for stats fetching
        
        while True:
            sets_input = input("  Number of sets (3/5/7/9, press Enter for 3): ").strip()
            if not sets_input:
                sets_to_play = 3
                break
            
            try:
                sets_to_play = int(sets_input)
                if sets_to_play > 0 and sets_to_play % 2 != 0:
                    print(f"  [OK] Playing best {sets_to_play // 2 + 1}/{sets_to_play} sets")
                    break
                else:
                    print("    ✗ Please enter a positive ODD number (e.g., 3, 5, 7).")
            except ValueError:
                print("    ✗ Invalid input. Please enter a number.")
        
        # Load historical data
        print("\nLoading historical data...")
        try:
            df = load_data()
        except FileNotFoundError:
            print(f"\n[ERROR] Could not find 'atp_data.csv'. Please ensure it's in the project directory.")
            return

        strict_year = False
        if p1_y or p2_y:
            print(f"\nERA FILTERING OPTIONS:")
            print(f"1. Cumulative (Stats from start of career up to the year provided) [Default]")
            print(f"2. Strict (Stats ONLY for the specific year(s) provided)")
            era_choice = input("Select mode (1/2): ").strip()
            strict_year = (era_choice == '2')

        # Predict
        print("Making predictions...")
        predictions = predictor.predict_from_names(
            player1_name, player2_name, df, surface, 'M' if not tourney_level else tourney_level,
            player1_year=p1_y, player2_year=p2_y, strict_year=strict_year
        )
        
        if predictions:
            predictor.display_predictions(predictions, player1_name, player2_name, sets_to_play=sets_to_play)
        else:
            print("\n✗ Could not generate predictions. Players may not be in dataset.")
            
    except Exception as e:
        print(f"\n✗ Error making prediction: {e}")
        import traceback
        traceback.print_exc()


def option_4_simulate():
    """Simulate tournament."""
    print("\n[Option 4] Simulate tournament")
    
    try:
        # Check if models exist
        if not os.path.exists('models') or len(os.listdir('models')) == 0:
            print("\n✗ No trained models found!")
            print("  Please train models first (Option 2).")
            return
        
        # Try GUI mode first
        print("\nSelect input mode:")
        print("  1. Bracket GUI (opens visual bracket window)")
        print("  2. Terminal (text-based input)")
        mode = input("  Choice (1/2, default 1): ").strip()
        
        if mode != '2':
            try:
                success = launch_tournament_gui()
                if success:
                    # GUI flow completed; return to main menu
                    return
                else:
                    # pywebview missing or user closed GUI setup; continue in terminal mode
                    print("  GUI unavailable or cancelled. Falling back to terminal input...")
            except Exception as gui_err:
                print(f"  GUI failed ({gui_err}). Falling back to terminal mode...")
        
        simulator = TournamentSimulator()
        
        # Get tournament details
        print("\nEnter tournament details:")
        draw_size_input = input("  Draw size (8/16/32/64/128, press Enter for 8): ").strip()
        draw_size = int(draw_size_input) if draw_size_input else 8
        
        surface = input("  Surface (Hard/Clay/Grass, press Enter for Hard): ").strip()
        if not surface:
            surface = 'Hard'
        
        tourney_level = input("  Tournament level (M/A/G, press Enter for M): ").strip().upper()
        if not tourney_level:
            tourney_level = 'M'
        
        # Tournament Name Mapping
        TOURNAMENT_MAP = {
            ('HARD', 'G'): ["Australian Open", "US Open"],
            ('CLAY', 'G'): ["French Open (Roland Garros)"],
            ('GRASS', 'G'): ["Wimbledon"],
            ('HARD', 'M'): ["Indian Wells Masters", "Miami Open", "Canadian Open", "Cincinnati Masters", "Shanghai Masters", "Paris Masters"],
            ('CLAY', 'M'): ["Monte Carlo Masters", "Madrid Open", "Italian Open (Rome)"],
            ('HARD', 'F'): ["ATP Finals"],
            ('HARD', 'A'): ["Rotterdam Open", "Dubai Championships", "Acapulco Open", "Tokyo Open", "Vienna Open", "Basel Open"],
            ('CLAY', 'A'): ["Rio Open", "Barcelona Open", "Hamburg Open"],
            ('GRASS', 'A'): ["Halle Open", "Queen's Club Championships"],
        }
        
        key = (surface.upper(), tourney_level)
        possible_tournaments = TOURNAMENT_MAP.get(key, [])
        tournament_name = "the tournament"
        
        if possible_tournaments:
            print(f"\nMatching tournaments for {surface} {tourney_level}:")
            for i, name in enumerate(possible_tournaments):
                print(f"{i+1}. {name}")
            print(f"{len(possible_tournaments)+1}. Custom Name")
            
            t_choice = input(f"Select tournament (1-{len(possible_tournaments)+1}, default 1): ").strip()
            if not t_choice or t_choice == '1':
                tournament_name = possible_tournaments[0]
            elif t_choice.isdigit() and 1 <= int(t_choice) <= len(possible_tournaments):
                tournament_name = possible_tournaments[int(t_choice)-1]
            else:
                custom = input("Enter tournament name: ").strip()
                if custom:
                    tournament_name = custom
        else:
            custom = input(f"Enter tournament name (default: 'the {surface} tournament'): ").strip()
            tournament_name = custom if custom else f"the {surface} tournament"
        
        # Input Method Selection
        print(f"\nSelect Player Input Method for {draw_size} entries:")
        print("1. Manual Entry (Type one by one)")
        print("2. Bulk Input (Paste comma-separated list)")
        print("3. Auto-Fill (Top ranked players from a specific year)")
        method = input("Choice (1/2/3, default 1): ").strip()
        
        players = []
        
        if method == '3':
            # Auto-Fill
            year_input = input("Enter Year (e.g. 2016): ").strip()
            if year_input.isdigit():
                year = int(year_input)
                print(f"Fetching top {draw_size} players from {year}...")
                players = simulator.get_top_players(year, count=draw_size)
                if not players:
                    print("Failed to fetch players. Please try manual entry.")
                else:
                    print(f"Successfully loaded {len(players)} players.")
                    for seed, name in players[:8]:
                        print(f"  [{seed}] {name}")
                    if len(players) > 8:
                        print(f"  ... and {len(players)-8} more.")
            else:
                print("Invalid year.")
        
        elif method == '2':
            # Bulk Input
            print(f"\nPaste names (comma separated) or type 'PASTE' for multi-line mode:")
            raw = input("> ")
            if raw.upper() == 'PASTE':
                print("Enter names (one per line, end with empty line):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line.strip())
                names = [l for l in lines if l]
            else:
                names = [n.strip() for n in raw.replace('\n', ',').split(',') if n.strip()]
            
            # Take top N if too many, or pad if too few?
            if len(names) > draw_size:
                print(f"Warning: You provided {len(names)} names, truncating to {draw_size}.")
                names = names[:draw_size]

            print(f"\nValidating {len(names)} players...")
            for i, name in enumerate(names):
                print(f"  Validating [{i+1}/{len(names)}]: {name}")
                validated_name = get_validated_player_name(simulator, name, label="player")
                players.append({'seed': i+1, 'name': validated_name, 'year': None})
        
        # Fallback to Manual if empty or Method 1
        if not players:
            if method != '1':
                print("\nFalling back to Manual Entry...")
            
            for i in range(len(players), draw_size):
                p_input = input(f"  Player {i+1}: ").strip()
                # Handle 'seed, name, year' logic
                parts = [p.strip() for p in p_input.split(',')]
                seed, name, year = None, None, None
                
                if len(parts) == 3:
                    try: seed, name, year = int(parts[0]), parts[1], int(parts[2])
                    except: name, year = parts[1], None
                elif len(parts) == 2:
                    if parts[0].isdigit() and int(parts[0]) < 200: seed, name = int(parts[0]), parts[1]
                    elif parts[1].isdigit() and int(parts[1]) > 1900: name, year = parts[0], int(parts[1])
                    else: name = parts[0]
                else:
                    name = p_input

                validated_name = get_validated_player_name(simulator, name, label=f"Player {i+1}")
                
                # Validate Year if provided
                v_year = None
                if year:
                    v_year, err = get_validated_year(simulator, validated_name, year)
                    if err:
                        print(f"    {err} Skipping year.")
                
                players.append({'seed': seed or (i+1), 'name': validated_name, 'year': v_year})
        
        if len(players) < draw_size:
            print(f"\n✗ Not enough players! Need {draw_size}, got {len(players)}")
            return

        # --- Late-Pro Cutoff Logic (Optional) ---
        if method in ['1', '2']: # Manual or Bulk
            while True:
                print(f"\nLATE-PRO CUTOFF OPTION:")
                print(f"1. Apply Late-Pro Cutoff (Stats for each player capped at the latest pro debut in the group)")
                print(f"2. Explain Late-Pro Cutoff")
                print(f"3. Skip (Use latest available stats for everyone)")
                cutoff_choice = input("Select option (1/2/3): ").strip()

                if cutoff_choice == '2':
                    print("\n" + "-"*60)
                    print("EXPLANATION: LATE-PRO CUTOFF")
                    print("-"*60)
                    print("This feature handles cross-era matches by ensuring everyone's statistics stop at")
                    print("the moment the 'newest' player in your list turned professional. For example,")
                    print("if you simulate a match between Roger Federer and Carlos Alcaraz, applying the")
                    print("cutoff will use Federer's stats up until 2022 (when Alcaraz debuted), rather than")
                    print("his current career-end stats. This creates a fairer 'meeting point' in time.")
                    print("Note: If you explicitly assigned a year to a player, that year is ALWAYS used.")
                    print("-"*60)
                    continue
                elif cutoff_choice == '1':
                    print("\nCalculating Late-Pro Cutoff...")
                    player_debuts = {}
                    for p in players:
                        if not p.get('year'): # Find debuts for those without explicit years
                            debut = simulator.get_player_debut_year(p['name'])
                            if debut:
                                player_debuts[p['name']] = debut
                    
                    if player_debuts:
                        global_cutoff = max(player_debuts.values())
                        latest_players = [name for name, year in player_debuts.items() if year == global_cutoff]
                        
                        print(f"  Late-Pro Cutoff Year: {global_cutoff}")
                        print(f"  Triggered by: {', '.join(latest_players)}")
                        print(f"  Applying this cutoff to all players without explicit years...")
                        
                        for p in players:
                            if not p.get('year'):
                                p['year'] = global_cutoff
                    else:
                        print("  Could not determine debut years for any players. Skipping.")
                    break
                else:
                    break

        # Run a standard simulation
        show_details_input = input("\nShow detailed match stats (head-to-head details)? (y/n, default n): ").strip().lower()
        show_details = show_details_input == 'y'
        
        results = simulator.simulate_tournament(
            players, surface=surface, tourney_level=tourney_level, 
            use_model='average', draw_size=draw_size, show_details=show_details
        )
        
        # Final Announcement
        print("\n" + "*"*60)
        print(f"🏆 {results['champion']} is the winner of {tournament_name}!")
        print("*"*60 + "\n")
        
    except FileNotFoundError:
        print("\n✗ Error: atp_data.csv not found!")
    except Exception as e:
        print(f"\n✗ Error simulating tournament: {e}")
        import traceback
        traceback.print_exc()


def option_5_performance():
    """View model performance."""
    print("\n[Option 5] View model performance")
    
    try:
        if not os.path.exists('models/evaluation_results.pkl'):
            print("\n✗ No evaluation results found!")
            print("  Please train models first (Option 2).")
            return
        
        import pickle
        with open('models/evaluation_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        
        import pandas as pd
        results_df = pd.DataFrame(results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss']]
        results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Log Loss']
        
        print("\n" + results_df.round(4).to_string())
        
        print("\nBest Models:")
        print(f"  Accuracy: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.4f})")
        print(f"  F1-Score: {results_df['F1-Score'].idxmax()} ({results_df['F1-Score'].max():.4f})")
        print(f"  ROC-AUC: {results_df['ROC-AUC'].idxmax()} ({results_df['ROC-AUC'].max():.4f})")
        
        # Feature importance for models that have it
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        for model_name in ['random_forest', 'xgboost']:
            if model_name in results and 'feature_importance' in results[model_name]:
                print(f"\n{model_name.replace('_', ' ').title()}:")
                importance = results[model_name]['feature_importance']
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                print("  Top 10 Features:")
                for feat, imp in sorted_importance[:10]:
                    print(f"    {feat}: {imp:.4f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n✗ Error viewing performance: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("\n" + "="*60)
    print("Welcome to Tennis Tournament Predictor!")
    print("="*60)
    print("\nThis tool uses machine learning to predict tennis match outcomes")
    print("and simulate tournament brackets.")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            option_1_eda()
        elif choice == '2':
            option_2_train()
        elif choice == '3':
            option_3_predict()
        elif choice == '4':
            option_4_simulate()
        elif choice == '5':
            option_5_performance()
        elif choice == '6':
            print("\n[Option 6] Scraping latest ATP data...")
            try:
                scrape_latest_data()
            except Exception as e:
                print(f"\n✗ Error scraping data: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '7':
            print("\nThank you for using Tennis Tournament Predictor!")
            print("Goodbye!\n")
            break
        else:
            print("\n✗ Invalid choice! Please enter a number between 1 and 7.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)