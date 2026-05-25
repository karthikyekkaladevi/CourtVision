"""
Data loading and exploration module for tennis tournament predictor.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_path='tennis_atp-master', start_year=1968, end_year=None, include_types=['tour']):
    """
    Load ATP match data from a file or a directory of yearly CSV files.

    Args:
        data_path: Path to the CSV file or directory containing multi-file data
        start_year: Start year to load (if directory)
        end_year: End year to load (if directory); defaults to the current calendar year
        include_types: List of types to include: 'tour', 'doubles', 'qual_chall', 'futures', 'amateur'

    Returns:
        DataFrame with match data
    """
    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year
    try:
        path = Path(data_path)
        
        if path.is_file():
            print(f"Reading single data file: {data_path}...")
            df = pd.read_csv(data_path, low_memory=False)
        elif path.is_dir():
            print(f"Reading tournament data from directory: {data_path} ({start_year}-{end_year})...")
            
            # Map types to file patterns
            type_patterns = {
                'tour': "atp_matches_[0-9][0-9][0-9][0-9].csv",
                'doubles': "atp_matches_doubles_[0-9][0-9][0-9][0-9].csv",
                'qual_chall': "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv",
                'futures': "atp_matches_futures_[0-9][0-9][0-9][0-9].csv",
                'amateur': "atp_matches_amateur.csv"
            }
            
            all_valid_files = []
            for t in include_types:
                if t in type_patterns:
                    pattern = type_patterns[t]
                    files = sorted(list(path.glob(pattern)))
                    
                    # Filter by year range if pattern has a year
                    if "[0-9]" in pattern:
                        for f in files:
                            try:
                                # Extract year from filename (e.g. atp_matches_2023.csv or atp_matches_doubles_2023.csv)
                                parts = f.stem.split('_')
                                year = int(parts[-1])
                                if start_year <= year <= end_year:
                                    all_valid_files.append(f)
                            except (ValueError, IndexError):
                                continue
                    else:
                        # Non-yearly files like amateur
                        all_valid_files.extend(files)
            
            if not all_valid_files:
                print(f"[ERROR] No matching files found in {data_path} for types {include_types}")
                return pd.DataFrame()
                
            print(f"  Found {len(all_valid_files)} match files. Concatenating...")
            df_list = []
            
            try:
                from tqdm import tqdm
                iterator = tqdm(all_valid_files, desc="Loading CSVs")
            except ImportError:
                print("  (Tip: install 'tqdm' for a progress bar)")
                iterator = all_valid_files
                
            for f in iterator:
                curr_df = pd.read_csv(f, low_memory=False)
                df_list.append(curr_df)
            
            df = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            print(f"[ERROR] path '{data_path}' is neither a file nor a directory!")
            return pd.DataFrame()

        # Remove rows that are repeated headers (found in some datasets)
        if not df.empty and 'winner_rank' in df.columns:
            header_col = 'winner_rank'
            df = df[df[header_col] != header_col].reset_index(drop=True)
            
        # Convert numeric columns to proper numeric types
        numeric_cols = [
            'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
            'winner_age', 'loser_age', 'winner_ht', 'loser_ht',
            'minutes', 'draw_size', 'match_num',
            'winner_seed', 'loser_seed', 'winner_id', 'loser_id',
            'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'w_SvGms', 'w_bpSaved', 'w_bpFaced',
            'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
            'l_SvGms', 'l_bpSaved', 'l_bpFaced',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"[OK] Successfully loaded total data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File '{csv_path}' not found!")
        raise
    except Exception as e:
        print(f"[ERROR] Loading data: {e}")
        raise


def explore_data(df):
    """
    Display basic statistics about the dataset.
    
    Args:
        df: DataFrame with match data
    """
    print("\n" + "="*60)
    print("DATA EXPLORATION SUMMARY")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Date range
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        date_range = df['tourney_date'].dropna()
        if len(date_range) > 0:
            print(f"\nDate Range: {date_range.min().strftime('%Y-%m-%d')} to {date_range.max().strftime('%Y-%m-%d')}")
    
    # Unique counts
    if 'winner_name' in df.columns:
        unique_winners = df['winner_name'].nunique()
        print(f"Unique Winners: {unique_winners:,}")
    
    if 'loser_name' in df.columns:
        unique_losers = df['loser_name'].nunique()
        print(f"Unique Losers: {unique_losers:,}")
    
    if 'tourney_name' in df.columns:
        unique_tournaments = df['tourney_name'].nunique()
        print(f"Unique Tournaments: {unique_tournaments:,}")
    
    # Missing values summary
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.head(20).to_string())
        if len(missing_df) > 20:
            print(f"... and {len(missing_df) - 20} more columns with missing values")
    else:
        print("No missing values found!")
    
    # Data types
    print("\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    print(dtype_counts.to_string())
    
    print("\n" + "="*60)


def get_relevant_columns(df):
    """
    Identify and return relevant columns for prediction.
    
    Args:
        df: DataFrame with match data
        
    Returns:
        List of relevant column names
    """
    relevant_cols = [
        # Tournament info
        'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date',
        # Match info
        'match_num', 'round', 'score', 'best_of', 'minutes',
        # Winner info
        'winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 
        'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points',
        # Loser info
        'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
        'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points',
        # Match statistics
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms',
        'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
        'l_bpSaved', 'l_bpFaced'
    ]
    
    # Filter to only columns that exist in the dataframe
    available_cols = [col for col in relevant_cols if col in df.columns]
    
    return available_cols


if __name__ == "__main__":
    # Test the data loader
    df = load_data()
    explore_data(df)
    relevant_cols = get_relevant_columns(df)
    print(f"\nRelevant columns for prediction: {len(relevant_cols)}")
    print(f"Sample columns: {relevant_cols[:10]}...")
