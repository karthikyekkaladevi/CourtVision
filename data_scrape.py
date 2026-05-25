"""
Data scraping module for Tennis Tournament Predictor.
Fetches latest ATP match data from the TML-Database on GitHub
and saves it in the Sackmann format used by this project.
"""
import pandas as pd
import requests
import os
import io
from datetime import datetime


# Paths — relative to THIS project's directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "tennis_atp-master")
PLAYERS_FILE = os.path.join(DATA_DIR, "atp_players.csv")

# URLs
TML_BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"

# Sackmann column order (must match existing atp_matches_YYYY.csv files)
SACKMANN_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"
]


def fetch_tml_data(year):
    """Fetch ATP match data for a given year from the TML-Database."""
    url = f"{TML_BASE_URL}/{year}.csv"
    print(f"  Fetching ATP data for {year} from TML-Database...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            print(f"    ✓ Received {len(df)} rows")
            return df
        else:
            print(f"    ✗ Failed (HTTP {response.status_code})")
            return None
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Network error: {e}")
        return None


def map_tml_to_sackmann(tml_df):
    """
    Map TML-Database columns to Sackmann format.
    TML uses the same player IDs as Sackmann, so no ID remapping is needed.
    The main difference is TML includes an 'indoor' column that Sackmann doesn't have.
    """
    # Drop 'indoor' column if present (not in Sackmann format)
    if 'indoor' in tml_df.columns:
        tml_df = tml_df.drop(columns=['indoor'])

    # Add any missing columns as NaN
    for col in SACKMANN_COLS:
        if col not in tml_df.columns:
            tml_df[col] = None

    # Reorder to match Sackmann exactly
    return tml_df[SACKMANN_COLS]


def scrape_latest_data(years=None):
    """
    Scrape the latest ATP match data and save to the project's tennis_atp-master directory.
    
    Args:
        years: List of years to scrape (e.g. [2025, 2026]). 
               Defaults to current year and previous year.
    
    Returns:
        dict with year -> number of matches saved, or None on failure.
    """
    if years is None:
        current_year = datetime.now().year
        years = [current_year - 1, current_year]

    print(f"\n{'='*60}")
    print(f"SCRAPING LATEST ATP DATA")
    print(f"{'='*60}")
    print(f"  Target years: {', '.join(str(y) for y in years)}")
    print(f"  Output directory: {DATA_DIR}")

    if not os.path.isdir(DATA_DIR):
        print(f"\n  ✗ Data directory not found: {DATA_DIR}")
        print(f"    Please ensure 'tennis_atp-master' exists in the project root.")
        return None

    results = {}
    for year in years:
        print(f"\n  --- {year} ---")
        tml_df = fetch_tml_data(year)

        if tml_df is not None:
            # Drop rows with missing winner/loser names
            before = len(tml_df)
            tml_df = tml_df.dropna(subset=['winner_name', 'loser_name'])
            dropped = before - len(tml_df)
            if dropped > 0:
                print(f"    Dropped {dropped} rows with missing player names")

            sackmann_df = map_tml_to_sackmann(tml_df)
            out_path = os.path.join(DATA_DIR, f"atp_matches_{year}.csv")
            sackmann_df.to_csv(out_path, index=False)
            results[year] = len(sackmann_df)
            print(f"    ✓ Saved {len(sackmann_df)} matches → atp_matches_{year}.csv")
        else:
            results[year] = 0
            print(f"    ✗ No data available for {year}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SCRAPE SUMMARY")
    print(f"{'='*60}")
    total = sum(results.values())
    for year, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {year}: {count:,} matches")
    print(f"  Total: {total:,} matches saved")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    scrape_latest_data()
