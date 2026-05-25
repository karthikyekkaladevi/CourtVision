#!/usr/bin/env python3
"""
auto_update_and_retrain.py
==========================
Automatically updates the ATP tennis match dataset with newly completed
tournaments and retrains all prediction models before every prediction run.

This script integrates directly with the existing project pipeline:
  - data_loader.py      (load_data)
  - feature_engineering.py (prepare_features_for_training, split_data_by_date)
  - model_trainer.py    (ModelTrainer)

Data Source:
  TML-Database on GitHub — mirrors Jeff Sackmann's ATP dataset format,
  updated weekly with completed tour-level matches.
  URL: https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv

Dataset Schema:
  Sackmann format (49 columns): tourney_id, tourney_name, surface,
  draw_size, tourney_level, tourney_date, match_num, winner_*/loser_*,
  score, best_of, round, minutes, w_ace ... l_bpFaced,
  winner_rank, winner_rank_points, loser_rank, loser_rank_points

Tournament Levels Included:
  G = Grand Slams
  M = Masters 1000
  A = ATP 500 / ATP 250 (main tour events)
  S = ATP 500 (some sources use this)
  F = ATP Finals / year-end championship
  (Challenger, Futures, ITF, Davis Cup, Olympics are EXCLUDED)

Usage:
  python auto_update_and_retrain.py                          # Full update + retrain all
  python auto_update_and_retrain.py --update-only            # Only update the dataset
  python auto_update_and_retrain.py --retrain-only           # Only retrain models
  python auto_update_and_retrain.py --years 2025 2026        # Specific years to fetch
  python auto_update_and_retrain.py --models xgboost logistic_regression
  python auto_update_and_retrain.py --no-backup              # Skip backup step
  python auto_update_and_retrain.py --dry-run                # Preview without writing
  python auto_update_and_retrain.py --elo                    # Also compute ELO ratings

Required pip packages (beyond existing project deps):
  pip install requests pandas numpy scikit-learn xgboost tqdm
  pip install python-dateutil unicodedata2
  (beautifulsoup4 is included for future HTML scraping fallback)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import io
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1: CONFIGURATION
# All paths and constants are defined here for easy modification.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# --- Data paths ---
DATA_DIR = PROJECT_DIR / "tennis_atp-master"          # Directory with per-year CSV files
MODELS_DIR = PROJECT_DIR / "models"                   # Where trained models are saved
BACKUPS_DIR = PROJECT_DIR / "backups"                 # Timestamped backups live here
ELO_OUTPUT_PATH = PROJECT_DIR / "models" / "elo_ratings.json"

# --- Sackmann column schema (must exactly match existing CSV files) ---
SACKMANN_COLS: List[str] = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

# ATP main-tour levels to keep. Excludes: C (Challenger), D (Davis Cup),
# O (Olympics), E (Exhibition), I (ITF), blank rows, etc.
ATP_MAIN_TOUR_LEVELS = {"G", "M", "A", "S", "F"}

# Jeff Sackmann's official ATP dataset — the authoritative source, updated weekly.
# TML-Database (previously used by data_scrape.py) lags significantly behind.
TML_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

# HTTP request settings
REQUEST_TIMEOUT_SEC = 45
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 3.0   # exponential base; actual wait = RETRY_BACKOFF_SEC ** attempt

# Keep at most this many timestamped backups (oldest are deleted)
MAX_BACKUP_VERSIONS = 5

# Numeric columns that need coercion after load
NUMERIC_COLS: List[str] = [
    "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
    "winner_age", "loser_age", "winner_ht", "loser_ht",
    "minutes", "draw_size", "match_num",
    "winner_seed", "loser_seed", "winner_id", "loser_id",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

# Valid models that ModelTrainer knows about
VALID_MODELS = [
    "logistic_regression", "knn", "svm",
    "random_forest", "xgboost", "neural_network",
]

# ELO default rating and K-factor per tournament level
ELO_DEFAULT_RATING = 1500.0
ELO_K_FACTORS: Dict[str, float] = {
    "G": 40.0,   # Grand Slams — highest stakes
    "M": 30.0,   # Masters 1000
    "A": 24.0,   # ATP 500 / 250
    "S": 24.0,
    "F": 28.0,   # ATP Finals
}
ELO_K_DEFAULT = 20.0

# ============================================================
# SECTION 2: LOGGING
# ============================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_DIR / "auto_update.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("auto_update")


# ============================================================
# SECTION 3: BACKUP UTILITIES
# ============================================================

def _backup_dir_path() -> Path:
    """Return a timestamped backup directory path (not yet created)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BACKUPS_DIR / ts


def backup_data_files(dry_run: bool = False) -> Optional[Path]:
    """
    Copy all atp_matches_YYYY.csv files into a timestamped backup folder.

    Args:
        dry_run: If True, log what would be backed up but don't copy.

    Returns:
        Path to the backup directory created, or None if nothing to back up.
    """
    csv_files = sorted(DATA_DIR.glob("atp_matches_*.csv"))
    if not csv_files:
        log.info("Backup: no CSV files found in %s — skipping.", DATA_DIR)
        return None

    dest = _backup_dir_path() / "tennis_atp-master"
    log.info("Backing up %d CSV file(s) → %s", len(csv_files), dest)

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for f in csv_files:
            shutil.copy2(f, dest / f.name)
        log.info("Data backup complete.")

    _prune_old_backups(dry_run)
    return dest.parent


def backup_models(dry_run: bool = False) -> Optional[Path]:
    """
    Copy the entire models/ directory into a timestamped backup folder.

    Args:
        dry_run: If True, log what would be backed up but don't copy.

    Returns:
        Path to the backup directory, or None if models directory is empty.
    """
    if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
        log.info("Backup: models/ directory is empty — skipping.")
        return None

    dest = _backup_dir_path() / "models"
    log.info("Backing up models/ → %s", dest)

    if not dry_run:
        shutil.copytree(MODELS_DIR, dest, dirs_exist_ok=True)
        log.info("Model backup complete.")

    return dest.parent


def _prune_old_backups(dry_run: bool = False) -> None:
    """Delete the oldest backup directories beyond MAX_BACKUP_VERSIONS."""
    if not BACKUPS_DIR.exists():
        return
    versions = sorted(
        [d for d in BACKUPS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    to_delete = versions[: max(0, len(versions) - MAX_BACKUP_VERSIONS)]
    for old in to_delete:
        log.info("Pruning old backup: %s", old)
        if not dry_run:
            shutil.rmtree(old, ignore_errors=True)


# ============================================================
# SECTION 4: DATA DETECTION
# ============================================================

def get_latest_tournament_date() -> Optional[datetime]:
    """
    Scan all local atp_matches_YYYY.csv files and return the most recent
    tourney_date found. This tells us what's already in the dataset.

    Returns:
        datetime of the latest tournament, or None if no data found.
    """
    csv_files = sorted(DATA_DIR.glob("atp_matches_[0-9][0-9][0-9][0-9].csv"))
    if not csv_files:
        log.warning("No local CSV files found in %s.", DATA_DIR)
        return None

    latest_date: Optional[datetime] = None

    for path in csv_files:
        try:
            # Only read the date column to be fast
            df = pd.read_csv(path, usecols=["tourney_date"], low_memory=False)
            df["tourney_date"] = pd.to_datetime(
                df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
            )
            col_max = df["tourney_date"].max()
            if pd.notna(col_max):
                if latest_date is None or col_max > latest_date:
                    latest_date = col_max.to_pydatetime()
        except Exception as exc:
            log.warning("Could not read date from %s: %s", path.name, exc)

    if latest_date:
        log.info(
            "Latest tournament date already in dataset: %s",
            latest_date.strftime("%Y-%m-%d"),
        )
    return latest_date


def get_years_needing_update(
    latest_date: Optional[datetime],
    force_years: Optional[List[int]] = None,
) -> List[int]:
    """
    Determine which years need to be fetched from TML-Database.

    Logic:
      - Always re-fetch the current calendar year (ongoing season).
      - If the dataset ends in a previous year, also fetch all intervening years.
      - If force_years is supplied, use those exactly.

    Args:
        latest_date: Most recent tourney_date in the local dataset.
        force_years: If set, overrides the auto-detection.

    Returns:
        Sorted list of years to fetch.
    """
    today = datetime.now()

    if force_years:
        log.info("Using user-specified years: %s", force_years)
        return sorted(force_years)

    if latest_date is None:
        # No local data at all — fetch from two years back to be safe
        years = list(range(today.year - 1, today.year + 1))
        log.info(
            "No local data detected. Will fetch years: %s",
            years,
        )
        return years

    # Re-fetch everything from the year the dataset last has data in
    # (in case the current-year file is partial) through the current year.
    start_year = latest_date.year
    years = list(range(start_year, today.year + 1))
    log.info("Will fetch/refresh years: %s", years)
    return years


# ============================================================
# SECTION 5: NAME NORMALIZATION
# ============================================================

def normalize_player_name(name: str) -> str:
    """
    Normalize a player name for deduplication and consistent storage.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Decompose Unicode → remove accent diacritics (é→e, ñ→n, etc.).
      3. Re-encode as ASCII, dropping any remaining non-ASCII characters.
      4. Title-case for consistent capitalization.

    Note: Sackmann uses UTF-8 names with accents (e.g. "Rafael Nadal" is
    already ASCII, but "Jiří Lehečka" has diacritics). We normalize for
    deduplication only; the stored value in CSV preserves the original form
    from the source.

    Args:
        name: Raw player name string.

    Returns:
        Normalized ASCII name string.
    """
    if not isinstance(name, str):
        return str(name)
    # NFKD decomposition separates base characters from diacritics
    nfkd = unicodedata.normalize("NFKD", name.strip())
    ascii_bytes = nfkd.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii").strip().title()


# ============================================================
# SECTION 6: DATA FETCHING
# ============================================================

def fetch_year_from_tml(year: int) -> Optional[pd.DataFrame]:
    """
    Fetch ATP match data for a given calendar year from TML-Database with
    automatic retries and exponential back-off.

    The TML-Database mirrors Sackmann's format, so minimal mapping is needed.

    Args:
        year: Calendar year (e.g. 2026).

    Returns:
        DataFrame in Sackmann column order, or None if all retries failed.
    """
    url = f"{TML_BASE_URL}/atp_matches_{year}.csv"
    log.info("  Fetching %d from Sackmann repo: %s", year, url)

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            log.info("    ✓ Received %d rows for %d", len(df), year)
            return df
        except requests.exceptions.HTTPError as exc:
            # 404 means year not available yet — no point retrying
            if exc.response is not None and exc.response.status_code == 404:
                log.warning("    Year %d not found on Sackmann repo (HTTP 404). Skipping.", year)
                return None
            last_exc = exc
        except requests.exceptions.RequestException as exc:
            last_exc = exc

        wait = RETRY_BACKOFF_SEC ** attempt
        log.warning(
            "    Attempt %d/%d failed (%s). Retrying in %.1fs…",
            attempt, MAX_RETRIES, last_exc, wait,
        )
        time.sleep(wait)

    log.error("    ✗ All %d attempts failed for year %d: %s", MAX_RETRIES, year, last_exc)
    return None


# ============================================================
# SECTION 7: DATA NORMALIZATION & FILTERING
# ============================================================

def filter_to_atp_main_tour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only ATP 250 / 500 / Masters 1000 / Grand Slams / ATP Finals.
    Removes Challenger (C), Davis Cup (D), Olympics (O), Exhibitions, etc.

    Sackmann tourney_level codes:
      G = Grand Slam          M = Masters 1000
      A = ATP 500/250 tour    S = ATP 500 (alt code)
      F = ATP Finals          C = Challenger  (excluded)
      D = Davis Cup           (excluded)      O = Olympics (excluded)

    Args:
        df: Raw DataFrame from TML/Sackmann source.

    Returns:
        Filtered DataFrame containing only main-tour singles.
    """
    if "tourney_level" not in df.columns:
        log.warning("    'tourney_level' column missing; cannot filter. Returning all rows.")
        return df

    before = len(df)
    df = df[df["tourney_level"].isin(ATP_MAIN_TOUR_LEVELS)].copy()
    removed = before - len(df)
    if removed:
        log.info("    Filtered out %d non-main-tour rows (Challengers / Davis Cup / etc.)", removed)
    return df


def map_to_sackmann_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a TML-fetched DataFrame into the exact Sackmann column schema.

    TML is nearly identical to Sackmann, except:
      - May include an 'indoor' column not present in Sackmann.
      - Column order may differ.
      - Any extra columns are silently dropped.
      - Any missing Sackmann columns are added as NaN.

    Args:
        raw_df: DataFrame as returned from TML-Database.

    Returns:
        DataFrame with exactly SACKMANN_COLS in Sackmann column order.
    """
    df = raw_df.copy()

    # Drop TML-specific extras that are not in Sackmann
    extras = [c for c in df.columns if c not in SACKMANN_COLS]
    if extras:
        df = df.drop(columns=extras)

    # Add any missing Sackmann columns as NaN
    for col in SACKMANN_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder to exact Sackmann order
    return df[SACKMANN_COLS]


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that should be numeric but may have come in as strings.
    Uses errors='coerce' so unparseable values become NaN instead of raising.
    """
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_match_id(row: pd.Series) -> str:
    """
    Construct a deterministic unique identifier for a match.

    Uses tourney_id + match_num as the canonical key, which is how
    Sackmann uniquely identifies every match. Falls back to a composite
    of date, player names, and round when those fields are missing.

    Args:
        row: A single DataFrame row (pd.Series).

    Returns:
        String match identifier.
    """
    tourney_id = str(row.get("tourney_id", "")).strip()
    match_num = str(row.get("match_num", "")).strip()

    if tourney_id and match_num and tourney_id != "nan" and match_num != "nan":
        return f"{tourney_id}__{match_num}"

    # Fallback composite key
    date = str(row.get("tourney_date", "")).strip()
    w = normalize_player_name(str(row.get("winner_name", "")))
    l = normalize_player_name(str(row.get("loser_name", "")))
    rnd = str(row.get("round", "")).strip()
    return f"{date}__{w}__{l}__{rnd}"


# ============================================================
# SECTION 8: LOCAL FILE MANAGEMENT & MERGING
# ============================================================

def load_existing_year(year: int) -> pd.DataFrame:
    """
    Load the existing local CSV for a calendar year.

    Args:
        year: Calendar year (e.g. 2025).

    Returns:
        DataFrame, or empty DataFrame if file doesn't exist.
    """
    path = DATA_DIR / f"atp_matches_{year}.csv"
    if not path.exists():
        log.info("    No existing file for %d — will create from scratch.", year)
        return pd.DataFrame(columns=SACKMANN_COLS)
    try:
        df = pd.read_csv(path, low_memory=False)
        # Ensure all Sackmann columns are present (schema migration safety)
        for col in SACKMANN_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[SACKMANN_COLS]
    except Exception as exc:
        log.error("    Error loading existing file %s: %s", path.name, exc)
        return pd.DataFrame(columns=SACKMANN_COLS)


def merge_new_into_existing(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Merge newly fetched matches into the existing local dataset, keeping
    only matches that aren't already present (based on match ID).

    This is an additive, non-destructive merge:
      - Existing rows are NEVER modified or removed.
      - Only genuinely new matches are appended.
      - Deduplication is by match_id (tourney_id + match_num).

    Args:
        existing_df: Current local data for a year.
        new_df:      Freshly fetched data from TML for the same year.

    Returns:
        Tuple of (merged DataFrame, count of newly appended rows).
    """
    if new_df.empty:
        return existing_df, 0

    # Build match IDs for both datasets
    existing_ids: set = set()
    if not existing_df.empty:
        existing_ids = set(existing_df.apply(build_match_id, axis=1).tolist())

    new_ids = new_df.apply(build_match_id, axis=1)
    is_new_mask = ~new_ids.isin(existing_ids)
    truly_new = new_df[is_new_mask].copy()

    added = len(truly_new)
    if added == 0:
        log.info("    No new matches to add — dataset already up-to-date for this year.")
        return existing_df, 0

    log.info(
        "    Adding %d new match(es) (dataset had %d, source has %d).",
        added, len(existing_df), len(new_df),
    )

    merged = pd.concat([existing_df, truly_new], ignore_index=True)

    # Sort by date and match_num for consistency
    if "tourney_date" in merged.columns:
        merged["tourney_date"] = pd.to_numeric(merged["tourney_date"], errors="coerce")
        merged = merged.sort_values(
            ["tourney_date", "match_num"], ascending=True, na_position="last"
        ).reset_index(drop=True)

    return merged, added


def save_year_data(year: int, df: pd.DataFrame, dry_run: bool = False) -> None:
    """
    Write the updated DataFrame for a given year back to CSV.

    Preserves integer formatting for numeric columns (no trailing .0).

    Args:
        year:    Calendar year.
        df:      DataFrame in Sackmann column order.
        dry_run: If True, log intent but don't write.
    """
    path = DATA_DIR / f"atp_matches_{year}.csv"
    if dry_run:
        log.info("    [DRY RUN] Would write %d rows → %s", len(df), path.name)
        return
    df.to_csv(path, index=False)
    log.info("    ✓ Saved %d rows → %s", len(df), path.name)


# ============================================================
# SECTION 9: ELO RATING SYSTEM (BONUS FEATURE)
# ============================================================

class EloRatingSystem:
    """
    Computes and maintains ELO ratings for ATP players.

    Features:
      - Global ELO rating per player.
      - Surface-specific ELO (Hard / Clay / Grass / Carpet).
      - K-factor varies by tournament prestige (Grand Slam > Masters > etc.).
      - Ratings are computed sequentially in match date order to prevent leakage.

    The ELO formula:
      E_a = 1 / (1 + 10^((R_b - R_a) / 400))   (expected win probability)
      R_a' = R_a + K * (S_a - E_a)               (updated rating)
        where S_a = 1 for a win, 0 for a loss.
    """

    def __init__(self, default_rating: float = ELO_DEFAULT_RATING):
        self.default_rating = default_rating
        # global_ratings[player_name] = float
        self.global_ratings: Dict[str, float] = {}
        # surface_ratings[(player_name, surface)] = float
        self.surface_ratings: Dict[Tuple[str, str], float] = {}

    def _get_global(self, player: str) -> float:
        return self.global_ratings.get(player, self.default_rating)

    def _get_surface(self, player: str, surface: str) -> float:
        return self.surface_ratings.get((player, surface), self.default_rating)

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Logistic win-probability for player A given ratings."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        winner: str,
        loser: str,
        surface: str,
        tourney_level: str,
    ) -> None:
        """
        Update ELO ratings after a single completed match.

        Args:
            winner:       Name of the match winner.
            loser:        Name of the match loser.
            surface:      Court surface (Hard / Clay / Grass / Carpet).
            tourney_level: Sackmann level code (G / M / A / S / F).
        """
        k = ELO_K_FACTORS.get(str(tourney_level).upper(), ELO_K_DEFAULT)
        surface = str(surface).strip().title()

        # --- Global ELO ---
        r_w = self._get_global(winner)
        r_l = self._get_global(loser)
        e_w = self._expected(r_w, r_l)
        self.global_ratings[winner] = r_w + k * (1.0 - e_w)
        self.global_ratings[loser] = r_l + k * (0.0 - (1.0 - e_w))

        # --- Surface ELO ---
        sr_w = self._get_surface(winner, surface)
        sr_l = self._get_surface(loser, surface)
        se_w = self._expected(sr_w, sr_l)
        self.surface_ratings[(winner, surface)] = sr_w + k * (1.0 - se_w)
        self.surface_ratings[(loser, surface)] = sr_l + k * (0.0 - (1.0 - se_w))

    def compute_from_dataframe(self, df: pd.DataFrame) -> "EloRatingSystem":
        """
        Process all matches chronologically and build ELO ratings.

        The DataFrame must include: tourney_date, winner_name, loser_name,
        surface, tourney_level.

        Args:
            df: Full historical match DataFrame (all years).

        Returns:
            self (for chaining).
        """
        log.info("Computing ELO ratings from %d matches…", len(df))
        df = df.copy()
        df["tourney_date"] = pd.to_datetime(
            df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        df = df.sort_values("tourney_date", na_position="last").reset_index(drop=True)

        processed = 0
        for _, row in df.iterrows():
            winner = str(row.get("winner_name", "")).strip()
            loser = str(row.get("loser_name", "")).strip()
            surface = str(row.get("surface", "Hard")).strip()
            level = str(row.get("tourney_level", "A")).strip()

            if not winner or not loser or winner == "nan" or loser == "nan":
                continue

            self.update(winner, loser, surface, level)
            processed += 1

        log.info("ELO computation done — %d matches processed, %d unique players.",
                 processed, len(self.global_ratings))
        return self

    def get_rating(self, player: str, surface: Optional[str] = None) -> float:
        """
        Retrieve a player's current ELO rating.

        Args:
            player:  Player name.
            surface: If supplied, returns surface-specific ELO; otherwise global.

        Returns:
            ELO rating (float).
        """
        if surface:
            return self._get_surface(player, surface.strip().title())
        return self._get_global(player)

    def top_players(self, n: int = 20, surface: Optional[str] = None) -> List[Tuple[str, float]]:
        """Return the top N players by ELO, optionally for a specific surface."""
        if surface:
            s = surface.strip().title()
            rated = {
                name: self._get_surface(name, s)
                for name in self.global_ratings  # only players who have played
            }
        else:
            rated = dict(self.global_ratings)

        return sorted(rated.items(), key=lambda x: x[1], reverse=True)[:n]

    def save(self, path: Path, dry_run: bool = False) -> None:
        """
        Serialise ELO ratings to a JSON file.

        Output structure:
          {
            "computed_at": "2026-05-24T12:00:00",
            "global": {"Player Name": 1623.4, ...},
            "surface": {"Player Name|Hard": 1598.7, ...}
          }

        Args:
            path:    Output file path.
            dry_run: If True, don't write.
        """
        payload = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "global": self.global_ratings,
            "surface": {f"{p}|{s}": r for (p, s), r in self.surface_ratings.items()},
        }
        if dry_run:
            log.info("[DRY RUN] Would save ELO ratings → %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("ELO ratings saved → %s (%d players)", path, len(self.global_ratings))

    @classmethod
    def load(cls, path: Path) -> "EloRatingSystem":
        """
        Load previously saved ELO ratings from JSON.

        Args:
            path: Path to the JSON file.

        Returns:
            EloRatingSystem with ratings restored.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        elo = cls()
        elo.global_ratings = payload.get("global", {})
        for key, rating in payload.get("surface", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                elo.surface_ratings[(parts[0], parts[1])] = rating
        log.info("ELO ratings loaded from %s (%d players)", path, len(elo.global_ratings))
        return elo


# ============================================================
# SECTION 10: MAIN UPDATE PIPELINE
# ============================================================

def run_update(
    force_years: Optional[List[int]] = None,
    compute_elo: bool = False,
    no_backup: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Full data update pipeline:
      1. Back up existing CSV files.
      2. Detect what years need refreshing.
      3. Fetch each year from TML-Database.
      4. Normalize & filter to ATP main tour only.
      5. Merge new matches into existing local files (no duplicates).
      6. Save updated CSV files.
      7. Optionally compute and save ELO ratings.

    Args:
        force_years:  If set, override auto-detected year list.
        compute_elo:  If True, recompute ELO ratings after update.
        no_backup:    If True, skip the backup step.
        dry_run:      If True, perform all reads/merges but don't write anything.

    Returns:
        Dict mapping year → number of new matches added.
    """
    log.info("=" * 60)
    log.info("STEP 1/3 — DATA UPDATE")
    log.info("=" * 60)

    # 1. Backup
    if not no_backup:
        backup_data_files(dry_run=dry_run)

    # 2. Detect what needs updating
    latest_date = get_latest_tournament_date()
    years = get_years_needing_update(latest_date, force_years)

    if not years:
        log.info("Dataset is already up-to-date. Nothing to fetch.")
        return {}

    summary: Dict[str, int] = {}
    total_added = 0

    for year in years:
        log.info("")
        log.info("--- Processing year %d ---", year)

        # 3. Fetch from TML
        raw_df = fetch_year_from_tml(year)
        if raw_df is None:
            log.warning("Skipping %d — no data fetched.", year)
            summary[str(year)] = 0
            continue

        # Drop rows with missing player names (unparseable / incomplete results)
        before = len(raw_df)
        raw_df = raw_df.dropna(subset=["winner_name", "loser_name"])
        raw_df = raw_df[
            raw_df["winner_name"].astype(str).str.strip().ne("") &
            raw_df["loser_name"].astype(str).str.strip().ne("")
        ]
        if len(raw_df) < before:
            log.info("    Dropped %d rows with missing player names.", before - len(raw_df))

        # 4. Filter & normalize
        raw_df = filter_to_atp_main_tour(raw_df)
        mapped_df = map_to_sackmann_format(raw_df)
        mapped_df = coerce_numeric_columns(mapped_df)

        # 5. Load existing local data and merge
        existing_df = load_existing_year(year)
        merged_df, added = merge_new_into_existing(existing_df, mapped_df)

        # 6. Save
        save_year_data(year, merged_df, dry_run=dry_run)

        summary[str(year)] = added
        total_added += added

    # 7. ELO (optional)
    if compute_elo and not dry_run:
        _recompute_elo()

    log.info("")
    log.info("=" * 60)
    log.info("DATA UPDATE SUMMARY")
    log.info("=" * 60)
    for yr, cnt in summary.items():
        log.info("  %s: %d new match(es) added", yr, cnt)
    log.info("  Total new matches: %d", total_added)
    log.info("=" * 60)

    return summary


def _recompute_elo() -> None:
    """Internal helper: loads all data and recomputes ELO ratings."""
    try:
        # Import lazily to avoid circular issues
        sys.path.insert(0, str(PROJECT_DIR))
        from data_loader import load_data  # type: ignore

        log.info("Loading full dataset for ELO computation…")
        df = load_data(data_path=str(DATA_DIR), include_types=["tour"])
        if df.empty:
            log.warning("No data loaded; skipping ELO computation.")
            return
        elo = EloRatingSystem()
        elo.compute_from_dataframe(df)
        elo.save(ELO_OUTPUT_PATH)
        # Print top 10 players as a sanity check
        top = elo.top_players(n=10)
        log.info("Top 10 ELO ratings (global):")
        for i, (name, rating) in enumerate(top, 1):
            log.info("  %2d. %-30s %.1f", i, name, rating)
    except Exception as exc:
        log.error("ELO computation failed: %s", exc, exc_info=True)


# ============================================================
# SECTION 11: RETRAINING PIPELINE
# ============================================================

def retrain_model(
    selected_models: Optional[List[str]] = None,
    no_backup: bool = False,
    dry_run: bool = False,
    start_year: int = 1968,
    end_year: Optional[int] = None,
) -> bool:
    """
    Full model retraining pipeline using the existing project infrastructure.

    Steps:
      1. Back up existing models.
      2. Load the full dataset using data_loader.load_data().
      3. Run temporal symmetric feature engineering.
      4. Split into train/test by date.
      5. Train selected (or all) models via ModelTrainer.
      6. Save trained models, scalers, evaluation results, and feature names.

    Why retrain from scratch?
      feature_engineering.compute_symmetric_features() makes a single
      sequential pass over ALL data sorted by date to build rolling stats
      (win rates, form, h2h, surface win rates). Adding new matches changes
      the rolling stats for every subsequent match, so incremental-update
      of trained weights is not feasible — a full retrain is both correct
      and consistent with the existing pipeline.

    Args:
        selected_models: List of model keys to train (None = train all).
                         Valid keys: logistic_regression, knn, svm,
                         random_forest, xgboost, neural_network.
        no_backup:       If True, skip backing up existing models.
        dry_run:         If True, perform feature engineering but don't train.
        start_year:      Earliest year of data to include in training.
        end_year:        Latest year to include (default: current year).

    Returns:
        True if training completed successfully, False otherwise.
    """
    log.info("=" * 60)
    log.info("STEP 2/3 — MODEL RETRAINING")
    log.info("=" * 60)

    # Validate model selection
    if selected_models:
        invalid = [m for m in selected_models if m not in VALID_MODELS]
        if invalid:
            log.error(
                "Unknown model key(s): %s. Valid: %s", invalid, VALID_MODELS
            )
            return False
        log.info("Selected models: %s", selected_models)
    else:
        log.info("Training all models: %s", VALID_MODELS)

    if end_year is None:
        end_year = datetime.now().year

    # 1. Backup existing models
    if not no_backup:
        backup_models(dry_run=dry_run)

    if dry_run:
        log.info("[DRY RUN] Would retrain models. Skipping actual training.")
        return True

    # 2. Import project modules
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from data_loader import load_data  # type: ignore
        from feature_engineering import (  # type: ignore
            prepare_features_for_training,
            split_data_by_date,
        )
        from model_trainer import ModelTrainer  # type: ignore
    except ImportError as exc:
        log.error(
            "Cannot import project module: %s\n"
            "Make sure auto_update_and_retrain.py is in the project root.",
            exc,
        )
        return False

    # 3. Load data
    log.info(
        "Loading data from '%s' (years %d–%d)…", DATA_DIR, start_year, end_year
    )
    try:
        df = load_data(
            data_path=str(DATA_DIR),
            start_year=start_year,
            end_year=end_year,
            include_types=["tour"],
        )
    except Exception as exc:
        log.error("Failed to load data: %s", exc, exc_info=True)
        return False

    if df.empty:
        log.error(
            "No data loaded from '%s'. Cannot retrain.", DATA_DIR
        )
        return False

    log.info("Loaded %d rows across %d columns.", len(df), len(df.columns))

    # 4. Feature engineering — single temporal symmetric pass (no leakage)
    log.info("Running temporal symmetric feature engineering…")
    try:
        _, _, feature_names, encoders, df_feats = prepare_features_for_training(df)
    except Exception as exc:
        log.error("Feature engineering failed: %s", exc, exc_info=True)
        return False

    log.info("Feature engineering complete — %d features, %d usable rows.",
             len(feature_names), len(df_feats))

    # 5. Temporal train / test split (most recent 20% = test)
    log.info("Splitting data temporally (80%% train / 20%% test by date)…")
    try:
        train_df, test_df = split_data_by_date(df_feats, test_size=0.2)
    except Exception as exc:
        log.error("Train/test split failed: %s", exc, exc_info=True)
        return False

    X_train = train_df[feature_names].values
    y_train = train_df["target"].values
    X_test = test_df[feature_names].values
    y_test = test_df["target"].values

    log.info(
        "Train: %d rows (up to %s)  |  Test: %d rows (from %s)",
        len(X_train),
        train_df["tourney_date"].max().strftime("%Y-%m-%d"),
        len(X_test),
        test_df["tourney_date"].min().strftime("%Y-%m-%d"),
    )

    # 6. Train
    log.info("Starting model training…")
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        trainer = ModelTrainer(models_dir=str(MODELS_DIR))
        trainer.train_all_models(
            X_train, y_train, X_test, y_test,
            feature_names,
            selected_keys=selected_models if selected_models else None,
        )
    except Exception as exc:
        log.error("Model training failed: %s", exc, exc_info=True)
        return False

    log.info("Model training complete. Artifacts saved to '%s/'.", MODELS_DIR)
    return True


def save_update_manifest(
    update_summary: Dict[str, int],
    models_trained: List[str],
    dry_run: bool = False,
) -> None:
    """
    Write a JSON manifest recording what was updated and when.
    Useful for auditing and for the next run to quickly check status.

    Args:
        update_summary: Dict of year → new rows added.
        models_trained: List of model keys retrained.
        dry_run:        If True, don't write.
    """
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "data_update": update_summary,
        "models_retrained": models_trained,
        "dry_run": dry_run,
    }
    path = PROJECT_DIR / "auto_update_manifest.json"
    if dry_run:
        log.info("[DRY RUN] Would write manifest → %s", path.name)
        log.info("Manifest content: %s", json.dumps(manifest, indent=2))
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest written → %s", path.name)


# ============================================================
# SECTION 12: CLI & MAIN ORCHESTRATION
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace with parsed argument values.
    """
    parser = argparse.ArgumentParser(
        prog="auto_update_and_retrain",
        description=(
            "Automatically update the ATP tennis dataset from TML-Database "
            "and retrain all prediction models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_update_and_retrain.py
  python auto_update_and_retrain.py --update-only
  python auto_update_and_retrain.py --retrain-only
  python auto_update_and_retrain.py --years 2025 2026
  python auto_update_and_retrain.py --models xgboost random_forest
  python auto_update_and_retrain.py --no-backup --dry-run
  python auto_update_and_retrain.py --elo
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update-only",
        action="store_true",
        help="Only update the dataset; skip model retraining.",
    )
    mode.add_argument(
        "--retrain-only",
        action="store_true",
        help="Only retrain models using existing local data; skip fetching.",
    )

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        metavar="YEAR",
        help="Specific calendar years to fetch/update (e.g. --years 2025 2026).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=VALID_MODELS,
        metavar="MODEL",
        help=(
            "Models to retrain. Default: all. "
            f"Choices: {', '.join(VALID_MODELS)}"
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1968,
        metavar="YEAR",
        help="Earliest year of data to include when retraining (default: 1968).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the automatic backup of CSV files and models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview all operations without writing any files.",
    )
    parser.add_argument(
        "--elo",
        action="store_true",
        help="Compute and save ELO ratings after data update.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG for extra detail.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point. Orchestrates:
      1. Data update (fetch → filter → merge → save)
      2. Model retraining (load → feature-eng → split → train → save)
      3. ELO recomputation (optional)

    The function exits with code 0 on success, 1 on any failure.
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("AUTO UPDATE & RETRAIN — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    if args.dry_run:
        log.info("*** DRY RUN MODE — no files will be modified ***")

    # Ensure data directory exists
    if not DATA_DIR.exists():
        log.error(
            "Data directory not found: %s\n"
            "Please ensure the 'tennis_atp-master' folder exists in the project root.",
            DATA_DIR,
        )
        sys.exit(1)

    update_summary: Dict[str, int] = {}
    retrain_success = True

    # ---- Phase 1: Data Update ----
    if not args.retrain_only:
        try:
            update_summary = run_update(
                force_years=args.years,
                compute_elo=args.elo,
                no_backup=args.no_backup,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            log.warning("Data update interrupted by user.")
            sys.exit(1)
        except Exception as exc:
            log.error("Data update failed unexpectedly: %s", exc, exc_info=True)
            sys.exit(1)
    else:
        log.info("--retrain-only: skipping data update phase.")
        if args.elo:
            _recompute_elo()

    # ---- Phase 2: Model Retraining ----
    if not args.update_only:
        try:
            retrain_success = retrain_model(
                selected_models=args.models,
                no_backup=args.no_backup,
                dry_run=args.dry_run,
                start_year=args.start_year,
            )
        except KeyboardInterrupt:
            log.warning("Retraining interrupted by user.")
            sys.exit(1)
        except Exception as exc:
            log.error("Retraining failed unexpectedly: %s", exc, exc_info=True)
            retrain_success = False
    else:
        log.info("--update-only: skipping model retraining phase.")

    # ---- Phase 3: Write manifest ----
    models_trained = args.models if args.models else (VALID_MODELS if not args.update_only else [])
    save_update_manifest(update_summary, models_trained, dry_run=args.dry_run)

    # ---- Final summary ----
    log.info("")
    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info("=" * 60)

    if update_summary:
        total_new = sum(update_summary.values())
        log.info("Data update: %d new matches added across %d year(s).",
                 total_new, len(update_summary))
    elif not args.retrain_only:
        log.info("Data update: dataset already up-to-date.")

    if not args.update_only:
        if retrain_success:
            log.info("Model retraining: completed successfully.")
        else:
            log.error("Model retraining: FAILED.")

    if args.elo:
        elo_status = "saved" if ELO_OUTPUT_PATH.exists() else "not saved"
        log.info("ELO ratings: %s → %s", elo_status, ELO_OUTPUT_PATH)

    log.info("=" * 60)

    if not retrain_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
