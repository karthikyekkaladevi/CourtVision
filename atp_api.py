"""
ATP API client and Wikipedia draw scraper for CourtVision.
Uses Ball Don't Lie API for tournament calendar and rankings,
and Wikipedia for actual tournament draws.
"""
import os
import requests
import time
import re
from datetime import date
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ──────────────────────────────────────────────

API_BASE = "https://api.balldontlie.io/atp/v1"
API_KEY = os.getenv("ATP_API_KEY")
RATE_LIMIT_DELAY = 2.0

CATEGORY_MAP = {
    'Grand Slam': 'G',
    'Masters 1000': 'M',
    'ATP 500': 'A',
    'ATP 250': 'A',
    'ATP Finals': 'F',
}

# Maps tournament names (from API) to Wikipedia URL slugs.
# Format: {year}_SLUG
WIKI_TOURNAMENT_MAP = {
    'Australian Open': 'Australian_Open_%E2%80%93_Men%27s_singles',
    'Roland Garros': 'French_Open_%E2%80%93_Men%27s_singles',
    'Wimbledon': 'Wimbledon_Championships_%E2%80%93_Men%27s_singles',
    'US Open': 'US_Open_%E2%80%93_Men%27s_singles',
    'BNP Paribas Open': 'Indian_Wells_Masters_%E2%80%93_Men%27s_singles',
    'Miami Open presented by Itau': 'Miami_Open_%E2%80%93_Men%27s_singles',
    'Rolex Monte-Carlo Masters': 'Monte-Carlo_Masters_%E2%80%93_Men%27s_singles',
    'Mutua Madrid Open': 'Madrid_Open_%E2%80%93_Men%27s_singles',
    'Internazionali BNL d\'Italia': 'Italian_Open_%E2%80%93_Men%27s_singles',
    'National Bank Open presented by Rogers': 'Canadian_Open_%E2%80%93_Men%27s_singles',
    'Cincinnati Open': 'Cincinnati_Open_%E2%80%93_Men%27s_singles',
    'Shanghai Masters': 'Shanghai_Masters_%E2%80%93_Men%27s_singles',
}


# ─── A. Ball Don't Lie API Client ───────────────────────────

def _api_get(endpoint, params=None):
    """Make an authenticated API request with rate limiting and retry."""
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    headers = {"Authorization": API_KEY}

    for attempt in range(3):
        time.sleep(RATE_LIMIT_DELAY)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = RATE_LIMIT_DELAY * (2 ** (attempt + 1))
                print(f"  Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  API error: HTTP {resp.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  API request failed: {e}")
            return None

    print("  API request failed after 3 retries.")
    return None


def fetch_tournament_calendar():
    """Fetch the full ATP tournament calendar for the current season."""
    tournaments = []
    cursor = None

    print("  Fetching ATP calendar...")
    while True:
        params = {"per_page": 25}
        if cursor:
            params["cursor"] = cursor

        data = _api_get("tournaments", params)
        if not data or "data" not in data:
            break

        tournaments.extend(data["data"])
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor

    print(f"  Found {len(tournaments)} tournaments.")
    return tournaments


def find_next_tournament(calendar=None):
    """
    Find the next upcoming tournament.
    Returns (next_tournament, upcoming_list) where upcoming_list has up to 5 options.
    """
    if calendar is None:
        calendar = fetch_tournament_calendar()

    today = date.today().isoformat()
    upcoming = [t for t in calendar if t.get("end_date", t.get("start_date", "")) >= today]
    # Sort by start date
    upcoming.sort(key=lambda t: t["start_date"])

    if not upcoming:
        return None, []

    return upcoming[0], upcoming[:5]


def map_category_to_level(category):
    """Map API category string to simulator tourney_level code."""
    if not category:
        return 'A'
    return CATEGORY_MAP.get(category, 'A')


def fetch_top_ranked_players(count=128):
    """Fetch top N ranked players from current ATP rankings."""
    players = []
    cursor = None

    print(f"  Fetching top {count} ranked players...")
    while len(players) < count:
        params = {"per_page": min(25, count - len(players))}
        if cursor:
            params["cursor"] = cursor

        data = _api_get("rankings", params)
        if not data or "data" not in data:
            break

        for entry in data["data"]:
            player = entry.get("player", {})
            players.append({
                "name": player.get("full_name", "Unknown"),
                "seed": len(players) + 1,
                "year": None,
            })
            if len(players) >= count:
                break

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor

    print(f"  Fetched {len(players)} players.")
    return players


# ─── B. Wikipedia Draw Scraper ──────────────────────────────

def _build_wikipedia_url(tournament_name, year):
    """Build Wikipedia URL for a tournament's men's singles draw."""
    # Try exact match first
    for api_name, slug in WIKI_TOURNAMENT_MAP.items():
        if api_name.lower() in tournament_name.lower() or tournament_name.lower() in api_name.lower():
            return f"https://en.wikipedia.org/wiki/{year}_{slug}"

    # Try partial match on key words
    name_lower = tournament_name.lower()
    for keyword in ['wimbledon', 'australian', 'roland', 'french', 'us open']:
        if keyword in name_lower:
            for api_name, slug in WIKI_TOURNAMENT_MAP.items():
                if keyword in api_name.lower():
                    return f"https://en.wikipedia.org/wiki/{year}_{slug}"

    return None


def scrape_wikipedia_draw(tournament_name, year, draw_size=128):
    """
    Scrape the men's singles draw from Wikipedia.
    Returns list of players in draw order, or None on failure.
    """
    url = _build_wikipedia_url(tournament_name, year)
    if not url:
        print(f"  No Wikipedia URL mapping for '{tournament_name}'.")
        return None

    print(f"  Fetching draw from Wikipedia...")
    try:
        resp = requests.get(url, headers={"User-Agent": "CourtVision/1.0"}, timeout=15)
        if resp.status_code != 200:
            print(f"  Wikipedia returned HTTP {resp.status_code}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  Wikipedia request failed: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find section headers (Section 1 through Section 8 for Grand Slams)
    section_headers = []
    for tag in soup.find_all(["h3", "h4"]):
        text = tag.get_text(strip=True)
        if re.match(r"Section \d+", text):
            section_headers.append(tag)

    if not section_headers:
        print("  Could not find draw sections on Wikipedia page.")
        return None

    all_players = []

    for header in section_headers:
        table = header.find_next("table")
        if not table:
            continue

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue

            seed_cell = cells[1]
            name_cell = cells[2]

            # Get player name from link title (full name) or link text
            player_link = None
            for link in name_cell.find_all("a"):
                href = link.get("href", "")
                title = link.get("title", "")
                # Skip country/flag links
                if "/wiki/" in href and title and not _is_country_link(link):
                    player_link = link
                    break

            if not player_link:
                continue

            full_name = player_link.get("title", player_link.get_text(strip=True))
            if not full_name or len(full_name) < 3:
                continue

            # Clean Wikipedia disambiguation (e.g., "Nuno Borges (tennis)" -> "Nuno Borges")
            full_name = re.sub(r'\s*\((?:tennis|born \d{4})\)\s*$', '', full_name).strip()

            # Skip non-player entries
            if full_name.lower() in ["first round", "second round", "third round", "fourth round"]:
                continue

            # Parse seed/entry
            seed_text = seed_cell.get_text(strip=True)
            seed = None
            entry = None
            if seed_text.isdigit():
                seed = int(seed_text)
            elif seed_text in ("WC", "Q", "LL", "PR", "Alt", "SE"):
                entry = seed_text

            all_players.append({
                "name": full_name,
                "seed": seed,
                "entry": entry,
                "year": None,
            })

    if not all_players:
        print("  No players extracted from Wikipedia.")
        return None

    print(f"  Extracted {len(all_players)} players from draw.")
    return all_players


def _is_country_link(link):
    """Check if a Wikipedia link is a country/flag link rather than a player link."""
    # Flag icons are always img tags inside links
    if link.find("img"):
        return True
    title = link.get("title", "")
    if not title:
        return True
    # Player disambiguation like "(tennis)" or "(born 2003)" is a strong player signal
    if re.search(r'\((?:tennis|born \d{4})\)', title):
        return False
    # Player names have at least a first and last name (space-separated)
    # Single-word titles are countries (Italy, Serbia, Croatia, etc.)
    # Multi-word countries (United States, South Africa) don't have disambiguation
    # and are already caught by the img check on flag links
    if ' ' in title:
        return False
    return True


# ─── C. Reverse Seed Mapping ───────────────────────────────

def _get_seed_distribution(n):
    """
    Replicate TournamentSimulator._get_seed_distribution() to compute
    standard bracket seed positions. Must match exactly.
    """
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2]

    prev = _get_seed_distribution(n // 2)
    res = []
    for i, x in enumerate(prev):
        if i % 2 == 0:
            res.extend([x, n + 1 - x])
        else:
            res.extend([n + 1 - x, x])
    return res


def prepare_draw_for_simulator(players_in_draw_order, draw_size):
    """
    Assign synthetic seeds so that create_bracket() reconstructs
    the exact draw order from Wikipedia.

    create_bracket sorts by seed, then maps distribution[i] -> player.
    We need: player at draw_position i gets seed = distribution[i].
    Then when sorted by seed and re-distributed, they land back in position i.
    """
    distribution = _get_seed_distribution(draw_size)

    # Pad with BYEs if needed
    while len(players_in_draw_order) < draw_size:
        players_in_draw_order.append({"name": "BYE", "seed": None, "year": None})

    # Truncate if too many
    players_in_draw_order = players_in_draw_order[:draw_size]

    # Preserve original seeds for display, assign synthetic seeds for bracket placement
    for i, player in enumerate(players_in_draw_order):
        player["original_seed"] = player.get("seed")
        player["seed"] = distribution[i]

    return players_in_draw_order


def build_fallback_draw(ranked_players, draw_size):
    """
    Build a draw from ranked players (no reverse mapping needed —
    create_bracket handles standard seed distribution for sequential seeds).
    """
    players = ranked_players[:draw_size]
    # Ensure seeds are 1..N
    for i, p in enumerate(players):
        p["seed"] = i + 1
    return players
