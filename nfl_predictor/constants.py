"""
NFL Constants - Teams, divisions, conferences, and reference data.

This module contains all the static NFL data used throughout the prediction system.
"""

from typing import Dict, List, Tuple

# NFL Team Structure by Conference and Division
NFL_TEAMS: Dict[str, Dict[str, List[str]]] = {
    "AFC": {
        "East": ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"],
        "North": ["Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers"],
        "South": ["Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans"],
        "West": ["Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers"]
    },
    "NFC": {
        "East": ["Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders"],
        "North": ["Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings"],
        "South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
        "West": ["Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"]
    }
}

# All 32 NFL teams as a flat list
ALL_NFL_TEAMS: List[str] = []
for conf, divisions in NFL_TEAMS.items():
    for div, teams in divisions.items():
        ALL_NFL_TEAMS.extend(teams)

# NFL Divisions with abbreviations (flat structure)
NFL_DIVISIONS_FLAT: Dict[str, List[str]] = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SEA", "SF"],
}

# NFL Divisions by conference (hierarchical structure)
NFL_DIVISIONS: Dict[str, Dict[str, List[str]]] = {
    "AFC": {
        "AFC East": ["BUF", "MIA", "NE", "NYJ"],
        "AFC North": ["BAL", "CIN", "CLE", "PIT"],
        "AFC South": ["HOU", "IND", "JAX", "TEN"],
        "AFC West": ["DEN", "KC", "LAC", "LV"],
    },
    "NFC": {
        "NFC East": ["DAL", "NYG", "PHI", "WAS"],
        "NFC North": ["CHI", "DET", "GB", "MIN"],
        "NFC South": ["ATL", "CAR", "NO", "TB"],
        "NFC West": ["ARI", "LAR", "SEA", "SF"],
    },
}

# List of conference names
NFL_CONFERENCES: List[str] = ["AFC", "NFC"]

# Team abbreviation to full name mapping
TEAM_ABBREVIATIONS: Dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

# Reverse mapping: full name to abbreviation
TEAM_TO_ABBREVIATION: Dict[str, str] = {v: k for k, v in TEAM_ABBREVIATIONS.items()}

# Team to division mapping (using abbreviations)
TEAM_TO_DIVISION: Dict[str, str] = {
    team: div for div, teams in NFL_DIVISIONS_FLAT.items() for team in teams
}

# Team to conference mapping (using abbreviations)
TEAM_TO_CONFERENCE: Dict[str, str] = {
    team: "AFC" if div.startswith("AFC") else "NFC"
    for team, div in TEAM_TO_DIVISION.items()
}

# Stadium coordinates for travel distance calculation (latitude, longitude)
STADIUM_COORDINATES: Dict[str, Tuple[float, float]] = {
    "Buffalo Bills": (42.7738, -78.7870),
    "Miami Dolphins": (25.9580, -80.2389),
    "New England Patriots": (42.0909, -71.2643),
    "New York Jets": (40.8135, -74.0745),
    "Baltimore Ravens": (39.2780, -76.6227),
    "Cincinnati Bengals": (39.0955, -84.5160),
    "Cleveland Browns": (41.5061, -81.6995),
    "Pittsburgh Steelers": (40.4468, -80.0158),
    "Houston Texans": (29.6847, -95.4107),
    "Indianapolis Colts": (39.7601, -86.1639),
    "Jacksonville Jaguars": (30.3239, -81.6373),
    "Tennessee Titans": (36.1665, -86.7713),
    "Denver Broncos": (39.7439, -105.0201),
    "Kansas City Chiefs": (39.0489, -94.4839),
    "Las Vegas Raiders": (36.0909, -115.1833),
    "Los Angeles Chargers": (33.9535, -118.3392),
    "Dallas Cowboys": (32.7473, -97.0945),
    "New York Giants": (40.8135, -74.0745),
    "Philadelphia Eagles": (39.9008, -75.1675),
    "Washington Commanders": (38.9076, -76.8645),
    "Chicago Bears": (41.8623, -87.6167),
    "Detroit Lions": (42.3400, -83.0456),
    "Green Bay Packers": (44.5013, -88.0622),
    "Minnesota Vikings": (44.9736, -93.2575),
    "Atlanta Falcons": (33.7553, -84.4006),
    "Carolina Panthers": (35.2258, -80.8528),
    "New Orleans Saints": (29.9511, -90.0812),
    "Tampa Bay Buccaneers": (27.9759, -82.5033),
    "Arizona Cardinals": (33.5276, -112.2626),
    "Los Angeles Rams": (33.9535, -118.3392),
    "San Francisco 49ers": (37.4032, -121.9698),
    "Seattle Seahawks": (47.5952, -122.3316),
}

# Stadium altitudes (feet above sea level)
STADIUM_ALTITUDES: Dict[str, int] = {
    "Denver Broncos": 5280,
    "Arizona Cardinals": 1086,
    "Las Vegas Raiders": 2001,
    "Kansas City Chiefs": 820,
    # Most others are near sea level (default to 0)
}

# Dome/indoor stadiums
DOME_STADIUMS: set = {
    "Arizona Cardinals", "Atlanta Falcons", "Dallas Cowboys",
    "Detroit Lions", "Houston Texans", "Indianapolis Colts",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams",
    "Minnesota Vikings", "New Orleans Saints",
}

# Dome teams by abbreviation
DOME_TEAMS: set = {"LV", "DET", "MIN", "NO", "ATL", "IND", "DAL", "LAR", "LAC", "ARI", "HOU"}

# Rivalry definitions with intensity multipliers
# Alias for backwards compatibility
NFL_RIVALRIES = RIVALRIES = {
    # AFC East
    ("BUF", "MIA"): 1.2,
    ("BUF", "NE"): 1.3,
    ("BUF", "NYJ"): 1.1,
    ("MIA", "NE"): 1.2,
    ("MIA", "NYJ"): 1.1,
    ("NE", "NYJ"): 1.3,
    # AFC North
    ("BAL", "CIN"): 1.2,
    ("BAL", "CLE"): 1.3,
    ("BAL", "PIT"): 1.5,
    ("CIN", "CLE"): 1.2,
    ("CIN", "PIT"): 1.2,
    ("CLE", "PIT"): 1.4,
    # AFC South
    ("HOU", "IND"): 1.2,
    ("HOU", "JAX"): 1.1,
    ("HOU", "TEN"): 1.2,
    ("IND", "JAX"): 1.1,
    ("IND", "TEN"): 1.3,
    ("JAX", "TEN"): 1.2,
    # AFC West
    ("DEN", "KC"): 1.3,
    ("DEN", "LAC"): 1.2,
    ("DEN", "LV"): 1.3,
    ("KC", "LAC"): 1.2,
    ("KC", "LV"): 1.3,
    ("LAC", "LV"): 1.2,
    # NFC East
    ("DAL", "NYG"): 1.4,
    ("DAL", "PHI"): 1.5,
    ("DAL", "WAS"): 1.4,
    ("NYG", "PHI"): 1.4,
    ("NYG", "WAS"): 1.2,
    ("PHI", "WAS"): 1.3,
    # NFC North
    ("CHI", "DET"): 1.2,
    ("CHI", "GB"): 1.5,
    ("CHI", "MIN"): 1.2,
    ("DET", "GB"): 1.3,
    ("DET", "MIN"): 1.2,
    ("GB", "MIN"): 1.4,
    # NFC South
    ("ATL", "CAR"): 1.1,
    ("ATL", "NO"): 1.4,
    ("ATL", "TB"): 1.2,
    ("CAR", "NO"): 1.2,
    ("CAR", "TB"): 1.2,
    ("NO", "TB"): 1.3,
    # NFC West
    ("ARI", "LAR"): 1.2,
    ("ARI", "SEA"): 1.2,
    ("ARI", "SF"): 1.2,
    ("LAR", "SEA"): 1.3,
    ("LAR", "SF"): 1.4,
    ("SEA", "SF"): 1.5,
}


def get_team_division(team: str) -> Tuple[str, str]:
    """
    Get conference and division for a team (full name).

    Args:
        team: Full team name (e.g., "Buffalo Bills")

    Returns:
        Tuple of (conference, division) e.g., ("AFC", "East")

    Raises:
        ValueError: If team not found
    """
    for conf, divisions in NFL_TEAMS.items():
        for div, teams in divisions.items():
            if team in teams:
                return conf, div
    raise ValueError(f"Unknown team: {team}")


def are_division_rivals(team1: str, team2: str) -> bool:
    """
    Check if two teams are in the same division.

    Args:
        team1: First team name
        team2: Second team name

    Returns:
        True if both teams are in the same division
    """
    try:
        conf1, div1 = get_team_division(team1)
        conf2, div2 = get_team_division(team2)
        return conf1 == conf2 and div1 == div2
    except ValueError:
        return False


def get_rivalry_intensity(team1: str, team2: str) -> float:
    """
    Get the rivalry intensity multiplier between two teams.

    Args:
        team1: First team abbreviation
        team2: Second team abbreviation

    Returns:
        Rivalry intensity multiplier (1.0 = no rivalry, higher = more intense)
    """
    key = tuple(sorted([team1, team2]))
    return RIVALRIES.get(key, 1.0)
