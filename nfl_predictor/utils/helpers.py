"""
Helpers - General helper functions.

This module contains:
- get_current_week: Determine current NFL week
- get_season_weeks: Get list of weeks in a season
- calculate_days_between: Calculate days between dates
- format_spread: Format spread for display
- format_probability: Format probability for display
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def get_current_week(season: Optional[int] = None) -> int:
    """
    Determine the current NFL week based on date.

    Args:
        season: Season year (defaults to current year)

    Returns:
        Current week number (1-18 for regular season)
    """
    today = datetime.now()

    if season is None:
        # NFL season spans two calendar years
        # Season starts in September
        if today.month >= 9:
            season = today.year
        else:
            season = today.year - 1

    # Approximate NFL week 1 start (first Thursday after Labor Day)
    # Labor Day is first Monday of September
    sept_1 = datetime(season, 9, 1)
    days_until_monday = (7 - sept_1.weekday()) % 7
    if sept_1.weekday() == 0:  # If Sept 1 is Monday
        labor_day = sept_1
    else:
        labor_day = sept_1 + timedelta(days=days_until_monday)

    # Week 1 starts the Thursday after Labor Day
    week_1_start = labor_day + timedelta(days=3)

    # Calculate weeks since start
    days_since_start = (today - week_1_start).days

    if days_since_start < 0:
        return 0  # Preseason

    week = (days_since_start // 7) + 1

    # Cap at 18 for regular season
    return min(week, 18)


def get_season_weeks(include_playoffs: bool = False) -> List[int]:
    """
    Get list of weeks in an NFL season.

    Args:
        include_playoffs: Whether to include playoff weeks

    Returns:
        List of week numbers
    """
    if include_playoffs:
        return list(range(1, 23))  # Weeks 1-22
    return list(range(1, 19))  # Weeks 1-18


def get_playoff_weeks() -> List[Tuple[int, str]]:
    """
    Get playoff weeks with round names.

    Returns:
        List of (week_number, round_name) tuples
    """
    return [
        (19, "Wild Card"),
        (20, "Divisional"),
        (21, "Conference Championship"),
        (22, "Super Bowl"),
    ]


def calculate_days_between(
    date1: datetime,
    date2: datetime
) -> int:
    """
    Calculate days between two dates.

    Args:
        date1: First date
        date2: Second date

    Returns:
        Number of days (absolute value)
    """
    return abs((date2 - date1).days)


def format_spread(spread: float, include_sign: bool = True) -> str:
    """
    Format spread for display.

    Args:
        spread: Point spread value
        include_sign: Whether to include +/- sign

    Returns:
        Formatted spread string
    """
    if include_sign:
        return f"{spread:+.1f}"
    return f"{spread:.1f}"


def format_probability(prob: float, as_percentage: bool = True) -> str:
    """
    Format probability for display.

    Args:
        prob: Probability value (0-1)
        as_percentage: Whether to format as percentage

    Returns:
        Formatted probability string
    """
    if as_percentage:
        return f"{prob:.1%}"
    return f"{prob:.3f}"


def format_record(wins: int, losses: int, ties: int = 0) -> str:
    """
    Format win-loss-tie record.

    Args:
        wins: Number of wins
        losses: Number of losses
        ties: Number of ties

    Returns:
        Formatted record string
    """
    if ties > 0:
        return f"{wins}-{losses}-{ties}"
    return f"{wins}-{losses}"


def calculate_win_percentage(wins: int, losses: int, ties: int = 0) -> float:
    """
    Calculate win percentage.

    Args:
        wins: Number of wins
        losses: Number of losses
        ties: Number of ties (count as half wins)

    Returns:
        Win percentage (0-1)
    """
    total = wins + losses + ties
    if total == 0:
        return 0.0
    return (wins + 0.5 * ties) / total


def get_division_for_team(team: str) -> Optional[str]:
    """
    Get division for a team.

    Args:
        team: Team abbreviation

    Returns:
        Division name or None if not found
    """
    from nfl_predictor.constants import NFL_DIVISIONS

    for conference, divisions in NFL_DIVISIONS.items():
        for division, teams in divisions.items():
            if team in teams:
                return division
    return None


def get_conference_for_team(team: str) -> Optional[str]:
    """
    Get conference for a team.

    Args:
        team: Team abbreviation

    Returns:
        Conference name (AFC/NFC) or None if not found
    """
    from nfl_predictor.constants import NFL_DIVISIONS

    for conference, divisions in NFL_DIVISIONS.items():
        for division, teams in divisions.items():
            if team in teams:
                return conference
    return None


def is_division_game(team1: str, team2: str) -> bool:
    """
    Check if two teams are in the same division.

    Args:
        team1: First team abbreviation
        team2: Second team abbreviation

    Returns:
        True if teams are in same division
    """
    div1 = get_division_for_team(team1)
    div2 = get_division_for_team(team2)
    return div1 is not None and div1 == div2


def is_conference_game(team1: str, team2: str) -> bool:
    """
    Check if two teams are in the same conference.

    Args:
        team1: First team abbreviation
        team2: Second team abbreviation

    Returns:
        True if teams are in same conference
    """
    conf1 = get_conference_for_team(team1)
    conf2 = get_conference_for_team(team2)
    return conf1 is not None and conf1 == conf2


def get_timezone_difference(team1: str, team2: str) -> int:
    """
    Get timezone difference between two teams (simplified).

    Args:
        team1: First team abbreviation
        team2: Second team abbreviation

    Returns:
        Timezone difference in hours (0-3)
    """
    # Simplified timezone mapping
    east_coast = {"MIA", "BUF", "NYJ", "NYG", "NE", "PHI", "PIT", "BAL", "CLE", "CIN", "WAS", "CAR", "ATL", "TB", "JAX"}
    central = {"MIN", "GB", "CHI", "DET", "NO", "DAL", "HOU", "TEN", "IND", "KC"}
    mountain = {"DEN", "ARI"}
    west_coast = {"SEA", "SF", "LAR", "LAC", "LV"}

    def get_tz(team: str) -> int:
        if team in east_coast:
            return 0
        elif team in central:
            return 1
        elif team in mountain:
            return 2
        elif team in west_coast:
            return 3
        return 1  # Default to central

    return abs(get_tz(team1) - get_tz(team2))
