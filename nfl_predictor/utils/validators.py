"""
Validators - Input validation utilities.

This module contains:
- ValidationError: Custom validation exception
- validate_team: Validate team abbreviation
- validate_week: Validate week number
- validate_season: Validate season year
- validate_spread: Validate point spread
- validate_probability: Validate probability value
"""

from typing import Optional, List

from nfl_predictor.constants import TEAM_ABBREVIATIONS

# Set of valid team abbreviations
NFL_TEAM_ABBREVS = set(TEAM_ABBREVIATIONS.keys())


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: any = None):
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
        """
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.field:
            return f"Validation error on '{self.field}': {self.message} (got: {self.value})"
        return self.message


def validate_team(team: str, raise_error: bool = True) -> bool:
    """
    Validate team abbreviation.

    Args:
        team: Team abbreviation to validate
        raise_error: Whether to raise exception on invalid

    Returns:
        True if valid, False if invalid (when raise_error=False)

    Raises:
        ValidationError: If team is invalid and raise_error=True
    """
    if not team:
        if raise_error:
            raise ValidationError("Team cannot be empty", field="team", value=team)
        return False

    team_upper = team.upper()
    if team_upper not in NFL_TEAM_ABBREVS:
        if raise_error:
            raise ValidationError(
                f"Invalid team abbreviation. Valid teams: {', '.join(sorted(NFL_TEAM_ABBREVS))}",
                field="team",
                value=team
            )
        return False

    return True


def validate_week(week: int, raise_error: bool = True) -> bool:
    """
    Validate week number.

    Args:
        week: Week number to validate (1-18 regular season, up to 22 with playoffs)
        raise_error: Whether to raise exception on invalid

    Returns:
        True if valid, False if invalid (when raise_error=False)

    Raises:
        ValidationError: If week is invalid and raise_error=True
    """
    if not isinstance(week, int):
        if raise_error:
            raise ValidationError(
                "Week must be an integer",
                field="week",
                value=week
            )
        return False

    if week < 1 or week > 22:
        if raise_error:
            raise ValidationError(
                "Week must be between 1 and 22",
                field="week",
                value=week
            )
        return False

    return True


def validate_season(season: int, raise_error: bool = True) -> bool:
    """
    Validate season year.

    Args:
        season: Season year to validate
        raise_error: Whether to raise exception on invalid

    Returns:
        True if valid, False if invalid (when raise_error=False)

    Raises:
        ValidationError: If season is invalid and raise_error=True
    """
    if not isinstance(season, int):
        if raise_error:
            raise ValidationError(
                "Season must be an integer",
                field="season",
                value=season
            )
        return False

    # NFL seasons from 1920 to current + 1
    min_season = 1920
    max_season = 2030  # Allow some future years

    if season < min_season or season > max_season:
        if raise_error:
            raise ValidationError(
                f"Season must be between {min_season} and {max_season}",
                field="season",
                value=season
            )
        return False

    return True


def validate_spread(spread: float, raise_error: bool = True) -> bool:
    """
    Validate point spread.

    Args:
        spread: Point spread to validate
        raise_error: Whether to raise exception on invalid

    Returns:
        True if valid, False if invalid (when raise_error=False)

    Raises:
        ValidationError: If spread is invalid and raise_error=True
    """
    if not isinstance(spread, (int, float)):
        if raise_error:
            raise ValidationError(
                "Spread must be a number",
                field="spread",
                value=spread
            )
        return False

    # Reasonable spread range
    max_spread = 50.0

    if abs(spread) > max_spread:
        if raise_error:
            raise ValidationError(
                f"Spread magnitude cannot exceed {max_spread}",
                field="spread",
                value=spread
            )
        return False

    return True


def validate_probability(prob: float, raise_error: bool = True) -> bool:
    """
    Validate probability value.

    Args:
        prob: Probability to validate (should be 0-1)
        raise_error: Whether to raise exception on invalid

    Returns:
        True if valid, False if invalid (when raise_error=False)

    Raises:
        ValidationError: If probability is invalid and raise_error=True
    """
    if not isinstance(prob, (int, float)):
        if raise_error:
            raise ValidationError(
                "Probability must be a number",
                field="probability",
                value=prob
            )
        return False

    if prob < 0.0 or prob > 1.0:
        if raise_error:
            raise ValidationError(
                "Probability must be between 0 and 1",
                field="probability",
                value=prob
            )
        return False

    return True


def validate_game_input(
    home_team: str,
    away_team: str,
    week: int,
    season: int,
    raise_error: bool = True
) -> bool:
    """
    Validate complete game input.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        week: Week number
        season: Season year
        raise_error: Whether to raise exception on invalid

    Returns:
        True if all inputs valid

    Raises:
        ValidationError: If any input is invalid and raise_error=True
    """
    all_valid = True

    if not validate_team(home_team, raise_error):
        all_valid = False
    if not validate_team(away_team, raise_error):
        all_valid = False
    if not validate_week(week, raise_error):
        all_valid = False
    if not validate_season(season, raise_error):
        all_valid = False

    # Teams must be different
    if home_team and away_team and home_team.upper() == away_team.upper():
        if raise_error:
            raise ValidationError(
                "Home and away teams must be different",
                field="teams",
                value=f"{home_team} vs {away_team}"
            )
        all_valid = False

    return all_valid


def validate_scores(
    home_score: int,
    away_score: int,
    raise_error: bool = True
) -> bool:
    """
    Validate game scores.

    Args:
        home_score: Home team score
        away_score: Away team score
        raise_error: Whether to raise exception on invalid

    Returns:
        True if scores valid

    Raises:
        ValidationError: If scores invalid and raise_error=True
    """
    for name, score in [("home_score", home_score), ("away_score", away_score)]:
        if not isinstance(score, int):
            if raise_error:
                raise ValidationError(
                    "Score must be an integer",
                    field=name,
                    value=score
                )
            return False

        if score < 0:
            if raise_error:
                raise ValidationError(
                    "Score cannot be negative",
                    field=name,
                    value=score
                )
            return False

        if score > 100:  # Sanity check
            if raise_error:
                raise ValidationError(
                    "Score exceeds reasonable maximum",
                    field=name,
                    value=score
                )
            return False

    return True
