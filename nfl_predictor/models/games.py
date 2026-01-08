"""
Game Models - Data classes for games, matchups, and predictions.

This module contains:
- GameType: Enum for game types (regular season, playoffs, etc.)
- GameLocation: Enum for home/away/neutral
- TeamGameContext: Situational context for a team
- NFLGame: Core game/matchup data model
- GamePrediction: Complete prediction for a game
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum

from nfl_predictor.constants import are_division_rivals


class GameType(Enum):
    """Types of NFL games."""
    REGULAR_SEASON = "regular_season"
    WILD_CARD = "wild_card"
    DIVISIONAL = "divisional"
    CONFERENCE = "conference"
    SUPER_BOWL = "super_bowl"
    PRESEASON = "preseason"


class GameLocation(Enum):
    """Game location types."""
    HOME = "home"
    AWAY = "away"
    NEUTRAL = "neutral"  # e.g., Super Bowl, international games


@dataclass
class TeamGameContext:
    """
    Context for a team entering a specific game.

    Captures rest days, travel distance, timezone changes,
    previous game info, and other situational factors.

    Attributes:
        team: Team name
        days_rest: Days since last game (7 = normal, 10+ = bye)
        is_coming_off_bye: Whether team had bye week
        previous_game_was_mnf: Late Monday game = less rest
        travel_miles: Distance traveled for this game
        timezone_change: Hours of timezone shift (-3 to +3)
        is_home: Playing at home
        stadium_altitude: Altitude in feet (Denver = 5280)
        stadium_dome: Indoor stadium
        previous_opponent_quality: Power rating of last opponent
        is_division_game: Playing division rival
        is_primetime: SNF, MNF, or TNF
        is_short_week: Thursday game after Sunday
        consecutive_road_games: Number of consecutive away games
    """
    team: str
    days_rest: int = 7
    is_coming_off_bye: bool = False
    previous_game_was_mnf: bool = False
    travel_miles: float = 0.0
    timezone_change: int = 0
    is_home: bool = True
    stadium_altitude: int = 0
    stadium_dome: bool = False
    previous_opponent_quality: float = 0.0
    is_division_game: bool = False
    is_primetime: bool = False
    is_short_week: bool = False
    consecutive_road_games: int = 0


@dataclass
class NFLGame:
    """
    Represents a single NFL game/matchup.

    This is the core data model for game predictions.

    Attributes:
        game_id: Unique identifier for the game
        season: NFL season year
        week: Week number (1-18 regular, 19-22 playoffs)
        game_date: Date of the game
        game_time: Scheduled start time (EST)

        home_team: Home team name
        away_team: Away team name
        location: Game location type
        stadium: Stadium name

        game_type: Type of game (regular, playoff, etc.)
        is_divisional: Division rivalry game

        # Situational context
        home_context: TeamGameContext for home team
        away_context: TeamGameContext for away team

        # Vegas lines (for comparison/validation)
        vegas_spread: Vegas spread (negative = home favored)
        vegas_total: Over/under total
        vegas_home_ml: Home team moneyline
        vegas_away_ml: Away team moneyline

        # Results (filled in after game)
        home_score: Final home score
        away_score: Final away score
        is_completed: Whether game has been played
    """
    game_id: str
    season: int
    week: int
    game_date: date

    home_team: str
    away_team: str

    # Optional fields with defaults
    game_time: str = "1:00 PM"
    location: GameLocation = GameLocation.HOME
    stadium: str = ""
    game_type: GameType = GameType.REGULAR_SEASON
    is_divisional: bool = False

    # Context (optional - calculated if not provided)
    home_context: Optional[TeamGameContext] = None
    away_context: Optional[TeamGameContext] = None

    # Vegas lines (optional)
    vegas_spread: Optional[float] = None
    vegas_total: Optional[float] = None
    vegas_home_ml: Optional[int] = None
    vegas_away_ml: Optional[int] = None

    # Results
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    is_completed: bool = False

    def __post_init__(self):
        """Initialize derived fields."""
        # Check if divisional game
        if not self.is_divisional:
            self.is_divisional = are_division_rivals(self.home_team, self.away_team)

        # Generate game_id if not provided
        if not self.game_id:
            self.game_id = f"{self.season}_W{self.week}_{self.away_team[:3]}@{self.home_team[:3]}"

    @property
    def winner(self) -> Optional[str]:
        """Get the winning team if game is completed."""
        if not self.is_completed or self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return self.home_team
        elif self.away_score > self.home_score:
            return self.away_team
        return "TIE"

    @property
    def actual_spread(self) -> Optional[float]:
        """Get actual game spread (negative = home won by that amount)."""
        if self.home_score is None or self.away_score is None:
            return None
        return self.away_score - self.home_score

    @property
    def total_points(self) -> Optional[int]:
        """Get total points scored."""
        if self.home_score is None or self.away_score is None:
            return None
        return self.home_score + self.away_score

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "game_date": self.game_date.isoformat() if self.game_date else None,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "game_type": self.game_type.value,
            "is_divisional": self.is_divisional,
            "vegas_spread": self.vegas_spread,
            "vegas_total": self.vegas_total,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "is_completed": self.is_completed
        }


@dataclass
class GamePrediction:
    """
    Complete prediction for a single game.

    Attributes:
        game: The NFLGame being predicted

        # Power ratings at prediction time
        home_power_rating: Blended power rating for home team
        away_power_rating: Blended power rating for away team

        # Adjustments
        home_field_advantage: HFA points adjustment
        rest_travel_adjustment: Rest/travel points adjustment
        total_adjustment: Sum of all adjustments (+ = home)

        # Core predictions
        predicted_spread: Predicted point spread (- = home favored)
        home_win_probability: Probability home team wins (0-1)
        away_win_probability: Probability away team wins (0-1)

        # Confidence
        confidence_tier: High/Medium/Low based on data quality
        confidence_score: Numerical confidence (0-100)

        # Comparison to Vegas (if available)
        spread_vs_vegas: Our spread - Vegas spread
        edge: Perceived edge (positive = bet home, negative = bet away)

        # Metadata
        prediction_timestamp: When prediction was generated
        notes: Any relevant notes
    """
    game: NFLGame

    # Power ratings
    home_power_rating: float
    away_power_rating: float

    # Adjustments
    home_field_advantage: float = 0.0
    rest_travel_adjustment: float = 0.0
    total_adjustment: float = 0.0

    # Predictions (calculated in post_init)
    predicted_spread: float = 0.0
    home_win_probability: float = 0.5
    away_win_probability: float = 0.5

    # Confidence
    confidence_tier: str = "Medium"
    confidence_score: int = 50

    # Vegas comparison
    spread_vs_vegas: Optional[float] = None
    edge: Optional[float] = None

    # Metadata
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate predictions from inputs."""
        # Import here to avoid circular dependency
        from nfl_predictor.predictions.probability import LogisticWinProbability

        # Calculate total adjustment
        self.total_adjustment = self.home_field_advantage + self.rest_travel_adjustment

        # Calculate spread
        # Spread = Away rating - Home rating - Total adjustment
        # Negative spread = home team favored
        raw_diff = self.home_power_rating - self.away_power_rating
        self.predicted_spread = -(raw_diff + self.total_adjustment)
        self.predicted_spread = round(self.predicted_spread, 1)

        # Calculate win probability
        prob_model = LogisticWinProbability(scale=6.0)
        adjusted_diff = raw_diff + self.total_adjustment
        self.home_win_probability = prob_model.calculate(adjusted_diff)
        self.away_win_probability = round(1.0 - self.home_win_probability, 4)

        # Compare to Vegas if available
        if self.game.vegas_spread is not None:
            self.spread_vs_vegas = round(self.predicted_spread - self.game.vegas_spread, 1)
            # Edge: positive means model likes home more than Vegas
            self.edge = -self.spread_vs_vegas

    @property
    def pick(self) -> str:
        """Get the predicted winner."""
        if self.predicted_spread < 0:
            return self.game.home_team
        elif self.predicted_spread > 0:
            return self.game.away_team
        return "PICK"  # Dead even

    @property
    def pick_ats(self) -> Optional[str]:
        """
        Get the against-the-spread pick.

        Returns which team to bet on vs the Vegas spread.
        """
        if self.edge is None or abs(self.edge) < 1.0:
            return None  # No edge

        if self.edge > 0:
            return self.game.home_team  # Model likes home more than Vegas
        return self.game.away_team  # Model likes away more than Vegas

    @property
    def formatted_spread(self) -> str:
        """Format spread with team name."""
        if self.predicted_spread < 0:
            return f"{self.game.home_team} {self.predicted_spread}"
        elif self.predicted_spread > 0:
            return f"{self.game.away_team} -{self.predicted_spread}"
        return "PICK"

    def is_correct(self) -> Optional[bool]:
        """Check if prediction was correct (if game completed)."""
        if not self.game.is_completed:
            return None

        predicted_winner = self.pick
        actual_winner = self.game.winner

        return predicted_winner == actual_winner

    def is_correct_ats(self) -> Optional[bool]:
        """Check if ATS prediction was correct."""
        if not self.game.is_completed or self.game.vegas_spread is None:
            return None

        actual_spread = self.game.actual_spread  # Away - Home
        vegas_spread = self.game.vegas_spread

        # Did home team cover?
        home_covered = actual_spread < vegas_spread

        if self.pick_ats == self.game.home_team:
            return home_covered
        elif self.pick_ats == self.game.away_team:
            return not home_covered
        return None  # No pick made

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "game_id": self.game.game_id,
            "home_team": self.game.home_team,
            "away_team": self.game.away_team,
            "week": self.game.week,
            "home_power_rating": self.home_power_rating,
            "away_power_rating": self.away_power_rating,
            "home_field_advantage": self.home_field_advantage,
            "rest_travel_adjustment": self.rest_travel_adjustment,
            "predicted_spread": self.predicted_spread,
            "formatted_spread": self.formatted_spread,
            "home_win_probability": self.home_win_probability,
            "away_win_probability": self.away_win_probability,
            "pick": self.pick,
            "confidence_tier": self.confidence_tier,
            "vegas_spread": self.game.vegas_spread,
            "spread_vs_vegas": self.spread_vs_vegas,
            "edge": self.edge,
            "pick_ats": self.pick_ats
        }
