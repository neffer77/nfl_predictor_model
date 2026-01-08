"""
Team Models - Data classes for team ratings and performance data.

This module contains:
- TeamPreseasonData: Preseason projections and power ratings
- TeamInSeasonData: In-season EPA-based performance data
- BlendedTeamRating: Combined rating from preseason and in-season data
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class TeamPreseasonData:
    """
    Preseason projection data for a single NFL team.

    Sources for this data typically include:
    - Vegas win totals (over/under)
    - Power rankings from ESPN, PFF, Football Outsiders
    - Preseason EPA projections
    - Strength of schedule analysis

    Attributes:
        team: Official team name (e.g., "Buffalo Bills")
        season: NFL season year
        projected_wins: Vegas over/under win total (e.g., 10.5)
        win_total_odds: Vig-adjusted probability implied by odds
        preseason_power_rating: Composite power rating (-10 to +10 scale)
        offensive_ranking: Preseason offensive rank (1-32)
        defensive_ranking: Preseason defensive rank (1-32)
        overall_ranking: Preseason overall rank (1-32)
        strength_of_schedule: SOS rating (0.0 = average, positive = harder)
        division: NFL division
        conference: AFC or NFC
        playoff_odds: Preseason playoff probability (0-100)
        division_odds: Preseason division winner probability (0-100)
        superbowl_odds: Preseason Super Bowl probability (0-100)
        notes: Any relevant notes about projections
        source: Data source (e.g., "Vegas Consensus", "ESPN FPI")
        timestamp: When this data was recorded
    """
    team: str
    season: int
    projected_wins: float

    # Optional detailed projections
    win_total_odds: Optional[float] = None
    preseason_power_rating: float = 0.0
    offensive_ranking: Optional[int] = None
    defensive_ranking: Optional[int] = None
    overall_ranking: Optional[int] = None
    strength_of_schedule: float = 0.0

    # Team info
    division: str = ""
    conference: str = ""

    # Probability projections
    playoff_odds: Optional[float] = None
    division_odds: Optional[float] = None
    superbowl_odds: Optional[float] = None

    # Metadata
    notes: List[str] = field(default_factory=list)
    source: str = "Manual Entry"
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate and derive fields after initialization."""
        # Derive power rating from wins if not provided
        if self.preseason_power_rating == 0.0 and self.projected_wins:
            self.preseason_power_rating = self._wins_to_power_rating(self.projected_wins)

        # Validate win total
        if not 0 <= self.projected_wins <= 17:
            raise ValueError(f"Projected wins must be 0-17, got {self.projected_wins}")

        # Validate rankings if provided
        if self.overall_ranking and not 1 <= self.overall_ranking <= 32:
            raise ValueError(f"Overall ranking must be 1-32, got {self.overall_ranking}")

    def _wins_to_power_rating(self, wins: float) -> float:
        """
        Convert projected wins to power rating.

        Scale: -10 (worst) to +10 (best)
        8.5 wins = 0 (average)
        Each win above/below average = ~1.18 power rating points
        """
        average_wins = 8.5
        points_per_win = 10.0 / 8.5  # ~1.18
        return round((wins - average_wins) * points_per_win, 2)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "team": self.team,
            "season": self.season,
            "projected_wins": self.projected_wins,
            "preseason_power_rating": self.preseason_power_rating,
            "offensive_ranking": self.offensive_ranking,
            "defensive_ranking": self.defensive_ranking,
            "overall_ranking": self.overall_ranking,
            "strength_of_schedule": self.strength_of_schedule,
            "division": self.division,
            "conference": self.conference,
            "playoff_odds": self.playoff_odds,
            "source": self.source
        }


@dataclass
class TeamInSeasonData:
    """
    In-season performance data based on EPA (Expected Points Added).

    EPA measures the value of each play relative to expected points.
    Positive EPA = good play, Negative EPA = bad play.
    Typical EPA per play ranges from -0.3 (bad) to +0.3 (elite).

    Attributes:
        team: Official team name
        season: NFL season year
        week: Current week of data (1-18)
        games_played: Number of games in sample

        # Offensive EPA
        offensive_epa_per_play: Total offensive EPA per play
        passing_epa_per_play: Passing plays only
        rushing_epa_per_play: Rushing plays only
        total_offensive_plays: Number of offensive plays

        # Defensive EPA (lower is better - points allowed)
        defensive_epa_per_play: Total defensive EPA allowed per play
        pass_defense_epa_per_play: EPA allowed on passes
        rush_defense_epa_per_play: EPA allowed on rushes
        total_defensive_plays: Number of defensive plays

        # Derived ratings
        net_epa_per_play: Offensive EPA - Defensive EPA
        epa_power_rating: Converted to power rating scale

        # Record
        wins: Current wins
        losses: Current losses
        ties: Current ties
        point_differential: Points scored - points allowed
    """
    team: str
    season: int
    week: int
    games_played: int

    # Offensive EPA
    offensive_epa_per_play: float = 0.0
    passing_epa_per_play: Optional[float] = None
    rushing_epa_per_play: Optional[float] = None
    total_offensive_plays: int = 0

    # Defensive EPA (from opponent's perspective - lower is better for defense)
    defensive_epa_per_play: float = 0.0
    pass_defense_epa_per_play: Optional[float] = None
    rush_defense_epa_per_play: Optional[float] = None
    total_defensive_plays: int = 0

    # Derived (calculated in post_init)
    net_epa_per_play: float = 0.0
    epa_power_rating: float = 0.0

    # Record
    wins: int = 0
    losses: int = 0
    ties: int = 0
    point_differential: int = 0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        # Net EPA = Offensive EPA - Defensive EPA allowed
        # Higher is better (good offense, bad defense for opponents)
        self.net_epa_per_play = round(
            self.offensive_epa_per_play - self.defensive_epa_per_play, 4
        )

        # Convert to power rating scale
        self.epa_power_rating = self._epa_to_power_rating(self.net_epa_per_play)

        # Validate EPA ranges (typical range is -0.4 to +0.4)
        if not -0.5 <= self.offensive_epa_per_play <= 0.5:
            self.notes.append(f"Warning: Offensive EPA {self.offensive_epa_per_play} outside typical range")

        if not -0.5 <= self.defensive_epa_per_play <= 0.5:
            self.notes.append(f"Warning: Defensive EPA {self.defensive_epa_per_play} outside typical range")

    def _epa_to_power_rating(self, net_epa: float) -> float:
        """
        Convert net EPA per play to power rating scale.

        Typical net EPA ranges from -0.3 (bad) to +0.3 (elite).
        Scale to -10 to +10 power rating.
        """
        # ~0.03 net EPA per play = 1 power rating point
        conversion_factor = 10.0 / 0.3
        return round(net_epa * conversion_factor, 2)

    @property
    def record(self) -> str:
        """Return win-loss-tie record as string."""
        if self.ties > 0:
            return f"{self.wins}-{self.losses}-{self.ties}"
        return f"{self.wins}-{self.losses}"

    @property
    def win_percentage(self) -> float:
        """Calculate win percentage."""
        total_games = self.wins + self.losses + self.ties
        if total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / total_games

    def has_sufficient_data(self, min_plays: int = 100) -> bool:
        """Check if there's enough data for reliable EPA."""
        return (
            self.total_offensive_plays >= min_plays and
            self.total_defensive_plays >= min_plays
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "team": self.team,
            "season": self.season,
            "week": self.week,
            "games_played": self.games_played,
            "offensive_epa_per_play": self.offensive_epa_per_play,
            "defensive_epa_per_play": self.defensive_epa_per_play,
            "net_epa_per_play": self.net_epa_per_play,
            "epa_power_rating": self.epa_power_rating,
            "record": self.record,
            "point_differential": self.point_differential
        }


@dataclass
class BlendedTeamRating:
    """
    Result of blending preseason and in-season ratings.

    Attributes:
        team: Team name
        season: NFL season
        week: Current week

        # Component ratings
        preseason_rating: From preseason projections
        in_season_rating: From in-season EPA (None if no games)

        # Blending
        preseason_weight: Weight applied to preseason
        in_season_weight: Weight applied to in-season

        # Final output
        power_rating: Final blended power rating
        confidence: Confidence level based on data quality

        # Context
        games_played: Number of games in sample
        sufficient_data: Whether there's enough in-season data
        notes: Any relevant notes
    """
    team: str
    season: int
    week: int

    # Component ratings
    preseason_rating: float
    in_season_rating: Optional[float]

    # Blending weights
    preseason_weight: float
    in_season_weight: float

    # Final output
    power_rating: float = 0.0
    confidence: str = "Medium"

    # Context
    games_played: int = 0
    sufficient_data: bool = True
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate final power rating."""
        if self.in_season_rating is None:
            # No in-season data - use preseason only
            self.power_rating = self.preseason_rating
            self.confidence = "Low"
            self.notes.append("No in-season data available")
        else:
            # Blend the ratings
            self.power_rating = round(
                (self.preseason_rating * self.preseason_weight) +
                (self.in_season_rating * self.in_season_weight),
                2
            )

            # Determine confidence
            self._calculate_confidence()

    def _calculate_confidence(self):
        """Determine confidence level based on data quality."""
        if self.games_played >= 8:
            self.confidence = "High"
        elif self.games_played >= 4:
            self.confidence = "Medium"
        elif self.games_played >= 1:
            self.confidence = "Low"
        else:
            self.confidence = "Very Low"

        if not self.sufficient_data:
            self.confidence = "Low"
            self.notes.append("Insufficient play data for reliable EPA")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "team": self.team,
            "season": self.season,
            "week": self.week,
            "preseason_rating": self.preseason_rating,
            "in_season_rating": self.in_season_rating,
            "preseason_weight": self.preseason_weight,
            "in_season_weight": self.in_season_weight,
            "power_rating": self.power_rating,
            "confidence": self.confidence,
            "games_played": self.games_played,
            "notes": self.notes
        }
