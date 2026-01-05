"""
NFL Prediction Model - Epics 1-6: Complete Prediction System

This module implements a complete NFL prediction system:
- Core rating system blending preseason projections with in-season EPA data
- Game predictions with spreads and win probabilities
- Comprehensive backtesting and performance analysis
- Data integration layer for fetching real NFL data from multiple sources
- Multi-format output and export capabilities
- Injury impact modeling and adjustments

Epic 1 Stories (Team Ratings):
- Story 1.1: Team Preseason Data Model
- Story 1.2: Team In-Season EPA Data Model  
- Story 1.3: Preseason-to-In-Season Decay Curve
- Story 1.4: Blended Team Rating Calculator
- Story 1.5: NFL Season Model (Full 32-Team Integration)
- Story 1.6: Rating Validation & Edge Cases

Epic 2 Stories (Game Predictions):
- Story 2.1: Game/Matchup Data Model
- Story 2.2: Home Field Advantage Calculator
- Story 2.3: Rest & Travel Adjustments
- Story 2.4: Win Probability Calculator
- Story 2.5: Spread/Line Prediction
- Story 2.6: Weekly Schedule & Predictions Manager

Epic 3 Stories (Backtesting & Validation):
- Story 3.1: Prediction Result Tracker
- Story 3.2: Accuracy Metrics Calculator
- Story 3.3: Calibration Analysis
- Story 3.4: Performance by Category Analytics
- Story 3.5: Season Backtester
- Story 3.6: Performance Report Generator

Epic 4 Stories (Data Integration & Automation):
- Story 4.1: Data Source Abstraction Layer
- Story 4.2: EPA Data Fetcher
- Story 4.3: Schedule & Results Fetcher
- Story 4.4: Vegas Lines Fetcher
- Story 4.5: Data Pipeline Orchestrator
- Story 4.6: Weekly Automation Manager

Epic 5 Stories (Output & Export):
- Story 5.1: Export Format Handlers (CSV, JSON, Markdown)
- Story 5.2: Weekly Pick Sheet Generator
- Story 5.3: Power Rankings Exporter
- Story 5.4: Betting Analysis Report
- Story 5.5: Season Dashboard Data Generator
- Story 5.6: Email/Notification Formatter

Epic 6 Stories (Injury Impact Modeling):
- Story 6.1: Player Value Model (Positional Importance, VOR)
- Story 6.2: Injury Report Data Model (NFL Designations)
- Story 6.3: Injury Data Fetcher (Mock + Real Sources)
- Story 6.4: Injury Impact Calculator
- Story 6.5: Injury-Adjusted Predictions
- Story 6.6: Injury Tracking & Validation

Author: Connor's NFL Prediction System
Version: 6.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from enum import Enum
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import math


# =============================================================================
# Story 1.1: Team Preseason Data Model
# =============================================================================

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


# =============================================================================
# Story 1.2: Team In-Season EPA Data Model
# =============================================================================

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


# =============================================================================
# Story 1.3: Preseason-to-In-Season Decay Curve
# =============================================================================

class DecayCurveType(Enum):
    """Types of decay curves for blending weights."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    CUSTOM = "custom"


@dataclass
class DecayCurveConfig:
    """
    Configuration for the preseason-to-in-season decay curve.
    
    The decay curve determines how much weight to give preseason projections
    vs in-season EPA data as the season progresses.
    
    Attributes:
        curve_type: Type of decay function
        start_weight: Preseason weight at week 0 (typically 0.67-0.80)
        floor_weight: Minimum preseason weight (typically 0.15-0.25)
        transition_weeks: Weeks until floor is reached (typically 8-12)
        inflection_point: For sigmoid, week where weight = 50% (typically 4-6)
        steepness: For sigmoid, how sharp the transition is
    """
    curve_type: DecayCurveType = DecayCurveType.EXPONENTIAL
    start_weight: float = 0.67
    floor_weight: float = 0.20
    transition_weeks: int = 10
    inflection_point: float = 5.0
    steepness: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.start_weight <= 1.0:
            raise ValueError(f"start_weight must be (0, 1], got {self.start_weight}")
        if not 0 <= self.floor_weight < self.start_weight:
            raise ValueError(f"floor_weight must be [0, start_weight), got {self.floor_weight}")
        if self.transition_weeks < 1:
            raise ValueError(f"transition_weeks must be >= 1, got {self.transition_weeks}")


class PreseasonDecayCurve:
    """
    Calculates blending weights for preseason vs in-season data.
    
    As the season progresses, we want to trust in-season EPA more and
    preseason projections less. This class implements various decay
    functions to smoothly transition between the two.
    
    Usage:
        curve = PreseasonDecayCurve()
        preseason_weight = curve.get_preseason_weight(week=5)
        inseason_weight = curve.get_inseason_weight(week=5)
    """
    
    def __init__(self, config: Optional[DecayCurveConfig] = None):
        """Initialize with optional custom configuration."""
        self.config = config or DecayCurveConfig()
        self._cache: Dict[int, float] = {}
    
    def get_preseason_weight(self, week: int) -> float:
        """
        Get the preseason weight for a given week.
        
        Args:
            week: Current NFL week (0 = preseason, 1-18 = regular season)
        
        Returns:
            Weight for preseason data (0.0 to 1.0)
        """
        if week < 0:
            raise ValueError(f"Week must be >= 0, got {week}")
        
        # Check cache
        if week in self._cache:
            return self._cache[week]
        
        # Calculate based on curve type
        if self.config.curve_type == DecayCurveType.LINEAR:
            weight = self._linear_decay(week)
        elif self.config.curve_type == DecayCurveType.EXPONENTIAL:
            weight = self._exponential_decay(week)
        elif self.config.curve_type == DecayCurveType.LOGARITHMIC:
            weight = self._logarithmic_decay(week)
        elif self.config.curve_type == DecayCurveType.SIGMOID:
            weight = self._sigmoid_decay(week)
        else:
            weight = self._exponential_decay(week)  # Default
        
        # Apply floor
        weight = max(weight, self.config.floor_weight)
        weight = round(weight, 4)
        
        # Cache result
        self._cache[week] = weight
        return weight
    
    def get_inseason_weight(self, week: int) -> float:
        """
        Get the in-season weight for a given week.
        
        This is simply 1 - preseason_weight.
        """
        return round(1.0 - self.get_preseason_weight(week), 4)
    
    def get_weights(self, week: int) -> Tuple[float, float]:
        """
        Get both weights as a tuple.
        
        Returns:
            (preseason_weight, inseason_weight)
        """
        pre = self.get_preseason_weight(week)
        return (pre, round(1.0 - pre, 4))
    
    def _linear_decay(self, week: int) -> float:
        """Linear decay from start_weight to floor_weight."""
        if week >= self.config.transition_weeks:
            return self.config.floor_weight
        
        decay_rate = (self.config.start_weight - self.config.floor_weight) / self.config.transition_weeks
        return self.config.start_weight - (decay_rate * week)
    
    def _exponential_decay(self, week: int) -> float:
        """
        Exponential decay with half-life based on transition_weeks.
        
        Weight = floor + (start - floor) * e^(-k * week)
        where k is chosen so weight reaches ~floor at transition_weeks
        """
        if week == 0:
            return self.config.start_weight
        
        # Calculate decay constant (reaches ~5% of original at transition_weeks)
        k = 3.0 / self.config.transition_weeks
        
        decay_range = self.config.start_weight - self.config.floor_weight
        return self.config.floor_weight + decay_range * math.exp(-k * week)
    
    def _logarithmic_decay(self, week: int) -> float:
        """Logarithmic decay - fast initial drop, then slow decline."""
        if week == 0:
            return self.config.start_weight
        
        # log(1) = 0, so we start at week 1 effectively
        log_factor = math.log(week + 1) / math.log(self.config.transition_weeks + 1)
        decay_range = self.config.start_weight - self.config.floor_weight
        return self.config.start_weight - (decay_range * log_factor)
    
    def _sigmoid_decay(self, week: int) -> float:
        """
        Sigmoid decay - smooth S-curve transition.
        
        Useful when you want gradual change early and late, with
        faster transition in the middle.
        """
        # Sigmoid: 1 / (1 + e^(k*(x-midpoint)))
        # Scaled and shifted to go from start_weight to floor_weight
        x = week - self.config.inflection_point
        sigmoid = 1 / (1 + math.exp(self.config.steepness * x))
        
        decay_range = self.config.start_weight - self.config.floor_weight
        return self.config.floor_weight + (decay_range * sigmoid)
    
    def get_all_weights(self, max_week: int = 18) -> List[Dict]:
        """
        Get weights for all weeks (useful for visualization).
        
        Returns:
            List of dicts with week, preseason_weight, inseason_weight
        """
        weights = []
        for week in range(0, max_week + 1):
            pre, ins = self.get_weights(week)
            weights.append({
                "week": week,
                "preseason_weight": pre,
                "inseason_weight": ins
            })
        return weights
    
    def clear_cache(self):
        """Clear the weight cache."""
        self._cache = {}


# =============================================================================
# Story 1.4: Blended Team Rating Calculator
# =============================================================================

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


class TeamRatingCalculator:
    """
    Calculates blended team ratings from preseason and in-season data.
    
    This is the core of the rating system - it takes preseason projections
    and in-season EPA data and produces a single power rating for each team.
    
    Usage:
        calculator = TeamRatingCalculator()
        rating = calculator.calculate_rating(preseason_data, inseason_data, week=5)
    """
    
    def __init__(
        self,
        decay_curve: Optional[PreseasonDecayCurve] = None,
        min_plays_for_reliability: int = 100
    ):
        """
        Initialize the calculator.
        
        Args:
            decay_curve: Custom decay curve (uses default if None)
            min_plays_for_reliability: Minimum plays for reliable EPA
        """
        self.decay_curve = decay_curve or PreseasonDecayCurve()
        self.min_plays = min_plays_for_reliability
    
    def calculate_rating(
        self,
        preseason: TeamPreseasonData,
        inseason: Optional[TeamInSeasonData],
        week: int
    ) -> BlendedTeamRating:
        """
        Calculate blended rating for a team.
        
        Args:
            preseason: Preseason projection data
            inseason: In-season EPA data (None if no games played)
            week: Current week number
        
        Returns:
            BlendedTeamRating with final power rating
        """
        # Get blending weights
        preseason_weight, inseason_weight = self.decay_curve.get_weights(week)
        
        # Extract ratings
        preseason_rating = preseason.preseason_power_rating
        
        inseason_rating = None
        games_played = 0
        sufficient_data = True
        
        if inseason:
            inseason_rating = inseason.epa_power_rating
            games_played = inseason.games_played
            sufficient_data = inseason.has_sufficient_data(self.min_plays)
            
            # Adjust weights if insufficient data
            if not sufficient_data:
                # Increase preseason weight when EPA is unreliable
                adjustment = 0.15
                preseason_weight = min(1.0, preseason_weight + adjustment)
                inseason_weight = 1.0 - preseason_weight
        
        return BlendedTeamRating(
            team=preseason.team,
            season=preseason.season,
            week=week,
            preseason_rating=preseason_rating,
            in_season_rating=inseason_rating,
            preseason_weight=preseason_weight,
            in_season_weight=inseason_weight,
            games_played=games_played,
            sufficient_data=sufficient_data
        )
    
    def calculate_rating_delta(
        self,
        current: BlendedTeamRating,
        previous: Optional[BlendedTeamRating]
    ) -> float:
        """Calculate rating change from previous week."""
        if previous is None:
            return 0.0
        return round(current.power_rating - previous.power_rating, 2)


# =============================================================================
# Story 1.5: NFL Season Model (Full 32-Team Integration)
# =============================================================================

# NFL Team Constants
NFL_TEAMS = {
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

# Flatten to list
ALL_NFL_TEAMS = []
for conf, divisions in NFL_TEAMS.items():
    for div, teams in divisions.items():
        ALL_NFL_TEAMS.extend(teams)


def get_team_division(team: str) -> Tuple[str, str]:
    """Get conference and division for a team."""
    for conf, divisions in NFL_TEAMS.items():
        for div, teams in divisions.items():
            if team in teams:
                return conf, div
    raise ValueError(f"Unknown team: {team}")


def are_division_rivals(team1: str, team2: str) -> bool:
    """Check if two teams are in the same division."""
    try:
        conf1, div1 = get_team_division(team1)
        conf2, div2 = get_team_division(team2)
        return conf1 == conf2 and div1 == div2
    except ValueError:
        return False


class NFLSeasonModel:
    """
    Full season model managing all 32 NFL teams.
    
    This class maintains preseason projections and in-season data for
    all teams, calculating blended ratings on demand.
    
    Usage:
        model = NFLSeasonModel(season=2025)
        model.load_preseason_data(preseason_dict)
        model.update_inseason_data(week=5, inseason_dict)
        ratings = model.get_power_rankings(week=5)
    """
    
    def __init__(
        self,
        season: int,
        decay_curve: Optional[PreseasonDecayCurve] = None
    ):
        """
        Initialize the season model.
        
        Args:
            season: NFL season year
            decay_curve: Custom decay curve configuration
        """
        self.season = season
        self.decay_curve = decay_curve or PreseasonDecayCurve()
        self.calculator = TeamRatingCalculator(self.decay_curve)
        
        # Data storage
        self._preseason: Dict[str, TeamPreseasonData] = {}
        self._inseason: Dict[str, Dict[int, TeamInSeasonData]] = {}  # team -> week -> data
        self._ratings_cache: Dict[str, Dict[int, BlendedTeamRating]] = {}  # team -> week -> rating
        
        # Current week tracking
        self.current_week = 0
    
    def load_preseason_data(self, data: Dict[str, TeamPreseasonData]) -> None:
        """
        Load preseason projections for all teams.
        
        Args:
            data: Dict mapping team name to TeamPreseasonData
        """
        self._preseason = data.copy()
        self._validate_all_teams_present()
    
    def load_preseason_from_wins(self, projected_wins: Dict[str, float]) -> None:
        """
        Convenience method to load preseason data from just win totals.
        
        Args:
            projected_wins: Dict mapping team name to projected wins
        """
        for team, wins in projected_wins.items():
            conf, div = get_team_division(team)
            self._preseason[team] = TeamPreseasonData(
                team=team,
                season=self.season,
                projected_wins=wins,
                division=div,
                conference=conf,
                source="Projected Wins Input"
            )
        self._validate_all_teams_present()
    
    def update_inseason_data(
        self,
        week: int,
        data: Dict[str, TeamInSeasonData]
    ) -> None:
        """
        Update in-season EPA data for a specific week.
        
        Args:
            week: Week number
            data: Dict mapping team name to TeamInSeasonData
        """
        for team, team_data in data.items():
            if team not in self._inseason:
                self._inseason[team] = {}
            self._inseason[team][week] = team_data
        
        self.current_week = max(self.current_week, week)
        
        # Clear affected cache entries
        for team in data.keys():
            if team in self._ratings_cache and week in self._ratings_cache[team]:
                del self._ratings_cache[team][week]
    
    def get_team_rating(self, team: str, week: int) -> BlendedTeamRating:
        """
        Get blended rating for a single team.
        
        Args:
            team: Team name
            week: Week number
        
        Returns:
            BlendedTeamRating for the team
        """
        # Check cache
        if team in self._ratings_cache and week in self._ratings_cache.get(team, {}):
            return self._ratings_cache[team][week]
        
        # Get preseason data
        if team not in self._preseason:
            raise ValueError(f"No preseason data for {team}")
        preseason = self._preseason[team]
        
        # Get most recent in-season data up to this week
        inseason = self._get_latest_inseason(team, week)
        
        # Calculate rating
        rating = self.calculator.calculate_rating(preseason, inseason, week)
        
        # Cache result
        if team not in self._ratings_cache:
            self._ratings_cache[team] = {}
        self._ratings_cache[team][week] = rating
        
        return rating
    
    def get_all_ratings(self, week: int) -> Dict[str, BlendedTeamRating]:
        """
        Get ratings for all 32 teams.
        
        Args:
            week: Week number
        
        Returns:
            Dict mapping team name to BlendedTeamRating
        """
        return {team: self.get_team_rating(team, week) for team in ALL_NFL_TEAMS}
    
    def get_power_rankings(self, week: int) -> List[Dict]:
        """
        Get power rankings sorted by rating.
        
        Args:
            week: Week number
        
        Returns:
            List of dicts with rank, team, rating, etc.
        """
        ratings = self.get_all_ratings(week)
        
        # Sort by power rating descending
        sorted_teams = sorted(
            ratings.items(),
            key=lambda x: x[1].power_rating,
            reverse=True
        )
        
        rankings = []
        for rank, (team, rating) in enumerate(sorted_teams, 1):
            # Get previous week's ranking if available
            prev_rank = None
            if week > 0:
                try:
                    prev_ratings = self.get_all_ratings(week - 1)
                    prev_sorted = sorted(
                        prev_ratings.items(),
                        key=lambda x: x[1].power_rating,
                        reverse=True
                    )
                    for pr, (pt, _) in enumerate(prev_sorted, 1):
                        if pt == team:
                            prev_rank = pr
                            break
                except:
                    pass
            
            rank_change = (prev_rank - rank) if prev_rank else None
            
            rankings.append({
                "rank": rank,
                "team": team,
                "power_rating": rating.power_rating,
                "preseason_rating": rating.preseason_rating,
                "inseason_rating": rating.in_season_rating,
                "games_played": rating.games_played,
                "confidence": rating.confidence,
                "rank_change": rank_change,
                "division": self._preseason[team].division if team in self._preseason else "",
                "conference": self._preseason[team].conference if team in self._preseason else ""
            })
        
        return rankings
    
    def get_matchup_rating_diff(
        self,
        home_team: str,
        away_team: str,
        week: int
    ) -> Dict:
        """
        Get rating differential for a matchup.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            week: Week number
        
        Returns:
            Dict with rating info and differential
        """
        home_rating = self.get_team_rating(home_team, week)
        away_rating = self.get_team_rating(away_team, week)
        
        diff = home_rating.power_rating - away_rating.power_rating
        is_divisional = are_division_rivals(home_team, away_team)
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_rating": home_rating.power_rating,
            "away_rating": away_rating.power_rating,
            "rating_differential": round(diff, 2),
            "is_divisional": is_divisional,
            "home_confidence": home_rating.confidence,
            "away_confidence": away_rating.confidence
        }
    
    def _get_latest_inseason(
        self,
        team: str,
        max_week: int
    ) -> Optional[TeamInSeasonData]:
        """Get most recent in-season data up to specified week."""
        if team not in self._inseason:
            return None
        
        team_data = self._inseason[team]
        
        # Find latest week <= max_week
        available_weeks = [w for w in team_data.keys() if w <= max_week]
        if not available_weeks:
            return None
        
        latest_week = max(available_weeks)
        return team_data[latest_week]
    
    def _validate_all_teams_present(self) -> None:
        """Validate that all 32 teams have preseason data."""
        missing = set(ALL_NFL_TEAMS) - set(self._preseason.keys())
        if missing:
            raise ValueError(f"Missing preseason data for teams: {missing}")
    
    def get_decay_weights(self, week: int) -> Tuple[float, float]:
        """Get the decay curve weights for a week."""
        return self.decay_curve.get_weights(week)
    
    def clear_cache(self) -> None:
        """Clear all cached ratings."""
        self._ratings_cache = {}


# =============================================================================
# Story 1.6: Rating Validation & Edge Cases
# =============================================================================

class RatingValidator:
    """
    Validates ratings and handles edge cases.
    
    Edge cases handled:
    - Teams with no games played (use preseason only)
    - Teams with insufficient plays for reliable EPA
    - Extreme EPA values (statistical outliers)
    - Bye weeks (use previous week's data)
    - Season start (week 0/1 handling)
    """
    
    # Valid rating ranges
    MIN_POWER_RATING = -12.0
    MAX_POWER_RATING = 12.0
    
    # EPA valid ranges
    MIN_EPA_PER_PLAY = -0.5
    MAX_EPA_PER_PLAY = 0.5
    
    # Minimum data thresholds
    MIN_PLAYS_RELIABLE = 100
    MIN_GAMES_RELIABLE = 3
    
    @classmethod
    def validate_preseason(cls, data: TeamPreseasonData) -> List[str]:
        """
        Validate preseason data.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if not 0 <= data.projected_wins <= 17:
            issues.append(f"Invalid projected wins: {data.projected_wins}")
        
        if not cls.MIN_POWER_RATING <= data.preseason_power_rating <= cls.MAX_POWER_RATING:
            issues.append(f"Power rating out of range: {data.preseason_power_rating}")
        
        if data.offensive_ranking and not 1 <= data.offensive_ranking <= 32:
            issues.append(f"Invalid offensive ranking: {data.offensive_ranking}")
        
        if data.defensive_ranking and not 1 <= data.defensive_ranking <= 32:
            issues.append(f"Invalid defensive ranking: {data.defensive_ranking}")
        
        return issues
    
    @classmethod
    def validate_inseason(cls, data: TeamInSeasonData) -> List[str]:
        """
        Validate in-season data.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if not cls.MIN_EPA_PER_PLAY <= data.offensive_epa_per_play <= cls.MAX_EPA_PER_PLAY:
            issues.append(f"Offensive EPA out of typical range: {data.offensive_epa_per_play}")
        
        if not cls.MIN_EPA_PER_PLAY <= data.defensive_epa_per_play <= cls.MAX_EPA_PER_PLAY:
            issues.append(f"Defensive EPA out of typical range: {data.defensive_epa_per_play}")
        
        if data.games_played < 0 or data.games_played > 18:
            issues.append(f"Invalid games played: {data.games_played}")
        
        if data.week < 1 or data.week > 18:
            issues.append(f"Invalid week: {data.week}")
        
        if data.games_played < cls.MIN_GAMES_RELIABLE:
            issues.append(f"Low sample size: {data.games_played} games (recommend >= {cls.MIN_GAMES_RELIABLE})")
        
        if not data.has_sufficient_data(cls.MIN_PLAYS_RELIABLE):
            issues.append(f"Insufficient plays for reliable EPA")
        
        return issues
    
    @classmethod
    def validate_blended_rating(cls, rating: BlendedTeamRating) -> List[str]:
        """
        Validate blended rating.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if not cls.MIN_POWER_RATING <= rating.power_rating <= cls.MAX_POWER_RATING:
            issues.append(f"Final rating out of range: {rating.power_rating}")
        
        # Check weight validity
        if not (0 <= rating.preseason_weight <= 1):
            issues.append(f"Invalid preseason weight: {rating.preseason_weight}")
        
        if not (0 <= rating.in_season_weight <= 1):
            issues.append(f"Invalid in-season weight: {rating.in_season_weight}")
        
        weight_sum = rating.preseason_weight + rating.in_season_weight
        if not 0.99 <= weight_sum <= 1.01:
            issues.append(f"Weights don't sum to 1: {weight_sum}")
        
        return issues
    
    @classmethod
    def clamp_epa(cls, epa: float) -> float:
        """Clamp EPA to valid range."""
        return max(cls.MIN_EPA_PER_PLAY, min(cls.MAX_EPA_PER_PLAY, epa))
    
    @classmethod
    def clamp_rating(cls, rating: float) -> float:
        """Clamp power rating to valid range."""
        return max(cls.MIN_POWER_RATING, min(cls.MAX_POWER_RATING, rating))


# =============================================================================
# =============================================================================
# EPIC 2: GAME PREDICTION SYSTEM
# =============================================================================
# =============================================================================


# =============================================================================
# Story 2.1: Game/Matchup Data Model
# =============================================================================

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


# =============================================================================
# Story 2.2: Home Field Advantage Calculator
# =============================================================================

@dataclass
class HomeFieldAdvantageConfig:
    """
    Configuration for home field advantage calculation.
    
    Research shows HFA has declined in recent years from ~3.0 to ~1.5-2.0 points.
    Various factors can modify the base advantage.
    
    Attributes:
        base_advantage: Base home field advantage in points (default 2.0)
        divisional_boost: Extra points for divisional home games
        altitude_bonus: Extra points for high-altitude home (Denver)
        dome_advantage: Points for dome team hosting outdoor team in bad weather
        crowd_noise_factor: Multiplier for loud stadiums (1.0 = normal)
        primetime_boost: Extra points for primetime home games
        playoff_multiplier: HFA multiplier for playoff games
    """
    base_advantage: float = 2.0
    divisional_boost: float = 0.3
    altitude_bonus: float = 0.5  # Denver effect
    dome_advantage: float = 0.3
    crowd_noise_factor: float = 1.0
    primetime_boost: float = 0.2
    playoff_multiplier: float = 1.2
    
    # Stadium-specific adjustments (team -> adjustment)
    stadium_adjustments: Dict[str, float] = field(default_factory=dict)


# Stadium coordinates for travel distance calculation
STADIUM_COORDINATES = {
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
STADIUM_ALTITUDES = {
    "Denver Broncos": 5280,
    "Arizona Cardinals": 1086,
    "Las Vegas Raiders": 2001,
    "Kansas City Chiefs": 820,
    # Most others are near sea level
}

# Dome/indoor stadiums
DOME_STADIUMS = {
    "Arizona Cardinals", "Atlanta Falcons", "Dallas Cowboys",
    "Detroit Lions", "Houston Texans", "Indianapolis Colts",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams",
    "Minnesota Vikings", "New Orleans Saints",
}


class HomeFieldAdvantageCalculator:
    """
    Calculates home field advantage for a specific matchup.
    
    Considers:
    - Base HFA (~2.0 points)
    - Divisional games (slight boost)
    - Altitude (Denver advantage)
    - Dome vs outdoor
    - Primetime games
    - Playoff games
    
    Usage:
        calc = HomeFieldAdvantageCalculator()
        hfa = calc.calculate(game)
    """
    
    def __init__(self, config: Optional[HomeFieldAdvantageConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or HomeFieldAdvantageConfig()
    
    def calculate(self, game: NFLGame) -> float:
        """
        Calculate home field advantage for a game.
        
        Args:
            game: NFLGame object
        
        Returns:
            Home field advantage in points (positive = home team advantage)
        """
        # Neutral site = no HFA
        if game.location == GameLocation.NEUTRAL:
            return 0.0
        
        # Start with base advantage
        hfa = self.config.base_advantage
        
        # Divisional boost
        if game.is_divisional:
            hfa += self.config.divisional_boost
        
        # Altitude bonus (Denver)
        home_altitude = STADIUM_ALTITUDES.get(game.home_team, 0)
        away_altitude = STADIUM_ALTITUDES.get(game.away_team, 0)
        if home_altitude > 4000 and away_altitude < 2000:
            hfa += self.config.altitude_bonus
        
        # Dome advantage (dome team at home vs outdoor team)
        home_is_dome = game.home_team in DOME_STADIUMS
        away_is_dome = game.away_team in DOME_STADIUMS
        if home_is_dome and not away_is_dome:
            hfa += self.config.dome_advantage
        
        # Primetime boost
        if game.home_context and game.home_context.is_primetime:
            hfa += self.config.primetime_boost
        
        # Playoff multiplier
        if game.game_type != GameType.REGULAR_SEASON:
            hfa *= self.config.playoff_multiplier
        
        # Stadium-specific adjustments
        if game.home_team in self.config.stadium_adjustments:
            hfa += self.config.stadium_adjustments[game.home_team]
        
        return round(hfa, 2)
    
    def get_travel_distance(self, away_team: str, home_team: str) -> float:
        """
        Calculate travel distance in miles between stadiums.
        
        Uses Haversine formula for great-circle distance.
        """
        if away_team not in STADIUM_COORDINATES or home_team not in STADIUM_COORDINATES:
            return 0.0
        
        lat1, lon1 = STADIUM_COORDINATES[away_team]
        lat2, lon2 = STADIUM_COORDINATES[home_team]
        
        # Haversine formula
        R = 3959  # Earth's radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return round(R * c, 1)
    
    def get_timezone_change(self, away_team: str, home_team: str) -> int:
        """
        Calculate timezone change for away team.
        
        Returns hours of timezone shift (-3 to +3).
        Negative = traveling west, Positive = traveling east.
        """
        # Approximate timezone by longitude
        away_coords = STADIUM_COORDINATES.get(away_team)
        home_coords = STADIUM_COORDINATES.get(home_team)
        
        if not away_coords or not home_coords:
            return 0
        
        # Rough conversion: 15 degrees longitude = 1 timezone
        lon_diff = home_coords[1] - away_coords[1]
        tz_change = round(lon_diff / 15)
        
        return max(-3, min(3, tz_change))


# =============================================================================
# Story 2.3: Rest & Travel Adjustments
# =============================================================================

@dataclass
class RestTravelConfig:
    """
    Configuration for rest and travel adjustments.
    
    Based on research showing:
    - Teams coming off bye week have ~1.5 point advantage
    - Short week (Thursday games) has ~1.0 point disadvantage
    - Cross-country travel has ~0.5 point disadvantage
    - East-to-West travel is harder than West-to-East
    
    Attributes:
        bye_week_advantage: Points advantage for team coming off bye
        short_week_penalty: Points penalty for Thursday game
        extra_rest_per_day: Points per extra day of rest (above 7)
        travel_penalty_per_1000_miles: Points penalty per 1000 miles
        timezone_penalty_per_hour: Points penalty per hour timezone change
        consecutive_road_penalty: Points penalty per consecutive road game
        mnf_short_turnaround: Extra penalty for MNF followed by early game
    """
    bye_week_advantage: float = 1.5
    short_week_penalty: float = 1.0
    extra_rest_per_day: float = 0.15
    travel_penalty_per_1000_miles: float = 0.3
    timezone_penalty_per_hour: float = 0.15
    consecutive_road_penalty: float = 0.2
    mnf_short_turnaround: float = 0.3
    max_adjustment: float = 3.0  # Cap total adjustment


class RestTravelAdjuster:
    """
    Calculates rest and travel adjustments for game predictions.
    
    This adjusts the baseline prediction based on situational factors
    like bye weeks, short weeks, and travel distance.
    
    Usage:
        adjuster = RestTravelAdjuster()
        adjustment = adjuster.calculate_adjustment(home_context, away_context)
    """
    
    def __init__(self, config: Optional[RestTravelConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or RestTravelConfig()
        self.hfa_calc = HomeFieldAdvantageCalculator()
    
    def calculate_adjustment(
        self,
        home_context: TeamGameContext,
        away_context: TeamGameContext
    ) -> float:
        """
        Calculate net rest/travel adjustment.
        
        Returns adjustment from home team perspective.
        Positive = favors home team, Negative = favors away team.
        """
        home_adj = self._calculate_team_adjustment(home_context, is_home=True)
        away_adj = self._calculate_team_adjustment(away_context, is_home=False)
        
        # Net adjustment (positive = home advantage)
        net_adjustment = home_adj - away_adj
        
        # Apply cap
        net_adjustment = max(-self.config.max_adjustment,
                            min(self.config.max_adjustment, net_adjustment))
        
        return round(net_adjustment, 2)
    
    def _calculate_team_adjustment(
        self,
        context: TeamGameContext,
        is_home: bool
    ) -> float:
        """Calculate adjustment for a single team."""
        adjustment = 0.0
        
        # Bye week advantage
        if context.is_coming_off_bye:
            adjustment += self.config.bye_week_advantage
        
        # Short week penalty
        if context.is_short_week:
            adjustment -= self.config.short_week_penalty
        
        # Extra rest (above normal 7 days, capped at 14)
        if context.days_rest > 7:
            extra_days = min(context.days_rest - 7, 7)
            adjustment += extra_days * self.config.extra_rest_per_day
        
        # Less rest penalty (below 7 days)
        if context.days_rest < 7:
            missing_days = 7 - context.days_rest
            adjustment -= missing_days * self.config.extra_rest_per_day * 1.5
        
        # Monday night short turnaround
        if context.previous_game_was_mnf and context.days_rest < 7:
            adjustment -= self.config.mnf_short_turnaround
        
        # Travel penalty (away team only)
        if not is_home:
            # Travel distance
            if context.travel_miles > 0:
                travel_penalty = (context.travel_miles / 1000) * self.config.travel_penalty_per_1000_miles
                adjustment -= travel_penalty
            
            # Timezone change
            if context.timezone_change != 0:
                # West-to-East is harder (positive timezone change)
                tz_penalty = abs(context.timezone_change) * self.config.timezone_penalty_per_hour
                if context.timezone_change > 0:  # Traveling east
                    tz_penalty *= 1.2
                adjustment -= tz_penalty
            
            # Consecutive road games
            if context.consecutive_road_games > 1:
                adjustment -= (context.consecutive_road_games - 1) * self.config.consecutive_road_penalty
        
        return adjustment
    
    def build_game_context(
        self,
        team: str,
        is_home: bool,
        days_rest: int = 7,
        previous_opponent: Optional[str] = None,
        away_team: Optional[str] = None,
        home_team: Optional[str] = None,
        is_primetime: bool = False,
        is_short_week: bool = False,
        is_coming_off_bye: bool = False,
        previous_game_was_mnf: bool = False,
        consecutive_road_games: int = 0
    ) -> TeamGameContext:
        """
        Build a TeamGameContext with calculated fields.
        
        This is a convenience method to construct context with
        travel distance and timezone automatically calculated.
        """
        travel_miles = 0.0
        timezone_change = 0
        
        if not is_home and home_team:
            travel_miles = self.hfa_calc.get_travel_distance(team, home_team)
            timezone_change = self.hfa_calc.get_timezone_change(team, home_team)
        
        return TeamGameContext(
            team=team,
            days_rest=days_rest,
            is_coming_off_bye=is_coming_off_bye,
            previous_game_was_mnf=previous_game_was_mnf,
            travel_miles=travel_miles,
            timezone_change=timezone_change,
            is_home=is_home,
            is_primetime=is_primetime,
            is_short_week=is_short_week,
            consecutive_road_games=consecutive_road_games
        )


# =============================================================================
# Story 2.4: Win Probability Calculator
# =============================================================================

class WinProbabilityModel(ABC):
    """Abstract base class for win probability models."""
    
    @abstractmethod
    def calculate(self, rating_diff: float) -> float:
        """
        Calculate win probability from rating differential.
        
        Args:
            rating_diff: Home team rating - Away team rating
        
        Returns:
            Home team win probability (0.0 to 1.0)
        """
        pass


class LogisticWinProbability(WinProbabilityModel):
    """
    Logistic regression-based win probability model.
    
    Uses the formula: P(home_win) = 1 / (1 + 10^(-diff/scale))
    
    This is similar to Elo rating conversion.
    """
    
    def __init__(self, scale: float = 6.0):
        """
        Initialize with scale factor.
        
        Args:
            scale: Points per "rating unit" for probability conversion.
                   Higher scale = ratings have less impact on probability.
                   Default 6.0 means ~6 points = 1 "unit" difference.
        """
        self.scale = scale
    
    def calculate(self, rating_diff: float) -> float:
        """Calculate win probability using logistic function."""
        # Prevent overflow for extreme values
        if rating_diff > 30:
            return 0.99
        if rating_diff < -30:
            return 0.01
        
        prob = 1.0 / (1.0 + 10 ** (-rating_diff / self.scale))
        return round(prob, 4)


class LinearWinProbability(WinProbabilityModel):
    """
    Simple linear win probability model.
    
    Caps at 0.05 and 0.95 to avoid extreme probabilities.
    """
    
    def __init__(self, points_per_percent: float = 0.25):
        """
        Initialize with conversion rate.
        
        Args:
            points_per_percent: Rating points per 1% win probability.
                               Default 0.25 means 4 points = 16% swing.
        """
        self.points_per_percent = points_per_percent
    
    def calculate(self, rating_diff: float) -> float:
        """Calculate win probability using linear conversion."""
        base_prob = 0.50
        adjustment = rating_diff / (100 * self.points_per_percent)
        prob = base_prob + adjustment
        
        # Cap between 5% and 95%
        return round(max(0.05, min(0.95, prob)), 4)


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


# =============================================================================
# Story 2.5: Spread/Line Prediction
# =============================================================================

class SpreadPredictor:
    """
    Generates spread predictions for NFL games.
    
    This is the main interface for generating game predictions.
    It combines team ratings, home field advantage, and rest/travel
    adjustments to produce spread predictions.
    
    Usage:
        predictor = SpreadPredictor(season_model)
        prediction = predictor.predict_game(game, week=5)
    """
    
    def __init__(
        self,
        season_model: 'NFLSeasonModel',
        hfa_calculator: Optional[HomeFieldAdvantageCalculator] = None,
        rest_adjuster: Optional[RestTravelAdjuster] = None,
        win_prob_model: Optional[WinProbabilityModel] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            season_model: NFLSeasonModel with team ratings
            hfa_calculator: Custom HFA calculator (uses default if None)
            rest_adjuster: Custom rest/travel adjuster (uses default if None)
            win_prob_model: Custom win probability model (uses default if None)
        """
        self.season_model = season_model
        self.hfa_calc = hfa_calculator or HomeFieldAdvantageCalculator()
        self.rest_adjuster = rest_adjuster or RestTravelAdjuster()
        self.win_prob_model = win_prob_model or LogisticWinProbability()
    
    def predict_game(
        self,
        game: NFLGame,
        week: Optional[int] = None
    ) -> GamePrediction:
        """
        Generate prediction for a single game.
        
        Args:
            game: NFLGame to predict
            week: Week number for ratings (uses game.week if None)
        
        Returns:
            GamePrediction with spread and win probability
        """
        week = week or game.week
        
        # Get team ratings
        home_rating = self.season_model.get_team_rating(game.home_team, week)
        away_rating = self.season_model.get_team_rating(game.away_team, week)
        
        # Calculate home field advantage
        hfa = self.hfa_calc.calculate(game)
        
        # Calculate rest/travel adjustment
        rest_adj = 0.0
        if game.home_context and game.away_context:
            rest_adj = self.rest_adjuster.calculate_adjustment(
                game.home_context,
                game.away_context
            )
        
        # Determine confidence tier
        confidence_tier = self._determine_confidence(home_rating, away_rating)
        confidence_score = self._calculate_confidence_score(home_rating, away_rating)
        
        return GamePrediction(
            game=game,
            home_power_rating=home_rating.power_rating,
            away_power_rating=away_rating.power_rating,
            home_field_advantage=hfa,
            rest_travel_adjustment=rest_adj,
            confidence_tier=confidence_tier,
            confidence_score=confidence_score
        )
    
    def predict_games(
        self,
        games: List[NFLGame],
        week: Optional[int] = None
    ) -> List[GamePrediction]:
        """
        Generate predictions for multiple games.
        
        Args:
            games: List of NFLGame objects
            week: Week number for ratings
        
        Returns:
            List of GamePrediction objects
        """
        return [self.predict_game(game, week) for game in games]
    
    def _determine_confidence(
        self,
        home_rating: BlendedTeamRating,
        away_rating: BlendedTeamRating
    ) -> str:
        """Determine confidence tier based on data quality."""
        # Both high confidence = high
        if home_rating.confidence == "High" and away_rating.confidence == "High":
            return "High"
        
        # Both medium or mixed high/medium = medium
        if (home_rating.confidence in ["High", "Medium"] and
            away_rating.confidence in ["High", "Medium"]):
            return "Medium"
        
        return "Low"
    
    def _calculate_confidence_score(
        self,
        home_rating: BlendedTeamRating,
        away_rating: BlendedTeamRating
    ) -> int:
        """Calculate numerical confidence score (0-100)."""
        score = 50  # Base
        
        # Add points for data quality
        confidence_points = {"High": 20, "Medium": 10, "Low": 0, "Very Low": -10}
        score += confidence_points.get(home_rating.confidence, 0)
        score += confidence_points.get(away_rating.confidence, 0)
        
        # Add points for games played
        games = home_rating.games_played + away_rating.games_played
        score += min(games, 10)  # Max 10 points from games
        
        return max(0, min(100, score))


# =============================================================================
# Story 2.6: Weekly Schedule & Predictions Manager
# =============================================================================

class WeeklyScheduleManager:
    """
    Manages weekly NFL schedules and predictions.
    
    This class coordinates the generation of predictions for
    a full week of games and tracks results.
    
    Usage:
        manager = WeeklyScheduleManager(season_model, season=2025)
        manager.add_games(week5_games)
        predictions = manager.get_week_predictions(week=5)
    """
    
    def __init__(
        self,
        season_model: 'NFLSeasonModel',
        season: int,
        predictor: Optional[SpreadPredictor] = None
    ):
        """
        Initialize the manager.
        
        Args:
            season_model: NFLSeasonModel with team ratings
            season: NFL season year
            predictor: Custom predictor (creates default if None)
        """
        self.season_model = season_model
        self.season = season
        self.predictor = predictor or SpreadPredictor(season_model)
        
        # Storage
        self._games: Dict[int, List[NFLGame]] = {}  # week -> games
        self._predictions: Dict[int, List[GamePrediction]] = {}  # week -> predictions
        self._results: Dict[str, dict] = {}  # game_id -> result info
    
    def add_game(self, game: NFLGame) -> None:
        """Add a single game to the schedule."""
        week = game.week
        if week not in self._games:
            self._games[week] = []
        self._games[week].append(game)
        
        # Clear cached predictions for this week
        if week in self._predictions:
            del self._predictions[week]
    
    def add_games(self, games: List[NFLGame]) -> None:
        """Add multiple games to the schedule."""
        for game in games:
            self.add_game(game)
    
    def get_week_games(self, week: int) -> List[NFLGame]:
        """Get all games for a specific week."""
        return self._games.get(week, [])
    
    def get_week_predictions(
        self,
        week: int,
        force_refresh: bool = False
    ) -> List[GamePrediction]:
        """
        Get predictions for all games in a week.
        
        Args:
            week: Week number
            force_refresh: Regenerate predictions even if cached
        
        Returns:
            List of GamePrediction objects
        """
        if week not in self._predictions or force_refresh:
            games = self.get_week_games(week)
            self._predictions[week] = self.predictor.predict_games(games, week)
        
        return self._predictions[week]
    
    def update_game_result(
        self,
        game_id: str,
        home_score: int,
        away_score: int
    ) -> None:
        """
        Update a game with its final result.
        
        Args:
            game_id: Game identifier
            home_score: Final home team score
            away_score: Final away team score
        """
        # Find and update the game
        for week_games in self._games.values():
            for game in week_games:
                if game.game_id == game_id:
                    game.home_score = home_score
                    game.away_score = away_score
                    game.is_completed = True
                    
                    # Store result info
                    self._results[game_id] = {
                        "home_score": home_score,
                        "away_score": away_score,
                        "winner": game.winner,
                        "actual_spread": game.actual_spread
                    }
                    return
    
    def get_week_summary(self, week: int) -> Dict:
        """
        Get summary statistics for a week's predictions.
        
        Returns dict with:
        - total_games
        - games_completed
        - correct_picks
        - correct_ats
        - accuracy percentages
        """
        predictions = self.get_week_predictions(week)
        
        total = len(predictions)
        completed = sum(1 for p in predictions if p.game.is_completed)
        
        if completed == 0:
            return {
                "week": week,
                "total_games": total,
                "games_completed": 0,
                "correct_picks": 0,
                "correct_ats": 0,
                "pick_accuracy": None,
                "ats_accuracy": None
            }
        
        correct_picks = sum(1 for p in predictions if p.is_correct() is True)
        correct_ats = sum(1 for p in predictions if p.is_correct_ats() is True)
        ats_eligible = sum(1 for p in predictions if p.is_correct_ats() is not None)
        
        return {
            "week": week,
            "total_games": total,
            "games_completed": completed,
            "correct_picks": correct_picks,
            "correct_ats": correct_ats,
            "pick_accuracy": round(correct_picks / completed * 100, 1),
            "ats_accuracy": round(correct_ats / ats_eligible * 100, 1) if ats_eligible > 0 else None
        }
    
    def get_season_summary(self) -> Dict:
        """Get summary statistics for the entire season."""
        all_weeks = sorted(self._games.keys())
        
        if not all_weeks:
            return {"error": "No games loaded"}
        
        total_games = 0
        completed_games = 0
        correct_picks = 0
        correct_ats = 0
        ats_eligible = 0
        
        week_summaries = []
        
        for week in all_weeks:
            summary = self.get_week_summary(week)
            week_summaries.append(summary)
            
            total_games += summary["total_games"]
            completed_games += summary["games_completed"]
            correct_picks += summary["correct_picks"]
            correct_ats += summary["correct_ats"]
        
        return {
            "season": self.season,
            "weeks_loaded": len(all_weeks),
            "total_games": total_games,
            "completed_games": completed_games,
            "correct_picks": correct_picks,
            "correct_ats": correct_ats,
            "pick_accuracy": round(correct_picks / completed_games * 100, 1) if completed_games > 0 else None,
            "ats_accuracy": round(correct_ats / ats_eligible * 100, 1) if ats_eligible > 0 else None,
            "week_summaries": week_summaries
        }
    
    def get_predictions_by_confidence(
        self,
        week: int,
        min_confidence: int = 0
    ) -> Dict[str, List[GamePrediction]]:
        """
        Get predictions grouped by confidence tier.
        
        Args:
            week: Week number
            min_confidence: Minimum confidence score to include
        
        Returns:
            Dict with keys "High", "Medium", "Low" containing predictions
        """
        predictions = self.get_week_predictions(week)
        predictions = [p for p in predictions if p.confidence_score >= min_confidence]
        
        grouped = {"High": [], "Medium": [], "Low": []}
        for pred in predictions:
            grouped[pred.confidence_tier].append(pred)
        
        return grouped
    
    def get_best_bets(
        self,
        week: int,
        min_edge: float = 2.0,
        min_confidence: int = 50
    ) -> List[GamePrediction]:
        """
        Get predictions with the best betting edges.
        
        Args:
            week: Week number
            min_edge: Minimum edge vs Vegas spread
            min_confidence: Minimum confidence score
        
        Returns:
            List of predictions with good edges, sorted by edge magnitude
        """
        predictions = self.get_week_predictions(week)
        
        best = []
        for pred in predictions:
            if (pred.edge is not None and
                abs(pred.edge) >= min_edge and
                pred.confidence_score >= min_confidence):
                best.append(pred)
        
        # Sort by edge magnitude
        best.sort(key=lambda p: abs(p.edge or 0), reverse=True)
        return best
    
    def print_week_predictions(self, week: int) -> None:
        """Print formatted predictions for a week."""
        predictions = self.get_week_predictions(week)
        
        print(f"\n{'='*70}")
        print(f"WEEK {week} PREDICTIONS")
        print(f"{'='*70}")
        
        # Sort by game date/time
        predictions.sort(key=lambda p: p.game.game_date)
        
        for pred in predictions:
            game = pred.game
            print(f"\n{game.away_team} @ {game.home_team}")
            print(f"  Spread: {pred.formatted_spread}")
            print(f"  Win Prob: {game.home_team} {pred.home_win_probability:.1%} | "
                  f"{game.away_team} {pred.away_win_probability:.1%}")
            print(f"  Confidence: {pred.confidence_tier} ({pred.confidence_score})")
            
            if game.vegas_spread is not None:
                print(f"  Vegas: {game.home_team} {game.vegas_spread:+.1f}")
                if pred.edge:
                    edge_dir = game.home_team if pred.edge > 0 else game.away_team
                    print(f"  Edge: {abs(pred.edge):.1f} pts on {edge_dir}")


# =============================================================================
# =============================================================================
# EPIC 3: BACKTESTING & MODEL VALIDATION
# =============================================================================
# =============================================================================


# =============================================================================
# Story 3.1: Prediction Result Tracker
# =============================================================================

@dataclass
class PredictionResult:
    """
    Tracks the result of a single prediction after the game is played.
    
    This stores both the prediction and actual outcome for analysis.
    
    Attributes:
        prediction: The original GamePrediction
        
        # Actual results
        actual_winner: Team that won
        actual_spread: Actual point spread (away - home)
        actual_total: Total points scored
        
        # Straight-up (SU) results
        predicted_winner: Team model predicted to win
        su_correct: Whether straight-up pick was correct
        
        # Against-the-spread (ATS) results
        vegas_spread: Vegas spread at game time
        ats_pick: Team model picked ATS (or None)
        ats_correct: Whether ATS pick was correct (or None if no pick)
        spread_error: Predicted spread - actual spread
        
        # Win probability calibration
        predicted_win_prob: Predicted probability for winner
        
        # Metadata
        game_id: Unique game identifier
        season: NFL season
        week: Week number
    """
    prediction: GamePrediction
    
    # Actual results
    actual_winner: str
    actual_spread: float
    actual_total: int
    
    # SU results
    predicted_winner: str = ""
    su_correct: bool = False
    
    # ATS results
    vegas_spread: Optional[float] = None
    ats_pick: Optional[str] = None
    ats_correct: Optional[bool] = None
    spread_error: float = 0.0
    
    # Calibration
    predicted_win_prob: float = 0.5
    
    # Metadata
    game_id: str = ""
    season: int = 0
    week: int = 0
    
    def __post_init__(self):
        """Calculate derived fields."""
        game = self.prediction.game
        
        # Set metadata
        self.game_id = game.game_id
        self.season = game.season
        self.week = game.week
        
        # Determine predicted winner
        self.predicted_winner = self.prediction.pick
        
        # Check straight-up correctness
        self.su_correct = (self.predicted_winner == self.actual_winner)
        
        # Store win probability for the actual winner
        if self.actual_winner == game.home_team:
            self.predicted_win_prob = self.prediction.home_win_probability
        else:
            self.predicted_win_prob = self.prediction.away_win_probability
        
        # Calculate spread error
        self.spread_error = round(
            self.prediction.predicted_spread - self.actual_spread, 1
        )
        
        # ATS results
        self.vegas_spread = game.vegas_spread
        self.ats_pick = self.prediction.pick_ats
        
        if self.vegas_spread is not None and self.ats_pick is not None:
            # Did home team cover?
            home_covered = self.actual_spread < self.vegas_spread
            
            if self.ats_pick == game.home_team:
                self.ats_correct = home_covered
            else:
                self.ats_correct = not home_covered
    
    @property
    def home_team(self) -> str:
        return self.prediction.game.home_team
    
    @property
    def away_team(self) -> str:
        return self.prediction.game.away_team
    
    @property
    def is_upset(self) -> bool:
        """Was the actual result an upset vs prediction?"""
        return not self.su_correct
    
    @property
    def confidence_tier(self) -> str:
        return self.prediction.confidence_tier
    
    @property
    def abs_spread_error(self) -> float:
        return abs(self.spread_error)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for analysis."""
        return {
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_winner": self.predicted_winner,
            "actual_winner": self.actual_winner,
            "su_correct": self.su_correct,
            "predicted_spread": self.prediction.predicted_spread,
            "actual_spread": self.actual_spread,
            "spread_error": self.spread_error,
            "vegas_spread": self.vegas_spread,
            "ats_pick": self.ats_pick,
            "ats_correct": self.ats_correct,
            "predicted_win_prob": self.predicted_win_prob,
            "confidence_tier": self.confidence_tier
        }


class PredictionResultTracker:
    """
    Tracks and stores prediction results across a season.
    
    Usage:
        tracker = PredictionResultTracker()
        tracker.add_result(prediction, home_score=24, away_score=17)
        metrics = tracker.get_accuracy_metrics()
    """
    
    def __init__(self):
        """Initialize the tracker."""
        self._results: List[PredictionResult] = []
        self._by_week: Dict[int, List[PredictionResult]] = {}
        self._by_team: Dict[str, List[PredictionResult]] = {}
        self._by_confidence: Dict[str, List[PredictionResult]] = {}
    
    def add_result(
        self,
        prediction: GamePrediction,
        home_score: int,
        away_score: int
    ) -> PredictionResult:
        """
        Add a prediction result.
        
        Args:
            prediction: The GamePrediction object
            home_score: Actual home team score
            away_score: Actual away team score
        
        Returns:
            PredictionResult object
        """
        game = prediction.game
        
        # Determine winner
        if home_score > away_score:
            actual_winner = game.home_team
        elif away_score > home_score:
            actual_winner = game.away_team
        else:
            actual_winner = "TIE"
        
        # Create result
        result = PredictionResult(
            prediction=prediction,
            actual_winner=actual_winner,
            actual_spread=away_score - home_score,
            actual_total=home_score + away_score
        )
        
        # Store in various indices
        self._results.append(result)
        
        # By week
        if result.week not in self._by_week:
            self._by_week[result.week] = []
        self._by_week[result.week].append(result)
        
        # By team
        for team in [game.home_team, game.away_team]:
            if team not in self._by_team:
                self._by_team[team] = []
            self._by_team[team].append(result)
        
        # By confidence
        conf = result.confidence_tier
        if conf not in self._by_confidence:
            self._by_confidence[conf] = []
        self._by_confidence[conf].append(result)
        
        return result
    
    def get_all_results(self) -> List[PredictionResult]:
        """Get all tracked results."""
        return self._results.copy()
    
    def get_results_by_week(self, week: int) -> List[PredictionResult]:
        """Get results for a specific week."""
        return self._by_week.get(week, [])
    
    def get_results_by_team(self, team: str) -> List[PredictionResult]:
        """Get results involving a specific team."""
        return self._by_team.get(team, [])
    
    def get_results_by_confidence(self, tier: str) -> List[PredictionResult]:
        """Get results for a specific confidence tier."""
        return self._by_confidence.get(tier, [])
    
    @property
    def total_predictions(self) -> int:
        return len(self._results)
    
    def clear(self) -> None:
        """Clear all tracked results."""
        self._results = []
        self._by_week = {}
        self._by_team = {}
        self._by_confidence = {}


# =============================================================================
# Story 3.2: Accuracy Metrics Calculator
# =============================================================================

@dataclass
class AccuracyMetrics:
    """
    Comprehensive accuracy metrics for model evaluation.
    
    Attributes:
        # Sample info
        total_games: Number of games evaluated
        
        # Straight-up (SU) accuracy
        su_correct: Number of correct SU picks
        su_accuracy: Percentage of correct SU picks
        
        # Against-the-spread (ATS) accuracy
        ats_games: Games with ATS picks
        ats_correct: Number of correct ATS picks
        ats_accuracy: Percentage of correct ATS picks
        
        # Spread prediction error
        mean_absolute_error: Average absolute spread error
        mean_error: Average spread error (bias indicator)
        rmse: Root mean squared error
        
        # Calibration
        brier_score: Brier score for probability calibration
        log_loss: Log loss for probability predictions
        
        # By confidence tier
        high_conf_accuracy: Accuracy for high confidence picks
        med_conf_accuracy: Accuracy for medium confidence picks
        low_conf_accuracy: Accuracy for low confidence picks
        
        # ROI (if tracking bets)
        theoretical_roi: ROI if betting every game at -110
    """
    # Sample info
    total_games: int = 0
    
    # SU accuracy
    su_correct: int = 0
    su_accuracy: float = 0.0
    
    # ATS accuracy
    ats_games: int = 0
    ats_correct: int = 0
    ats_accuracy: float = 0.0
    
    # Spread error
    mean_absolute_error: float = 0.0
    mean_error: float = 0.0
    rmse: float = 0.0
    
    # Calibration
    brier_score: float = 0.0
    log_loss: float = 0.0
    
    # By confidence
    high_conf_accuracy: Optional[float] = None
    med_conf_accuracy: Optional[float] = None
    low_conf_accuracy: Optional[float] = None
    
    # ROI
    theoretical_roi: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_games": self.total_games,
            "su_correct": self.su_correct,
            "su_accuracy": self.su_accuracy,
            "ats_games": self.ats_games,
            "ats_correct": self.ats_correct,
            "ats_accuracy": self.ats_accuracy,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_error": self.mean_error,
            "rmse": self.rmse,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "high_conf_accuracy": self.high_conf_accuracy,
            "med_conf_accuracy": self.med_conf_accuracy,
            "low_conf_accuracy": self.low_conf_accuracy,
            "theoretical_roi": self.theoretical_roi
        }
    
    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"Total Games: {self.total_games}",
            f"SU Record: {self.su_correct}-{self.total_games - self.su_correct} ({self.su_accuracy:.1%})",
            f"ATS Record: {self.ats_correct}-{self.ats_games - self.ats_correct} ({self.ats_accuracy:.1%})" if self.ats_games > 0 else "ATS: N/A",
            f"MAE: {self.mean_absolute_error:.2f} pts",
            f"RMSE: {self.rmse:.2f} pts",
            f"Brier Score: {self.brier_score:.4f}",
        ]
        
        if self.high_conf_accuracy is not None:
            lines.append(f"High Conf: {self.high_conf_accuracy:.1%}")
        if self.med_conf_accuracy is not None:
            lines.append(f"Med Conf: {self.med_conf_accuracy:.1%}")
        if self.low_conf_accuracy is not None:
            lines.append(f"Low Conf: {self.low_conf_accuracy:.1%}")
        
        return "\n".join(lines)


class AccuracyCalculator:
    """
    Calculates comprehensive accuracy metrics from prediction results.
    
    Usage:
        calc = AccuracyCalculator()
        metrics = calc.calculate(results)
    """
    
    def calculate(self, results: List[PredictionResult]) -> AccuracyMetrics:
        """
        Calculate accuracy metrics from a list of results.
        
        Args:
            results: List of PredictionResult objects
        
        Returns:
            AccuracyMetrics with all calculated metrics
        """
        if not results:
            return AccuracyMetrics()
        
        metrics = AccuracyMetrics(total_games=len(results))
        
        # SU accuracy
        metrics.su_correct = sum(1 for r in results if r.su_correct)
        metrics.su_accuracy = metrics.su_correct / len(results)
        
        # ATS accuracy
        ats_results = [r for r in results if r.ats_correct is not None]
        metrics.ats_games = len(ats_results)
        if ats_results:
            metrics.ats_correct = sum(1 for r in ats_results if r.ats_correct)
            metrics.ats_accuracy = metrics.ats_correct / len(ats_results)
        
        # Spread error metrics
        spread_errors = [r.spread_error for r in results]
        metrics.mean_error = sum(spread_errors) / len(spread_errors)
        metrics.mean_absolute_error = sum(abs(e) for e in spread_errors) / len(spread_errors)
        metrics.rmse = math.sqrt(sum(e**2 for e in spread_errors) / len(spread_errors))
        
        # Calibration metrics
        metrics.brier_score = self._calculate_brier_score(results)
        metrics.log_loss = self._calculate_log_loss(results)
        
        # By confidence tier
        high_conf = [r for r in results if r.confidence_tier == "High"]
        med_conf = [r for r in results if r.confidence_tier == "Medium"]
        low_conf = [r for r in results if r.confidence_tier == "Low"]
        
        if high_conf:
            metrics.high_conf_accuracy = sum(1 for r in high_conf if r.su_correct) / len(high_conf)
        if med_conf:
            metrics.med_conf_accuracy = sum(1 for r in med_conf if r.su_correct) / len(med_conf)
        if low_conf:
            metrics.low_conf_accuracy = sum(1 for r in low_conf if r.su_correct) / len(low_conf)
        
        # Theoretical ROI (betting every game at -110)
        metrics.theoretical_roi = self._calculate_theoretical_roi(results)
        
        return metrics
    
    def _calculate_brier_score(self, results: List[PredictionResult]) -> float:
        """
        Calculate Brier score for probability calibration.
        
        Brier score = mean((predicted_prob - actual_outcome)^2)
        Lower is better. 0 = perfect, 0.25 = random guessing.
        """
        brier_sum = 0.0
        for r in results:
            # Actual outcome: 1 if predicted team won, 0 otherwise
            actual = 1.0 if r.su_correct else 0.0
            # Use the win probability for the team we predicted
            if r.predicted_winner == r.home_team:
                pred_prob = r.prediction.home_win_probability
            else:
                pred_prob = r.prediction.away_win_probability
            
            brier_sum += (pred_prob - actual) ** 2
        
        return round(brier_sum / len(results), 4)
    
    def _calculate_log_loss(self, results: List[PredictionResult]) -> float:
        """
        Calculate log loss for probability predictions.
        
        Log loss penalizes confident wrong predictions heavily.
        Lower is better.
        """
        eps = 1e-15  # Prevent log(0)
        log_loss_sum = 0.0
        
        for r in results:
            actual = 1.0 if r.su_correct else 0.0
            if r.predicted_winner == r.home_team:
                pred_prob = r.prediction.home_win_probability
            else:
                pred_prob = r.prediction.away_win_probability
            
            # Clip probability to prevent log(0)
            pred_prob = max(eps, min(1 - eps, pred_prob))
            
            log_loss_sum -= (actual * math.log(pred_prob) + 
                           (1 - actual) * math.log(1 - pred_prob))
        
        return round(log_loss_sum / len(results), 4)
    
    def _calculate_theoretical_roi(self, results: List[PredictionResult]) -> float:
        """
        Calculate theoretical ROI if betting every SU pick at -110.
        
        At -110 odds, you risk 110 to win 100.
        Break-even is 52.38% accuracy.
        """
        if not results:
            return 0.0
        
        wins = sum(1 for r in results if r.su_correct)
        losses = len(results) - wins
        
        # Profit = wins * 100 - losses * 110
        profit = wins * 100 - losses * 110
        total_risked = len(results) * 110
        
        return round((profit / total_risked) * 100, 2)


# =============================================================================
# Story 3.3: Calibration Analysis
# =============================================================================

@dataclass
class CalibrationBucket:
    """
    A bucket for calibration analysis.
    
    Groups predictions by predicted probability range.
    """
    min_prob: float
    max_prob: float
    predictions: int = 0
    wins: int = 0
    
    @property
    def actual_rate(self) -> float:
        """Actual win rate in this bucket."""
        return self.wins / self.predictions if self.predictions > 0 else 0.0
    
    @property
    def expected_rate(self) -> float:
        """Expected win rate (midpoint of bucket)."""
        return (self.min_prob + self.max_prob) / 2
    
    @property
    def calibration_error(self) -> float:
        """Difference between actual and expected rate."""
        return self.actual_rate - self.expected_rate


class CalibrationAnalyzer:
    """
    Analyzes probability calibration of predictions.
    
    A well-calibrated model should have:
    - Games predicted at 70% win prob should win ~70% of the time
    - Games predicted at 90% win prob should win ~90% of the time
    
    Usage:
        analyzer = CalibrationAnalyzer()
        analysis = analyzer.analyze(results)
    """
    
    def __init__(self, num_buckets: int = 10):
        """
        Initialize analyzer.
        
        Args:
            num_buckets: Number of probability buckets (default 10 = 10% ranges)
        """
        self.num_buckets = num_buckets
    
    def analyze(self, results: List[PredictionResult]) -> Dict:
        """
        Perform calibration analysis.
        
        Args:
            results: List of PredictionResult objects
        
        Returns:
            Dict with calibration analysis results
        """
        # Create buckets
        bucket_size = 1.0 / self.num_buckets
        buckets = []
        for i in range(self.num_buckets):
            buckets.append(CalibrationBucket(
                min_prob=i * bucket_size,
                max_prob=(i + 1) * bucket_size
            ))
        
        # Assign results to buckets
        for r in results:
            # Get win probability for the predicted winner
            if r.predicted_winner == r.home_team:
                prob = r.prediction.home_win_probability
            else:
                prob = r.prediction.away_win_probability
            
            # Find bucket
            bucket_idx = min(int(prob / bucket_size), self.num_buckets - 1)
            buckets[bucket_idx].predictions += 1
            if r.su_correct:
                buckets[bucket_idx].wins += 1
        
        # Calculate metrics
        ece = self._calculate_ece(buckets)  # Expected Calibration Error
        mce = self._calculate_mce(buckets)  # Maximum Calibration Error
        
        # Build result
        bucket_data = []
        for b in buckets:
            if b.predictions > 0:
                bucket_data.append({
                    "range": f"{b.min_prob:.0%}-{b.max_prob:.0%}",
                    "predictions": b.predictions,
                    "wins": b.wins,
                    "expected_rate": round(b.expected_rate, 3),
                    "actual_rate": round(b.actual_rate, 3),
                    "calibration_error": round(b.calibration_error, 3)
                })
        
        return {
            "expected_calibration_error": round(ece, 4),
            "maximum_calibration_error": round(mce, 4),
            "buckets": bucket_data,
            "is_well_calibrated": ece < 0.05,  # ECE < 5% is good
            "calibration_assessment": self._assess_calibration(ece)
        }
    
    def _calculate_ece(self, buckets: List[CalibrationBucket]) -> float:
        """
        Calculate Expected Calibration Error.
        
        ECE = sum(|bucket_size / total| * |actual_rate - expected_rate|)
        """
        total = sum(b.predictions for b in buckets)
        if total == 0:
            return 0.0
        
        ece = 0.0
        for b in buckets:
            if b.predictions > 0:
                weight = b.predictions / total
                ece += weight * abs(b.actual_rate - b.expected_rate)
        
        return ece
    
    def _calculate_mce(self, buckets: List[CalibrationBucket]) -> float:
        """
        Calculate Maximum Calibration Error.
        
        MCE = max(|actual_rate - expected_rate|) across non-empty buckets
        """
        errors = [abs(b.calibration_error) for b in buckets if b.predictions > 0]
        return max(errors) if errors else 0.0
    
    def _assess_calibration(self, ece: float) -> str:
        """Provide human-readable calibration assessment."""
        if ece < 0.02:
            return "Excellent - Very well calibrated"
        elif ece < 0.05:
            return "Good - Well calibrated"
        elif ece < 0.10:
            return "Fair - Slight miscalibration"
        elif ece < 0.15:
            return "Poor - Significant miscalibration"
        else:
            return "Bad - Severely miscalibrated"


# =============================================================================
# Story 3.4: Performance by Category Analytics
# =============================================================================

class PerformanceAnalyzer:
    """
    Analyzes model performance across different categories.
    
    Categories include:
    - By week of season
    - By team
    - By favorite/underdog
    - By spread size
    - By divisional vs non-divisional
    - By home/away
    - By confidence tier
    
    Usage:
        analyzer = PerformanceAnalyzer()
        report = analyzer.full_analysis(results)
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.accuracy_calc = AccuracyCalculator()
    
    def analyze_by_week(
        self,
        results: List[PredictionResult]
    ) -> List[Dict]:
        """Analyze accuracy by week of season."""
        by_week: Dict[int, List[PredictionResult]] = {}
        for r in results:
            if r.week not in by_week:
                by_week[r.week] = []
            by_week[r.week].append(r)
        
        analysis = []
        for week in sorted(by_week.keys()):
            week_results = by_week[week]
            correct = sum(1 for r in week_results if r.su_correct)
            analysis.append({
                "week": week,
                "games": len(week_results),
                "correct": correct,
                "accuracy": round(correct / len(week_results), 3)
            })
        
        return analysis
    
    def analyze_by_team(
        self,
        results: List[PredictionResult]
    ) -> List[Dict]:
        """Analyze accuracy by team (when picking for/against)."""
        team_picks: Dict[str, Dict] = {}
        
        for r in results:
            # Track when we picked this team
            picked = r.predicted_winner
            if picked not in team_picks:
                team_picks[picked] = {"picked_for": 0, "correct_for": 0}
            team_picks[picked]["picked_for"] += 1
            if r.su_correct:
                team_picks[picked]["correct_for"] += 1
            
            # Track when we picked against this team
            opponent = r.away_team if r.home_team == picked else r.home_team
            if opponent not in team_picks:
                team_picks[opponent] = {"picked_for": 0, "correct_for": 0}
        
        analysis = []
        for team, data in sorted(team_picks.items()):
            if data["picked_for"] > 0:
                analysis.append({
                    "team": team,
                    "times_picked": data["picked_for"],
                    "correct": data["correct_for"],
                    "accuracy": round(data["correct_for"] / data["picked_for"], 3)
                })
        
        return sorted(analysis, key=lambda x: x["accuracy"], reverse=True)
    
    def analyze_by_spread_size(
        self,
        results: List[PredictionResult]
    ) -> List[Dict]:
        """Analyze accuracy by predicted spread size."""
        buckets = [
            ("Pick'em (0-2)", 0, 2),
            ("Small (2.5-5)", 2.5, 5),
            ("Medium (5.5-9)", 5.5, 9),
            ("Large (9.5-13)", 9.5, 13),
            ("Blowout (13.5+)", 13.5, 100),
        ]
        
        analysis = []
        for name, min_spread, max_spread in buckets:
            bucket_results = [
                r for r in results
                if min_spread <= abs(r.prediction.predicted_spread) <= max_spread
            ]
            
            if bucket_results:
                correct = sum(1 for r in bucket_results if r.su_correct)
                analysis.append({
                    "category": name,
                    "games": len(bucket_results),
                    "correct": correct,
                    "accuracy": round(correct / len(bucket_results), 3)
                })
        
        return analysis
    
    def analyze_favorites_vs_underdogs(
        self,
        results: List[PredictionResult]
    ) -> Dict:
        """Analyze accuracy when picking favorites vs underdogs."""
        favorites = []
        underdogs = []
        
        for r in results:
            spread = r.prediction.predicted_spread
            # Negative spread = home team is favorite
            if spread < 0:
                home_is_favorite = True
            else:
                home_is_favorite = False
            
            # Did we pick the favorite or underdog?
            picked_home = (r.predicted_winner == r.home_team)
            picked_favorite = (picked_home == home_is_favorite)
            
            if picked_favorite:
                favorites.append(r)
            else:
                underdogs.append(r)
        
        fav_correct = sum(1 for r in favorites if r.su_correct)
        dog_correct = sum(1 for r in underdogs if r.su_correct)
        
        return {
            "favorites": {
                "games": len(favorites),
                "correct": fav_correct,
                "accuracy": round(fav_correct / len(favorites), 3) if favorites else 0
            },
            "underdogs": {
                "games": len(underdogs),
                "correct": dog_correct,
                "accuracy": round(dog_correct / len(underdogs), 3) if underdogs else 0
            }
        }
    
    def analyze_divisional_vs_non(
        self,
        results: List[PredictionResult]
    ) -> Dict:
        """Analyze accuracy for divisional vs non-divisional games."""
        divisional = [r for r in results if r.prediction.game.is_divisional]
        non_div = [r for r in results if not r.prediction.game.is_divisional]
        
        div_correct = sum(1 for r in divisional if r.su_correct)
        non_correct = sum(1 for r in non_div if r.su_correct)
        
        return {
            "divisional": {
                "games": len(divisional),
                "correct": div_correct,
                "accuracy": round(div_correct / len(divisional), 3) if divisional else 0
            },
            "non_divisional": {
                "games": len(non_div),
                "correct": non_correct,
                "accuracy": round(non_correct / len(non_div), 3) if non_div else 0
            }
        }
    
    def analyze_by_confidence(
        self,
        results: List[PredictionResult]
    ) -> Dict:
        """Analyze accuracy by confidence tier."""
        by_conf: Dict[str, List[PredictionResult]] = {}
        for r in results:
            tier = r.confidence_tier
            if tier not in by_conf:
                by_conf[tier] = []
            by_conf[tier].append(r)
        
        analysis = {}
        for tier, tier_results in by_conf.items():
            correct = sum(1 for r in tier_results if r.su_correct)
            analysis[tier] = {
                "games": len(tier_results),
                "correct": correct,
                "accuracy": round(correct / len(tier_results), 3)
            }
        
        return analysis
    
    def full_analysis(self, results: List[PredictionResult]) -> Dict:
        """
        Perform comprehensive performance analysis.
        
        Returns:
            Dict with all analysis categories
        """
        return {
            "overall": self.accuracy_calc.calculate(results).to_dict(),
            "by_week": self.analyze_by_week(results),
            "by_team": self.analyze_by_team(results),
            "by_spread_size": self.analyze_by_spread_size(results),
            "favorites_vs_underdogs": self.analyze_favorites_vs_underdogs(results),
            "divisional_vs_non": self.analyze_divisional_vs_non(results),
            "by_confidence": self.analyze_by_confidence(results)
        }


# =============================================================================
# Story 3.5: Season Backtester
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.
    
    Attributes:
        start_week: First week to include predictions (default 1)
        end_week: Last week to include (default 18)
        min_confidence_score: Minimum confidence to include pick
        require_vegas_line: Only include games with Vegas spreads
        ats_min_edge: Minimum edge for ATS picks (e.g., 2.0 points)
    """
    start_week: int = 1
    end_week: int = 18
    min_confidence_score: int = 0
    require_vegas_line: bool = False
    ats_min_edge: float = 0.0


class SeasonBacktester:
    """
    Runs backtest of model predictions against historical results.
    
    This simulates what would have happened if you used the model
    to make picks throughout a season.
    
    Usage:
        backtester = SeasonBacktester(model, schedule_manager)
        report = backtester.run_backtest()
    """
    
    def __init__(
        self,
        season_model: 'NFLSeasonModel',
        schedule_manager: WeeklyScheduleManager,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Args:
            season_model: The NFLSeasonModel to use
            schedule_manager: WeeklyScheduleManager with games and results
            config: BacktestConfig for filtering
        """
        self.model = season_model
        self.schedule = schedule_manager
        self.config = config or BacktestConfig()
        
        self.tracker = PredictionResultTracker()
        self.accuracy_calc = AccuracyCalculator()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_backtest(self) -> Dict:
        """
        Run the full backtest.
        
        Returns:
            Dict with comprehensive backtest results
        """
        self.tracker.clear()
        
        # Process each week
        weekly_results = []
        
        for week in range(self.config.start_week, self.config.end_week + 1):
            week_games = self.schedule.get_week_games(week)
            if not week_games:
                continue
            
            week_data = {"week": week, "games": 0, "correct": 0}
            
            for game in week_games:
                # Skip incomplete games
                if not game.is_completed:
                    continue
                
                # Apply filters
                if self.config.require_vegas_line and game.vegas_spread is None:
                    continue
                
                # Generate prediction
                prediction = self.schedule.predictor.predict_game(game, week)
                
                # Apply confidence filter
                if prediction.confidence_score < self.config.min_confidence_score:
                    continue
                
                # Record result
                result = self.tracker.add_result(
                    prediction,
                    game.home_score,
                    game.away_score
                )
                
                week_data["games"] += 1
                if result.su_correct:
                    week_data["correct"] += 1
            
            if week_data["games"] > 0:
                week_data["accuracy"] = round(
                    week_data["correct"] / week_data["games"], 3
                )
                weekly_results.append(week_data)
        
        # Calculate overall metrics
        all_results = self.tracker.get_all_results()
        
        if not all_results:
            return {"error": "No completed games found for backtest"}
        
        metrics = self.accuracy_calc.calculate(all_results)
        calibration = self.calibration_analyzer.analyze(all_results)
        performance = self.performance_analyzer.full_analysis(all_results)
        
        return {
            "config": {
                "start_week": self.config.start_week,
                "end_week": self.config.end_week,
                "min_confidence": self.config.min_confidence_score,
                "require_vegas": self.config.require_vegas_line
            },
            "summary": {
                "total_games": metrics.total_games,
                "su_record": f"{metrics.su_correct}-{metrics.total_games - metrics.su_correct}",
                "su_accuracy": f"{metrics.su_accuracy:.1%}",
                "ats_record": f"{metrics.ats_correct}-{metrics.ats_games - metrics.ats_correct}" if metrics.ats_games > 0 else "N/A",
                "ats_accuracy": f"{metrics.ats_accuracy:.1%}" if metrics.ats_games > 0 else "N/A",
                "mae": f"{metrics.mean_absolute_error:.2f} pts",
                "rmse": f"{metrics.rmse:.2f} pts",
                "brier_score": metrics.brier_score,
                "theoretical_roi": f"{metrics.theoretical_roi:.1f}%"
            },
            "weekly_results": weekly_results,
            "calibration": calibration,
            "performance_breakdown": performance,
            "metrics": metrics.to_dict()
        }
    
    def run_rolling_backtest(
        self,
        window_size: int = 4
    ) -> List[Dict]:
        """
        Run rolling window backtest to see how accuracy changes.
        
        Args:
            window_size: Number of weeks in each window
        
        Returns:
            List of accuracy data for each rolling window
        """
        results = []
        
        for start in range(self.config.start_week, 
                          self.config.end_week - window_size + 2):
            end = start + window_size - 1
            
            # Create temp config for this window
            temp_config = BacktestConfig(
                start_week=start,
                end_week=end,
                min_confidence_score=self.config.min_confidence_score,
                require_vegas_line=self.config.require_vegas_line
            )
            
            temp_backtester = SeasonBacktester(
                self.model,
                self.schedule,
                temp_config
            )
            
            window_results = temp_backtester.run_backtest()
            
            if "error" not in window_results:
                results.append({
                    "window": f"Week {start}-{end}",
                    "games": window_results["summary"]["total_games"],
                    "su_accuracy": window_results["summary"]["su_accuracy"],
                    "mae": window_results["summary"]["mae"]
                })
        
        return results


# =============================================================================
# Story 3.6: Performance Report Generator
# =============================================================================

class PerformanceReportGenerator:
    """
    Generates formatted performance reports.
    
    Outputs can be:
    - Text summary
    - Detailed breakdown
    - Weekly report
    
    Usage:
        generator = PerformanceReportGenerator()
        report = generator.generate_season_report(backtest_results)
    """
    
    def generate_season_report(self, backtest_results: Dict) -> str:
        """
        Generate a comprehensive season report.
        
        Args:
            backtest_results: Output from SeasonBacktester.run_backtest()
        
        Returns:
            Formatted string report
        """
        if "error" in backtest_results:
            return f"Error: {backtest_results['error']}"
        
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("NFL PREDICTION MODEL - SEASON PERFORMANCE REPORT")
        lines.append("=" * 70)
        
        # Summary
        summary = backtest_results["summary"]
        lines.append("\n📊 OVERALL PERFORMANCE")
        lines.append("-" * 40)
        lines.append(f"Total Games:      {summary['total_games']}")
        lines.append(f"Straight-Up:      {summary['su_record']} ({summary['su_accuracy']})")
        lines.append(f"Against Spread:   {summary['ats_record']} ({summary['ats_accuracy']})")
        lines.append(f"Mean Abs Error:   {summary['mae']}")
        lines.append(f"RMSE:             {summary['rmse']}")
        lines.append(f"Brier Score:      {summary['brier_score']}")
        lines.append(f"Theoretical ROI:  {summary['theoretical_roi']}")
        
        # Calibration
        cal = backtest_results["calibration"]
        lines.append(f"\n📈 PROBABILITY CALIBRATION")
        lines.append("-" * 40)
        lines.append(f"Expected Cal Error: {cal['expected_calibration_error']:.4f}")
        lines.append(f"Assessment:         {cal['calibration_assessment']}")
        
        # By confidence
        by_conf = backtest_results["performance_breakdown"]["by_confidence"]
        lines.append(f"\n🎯 ACCURACY BY CONFIDENCE")
        lines.append("-" * 40)
        for tier in ["High", "Medium", "Low"]:
            if tier in by_conf:
                data = by_conf[tier]
                lines.append(f"{tier:10} {data['correct']}/{data['games']} ({data['accuracy']:.1%})")
        
        # By spread size
        by_spread = backtest_results["performance_breakdown"]["by_spread_size"]
        lines.append(f"\n📏 ACCURACY BY SPREAD SIZE")
        lines.append("-" * 40)
        for bucket in by_spread:
            lines.append(
                f"{bucket['category']:20} {bucket['correct']}/{bucket['games']} ({bucket['accuracy']:.1%})"
            )
        
        # Favorites vs underdogs
        fav_dog = backtest_results["performance_breakdown"]["favorites_vs_underdogs"]
        lines.append(f"\n⭐ FAVORITES VS UNDERDOGS")
        lines.append("-" * 40)
        fav = fav_dog["favorites"]
        dog = fav_dog["underdogs"]
        lines.append(f"Favorites:  {fav['correct']}/{fav['games']} ({fav['accuracy']:.1%})")
        lines.append(f"Underdogs:  {dog['correct']}/{dog['games']} ({dog['accuracy']:.1%})")
        
        # Divisional
        div = backtest_results["performance_breakdown"]["divisional_vs_non"]
        lines.append(f"\n🏈 DIVISIONAL VS NON-DIVISIONAL")
        lines.append("-" * 40)
        d = div["divisional"]
        nd = div["non_divisional"]
        lines.append(f"Divisional:     {d['correct']}/{d['games']} ({d['accuracy']:.1%})")
        lines.append(f"Non-Divisional: {nd['correct']}/{nd['games']} ({nd['accuracy']:.1%})")
        
        # Weekly breakdown
        lines.append(f"\n📅 WEEKLY BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"{'Week':<8} {'Games':<8} {'Correct':<10} {'Accuracy':<10}")
        for week in backtest_results["weekly_results"]:
            lines.append(
                f"{week['week']:<8} {week['games']:<8} {week['correct']:<10} {week['accuracy']:.1%}"
            )
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def generate_weekly_report(
        self,
        tracker: PredictionResultTracker,
        week: int
    ) -> str:
        """Generate a report for a single week."""
        results = tracker.get_results_by_week(week)
        
        if not results:
            return f"No results found for week {week}"
        
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"WEEK {week} RESULTS")
        lines.append(f"{'='*60}")
        
        correct = sum(1 for r in results if r.su_correct)
        lines.append(f"\nRecord: {correct}-{len(results) - correct}")
        lines.append(f"Accuracy: {correct / len(results):.1%}")
        
        lines.append(f"\n{'Game':<35} {'Pick':<10} {'Result':<10} {'Spread Err':<10}")
        lines.append("-" * 65)
        
        for r in results:
            game_str = f"{r.away_team[:12]} @ {r.home_team[:12]}"
            pick_str = r.predicted_winner[:10]
            result_str = "✓" if r.su_correct else "✗"
            err_str = f"{r.spread_error:+.1f}"
            lines.append(f"{game_str:<35} {pick_str:<10} {result_str:<10} {err_str:<10}")
        
        return "\n".join(lines)
    
    def generate_best_and_worst(
        self,
        results: List[PredictionResult],
        n: int = 5
    ) -> str:
        """Generate report of best and worst predictions."""
        lines = []
        
        # Sort by spread error
        sorted_by_error = sorted(results, key=lambda r: r.abs_spread_error)
        
        lines.append("\n🎯 MOST ACCURATE PREDICTIONS")
        lines.append("-" * 50)
        for r in sorted_by_error[:n]:
            lines.append(
                f"{r.away_team} @ {r.home_team}: "
                f"Pred {r.prediction.predicted_spread:+.1f}, "
                f"Actual {r.actual_spread:+.1f} "
                f"(Error: {r.spread_error:+.1f})"
            )
        
        lines.append("\n💥 LEAST ACCURATE PREDICTIONS")
        lines.append("-" * 50)
        for r in sorted_by_error[-n:]:
            lines.append(
                f"{r.away_team} @ {r.home_team}: "
                f"Pred {r.prediction.predicted_spread:+.1f}, "
                f"Actual {r.actual_spread:+.1f} "
                f"(Error: {r.spread_error:+.1f})"
            )
        
        return "\n".join(lines)


# =============================================================================
# =============================================================================
# EPIC 4: DATA INTEGRATION & AUTOMATION
# =============================================================================
# =============================================================================


# =============================================================================
# Story 4.1: Data Source Abstraction Layer
# =============================================================================

class DataSourceType(Enum):
    """Types of data sources."""
    EPA = "epa"
    SCHEDULE = "schedule"
    VEGAS_LINES = "vegas_lines"
    GAME_RESULTS = "game_results"
    PRESEASON = "preseason"
    INJURIES = "injuries"
    WEATHER = "weather"


class DataFetchStatus(Enum):
    """Status of a data fetch operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CACHED = "cached"
    RATE_LIMITED = "rate_limited"


@dataclass
class DataFetchResult:
    """
    Result of a data fetch operation.
    
    Attributes:
        source: Name of the data source
        source_type: Type of data fetched
        status: Status of the fetch
        data: The fetched data (type varies)
        timestamp: When the fetch occurred
        cache_valid_until: When cached data expires
        error_message: Error message if failed
        records_fetched: Number of records retrieved
        metadata: Additional source-specific metadata
    """
    source: str
    source_type: DataSourceType
    status: DataFetchStatus
    data: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    cache_valid_until: Optional[datetime] = None
    error_message: Optional[str] = None
    records_fetched: int = 0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.status in [DataFetchStatus.SUCCESS, DataFetchStatus.CACHED]
    
    @property
    def is_fresh(self) -> bool:
        """Check if cached data is still valid."""
        if self.cache_valid_until is None:
            return True
        return datetime.now() < self.cache_valid_until


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    Data sources provide a consistent interface for fetching
    NFL data from various providers (APIs, web scraping, files).
    """
    
    def __init__(self, name: str, source_type: DataSourceType):
        """
        Initialize data source.
        
        Args:
            name: Human-readable name for the source
            source_type: Type of data this source provides
        """
        self.name = name
        self.source_type = source_type
        self._cache: Dict[str, DataFetchResult] = {}
        self._last_fetch: Optional[datetime] = None
        self._rate_limit_until: Optional[datetime] = None
    
    @abstractmethod
    def fetch(self, **kwargs) -> DataFetchResult:
        """
        Fetch data from the source.
        
        Returns:
            DataFetchResult with the fetched data
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is currently available."""
        pass
    
    def get_cached(self, cache_key: str) -> Optional[DataFetchResult]:
        """Get cached result if available and fresh."""
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.is_fresh:
                return cached
        return None
    
    def set_cache(
        self,
        cache_key: str,
        result: DataFetchResult,
        ttl_minutes: int = 60
    ) -> None:
        """Cache a result with time-to-live."""
        result.cache_valid_until = datetime.now() + timedelta(minutes=ttl_minutes)
        result.status = DataFetchStatus.CACHED
        self._cache[cache_key] = result
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache = {}
    
    def is_rate_limited(self) -> bool:
        """Check if we're currently rate limited."""
        if self._rate_limit_until is None:
            return False
        return datetime.now() < self._rate_limit_until
    
    def set_rate_limit(self, seconds: int) -> None:
        """Set rate limit for this source."""
        self._rate_limit_until = datetime.now() + timedelta(seconds=seconds)


class DataSourceRegistry:
    """
    Registry for managing multiple data sources.
    
    Allows registering sources and fetching from the best available.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._sources: Dict[DataSourceType, List[DataSource]] = {}
    
    def register(self, source: DataSource) -> None:
        """Register a data source."""
        if source.source_type not in self._sources:
            self._sources[source.source_type] = []
        self._sources[source.source_type].append(source)
    
    def get_sources(self, source_type: DataSourceType) -> List[DataSource]:
        """Get all sources of a given type."""
        return self._sources.get(source_type, [])
    
    def fetch_best(
        self,
        source_type: DataSourceType,
        **kwargs
    ) -> DataFetchResult:
        """
        Fetch from the best available source of a given type.
        
        Tries sources in order until one succeeds.
        """
        sources = self.get_sources(source_type)
        
        if not sources:
            return DataFetchResult(
                source="none",
                source_type=source_type,
                status=DataFetchStatus.FAILED,
                error_message=f"No sources registered for {source_type.value}"
            )
        
        for source in sources:
            if not source.is_available():
                continue
            if source.is_rate_limited():
                continue
            
            result = source.fetch(**kwargs)
            if result.is_success:
                return result
        
        return DataFetchResult(
            source="all",
            source_type=source_type,
            status=DataFetchStatus.FAILED,
            error_message="All sources failed or unavailable"
        )


# =============================================================================
# Story 4.2: EPA Data Fetcher
# =============================================================================

# Team name mappings for different data sources
TEAM_NAME_ALIASES = {
    # Standard -> [aliases]
    "Arizona Cardinals": ["ARI", "Cardinals", "Arizona"],
    "Atlanta Falcons": ["ATL", "Falcons", "Atlanta"],
    "Baltimore Ravens": ["BAL", "Ravens", "Baltimore"],
    "Buffalo Bills": ["BUF", "Bills", "Buffalo"],
    "Carolina Panthers": ["CAR", "Panthers", "Carolina"],
    "Chicago Bears": ["CHI", "Bears", "Chicago"],
    "Cincinnati Bengals": ["CIN", "Bengals", "Cincinnati"],
    "Cleveland Browns": ["CLE", "Browns", "Cleveland"],
    "Dallas Cowboys": ["DAL", "Cowboys", "Dallas"],
    "Denver Broncos": ["DEN", "Broncos", "Denver"],
    "Detroit Lions": ["DET", "Lions", "Detroit"],
    "Green Bay Packers": ["GB", "GNB", "Packers", "Green Bay"],
    "Houston Texans": ["HOU", "Texans", "Houston"],
    "Indianapolis Colts": ["IND", "Colts", "Indianapolis"],
    "Jacksonville Jaguars": ["JAX", "JAC", "Jaguars", "Jacksonville"],
    "Kansas City Chiefs": ["KC", "KAN", "Chiefs", "Kansas City"],
    "Las Vegas Raiders": ["LV", "LVR", "Raiders", "Las Vegas", "Oakland Raiders", "OAK"],
    "Los Angeles Chargers": ["LAC", "Chargers", "San Diego Chargers", "SD"],
    "Los Angeles Rams": ["LAR", "LA", "Rams", "St. Louis Rams", "STL"],
    "Miami Dolphins": ["MIA", "Dolphins", "Miami"],
    "Minnesota Vikings": ["MIN", "Vikings", "Minnesota"],
    "New England Patriots": ["NE", "NEP", "Patriots", "New England"],
    "New Orleans Saints": ["NO", "NOR", "Saints", "New Orleans"],
    "New York Giants": ["NYG", "Giants"],
    "New York Jets": ["NYJ", "Jets"],
    "Philadelphia Eagles": ["PHI", "Eagles", "Philadelphia"],
    "Pittsburgh Steelers": ["PIT", "Steelers", "Pittsburgh"],
    "San Francisco 49ers": ["SF", "SFO", "49ers", "San Francisco", "Niners"],
    "Seattle Seahawks": ["SEA", "Seahawks", "Seattle"],
    "Tampa Bay Buccaneers": ["TB", "TBB", "TAM", "Buccaneers", "Tampa Bay", "Bucs"],
    "Tennessee Titans": ["TEN", "Titans", "Tennessee"],
    "Washington Commanders": ["WAS", "WSH", "Commanders", "Washington", 
                              "Washington Football Team", "WFT", "Redskins"],
}

# Reverse lookup: alias -> standard name
TEAM_ALIAS_LOOKUP: Dict[str, str] = {}
for standard, aliases in TEAM_NAME_ALIASES.items():
    TEAM_ALIAS_LOOKUP[standard.lower()] = standard
    for alias in aliases:
        TEAM_ALIAS_LOOKUP[alias.lower()] = standard


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name to the standard format.
    
    Args:
        name: Team name in any format
    
    Returns:
        Standard team name (e.g., "Buffalo Bills")
    
    Raises:
        ValueError: If team name is not recognized
    """
    lookup = name.strip().lower()
    if lookup in TEAM_ALIAS_LOOKUP:
        return TEAM_ALIAS_LOOKUP[lookup]
    
    # Try partial match
    for alias, standard in TEAM_ALIAS_LOOKUP.items():
        if alias in lookup or lookup in alias:
            return standard
    
    raise ValueError(f"Unknown team name: {name}")


@dataclass
class EPADataPoint:
    """
    EPA data for a single team at a point in time.
    
    Attributes:
        team: Standard team name
        season: NFL season year
        week: Week number (0 = preseason)
        
        # Core EPA metrics
        offensive_epa_per_play: Offensive EPA/play
        defensive_epa_per_play: Defensive EPA allowed/play
        net_epa_per_play: Offensive - Defensive
        
        # Split metrics
        pass_epa_offense: Passing EPA/play
        rush_epa_offense: Rushing EPA/play
        pass_epa_defense: Pass defense EPA allowed/play
        rush_epa_defense: Rush defense EPA allowed/play
        
        # Volume
        total_plays: Total plays in sample
        games_played: Games in sample
        
        # Source info
        source: Data source name
        timestamp: When data was fetched
    """
    team: str
    season: int
    week: int
    
    offensive_epa_per_play: float
    defensive_epa_per_play: float
    net_epa_per_play: float = 0.0
    
    pass_epa_offense: Optional[float] = None
    rush_epa_offense: Optional[float] = None
    pass_epa_defense: Optional[float] = None
    rush_epa_defense: Optional[float] = None
    
    total_plays: int = 0
    games_played: int = 0
    
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.net_epa_per_play = round(
            self.offensive_epa_per_play - self.defensive_epa_per_play, 4
        )
    
    def to_in_season_data(self, wins: int = 0, losses: int = 0) -> TeamInSeasonData:
        """Convert to TeamInSeasonData for use in the model."""
        return TeamInSeasonData(
            team=self.team,
            season=self.season,
            week=self.week,
            games_played=self.games_played,
            offensive_epa_per_play=self.offensive_epa_per_play,
            defensive_epa_per_play=self.defensive_epa_per_play,
            passing_epa_per_play=self.pass_epa_offense,
            rushing_epa_per_play=self.rush_epa_offense,
            pass_defense_epa_per_play=self.pass_epa_defense,
            rush_defense_epa_per_play=self.rush_epa_defense,
            total_offensive_plays=self.total_plays // 2,
            total_defensive_plays=self.total_plays // 2,
            wins=wins,
            losses=losses
        )


class MockEPADataSource(DataSource):
    """
    Mock EPA data source for testing and demonstration.
    
    Generates realistic-looking EPA data without external dependencies.
    In production, replace with actual data source implementations.
    """
    
    def __init__(self):
        """Initialize mock data source."""
        super().__init__("Mock EPA", DataSourceType.EPA)
        self._mock_data = self._generate_mock_data()
    
    def fetch(
        self,
        season: int = 2025,
        week: Optional[int] = None,
        team: Optional[str] = None,
        **kwargs
    ) -> DataFetchResult:
        """
        Fetch mock EPA data.
        
        Args:
            season: NFL season year
            week: Specific week (None = all weeks)
            team: Specific team (None = all teams)
        
        Returns:
            DataFetchResult with EPA data
        """
        # Check cache
        cache_key = f"{season}_{week}_{team}"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        # Filter mock data
        data = []
        for epa_point in self._mock_data:
            if epa_point.season != season:
                continue
            if week is not None and epa_point.week != week:
                continue
            if team is not None and epa_point.team != team:
                continue
            data.append(epa_point)
        
        result = DataFetchResult(
            source=self.name,
            source_type=self.source_type,
            status=DataFetchStatus.SUCCESS,
            data={"epa_data": data},
            records_fetched=len(data),
            metadata={"season": season, "week": week}
        )
        
        self.set_cache(cache_key, result, ttl_minutes=30)
        return result
    
    def is_available(self) -> bool:
        """Mock source is always available."""
        return True
    
    def _generate_mock_data(self) -> List[EPADataPoint]:
        """Generate realistic mock EPA data for all teams."""
        import random
        random.seed(42)  # Reproducible
        
        data = []
        
        # Base EPA values by team tier
        tier_ranges = {
            "elite": (0.10, 0.20, -0.10, 0.00),      # off_min, off_max, def_min, def_max
            "good": (0.02, 0.12, -0.05, 0.05),
            "average": (-0.05, 0.05, -0.03, 0.08),
            "below_avg": (-0.10, 0.00, 0.00, 0.12),
            "poor": (-0.18, -0.05, 0.05, 0.18),
        }
        
        team_tiers = {
            "elite": ["Buffalo Bills", "Kansas City Chiefs", "San Francisco 49ers", 
                     "Baltimore Ravens", "Detroit Lions"],
            "good": ["Philadelphia Eagles", "Miami Dolphins", "Dallas Cowboys",
                    "Cincinnati Bengals", "Houston Texans", "Los Angeles Chargers"],
            "average": ["Green Bay Packers", "Seattle Seahawks", "Pittsburgh Steelers",
                       "Los Angeles Rams", "Tampa Bay Buccaneers", "Minnesota Vikings",
                       "Denver Broncos", "Jacksonville Jaguars", "Cleveland Browns"],
            "below_avg": ["Indianapolis Colts", "Atlanta Falcons", "New Orleans Saints",
                         "Chicago Bears", "Las Vegas Raiders", "New York Jets",
                         "Arizona Cardinals", "Tennessee Titans", "Washington Commanders"],
            "poor": ["New England Patriots", "New York Giants", "Carolina Panthers"]
        }
        
        for tier, teams in team_tiers.items():
            off_min, off_max, def_min, def_max = tier_ranges[tier]
            
            for team in teams:
                for week in range(1, 19):
                    # Add some weekly variance
                    off_epa = random.uniform(off_min, off_max) + random.gauss(0, 0.03)
                    def_epa = random.uniform(def_min, def_max) + random.gauss(0, 0.03)
                    
                    data.append(EPADataPoint(
                        team=team,
                        season=2025,
                        week=week,
                        offensive_epa_per_play=round(off_epa, 4),
                        defensive_epa_per_play=round(def_epa, 4),
                        pass_epa_offense=round(off_epa * 1.2, 4),
                        rush_epa_offense=round(off_epa * 0.7, 4),
                        pass_epa_defense=round(def_epa * 1.1, 4),
                        rush_epa_defense=round(def_epa * 0.8, 4),
                        total_plays=random.randint(55, 75) * week,
                        games_played=week,
                        source="MockEPA"
                    ))
        
        return data


class NFLFastRDataSource(DataSource):
    """
    Data source for nflfastR/nflverse data.
    
    nflfastR is an R package that provides play-by-play data.
    This would typically fetch from their data repository or
    a local database populated from nflfastR.
    
    Note: This is a template - actual implementation would
    require R integration or pre-processed data files.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize nflfastR data source.
        
        Args:
            data_path: Path to local nflfastR data files
        """
        super().__init__("nflfastR", DataSourceType.EPA)
        self.data_path = data_path
        self._available = data_path is not None
    
    def fetch(
        self,
        season: int = 2025,
        week: Optional[int] = None,
        **kwargs
    ) -> DataFetchResult:
        """
        Fetch EPA data from nflfastR.
        
        In production, this would:
        1. Load play-by-play data from parquet/csv files
        2. Calculate EPA aggregates per team
        3. Return structured EPADataPoint objects
        """
        if not self._available:
            return DataFetchResult(
                source=self.name,
                source_type=self.source_type,
                status=DataFetchStatus.FAILED,
                error_message="nflfastR data path not configured"
            )
        
        # Template for actual implementation
        return DataFetchResult(
            source=self.name,
            source_type=self.source_type,
            status=DataFetchStatus.FAILED,
            error_message="nflfastR integration not implemented - use MockEPADataSource for testing"
        )
    
    def is_available(self) -> bool:
        """Check if nflfastR data is available."""
        return self._available


# =============================================================================
# Story 4.3: Schedule & Results Fetcher
# =============================================================================

@dataclass
class ScheduleEntry:
    """
    A single game in the schedule.
    
    Attributes:
        game_id: Unique identifier
        season: NFL season year
        week: Week number
        game_date: Date of game
        game_time: Start time
        
        home_team: Home team name
        away_team: Away team name
        
        # Location
        stadium: Stadium name
        is_neutral: Neutral site game
        
        # Results (if completed)
        home_score: Home team final score
        away_score: Away team final score
        is_completed: Whether game has been played
        
        # Broadcast
        network: TV network
        is_primetime: SNF/MNF/TNF
        
        # Source
        source: Data source name
    """
    game_id: str
    season: int
    week: int
    game_date: date
    game_time: str
    
    home_team: str
    away_team: str
    
    stadium: str = ""
    is_neutral: bool = False
    
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    is_completed: bool = False
    
    network: str = ""
    is_primetime: bool = False
    
    source: str = "unknown"
    
    def to_nfl_game(self) -> NFLGame:
        """Convert to NFLGame for use in predictions."""
        return NFLGame(
            game_id=self.game_id,
            season=self.season,
            week=self.week,
            game_date=self.game_date,
            game_time=self.game_time,
            home_team=self.home_team,
            away_team=self.away_team,
            stadium=self.stadium,
            location=GameLocation.NEUTRAL if self.is_neutral else GameLocation.HOME,
            home_score=self.home_score,
            away_score=self.away_score,
            is_completed=self.is_completed
        )


class MockScheduleDataSource(DataSource):
    """
    Mock schedule data source for testing.
    
    Generates a realistic NFL schedule.
    """
    
    def __init__(self):
        """Initialize mock schedule source."""
        super().__init__("Mock Schedule", DataSourceType.SCHEDULE)
    
    def fetch(
        self,
        season: int = 2025,
        week: Optional[int] = None,
        **kwargs
    ) -> DataFetchResult:
        """
        Fetch mock schedule data.
        
        Args:
            season: NFL season year
            week: Specific week (None = all weeks)
        """
        schedule = self._generate_mock_schedule(season)
        
        if week is not None:
            schedule = [g for g in schedule if g.week == week]
        
        return DataFetchResult(
            source=self.name,
            source_type=self.source_type,
            status=DataFetchStatus.SUCCESS,
            data={"schedule": schedule},
            records_fetched=len(schedule)
        )
    
    def is_available(self) -> bool:
        return True
    
    def _generate_mock_schedule(self, season: int) -> List[ScheduleEntry]:
        """Generate a mock NFL schedule."""
        import random
        random.seed(season)
        
        schedule = []
        teams = ALL_NFL_TEAMS.copy()
        
        # Generate 16 games per week (standard slate)
        for week in range(1, 19):
            random.shuffle(teams)
            week_start = date(season, 9, 7) + timedelta(weeks=week-1)
            
            for i in range(0, 32, 2):
                home = teams[i]
                away = teams[i + 1]
                
                game_date = week_start + timedelta(days=random.choice([0, 0, 0, 4]))  # Mostly Sunday
                
                schedule.append(ScheduleEntry(
                    game_id=f"{season}_W{week}_{away[:3]}@{home[:3]}",
                    season=season,
                    week=week,
                    game_date=game_date,
                    game_time="1:00 PM",
                    home_team=home,
                    away_team=away,
                    source="MockSchedule"
                ))
        
        return schedule


# =============================================================================
# Story 4.4: Vegas Lines Fetcher
# =============================================================================

@dataclass
class VegasLine:
    """
    Vegas betting line for a game.
    
    Attributes:
        game_id: Game identifier
        
        # Spread
        spread: Point spread (negative = home favored)
        spread_home_odds: Odds for home spread (-110 typical)
        spread_away_odds: Odds for away spread
        
        # Total
        total: Over/under total points
        over_odds: Odds for over
        under_odds: Odds for under
        
        # Moneyline
        home_ml: Home team moneyline
        away_ml: Away team moneyline
        
        # Metadata
        book: Sportsbook name
        timestamp: When line was captured
        is_opening: Opening line vs current
    """
    game_id: str
    
    spread: float
    spread_home_odds: int = -110
    spread_away_odds: int = -110
    
    total: float = 45.0
    over_odds: int = -110
    under_odds: int = -110
    
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None
    
    book: str = "Consensus"
    timestamp: datetime = field(default_factory=datetime.now)
    is_opening: bool = False
    
    @property
    def home_implied_prob(self) -> float:
        """Calculate implied probability for home team from ML."""
        if self.home_ml is None:
            return 0.5
        
        if self.home_ml < 0:
            return abs(self.home_ml) / (abs(self.home_ml) + 100)
        else:
            return 100 / (self.home_ml + 100)
    
    @property
    def away_implied_prob(self) -> float:
        """Calculate implied probability for away team from ML."""
        if self.away_ml is None:
            return 0.5
        
        if self.away_ml < 0:
            return abs(self.away_ml) / (abs(self.away_ml) + 100)
        else:
            return 100 / (self.away_ml + 100)


class MockVegasDataSource(DataSource):
    """
    Mock Vegas lines data source for testing.
    
    Generates realistic betting lines based on team quality.
    """
    
    def __init__(self):
        """Initialize mock Vegas source."""
        super().__init__("Mock Vegas", DataSourceType.VEGAS_LINES)
    
    def fetch(
        self,
        season: int = 2025,
        week: Optional[int] = None,
        game_id: Optional[str] = None,
        **kwargs
    ) -> DataFetchResult:
        """Fetch mock Vegas lines."""
        lines = self._generate_mock_lines(season, week)
        
        if game_id:
            lines = [l for l in lines if l.game_id == game_id]
        
        return DataFetchResult(
            source=self.name,
            source_type=self.source_type,
            status=DataFetchStatus.SUCCESS,
            data={"lines": lines},
            records_fetched=len(lines)
        )
    
    def is_available(self) -> bool:
        return True
    
    def _generate_mock_lines(
        self,
        season: int,
        week: Optional[int]
    ) -> List[VegasLine]:
        """Generate mock Vegas lines."""
        import random
        random.seed(season * 100 + (week or 0))
        
        # Get schedule first
        schedule_source = MockScheduleDataSource()
        schedule_result = schedule_source.fetch(season=season, week=week)
        schedule = schedule_result.data["schedule"]
        
        lines = []
        for game in schedule:
            # Base spread on team "quality" (mock)
            base_spread = random.uniform(-14, 14)
            # Add home field advantage
            spread = round(base_spread - 2.5, 1)
            # Normalize to 0.5 increments
            spread = round(spread * 2) / 2
            
            # Generate moneylines from spread
            if spread < 0:  # Home favored
                home_ml = int(-110 * (1 + abs(spread) / 7))
                away_ml = int(100 * (1 + abs(spread) / 10))
            else:  # Away favored
                home_ml = int(100 * (1 + abs(spread) / 10))
                away_ml = int(-110 * (1 + abs(spread) / 7))
            
            lines.append(VegasLine(
                game_id=game.game_id,
                spread=spread,
                total=round(random.uniform(38, 54), 1),
                home_ml=home_ml,
                away_ml=away_ml,
                book="MockBook"
            ))
        
        return lines


# =============================================================================
# Story 4.5: Data Pipeline Orchestrator
# =============================================================================

@dataclass
class PipelineStepResult:
    """Result of a single pipeline step."""
    step_name: str
    status: DataFetchStatus
    duration_seconds: float
    records_processed: int = 0
    error_message: Optional[str] = None
    data: Optional[Dict] = None


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    pipeline_name: str
    start_time: datetime
    end_time: datetime
    overall_status: DataFetchStatus
    steps: List[PipelineStepResult] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def is_success(self) -> bool:
        return self.overall_status == DataFetchStatus.SUCCESS
    
    def summary(self) -> str:
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"Status: {self.overall_status.value}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Steps:"
        ]
        for step in self.steps:
            status_icon = "✓" if step.status == DataFetchStatus.SUCCESS else "✗"
            lines.append(f"  {status_icon} {step.step_name}: {step.records_processed} records")
        return "\n".join(lines)


class DataPipeline:
    """
    Orchestrates data fetching and model updates.
    
    A pipeline defines a series of steps to:
    1. Fetch data from various sources
    2. Transform/normalize the data
    3. Update the prediction model
    4. Generate predictions
    
    Usage:
        pipeline = DataPipeline("Weekly Update")
        pipeline.add_step("fetch_epa", fetch_epa_func)
        pipeline.add_step("update_model", update_model_func)
        result = pipeline.run(season=2025, week=10)
    """
    
    def __init__(self, name: str):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name for logging
        """
        self.name = name
        self._steps: List[Tuple[str, callable]] = []
        self._context: Dict = {}
    
    def add_step(self, name: str, func: callable) -> 'DataPipeline':
        """
        Add a step to the pipeline.
        
        Args:
            name: Step name
            func: Callable that takes context dict and returns data
        
        Returns:
            Self for chaining
        """
        self._steps.append((name, func))
        return self
    
    def run(self, **initial_context) -> PipelineResult:
        """
        Run the pipeline.
        
        Args:
            **initial_context: Initial context values
        
        Returns:
            PipelineResult with status and data
        """
        start_time = datetime.now()
        self._context = initial_context.copy()
        step_results = []
        overall_status = DataFetchStatus.SUCCESS
        
        for step_name, step_func in self._steps:
            step_start = datetime.now()
            
            try:
                result = step_func(self._context)
                
                # Store result in context for next step
                self._context[f"{step_name}_result"] = result
                
                records = 0
                if isinstance(result, dict) and "records" in result:
                    records = result["records"]
                elif isinstance(result, DataFetchResult):
                    records = result.records_fetched
                
                step_results.append(PipelineStepResult(
                    step_name=step_name,
                    status=DataFetchStatus.SUCCESS,
                    duration_seconds=(datetime.now() - step_start).total_seconds(),
                    records_processed=records,
                    data=result if isinstance(result, dict) else None
                ))
                
            except Exception as e:
                step_results.append(PipelineStepResult(
                    step_name=step_name,
                    status=DataFetchStatus.FAILED,
                    duration_seconds=(datetime.now() - step_start).total_seconds(),
                    error_message=str(e)
                ))
                overall_status = DataFetchStatus.PARTIAL
        
        return PipelineResult(
            pipeline_name=self.name,
            start_time=start_time,
            end_time=datetime.now(),
            overall_status=overall_status,
            steps=step_results
        )
    
    def get_context(self) -> Dict:
        """Get current pipeline context."""
        return self._context.copy()


class WeeklyUpdatePipeline:
    """
    Pre-built pipeline for weekly model updates.
    
    Steps:
    1. Fetch latest EPA data
    2. Fetch week schedule
    3. Fetch Vegas lines
    4. Update model with new data
    5. Generate predictions
    
    Usage:
        pipeline = WeeklyUpdatePipeline(model, registry)
        result = pipeline.run(season=2025, week=10)
        predictions = result.get_predictions()
    """
    
    def __init__(
        self,
        model: 'NFLSeasonModel',
        registry: DataSourceRegistry
    ):
        """
        Initialize weekly update pipeline.
        
        Args:
            model: NFLSeasonModel to update
            registry: DataSourceRegistry with data sources
        """
        self.model = model
        self.registry = registry
        self._predictions: List[GamePrediction] = []
    
    def run(self, season: int, week: int) -> PipelineResult:
        """
        Run the weekly update pipeline.
        
        Args:
            season: NFL season year
            week: Current week number
        
        Returns:
            PipelineResult with status
        """
        pipeline = DataPipeline(f"Weekly Update - Week {week}")
        
        # Step 1: Fetch EPA data
        def fetch_epa(ctx):
            result = self.registry.fetch_best(
                DataSourceType.EPA,
                season=season,
                week=week
            )
            if result.is_success and result.data:
                ctx["epa_data"] = result.data.get("epa_data", [])
            return {"records": result.records_fetched}
        
        # Step 2: Fetch schedule
        def fetch_schedule(ctx):
            result = self.registry.fetch_best(
                DataSourceType.SCHEDULE,
                season=season,
                week=week
            )
            if result.is_success and result.data:
                ctx["schedule"] = result.data.get("schedule", [])
            return {"records": result.records_fetched}
        
        # Step 3: Fetch Vegas lines
        def fetch_vegas(ctx):
            result = self.registry.fetch_best(
                DataSourceType.VEGAS_LINES,
                season=season,
                week=week
            )
            if result.is_success and result.data:
                ctx["vegas_lines"] = result.data.get("lines", [])
            return {"records": result.records_fetched}
        
        # Step 4: Update model
        def update_model(ctx):
            epa_data = ctx.get("epa_data", [])
            updated = 0
            
            if epa_data:
                # Convert EPA data to TeamInSeasonData
                in_season_data = {}
                for epa_point in epa_data:
                    if isinstance(epa_point, EPADataPoint):
                        in_season_data[epa_point.team] = epa_point.to_in_season_data()
                        updated += 1
                
                if in_season_data:
                    self.model.update_inseason_data(week, in_season_data)
            
            return {"records": updated}
        
        # Step 5: Generate predictions
        def generate_predictions(ctx):
            schedule = ctx.get("schedule", [])
            vegas_lines = {l.game_id: l for l in ctx.get("vegas_lines", [])}
            
            self._predictions = []
            predictor = SpreadPredictor(self.model)
            
            for entry in schedule:
                if isinstance(entry, ScheduleEntry):
                    game = entry.to_nfl_game()
                    
                    # Add Vegas line if available
                    if game.game_id in vegas_lines:
                        line = vegas_lines[game.game_id]
                        game.vegas_spread = line.spread
                        game.vegas_total = line.total
                    
                    pred = predictor.predict_game(game, week)
                    self._predictions.append(pred)
            
            return {"records": len(self._predictions)}
        
        # Build and run pipeline
        pipeline.add_step("fetch_epa", fetch_epa)
        pipeline.add_step("fetch_schedule", fetch_schedule)
        pipeline.add_step("fetch_vegas", fetch_vegas)
        pipeline.add_step("update_model", update_model)
        pipeline.add_step("generate_predictions", generate_predictions)
        
        return pipeline.run(season=season, week=week)
    
    def get_predictions(self) -> List[GamePrediction]:
        """Get predictions from the last run."""
        return self._predictions.copy()


# =============================================================================
# Story 4.6: Weekly Automation Manager
# =============================================================================

@dataclass
class AutomationSchedule:
    """
    Schedule for automated tasks.
    
    Attributes:
        name: Task name
        day_of_week: Day to run (0=Mon, 6=Sun)
        hour: Hour to run (0-23)
        minute: Minute to run (0-59)
        enabled: Whether task is enabled
    """
    name: str
    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int
    minute: int = 0
    enabled: bool = True
    
    def should_run(self, now: datetime) -> bool:
        """Check if task should run at the given time."""
        if not self.enabled:
            return False
        return (now.weekday() == self.day_of_week and
                now.hour == self.hour and
                now.minute == self.minute)
    
    @property
    def day_name(self) -> str:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                "Friday", "Saturday", "Sunday"]
        return days[self.day_of_week]


class WeeklyAutomationManager:
    """
    Manages automated weekly prediction tasks.
    
    Handles:
    - Scheduling data updates
    - Running prediction pipelines
    - Sending notifications
    - Logging results
    
    Usage:
        manager = WeeklyAutomationManager(model)
        manager.register_data_source(epa_source)
        manager.schedule_weekly_update(day=1, hour=6)  # Tuesday 6am
        manager.run_scheduled_tasks()
    """
    
    def __init__(self, model: 'NFLSeasonModel'):
        """
        Initialize automation manager.
        
        Args:
            model: NFLSeasonModel to manage
        """
        self.model = model
        self.registry = DataSourceRegistry()
        self._schedules: List[AutomationSchedule] = []
        self._run_history: List[Dict] = []
        self._notification_handlers: List[callable] = []
    
    def register_data_source(self, source: DataSource) -> None:
        """Register a data source."""
        self.registry.register(source)
    
    def schedule_weekly_update(
        self,
        day: int = 1,  # Tuesday
        hour: int = 6,
        minute: int = 0
    ) -> None:
        """
        Schedule the weekly update task.
        
        Args:
            day: Day of week (0=Mon, 6=Sun)
            hour: Hour (0-23)
            minute: Minute (0-59)
        """
        self._schedules.append(AutomationSchedule(
            name="weekly_update",
            day_of_week=day,
            hour=hour,
            minute=minute
        ))
    
    def schedule_prediction_refresh(
        self,
        day: int = 3,  # Thursday
        hour: int = 18,
        minute: int = 0
    ) -> None:
        """Schedule prediction refresh before Thursday games."""
        self._schedules.append(AutomationSchedule(
            name="prediction_refresh",
            day_of_week=day,
            hour=hour,
            minute=minute
        ))
    
    def add_notification_handler(self, handler: callable) -> None:
        """
        Add a notification handler.
        
        Handler receives: (event_type: str, data: dict)
        """
        self._notification_handlers.append(handler)
    
    def run_weekly_update(self, season: int, week: int) -> PipelineResult:
        """
        Run the weekly update pipeline.
        
        Args:
            season: NFL season year
            week: Current week number
        
        Returns:
            PipelineResult with status
        """
        pipeline = WeeklyUpdatePipeline(self.model, self.registry)
        result = pipeline.run(season, week)
        
        # Log result
        self._run_history.append({
            "task": "weekly_update",
            "season": season,
            "week": week,
            "timestamp": datetime.now(),
            "status": result.overall_status.value,
            "duration": result.duration_seconds
        })
        
        # Send notifications
        self._notify("weekly_update_complete", {
            "season": season,
            "week": week,
            "status": result.overall_status.value,
            "predictions_count": len(pipeline.get_predictions())
        })
        
        return result
    
    def run_scheduled_tasks(self, now: Optional[datetime] = None) -> List[str]:
        """
        Run any scheduled tasks that are due.
        
        Args:
            now: Current datetime (uses actual now if None)
        
        Returns:
            List of task names that were run
        """
        now = now or datetime.now()
        tasks_run = []
        
        for schedule in self._schedules:
            if schedule.should_run(now):
                # Determine current season/week (simplified)
                season = now.year if now.month >= 9 else now.year - 1
                week = self._estimate_current_week(now)
                
                if schedule.name == "weekly_update":
                    self.run_weekly_update(season, week)
                    tasks_run.append(schedule.name)
                
                elif schedule.name == "prediction_refresh":
                    # Just regenerate predictions
                    pipeline = WeeklyUpdatePipeline(self.model, self.registry)
                    pipeline.run(season, week)
                    tasks_run.append(schedule.name)
        
        return tasks_run
    
    def get_run_history(self, limit: int = 10) -> List[Dict]:
        """Get recent run history."""
        return self._run_history[-limit:]
    
    def get_schedules(self) -> List[Dict]:
        """Get all scheduled tasks."""
        return [
            {
                "name": s.name,
                "day": s.day_name,
                "time": f"{s.hour:02d}:{s.minute:02d}",
                "enabled": s.enabled
            }
            for s in self._schedules
        ]
    
    def _estimate_current_week(self, now: datetime) -> int:
        """Estimate current NFL week from date."""
        # NFL season typically starts first Thursday after Labor Day
        season_start = date(now.year, 9, 1)
        # Find first Thursday
        while season_start.weekday() != 3:  # Thursday
            season_start += timedelta(days=1)
        
        # Calculate weeks since start
        days_since = (now.date() - season_start).days
        if days_since < 0:
            return 0  # Preseason
        
        return min(18, (days_since // 7) + 1)
    
    def _notify(self, event_type: str, data: Dict) -> None:
        """Send notification to all handlers."""
        for handler in self._notification_handlers:
            try:
                handler(event_type, data)
            except Exception:
                pass  # Don't let notification failures break the pipeline


# =============================================================================
# =============================================================================
# EPIC 5: OUTPUT & EXPORT
# =============================================================================
# =============================================================================


# =============================================================================
# Story 5.1: Export Format Abstraction
# =============================================================================

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


@dataclass
class ExportConfig:
    """
    Configuration for exports.
    
    Attributes:
        format: Output format
        include_metadata: Include timestamps, version info
        include_ratings: Include team power ratings
        include_vegas_comparison: Include Vegas line comparisons
        include_confidence: Include confidence scores
        include_analysis: Include detailed analysis text
        sort_by: Field to sort predictions by
        filter_min_edge: Minimum edge to include (for best bets)
        output_path: File path for output (None for string return)
        pretty_print: Pretty-print JSON output
    """
    format: ExportFormat = ExportFormat.JSON
    include_metadata: bool = True
    include_ratings: bool = True
    include_vegas_comparison: bool = True
    include_confidence: bool = True
    include_analysis: bool = False
    sort_by: str = "game_date"
    filter_min_edge: Optional[float] = None
    output_path: Optional[str] = None
    pretty_print: bool = True


class Exporter(ABC):
    """
    Abstract base class for all exporters.
    
    Exporters convert prediction data to various output formats.
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
    
    @abstractmethod
    def export_predictions(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int
    ) -> str:
        """
        Export predictions to string.
        
        Args:
            predictions: List of GamePrediction objects
            week: Week number
            season: NFL season year
        
        Returns:
            Formatted string in the target format
        """
        pass
    
    @abstractmethod
    def export_rankings(
        self,
        rankings: List[Dict],
        week: int,
        season: int
    ) -> str:
        """
        Export power rankings to string.
        
        Args:
            rankings: List of ranking dictionaries
            week: Week number
            season: NFL season year
        
        Returns:
            Formatted string in the target format
        """
        pass
    
    @abstractmethod
    def export_backtest_results(
        self,
        results: Dict
    ) -> str:
        """
        Export backtest results to string.
        
        Args:
            results: Backtest results dictionary
        
        Returns:
            Formatted string in the target format
        """
        pass
    
    def save_to_file(self, content: str, filepath: str) -> bool:
        """
        Save content to file.
        
        Args:
            content: String content to save
            filepath: Output file path
        
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def _filter_predictions(
        self,
        predictions: List[GamePrediction]
    ) -> List[GamePrediction]:
        """Apply configured filters to predictions."""
        filtered = predictions.copy()
        
        # Filter by minimum edge
        if self.config.filter_min_edge is not None:
            filtered = [
                p for p in filtered
                if p.edge is not None and abs(p.edge) >= self.config.filter_min_edge
            ]
        
        # Sort
        if self.config.sort_by == "game_date":
            filtered.sort(key=lambda p: p.game.game_date)
        elif self.config.sort_by == "spread":
            filtered.sort(key=lambda p: abs(p.predicted_spread))
        elif self.config.sort_by == "edge":
            filtered.sort(key=lambda p: abs(p.edge or 0), reverse=True)
        elif self.config.sort_by == "confidence":
            filtered.sort(key=lambda p: p.confidence_score, reverse=True)
        
        return filtered
    
    def _get_metadata(self, week: int, season: int) -> Dict:
        """Generate metadata for export."""
        return {
            "generated_at": datetime.now().isoformat(),
            "season": season,
            "week": week,
            "model_version": "5.0.0",
            "export_format": self.config.format.value
        }


# =============================================================================
# Story 5.2: JSON Exporter
# =============================================================================

class JSONExporter(Exporter):
    """
    Exports data to JSON format.
    
    Produces structured JSON suitable for:
    - API responses
    - Data storage
    - Frontend consumption
    - Integration with other tools
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize JSON exporter."""
        config = config or ExportConfig(format=ExportFormat.JSON)
        super().__init__(config)
    
    def export_predictions(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int
    ) -> str:
        """Export predictions to JSON."""
        import json
        
        filtered = self._filter_predictions(predictions)
        
        output = {
            "predictions": []
        }
        
        if self.config.include_metadata:
            output["metadata"] = self._get_metadata(week, season)
        
        for pred in filtered:
            game_data = {
                "game_id": pred.game.game_id,
                "away_team": pred.game.away_team,
                "home_team": pred.game.home_team,
                "game_date": pred.game.game_date.isoformat() if pred.game.game_date else None,
                "predicted_spread": pred.predicted_spread,
                "formatted_spread": pred.formatted_spread,
                "pick": pred.pick,
                "home_win_probability": pred.home_win_probability,
                "away_win_probability": pred.away_win_probability
            }
            
            if self.config.include_ratings:
                game_data["home_power_rating"] = pred.home_power_rating
                game_data["away_power_rating"] = pred.away_power_rating
            
            if self.config.include_vegas_comparison and pred.game.vegas_spread is not None:
                game_data["vegas_spread"] = pred.game.vegas_spread
                game_data["spread_vs_vegas"] = pred.spread_vs_vegas
                game_data["edge"] = pred.edge
                game_data["pick_ats"] = pred.pick_ats
            
            if self.config.include_confidence:
                game_data["confidence_tier"] = pred.confidence_tier
                game_data["confidence_score"] = pred.confidence_score
            
            output["predictions"].append(game_data)
        
        if self.config.pretty_print:
            return json.dumps(output, indent=2, default=str)
        return json.dumps(output, default=str)
    
    def export_rankings(
        self,
        rankings: List[Dict],
        week: int,
        season: int
    ) -> str:
        """Export power rankings to JSON."""
        import json
        
        output = {
            "rankings": rankings
        }
        
        if self.config.include_metadata:
            output["metadata"] = self._get_metadata(week, season)
        
        if self.config.pretty_print:
            return json.dumps(output, indent=2, default=str)
        return json.dumps(output, default=str)
    
    def export_backtest_results(self, results: Dict) -> str:
        """Export backtest results to JSON."""
        import json
        
        if self.config.pretty_print:
            return json.dumps(results, indent=2, default=str)
        return json.dumps(results, default=str)
    
    def export_full_week(
        self,
        predictions: List[GamePrediction],
        rankings: List[Dict],
        week: int,
        season: int
    ) -> str:
        """Export complete weekly data package."""
        import json
        
        filtered = self._filter_predictions(predictions)
        
        output = {
            "predictions": [self._prediction_to_dict(p) for p in filtered],
            "rankings": rankings
        }
        
        if self.config.include_metadata:
            output["metadata"] = self._get_metadata(week, season)
        
        if self.config.pretty_print:
            return json.dumps(output, indent=2, default=str)
        return json.dumps(output, default=str)
    
    def _prediction_to_dict(self, pred: GamePrediction) -> Dict:
        """Convert prediction to dictionary."""
        return {
            "game_id": pred.game.game_id,
            "away_team": pred.game.away_team,
            "home_team": pred.game.home_team,
            "game_date": pred.game.game_date.isoformat() if pred.game.game_date else None,
            "predicted_spread": pred.predicted_spread,
            "formatted_spread": pred.formatted_spread,
            "pick": pred.pick,
            "home_win_probability": pred.home_win_probability,
            "away_win_probability": pred.away_win_probability,
            "home_power_rating": pred.home_power_rating,
            "away_power_rating": pred.away_power_rating,
            "vegas_spread": pred.game.vegas_spread,
            "edge": pred.edge,
            "confidence_tier": pred.confidence_tier,
            "confidence_score": pred.confidence_score
        }


# =============================================================================
# Story 5.3: CSV Exporter
# =============================================================================

class CSVExporter(Exporter):
    """
    Exports data to CSV format.
    
    Produces tabular data suitable for:
    - Spreadsheet analysis
    - Data science tools (pandas, R)
    - Database import
    - Simple sharing
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize CSV exporter."""
        config = config or ExportConfig(format=ExportFormat.CSV)
        super().__init__(config)
    
    def export_predictions(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int
    ) -> str:
        """Export predictions to CSV."""
        filtered = self._filter_predictions(predictions)
        
        # Build header
        headers = [
            "game_id", "season", "week", "game_date",
            "away_team", "home_team",
            "predicted_spread", "pick",
            "home_win_prob", "away_win_prob"
        ]
        
        if self.config.include_ratings:
            headers.extend(["home_rating", "away_rating"])
        
        if self.config.include_vegas_comparison:
            headers.extend(["vegas_spread", "edge", "pick_ats"])
        
        if self.config.include_confidence:
            headers.extend(["confidence_tier", "confidence_score"])
        
        lines = [",".join(headers)]
        
        # Build rows
        for pred in filtered:
            row = [
                pred.game.game_id,
                str(season),
                str(week),
                pred.game.game_date.isoformat() if pred.game.game_date else "",
                pred.game.away_team,
                pred.game.home_team,
                f"{pred.predicted_spread:.1f}",
                pred.pick,
                f"{pred.home_win_probability:.4f}",
                f"{pred.away_win_probability:.4f}"
            ]
            
            if self.config.include_ratings:
                row.extend([
                    f"{pred.home_power_rating:.2f}",
                    f"{pred.away_power_rating:.2f}"
                ])
            
            if self.config.include_vegas_comparison:
                row.extend([
                    f"{pred.game.vegas_spread:.1f}" if pred.game.vegas_spread else "",
                    f"{pred.edge:.1f}" if pred.edge else "",
                    pred.pick_ats or ""
                ])
            
            if self.config.include_confidence:
                row.extend([
                    pred.confidence_tier,
                    str(pred.confidence_score)
                ])
            
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    def export_rankings(
        self,
        rankings: List[Dict],
        week: int,
        season: int
    ) -> str:
        """Export power rankings to CSV."""
        headers = ["rank", "team", "power_rating", "conference", "division"]
        lines = [",".join(headers)]
        
        for r in rankings:
            row = [
                str(r.get("rank", "")),
                r.get("team", ""),
                f"{r.get('power_rating', 0):.2f}",
                r.get("conference", ""),
                r.get("division", "")
            ]
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    def export_backtest_results(self, results: Dict) -> str:
        """Export backtest weekly results to CSV."""
        weekly = results.get("weekly_results", [])
        
        if not weekly:
            return "week,games,correct,accuracy\n"
        
        headers = ["week", "games", "correct", "accuracy"]
        lines = [",".join(headers)]
        
        for w in weekly:
            row = [
                str(w.get("week", "")),
                str(w.get("games", "")),
                str(w.get("correct", "")),
                f"{w.get('accuracy', 0):.3f}"
            ]
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    def export_prediction_results(
        self,
        results: List[PredictionResult]
    ) -> str:
        """Export prediction results for analysis."""
        headers = [
            "game_id", "week", "home_team", "away_team",
            "predicted_winner", "actual_winner", "su_correct",
            "predicted_spread", "actual_spread", "spread_error",
            "vegas_spread", "ats_pick", "ats_correct",
            "predicted_win_prob", "confidence_tier"
        ]
        
        lines = [",".join(headers)]
        
        for r in results:
            row = [
                r.game_id,
                str(r.week),
                r.home_team,
                r.away_team,
                r.predicted_winner,
                r.actual_winner,
                str(r.su_correct),
                f"{r.prediction.predicted_spread:.1f}",
                f"{r.actual_spread:.1f}",
                f"{r.spread_error:.1f}",
                f"{r.vegas_spread:.1f}" if r.vegas_spread else "",
                r.ats_pick or "",
                str(r.ats_correct) if r.ats_correct is not None else "",
                f"{r.predicted_win_prob:.4f}",
                r.confidence_tier
            ]
            lines.append(",".join(row))
        
        return "\n".join(lines)


# =============================================================================
# Story 5.4: Markdown Report Generator
# =============================================================================

class MarkdownExporter(Exporter):
    """
    Exports data to Markdown format.
    
    Produces formatted Markdown suitable for:
    - GitHub/GitLab documentation
    - Notion, Obsidian notes
    - Blog posts
    - Email (rendered)
    - Static site generators
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize Markdown exporter."""
        config = config or ExportConfig(format=ExportFormat.MARKDOWN)
        super().__init__(config)
    
    def export_predictions(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int
    ) -> str:
        """Export predictions to Markdown."""
        filtered = self._filter_predictions(predictions)
        
        lines = []
        
        # Header
        lines.append(f"# NFL Week {week} Predictions")
        lines.append(f"**{season} Season**")
        lines.append("")
        
        if self.config.include_metadata:
            lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
            lines.append("")
        
        # Summary
        lines.append("## Quick Picks")
        lines.append("")
        lines.append("| Game | Pick | Spread | Win Prob |")
        lines.append("|------|------|--------|----------|")
        
        for pred in filtered:
            game = f"{pred.game.away_team} @ {pred.game.home_team}"
            spread = pred.formatted_spread
            prob = f"{max(pred.home_win_probability, pred.away_win_probability):.0%}"
            lines.append(f"| {game} | **{pred.pick}** | {spread} | {prob} |")
        
        lines.append("")
        
        # Detailed predictions
        if self.config.include_analysis:
            lines.append("## Detailed Analysis")
            lines.append("")
            
            for pred in filtered:
                lines.append(f"### {pred.game.away_team} @ {pred.game.home_team}")
                lines.append("")
                lines.append(f"**Pick: {pred.pick}** ({pred.formatted_spread})")
                lines.append("")
                lines.append(f"- Home Win Probability: {pred.home_win_probability:.1%}")
                lines.append(f"- Away Win Probability: {pred.away_win_probability:.1%}")
                
                if self.config.include_ratings:
                    lines.append(f"- {pred.game.home_team} Power Rating: {pred.home_power_rating:+.2f}")
                    lines.append(f"- {pred.game.away_team} Power Rating: {pred.away_power_rating:+.2f}")
                
                if self.config.include_vegas_comparison and pred.game.vegas_spread is not None:
                    lines.append(f"- Vegas Spread: {pred.game.vegas_spread:+.1f}")
                    if pred.edge:
                        edge_team = pred.game.home_team if pred.edge > 0 else pred.game.away_team
                        lines.append(f"- **Edge: {abs(pred.edge):.1f} pts on {edge_team}**")
                
                if self.config.include_confidence:
                    lines.append(f"- Confidence: {pred.confidence_tier} ({pred.confidence_score}/100)")
                
                lines.append("")
        
        # Best bets section
        if self.config.include_vegas_comparison:
            best_bets = [p for p in filtered if p.edge and abs(p.edge) >= 2.0]
            if best_bets:
                lines.append("## Best Bets (2+ pt edge)")
                lines.append("")
                for pred in sorted(best_bets, key=lambda x: abs(x.edge or 0), reverse=True):
                    edge_team = pred.game.home_team if pred.edge > 0 else pred.game.away_team
                    lines.append(f"- **{edge_team}** ({abs(pred.edge):.1f} pt edge) vs Vegas")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_rankings(
        self,
        rankings: List[Dict],
        week: int,
        season: int
    ) -> str:
        """Export power rankings to Markdown."""
        lines = []
        
        lines.append(f"# NFL Power Rankings - Week {week}")
        lines.append(f"**{season} Season**")
        lines.append("")
        
        if self.config.include_metadata:
            lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
            lines.append("")
        
        lines.append("| Rank | Team | Rating | Conference |")
        lines.append("|------|------|--------|------------|")
        
        for r in rankings:
            rank = r.get("rank", "")
            team = r.get("team", "")
            rating = r.get("power_rating", 0)
            conf = r.get("conference", "")
            lines.append(f"| {rank} | {team} | {rating:+.2f} | {conf} |")
        
        lines.append("")
        
        # Tier breakdown
        lines.append("## Tier Breakdown")
        lines.append("")
        
        tiers = [
            ("Elite (Top 5)", rankings[:5]),
            ("Contenders (6-12)", rankings[5:12]),
            ("Playoff Bubble (13-20)", rankings[12:20]),
            ("Rebuilding (21-28)", rankings[20:28]),
            ("Bottom Tier (29-32)", rankings[28:])
        ]
        
        for tier_name, tier_teams in tiers:
            if tier_teams:
                lines.append(f"### {tier_name}")
                for r in tier_teams:
                    lines.append(f"- {r.get('team')} ({r.get('power_rating', 0):+.2f})")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_backtest_results(self, results: Dict) -> str:
        """Export backtest results to Markdown."""
        lines = []
        
        summary = results.get("summary", {})
        
        lines.append("# Backtest Results")
        lines.append("")
        
        if self.config.include_metadata:
            lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
            lines.append("")
        
        # Summary stats
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Games:** {summary.get('total_games', 'N/A')}")
        lines.append(f"- **Straight-Up Record:** {summary.get('su_record', 'N/A')} ({summary.get('su_accuracy', 'N/A')})")
        lines.append(f"- **Against-the-Spread:** {summary.get('ats_record', 'N/A')} ({summary.get('ats_accuracy', 'N/A')})")
        lines.append(f"- **Mean Absolute Error:** {summary.get('mae', 'N/A')}")
        lines.append(f"- **RMSE:** {summary.get('rmse', 'N/A')}")
        lines.append(f"- **Brier Score:** {summary.get('brier_score', 'N/A')}")
        lines.append(f"- **Theoretical ROI:** {summary.get('theoretical_roi', 'N/A')}")
        lines.append("")
        
        # Weekly breakdown
        weekly = results.get("weekly_results", [])
        if weekly:
            lines.append("## Weekly Results")
            lines.append("")
            lines.append("| Week | Games | Correct | Accuracy |")
            lines.append("|------|-------|---------|----------|")
            
            for w in weekly:
                lines.append(
                    f"| {w.get('week', '')} | {w.get('games', '')} | "
                    f"{w.get('correct', '')} | {w.get('accuracy', 0):.1%} |"
                )
            lines.append("")
        
        # Calibration
        calibration = results.get("calibration", {})
        if calibration:
            lines.append("## Calibration Analysis")
            lines.append("")
            lines.append(f"- **Expected Calibration Error:** {calibration.get('expected_calibration_error', 'N/A')}")
            lines.append(f"- **Assessment:** {calibration.get('calibration_assessment', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_weekly_newsletter(
        self,
        predictions: List[GamePrediction],
        rankings: List[Dict],
        week: int,
        season: int,
        previous_results: Optional[Dict] = None
    ) -> str:
        """Generate a weekly newsletter-style report."""
        lines = []
        
        lines.append(f"# 🏈 NFL Week {week} Preview")
        lines.append(f"### {season} Season")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%B %d, %Y')}*")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Previous week results
        if previous_results:
            lines.append("## 📊 Last Week's Results")
            lines.append("")
            summary = previous_results.get("summary", {})
            lines.append(f"Record: **{summary.get('su_record', 'N/A')}** ({summary.get('su_accuracy', 'N/A')})")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Power rankings snapshot
        lines.append("## 📈 Power Rankings (Top 10)")
        lines.append("")
        for r in rankings[:10]:
            lines.append(f"{r.get('rank')}. **{r.get('team')}** ({r.get('power_rating', 0):+.2f})")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # This week's picks
        filtered = self._filter_predictions(predictions)
        
        lines.append("## 🎯 This Week's Picks")
        lines.append("")
        
        for pred in filtered:
            emoji = "🏠" if pred.pick == pred.game.home_team else "✈️"
            lines.append(f"### {pred.game.away_team} @ {pred.game.home_team}")
            lines.append("")
            lines.append(f"{emoji} **Pick: {pred.pick}** ({pred.formatted_spread})")
            lines.append("")
            lines.append(f"Win Probability: {max(pred.home_win_probability, pred.away_win_probability):.0%}")
            
            if pred.edge and abs(pred.edge) >= 2.0:
                lines.append(f"🔥 **{abs(pred.edge):.1f} pt edge vs Vegas!**")
            
            lines.append("")
        
        # Best bets summary
        best_bets = [p for p in filtered if p.edge and abs(p.edge) >= 2.0]
        if best_bets:
            lines.append("---")
            lines.append("")
            lines.append("## 💰 Best Bets")
            lines.append("")
            for pred in sorted(best_bets, key=lambda x: abs(x.edge or 0), reverse=True):
                edge_team = pred.game.home_team if pred.edge > 0 else pred.game.away_team
                lines.append(f"1. **{edge_team}** - {abs(pred.edge):.1f} point edge")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("*Good luck this week! 🍀*")
        
        return "\n".join(lines)


# =============================================================================
# Story 5.5: HTML Dashboard Generator
# =============================================================================

class HTMLExporter(Exporter):
    """
    Exports data to HTML format.
    
    Produces styled HTML suitable for:
    - Web pages
    - Email newsletters
    - PDF generation
    - Interactive dashboards
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize HTML exporter."""
        config = config or ExportConfig(format=ExportFormat.HTML)
        super().__init__(config)
    
    def export_predictions(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int
    ) -> str:
        """Export predictions to HTML."""
        filtered = self._filter_predictions(predictions)
        
        html = self._get_html_header(f"NFL Week {week} Predictions - {season}")
        
        html += f"""
        <div class="container">
            <h1>🏈 NFL Week {week} Predictions</h1>
            <p class="subtitle">{season} Season</p>
            <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <table class="predictions">
                <thead>
                    <tr>
                        <th>Game</th>
                        <th>Pick</th>
                        <th>Spread</th>
                        <th>Win Prob</th>
                        {'<th>Edge</th>' if self.config.include_vegas_comparison else ''}
                        {'<th>Confidence</th>' if self.config.include_confidence else ''}
                    </tr>
                </thead>
                <tbody>
        """
        
        for pred in filtered:
            game = f"{pred.game.away_team} @ {pred.game.home_team}"
            prob = max(pred.home_win_probability, pred.away_win_probability)
            prob_class = "high-prob" if prob >= 0.7 else "med-prob" if prob >= 0.55 else "low-prob"
            
            edge_cell = ""
            if self.config.include_vegas_comparison:
                if pred.edge and abs(pred.edge) >= 2.0:
                    edge_team = pred.game.home_team if pred.edge > 0 else pred.game.away_team
                    edge_cell = f'<td class="edge-highlight">{abs(pred.edge):.1f} pts ({edge_team})</td>'
                elif pred.edge:
                    edge_cell = f'<td>{abs(pred.edge):.1f} pts</td>'
                else:
                    edge_cell = '<td>-</td>'
            
            conf_cell = ""
            if self.config.include_confidence:
                conf_class = pred.confidence_tier.lower()
                conf_cell = f'<td class="conf-{conf_class}">{pred.confidence_tier}</td>'
            
            html += f"""
                    <tr>
                        <td>{game}</td>
                        <td class="pick"><strong>{pred.pick}</strong></td>
                        <td>{pred.formatted_spread}</td>
                        <td class="{prob_class}">{prob:.0%}</td>
                        {edge_cell}
                        {conf_cell}
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        html += self._get_html_footer()
        return html
    
    def export_rankings(
        self,
        rankings: List[Dict],
        week: int,
        season: int
    ) -> str:
        """Export power rankings to HTML."""
        html = self._get_html_header(f"NFL Power Rankings - Week {week}")
        
        html += f"""
        <div class="container">
            <h1>📊 NFL Power Rankings</h1>
            <p class="subtitle">Week {week} - {season} Season</p>
            
            <table class="rankings">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Team</th>
                        <th>Rating</th>
                        <th>Tier</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for r in rankings:
            rank = r.get("rank", 0)
            if rank <= 5:
                tier = "Elite"
                tier_class = "elite"
            elif rank <= 12:
                tier = "Contender"
                tier_class = "contender"
            elif rank <= 20:
                tier = "Bubble"
                tier_class = "bubble"
            else:
                tier = "Rebuilding"
                tier_class = "rebuilding"
            
            rating = r.get("power_rating", 0)
            rating_class = "positive" if rating > 0 else "negative"
            
            html += f"""
                    <tr>
                        <td class="rank">{rank}</td>
                        <td class="team">{r.get('team', '')}</td>
                        <td class="rating {rating_class}">{rating:+.2f}</td>
                        <td class="tier {tier_class}">{tier}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        html += self._get_html_footer()
        return html
    
    def export_backtest_results(self, results: Dict) -> str:
        """Export backtest results to HTML."""
        summary = results.get("summary", {})
        
        html = self._get_html_header("Backtest Results")
        
        html += f"""
        <div class="container">
            <h1>📈 Model Backtest Results</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_games', 'N/A')}</div>
                    <div class="stat-label">Total Games</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('su_accuracy', 'N/A')}</div>
                    <div class="stat-label">SU Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('ats_accuracy', 'N/A')}</div>
                    <div class="stat-label">ATS Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('mae', 'N/A')}</div>
                    <div class="stat-label">Mean Abs Error</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('theoretical_roi', 'N/A')}</div>
                    <div class="stat-label">Theoretical ROI</div>
                </div>
            </div>
        </div>
        """
        
        html += self._get_html_footer()
        return html
    
    def export_dashboard(
        self,
        predictions: List[GamePrediction],
        rankings: List[Dict],
        week: int,
        season: int,
        backtest_results: Optional[Dict] = None
    ) -> str:
        """Generate a complete HTML dashboard."""
        html = self._get_html_header(f"NFL Predictions Dashboard - Week {week}")
        
        filtered = self._filter_predictions(predictions)
        
        html += f"""
        <div class="dashboard">
            <header class="dashboard-header">
                <h1>🏈 NFL Predictions Dashboard</h1>
                <p>Week {week} - {season} Season</p>
            </header>
            
            <div class="dashboard-grid">
        """
        
        # Best bets panel
        best_bets = [p for p in filtered if p.edge and abs(p.edge) >= 2.0]
        html += """
                <div class="panel best-bets">
                    <h2>💰 Best Bets</h2>
        """
        if best_bets:
            for pred in sorted(best_bets, key=lambda x: abs(x.edge or 0), reverse=True)[:5]:
                edge_team = pred.game.home_team if pred.edge > 0 else pred.game.away_team
                html += f"""
                    <div class="bet-card">
                        <strong>{edge_team}</strong>
                        <span class="edge">{abs(pred.edge):.1f} pt edge</span>
                    </div>
                """
        else:
            html += "<p>No high-edge bets this week</p>"
        html += "</div>"
        
        # Top ranked teams panel
        html += """
                <div class="panel top-teams">
                    <h2>📊 Top 5 Teams</h2>
        """
        for r in rankings[:5]:
            html += f"""
                    <div class="team-row">
                        <span class="rank">#{r.get('rank')}</span>
                        <span class="team-name">{r.get('team')}</span>
                        <span class="rating">{r.get('power_rating', 0):+.2f}</span>
                    </div>
            """
        html += "</div>"
        
        # Model performance panel
        if backtest_results:
            summary = backtest_results.get("summary", {})
            html += f"""
                <div class="panel performance">
                    <h2>📈 Model Performance</h2>
                    <div class="perf-stat">
                        <label>SU Record:</label>
                        <value>{summary.get('su_record', 'N/A')}</value>
                    </div>
                    <div class="perf-stat">
                        <label>Accuracy:</label>
                        <value>{summary.get('su_accuracy', 'N/A')}</value>
                    </div>
                    <div class="perf-stat">
                        <label>ROI:</label>
                        <value>{summary.get('theoretical_roi', 'N/A')}</value>
                    </div>
                </div>
            """
        
        html += """
            </div>
            
            <div class="predictions-section">
                <h2>🎯 All Predictions</h2>
                <table class="predictions-table">
                    <thead>
                        <tr>
                            <th>Game</th>
                            <th>Pick</th>
                            <th>Spread</th>
                            <th>Probability</th>
                            <th>Edge</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for pred in filtered:
            edge_display = f"{abs(pred.edge):.1f}" if pred.edge else "-"
            html += f"""
                        <tr>
                            <td>{pred.game.away_team} @ {pred.game.home_team}</td>
                            <td><strong>{pred.pick}</strong></td>
                            <td>{pred.formatted_spread}</td>
                            <td>{max(pred.home_win_probability, pred.away_win_probability):.0%}</td>
                            <td>{edge_display}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        html += self._get_html_footer()
        return html
    
    def _get_html_header(self, title: str) -> str:
        """Get HTML document header with CSS."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2b6cb0;
            --success: #38a169;
            --warning: #d69e2e;
            --danger: #e53e3e;
            --light: #f7fafc;
            --dark: #1a202c;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--light);
            color: var(--dark);
            line-height: 1.6;
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        
        .subtitle {{ color: #718096; font-size: 1.1rem; }}
        
        .meta {{ color: #a0aec0; font-size: 0.9rem; margin-bottom: 2rem; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        th, td {{ padding: 1rem; text-align: left; }}
        
        th {{
            background: var(--primary);
            color: white;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{ background: #f8fafc; }}
        
        tr:hover {{ background: #edf2f7; }}
        
        .pick {{ font-weight: bold; color: var(--primary); }}
        
        .high-prob {{ color: var(--success); font-weight: bold; }}
        .med-prob {{ color: var(--warning); }}
        .low-prob {{ color: var(--danger); }}
        
        .edge-highlight {{
            background: #fef3c7;
            color: #92400e;
            font-weight: bold;
        }}
        
        .conf-high {{ color: var(--success); }}
        .conf-medium {{ color: var(--warning); }}
        .conf-low {{ color: var(--danger); }}
        
        .positive {{ color: var(--success); }}
        .negative {{ color: var(--danger); }}
        
        .elite {{ color: #7c3aed; font-weight: bold; }}
        .contender {{ color: var(--secondary); }}
        .bubble {{ color: var(--warning); }}
        .rebuilding {{ color: #a0aec0; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }}
        
        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        
        .stat-label {{ color: #718096; margin-top: 0.5rem; }}
        
        .dashboard {{ padding: 2rem; }}
        
        .dashboard-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .panel {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .panel h2 {{ margin-bottom: 1rem; font-size: 1.2rem; }}
        
        .bet-card {{
            display: flex;
            justify-content: space-between;
            padding: 0.75rem;
            background: #fef3c7;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }}
        
        .edge {{ color: #92400e; font-weight: bold; }}
        
        .team-row {{
            display: flex;
            gap: 1rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid #edf2f7;
        }}
        
        .rank {{ font-weight: bold; color: var(--primary); width: 2rem; }}
        .team-name {{ flex: 1; }}
        .rating {{ font-weight: bold; }}
        
        .predictions-section {{ margin-top: 2rem; }}
        
        .predictions-table {{
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
"""
    
    def _get_html_footer(self) -> str:
        """Get HTML document footer."""
        return """
</body>
</html>
"""


# =============================================================================
# Story 5.6: Multi-Format Export Manager
# =============================================================================

class ExportManager:
    """
    Manages exports across multiple formats.
    
    Provides a unified interface for:
    - Exporting to any supported format
    - Batch exports to multiple formats
    - Saving to files
    - Format detection and validation
    
    Usage:
        manager = ExportManager()
        
        # Single format export
        json_output = manager.export_predictions(predictions, week=10, format="json")
        
        # Multi-format export
        outputs = manager.export_all_formats(predictions, week=10)
        
        # Save to files
        manager.save_weekly_exports(predictions, rankings, week=10, output_dir="./exports")
    """
    
    def __init__(self):
        """Initialize export manager with all exporters."""
        self._exporters: Dict[ExportFormat, Exporter] = {
            ExportFormat.JSON: JSONExporter(),
            ExportFormat.CSV: CSVExporter(),
            ExportFormat.MARKDOWN: MarkdownExporter(),
            ExportFormat.HTML: HTMLExporter(),
        }
    
    def get_exporter(self, format: Union[str, ExportFormat]) -> Exporter:
        """
        Get exporter for a specific format.
        
        Args:
            format: Format name or ExportFormat enum
        
        Returns:
            Exporter instance
        """
        if isinstance(format, str):
            format = ExportFormat(format.lower())
        
        if format not in self._exporters:
            raise ValueError(f"Unsupported format: {format}")
        
        return self._exporters[format]
    
    def set_config(self, format: Union[str, ExportFormat], config: ExportConfig) -> None:
        """Set configuration for a specific exporter."""
        exporter = self.get_exporter(format)
        exporter.config = config
    
    def export_predictions(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int = 2025,
        format: Union[str, ExportFormat] = ExportFormat.JSON
    ) -> str:
        """Export predictions in specified format."""
        exporter = self.get_exporter(format)
        return exporter.export_predictions(predictions, week, season)
    
    def export_rankings(
        self,
        rankings: List[Dict],
        week: int,
        season: int = 2025,
        format: Union[str, ExportFormat] = ExportFormat.JSON
    ) -> str:
        """Export rankings in specified format."""
        exporter = self.get_exporter(format)
        return exporter.export_rankings(rankings, week, season)
    
    def export_backtest(
        self,
        results: Dict,
        format: Union[str, ExportFormat] = ExportFormat.JSON
    ) -> str:
        """Export backtest results in specified format."""
        exporter = self.get_exporter(format)
        return exporter.export_backtest_results(results)
    
    def export_all_formats(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int = 2025
    ) -> Dict[str, str]:
        """
        Export predictions to all supported formats.
        
        Returns:
            Dict mapping format name to exported content
        """
        outputs = {}
        for format in self._exporters.keys():
            try:
                outputs[format.value] = self.export_predictions(
                    predictions, week, season, format
                )
            except Exception as e:
                outputs[format.value] = f"Error: {str(e)}"
        return outputs
    
    def save_weekly_exports(
        self,
        predictions: List[GamePrediction],
        rankings: List[Dict],
        week: int,
        season: int = 2025,
        output_dir: str = ".",
        formats: Optional[List[ExportFormat]] = None
    ) -> Dict[str, bool]:
        """
        Save weekly exports to files.
        
        Args:
            predictions: List of predictions
            rankings: Power rankings
            week: Week number
            season: Season year
            output_dir: Output directory
            formats: List of formats to export (all if None)
        
        Returns:
            Dict mapping filename to success status
        """
        formats = formats or list(self._exporters.keys())
        results = {}
        
        for format in formats:
            exporter = self.get_exporter(format)
            
            # Export predictions
            pred_content = exporter.export_predictions(predictions, week, season)
            pred_filename = f"{output_dir}/week{week}_predictions.{format.value}"
            results[pred_filename] = exporter.save_to_file(pred_content, pred_filename)
            
            # Export rankings
            rank_content = exporter.export_rankings(rankings, week, season)
            rank_filename = f"{output_dir}/week{week}_rankings.{format.value}"
            results[rank_filename] = exporter.save_to_file(rank_content, rank_filename)
        
        return results
    
    def generate_weekly_package(
        self,
        predictions: List[GamePrediction],
        rankings: List[Dict],
        week: int,
        season: int = 2025,
        backtest_results: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Generate a complete weekly content package.
        
        Returns all export formats plus special reports.
        """
        package = {}
        
        # Standard exports
        for format in self._exporters.keys():
            package[f"predictions_{format.value}"] = self.export_predictions(
                predictions, week, season, format
            )
            package[f"rankings_{format.value}"] = self.export_rankings(
                rankings, week, season, format
            )
        
        # Special reports
        md_exporter = self.get_exporter(ExportFormat.MARKDOWN)
        if isinstance(md_exporter, MarkdownExporter):
            package["newsletter"] = md_exporter.export_weekly_newsletter(
                predictions, rankings, week, season
            )
        
        html_exporter = self.get_exporter(ExportFormat.HTML)
        if isinstance(html_exporter, HTMLExporter):
            package["dashboard"] = html_exporter.export_dashboard(
                predictions, rankings, week, season, backtest_results
            )
        
        # JSON full package
        json_exporter = self.get_exporter(ExportFormat.JSON)
        if isinstance(json_exporter, JSONExporter):
            package["full_package"] = json_exporter.export_full_week(
                predictions, rankings, week, season
            )
        
        return package
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported format names."""
        return [f.value for f in ExportFormat]
    
    @staticmethod
    def get_file_extension(format: Union[str, ExportFormat]) -> str:
        """Get file extension for a format."""
        if isinstance(format, str):
            format = ExportFormat(format.lower())
        
        extensions = {
            ExportFormat.JSON: ".json",
            ExportFormat.CSV: ".csv",
            ExportFormat.MARKDOWN: ".md",
            ExportFormat.HTML: ".html",
            ExportFormat.TEXT: ".txt",
        }
        return extensions.get(format, ".txt")


# =============================================================================
# =============================================================================
# EPIC 6: INJURY IMPACT MODELING
# =============================================================================
# =============================================================================


# =============================================================================
# Story 6.1: Player Value Model
# =============================================================================

class Position(Enum):
    """NFL player positions."""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    OL = "OL"      # Offensive Line (generic)
    LT = "LT"      # Left Tackle
    LG = "LG"      # Left Guard
    C = "C"        # Center
    RG = "RG"      # Right Guard
    RT = "RT"      # Right Tackle
    DL = "DL"      # Defensive Line (generic)
    DE = "DE"      # Defensive End
    DT = "DT"      # Defensive Tackle
    EDGE = "EDGE"  # Edge Rusher
    LB = "LB"      # Linebacker
    ILB = "ILB"    # Inside Linebacker
    OLB = "OLB"    # Outside Linebacker
    CB = "CB"      # Cornerback
    S = "S"        # Safety
    FS = "FS"      # Free Safety
    SS = "SS"      # Strong Safety
    K = "K"        # Kicker
    P = "P"        # Punter
    LS = "LS"      # Long Snapper


@dataclass
class PositionalValue:
    """
    Value weights for each position.
    
    Represents how much a position impacts team performance.
    Values are in approximate points per game.
    """
    position: Position
    starter_value: float      # Points per game impact of starter
    replacement_value: float  # Points per game impact of replacement/backup
    max_vor: float           # Maximum Value Over Replacement
    
    @property
    def value_over_replacement(self) -> float:
        """Calculate VOR."""
        return self.starter_value - self.replacement_value


# Position value weights based on NFL analytics research
POSITION_VALUES: Dict[Position, PositionalValue] = {
    # Quarterback - Most valuable by far
    Position.QB: PositionalValue(Position.QB, 3.5, 0.5, 3.0),
    
    # Premium pass rushers
    Position.EDGE: PositionalValue(Position.EDGE, 1.2, 0.4, 0.8),
    Position.DE: PositionalValue(Position.DE, 1.0, 0.4, 0.6),
    
    # Offensive skill positions
    Position.WR: PositionalValue(Position.WR, 0.9, 0.3, 0.6),
    Position.RB: PositionalValue(Position.RB, 0.7, 0.4, 0.3),
    Position.TE: PositionalValue(Position.TE, 0.6, 0.3, 0.3),
    
    # Offensive line
    Position.LT: PositionalValue(Position.LT, 0.8, 0.3, 0.5),
    Position.RT: PositionalValue(Position.RT, 0.6, 0.3, 0.3),
    Position.LG: PositionalValue(Position.LG, 0.4, 0.2, 0.2),
    Position.RG: PositionalValue(Position.RG, 0.4, 0.2, 0.2),
    Position.C: PositionalValue(Position.C, 0.5, 0.2, 0.3),
    Position.OL: PositionalValue(Position.OL, 0.5, 0.25, 0.25),
    
    # Secondary
    Position.CB: PositionalValue(Position.CB, 0.9, 0.3, 0.6),
    Position.S: PositionalValue(Position.S, 0.6, 0.3, 0.3),
    Position.FS: PositionalValue(Position.FS, 0.6, 0.3, 0.3),
    Position.SS: PositionalValue(Position.SS, 0.5, 0.3, 0.2),
    
    # Linebackers
    Position.LB: PositionalValue(Position.LB, 0.6, 0.3, 0.3),
    Position.ILB: PositionalValue(Position.ILB, 0.6, 0.3, 0.3),
    Position.OLB: PositionalValue(Position.OLB, 0.7, 0.3, 0.4),
    
    # Interior defensive line
    Position.DT: PositionalValue(Position.DT, 0.7, 0.3, 0.4),
    Position.DL: PositionalValue(Position.DL, 0.6, 0.3, 0.3),
    
    # Special teams (minimal impact on spread)
    Position.K: PositionalValue(Position.K, 0.3, 0.15, 0.15),
    Position.P: PositionalValue(Position.P, 0.15, 0.1, 0.05),
    Position.LS: PositionalValue(Position.LS, 0.05, 0.02, 0.03),
}


@dataclass
class PlayerValue:
    """
    Individual player's value to their team.
    
    Attributes:
        player_id: Unique identifier
        name: Player's name
        team: Team name
        position: Primary position
        depth_chart_rank: 1 = starter, 2 = backup, etc.
        snap_percentage: Percentage of snaps played (0-100)
        epa_contribution: Estimated EPA added per game
        is_pro_bowl: Pro Bowl caliber player
        is_all_pro: All-Pro caliber player
        games_played: Games played this season
    """
    player_id: str
    name: str
    team: str
    position: Position
    depth_chart_rank: int = 1
    snap_percentage: float = 100.0
    epa_contribution: Optional[float] = None
    is_pro_bowl: bool = False
    is_all_pro: bool = False
    games_played: int = 0
    
    def __post_init__(self):
        """Calculate value if not provided."""
        if self.epa_contribution is None:
            self.epa_contribution = self._estimate_epa_contribution()
    
    def _estimate_epa_contribution(self) -> float:
        """Estimate EPA contribution based on position and tier."""
        base_value = POSITION_VALUES.get(self.position)
        if not base_value:
            return 0.3  # Default for unknown positions
        
        # Adjust for depth chart position
        depth_multiplier = {
            1: 1.0,    # Starter
            2: 0.5,    # Primary backup
            3: 0.25,   # Third string
        }.get(self.depth_chart_rank, 0.15)
        
        # Adjust for elite status
        elite_multiplier = 1.0
        if self.is_all_pro:
            elite_multiplier = 1.4
        elif self.is_pro_bowl:
            elite_multiplier = 1.2
        
        return base_value.starter_value * depth_multiplier * elite_multiplier
    
    @property
    def replacement_level(self) -> float:
        """Get replacement level value for this position."""
        base = POSITION_VALUES.get(self.position)
        if not base:
            return 0.1
        return base.replacement_value
    
    @property
    def value_over_replacement(self) -> float:
        """Calculate this player's VOR."""
        return max(0, self.epa_contribution - self.replacement_level)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "team": self.team,
            "position": self.position.value,
            "depth_chart_rank": self.depth_chart_rank,
            "snap_percentage": self.snap_percentage,
            "epa_contribution": self.epa_contribution,
            "replacement_level": self.replacement_level,
            "value_over_replacement": self.value_over_replacement,
            "is_pro_bowl": self.is_pro_bowl,
            "is_all_pro": self.is_all_pro,
        }


class PlayerRoster:
    """
    Manages a team's roster with player values.
    """
    
    def __init__(self, team: str):
        """Initialize roster for a team."""
        self.team = team
        self._players: Dict[str, PlayerValue] = {}
    
    def add_player(self, player: PlayerValue) -> None:
        """Add a player to the roster."""
        self._players[player.player_id] = player
    
    def get_player(self, player_id: str) -> Optional[PlayerValue]:
        """Get a player by ID."""
        return self._players.get(player_id)
    
    def get_player_by_name(self, name: str) -> Optional[PlayerValue]:
        """Get a player by name (partial match)."""
        name_lower = name.lower()
        for player in self._players.values():
            if name_lower in player.name.lower():
                return player
        return None
    
    def get_starters(self) -> List[PlayerValue]:
        """Get all starters (depth_chart_rank = 1)."""
        return [p for p in self._players.values() if p.depth_chart_rank == 1]
    
    def get_by_position(self, position: Position) -> List[PlayerValue]:
        """Get all players at a position."""
        return [p for p in self._players.values() if p.position == position]
    
    def get_total_roster_vor(self) -> float:
        """Calculate total VOR for the roster."""
        return sum(p.value_over_replacement for p in self._players.values())
    
    def get_offensive_vor(self) -> float:
        """Calculate offensive VOR."""
        offensive_positions = {
            Position.QB, Position.RB, Position.WR, Position.TE,
            Position.LT, Position.LG, Position.C, Position.RG, Position.RT, Position.OL
        }
        return sum(
            p.value_over_replacement 
            for p in self._players.values() 
            if p.position in offensive_positions
        )
    
    def get_defensive_vor(self) -> float:
        """Calculate defensive VOR."""
        defensive_positions = {
            Position.DE, Position.DT, Position.DL, Position.EDGE,
            Position.LB, Position.ILB, Position.OLB,
            Position.CB, Position.S, Position.FS, Position.SS
        }
        return sum(
            p.value_over_replacement 
            for p in self._players.values() 
            if p.position in defensive_positions
        )
    
    def __len__(self) -> int:
        return len(self._players)


# =============================================================================
# Story 6.2: Injury Report Data Model
# =============================================================================

class InjuryStatus(Enum):
    """NFL injury designations."""
    OUT = "Out"                    # Will not play (0%)
    DOUBTFUL = "Doubtful"          # Unlikely to play (~25%)
    QUESTIONABLE = "Questionable"  # May or may not play (~50-70%)
    PROBABLE = "Probable"          # Likely to play (~95%) - rarely used now
    IR = "IR"                      # Injured Reserve - out long term
    PUP = "PUP"                    # Physically Unable to Perform
    NFI = "NFI"                    # Non-Football Injury
    SUSPENDED = "Suspended"        # Suspended - not injury related
    HEALTHY = "Healthy"            # No injury designation


class PracticeStatus(Enum):
    """Practice participation status."""
    DNP = "DNP"        # Did Not Practice
    LIMITED = "LP"     # Limited Participation
    FULL = "FP"        # Full Participation
    UNKNOWN = "Unknown"


@dataclass
class InjuryReport:
    """
    Individual injury report for a player.
    
    Attributes:
        player_id: Player identifier
        player_name: Player's name
        team: Team name
        position: Player's position
        injury_type: Type of injury (knee, shoulder, concussion, etc.)
        status: Official injury designation
        practice_status: Practice participation
        game_id: Game this report applies to
        report_date: When this report was issued
        is_game_time_decision: Requires last-minute decision
        notes: Additional context
    """
    player_id: str
    player_name: str
    team: str
    position: Position
    injury_type: str
    status: InjuryStatus
    practice_status: PracticeStatus = PracticeStatus.UNKNOWN
    game_id: Optional[str] = None
    report_date: Optional[date] = None
    is_game_time_decision: bool = False
    notes: Optional[str] = None
    
    @property
    def play_probability(self) -> float:
        """
        Estimate probability of playing based on designation.
        
        Based on historical data of how often players with each
        designation actually play.
        """
        probabilities = {
            InjuryStatus.HEALTHY: 1.00,
            InjuryStatus.PROBABLE: 0.95,
            InjuryStatus.QUESTIONABLE: 0.65,  # Historical average ~65%
            InjuryStatus.DOUBTFUL: 0.20,
            InjuryStatus.OUT: 0.00,
            InjuryStatus.IR: 0.00,
            InjuryStatus.PUP: 0.00,
            InjuryStatus.NFI: 0.00,
            InjuryStatus.SUSPENDED: 0.00,
        }
        
        base_prob = probabilities.get(self.status, 0.5)
        
        # Adjust based on practice status
        if self.status == InjuryStatus.QUESTIONABLE:
            if self.practice_status == PracticeStatus.FULL:
                base_prob = 0.85  # Full practice = likely to play
            elif self.practice_status == PracticeStatus.LIMITED:
                base_prob = 0.60
            elif self.practice_status == PracticeStatus.DNP:
                base_prob = 0.35
        
        # Game-time decisions have more uncertainty
        if self.is_game_time_decision:
            # Move toward 50% uncertainty
            base_prob = base_prob * 0.8 + 0.5 * 0.2
        
        return round(base_prob, 2)
    
    @property
    def miss_probability(self) -> float:
        """Probability of NOT playing."""
        return 1.0 - self.play_probability
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "position": self.position.value,
            "injury_type": self.injury_type,
            "status": self.status.value,
            "practice_status": self.practice_status.value,
            "play_probability": self.play_probability,
            "is_game_time_decision": self.is_game_time_decision,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "notes": self.notes,
        }


@dataclass
class TeamInjuryReport:
    """
    Aggregated injury report for a team.
    
    Contains all injury reports for a team for a specific game.
    """
    team: str
    game_id: str
    week: int
    season: int
    report_date: date
    injuries: List[InjuryReport] = field(default_factory=list)
    
    def add_injury(self, injury: InjuryReport) -> None:
        """Add an injury to the report."""
        self.injuries.append(injury)
    
    def get_out_players(self) -> List[InjuryReport]:
        """Get all players ruled OUT."""
        return [i for i in self.injuries if i.status == InjuryStatus.OUT]
    
    def get_questionable_players(self) -> List[InjuryReport]:
        """Get all QUESTIONABLE players."""
        return [i for i in self.injuries if i.status == InjuryStatus.QUESTIONABLE]
    
    def get_doubtful_players(self) -> List[InjuryReport]:
        """Get all DOUBTFUL players."""
        return [i for i in self.injuries if i.status == InjuryStatus.DOUBTFUL]
    
    def get_by_position(self, position: Position) -> List[InjuryReport]:
        """Get injuries at a specific position."""
        return [i for i in self.injuries if i.position == position]
    
    def has_qb_injury(self) -> bool:
        """Check if QB is injured."""
        qb_injuries = self.get_by_position(Position.QB)
        return any(i.status != InjuryStatus.HEALTHY for i in qb_injuries)
    
    def get_total_expected_vor_loss(self, roster: Optional[PlayerRoster] = None) -> float:
        """
        Calculate expected VOR loss from injuries.
        
        Args:
            roster: Team roster for actual player values (uses estimates if None)
        """
        total_loss = 0.0
        
        for injury in self.injuries:
            if injury.status == InjuryStatus.HEALTHY:
                continue
            
            # Get player value
            if roster:
                player = roster.get_player(injury.player_id)
                if player:
                    vor = player.value_over_replacement
                else:
                    # Estimate based on position
                    pos_value = POSITION_VALUES.get(injury.position)
                    vor = pos_value.value_over_replacement if pos_value else 0.3
            else:
                pos_value = POSITION_VALUES.get(injury.position)
                vor = pos_value.value_over_replacement if pos_value else 0.3
            
            # Expected loss = VOR × probability of missing
            expected_loss = vor * injury.miss_probability
            total_loss += expected_loss
        
        return round(total_loss, 2)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "team": self.team,
            "game_id": self.game_id,
            "week": self.week,
            "season": self.season,
            "report_date": self.report_date.isoformat(),
            "total_injuries": len(self.injuries),
            "out_count": len(self.get_out_players()),
            "questionable_count": len(self.get_questionable_players()),
            "doubtful_count": len(self.get_doubtful_players()),
            "injuries": [i.to_dict() for i in self.injuries],
        }


# =============================================================================
# Story 6.3: Injury Data Fetcher
# =============================================================================

class InjuryDataSource(DataSource):
    """
    Abstract base for injury data sources.
    """
    
    def __init__(self, name: str):
        """Initialize injury data source."""
        super().__init__(name, DataSourceType.SCHEDULE)  # Reuse schedule type
        self._source_type_name = "INJURY"
    
    @abstractmethod
    def fetch_team_injuries(
        self,
        team: str,
        season: int,
        week: int
    ) -> Optional[TeamInjuryReport]:
        """Fetch injuries for a specific team."""
        pass
    
    @abstractmethod
    def fetch_week_injuries(
        self,
        season: int,
        week: int
    ) -> Dict[str, TeamInjuryReport]:
        """Fetch injuries for all teams for a week."""
        pass


class MockInjuryDataSource(InjuryDataSource):
    """
    Mock injury data source for testing.
    
    Generates realistic injury scenarios for testing.
    """
    
    def __init__(self):
        """Initialize mock injury source."""
        super().__init__("Mock Injuries")
        self._injury_cache: Dict[str, Dict[str, TeamInjuryReport]] = {}
    
    def fetch(
        self,
        season: int = 2025,
        week: Optional[int] = None,
        **kwargs
    ) -> DataFetchResult:
        """Fetch mock injury data."""
        if week is None:
            week = 1
        
        cache_key = f"{season}_W{week}"
        
        if cache_key in self._injury_cache:
            return DataFetchResult(
                source=self.name,
                source_type=self.source_type,
                status=DataFetchStatus.CACHED,
                data={"injuries": self._injury_cache[cache_key]},
                records_fetched=len(self._injury_cache[cache_key])
            )
        
        injuries = self._generate_mock_injuries(season, week)
        self._injury_cache[cache_key] = injuries
        
        return DataFetchResult(
            source=self.name,
            source_type=self.source_type,
            status=DataFetchStatus.SUCCESS,
            data={"injuries": injuries},
            records_fetched=len(injuries)
        )
    
    def is_available(self) -> bool:
        return True
    
    def fetch_team_injuries(
        self,
        team: str,
        season: int,
        week: int
    ) -> Optional[TeamInjuryReport]:
        """Fetch injuries for a specific team."""
        result = self.fetch(season=season, week=week)
        if result.is_success:
            injuries = result.data.get("injuries", {})
            return injuries.get(team)
        return None
    
    def fetch_week_injuries(
        self,
        season: int,
        week: int
    ) -> Dict[str, TeamInjuryReport]:
        """Fetch injuries for all teams."""
        result = self.fetch(season=season, week=week)
        if result.is_success:
            return result.data.get("injuries", {})
        return {}
    
    def _generate_mock_injuries(
        self,
        season: int,
        week: int
    ) -> Dict[str, TeamInjuryReport]:
        """Generate realistic mock injury data."""
        import random
        random.seed(season * 100 + week)  # Reproducible
        
        injuries = {}
        
        # Common injury types
        injury_types = [
            "knee", "ankle", "hamstring", "shoulder", "concussion",
            "back", "groin", "calf", "foot", "hand", "ribs", "illness"
        ]
        
        for team in ALL_NFL_TEAMS:
            report = TeamInjuryReport(
                team=team,
                game_id=f"{season}_W{week}_{team[:3]}",
                week=week,
                season=season,
                report_date=date.today()
            )
            
            # Each team has 0-5 injuries typically
            num_injuries = random.choices([0, 1, 2, 3, 4, 5], weights=[10, 25, 30, 20, 10, 5])[0]
            
            # Positions that could be injured (weighted by roster size)
            position_weights = [
                (Position.WR, 15), (Position.CB, 12), (Position.LB, 12),
                (Position.RB, 8), (Position.OL, 15), (Position.DL, 12),
                (Position.S, 8), (Position.TE, 6), (Position.EDGE, 8),
                (Position.QB, 3), (Position.K, 1),
            ]
            positions = [p for p, w in position_weights for _ in range(w)]
            
            for i in range(num_injuries):
                position = random.choice(positions)
                injury_type = random.choice(injury_types)
                
                # Determine status (weighted toward questionable)
                status = random.choices(
                    [InjuryStatus.OUT, InjuryStatus.DOUBTFUL, 
                     InjuryStatus.QUESTIONABLE, InjuryStatus.PROBABLE],
                    weights=[20, 15, 50, 15]
                )[0]
                
                # Practice status correlates with game status
                if status == InjuryStatus.OUT:
                    practice = PracticeStatus.DNP
                elif status == InjuryStatus.DOUBTFUL:
                    practice = random.choice([PracticeStatus.DNP, PracticeStatus.LIMITED])
                elif status == InjuryStatus.QUESTIONABLE:
                    practice = random.choice([PracticeStatus.LIMITED, PracticeStatus.FULL, PracticeStatus.DNP])
                else:
                    practice = PracticeStatus.FULL
                
                injury = InjuryReport(
                    player_id=f"{team[:3]}_{position.value}_{i}",
                    player_name=f"{team.split()[0]} {position.value}{i+1}",
                    team=team,
                    position=position,
                    injury_type=injury_type,
                    status=status,
                    practice_status=practice,
                    game_id=report.game_id,
                    report_date=report.report_date,
                    is_game_time_decision=(status == InjuryStatus.QUESTIONABLE and random.random() < 0.3)
                )
                
                report.add_injury(injury)
            
            injuries[team] = report
        
        return injuries


# =============================================================================
# Story 6.4: Injury Impact Calculator
# =============================================================================

@dataclass 
class InjuryImpact:
    """
    Calculated injury impact for a team.
    
    Attributes:
        team: Team name
        total_vor_loss: Total expected VOR loss from injuries
        offensive_impact: Points lost on offense
        defensive_impact: Points lost on defense
        rating_adjustment: Final adjustment to team rating
        key_injuries: List of highest-impact injuries
        uncertainty_level: How uncertain is the impact (GTD players)
        confidence: Confidence in this estimate
    """
    team: str
    total_vor_loss: float
    offensive_impact: float
    defensive_impact: float
    rating_adjustment: float
    key_injuries: List[Dict] = field(default_factory=list)
    uncertainty_level: str = "Low"  # Low, Medium, High
    confidence: str = "Medium"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "team": self.team,
            "total_vor_loss": self.total_vor_loss,
            "offensive_impact": self.offensive_impact,
            "defensive_impact": self.defensive_impact,
            "rating_adjustment": self.rating_adjustment,
            "key_injuries": self.key_injuries,
            "uncertainty_level": self.uncertainty_level,
            "confidence": self.confidence,
        }


class InjuryImpactCalculator:
    """
    Calculates rating adjustments based on injuries.
    
    Converts injury reports into point spread adjustments.
    """
    
    # Maximum adjustment to prevent extreme swings
    MAX_ADJUSTMENT = 5.0
    
    # Diminishing returns threshold for multiple injuries at same position
    DIMINISHING_RETURNS_THRESHOLD = 2
    
    def __init__(
        self,
        max_adjustment: float = 5.0,
        qb_multiplier: float = 1.2,
        key_player_threshold: float = 0.4
    ):
        """
        Initialize calculator.
        
        Args:
            max_adjustment: Cap on total injury adjustment
            qb_multiplier: Extra weight for QB injuries
            key_player_threshold: VOR threshold for "key injury" designation
        """
        self.max_adjustment = max_adjustment
        self.qb_multiplier = qb_multiplier
        self.key_player_threshold = key_player_threshold
    
    def calculate_impact(
        self,
        injury_report: TeamInjuryReport,
        roster: Optional[PlayerRoster] = None
    ) -> InjuryImpact:
        """
        Calculate injury impact for a team.
        
        Args:
            injury_report: Team's injury report
            roster: Optional roster for actual player values
        
        Returns:
            InjuryImpact with rating adjustment
        """
        offensive_positions = {
            Position.QB, Position.RB, Position.WR, Position.TE,
            Position.LT, Position.LG, Position.C, Position.RG, Position.RT, Position.OL
        }
        
        defensive_positions = {
            Position.DE, Position.DT, Position.DL, Position.EDGE,
            Position.LB, Position.ILB, Position.OLB,
            Position.CB, Position.S, Position.FS, Position.SS
        }
        
        offensive_loss = 0.0
        defensive_loss = 0.0
        key_injuries = []
        position_counts: Dict[Position, int] = {}
        gtd_count = 0
        
        for injury in injury_report.injuries:
            if injury.status == InjuryStatus.HEALTHY:
                continue
            
            # Get player value
            vor = self._get_player_vor(injury, roster)
            
            # Apply miss probability
            expected_loss = vor * injury.miss_probability
            
            # Apply diminishing returns for multiple injuries at same position
            pos = injury.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
            if position_counts[pos] > self.DIMINISHING_RETURNS_THRESHOLD:
                expected_loss *= 0.5  # 50% reduction for 3rd+ injury at position
            
            # Apply QB multiplier
            if injury.position == Position.QB:
                expected_loss *= self.qb_multiplier
            
            # Categorize as offensive or defensive
            if injury.position in offensive_positions:
                offensive_loss += expected_loss
            elif injury.position in defensive_positions:
                defensive_loss += expected_loss
            
            # Track key injuries
            if vor >= self.key_player_threshold:
                key_injuries.append({
                    "player": injury.player_name,
                    "position": injury.position.value,
                    "status": injury.status.value,
                    "vor": vor,
                    "expected_impact": round(expected_loss, 2),
                    "play_probability": injury.play_probability,
                })
            
            # Track game-time decisions for uncertainty
            if injury.is_game_time_decision:
                gtd_count += 1
        
        # Total loss (uncapped)
        total_loss = offensive_loss + defensive_loss
        
        # Apply cap
        rating_adjustment = -min(total_loss, self.max_adjustment)
        
        # Determine uncertainty level
        if gtd_count >= 3:
            uncertainty = "High"
        elif gtd_count >= 1:
            uncertainty = "Medium"
        else:
            uncertainty = "Low"
        
        # Determine confidence
        if uncertainty == "High" or total_loss > self.max_adjustment:
            confidence = "Low"
        elif uncertainty == "Medium" or len(key_injuries) > 2:
            confidence = "Medium"
        else:
            confidence = "High"
        
        return InjuryImpact(
            team=injury_report.team,
            total_vor_loss=round(total_loss, 2),
            offensive_impact=round(-offensive_loss, 2),
            defensive_impact=round(-defensive_loss, 2),
            rating_adjustment=round(rating_adjustment, 2),
            key_injuries=sorted(key_injuries, key=lambda x: x["vor"], reverse=True),
            uncertainty_level=uncertainty,
            confidence=confidence,
        )
    
    def _get_player_vor(
        self,
        injury: InjuryReport,
        roster: Optional[PlayerRoster]
    ) -> float:
        """Get player's VOR from roster or estimate."""
        if roster:
            player = roster.get_player(injury.player_id)
            if player:
                return player.value_over_replacement
        
        # Estimate from position
        pos_value = POSITION_VALUES.get(injury.position)
        if pos_value:
            return pos_value.value_over_replacement
        return 0.3  # Default
    
    def calculate_game_impact(
        self,
        home_report: TeamInjuryReport,
        away_report: TeamInjuryReport,
        home_roster: Optional[PlayerRoster] = None,
        away_roster: Optional[PlayerRoster] = None
    ) -> Dict[str, InjuryImpact]:
        """
        Calculate injury impacts for both teams in a game.
        
        Returns dict with 'home' and 'away' InjuryImpact objects.
        """
        return {
            "home": self.calculate_impact(home_report, home_roster),
            "away": self.calculate_impact(away_report, away_roster),
        }
    
    def get_spread_adjustment(
        self,
        home_impact: InjuryImpact,
        away_impact: InjuryImpact
    ) -> float:
        """
        Calculate net spread adjustment from injuries.
        
        Positive = home team benefits (away more injured)
        Negative = away team benefits (home more injured)
        """
        # Home team's injuries hurt them, away team's injuries help
        return away_impact.rating_adjustment - home_impact.rating_adjustment


# =============================================================================
# Story 6.5: Injury-Adjusted Predictions
# =============================================================================

@dataclass
class InjuryAdjustedPrediction:
    """
    Game prediction with injury adjustments.
    
    Extends GamePrediction with injury context.
    """
    base_prediction: GamePrediction
    home_injury_impact: Optional[InjuryImpact]
    away_injury_impact: Optional[InjuryImpact]
    injury_spread_adjustment: float
    adjusted_spread: float
    adjusted_home_win_prob: float
    adjusted_away_win_prob: float
    injury_confidence: str
    
    @property
    def game(self) -> NFLGame:
        return self.base_prediction.game
    
    @property
    def pick(self) -> str:
        """Adjusted pick based on injuries."""
        if self.adjusted_spread < 0:
            return self.game.home_team
        elif self.adjusted_spread > 0:
            return self.game.away_team
        return self.game.home_team  # Tie goes to home
    
    @property
    def formatted_adjusted_spread(self) -> str:
        """Format adjusted spread for display."""
        if self.adjusted_spread < 0:
            return f"{self.game.home_team} {self.adjusted_spread:.1f}"
        elif self.adjusted_spread > 0:
            return f"{self.game.away_team} {-self.adjusted_spread:.1f}"
        return "PICK"
    
    @property
    def injury_edge(self) -> Optional[float]:
        """Edge vs Vegas after injury adjustment."""
        if self.game.vegas_spread is None:
            return None
        # Vegas spread is from home perspective
        return self.adjusted_spread - self.game.vegas_spread
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "game_id": self.game.game_id,
            "home_team": self.game.home_team,
            "away_team": self.game.away_team,
            "base_spread": self.base_prediction.predicted_spread,
            "injury_adjustment": self.injury_spread_adjustment,
            "adjusted_spread": self.adjusted_spread,
            "formatted_spread": self.formatted_adjusted_spread,
            "pick": self.pick,
            "adjusted_home_win_prob": self.adjusted_home_win_prob,
            "adjusted_away_win_prob": self.adjusted_away_win_prob,
            "vegas_spread": self.game.vegas_spread,
            "injury_edge": self.injury_edge,
            "injury_confidence": self.injury_confidence,
            "home_injuries": self.home_injury_impact.to_dict() if self.home_injury_impact else None,
            "away_injuries": self.away_injury_impact.to_dict() if self.away_injury_impact else None,
        }


class InjuryAdjustedPredictor:
    """
    Predictor that incorporates injury adjustments.
    
    Wraps SpreadPredictor and adds injury impact.
    """
    
    def __init__(
        self,
        base_predictor: SpreadPredictor,
        injury_calculator: Optional[InjuryImpactCalculator] = None,
        injury_source: Optional[InjuryDataSource] = None
    ):
        """
        Initialize injury-adjusted predictor.
        
        Args:
            base_predictor: Base SpreadPredictor
            injury_calculator: Calculator for injury impacts
            injury_source: Source for injury data
        """
        self.base_predictor = base_predictor
        self.injury_calculator = injury_calculator or InjuryImpactCalculator()
        self.injury_source = injury_source or MockInjuryDataSource()
        self.win_prob_model = LogisticWinProbability()
    
    def predict_game(
        self,
        game: NFLGame,
        week: int,
        home_injuries: Optional[TeamInjuryReport] = None,
        away_injuries: Optional[TeamInjuryReport] = None,
        fetch_injuries: bool = True
    ) -> InjuryAdjustedPrediction:
        """
        Generate injury-adjusted prediction.
        
        Args:
            game: Game to predict
            week: Current week
            home_injuries: Pre-fetched home injury report
            away_injuries: Pre-fetched away injury report
            fetch_injuries: Whether to fetch if not provided
        """
        # Get base prediction
        base_pred = self.base_predictor.predict_game(game, week)
        
        # Fetch injuries if needed
        if home_injuries is None and fetch_injuries:
            home_injuries = self.injury_source.fetch_team_injuries(
                game.home_team, game.season, week
            )
        
        if away_injuries is None and fetch_injuries:
            away_injuries = self.injury_source.fetch_team_injuries(
                game.away_team, game.season, week
            )
        
        # Calculate impacts
        home_impact = None
        away_impact = None
        injury_adjustment = 0.0
        
        if home_injuries:
            home_impact = self.injury_calculator.calculate_impact(home_injuries)
        
        if away_injuries:
            away_impact = self.injury_calculator.calculate_impact(away_injuries)
        
        if home_impact and away_impact:
            injury_adjustment = self.injury_calculator.get_spread_adjustment(
                home_impact, away_impact
            )
        elif home_impact:
            injury_adjustment = -home_impact.rating_adjustment
        elif away_impact:
            injury_adjustment = away_impact.rating_adjustment
        
        # Apply adjustment to spread
        adjusted_spread = base_pred.predicted_spread + injury_adjustment
        
        # Recalculate win probability with adjusted spread
        adjusted_home_prob = self.win_prob_model.calculate(-adjusted_spread)
        adjusted_away_prob = 1.0 - adjusted_home_prob
        
        # Determine overall injury confidence
        if home_impact and away_impact:
            if home_impact.uncertainty_level == "High" or away_impact.uncertainty_level == "High":
                injury_confidence = "Low"
            elif home_impact.uncertainty_level == "Medium" or away_impact.uncertainty_level == "Medium":
                injury_confidence = "Medium"
            else:
                injury_confidence = "High"
        else:
            injury_confidence = "Medium"  # Missing data
        
        return InjuryAdjustedPrediction(
            base_prediction=base_pred,
            home_injury_impact=home_impact,
            away_injury_impact=away_impact,
            injury_spread_adjustment=round(injury_adjustment, 1),
            adjusted_spread=round(adjusted_spread, 1),
            adjusted_home_win_prob=round(adjusted_home_prob, 4),
            adjusted_away_win_prob=round(adjusted_away_prob, 4),
            injury_confidence=injury_confidence,
        )
    
    def predict_week(
        self,
        games: List[NFLGame],
        week: int,
        season: int
    ) -> List[InjuryAdjustedPrediction]:
        """
        Generate injury-adjusted predictions for all games in a week.
        """
        # Fetch all injuries at once
        all_injuries = self.injury_source.fetch_week_injuries(season, week)
        
        predictions = []
        for game in games:
            home_injuries = all_injuries.get(game.home_team)
            away_injuries = all_injuries.get(game.away_team)
            
            pred = self.predict_game(
                game, week,
                home_injuries=home_injuries,
                away_injuries=away_injuries,
                fetch_injuries=False
            )
            predictions.append(pred)
        
        return predictions


# =============================================================================
# Story 6.6: Injury Tracking & Validation
# =============================================================================

@dataclass
class InjuryPredictionResult:
    """
    Tracks injury prediction accuracy.
    """
    game_id: str
    team: str
    player_name: str
    position: str
    predicted_play_prob: float
    predicted_status: str
    actual_played: bool  # True if they actually played
    vor: float
    expected_impact: float
    actual_impact: float  # Will be 0 if played, VOR if didn't
    
    @property
    def prediction_correct(self) -> bool:
        """Was the play/no-play prediction correct?"""
        predicted_plays = self.predicted_play_prob >= 0.5
        return predicted_plays == self.actual_played
    
    @property
    def impact_error(self) -> float:
        """Error in impact prediction."""
        return abs(self.expected_impact - self.actual_impact)


class InjuryTracker:
    """
    Tracks injury prediction accuracy over time.
    """
    
    def __init__(self):
        """Initialize tracker."""
        self._results: List[InjuryPredictionResult] = []
        self._game_adjustments: List[Dict] = []
    
    def add_result(
        self,
        injury: InjuryReport,
        actually_played: bool,
        vor: float
    ) -> InjuryPredictionResult:
        """
        Record an injury prediction result.
        
        Args:
            injury: Original injury report
            actually_played: Whether player actually played
            vor: Player's VOR
        """
        expected_impact = vor * injury.miss_probability
        actual_impact = 0.0 if actually_played else vor
        
        result = InjuryPredictionResult(
            game_id=injury.game_id or "",
            team=injury.team,
            player_name=injury.player_name,
            position=injury.position.value,
            predicted_play_prob=injury.play_probability,
            predicted_status=injury.status.value,
            actual_played=actually_played,
            vor=vor,
            expected_impact=expected_impact,
            actual_impact=actual_impact,
        )
        
        self._results.append(result)
        return result
    
    def add_game_adjustment(
        self,
        game_id: str,
        predicted_adjustment: float,
        actual_spread_error: float,
        injuries_used: int
    ) -> None:
        """Track game-level adjustment accuracy."""
        self._game_adjustments.append({
            "game_id": game_id,
            "predicted_adjustment": predicted_adjustment,
            "actual_spread_error": actual_spread_error,
            "injuries_used": injuries_used,
        })
    
    def get_play_probability_accuracy(self) -> Dict[str, float]:
        """
        Analyze play probability prediction accuracy.
        
        Groups by predicted probability bucket and compares to actual rate.
        """
        buckets = {
            "0-20%": {"predicted": 0, "played": 0, "total": 0},
            "20-40%": {"predicted": 0, "played": 0, "total": 0},
            "40-60%": {"predicted": 0, "played": 0, "total": 0},
            "60-80%": {"predicted": 0, "played": 0, "total": 0},
            "80-100%": {"predicted": 0, "played": 0, "total": 0},
        }
        
        for result in self._results:
            prob = result.predicted_play_prob
            
            if prob < 0.20:
                bucket = "0-20%"
            elif prob < 0.40:
                bucket = "20-40%"
            elif prob < 0.60:
                bucket = "40-60%"
            elif prob < 0.80:
                bucket = "60-80%"
            else:
                bucket = "80-100%"
            
            buckets[bucket]["total"] += 1
            buckets[bucket]["predicted"] += prob
            if result.actual_played:
                buckets[bucket]["played"] += 1
        
        # Calculate accuracy by bucket
        accuracy = {}
        for bucket, data in buckets.items():
            if data["total"] > 0:
                avg_predicted = data["predicted"] / data["total"]
                actual_rate = data["played"] / data["total"]
                accuracy[bucket] = {
                    "count": data["total"],
                    "avg_predicted": round(avg_predicted, 3),
                    "actual_rate": round(actual_rate, 3),
                    "calibration_error": round(abs(avg_predicted - actual_rate), 3),
                }
        
        return accuracy
    
    def get_position_accuracy(self) -> Dict[str, Dict]:
        """Analyze accuracy by position."""
        by_position: Dict[str, List[InjuryPredictionResult]] = {}
        
        for result in self._results:
            pos = result.position
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(result)
        
        accuracy = {}
        for pos, results in by_position.items():
            correct = sum(1 for r in results if r.prediction_correct)
            total_impact_error = sum(r.impact_error for r in results)
            
            accuracy[pos] = {
                "count": len(results),
                "prediction_accuracy": round(correct / len(results), 3) if results else 0,
                "avg_impact_error": round(total_impact_error / len(results), 3) if results else 0,
            }
        
        return accuracy
    
    def get_summary(self) -> Dict:
        """Get overall summary statistics."""
        if not self._results:
            return {"total_tracked": 0}
        
        correct = sum(1 for r in self._results if r.prediction_correct)
        total_impact_error = sum(r.impact_error for r in self._results)
        
        return {
            "total_tracked": len(self._results),
            "overall_accuracy": round(correct / len(self._results), 3),
            "avg_impact_error": round(total_impact_error / len(self._results), 3),
            "by_probability": self.get_play_probability_accuracy(),
            "by_position": self.get_position_accuracy(),
            "game_adjustments": len(self._game_adjustments),
        }
    
    def generate_report(self) -> str:
        """Generate a text report of injury tracking results."""
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            "INJURY PREDICTION TRACKING REPORT",
            "=" * 60,
            "",
            f"Total Injuries Tracked: {summary['total_tracked']}",
            f"Overall Prediction Accuracy: {summary['overall_accuracy']:.1%}",
            f"Average Impact Error: {summary['avg_impact_error']:.2f} pts",
            "",
            "ACCURACY BY PROBABILITY BUCKET",
            "-" * 40,
        ]
        
        by_prob = summary.get("by_probability", {})
        for bucket, data in by_prob.items():
            lines.append(
                f"  {bucket}: Predicted {data['avg_predicted']:.0%}, "
                f"Actual {data['actual_rate']:.0%}, "
                f"Error {data['calibration_error']:.0%} (n={data['count']})"
            )
        
        lines.extend([
            "",
            "ACCURACY BY POSITION",
            "-" * 40,
        ])
        
        by_pos = summary.get("by_position", {})
        for pos, data in sorted(by_pos.items(), key=lambda x: x[1]["count"], reverse=True):
            lines.append(
                f"  {pos}: {data['prediction_accuracy']:.0%} accuracy, "
                f"{data['avg_impact_error']:.2f} avg error (n={data['count']})"
            )
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions for Epic 6
# =============================================================================

def create_sample_roster(team: str, with_stars: bool = True) -> PlayerRoster:
    """
    Create a sample roster for testing.
    
    Args:
        team: Team name
        with_stars: Include elite players
    """
    roster = PlayerRoster(team)
    
    # QB
    roster.add_player(PlayerValue(
        player_id=f"{team[:3]}_QB_1",
        name=f"{team} QB1",
        team=team,
        position=Position.QB,
        depth_chart_rank=1,
        is_pro_bowl=with_stars,
    ))
    
    # Skill positions
    for i in range(3):
        roster.add_player(PlayerValue(
            player_id=f"{team[:3]}_WR_{i+1}",
            name=f"{team} WR{i+1}",
            team=team,
            position=Position.WR,
            depth_chart_rank=i+1,
            is_pro_bowl=(i == 0 and with_stars),
        ))
    
    roster.add_player(PlayerValue(
        player_id=f"{team[:3]}_RB_1",
        name=f"{team} RB1",
        team=team,
        position=Position.RB,
        depth_chart_rank=1,
    ))
    
    roster.add_player(PlayerValue(
        player_id=f"{team[:3]}_TE_1",
        name=f"{team} TE1",
        team=team,
        position=Position.TE,
        depth_chart_rank=1,
    ))
    
    # O-Line
    for pos in [Position.LT, Position.LG, Position.C, Position.RG, Position.RT]:
        roster.add_player(PlayerValue(
            player_id=f"{team[:3]}_{pos.value}_1",
            name=f"{team} {pos.value}",
            team=team,
            position=pos,
            depth_chart_rank=1,
        ))
    
    # Defense
    for i in range(2):
        roster.add_player(PlayerValue(
            player_id=f"{team[:3]}_EDGE_{i+1}",
            name=f"{team} EDGE{i+1}",
            team=team,
            position=Position.EDGE,
            depth_chart_rank=i+1,
            is_pro_bowl=(i == 0 and with_stars),
        ))
    
    for i in range(2):
        roster.add_player(PlayerValue(
            player_id=f"{team[:3]}_CB_{i+1}",
            name=f"{team} CB{i+1}",
            team=team,
            position=Position.CB,
            depth_chart_rank=i+1,
        ))
    
    for i in range(2):
        roster.add_player(PlayerValue(
            player_id=f"{team[:3]}_S_{i+1}",
            name=f"{team} S{i+1}",
            team=team,
            position=Position.S,
            depth_chart_rank=i+1,
        ))
    
    for i in range(3):
        roster.add_player(PlayerValue(
            player_id=f"{team[:3]}_LB_{i+1}",
            name=f"{team} LB{i+1}",
            team=team,
            position=Position.LB,
            depth_chart_rank=i+1,
        ))
    
    return roster


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_preseason_data(season: int = 2025) -> Dict[str, TeamPreseasonData]:
    """
    Create default preseason data for all 32 teams.
    
    Uses reasonable defaults for a new season (all teams at 8.5 wins projected).
    Replace with actual Vegas lines for real predictions.
    """
    data = {}
    for team in ALL_NFL_TEAMS:
        conf, div = get_team_division(team)
        data[team] = TeamPreseasonData(
            team=team,
            season=season,
            projected_wins=8.5,  # League average
            division=div,
            conference=conf,
            source="Default (replace with Vegas lines)"
        )
    return data


def create_model_from_vegas_wins(
    season: int,
    vegas_wins: Dict[str, float],
    decay_config: Optional[DecayCurveConfig] = None
) -> NFLSeasonModel:
    """
    Convenience function to create a model from Vegas win totals.
    
    Args:
        season: NFL season year
        vegas_wins: Dict mapping team name to projected wins
        decay_config: Optional custom decay curve config
    
    Returns:
        Initialized NFLSeasonModel
    """
    decay_curve = PreseasonDecayCurve(decay_config) if decay_config else PreseasonDecayCurve()
    model = NFLSeasonModel(season=season, decay_curve=decay_curve)
    model.load_preseason_from_wins(vegas_wins)
    return model


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Example: Create a season model
    print("=" * 70)
    print("NFL Prediction Model - Epic 1 & 2 Demo")
    print("=" * 70)
    
    # Example Vegas win totals (2025 projections)
    sample_vegas_wins = {
        "Buffalo Bills": 11.5,
        "Miami Dolphins": 9.5,
        "New England Patriots": 4.5,
        "New York Jets": 8.5,
        "Baltimore Ravens": 11.5,
        "Cincinnati Bengals": 10.5,
        "Cleveland Browns": 7.5,
        "Pittsburgh Steelers": 8.5,
        "Houston Texans": 10.0,
        "Indianapolis Colts": 8.5,
        "Jacksonville Jaguars": 7.5,
        "Tennessee Titans": 6.5,
        "Denver Broncos": 8.5,
        "Kansas City Chiefs": 11.5,
        "Las Vegas Raiders": 6.5,
        "Los Angeles Chargers": 9.5,
        "Dallas Cowboys": 9.0,
        "New York Giants": 7.0,
        "Philadelphia Eagles": 10.5,
        "Washington Commanders": 8.0,
        "Chicago Bears": 8.5,
        "Detroit Lions": 11.0,
        "Green Bay Packers": 9.5,
        "Minnesota Vikings": 9.0,
        "Atlanta Falcons": 8.0,
        "Carolina Panthers": 5.5,
        "New Orleans Saints": 7.5,
        "Tampa Bay Buccaneers": 8.5,
        "Arizona Cardinals": 7.0,
        "Los Angeles Rams": 9.0,
        "San Francisco 49ers": 11.5,
        "Seattle Seahawks": 8.5,
    }
    
    # Create model
    model = create_model_from_vegas_wins(2025, sample_vegas_wins)
    
    # =========================================================================
    # EPIC 1 DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("EPIC 1: TEAM RATINGS")
    print("=" * 70)
    
    # Show preseason power rankings (Week 0)
    print("\n📊 PRESEASON POWER RANKINGS (Week 0)")
    print("-" * 60)
    rankings = model.get_power_rankings(week=0)
    for r in rankings[:10]:
        print(f"{r['rank']:2}. {r['team']:<25} Rating: {r['power_rating']:+6.2f}")
    print("...")
    
    # Simulate some in-season data
    print("\n📈 SIMULATING WEEK 5 DATA...")
    
    # Add sample in-season data for key teams
    week5_data = {
        "Buffalo Bills": TeamInSeasonData(
            team="Buffalo Bills", season=2025, week=5, games_played=5,
            offensive_epa_per_play=0.18, defensive_epa_per_play=-0.05,
            total_offensive_plays=320, total_defensive_plays=310,
            wins=4, losses=1, point_differential=65
        ),
        "Kansas City Chiefs": TeamInSeasonData(
            team="Kansas City Chiefs", season=2025, week=5, games_played=5,
            offensive_epa_per_play=0.12, defensive_epa_per_play=-0.08,
            total_offensive_plays=315, total_defensive_plays=325,
            wins=5, losses=0, point_differential=45
        ),
        "Detroit Lions": TeamInSeasonData(
            team="Detroit Lions", season=2025, week=5, games_played=5,
            offensive_epa_per_play=0.15, defensive_epa_per_play=0.02,
            total_offensive_plays=340, total_defensive_plays=300,
            wins=4, losses=1, point_differential=52
        ),
        "Carolina Panthers": TeamInSeasonData(
            team="Carolina Panthers", season=2025, week=5, games_played=5,
            offensive_epa_per_play=-0.15, defensive_epa_per_play=0.10,
            total_offensive_plays=280, total_defensive_plays=340,
            wins=1, losses=4, point_differential=-55
        ),
        "Denver Broncos": TeamInSeasonData(
            team="Denver Broncos", season=2025, week=5, games_played=5,
            offensive_epa_per_play=0.05, defensive_epa_per_play=-0.03,
            total_offensive_plays=300, total_defensive_plays=310,
            wins=3, losses=2, point_differential=15
        ),
        "Miami Dolphins": TeamInSeasonData(
            team="Miami Dolphins", season=2025, week=5, games_played=5,
            offensive_epa_per_play=0.14, defensive_epa_per_play=0.05,
            total_offensive_plays=330, total_defensive_plays=290,
            wins=3, losses=2, point_differential=20
        ),
    }
    
    model.update_inseason_data(week=5, data=week5_data)
    
    # Show decay curve weights
    preseason_w, inseason_w = model.get_decay_weights(5)
    print(f"\nWeek 5 Weights: Preseason={preseason_w:.2%}, In-Season={inseason_w:.2%}")
    
    # Show updated ratings for sample teams
    print("\n📊 WEEK 5 RATINGS (Sample Teams)")
    print("-" * 60)
    
    for team in ["Buffalo Bills", "Kansas City Chiefs", "Detroit Lions"]:
        rating = model.get_team_rating(team, week=5)
        print(f"\n{team}:")
        print(f"  Preseason Rating: {rating.preseason_rating:+.2f}")
        print(f"  In-Season Rating: {rating.in_season_rating:+.2f}")
        print(f"  Blended Rating:   {rating.power_rating:+.2f}")
        print(f"  Confidence:       {rating.confidence}")
    
    # =========================================================================
    # EPIC 2 DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("EPIC 2: GAME PREDICTIONS")
    print("=" * 70)
    
    # Story 2.1 & 2.2: Game Setup with Home Field Advantage
    print("\n🏟️  HOME FIELD ADVANTAGE CALCULATIONS")
    print("-" * 60)
    
    hfa_calc = HomeFieldAdvantageCalculator()
    
    # Example games
    games = [
        NFLGame(
            game_id="2025_W6_BUF@KC",
            season=2025, week=6,
            game_date=date(2025, 10, 12),
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            game_time="4:25 PM",
            vegas_spread=-1.5,  # KC favored by 1.5
            vegas_total=48.5
        ),
        NFLGame(
            game_id="2025_W6_MIA@DEN",
            season=2025, week=6,
            game_date=date(2025, 10, 12),
            home_team="Denver Broncos",
            away_team="Miami Dolphins",
            game_time="4:05 PM",
            vegas_spread=-2.5,
            vegas_total=44.5
        ),
        NFLGame(
            game_id="2025_W6_CAR@DET",
            season=2025, week=6,
            game_date=date(2025, 10, 12),
            home_team="Detroit Lions",
            away_team="Carolina Panthers",
            game_time="1:00 PM",
            vegas_spread=-13.5,
            vegas_total=46.5
        ),
    ]
    
    for game in games:
        hfa = hfa_calc.calculate(game)
        travel = hfa_calc.get_travel_distance(game.away_team, game.home_team)
        tz = hfa_calc.get_timezone_change(game.away_team, game.home_team)
        print(f"\n{game.away_team} @ {game.home_team}")
        print(f"  Home Field Advantage: {hfa:+.2f} pts")
        print(f"  Travel Distance: {travel:,.0f} miles")
        print(f"  Timezone Change: {tz:+d} hours")
    
    # Story 2.3: Rest & Travel Adjustments
    print("\n\n😴 REST & TRAVEL ADJUSTMENTS")
    print("-" * 60)
    
    rest_adjuster = RestTravelAdjuster()
    
    # Example: Bills coming off MNF, Chiefs had normal rest
    buf_context = rest_adjuster.build_game_context(
        team="Buffalo Bills",
        is_home=False,
        days_rest=6,
        previous_game_was_mnf=True,
        home_team="Kansas City Chiefs"
    )
    
    kc_context = rest_adjuster.build_game_context(
        team="Kansas City Chiefs",
        is_home=True,
        days_rest=7
    )
    
    rest_adj = rest_adjuster.calculate_adjustment(kc_context, buf_context)
    print(f"\nBUF @ KC - Rest/Travel Analysis:")
    print(f"  Bills: {buf_context.days_rest} days rest, {buf_context.travel_miles:.0f} mi travel")
    print(f"  Chiefs: {kc_context.days_rest} days rest (home)")
    print(f"  Net Adjustment: {rest_adj:+.2f} pts (+ = KC)")
    
    # Story 2.4 & 2.5: Win Probability & Spread Prediction
    print("\n\n🎯 SPREAD PREDICTIONS (WEEK 6)")
    print("-" * 60)
    
    # Add context to games
    games[0].home_context = kc_context
    games[0].away_context = buf_context
    
    predictor = SpreadPredictor(model)
    
    for game in games:
        pred = predictor.predict_game(game, week=5)  # Use week 5 ratings
        
        print(f"\n{game.away_team} @ {game.home_team}")
        print(f"  Power Ratings: {game.away_team} {pred.away_power_rating:+.2f} vs "
              f"{game.home_team} {pred.home_power_rating:+.2f}")
        print(f"  HFA: {pred.home_field_advantage:+.2f}, "
              f"Rest/Travel: {pred.rest_travel_adjustment:+.2f}")
        print(f"  Predicted Spread: {pred.formatted_spread}")
        print(f"  Win Prob: {game.home_team} {pred.home_win_probability:.1%} | "
              f"{game.away_team} {pred.away_win_probability:.1%}")
        print(f"  Vegas Spread: {game.home_team} {game.vegas_spread:+.1f}")
        if pred.edge:
            edge_team = game.home_team if pred.edge > 0 else game.away_team
            print(f"  Edge: {abs(pred.edge):.1f} pts on {edge_team}")
    
    # Story 2.6: Weekly Schedule Manager
    print("\n\n📅 WEEKLY SCHEDULE MANAGER")
    print("-" * 60)
    
    manager = WeeklyScheduleManager(model, season=2025)
    manager.add_games(games)
    
    # Simulate results
    manager.update_game_result("2025_W6_BUF@KC", home_score=24, away_score=27)
    manager.update_game_result("2025_W6_MIA@DEN", home_score=31, away_score=24)
    manager.update_game_result("2025_W6_CAR@DET", home_score=38, away_score=10)
    
    # Get summary
    summary = manager.get_week_summary(week=6)
    print(f"\nWeek 6 Results:")
    print(f"  Games: {summary['games_completed']}/{summary['total_games']}")
    print(f"  Correct Picks: {summary['correct_picks']}")
    print(f"  Pick Accuracy: {summary['pick_accuracy']}%")
    
    # Show best bets feature
    print("\n\n💰 BEST BETS (Edge >= 2.0 pts)")
    print("-" * 60)
    best_bets = manager.get_best_bets(week=6, min_edge=2.0)
    if best_bets:
        for bet in best_bets:
            edge_team = bet.game.home_team if bet.edge > 0 else bet.game.away_team
            print(f"  {edge_team}: {abs(bet.edge):.1f} pt edge")
    else:
        print("  No high-edge bets found")
    
    # =========================================================================
    # EPIC 3 DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("EPIC 3: BACKTESTING & VALIDATION")
    print("=" * 70)
    
    # Story 3.1: Prediction Result Tracking
    print("\n📝 PREDICTION RESULT TRACKING")
    print("-" * 60)
    
    tracker = PredictionResultTracker()
    
    # Track the predictions we made
    for game in games:
        pred = predictor.predict_game(game, week=5)
        if game.is_completed:
            result = tracker.add_result(pred, game.home_score, game.away_score)
            status = "✓" if result.su_correct else "✗"
            print(f"  {result.away_team} @ {result.home_team}: {status} "
                  f"(Error: {result.spread_error:+.1f} pts)")
    
    print(f"\n  Total tracked: {tracker.total_predictions} games")
    
    # Story 3.2: Accuracy Metrics
    print("\n\n📊 ACCURACY METRICS")
    print("-" * 60)
    
    accuracy_calc = AccuracyCalculator()
    metrics = accuracy_calc.calculate(tracker.get_all_results())
    
    print(f"  Straight-Up:    {metrics.su_correct}/{metrics.total_games} ({metrics.su_accuracy:.1%})")
    print(f"  Mean Abs Error: {metrics.mean_absolute_error:.2f} pts")
    print(f"  RMSE:           {metrics.rmse:.2f} pts")
    print(f"  Brier Score:    {metrics.brier_score:.4f}")
    print(f"  Theoretical ROI: {metrics.theoretical_roi:.1f}%")
    
    # Story 3.3: Calibration Analysis
    print("\n\n📈 CALIBRATION ANALYSIS")
    print("-" * 60)
    
    calibration_analyzer = CalibrationAnalyzer(num_buckets=5)
    calibration = calibration_analyzer.analyze(tracker.get_all_results())
    
    print(f"  Expected Calibration Error: {calibration['expected_calibration_error']:.4f}")
    print(f"  Assessment: {calibration['calibration_assessment']}")
    print(f"\n  Probability Buckets:")
    for bucket in calibration['buckets']:
        print(f"    {bucket['range']}: {bucket['wins']}/{bucket['predictions']} "
              f"(Expected: {bucket['expected_rate']:.0%}, Actual: {bucket['actual_rate']:.0%})")
    
    # Story 3.4: Performance by Category
    print("\n\n🔍 PERFORMANCE BY CATEGORY")
    print("-" * 60)
    
    perf_analyzer = PerformanceAnalyzer()
    
    # By spread size
    by_spread = perf_analyzer.analyze_by_spread_size(tracker.get_all_results())
    print("\n  By Spread Size:")
    for bucket in by_spread:
        print(f"    {bucket['category']}: {bucket['correct']}/{bucket['games']} ({bucket['accuracy']:.1%})")
    
    # Favorites vs underdogs
    fav_dog = perf_analyzer.analyze_favorites_vs_underdogs(tracker.get_all_results())
    print("\n  Favorites vs Underdogs:")
    print(f"    Favorites: {fav_dog['favorites']['correct']}/{fav_dog['favorites']['games']} "
          f"({fav_dog['favorites']['accuracy']:.1%})")
    print(f"    Underdogs: {fav_dog['underdogs']['correct']}/{fav_dog['underdogs']['games']} "
          f"({fav_dog['underdogs']['accuracy']:.1%})")
    
    # Story 3.5: Backtesting Demo (simulated multi-week)
    print("\n\n🔄 BACKTESTING FRAMEWORK")
    print("-" * 60)
    
    # Add more games for a fuller backtest demo
    extra_games = [
        NFLGame(
            game_id="2025_W5_NYJ@NE",
            season=2025, week=5,
            game_date=date(2025, 10, 5),
            home_team="New England Patriots",
            away_team="New York Jets",
            vegas_spread=3.0,
            home_score=17, away_score=24, is_completed=True
        ),
        NFLGame(
            game_id="2025_W5_CLE@BAL",
            season=2025, week=5,
            game_date=date(2025, 10, 5),
            home_team="Baltimore Ravens",
            away_team="Cleveland Browns",
            vegas_spread=-10.5,
            home_score=31, away_score=14, is_completed=True
        ),
        NFLGame(
            game_id="2025_W5_SF@ARI",
            season=2025, week=5,
            game_date=date(2025, 10, 5),
            home_team="Arizona Cardinals",
            away_team="San Francisco 49ers",
            vegas_spread=6.5,
            home_score=20, away_score=28, is_completed=True
        ),
    ]
    
    # Add to manager for backtesting
    manager.add_games(extra_games)
    
    # Create backtester
    backtest_config = BacktestConfig(
        start_week=5,
        end_week=6,
        require_vegas_line=True
    )
    
    backtester = SeasonBacktester(model, manager, backtest_config)
    backtest_results = backtester.run_backtest()
    
    if "error" not in backtest_results:
        print(f"\n  Backtest Results (Weeks 5-6):")
        print(f"    Total Games: {backtest_results['summary']['total_games']}")
        print(f"    SU Record:   {backtest_results['summary']['su_record']} ({backtest_results['summary']['su_accuracy']})")
        print(f"    MAE:         {backtest_results['summary']['mae']}")
        print(f"    RMSE:        {backtest_results['summary']['rmse']}")
    
    # Story 3.6: Performance Report
    print("\n\n📋 PERFORMANCE REPORT PREVIEW")
    print("-" * 60)
    
    report_generator = PerformanceReportGenerator()
    
    # Generate best/worst predictions
    all_results = []
    for game in games + extra_games:
        if game.is_completed:
            pred = predictor.predict_game(game, week=5)
            result = PredictionResult(
                prediction=pred,
                actual_winner=game.winner,
                actual_spread=game.actual_spread,
                actual_total=game.total_points
            )
            all_results.append(result)
    
    best_worst = report_generator.generate_best_and_worst(all_results, n=2)
    print(best_worst)
    
    # =========================================================================
    # EPIC 4 DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("EPIC 4: DATA INTEGRATION & AUTOMATION")
    print("=" * 70)
    
    # Story 4.1: Data Source Abstraction
    print("\n🔌 DATA SOURCE REGISTRY")
    print("-" * 60)
    
    registry = DataSourceRegistry()
    
    # Register mock data sources
    epa_source = MockEPADataSource()
    schedule_source = MockScheduleDataSource()
    vegas_source = MockVegasDataSource()
    
    registry.register(epa_source)
    registry.register(schedule_source)
    registry.register(vegas_source)
    
    print(f"  Registered sources:")
    print(f"    - {epa_source.name} ({epa_source.source_type.value})")
    print(f"    - {schedule_source.name} ({schedule_source.source_type.value})")
    print(f"    - {vegas_source.name} ({vegas_source.source_type.value})")
    
    # Story 4.2: EPA Data Fetching
    print("\n\n📊 EPA DATA FETCHING")
    print("-" * 60)
    
    epa_result = registry.fetch_best(
        DataSourceType.EPA,
        season=2025,
        week=10
    )
    
    print(f"  Fetch Status: {epa_result.status.value}")
    print(f"  Records: {epa_result.records_fetched}")
    
    if epa_result.is_success and epa_result.data:
        sample_epa = epa_result.data["epa_data"][:3]
        print(f"\n  Sample EPA Data (Week 10):")
        for epa in sample_epa:
            print(f"    {epa.team}: Off {epa.offensive_epa_per_play:+.3f}, "
                  f"Def {epa.defensive_epa_per_play:+.3f}, "
                  f"Net {epa.net_epa_per_play:+.3f}")
    
    # Story 4.3: Schedule Fetching
    print("\n\n📅 SCHEDULE FETCHING")
    print("-" * 60)
    
    schedule_result = registry.fetch_best(
        DataSourceType.SCHEDULE,
        season=2025,
        week=10
    )
    
    print(f"  Fetch Status: {schedule_result.status.value}")
    print(f"  Games: {schedule_result.records_fetched}")
    
    if schedule_result.is_success and schedule_result.data:
        sample_games = schedule_result.data["schedule"][:3]
        print(f"\n  Sample Games (Week 10):")
        for game in sample_games:
            print(f"    {game.away_team} @ {game.home_team} ({game.game_date})")
    
    # Story 4.4: Vegas Lines Fetching
    print("\n\n💰 VEGAS LINES FETCHING")
    print("-" * 60)
    
    vegas_result = registry.fetch_best(
        DataSourceType.VEGAS_LINES,
        season=2025,
        week=10
    )
    
    print(f"  Fetch Status: {vegas_result.status.value}")
    print(f"  Lines: {vegas_result.records_fetched}")
    
    if vegas_result.is_success and vegas_result.data:
        sample_lines = vegas_result.data["lines"][:3]
        print(f"\n  Sample Lines (Week 10):")
        for line in sample_lines:
            print(f"    {line.game_id}: Spread {line.spread:+.1f}, "
                  f"Total {line.total}, ML {line.home_ml}/{line.away_ml}")
    
    # Story 4.5: Data Pipeline
    print("\n\n🔄 DATA PIPELINE ORCHESTRATION")
    print("-" * 60)
    
    # Create a new model for the pipeline demo
    pipeline_model = create_model_from_vegas_wins(2025, sample_vegas_wins)
    
    pipeline = WeeklyUpdatePipeline(pipeline_model, registry)
    pipeline_result = pipeline.run(season=2025, week=10)
    
    print(f"\n  {pipeline_result.summary()}")
    
    predictions = pipeline.get_predictions()
    print(f"\n  Generated {len(predictions)} predictions")
    
    if predictions:
        print(f"\n  Sample Predictions:")
        for pred in predictions[:3]:
            print(f"    {pred.game.away_team} @ {pred.game.home_team}: "
                  f"{pred.formatted_spread}")
    
    # Story 4.6: Automation Manager
    print("\n\n⚙️  WEEKLY AUTOMATION MANAGER")
    print("-" * 60)
    
    automation = WeeklyAutomationManager(pipeline_model)
    automation.register_data_source(epa_source)
    automation.register_data_source(schedule_source)
    automation.register_data_source(vegas_source)
    
    # Schedule tasks
    automation.schedule_weekly_update(day=1, hour=6)  # Tuesday 6am
    automation.schedule_prediction_refresh(day=3, hour=18)  # Thursday 6pm
    
    print(f"  Scheduled Tasks:")
    for schedule in automation.get_schedules():
        print(f"    - {schedule['name']}: {schedule['day']} @ {schedule['time']}")
    
    # Simulate running an update
    print(f"\n  Running manual weekly update...")
    update_result = automation.run_weekly_update(season=2025, week=10)
    print(f"    Status: {update_result.overall_status.value}")
    print(f"    Duration: {update_result.duration_seconds:.2f}s")
    
    # Show run history
    history = automation.get_run_history(limit=5)
    print(f"\n  Run History: {len(history)} entries")
    
    # =========================================================================
    # EPIC 5 DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("EPIC 5: OUTPUT & EXPORT")
    print("=" * 70)
    
    # Get some predictions and rankings for export demos
    demo_predictions = predictions[:5] if predictions else []
    demo_rankings = pipeline_model.get_power_rankings(week=10)[:10]
    
    # Story 5.1 & 5.2: JSON Export
    print("\n📄 JSON EXPORT")
    print("-" * 60)
    
    json_exporter = JSONExporter()
    json_output = json_exporter.export_predictions(demo_predictions, week=10, season=2025)
    print(f"  JSON output length: {len(json_output)} characters")
    print(f"  Preview (first 200 chars):")
    print(f"    {json_output[:200]}...")
    
    # Story 5.3: CSV Export
    print("\n\n📊 CSV EXPORT")
    print("-" * 60)
    
    csv_exporter = CSVExporter()
    csv_output = csv_exporter.export_predictions(demo_predictions, week=10, season=2025)
    csv_lines = csv_output.split('\n')
    print(f"  CSV rows: {len(csv_lines)}")
    print(f"  Header: {csv_lines[0][:80]}...")
    if len(csv_lines) > 1:
        print(f"  First row: {csv_lines[1][:80]}...")
    
    # Story 5.4: Markdown Export
    print("\n\n📝 MARKDOWN EXPORT")
    print("-" * 60)
    
    md_exporter = MarkdownExporter(ExportConfig(include_analysis=False))
    md_output = md_exporter.export_predictions(demo_predictions, week=10, season=2025)
    print(f"  Markdown output length: {len(md_output)} characters")
    # Show first few lines
    md_lines = md_output.split('\n')[:8]
    for line in md_lines:
        print(f"  {line}")
    print("  ...")
    
    # Story 5.5: HTML Export
    print("\n\n🌐 HTML DASHBOARD")
    print("-" * 60)
    
    html_exporter = HTMLExporter()
    html_output = html_exporter.export_dashboard(
        demo_predictions, demo_rankings, week=10, season=2025
    )
    print(f"  HTML output length: {len(html_output)} characters")
    print(f"  Contains CSS: {'<style>' in html_output}")
    print(f"  Contains table: {'<table' in html_output}")
    print(f"  Contains dashboard: {'dashboard' in html_output}")
    
    # Story 5.6: Export Manager
    print("\n\n📦 EXPORT MANAGER")
    print("-" * 60)
    
    export_manager = ExportManager()
    print(f"  Supported formats: {export_manager.get_supported_formats()}")
    
    # Generate all formats
    all_exports = export_manager.export_all_formats(demo_predictions, week=10, season=2025)
    print(f"\n  Generated {len(all_exports)} format exports:")
    for format_name, content in all_exports.items():
        print(f"    - {format_name}: {len(content)} characters")
    
    # Generate weekly package
    print("\n  Generating weekly content package...")
    package = export_manager.generate_weekly_package(
        demo_predictions, demo_rankings, week=10, season=2025
    )
    print(f"  Package contains {len(package)} items:")
    for item_name in list(package.keys())[:6]:
        print(f"    - {item_name}")
    if len(package) > 6:
        print(f"    ... and {len(package) - 6} more")
    
    # Newsletter preview
    print("\n\n📰 NEWSLETTER PREVIEW")
    print("-" * 60)
    newsletter = md_exporter.export_weekly_newsletter(
        demo_predictions, demo_rankings, week=10, season=2025
    )
    newsletter_preview = '\n'.join(newsletter.split('\n')[:15])
    print(newsletter_preview)
    print("...")
    
    # =========================================================================
    # EPIC 6 DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("EPIC 6: INJURY IMPACT MODELING")
    print("=" * 70)
    
    # Story 6.1: Player Value Model
    print("\n🏈 PLAYER VALUE MODEL")
    print("-" * 60)
    
    print("\n  Position Values (Points per Game Impact):")
    print(f"  {'Position':<10} {'Starter':<10} {'Backup':<10} {'Max VOR':<10}")
    print("  " + "-" * 40)
    for pos in [Position.QB, Position.EDGE, Position.WR, Position.CB, Position.LT, Position.RB]:
        pv = POSITION_VALUES[pos]
        print(f"  {pos.value:<10} {pv.starter_value:<10.2f} {pv.replacement_value:<10.2f} {pv.max_vor:<10.2f}")
    
    # Create sample roster
    print("\n  Creating sample roster for Buffalo Bills...")
    bills_roster = create_sample_roster("Buffalo Bills", with_stars=True)
    print(f"  Roster size: {len(bills_roster)} players")
    print(f"  Total roster VOR: {bills_roster.get_total_roster_vor():.2f}")
    print(f"  Offensive VOR: {bills_roster.get_offensive_vor():.2f}")
    print(f"  Defensive VOR: {bills_roster.get_defensive_vor():.2f}")
    
    # Story 6.2 & 6.3: Injury Report Data Model & Fetcher
    print("\n\n🏥 INJURY REPORT SYSTEM")
    print("-" * 60)
    
    # Create mock injury source
    injury_source = MockInjuryDataSource()
    
    # Fetch injuries for Week 10
    injury_result = injury_source.fetch(season=2025, week=10)
    print(f"\n  Fetched injury data: {injury_result.status.value}")
    print(f"  Teams with injury reports: {injury_result.records_fetched}")
    
    # Show sample injury report
    injuries_data = injury_result.data.get("injuries", {})
    sample_team = "Buffalo Bills"
    if sample_team in injuries_data:
        team_injuries = injuries_data[sample_team]
        print(f"\n  {sample_team} Injury Report (Week 10):")
        print(f"    Total injuries: {len(team_injuries.injuries)}")
        print(f"    Out: {len(team_injuries.get_out_players())}")
        print(f"    Questionable: {len(team_injuries.get_questionable_players())}")
        print(f"    Doubtful: {len(team_injuries.get_doubtful_players())}")
        
        # Show individual injuries
        if team_injuries.injuries:
            print(f"\n    Injury Details:")
            for inj in team_injuries.injuries[:5]:  # Show first 5
                print(f"      - {inj.player_name} ({inj.position.value}): {inj.status.value} "
                      f"[{inj.injury_type}] - Play prob: {inj.play_probability:.0%}")
    
    # Story 6.4: Injury Impact Calculator
    print("\n\n📊 INJURY IMPACT CALCULATOR")
    print("-" * 60)
    
    calculator = InjuryImpactCalculator(max_adjustment=5.0)
    
    # Calculate impact for sample teams
    print("\n  Calculating injury impacts...")
    for team_name in ["Buffalo Bills", "Kansas City Chiefs", "Detroit Lions"]:
        if team_name in injuries_data:
            team_report = injuries_data[team_name]
            impact = calculator.calculate_impact(team_report)
            
            print(f"\n  {team_name}:")
            print(f"    Total VOR Loss: {impact.total_vor_loss:.2f}")
            print(f"    Offensive Impact: {impact.offensive_impact:+.2f} pts")
            print(f"    Defensive Impact: {impact.defensive_impact:+.2f} pts")
            print(f"    Rating Adjustment: {impact.rating_adjustment:+.2f} pts")
            print(f"    Uncertainty: {impact.uncertainty_level}")
            
            if impact.key_injuries:
                print(f"    Key Injuries:")
                for ki in impact.key_injuries[:3]:
                    print(f"      - {ki['player']} ({ki['position']}): "
                          f"{ki['status']}, VOR: {ki['vor']:.2f}")
    
    # Story 6.5: Injury-Adjusted Predictions
    print("\n\n🎯 INJURY-ADJUSTED PREDICTIONS")
    print("-" * 60)
    
    # Create injury-adjusted predictor
    base_predictor = SpreadPredictor(pipeline_model)
    injury_predictor = InjuryAdjustedPredictor(
        base_predictor=base_predictor,
        injury_calculator=calculator,
        injury_source=injury_source
    )
    
    # Create a sample game
    sample_game = NFLGame(
        game_id="2025_W10_KC@BUF",
        season=2025, week=10,
        game_date=date(2025, 11, 9),
        home_team="Buffalo Bills",
        away_team="Kansas City Chiefs",
        vegas_spread=-2.5
    )
    
    # Get injury-adjusted prediction
    injury_pred = injury_predictor.predict_game(sample_game, week=10)
    
    print(f"\n  Game: {sample_game.away_team} @ {sample_game.home_team}")
    print(f"\n  Base Prediction:")
    print(f"    Spread: {injury_pred.base_prediction.formatted_spread}")
    print(f"    Win Prob: {injury_pred.base_prediction.home_win_probability:.0%} / "
          f"{injury_pred.base_prediction.away_win_probability:.0%}")
    
    print(f"\n  Injury Adjustment: {injury_pred.injury_spread_adjustment:+.1f} pts")
    
    print(f"\n  Injury-Adjusted Prediction:")
    print(f"    Adjusted Spread: {injury_pred.formatted_adjusted_spread}")
    print(f"    Adjusted Win Prob: {injury_pred.adjusted_home_win_prob:.0%} / "
          f"{injury_pred.adjusted_away_win_prob:.0%}")
    print(f"    Pick: {injury_pred.pick}")
    print(f"    Injury Confidence: {injury_pred.injury_confidence}")
    
    if injury_pred.home_injury_impact:
        print(f"\n  Home ({sample_game.home_team}) Injuries: "
              f"{injury_pred.home_injury_impact.rating_adjustment:+.1f} impact")
    if injury_pred.away_injury_impact:
        print(f"  Away ({sample_game.away_team}) Injuries: "
              f"{injury_pred.away_injury_impact.rating_adjustment:+.1f} impact")
    
    # Story 6.6: Injury Tracking
    print("\n\n📈 INJURY TRACKING & VALIDATION")
    print("-" * 60)
    
    tracker = InjuryTracker()
    
    # Simulate some tracking results
    print("\n  Simulating injury outcome tracking...")
    
    # Add some mock results
    for team_name in ["Buffalo Bills", "Kansas City Chiefs"]:
        if team_name in injuries_data:
            team_report = injuries_data[team_name]
            for inj in team_report.injuries[:3]:
                # Simulate whether player actually played
                import random
                random.seed(hash(inj.player_id))
                actually_played = random.random() < inj.play_probability
                
                vor = POSITION_VALUES.get(inj.position, PositionalValue(inj.position, 0.5, 0.2, 0.3)).value_over_replacement
                tracker.add_result(inj, actually_played, vor)
    
    # Get summary
    summary = tracker.get_summary()
    print(f"\n  Tracking Summary:")
    print(f"    Total tracked: {summary['total_tracked']}")
    print(f"    Overall accuracy: {summary['overall_accuracy']:.0%}")
    print(f"    Avg impact error: {summary['avg_impact_error']:.2f} pts")
    
    # Show report preview
    print("\n  Tracking Report Preview:")
    report = tracker.generate_report()
    report_lines = report.split('\n')[:15]
    for line in report_lines:
        print(f"    {line}")
    print("    ...")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ EPICS 1-6 IMPLEMENTATION COMPLETE!")
    print("=" * 70)
    print("""
Implemented Features:
  
EPIC 1 - Team Ratings:
  ✓ TeamPreseasonData - Vegas lines to power ratings
  ✓ TeamInSeasonData - EPA-based performance tracking
  ✓ PreseasonDecayCurve - Smooth transition from preseason to in-season
  ✓ BlendedTeamRating - Combined rating with confidence tiers
  ✓ NFLSeasonModel - Full 32-team season management
  ✓ RatingValidator - Edge case handling

EPIC 2 - Game Predictions:
  ✓ NFLGame - Complete game/matchup data model
  ✓ HomeFieldAdvantageCalculator - HFA with altitude/dome/divisional factors
  ✓ RestTravelAdjuster - Bye week, short week, travel penalties
  ✓ WinProbabilityModel - Logistic & linear probability conversion
  ✓ GamePrediction - Spread, win prob, Vegas comparison
  ✓ SpreadPredictor - Main prediction interface
  ✓ WeeklyScheduleManager - Schedule & results tracking

EPIC 3 - Backtesting & Validation:
  ✓ PredictionResult - Track prediction outcomes
  ✓ PredictionResultTracker - Store/index all results
  ✓ AccuracyCalculator - SU, ATS, MAE, RMSE, Brier score, ROI
  ✓ CalibrationAnalyzer - Probability calibration analysis
  ✓ PerformanceAnalyzer - Breakdown by category
  ✓ SeasonBacktester - Full season backtesting
  ✓ PerformanceReportGenerator - Formatted reports

EPIC 4 - Data Integration & Automation:
  ✓ DataSource - Abstract base for all data sources
  ✓ DataSourceRegistry - Manage multiple sources with fallback
  ✓ EPADataPoint - Normalized EPA data structure
  ✓ MockEPADataSource - Testing/demo EPA source
  ✓ ScheduleEntry - Normalized schedule data
  ✓ VegasLine - Betting line data structure
  ✓ DataPipeline - Multi-step data orchestration
  ✓ WeeklyUpdatePipeline - Pre-built weekly update flow
  ✓ WeeklyAutomationManager - Scheduled task management
  ✓ Team name normalization across sources

EPIC 5 - Output & Export:
  ✓ ExportFormat - Enum of supported formats
  ✓ ExportConfig - Configurable export options
  ✓ Exporter - Abstract base class for exporters
  ✓ JSONExporter - Structured JSON output
  ✓ CSVExporter - Tabular data for spreadsheets
  ✓ MarkdownExporter - Documentation & newsletters
  ✓ HTMLExporter - Styled dashboards & reports
  ✓ ExportManager - Multi-format export orchestration
  ✓ Weekly newsletter generation
  ✓ Full dashboard HTML generation

EPIC 6 - Injury Impact Modeling:
  ✓ Position - NFL position enum with all positions
  ✓ PositionalValue - Position importance weights
  ✓ PlayerValue - Individual player VOR calculation
  ✓ PlayerRoster - Team roster management
  ✓ InjuryStatus - NFL injury designations (Out, Doubtful, Questionable, etc.)
  ✓ InjuryReport - Individual player injury data
  ✓ TeamInjuryReport - Aggregated team injuries
  ✓ MockInjuryDataSource - Testing injury data source
  ✓ InjuryImpact - Calculated injury effect on team
  ✓ InjuryImpactCalculator - VOR-based impact calculation
  ✓ InjuryAdjustedPrediction - Prediction with injury context
  ✓ InjuryAdjustedPredictor - Full injury-aware prediction pipeline
  ✓ InjuryTracker - Track injury prediction accuracy
  ✓ Calibration tracking by probability bucket
  ✓ Position-level accuracy analysis
""")
