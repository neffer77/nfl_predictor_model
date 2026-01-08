"""
Team Ratings - Rating system blending preseason and in-season data.

This module contains:
- DecayCurveType: Enum for decay curve types
- DecayCurveConfig: Configuration for decay curves
- PreseasonDecayCurve: Calculates blending weights over the season
- TeamRatingCalculator: Calculates blended team ratings
- NFLSeasonModel: Full 32-team season management
- RatingValidator: Validation and edge case handling
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

from nfl_predictor.constants import (
    NFL_TEAMS,
    ALL_NFL_TEAMS,
    get_team_division,
    are_division_rivals,
)
from nfl_predictor.models.teams import (
    TeamPreseasonData,
    TeamInSeasonData,
    BlendedTeamRating,
)


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
