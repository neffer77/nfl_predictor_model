"""
Parameters - Model parameters and tuning values.

This module contains:
- ModelParameters: Complete model parameter set
- DecayParameters: Decay curve parameters
- HomeFieldParameters: Home field advantage parameters
- EPAParameters: EPA scaling parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import json


@dataclass
class DecayParameters:
    """
    Decay curve parameters for preseason-to-in-season blending.

    Attributes:
        initial_preseason_weight: Starting weight for preseason data (week 1)
        decay_rate: Rate of exponential decay per week
        min_preseason_weight: Minimum preseason weight (asymptote)
        transition_week: Week where preseason/in-season weights equal
    """
    initial_preseason_weight: float = 0.8
    decay_rate: float = 0.15
    min_preseason_weight: float = 0.1
    transition_week: int = 8

    def get_preseason_weight(self, week: int) -> float:
        """
        Calculate preseason weight for a given week.

        Args:
            week: Week number (1-18)

        Returns:
            Weight for preseason data (0-1)
        """
        if week <= 0:
            return self.initial_preseason_weight

        import math
        weight = self.initial_preseason_weight * math.exp(-self.decay_rate * (week - 1))
        return max(weight, self.min_preseason_weight)

    def get_inseason_weight(self, week: int) -> float:
        """Get in-season weight (complement of preseason weight)."""
        return 1.0 - self.get_preseason_weight(week)


@dataclass
class HomeFieldParameters:
    """
    Home field advantage parameters.

    Attributes:
        base_hfa: Base home field advantage in points
        dome_bonus: Additional HFA for dome teams
        altitude_bonus: Additional HFA for high altitude (Denver)
        division_reduction: HFA reduction for division games
        rivalry_reduction: HFA reduction for rivalry games
        weather_max_reduction: Maximum HFA reduction for weather
    """
    base_hfa: float = 2.5
    dome_bonus: float = 0.5
    altitude_bonus: float = 1.0
    division_reduction: float = 0.3
    rivalry_reduction: float = 0.5
    weather_max_reduction: float = 1.0

    # Travel adjustments
    travel_per_timezone: float = 0.2
    short_week_penalty: float = 0.5
    thursday_game_penalty: float = 0.3

    def calculate_base_hfa(
        self,
        is_dome: bool = False,
        is_altitude: bool = False
    ) -> float:
        """
        Calculate base HFA with stadium bonuses.

        Args:
            is_dome: Whether home team plays in dome
            is_altitude: Whether home team is high altitude

        Returns:
            Base HFA in points
        """
        hfa = self.base_hfa

        if is_dome:
            hfa += self.dome_bonus
        if is_altitude:
            hfa += self.altitude_bonus

        return hfa


@dataclass
class EPAParameters:
    """
    EPA (Expected Points Added) scaling parameters.

    Attributes:
        offense_weight: Weight for offensive EPA
        defense_weight: Weight for defensive EPA
        special_teams_weight: Weight for special teams
        recent_games_emphasis: Weight for recent games
        scaling_factor: Overall EPA to spread conversion
    """
    offense_weight: float = 1.0
    defense_weight: float = 1.0
    special_teams_weight: float = 0.3
    recent_games_emphasis: float = 0.6
    scaling_factor: float = 3.5

    # EPA component weights
    pass_epa_weight: float = 0.55
    rush_epa_weight: float = 0.45

    def calculate_team_strength(
        self,
        off_epa: float,
        def_epa: float,
        st_epa: float = 0.0
    ) -> float:
        """
        Calculate overall team strength from EPA components.

        Args:
            off_epa: Offensive EPA per play
            def_epa: Defensive EPA per play (negative is better)
            st_epa: Special teams EPA

        Returns:
            Combined team strength score
        """
        return (
            off_epa * self.offense_weight -
            def_epa * self.defense_weight +
            st_epa * self.special_teams_weight
        ) * self.scaling_factor


@dataclass
class ProbabilityParameters:
    """
    Win probability model parameters.

    Attributes:
        model_type: Type of probability model ("logistic" or "linear")
        spread_stdev: Standard deviation of spread for probability
        home_baseline: Baseline home win probability at spread=0
    """
    model_type: str = "logistic"
    spread_stdev: float = 13.5
    home_baseline: float = 0.57


@dataclass
class ModelParameters:
    """
    Complete model parameter set.

    This dataclass contains all configurable parameters for the
    NFL prediction model.

    Attributes:
        decay: Decay curve parameters
        home_field: Home field advantage parameters
        epa: EPA scaling parameters
        probability: Win probability parameters
        version: Parameter version for tracking
    """
    decay: DecayParameters = field(default_factory=DecayParameters)
    home_field: HomeFieldParameters = field(default_factory=HomeFieldParameters)
    epa: EPAParameters = field(default_factory=EPAParameters)
    probability: ProbabilityParameters = field(default_factory=ProbabilityParameters)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "version": self.version,
            "decay": {
                "initial_preseason_weight": self.decay.initial_preseason_weight,
                "decay_rate": self.decay.decay_rate,
                "min_preseason_weight": self.decay.min_preseason_weight,
                "transition_week": self.decay.transition_week,
            },
            "home_field": {
                "base_hfa": self.home_field.base_hfa,
                "dome_bonus": self.home_field.dome_bonus,
                "altitude_bonus": self.home_field.altitude_bonus,
                "division_reduction": self.home_field.division_reduction,
                "travel_per_timezone": self.home_field.travel_per_timezone,
            },
            "epa": {
                "offense_weight": self.epa.offense_weight,
                "defense_weight": self.epa.defense_weight,
                "special_teams_weight": self.epa.special_teams_weight,
                "scaling_factor": self.epa.scaling_factor,
            },
            "probability": {
                "model_type": self.probability.model_type,
                "spread_stdev": self.probability.spread_stdev,
                "home_baseline": self.probability.home_baseline,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParameters':
        """Create parameters from dictionary."""
        params = cls()

        if "decay" in data:
            d = data["decay"]
            params.decay = DecayParameters(
                initial_preseason_weight=d.get("initial_preseason_weight", 0.8),
                decay_rate=d.get("decay_rate", 0.15),
                min_preseason_weight=d.get("min_preseason_weight", 0.1),
                transition_week=d.get("transition_week", 8),
            )

        if "home_field" in data:
            h = data["home_field"]
            params.home_field = HomeFieldParameters(
                base_hfa=h.get("base_hfa", 2.5),
                dome_bonus=h.get("dome_bonus", 0.5),
                altitude_bonus=h.get("altitude_bonus", 1.0),
                division_reduction=h.get("division_reduction", 0.3),
                travel_per_timezone=h.get("travel_per_timezone", 0.2),
            )

        if "epa" in data:
            e = data["epa"]
            params.epa = EPAParameters(
                offense_weight=e.get("offense_weight", 1.0),
                defense_weight=e.get("defense_weight", 1.0),
                special_teams_weight=e.get("special_teams_weight", 0.3),
                scaling_factor=e.get("scaling_factor", 3.5),
            )

        if "probability" in data:
            p = data["probability"]
            params.probability = ProbabilityParameters(
                model_type=p.get("model_type", "logistic"),
                spread_stdev=p.get("spread_stdev", 13.5),
                home_baseline=p.get("home_baseline", 0.57),
            )

        params.version = data.get("version", "1.0.0")
        return params

    def save_to_file(self, filepath: str) -> None:
        """Save parameters to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ModelParameters':
        """Load parameters from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Default parameter instance
DEFAULT_PARAMETERS = ModelParameters()
