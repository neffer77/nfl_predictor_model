"""
Predictions package - Core prediction engines for NFL games.

This package contains:
- ratings.py: Team rating calculations and season model
- spreads.py: Spread prediction with HFA and rest/travel adjustments
- probability.py: Win probability models
- weather.py: Weather impact adjustments
- injuries.py: Injury impact adjustments
- situational.py: Situational/motivation adjustments
- engine.py: Main prediction orchestration
"""

from nfl_predictor.predictions.ratings import (
    NFLSeasonModel,
    TeamRatingCalculator,
    PreseasonDecayCurve,
    DecayCurveConfig,
    DecayCurveType,
    RatingValidator,
)
from nfl_predictor.predictions.spreads import (
    SpreadPredictor,
    HomeFieldAdvantageCalculator,
    HomeFieldAdvantageConfig,
    RestTravelAdjuster,
    RestTravelConfig,
    WeeklyScheduleManager,
)
from nfl_predictor.predictions.probability import (
    WinProbabilityModel,
    LogisticWinProbability,
    LinearWinProbability,
)

__all__ = [
    # Ratings
    "NFLSeasonModel",
    "TeamRatingCalculator",
    "PreseasonDecayCurve",
    "DecayCurveConfig",
    "DecayCurveType",
    "RatingValidator",
    # Spreads
    "SpreadPredictor",
    "HomeFieldAdvantageCalculator",
    "HomeFieldAdvantageConfig",
    "RestTravelAdjuster",
    "RestTravelConfig",
    "WeeklyScheduleManager",
    # Probability
    "WinProbabilityModel",
    "LogisticWinProbability",
    "LinearWinProbability",
]
