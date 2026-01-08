"""
Models package - Data classes and enums for the NFL prediction system.

This package contains all the core data models:
- teams.py: Team ratings and EPA data
- games.py: Game/matchup models and predictions
- weather.py: Weather conditions and impacts
- injuries.py: Injury reports and player values
- standings.py: Team standings and playoff scenarios
- motivation.py: Motivational factors
- calibration.py: Historical data for calibration
"""

from nfl_predictor.models.teams import (
    TeamPreseasonData,
    TeamInSeasonData,
    BlendedTeamRating,
)
from nfl_predictor.models.games import (
    NFLGame,
    GameType,
    GameLocation,
    TeamGameContext,
    GamePrediction,
)

__all__ = [
    # Team models
    "TeamPreseasonData",
    "TeamInSeasonData",
    "BlendedTeamRating",
    # Game models
    "NFLGame",
    "GameType",
    "GameLocation",
    "TeamGameContext",
    "GamePrediction",
]
