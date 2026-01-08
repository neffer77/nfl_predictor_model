"""
Standings package - NFL standings tracking and playoff calculations.

This package contains:
- engine.py: Standings calculation engine
- playoffs.py: Playoff probability calculations
- rest.py: Rest and bye week analysis
"""

from nfl_predictor.standings.engine import (
    TeamStanding,
    DivisionStandings,
    ConferenceStandings,
    StandingsEngine,
)
from nfl_predictor.standings.playoffs import (
    PlayoffScenario,
    PlayoffProbabilityCalculator,
)
from nfl_predictor.standings.rest import (
    RestAnalysis,
    ByeWeekTracker,
)

__all__ = [
    "TeamStanding",
    "DivisionStandings",
    "ConferenceStandings",
    "StandingsEngine",
    "PlayoffScenario",
    "PlayoffProbabilityCalculator",
    "RestAnalysis",
    "ByeWeekTracker",
]
