"""
NFL Predictor - A comprehensive NFL game prediction system.

This package provides tools for:
- Team power ratings blending preseason projections with in-season EPA data
- Game spread and win probability predictions
- Weather, injury, and situational adjustments
- Standings tracking and playoff probability calculations
- Backtesting and model calibration
- Multi-format export capabilities

Example usage:
    from nfl_predictor import NFLSeasonModel, SpreadPredictor

    model = NFLSeasonModel(season=2025)
    model.load_preseason_from_wins(projected_wins)

    predictor = SpreadPredictor(model)
    prediction = predictor.predict_game(game, week=5)
"""

__version__ = "1.0.0"
__author__ = "Connor's NFL Prediction System"

# Core models
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

# Prediction engines
from nfl_predictor.predictions.ratings import (
    NFLSeasonModel,
    TeamRatingCalculator,
    PreseasonDecayCurve,
    DecayCurveConfig,
)
from nfl_predictor.predictions.spreads import (
    SpreadPredictor,
    HomeFieldAdvantageCalculator,
    RestTravelAdjuster,
)
from nfl_predictor.predictions.probability import (
    LogisticWinProbability,
    LinearWinProbability,
)

# Analysis and backtesting
from nfl_predictor.analysis.backtesting import (
    PredictionResult,
    PredictionResultTracker,
    SeasonBacktester,
)
from nfl_predictor.analysis.metrics import (
    AccuracyMetrics,
    CalibrationAnalysis,
)

# Standings
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

# Data sources and pipeline
from nfl_predictor.data.pipeline import (
    DataPipeline,
    WeeklyAutomationManager,
)
from nfl_predictor.data.stadiums import (
    StadiumInfo,
    STADIUM_DATA,
)

# Export formatters
from nfl_predictor.export.formatters import (
    CSVFormatter,
    JSONFormatter,
    MarkdownFormatter,
)
from nfl_predictor.export.reports import (
    WeeklyReport,
    SeasonReport,
    ReportGenerator,
)

# Configuration
from nfl_predictor.config.parameters import (
    ModelParameters,
    DecayParameters,
    HomeFieldParameters,
    EPAParameters,
    DEFAULT_PARAMETERS,
)
from nfl_predictor.config.settings import (
    Settings,
    get_settings,
)

# Utilities
from nfl_predictor.utils.validators import (
    validate_team,
    validate_week,
    validate_season,
    ValidationError,
)
from nfl_predictor.utils.helpers import (
    get_current_week,
    format_spread,
    format_probability,
    is_division_game,
    is_conference_game,
)

# Constants
from nfl_predictor.constants import (
    NFL_TEAMS,
    ALL_NFL_TEAMS,
    NFL_DIVISIONS,
    NFL_CONFERENCES,
    TEAM_TO_DIVISION,
    TEAM_TO_CONFERENCE,
    TEAM_ABBREVIATIONS,
    NFL_RIVALRIES,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "TeamPreseasonData",
    "TeamInSeasonData",
    "BlendedTeamRating",
    "NFLGame",
    "GameType",
    "GameLocation",
    "TeamGameContext",
    "GamePrediction",
    # Prediction engines
    "NFLSeasonModel",
    "TeamRatingCalculator",
    "PreseasonDecayCurve",
    "DecayCurveConfig",
    "SpreadPredictor",
    "HomeFieldAdvantageCalculator",
    "RestTravelAdjuster",
    "LogisticWinProbability",
    "LinearWinProbability",
    # Analysis
    "PredictionResult",
    "PredictionResultTracker",
    "SeasonBacktester",
    "AccuracyMetrics",
    "CalibrationAnalysis",
    # Standings
    "TeamStanding",
    "DivisionStandings",
    "ConferenceStandings",
    "StandingsEngine",
    "PlayoffScenario",
    "PlayoffProbabilityCalculator",
    "RestAnalysis",
    "ByeWeekTracker",
    # Data
    "DataPipeline",
    "WeeklyAutomationManager",
    "StadiumInfo",
    "STADIUM_DATA",
    # Export
    "CSVFormatter",
    "JSONFormatter",
    "MarkdownFormatter",
    "WeeklyReport",
    "SeasonReport",
    "ReportGenerator",
    # Config
    "ModelParameters",
    "DecayParameters",
    "HomeFieldParameters",
    "EPAParameters",
    "DEFAULT_PARAMETERS",
    "Settings",
    "get_settings",
    # Utils
    "validate_team",
    "validate_week",
    "validate_season",
    "ValidationError",
    "get_current_week",
    "format_spread",
    "format_probability",
    "is_division_game",
    "is_conference_game",
    # Constants
    "NFL_TEAMS",
    "ALL_NFL_TEAMS",
    "NFL_DIVISIONS",
    "NFL_CONFERENCES",
    "TEAM_TO_DIVISION",
    "TEAM_TO_CONFERENCE",
    "TEAM_ABBREVIATIONS",
    "NFL_RIVALRIES",
]
