"""
Data package - Data sources, fetchers, and pipelines.

This package contains:
- sources/: Data source implementations (EPA, schedule, Vegas lines, weather, injuries)
- stadiums.py: Stadium information and coordinates
- team_profiles.py: Team weather profiles
- pipeline.py: Data pipeline orchestration
"""

from nfl_predictor.data.stadiums import StadiumInfo, STADIUM_DATA
from nfl_predictor.data.pipeline import DataPipeline, WeeklyAutomationManager

__all__ = [
    "StadiumInfo",
    "STADIUM_DATA",
    "DataPipeline",
    "WeeklyAutomationManager",
]
