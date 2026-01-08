"""
Data Sources - Pluggable data source implementations.

This package contains:
- base.py: Abstract base classes for data sources
- epa.py: EPA data fetcher
- schedule.py: Schedule and results fetcher
- vegas.py: Vegas lines fetcher
- weather.py: Weather data source
- injuries.py: Injury data source
"""

from nfl_predictor.data.sources.base import (
    DataSource,
    EPADataSource,
    ScheduleDataSource,
    VegasLinesDataSource,
    WeatherDataSource,
    InjuryDataSource,
)

__all__ = [
    "DataSource",
    "EPADataSource",
    "ScheduleDataSource",
    "VegasLinesDataSource",
    "WeatherDataSource",
    "InjuryDataSource",
]
