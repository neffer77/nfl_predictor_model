"""
Data Pipeline - Orchestration of data fetching and processing.

This module contains:
- DataPipeline: Orchestrates data fetching from multiple sources
- WeeklyAutomationManager: Manages weekly data updates
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from abc import ABC

from nfl_predictor.data.sources.base import (
    EPADataSource,
    ScheduleDataSource,
    VegasLinesDataSource,
)


@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline."""
    cache_enabled: bool = True
    cache_ttl_minutes: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 5


class DataPipeline:
    """
    Orchestrates data fetching from multiple sources.

    This class coordinates fetching EPA, schedule, Vegas lines,
    and other data needed for predictions.

    Usage:
        pipeline = DataPipeline()
        pipeline.register_source("epa", epa_fetcher)
        data = pipeline.fetch_week_data(season=2025, week=5)
    """

    def __init__(self, config: Optional[DataPipelineConfig] = None):
        """Initialize the pipeline."""
        self.config = config or DataPipelineConfig()
        self._sources: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def register_source(self, name: str, source: Any) -> None:
        """
        Register a data source.

        Args:
            name: Source identifier (e.g., "epa", "schedule", "vegas")
            source: Data source instance
        """
        self._sources[name] = source

    def get_source(self, name: str) -> Optional[Any]:
        """Get a registered data source."""
        return self._sources.get(name)

    def fetch_week_data(self, season: int, week: int) -> Dict[str, Any]:
        """
        Fetch all data for a specific week.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            Dict with all fetched data
        """
        data = {
            "season": season,
            "week": week,
            "timestamp": datetime.now(),
        }

        # Fetch from each registered source
        if "epa" in self._sources:
            try:
                data["epa"] = self._sources["epa"].fetch_all_teams_epa(season, week)
            except Exception as e:
                data["epa_error"] = str(e)

        if "schedule" in self._sources:
            try:
                data["schedule"] = self._sources["schedule"].fetch_week_schedule(season, week)
            except Exception as e:
                data["schedule_error"] = str(e)

        if "vegas" in self._sources:
            try:
                data["vegas"] = self._sources["vegas"].fetch_week_lines(season, week)
            except Exception as e:
                data["vegas_error"] = str(e)

        return data

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache = {}
        self._cache_timestamps = {}


class WeeklyAutomationManager:
    """
    Manages automated weekly data updates.

    This class handles scheduling and running weekly updates
    to fetch latest data and regenerate predictions.

    Usage:
        manager = WeeklyAutomationManager(pipeline)
        manager.run_weekly_update(season=2025, week=5)
    """

    def __init__(self, pipeline: DataPipeline):
        """Initialize with a data pipeline."""
        self.pipeline = pipeline
        self._update_history: List[Dict] = []

    def run_weekly_update(self, season: int, week: int) -> Dict:
        """
        Run a complete weekly update.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            Dict with update results
        """
        start_time = datetime.now()

        result = {
            "season": season,
            "week": week,
            "start_time": start_time,
            "status": "started",
        }

        try:
            # Fetch all data
            data = self.pipeline.fetch_week_data(season, week)
            result["data"] = data
            result["status"] = "completed"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        result["end_time"] = datetime.now()
        result["duration_seconds"] = (result["end_time"] - start_time).total_seconds()

        self._update_history.append(result)
        return result

    def get_update_history(self) -> List[Dict]:
        """Get history of all updates."""
        return self._update_history.copy()

    def get_last_update(self) -> Optional[Dict]:
        """Get the most recent update."""
        return self._update_history[-1] if self._update_history else None
