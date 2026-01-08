"""
Base Data Sources - Abstract base classes for data source implementations.

This module defines the interface that all data sources must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from datetime import date


class DataSource(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        """Fetch data from the source."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        pass


class EPADataSource(DataSource):
    """Abstract base class for EPA data sources."""

    @abstractmethod
    def fetch_team_epa(self, team: str, season: int, week: int) -> Optional[Dict]:
        """
        Fetch EPA data for a specific team.

        Args:
            team: Team name or abbreviation
            season: NFL season year
            week: Week number

        Returns:
            Dict with EPA metrics or None if not available
        """
        pass

    @abstractmethod
    def fetch_all_teams_epa(self, season: int, week: int) -> Dict[str, Dict]:
        """
        Fetch EPA data for all teams.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            Dict mapping team name to EPA metrics
        """
        pass


class ScheduleDataSource(DataSource):
    """Abstract base class for schedule data sources."""

    @abstractmethod
    def fetch_week_schedule(self, season: int, week: int) -> List[Dict]:
        """
        Fetch schedule for a specific week.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            List of game dictionaries
        """
        pass

    @abstractmethod
    def fetch_game_result(self, game_id: str) -> Optional[Dict]:
        """
        Fetch result for a specific game.

        Args:
            game_id: Game identifier

        Returns:
            Dict with game result or None if not completed
        """
        pass


class VegasLinesDataSource(DataSource):
    """Abstract base class for Vegas lines data sources."""

    @abstractmethod
    def fetch_game_lines(self, game_id: str) -> Optional[Dict]:
        """
        Fetch Vegas lines for a specific game.

        Args:
            game_id: Game identifier

        Returns:
            Dict with spread, total, moneylines or None
        """
        pass

    @abstractmethod
    def fetch_week_lines(self, season: int, week: int) -> Dict[str, Dict]:
        """
        Fetch Vegas lines for all games in a week.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            Dict mapping game_id to lines
        """
        pass


class WeatherDataSource(DataSource):
    """Abstract base class for weather data sources."""

    @abstractmethod
    def fetch_game_weather(
        self,
        stadium: str,
        game_date: date,
        game_time: str
    ) -> Optional[Dict]:
        """
        Fetch weather for a specific game.

        Args:
            stadium: Stadium name or team
            game_date: Date of game
            game_time: Scheduled game time

        Returns:
            Dict with weather conditions or None
        """
        pass


class InjuryDataSource(DataSource):
    """Abstract base class for injury data sources."""

    @abstractmethod
    def fetch_team_injuries(self, team: str, week: int) -> List[Dict]:
        """
        Fetch injury report for a team.

        Args:
            team: Team name or abbreviation
            week: Week number

        Returns:
            List of player injury dictionaries
        """
        pass

    @abstractmethod
    def fetch_all_injuries(self, week: int) -> Dict[str, List[Dict]]:
        """
        Fetch injury reports for all teams.

        Args:
            week: Week number

        Returns:
            Dict mapping team to injury list
        """
        pass
