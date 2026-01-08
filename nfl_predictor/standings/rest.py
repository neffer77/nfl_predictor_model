"""
Rest Analysis - Bye week and rest advantage tracking.

This module contains:
- RestAnalysis: Analyze rest advantages
- ByeWeekTracker: Track and manage bye weeks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta

from nfl_predictor.constants import NFL_TEAMS


@dataclass
class RestAdvantage:
    """
    Rest advantage calculation for a game.

    Attributes:
        home_days_rest: Days of rest for home team
        away_days_rest: Days of rest for away team
        advantage: Rest advantage (positive favors home)
        category: Category of rest differential
    """
    home_days_rest: int
    away_days_rest: int
    advantage: int = 0
    category: str = ""

    def __post_init__(self):
        """Calculate derived fields."""
        self.advantage = self.home_days_rest - self.away_days_rest
        self._categorize()

    def _categorize(self) -> None:
        """Categorize the rest differential."""
        if self.advantage >= 4:
            self.category = "major_home_advantage"
        elif self.advantage >= 2:
            self.category = "home_advantage"
        elif self.advantage <= -4:
            self.category = "major_away_advantage"
        elif self.advantage <= -2:
            self.category = "away_advantage"
        else:
            self.category = "neutral"

    def get_spread_adjustment(self) -> float:
        """
        Get spread adjustment based on rest.

        Returns:
            Spread adjustment in points (negative favors home)
        """
        # Approximately 0.5 points per day of rest advantage
        return -self.advantage * 0.5


@dataclass
class TeamScheduleInfo:
    """
    Schedule information for a team.

    Attributes:
        team: Team abbreviation
        bye_week: Bye week number
        last_game_date: Date of last game
        next_game_date: Date of next game
        short_weeks: Number of short weeks (Thu games after Sun)
        monday_games: Number of Monday night games
        thursday_games: Number of Thursday games
    """
    team: str
    bye_week: Optional[int] = None
    last_game_date: Optional[datetime] = None
    next_game_date: Optional[datetime] = None
    short_weeks: int = 0
    monday_games: int = 0
    thursday_games: int = 0
    primetime_games: int = 0


class RestAnalysis:
    """
    Analyze rest advantages for games.

    This class tracks rest days between games and calculates
    rest advantages for matchups.

    Usage:
        analyzer = RestAnalysis()
        advantage = analyzer.calculate_advantage("KC", "LV", week=5)
    """

    # Standard days between games
    STANDARD_REST = 7  # Sunday to Sunday

    def __init__(self):
        """Initialize the analyzer."""
        self._team_rest: Dict[str, int] = {}
        self._last_game_dates: Dict[str, datetime] = {}

    def set_days_rest(self, team: str, days: int) -> None:
        """Set days of rest for a team."""
        self._team_rest[team] = days

    def get_days_rest(self, team: str) -> int:
        """Get days of rest for a team, default to standard."""
        return self._team_rest.get(team, self.STANDARD_REST)

    def calculate_advantage(
        self,
        home_team: str,
        away_team: str,
        week: Optional[int] = None
    ) -> RestAdvantage:
        """
        Calculate rest advantage for a matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week number (optional)

        Returns:
            RestAdvantage with calculated values
        """
        home_rest = self.get_days_rest(home_team)
        away_rest = self.get_days_rest(away_team)

        return RestAdvantage(
            home_days_rest=home_rest,
            away_days_rest=away_rest
        )

    def update_from_schedule(
        self,
        schedule: List[Dict],
        week: int
    ) -> None:
        """
        Update rest information from schedule.

        Args:
            schedule: List of game dictionaries
            week: Current week number
        """
        # Find previous week games for each team
        prev_week_games = [g for g in schedule if g.get("week") == week - 1]

        for game in prev_week_games:
            home = game.get("home_team")
            away = game.get("away_team")
            game_date = game.get("date")

            if game_date and home:
                # Calculate days until current week (assume Sunday)
                days_rest = 7  # Default

                day_of_week = game.get("day_of_week", "Sunday")
                if day_of_week == "Thursday":
                    days_rest = 10
                elif day_of_week == "Monday":
                    days_rest = 6

                self.set_days_rest(home, days_rest)

            if game_date and away:
                days_rest = 7
                day_of_week = game.get("day_of_week", "Sunday")
                if day_of_week == "Thursday":
                    days_rest = 10
                elif day_of_week == "Monday":
                    days_rest = 6

                self.set_days_rest(away, days_rest)


class ByeWeekTracker:
    """
    Track and manage bye weeks across the season.

    This class tracks which teams have byes in each week
    and provides utilities for bye-related analysis.

    Usage:
        tracker = ByeWeekTracker()
        tracker.set_bye_week("KC", 6)
        teams_on_bye = tracker.get_bye_teams(week=6)
    """

    # Typical bye week range
    MIN_BYE_WEEK = 5
    MAX_BYE_WEEK = 14

    def __init__(self):
        """Initialize the tracker."""
        self._team_byes: Dict[str, int] = {}
        self._week_byes: Dict[int, Set[str]] = {}

    def set_bye_week(self, team: str, week: int) -> None:
        """
        Set bye week for a team.

        Args:
            team: Team abbreviation
            week: Bye week number
        """
        # Remove from old week if exists
        old_week = self._team_byes.get(team)
        if old_week and old_week in self._week_byes:
            self._week_byes[old_week].discard(team)

        # Set new bye week
        self._team_byes[team] = week

        if week not in self._week_byes:
            self._week_byes[week] = set()
        self._week_byes[week].add(team)

    def get_bye_week(self, team: str) -> Optional[int]:
        """Get bye week for a team."""
        return self._team_byes.get(team)

    def get_bye_teams(self, week: int) -> Set[str]:
        """Get all teams on bye for a week."""
        return self._week_byes.get(week, set())

    def is_on_bye(self, team: str, week: int) -> bool:
        """Check if team is on bye for a week."""
        return self._team_byes.get(team) == week

    def is_post_bye(self, team: str, week: int) -> bool:
        """Check if team is coming off bye."""
        bye_week = self._team_byes.get(team)
        return bye_week is not None and week == bye_week + 1

    def get_post_bye_teams(self, week: int) -> Set[str]:
        """Get all teams coming off bye for a week."""
        return self.get_bye_teams(week - 1)

    def load_from_schedule(self, schedule: List[Dict]) -> None:
        """
        Load bye weeks from schedule data.

        Args:
            schedule: List of game dictionaries
        """
        # Find which weeks each team plays
        team_weeks: Dict[str, Set[int]] = {team: set() for team in NFL_TEAMS}

        for game in schedule:
            week = game.get("week")
            home = game.get("home_team")
            away = game.get("away_team")

            if week and home in team_weeks:
                team_weeks[home].add(week)
            if week and away in team_weeks:
                team_weeks[away].add(week)

        # Find bye week (gap in schedule during bye week range)
        for team, weeks in team_weeks.items():
            for bye_candidate in range(self.MIN_BYE_WEEK, self.MAX_BYE_WEEK + 1):
                if bye_candidate not in weeks:
                    self.set_bye_week(team, bye_candidate)
                    break

    def get_schedule_info(self, team: str) -> TeamScheduleInfo:
        """
        Get full schedule info for a team.

        Args:
            team: Team abbreviation

        Returns:
            TeamScheduleInfo with bye and schedule data
        """
        return TeamScheduleInfo(
            team=team,
            bye_week=self.get_bye_week(team)
        )

    def get_bye_summary(self) -> str:
        """Generate bye week summary report."""
        lines = [
            "BYE WEEK SUMMARY",
            "=" * 40,
            "",
        ]

        for week in range(self.MIN_BYE_WEEK, self.MAX_BYE_WEEK + 1):
            teams = self.get_bye_teams(week)
            if teams:
                teams_str = ", ".join(sorted(teams))
                lines.append(f"Week {week:2d}: {teams_str}")

        return "\n".join(lines)
