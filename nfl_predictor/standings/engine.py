"""
Standings Engine - NFL standings calculation and tracking.

This module contains:
- TeamStanding: Individual team record and tiebreakers
- DivisionStandings: Division standings with rankings
- ConferenceStandings: Conference standings
- StandingsEngine: Full standings calculation engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from nfl_predictor.constants import NFL_DIVISIONS, NFL_CONFERENCES


@dataclass
class TeamStanding:
    """
    Individual team standing with record and tiebreaker info.

    Attributes:
        team: Team abbreviation
        wins: Total wins
        losses: Total losses
        ties: Total ties
        division: Division name
        conference: Conference name
        division_wins: Wins within division
        division_losses: Losses within division
        conference_wins: Wins within conference
        conference_losses: Losses within conference
        points_for: Total points scored
        points_against: Total points allowed
        strength_of_victory: Combined opponent wins for games won
        strength_of_schedule: Combined opponent wins for all games
    """
    team: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    division: str = ""
    conference: str = ""

    # Division record
    division_wins: int = 0
    division_losses: int = 0
    division_ties: int = 0

    # Conference record
    conference_wins: int = 0
    conference_losses: int = 0
    conference_ties: int = 0

    # Points
    points_for: int = 0
    points_against: int = 0

    # Tiebreaker stats
    strength_of_victory: float = 0.0
    strength_of_schedule: float = 0.0

    # Head-to-head results
    head_to_head: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    # team -> (wins, losses, ties)

    @property
    def win_pct(self) -> float:
        """Calculate win percentage."""
        total = self.wins + self.losses + self.ties
        if total == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / total

    @property
    def division_win_pct(self) -> float:
        """Calculate division win percentage."""
        total = self.division_wins + self.division_losses + self.division_ties
        if total == 0:
            return 0.0
        return (self.division_wins + 0.5 * self.division_ties) / total

    @property
    def conference_win_pct(self) -> float:
        """Calculate conference win percentage."""
        total = self.conference_wins + self.conference_losses + self.conference_ties
        if total == 0:
            return 0.0
        return (self.conference_wins + 0.5 * self.conference_ties) / total

    @property
    def point_differential(self) -> int:
        """Calculate point differential."""
        return self.points_for - self.points_against

    @property
    def games_played(self) -> int:
        """Total games played."""
        return self.wins + self.losses + self.ties

    @property
    def record_str(self) -> str:
        """Formatted record string."""
        if self.ties > 0:
            return f"{self.wins}-{self.losses}-{self.ties}"
        return f"{self.wins}-{self.losses}"

    def add_game_result(
        self,
        opponent: str,
        won: bool,
        tied: bool,
        points_for: int,
        points_against: int,
        is_division: bool,
        is_conference: bool
    ) -> None:
        """
        Add a game result to the standing.

        Args:
            opponent: Opponent team abbreviation
            won: Whether team won
            tied: Whether game was a tie
            points_for: Points scored
            points_against: Points allowed
            is_division: Whether opponent is in same division
            is_conference: Whether opponent is in same conference
        """
        self.points_for += points_for
        self.points_against += points_against

        if tied:
            self.ties += 1
            if is_division:
                self.division_ties += 1
            if is_conference:
                self.conference_ties += 1
        elif won:
            self.wins += 1
            if is_division:
                self.division_wins += 1
            if is_conference:
                self.conference_wins += 1
        else:
            self.losses += 1
            if is_division:
                self.division_losses += 1
            if is_conference:
                self.conference_losses += 1

        # Update head-to-head
        if opponent not in self.head_to_head:
            self.head_to_head[opponent] = (0, 0, 0)
        w, l, t = self.head_to_head[opponent]
        if tied:
            self.head_to_head[opponent] = (w, l, t + 1)
        elif won:
            self.head_to_head[opponent] = (w + 1, l, t)
        else:
            self.head_to_head[opponent] = (w, l + 1, t)


@dataclass
class DivisionStandings:
    """
    Division standings with ranked teams.

    Attributes:
        division: Division name
        conference: Conference name
        teams: List of TeamStanding objects, sorted by rank
    """
    division: str
    conference: str
    teams: List[TeamStanding] = field(default_factory=list)

    def get_leader(self) -> Optional[TeamStanding]:
        """Get division leader."""
        return self.teams[0] if self.teams else None

    def get_rankings(self) -> List[Tuple[int, TeamStanding]]:
        """Get ranked list of teams."""
        return [(i + 1, team) for i, team in enumerate(self.teams)]


@dataclass
class ConferenceStandings:
    """
    Conference standings including wild card.

    Attributes:
        conference: Conference name (AFC/NFC)
        divisions: Dict of division name -> DivisionStandings
        wild_card_teams: Teams in wild card positions
    """
    conference: str
    divisions: Dict[str, DivisionStandings] = field(default_factory=dict)
    wild_card_teams: List[TeamStanding] = field(default_factory=list)

    def get_playoff_teams(self) -> List[TeamStanding]:
        """
        Get all playoff teams in seed order.

        Returns:
            List of 7 playoff teams (4 division winners + 3 wild cards)
        """
        # Get division winners
        division_winners = []
        for div_standings in self.divisions.values():
            leader = div_standings.get_leader()
            if leader:
                division_winners.append(leader)

        # Sort division winners by record
        division_winners.sort(key=lambda t: (-t.win_pct, -t.point_differential))

        # Combine with wild cards
        return division_winners + self.wild_card_teams[:3]


class StandingsEngine:
    """
    Full standings calculation engine.

    Handles standings calculation, tiebreakers, and playoff seeding
    according to NFL rules.

    Usage:
        engine = StandingsEngine()
        engine.add_game_result("KC", "LV", 24, 17, season=2025, week=5)
        standings = engine.get_standings()
    """

    def __init__(self):
        """Initialize the standings engine."""
        self._standings: Dict[str, TeamStanding] = {}
        self._initialize_teams()

    def _initialize_teams(self) -> None:
        """Initialize all teams with empty standings."""
        for conference, divisions in NFL_DIVISIONS.items():
            for division, teams in divisions.items():
                for team in teams:
                    self._standings[team] = TeamStanding(
                        team=team,
                        division=division,
                        conference=conference
                    )

    def add_game_result(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        season: int,
        week: int
    ) -> None:
        """
        Add a game result to standings.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_score: Home team score
            away_score: Away team score
            season: Season year
            week: Week number
        """
        if home_team not in self._standings or away_team not in self._standings:
            return

        home = self._standings[home_team]
        away = self._standings[away_team]

        # Determine game outcome
        tied = home_score == away_score
        home_won = home_score > away_score

        # Determine division/conference relationships
        same_division = home.division == away.division
        same_conference = home.conference == away.conference

        # Update home team
        home.add_game_result(
            opponent=away_team,
            won=home_won,
            tied=tied,
            points_for=home_score,
            points_against=away_score,
            is_division=same_division,
            is_conference=same_conference
        )

        # Update away team
        away.add_game_result(
            opponent=home_team,
            won=not home_won and not tied,
            tied=tied,
            points_for=away_score,
            points_against=home_score,
            is_division=same_division,
            is_conference=same_conference
        )

    def get_team_standing(self, team: str) -> Optional[TeamStanding]:
        """Get standing for a specific team."""
        return self._standings.get(team)

    def get_division_standings(self, division: str) -> DivisionStandings:
        """
        Get standings for a division.

        Args:
            division: Division name (e.g., "AFC West")

        Returns:
            DivisionStandings with teams sorted by rank
        """
        # Find teams in this division
        teams = [s for s in self._standings.values() if s.division == division]

        # Sort by NFL tiebreaker rules
        teams = self._sort_teams(teams, division_tiebreaker=True)

        # Determine conference
        conference = teams[0].conference if teams else ""

        return DivisionStandings(
            division=division,
            conference=conference,
            teams=teams
        )

    def get_conference_standings(self, conference: str) -> ConferenceStandings:
        """
        Get full conference standings including wild card.

        Args:
            conference: Conference name (AFC/NFC)

        Returns:
            ConferenceStandings with divisions and wild card
        """
        standings = ConferenceStandings(conference=conference)

        # Get division standings
        divisions = NFL_DIVISIONS.get(conference, {})
        for division in divisions:
            div_standings = self.get_division_standings(division)
            standings.divisions[division] = div_standings

        # Calculate wild card
        wild_card_candidates = []
        for div_standings in standings.divisions.values():
            # Add non-division-winners
            for team in div_standings.teams[1:]:
                wild_card_candidates.append(team)

        # Sort wild card candidates
        wild_card_candidates = self._sort_teams(
            wild_card_candidates,
            division_tiebreaker=False
        )
        standings.wild_card_teams = wild_card_candidates[:3]

        return standings

    def _sort_teams(
        self,
        teams: List[TeamStanding],
        division_tiebreaker: bool
    ) -> List[TeamStanding]:
        """
        Sort teams according to NFL tiebreaker rules.

        Args:
            teams: List of teams to sort
            division_tiebreaker: Whether to use division tiebreaker rules

        Returns:
            Sorted list of teams
        """
        def sort_key(team: TeamStanding) -> tuple:
            """Generate sort key for team."""
            return (
                -team.win_pct,  # Higher win pct first
                -team.division_win_pct if division_tiebreaker else -team.conference_win_pct,
                -team.point_differential,
                -team.points_for,
                team.team  # Alphabetical as final tiebreaker
            )

        return sorted(teams, key=sort_key)

    def get_standings_report(self) -> str:
        """Generate formatted standings report."""
        lines = [
            "=" * 70,
            "NFL STANDINGS",
            "=" * 70,
            "",
        ]

        for conference in ["AFC", "NFC"]:
            lines.append(f"\n{conference}")
            lines.append("-" * 70)

            conf_standings = self.get_conference_standings(conference)

            for division, div_standings in sorted(conf_standings.divisions.items()):
                lines.append(f"\n{division}")
                lines.append(f"{'Team':<6} {'W-L-T':<10} {'Pct':<8} {'PF':<6} {'PA':<6} {'Diff':<6}")
                lines.append("-" * 50)

                for team in div_standings.teams:
                    lines.append(
                        f"{team.team:<6} {team.record_str:<10} "
                        f"{team.win_pct:.3f}   {team.points_for:<6} "
                        f"{team.points_against:<6} {team.point_differential:+d}"
                    )

            # Wild card
            lines.append(f"\n{conference} Wild Card")
            lines.append("-" * 50)
            for i, team in enumerate(conf_standings.wild_card_teams, 1):
                lines.append(
                    f"  {i}. {team.team:<4} {team.record_str:<10} ({team.division})"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all standings to empty."""
        self._initialize_teams()
