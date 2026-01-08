"""
Playoffs - Playoff probability calculations.

This module contains:
- PlayoffScenario: Individual playoff scenario
- PlayoffProbabilityCalculator: Monte Carlo playoff probability simulation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import random

from nfl_predictor.standings.engine import StandingsEngine, TeamStanding


@dataclass
class PlayoffScenario:
    """
    Individual playoff scenario result.

    Attributes:
        seed: Playoff seed (1-7)
        division_winner: Whether team won division
        wild_card: Whether team is wild card
        bye_week: Whether team has first-round bye (seed 1)
        home_games: Expected home playoff games
    """
    team: str
    seed: int
    division_winner: bool = False
    wild_card: bool = False
    bye_week: bool = False
    home_games: int = 0

    @classmethod
    def from_seed(cls, team: str, seed: int) -> 'PlayoffScenario':
        """Create scenario from seed number."""
        return cls(
            team=team,
            seed=seed,
            division_winner=seed <= 4,
            wild_card=seed > 4,
            bye_week=seed == 1,
            home_games=max(0, 4 - seed) if seed <= 4 else 0
        )


@dataclass
class TeamPlayoffOdds:
    """
    Playoff odds for a single team.

    Attributes:
        team: Team abbreviation
        make_playoffs: Probability of making playoffs
        win_division: Probability of winning division
        get_bye: Probability of getting bye (seed 1)
        seed_distribution: Probability distribution over seeds
        simulations: Number of simulations run
    """
    team: str
    make_playoffs: float = 0.0
    win_division: float = 0.0
    get_bye: float = 0.0
    seed_distribution: Dict[int, float] = field(default_factory=dict)
    simulations: int = 0

    def get_report(self) -> str:
        """Generate formatted odds report."""
        lines = [
            f"{self.team} Playoff Odds",
            "-" * 30,
            f"Make Playoffs: {self.make_playoffs:.1%}",
            f"Win Division:  {self.win_division:.1%}",
            f"Get Bye:       {self.get_bye:.1%}",
            "",
            "Seed Distribution:",
        ]

        for seed in range(1, 8):
            prob = self.seed_distribution.get(seed, 0.0)
            if prob > 0:
                lines.append(f"  Seed {seed}: {prob:.1%}")

        return "\n".join(lines)


class PlayoffProbabilityCalculator:
    """
    Calculate playoff probabilities using Monte Carlo simulation.

    This class simulates the remainder of an NFL season many times
    to calculate playoff odds for each team.

    Usage:
        calculator = PlayoffProbabilityCalculator(engine, predictor)
        odds = calculator.calculate_odds(remaining_games, simulations=10000)
    """

    def __init__(
        self,
        standings_engine: StandingsEngine,
        win_probability_model: Optional[any] = None
    ):
        """
        Initialize the calculator.

        Args:
            standings_engine: Current standings engine
            win_probability_model: Model to predict game outcomes
        """
        self.engine = standings_engine
        self.win_model = win_probability_model
        self._simulation_results: List[Dict] = []

    def simulate_game(
        self,
        home_team: str,
        away_team: str,
        home_win_prob: Optional[float] = None
    ) -> tuple:
        """
        Simulate a single game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_win_prob: Pre-calculated win probability

        Returns:
            Tuple of (winner, home_score, away_score)
        """
        if home_win_prob is None:
            home_win_prob = 0.57  # Default home advantage

        # Simulate winner
        home_wins = random.random() < home_win_prob

        # Generate plausible scores
        if home_wins:
            home_score = random.randint(17, 35)
            away_score = random.randint(10, home_score - 1)
        else:
            away_score = random.randint(17, 35)
            home_score = random.randint(10, away_score - 1)

        winner = home_team if home_wins else away_team
        return winner, home_score, away_score

    def run_simulation(
        self,
        remaining_games: List[Dict],
        game_probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, PlayoffScenario]:
        """
        Run a single season simulation.

        Args:
            remaining_games: List of remaining games
                Each dict has: home_team, away_team, week
            game_probabilities: Dict of game_id -> home_win_prob

        Returns:
            Dict of team -> PlayoffScenario for playoff teams
        """
        game_probabilities = game_probabilities or {}

        # Create a copy of the standings engine for simulation
        sim_engine = StandingsEngine()

        # Copy current standings
        for team, standing in self.engine._standings.items():
            sim_engine._standings[team] = TeamStanding(
                team=standing.team,
                wins=standing.wins,
                losses=standing.losses,
                ties=standing.ties,
                division=standing.division,
                conference=standing.conference,
                division_wins=standing.division_wins,
                division_losses=standing.division_losses,
                conference_wins=standing.conference_wins,
                conference_losses=standing.conference_losses,
                points_for=standing.points_for,
                points_against=standing.points_against,
            )

        # Simulate remaining games
        for game in remaining_games:
            home = game["home_team"]
            away = game["away_team"]
            week = game.get("week", 0)

            game_id = f"{away}@{home}_{week}"
            prob = game_probabilities.get(game_id, 0.57)

            winner, home_score, away_score = self.simulate_game(home, away, prob)
            sim_engine.add_game_result(
                home, away, home_score, away_score,
                season=2025, week=week
            )

        # Determine playoff teams
        results = {}
        for conference in ["AFC", "NFC"]:
            conf_standings = sim_engine.get_conference_standings(conference)
            playoff_teams = conf_standings.get_playoff_teams()

            for seed, team in enumerate(playoff_teams, 1):
                results[team.team] = PlayoffScenario.from_seed(team.team, seed)

        return results

    def calculate_odds(
        self,
        remaining_games: List[Dict],
        game_probabilities: Optional[Dict[str, float]] = None,
        simulations: int = 10000
    ) -> Dict[str, TeamPlayoffOdds]:
        """
        Calculate playoff odds using Monte Carlo simulation.

        Args:
            remaining_games: List of remaining games
            game_probabilities: Dict of game_id -> home_win_prob
            simulations: Number of simulations to run

        Returns:
            Dict of team -> TeamPlayoffOdds
        """
        # Initialize odds tracking
        odds: Dict[str, TeamPlayoffOdds] = {}
        for team in self.engine._standings:
            odds[team] = TeamPlayoffOdds(team=team, simulations=simulations)
            odds[team].seed_distribution = {i: 0.0 for i in range(1, 8)}

        # Run simulations
        for _ in range(simulations):
            result = self.run_simulation(remaining_games, game_probabilities)

            for team, scenario in result.items():
                odds[team].make_playoffs += 1
                odds[team].seed_distribution[scenario.seed] += 1

                if scenario.division_winner:
                    odds[team].win_division += 1
                if scenario.bye_week:
                    odds[team].get_bye += 1

        # Convert to probabilities
        for team_odds in odds.values():
            team_odds.make_playoffs /= simulations
            team_odds.win_division /= simulations
            team_odds.get_bye /= simulations

            for seed in team_odds.seed_distribution:
                team_odds.seed_distribution[seed] /= simulations

        return odds

    def get_clinch_scenarios(
        self,
        team: str,
        remaining_games: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Determine clinching scenarios for a team.

        Args:
            team: Team abbreviation
            remaining_games: List of remaining games

        Returns:
            Dict with clinching scenarios
        """
        scenarios = {
            "clinch_playoff": [],
            "clinch_division": [],
            "clinch_bye": [],
            "eliminated": [],
        }

        # Simplified clinch detection
        standing = self.engine.get_team_standing(team)
        if not standing:
            return scenarios

        games_remaining = 17 - standing.games_played
        max_wins = standing.wins + games_remaining
        min_wins = standing.wins

        # Very basic clinch logic (would need more sophisticated analysis)
        if min_wins >= 10:
            scenarios["clinch_playoff"].append("Already clinched with 10+ wins")
        elif max_wins < 7:
            scenarios["eliminated"].append("Cannot reach 7 wins")

        return scenarios
