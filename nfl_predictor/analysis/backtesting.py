"""
Backtesting - Track and analyze prediction results.

This module contains:
- PredictionResult: Single game prediction vs actual outcome
- PredictionResultTracker: Tracks results across a season
- SeasonBacktester: Full season backtesting
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from nfl_predictor.models.games import GamePrediction, NFLGame


@dataclass
class PredictionResult:
    """
    Tracks the result of a single prediction after the game is played.

    Attributes:
        prediction: The original GamePrediction
        actual_winner: Team that won
        actual_spread: Actual point spread (away - home)
        actual_total: Total points scored
        su_correct: Whether straight-up pick was correct
        ats_correct: Whether ATS pick was correct
        spread_error: Predicted spread - actual spread
    """
    prediction: GamePrediction

    # Actual results
    actual_winner: str
    actual_spread: float
    actual_total: int

    # SU results
    predicted_winner: str = ""
    su_correct: bool = False

    # ATS results
    vegas_spread: Optional[float] = None
    ats_pick: Optional[str] = None
    ats_correct: Optional[bool] = None
    spread_error: float = 0.0

    # Calibration
    predicted_win_prob: float = 0.5

    # Metadata
    game_id: str = ""
    season: int = 0
    week: int = 0

    def __post_init__(self):
        """Calculate derived fields."""
        game = self.prediction.game

        # Set metadata
        self.game_id = game.game_id
        self.season = game.season
        self.week = game.week

        # Determine predicted winner
        self.predicted_winner = self.prediction.pick

        # Check straight-up correctness
        self.su_correct = (self.predicted_winner == self.actual_winner)

        # Store win probability for the actual winner
        if self.actual_winner == game.home_team:
            self.predicted_win_prob = self.prediction.home_win_probability
        else:
            self.predicted_win_prob = self.prediction.away_win_probability

        # Calculate spread error
        self.spread_error = round(
            self.prediction.predicted_spread - self.actual_spread, 1
        )

        # ATS results
        self.vegas_spread = game.vegas_spread
        self.ats_pick = self.prediction.pick_ats

        if self.vegas_spread is not None and self.ats_pick is not None:
            home_covered = self.actual_spread < self.vegas_spread
            if self.ats_pick == game.home_team:
                self.ats_correct = home_covered
            else:
                self.ats_correct = not home_covered

    @property
    def abs_spread_error(self) -> float:
        """Absolute spread error."""
        return abs(self.spread_error)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "predicted_winner": self.predicted_winner,
            "actual_winner": self.actual_winner,
            "su_correct": self.su_correct,
            "predicted_spread": self.prediction.predicted_spread,
            "actual_spread": self.actual_spread,
            "spread_error": self.spread_error,
            "vegas_spread": self.vegas_spread,
            "ats_pick": self.ats_pick,
            "ats_correct": self.ats_correct,
            "predicted_win_prob": self.predicted_win_prob,
        }


class PredictionResultTracker:
    """
    Tracks and stores prediction results across a season.

    Usage:
        tracker = PredictionResultTracker()
        tracker.add_result(prediction, home_score=24, away_score=17)
        metrics = tracker.get_accuracy_metrics()
    """

    def __init__(self):
        """Initialize the tracker."""
        self._results: List[PredictionResult] = []
        self._by_week: Dict[int, List[PredictionResult]] = {}
        self._by_team: Dict[str, List[PredictionResult]] = {}

    def add_result(
        self,
        prediction: GamePrediction,
        home_score: int,
        away_score: int
    ) -> PredictionResult:
        """
        Add a game result.

        Args:
            prediction: Original GamePrediction
            home_score: Actual home team score
            away_score: Actual away team score

        Returns:
            PredictionResult for the game
        """
        game = prediction.game
        actual_winner = game.home_team if home_score > away_score else game.away_team
        if home_score == away_score:
            actual_winner = "TIE"

        result = PredictionResult(
            prediction=prediction,
            actual_winner=actual_winner,
            actual_spread=away_score - home_score,
            actual_total=home_score + away_score,
        )

        self._results.append(result)

        # Index by week
        week = result.week
        if week not in self._by_week:
            self._by_week[week] = []
        self._by_week[week].append(result)

        # Index by team
        for team in [game.home_team, game.away_team]:
            if team not in self._by_team:
                self._by_team[team] = []
            self._by_team[team].append(result)

        return result

    def get_all_results(self) -> List[PredictionResult]:
        """Get all tracked results."""
        return self._results.copy()

    def get_week_results(self, week: int) -> List[PredictionResult]:
        """Get results for a specific week."""
        return self._by_week.get(week, [])

    def get_team_results(self, team: str) -> List[PredictionResult]:
        """Get results involving a specific team."""
        return self._by_team.get(team, [])

    def get_accuracy_summary(self) -> Dict:
        """
        Get accuracy summary statistics.

        Returns:
            Dict with accuracy metrics
        """
        if not self._results:
            return {"error": "No results tracked"}

        total = len(self._results)
        su_correct = sum(1 for r in self._results if r.su_correct)
        ats_results = [r for r in self._results if r.ats_correct is not None]
        ats_correct = sum(1 for r in ats_results if r.ats_correct)

        spread_errors = [r.abs_spread_error for r in self._results]
        mae = sum(spread_errors) / total

        return {
            "total_games": total,
            "su_correct": su_correct,
            "su_accuracy": round(su_correct / total * 100, 1),
            "ats_games": len(ats_results),
            "ats_correct": ats_correct,
            "ats_accuracy": round(ats_correct / len(ats_results) * 100, 1) if ats_results else None,
            "spread_mae": round(mae, 2),
        }


class SeasonBacktester:
    """
    Full season backtesting framework.

    Usage:
        backtester = SeasonBacktester(predictor)
        results = backtester.backtest_season(games, season=2024)
    """

    def __init__(self, predictor):
        """
        Initialize with a predictor.

        Args:
            predictor: SpreadPredictor instance
        """
        self.predictor = predictor
        self.tracker = PredictionResultTracker()

    def backtest_game(
        self,
        game: NFLGame,
        week: int
    ) -> Optional[PredictionResult]:
        """
        Backtest a single completed game.

        Args:
            game: NFLGame with completed scores
            week: Week number for ratings

        Returns:
            PredictionResult or None if game not completed
        """
        if not game.is_completed:
            return None

        prediction = self.predictor.predict_game(game, week)
        return self.tracker.add_result(
            prediction,
            home_score=game.home_score,
            away_score=game.away_score
        )

    def backtest_season(
        self,
        games: List[NFLGame],
        season: int
    ) -> Dict:
        """
        Backtest a full season of games.

        Args:
            games: List of completed NFLGame objects
            season: Season year

        Returns:
            Dict with backtest results
        """
        results = []
        for game in games:
            if game.is_completed and game.season == season:
                result = self.backtest_game(game, game.week)
                if result:
                    results.append(result)

        return {
            "season": season,
            "games_tested": len(results),
            "summary": self.tracker.get_accuracy_summary(),
            "results": results,
        }

    def get_report(self) -> str:
        """Generate formatted backtest report."""
        summary = self.tracker.get_accuracy_summary()

        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Total Games: {summary.get('total_games', 0)}",
            "",
            "STRAIGHT-UP",
            "-" * 40,
            f"  Correct: {summary.get('su_correct', 0)}",
            f"  Accuracy: {summary.get('su_accuracy', 0)}%",
            "",
            "AGAINST THE SPREAD",
            "-" * 40,
            f"  Games with Vegas line: {summary.get('ats_games', 0)}",
            f"  Correct: {summary.get('ats_correct', 0)}",
            f"  Accuracy: {summary.get('ats_accuracy', 'N/A')}%",
            "",
            "SPREAD ERROR",
            "-" * 40,
            f"  MAE: {summary.get('spread_mae', 0)} points",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)
