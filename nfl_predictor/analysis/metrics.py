"""
Metrics - Accuracy and calibration metrics for predictions.

This module contains:
- AccuracyMetrics: Calculate accuracy metrics
- CalibrationAnalysis: Analyze probability calibration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math

from nfl_predictor.analysis.backtesting import PredictionResult


@dataclass
class AccuracyMetrics:
    """
    Calculate and store accuracy metrics for predictions.

    Attributes:
        total_games: Number of games analyzed
        su_correct: Straight-up correct picks
        su_accuracy: Straight-up accuracy percentage
        ats_correct: Against-the-spread correct picks
        ats_accuracy: ATS accuracy percentage
        spread_mae: Mean Absolute Error for spread
        spread_rmse: Root Mean Square Error for spread
        total_mae: MAE for total points
        brier_score: Brier score for probability calibration
    """
    total_games: int = 0
    su_correct: int = 0
    su_accuracy: float = 0.0
    ats_games: int = 0
    ats_correct: int = 0
    ats_accuracy: float = 0.0
    spread_mae: float = 0.0
    spread_rmse: float = 0.0
    total_mae: float = 0.0
    brier_score: float = 0.0

    @classmethod
    def from_results(cls, results: List[PredictionResult]) -> 'AccuracyMetrics':
        """
        Calculate metrics from a list of results.

        Args:
            results: List of PredictionResult objects

        Returns:
            AccuracyMetrics with calculated values
        """
        if not results:
            return cls()

        metrics = cls()
        metrics.total_games = len(results)

        # Straight-up accuracy
        metrics.su_correct = sum(1 for r in results if r.su_correct)
        metrics.su_accuracy = round(metrics.su_correct / metrics.total_games * 100, 1)

        # ATS accuracy
        ats_results = [r for r in results if r.ats_correct is not None]
        metrics.ats_games = len(ats_results)
        if ats_results:
            metrics.ats_correct = sum(1 for r in ats_results if r.ats_correct)
            metrics.ats_accuracy = round(metrics.ats_correct / metrics.ats_games * 100, 1)

        # Spread errors
        spread_errors = [r.spread_error for r in results]
        metrics.spread_mae = round(sum(abs(e) for e in spread_errors) / len(spread_errors), 2)
        metrics.spread_rmse = round(math.sqrt(sum(e**2 for e in spread_errors) / len(spread_errors)), 2)

        # Brier score
        brier_scores = []
        for r in results:
            actual = 1.0 if r.actual_winner == r.prediction.game.home_team else 0.0
            prob = r.prediction.home_win_probability
            brier_scores.append((prob - actual) ** 2)
        metrics.brier_score = round(sum(brier_scores) / len(brier_scores), 4)

        return metrics

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_games": self.total_games,
            "su_correct": self.su_correct,
            "su_accuracy": self.su_accuracy,
            "ats_games": self.ats_games,
            "ats_correct": self.ats_correct,
            "ats_accuracy": self.ats_accuracy,
            "spread_mae": self.spread_mae,
            "spread_rmse": self.spread_rmse,
            "brier_score": self.brier_score,
        }

    def get_summary(self) -> str:
        """Get formatted summary."""
        return (
            f"Games: {self.total_games} | "
            f"SU: {self.su_accuracy}% | "
            f"ATS: {self.ats_accuracy}% | "
            f"MAE: {self.spread_mae}"
        )


@dataclass
class CalibrationAnalysis:
    """
    Analyze probability calibration.

    Buckets predictions by predicted probability and compares
    to actual win rate.
    """
    bucket_size: int = 10  # Percentage points per bucket
    buckets: Dict[int, Tuple[float, float, int]] = field(default_factory=dict)
    # bucket -> (predicted_avg, actual_rate, count)

    @classmethod
    def from_results(cls, results: List[PredictionResult], bucket_size: int = 10) -> 'CalibrationAnalysis':
        """
        Calculate calibration from results.

        Args:
            results: List of PredictionResult objects
            bucket_size: Size of each bucket in percentage points

        Returns:
            CalibrationAnalysis with bucket data
        """
        analysis = cls(bucket_size=bucket_size)

        # Group by bucket
        bucket_data: Dict[int, List[Tuple[float, bool]]] = {}

        for r in results:
            prob = r.prediction.home_win_probability
            actual_home_won = r.actual_winner == r.prediction.game.home_team

            # Determine bucket (0-9 for 10% buckets)
            bucket = min(int(prob * 100) // bucket_size, (100 // bucket_size) - 1)

            if bucket not in bucket_data:
                bucket_data[bucket] = []
            bucket_data[bucket].append((prob, actual_home_won))

        # Calculate stats for each bucket
        for bucket, data in bucket_data.items():
            if data:
                predicted_avg = sum(p for p, _ in data) / len(data)
                actual_rate = sum(1 for _, won in data if won) / len(data)
                analysis.buckets[bucket] = (
                    round(predicted_avg, 3),
                    round(actual_rate, 3),
                    len(data)
                )

        return analysis

    def get_calibration_error(self) -> float:
        """Calculate mean calibration error across buckets."""
        if not self.buckets:
            return 0.0

        errors = []
        for predicted, actual, count in self.buckets.values():
            errors.append(abs(predicted - actual) * count)

        total_count = sum(count for _, _, count in self.buckets.values())
        return round(sum(errors) / total_count, 4) if total_count > 0 else 0.0

    def get_report(self) -> str:
        """Generate calibration report."""
        lines = [
            "CALIBRATION ANALYSIS",
            "-" * 50,
            f"{'Bucket':<15} {'Predicted':<12} {'Actual':<12} {'Count':<10}",
            "-" * 50,
        ]

        for bucket in sorted(self.buckets.keys()):
            predicted, actual, count = self.buckets[bucket]
            low = bucket * self.bucket_size
            high = (bucket + 1) * self.bucket_size
            lines.append(
                f"{low}-{high}%{'':<8} {predicted:<12.1%} {actual:<12.1%} {count:<10}"
            )

        lines.append("-" * 50)
        lines.append(f"Mean Calibration Error: {self.get_calibration_error():.2%}")

        return "\n".join(lines)
