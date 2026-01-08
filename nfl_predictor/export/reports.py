"""
Reports - Report generation utilities.

This module contains:
- WeeklyReport: Weekly prediction report
- SeasonReport: Full season report
- ReportGenerator: Generate various report types
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from nfl_predictor.models.games import GamePrediction
from nfl_predictor.analysis.backtesting import PredictionResult
from nfl_predictor.analysis.metrics import AccuracyMetrics


@dataclass
class WeeklyReport:
    """
    Weekly prediction report.

    Attributes:
        season: Season year
        week: Week number
        predictions: List of predictions
        previous_week_results: Results from previous week
        accuracy_ytd: Year-to-date accuracy
        generated_at: Report generation timestamp
    """
    season: int
    week: int
    predictions: List[GamePrediction] = field(default_factory=list)
    previous_week_results: List[PredictionResult] = field(default_factory=list)
    accuracy_ytd: Optional[AccuracyMetrics] = None
    generated_at: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> Dict:
        """Get report summary."""
        summary = {
            "season": self.season,
            "week": self.week,
            "num_predictions": len(self.predictions),
            "generated_at": self.generated_at.isoformat(),
        }

        # Previous week stats
        if self.previous_week_results:
            correct = sum(1 for r in self.previous_week_results if r.su_correct)
            total = len(self.previous_week_results)
            summary["previous_week"] = {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            }

        # YTD stats
        if self.accuracy_ytd:
            summary["ytd"] = self.accuracy_ytd.to_dict()

        return summary

    def get_picks_by_confidence(self) -> List[GamePrediction]:
        """Get predictions sorted by confidence."""
        return sorted(
            self.predictions,
            key=lambda p: p.confidence,
            reverse=True
        )

    def get_best_bets(self, min_confidence: float = 0.6) -> List[GamePrediction]:
        """Get high-confidence picks."""
        return [p for p in self.predictions if p.confidence >= min_confidence]


@dataclass
class SeasonReport:
    """
    Full season report.

    Attributes:
        season: Season year
        weeks_completed: Number of completed weeks
        total_predictions: Total predictions made
        accuracy_metrics: Overall accuracy metrics
        weekly_breakdown: Accuracy by week
        team_performance: How model performed for each team
    """
    season: int
    weeks_completed: int = 0
    total_predictions: int = 0
    accuracy_metrics: Optional[AccuracyMetrics] = None
    weekly_breakdown: Dict[int, AccuracyMetrics] = field(default_factory=dict)
    team_performance: Dict[str, Dict] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def get_best_week(self) -> Optional[int]:
        """Get week with best accuracy."""
        if not self.weekly_breakdown:
            return None

        return max(
            self.weekly_breakdown.keys(),
            key=lambda w: self.weekly_breakdown[w].su_accuracy
        )

    def get_worst_week(self) -> Optional[int]:
        """Get week with worst accuracy."""
        if not self.weekly_breakdown:
            return None

        return min(
            self.weekly_breakdown.keys(),
            key=lambda w: self.weekly_breakdown[w].su_accuracy
        )

    def get_team_accuracy(self, team: str) -> Optional[Dict]:
        """Get accuracy stats for a specific team."""
        return self.team_performance.get(team)


class ReportGenerator:
    """
    Generate various report types.

    This class provides utilities for generating weekly, season,
    and custom reports in various formats.

    Usage:
        generator = ReportGenerator()
        weekly = generator.create_weekly_report(predictions, week=5, season=2025)
        print(generator.render_weekly_report(weekly))
    """

    def __init__(self):
        """Initialize the generator."""
        self._weekly_reports: Dict[int, WeeklyReport] = {}
        self._season_report: Optional[SeasonReport] = None

    def create_weekly_report(
        self,
        predictions: List[GamePrediction],
        week: int,
        season: int,
        previous_results: Optional[List[PredictionResult]] = None,
        accuracy_ytd: Optional[AccuracyMetrics] = None
    ) -> WeeklyReport:
        """
        Create a weekly report.

        Args:
            predictions: Week predictions
            week: Week number
            season: Season year
            previous_results: Results from previous week
            accuracy_ytd: Year-to-date accuracy

        Returns:
            WeeklyReport instance
        """
        report = WeeklyReport(
            season=season,
            week=week,
            predictions=predictions,
            previous_week_results=previous_results or [],
            accuracy_ytd=accuracy_ytd,
        )

        self._weekly_reports[week] = report
        return report

    def create_season_report(
        self,
        season: int,
        all_results: List[PredictionResult]
    ) -> SeasonReport:
        """
        Create a full season report.

        Args:
            season: Season year
            all_results: All prediction results

        Returns:
            SeasonReport instance
        """
        report = SeasonReport(season=season)

        if not all_results:
            return report

        # Overall metrics
        report.accuracy_metrics = AccuracyMetrics.from_results(all_results)
        report.total_predictions = len(all_results)

        # Group by week
        by_week: Dict[int, List[PredictionResult]] = {}
        for result in all_results:
            week = result.week
            if week not in by_week:
                by_week[week] = []
            by_week[week].append(result)

        report.weeks_completed = len(by_week)

        # Calculate weekly metrics
        for week, results in by_week.items():
            report.weekly_breakdown[week] = AccuracyMetrics.from_results(results)

        # Team performance
        by_team: Dict[str, List[PredictionResult]] = {}
        for result in all_results:
            for team in [result.prediction.game.home_team, result.prediction.game.away_team]:
                if team not in by_team:
                    by_team[team] = []
                by_team[team].append(result)

        for team, results in by_team.items():
            # Calculate accuracy when predicting this team
            predicted_this_team = [r for r in results if r.predicted_winner == team]
            correct = sum(1 for r in predicted_this_team if r.su_correct)
            total = len(predicted_this_team)

            report.team_performance[team] = {
                "games": len(results),
                "predicted_wins": total,
                "correct_picks": correct,
                "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            }

        self._season_report = report
        return report

    def render_weekly_report(
        self,
        report: WeeklyReport,
        format: str = "text"
    ) -> str:
        """
        Render weekly report to string.

        Args:
            report: WeeklyReport to render
            format: Output format ("text" or "markdown")

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._render_weekly_markdown(report)
        return self._render_weekly_text(report)

    def _render_weekly_text(self, report: WeeklyReport) -> str:
        """Render weekly report as plain text."""
        lines = [
            "=" * 60,
            f"NFL WEEK {report.week} PREDICTIONS ({report.season})",
            "=" * 60,
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # Previous week results
        if report.previous_week_results:
            correct = sum(1 for r in report.previous_week_results if r.su_correct)
            total = len(report.previous_week_results)
            lines.extend([
                f"WEEK {report.week - 1} RESULTS: {correct}/{total} ({correct/total*100:.1f}%)",
                "-" * 40,
                "",
            ])

        # Predictions
        lines.extend([
            "THIS WEEK'S PREDICTIONS",
            "-" * 40,
            f"{'Away':<6} {'Home':<6} {'Spread':<8} {'Win%':<6} {'Pick':<6}",
            "-" * 40,
        ])

        for pred in report.predictions:
            game = pred.game
            lines.append(
                f"{game.away_team:<6} {game.home_team:<6} "
                f"{pred.predicted_spread:+5.1f}   "
                f"{pred.home_win_probability:.0%}    {pred.pick:<6}"
            )

        # YTD stats
        if report.accuracy_ytd:
            lines.extend([
                "",
                "YEAR-TO-DATE",
                "-" * 40,
                report.accuracy_ytd.get_summary(),
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def _render_weekly_markdown(self, report: WeeklyReport) -> str:
        """Render weekly report as Markdown."""
        lines = [
            f"# NFL Week {report.week} Predictions ({report.season})",
            "",
            f"*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        # Previous week results
        if report.previous_week_results:
            correct = sum(1 for r in report.previous_week_results if r.su_correct)
            total = len(report.previous_week_results)
            lines.extend([
                f"## Week {report.week - 1} Results",
                "",
                f"**Record**: {correct}/{total} ({correct/total*100:.1f}%)",
                "",
            ])

        # Predictions table
        lines.extend([
            "## Predictions",
            "",
            "| Away | Home | Spread | Win % | Pick |",
            "|:----:|:----:|:------:|:-----:|:----:|",
        ])

        for pred in report.predictions:
            game = pred.game
            lines.append(
                f"| {game.away_team} | {game.home_team} | "
                f"{pred.predicted_spread:+.1f} | "
                f"{pred.home_win_probability:.0%} | **{pred.pick}** |"
            )

        # YTD stats
        if report.accuracy_ytd:
            ytd = report.accuracy_ytd
            lines.extend([
                "",
                "## Year-to-Date",
                "",
                f"- **Games**: {ytd.total_games}",
                f"- **SU Accuracy**: {ytd.su_accuracy}%",
                f"- **ATS Accuracy**: {ytd.ats_accuracy}%",
                f"- **Spread MAE**: {ytd.spread_mae}",
            ])

        return "\n".join(lines)

    def render_season_report(
        self,
        report: SeasonReport,
        format: str = "text"
    ) -> str:
        """
        Render season report to string.

        Args:
            report: SeasonReport to render
            format: Output format

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"NFL {report.season} SEASON REPORT",
            "=" * 70,
            "",
        ]

        if report.accuracy_metrics:
            metrics = report.accuracy_metrics
            lines.extend([
                "OVERALL PERFORMANCE",
                "-" * 50,
                f"Total Predictions: {report.total_predictions}",
                f"Weeks Completed:   {report.weeks_completed}",
                "",
                f"Straight-Up:       {metrics.su_correct}/{metrics.total_games} ({metrics.su_accuracy}%)",
                f"Against Spread:    {metrics.ats_correct}/{metrics.ats_games} ({metrics.ats_accuracy}%)",
                f"Spread MAE:        {metrics.spread_mae}",
                f"Brier Score:       {metrics.brier_score}",
                "",
            ])

        # Best/Worst weeks
        best_week = report.get_best_week()
        worst_week = report.get_worst_week()

        if best_week and worst_week:
            lines.extend([
                "NOTABLE WEEKS",
                "-" * 50,
                f"Best Week:   Week {best_week} ({report.weekly_breakdown[best_week].su_accuracy}%)",
                f"Worst Week:  Week {worst_week} ({report.weekly_breakdown[worst_week].su_accuracy}%)",
                "",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)
