"""
Markdown Formatter - Export predictions and results to Markdown format.

This module contains:
- MarkdownFormatter: Format predictions and results as Markdown
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from nfl_predictor.models.games import GamePrediction


class MarkdownFormatter:
    """
    Format predictions and results as Markdown.

    Usage:
        formatter = MarkdownFormatter()
        md_string = formatter.format_predictions(predictions, week=5, season=2025)
        formatter.write_to_file(predictions, "predictions.md")
    """

    def __init__(self, include_toc: bool = False):
        """
        Initialize the formatter.

        Args:
            include_toc: Whether to include table of contents
        """
        self.include_toc = include_toc

    def format_predictions(
        self,
        predictions: List[GamePrediction],
        week: Optional[int] = None,
        season: Optional[int] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Format predictions as Markdown string.

        Args:
            predictions: List of GamePrediction objects
            week: Week number
            season: Season year
            title: Custom title

        Returns:
            Markdown formatted string
        """
        if not predictions:
            return "No predictions available."

        # Build header
        if title:
            header = f"# {title}\n\n"
        elif week and season:
            header = f"# NFL Week {week} Predictions ({season})\n\n"
        else:
            header = "# NFL Predictions\n\n"

        lines = [header]
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

        # Predictions table
        lines.append("## Game Predictions\n\n")
        lines.append("| Away | Home | Spread | Win % | Pick | Confidence |")
        lines.append("|:----:|:----:|:------:|:-----:|:----:|:----------:|")

        for pred in predictions:
            game = pred.game
            spread_str = f"{pred.predicted_spread:+.1f}"
            win_pct = f"{pred.home_win_probability:.0%}"
            conf = f"{pred.confidence:.0%}"

            lines.append(
                f"| {game.away_team} | {game.home_team} | "
                f"{spread_str} | {win_pct} | **{pred.pick}** | {conf} |"
            )

        lines.append("\n")

        # Add legend
        lines.append("### Legend\n")
        lines.append("- **Spread**: Predicted point spread (negative = home favored)\n")
        lines.append("- **Win %**: Home team win probability\n")
        lines.append("- **Confidence**: Model confidence in the pick\n")

        return "\n".join(lines)

    def format_standings(
        self,
        standings: Dict,
        title: str = "NFL Standings"
    ) -> str:
        """
        Format standings as Markdown.

        Args:
            standings: Standings dictionary by conference/division
            title: Title for the document

        Returns:
            Markdown formatted string
        """
        lines = [f"# {title}\n\n"]

        for conference in ["AFC", "NFC"]:
            lines.append(f"## {conference}\n\n")

            conf_data = standings.get(conference, {})
            for division, teams in conf_data.items():
                lines.append(f"### {division}\n\n")
                lines.append("| Team | W | L | T | Pct | PF | PA | Diff |")
                lines.append("|:----:|:-:|:-:|:-:|:---:|:--:|:--:|:----:|")

                for team in teams:
                    pct = f"{team.get('win_pct', 0):.3f}"
                    diff = team.get('point_diff', 0)
                    diff_str = f"+{diff}" if diff > 0 else str(diff)

                    lines.append(
                        f"| {team.get('team', '')} | "
                        f"{team.get('wins', 0)} | {team.get('losses', 0)} | "
                        f"{team.get('ties', 0)} | {pct} | "
                        f"{team.get('points_for', 0)} | {team.get('points_against', 0)} | "
                        f"{diff_str} |"
                    )

                lines.append("\n")

        return "\n".join(lines)

    def format_results(
        self,
        results: List[Dict],
        title: str = "Backtest Results"
    ) -> str:
        """
        Format backtest results as Markdown.

        Args:
            results: List of result dictionaries
            title: Title for the document

        Returns:
            Markdown formatted string
        """
        lines = [f"# {title}\n\n"]

        if not results:
            return lines[0] + "No results available.\n"

        # Summary stats
        total = len(results)
        su_correct = sum(1 for r in results if r.get("su_correct"))
        su_pct = su_correct / total * 100 if total > 0 else 0

        ats_results = [r for r in results if r.get("ats_correct") is not None]
        ats_correct = sum(1 for r in ats_results if r.get("ats_correct"))
        ats_pct = ats_correct / len(ats_results) * 100 if ats_results else 0

        lines.append("## Summary\n\n")
        lines.append(f"- **Total Games**: {total}\n")
        lines.append(f"- **Straight-Up**: {su_correct}/{total} ({su_pct:.1f}%)\n")
        if ats_results:
            lines.append(f"- **Against the Spread**: {ats_correct}/{len(ats_results)} ({ats_pct:.1f}%)\n")
        lines.append("\n")

        # Results table
        lines.append("## Detailed Results\n\n")
        lines.append("| Week | Game | Pick | Actual | Correct | Spread Error |")
        lines.append("|:----:|:-----|:----:|:------:|:-------:|:------------:|")

        for r in results:
            week = r.get("week", "")
            game_id = r.get("game_id", "")
            pick = r.get("predicted_winner", "")
            actual = r.get("actual_winner", "")
            correct = "✓" if r.get("su_correct") else "✗"
            spread_err = r.get("spread_error", 0)

            lines.append(f"| {week} | {game_id} | {pick} | {actual} | {correct} | {spread_err:+.1f} |")

        return "\n".join(lines)

    def format_weekly_report(
        self,
        predictions: List[GamePrediction],
        results: Optional[List[Dict]] = None,
        week: int = 0,
        season: int = 0
    ) -> str:
        """
        Format complete weekly report with predictions and optional results.

        Args:
            predictions: Week predictions
            results: Previous week results
            week: Week number
            season: Season year

        Returns:
            Markdown formatted string
        """
        lines = [f"# NFL Week {week} Report ({season})\n\n"]
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

        # Table of contents
        if self.include_toc:
            lines.append("## Contents\n")
            lines.append("1. [Predictions](#predictions)\n")
            if results:
                lines.append("2. [Last Week Results](#last-week-results)\n")
            lines.append("\n")

        # Predictions section
        lines.append("## Predictions\n\n")
        lines.append("| Away | Home | Spread | Win % | Pick |")
        lines.append("|:----:|:----:|:------:|:-----:|:----:|")

        for pred in predictions:
            game = pred.game
            lines.append(
                f"| {game.away_team} | {game.home_team} | "
                f"{pred.predicted_spread:+.1f} | "
                f"{pred.home_win_probability:.0%} | **{pred.pick}** |"
            )

        lines.append("\n")

        # Results section
        if results:
            lines.append("## Last Week Results\n\n")
            total = len(results)
            correct = sum(1 for r in results if r.get("su_correct"))

            lines.append(f"**Record**: {correct}/{total} ({correct/total*100:.1f}%)\n\n")

            lines.append("| Game | Pick | Result | ✓/✗ |")
            lines.append("|:-----|:----:|:------:|:---:|")

            for r in results:
                game_id = r.get("game_id", "")
                pick = r.get("predicted_winner", "")
                actual = r.get("actual_winner", "")
                icon = "✓" if r.get("su_correct") else "✗"
                lines.append(f"| {game_id} | {pick} | {actual} | {icon} |")

        return "\n".join(lines)

    def write_to_file(
        self,
        content: str,
        filepath: str
    ) -> None:
        """
        Write Markdown content to a file.

        Args:
            content: Markdown content
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            f.write(content)
