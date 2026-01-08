"""
CSV Formatter - Export predictions and results to CSV format.

This module contains:
- CSVFormatter: Format predictions and results as CSV
"""

import csv
import io
from typing import List, Dict, Any, Optional
from dataclasses import fields, is_dataclass

from nfl_predictor.models.games import GamePrediction


class CSVFormatter:
    """
    Format predictions and results as CSV.

    Usage:
        formatter = CSVFormatter()
        csv_string = formatter.format_predictions(predictions)
        formatter.write_to_file(predictions, "predictions.csv")
    """

    def __init__(self, delimiter: str = ",", include_headers: bool = True):
        """
        Initialize the formatter.

        Args:
            delimiter: CSV delimiter character
            include_headers: Whether to include header row
        """
        self.delimiter = delimiter
        self.include_headers = include_headers

    def format_predictions(self, predictions: List[GamePrediction]) -> str:
        """
        Format predictions as CSV string.

        Args:
            predictions: List of GamePrediction objects

        Returns:
            CSV formatted string
        """
        if not predictions:
            return ""

        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter)

        # Write headers
        if self.include_headers:
            headers = [
                "game_id",
                "season",
                "week",
                "away_team",
                "home_team",
                "predicted_spread",
                "home_win_probability",
                "away_win_probability",
                "pick",
                "pick_ats",
                "confidence",
                "vegas_spread",
                "vegas_total",
            ]
            writer.writerow(headers)

        # Write data rows
        for pred in predictions:
            game = pred.game
            row = [
                game.game_id,
                game.season,
                game.week,
                game.away_team,
                game.home_team,
                round(pred.predicted_spread, 1),
                round(pred.home_win_probability, 3),
                round(pred.away_win_probability, 3),
                pred.pick,
                pred.pick_ats or "",
                round(pred.confidence, 3),
                game.vegas_spread or "",
                game.vegas_total or "",
            ]
            writer.writerow(row)

        return output.getvalue()

    def format_results(self, results: List[Dict]) -> str:
        """
        Format backtest results as CSV string.

        Args:
            results: List of result dictionaries

        Returns:
            CSV formatted string
        """
        if not results:
            return ""

        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter)

        # Get headers from first result
        if self.include_headers and results:
            writer.writerow(results[0].keys())

        # Write data rows
        for result in results:
            writer.writerow(result.values())

        return output.getvalue()

    def format_standings(self, standings: List[Dict]) -> str:
        """
        Format standings as CSV string.

        Args:
            standings: List of team standing dictionaries

        Returns:
            CSV formatted string
        """
        if not standings:
            return ""

        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter)

        # Standard standings headers
        if self.include_headers:
            headers = [
                "team",
                "wins",
                "losses",
                "ties",
                "win_pct",
                "division",
                "conference",
                "points_for",
                "points_against",
                "point_diff",
            ]
            writer.writerow(headers)

        # Write data
        for team in standings:
            row = [
                team.get("team", ""),
                team.get("wins", 0),
                team.get("losses", 0),
                team.get("ties", 0),
                team.get("win_pct", 0.0),
                team.get("division", ""),
                team.get("conference", ""),
                team.get("points_for", 0),
                team.get("points_against", 0),
                team.get("point_diff", 0),
            ]
            writer.writerow(row)

        return output.getvalue()

    def write_to_file(
        self,
        data: List[Any],
        filepath: str,
        format_type: str = "predictions"
    ) -> None:
        """
        Write formatted data to a file.

        Args:
            data: Data to format
            filepath: Output file path
            format_type: Type of data ("predictions", "results", "standings")
        """
        if format_type == "predictions":
            content = self.format_predictions(data)
        elif format_type == "results":
            content = self.format_results(data)
        elif format_type == "standings":
            content = self.format_standings(data)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

        with open(filepath, "w", newline="") as f:
            f.write(content)

    def dataclass_to_csv(self, items: List[Any]) -> str:
        """
        Convert list of dataclass instances to CSV.

        Args:
            items: List of dataclass instances

        Returns:
            CSV formatted string
        """
        if not items:
            return ""

        if not is_dataclass(items[0]):
            raise ValueError("Items must be dataclass instances")

        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter)

        # Get field names from dataclass
        field_names = [f.name for f in fields(items[0])]

        if self.include_headers:
            writer.writerow(field_names)

        for item in items:
            row = [getattr(item, name) for name in field_names]
            writer.writerow(row)

        return output.getvalue()
