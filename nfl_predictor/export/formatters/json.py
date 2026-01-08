"""
JSON Formatter - Export predictions and results to JSON format.

This module contains:
- JSONFormatter: Format predictions and results as JSON
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict, is_dataclass
from datetime import datetime

from nfl_predictor.models.games import GamePrediction


class JSONFormatter:
    """
    Format predictions and results as JSON.

    Usage:
        formatter = JSONFormatter()
        json_string = formatter.format_predictions(predictions)
        formatter.write_to_file(predictions, "predictions.json")
    """

    def __init__(self, indent: int = 2, sort_keys: bool = False):
        """
        Initialize the formatter.

        Args:
            indent: JSON indentation level
            sort_keys: Whether to sort dictionary keys
        """
        self.indent = indent
        self.sort_keys = sort_keys

    def _json_serial(self, obj: Any) -> Any:
        """JSON serializer for objects not serializable by default."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def format_predictions(
        self,
        predictions: List[GamePrediction],
        include_metadata: bool = True
    ) -> str:
        """
        Format predictions as JSON string.

        Args:
            predictions: List of GamePrediction objects
            include_metadata: Whether to include metadata wrapper

        Returns:
            JSON formatted string
        """
        prediction_dicts = []

        for pred in predictions:
            game = pred.game
            pred_dict = {
                "game_id": game.game_id,
                "season": game.season,
                "week": game.week,
                "away_team": game.away_team,
                "home_team": game.home_team,
                "predicted_spread": round(pred.predicted_spread, 1),
                "home_win_probability": round(pred.home_win_probability, 3),
                "away_win_probability": round(pred.away_win_probability, 3),
                "pick": pred.pick,
                "pick_ats": pred.pick_ats,
                "confidence": round(pred.confidence, 3),
                "vegas_spread": game.vegas_spread,
                "vegas_total": game.vegas_total,
            }
            prediction_dicts.append(pred_dict)

        if include_metadata:
            output = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "count": len(predictions),
                    "type": "predictions",
                },
                "predictions": prediction_dicts,
            }
        else:
            output = prediction_dicts

        return json.dumps(
            output,
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=self._json_serial
        )

    def format_results(
        self,
        results: List[Dict],
        include_metadata: bool = True
    ) -> str:
        """
        Format backtest results as JSON string.

        Args:
            results: List of result dictionaries
            include_metadata: Whether to include metadata wrapper

        Returns:
            JSON formatted string
        """
        if include_metadata:
            output = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "count": len(results),
                    "type": "backtest_results",
                },
                "results": results,
            }
        else:
            output = results

        return json.dumps(
            output,
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=self._json_serial
        )

    def format_standings(
        self,
        standings: Dict,
        include_metadata: bool = True
    ) -> str:
        """
        Format standings as JSON string.

        Args:
            standings: Standings dictionary
            include_metadata: Whether to include metadata wrapper

        Returns:
            JSON formatted string
        """
        if include_metadata:
            output = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "type": "standings",
                },
                "standings": standings,
            }
        else:
            output = standings

        return json.dumps(
            output,
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=self._json_serial
        )

    def format_weekly_data(
        self,
        week: int,
        season: int,
        predictions: List[GamePrediction],
        standings: Optional[Dict] = None,
        ratings: Optional[Dict] = None
    ) -> str:
        """
        Format complete weekly data package.

        Args:
            week: Week number
            season: Season year
            predictions: Week predictions
            standings: Current standings
            ratings: Team ratings

        Returns:
            JSON formatted string
        """
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "season": season,
                "week": week,
                "type": "weekly_package",
            },
            "predictions": [],
            "standings": standings or {},
            "ratings": ratings or {},
        }

        for pred in predictions:
            game = pred.game
            output["predictions"].append({
                "game_id": game.game_id,
                "away_team": game.away_team,
                "home_team": game.home_team,
                "predicted_spread": round(pred.predicted_spread, 1),
                "home_win_probability": round(pred.home_win_probability, 3),
                "pick": pred.pick,
            })

        return json.dumps(
            output,
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=self._json_serial
        )

    def write_to_file(
        self,
        data: Any,
        filepath: str,
        format_type: str = "predictions"
    ) -> None:
        """
        Write formatted data to a file.

        Args:
            data: Data to format
            filepath: Output file path
            format_type: Type of data
        """
        if format_type == "predictions":
            content = self.format_predictions(data)
        elif format_type == "results":
            content = self.format_results(data)
        elif format_type == "standings":
            content = self.format_standings(data)
        else:
            # Generic JSON dump
            content = json.dumps(
                data,
                indent=self.indent,
                sort_keys=self.sort_keys,
                default=self._json_serial
            )

        with open(filepath, "w") as f:
            f.write(content)

    def parse_file(self, filepath: str) -> Dict:
        """
        Parse a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Parsed JSON data
        """
        with open(filepath, "r") as f:
            return json.load(f)
