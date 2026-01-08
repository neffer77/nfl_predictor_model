"""
Spread Prediction - Home field advantage, rest/travel adjustments, and spread prediction.

This module contains:
- HomeFieldAdvantageConfig: Configuration for HFA
- HomeFieldAdvantageCalculator: Calculates HFA for matchups
- RestTravelConfig: Configuration for rest/travel adjustments
- RestTravelAdjuster: Calculates rest and travel impacts
- SpreadPredictor: Main prediction interface
- WeeklyScheduleManager: Manages weekly schedules and predictions
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from nfl_predictor.constants import (
    STADIUM_COORDINATES,
    STADIUM_ALTITUDES,
    DOME_STADIUMS,
)
from nfl_predictor.models.games import (
    NFLGame,
    GameType,
    GameLocation,
    TeamGameContext,
    GamePrediction,
)
from nfl_predictor.models.teams import BlendedTeamRating
from nfl_predictor.predictions.probability import (
    WinProbabilityModel,
    LogisticWinProbability,
)


@dataclass
class HomeFieldAdvantageConfig:
    """
    Configuration for home field advantage calculation.

    Research shows HFA has declined in recent years from ~3.0 to ~1.5-2.0 points.
    Various factors can modify the base advantage.

    Attributes:
        base_advantage: Base home field advantage in points (default 2.0)
        divisional_boost: Extra points for divisional home games
        altitude_bonus: Extra points for high-altitude home (Denver)
        dome_advantage: Points for dome team hosting outdoor team in bad weather
        crowd_noise_factor: Multiplier for loud stadiums (1.0 = normal)
        primetime_boost: Extra points for primetime home games
        playoff_multiplier: HFA multiplier for playoff games
    """
    base_advantage: float = 2.0
    divisional_boost: float = 0.3
    altitude_bonus: float = 0.5  # Denver effect
    dome_advantage: float = 0.3
    crowd_noise_factor: float = 1.0
    primetime_boost: float = 0.2
    playoff_multiplier: float = 1.2

    # Stadium-specific adjustments (team -> adjustment)
    stadium_adjustments: Dict[str, float] = field(default_factory=dict)


class HomeFieldAdvantageCalculator:
    """
    Calculates home field advantage for a specific matchup.

    Considers:
    - Base HFA (~2.0 points)
    - Divisional games (slight boost)
    - Altitude (Denver advantage)
    - Dome vs outdoor
    - Primetime games
    - Playoff games

    Usage:
        calc = HomeFieldAdvantageCalculator()
        hfa = calc.calculate(game)
    """

    def __init__(self, config: Optional[HomeFieldAdvantageConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or HomeFieldAdvantageConfig()

    def calculate(self, game: NFLGame) -> float:
        """
        Calculate home field advantage for a game.

        Args:
            game: NFLGame object

        Returns:
            Home field advantage in points (positive = home team advantage)
        """
        # Neutral site = no HFA
        if game.location == GameLocation.NEUTRAL:
            return 0.0

        # Start with base advantage
        hfa = self.config.base_advantage

        # Divisional boost
        if game.is_divisional:
            hfa += self.config.divisional_boost

        # Altitude bonus (Denver)
        home_altitude = STADIUM_ALTITUDES.get(game.home_team, 0)
        away_altitude = STADIUM_ALTITUDES.get(game.away_team, 0)
        if home_altitude > 4000 and away_altitude < 2000:
            hfa += self.config.altitude_bonus

        # Dome advantage (dome team at home vs outdoor team)
        home_is_dome = game.home_team in DOME_STADIUMS
        away_is_dome = game.away_team in DOME_STADIUMS
        if home_is_dome and not away_is_dome:
            hfa += self.config.dome_advantage

        # Primetime boost
        if game.home_context and game.home_context.is_primetime:
            hfa += self.config.primetime_boost

        # Playoff multiplier
        if game.game_type != GameType.REGULAR_SEASON:
            hfa *= self.config.playoff_multiplier

        # Stadium-specific adjustments
        if game.home_team in self.config.stadium_adjustments:
            hfa += self.config.stadium_adjustments[game.home_team]

        return round(hfa, 2)

    def get_travel_distance(self, away_team: str, home_team: str) -> float:
        """
        Calculate travel distance in miles between stadiums.

        Uses Haversine formula for great-circle distance.
        """
        if away_team not in STADIUM_COORDINATES or home_team not in STADIUM_COORDINATES:
            return 0.0

        lat1, lon1 = STADIUM_COORDINATES[away_team]
        lat2, lon2 = STADIUM_COORDINATES[home_team]

        # Haversine formula
        R = 3959  # Earth's radius in miles

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return round(R * c, 1)

    def get_timezone_change(self, away_team: str, home_team: str) -> int:
        """
        Calculate timezone change for away team.

        Returns hours of timezone shift (-3 to +3).
        Negative = traveling west, Positive = traveling east.
        """
        # Approximate timezone by longitude
        away_coords = STADIUM_COORDINATES.get(away_team)
        home_coords = STADIUM_COORDINATES.get(home_team)

        if not away_coords or not home_coords:
            return 0

        # Rough conversion: 15 degrees longitude = 1 timezone
        lon_diff = home_coords[1] - away_coords[1]
        tz_change = round(lon_diff / 15)

        return max(-3, min(3, tz_change))


@dataclass
class RestTravelConfig:
    """
    Configuration for rest and travel adjustments.

    Based on research showing:
    - Teams coming off bye week have ~1.5 point advantage
    - Short week (Thursday games) has ~1.0 point disadvantage
    - Cross-country travel has ~0.5 point disadvantage
    - East-to-West travel is harder than West-to-East

    Attributes:
        bye_week_advantage: Points advantage for team coming off bye
        short_week_penalty: Points penalty for Thursday game
        extra_rest_per_day: Points per extra day of rest (above 7)
        travel_penalty_per_1000_miles: Points penalty per 1000 miles
        timezone_penalty_per_hour: Points penalty per hour timezone change
        consecutive_road_penalty: Points penalty per consecutive road game
        mnf_short_turnaround: Extra penalty for MNF followed by early game
    """
    bye_week_advantage: float = 1.5
    short_week_penalty: float = 1.0
    extra_rest_per_day: float = 0.15
    travel_penalty_per_1000_miles: float = 0.3
    timezone_penalty_per_hour: float = 0.15
    consecutive_road_penalty: float = 0.2
    mnf_short_turnaround: float = 0.3
    max_adjustment: float = 3.0  # Cap total adjustment


class RestTravelAdjuster:
    """
    Calculates rest and travel adjustments for game predictions.

    This adjusts the baseline prediction based on situational factors
    like bye weeks, short weeks, and travel distance.

    Usage:
        adjuster = RestTravelAdjuster()
        adjustment = adjuster.calculate_adjustment(home_context, away_context)
    """

    def __init__(self, config: Optional[RestTravelConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or RestTravelConfig()
        self.hfa_calc = HomeFieldAdvantageCalculator()

    def calculate_adjustment(
        self,
        home_context: TeamGameContext,
        away_context: TeamGameContext
    ) -> float:
        """
        Calculate net rest/travel adjustment.

        Returns adjustment from home team perspective.
        Positive = favors home team, Negative = favors away team.
        """
        home_adj = self._calculate_team_adjustment(home_context, is_home=True)
        away_adj = self._calculate_team_adjustment(away_context, is_home=False)

        # Net adjustment (positive = home advantage)
        net_adjustment = home_adj - away_adj

        # Apply cap
        net_adjustment = max(-self.config.max_adjustment,
                            min(self.config.max_adjustment, net_adjustment))

        return round(net_adjustment, 2)

    def _calculate_team_adjustment(
        self,
        context: TeamGameContext,
        is_home: bool
    ) -> float:
        """Calculate adjustment for a single team."""
        adjustment = 0.0

        # Bye week advantage
        if context.is_coming_off_bye:
            adjustment += self.config.bye_week_advantage

        # Short week penalty
        if context.is_short_week:
            adjustment -= self.config.short_week_penalty

        # Extra rest (above normal 7 days, capped at 14)
        if context.days_rest > 7:
            extra_days = min(context.days_rest - 7, 7)
            adjustment += extra_days * self.config.extra_rest_per_day

        # Less rest penalty (below 7 days)
        if context.days_rest < 7:
            missing_days = 7 - context.days_rest
            adjustment -= missing_days * self.config.extra_rest_per_day * 1.5

        # Monday night short turnaround
        if context.previous_game_was_mnf and context.days_rest < 7:
            adjustment -= self.config.mnf_short_turnaround

        # Travel penalty (away team only)
        if not is_home:
            # Travel distance
            if context.travel_miles > 0:
                travel_penalty = (context.travel_miles / 1000) * self.config.travel_penalty_per_1000_miles
                adjustment -= travel_penalty

            # Timezone change
            if context.timezone_change != 0:
                # West-to-East is harder (positive timezone change)
                tz_penalty = abs(context.timezone_change) * self.config.timezone_penalty_per_hour
                if context.timezone_change > 0:  # Traveling east
                    tz_penalty *= 1.2
                adjustment -= tz_penalty

            # Consecutive road games
            if context.consecutive_road_games > 1:
                adjustment -= (context.consecutive_road_games - 1) * self.config.consecutive_road_penalty

        return adjustment

    def build_game_context(
        self,
        team: str,
        is_home: bool,
        days_rest: int = 7,
        previous_opponent: Optional[str] = None,
        away_team: Optional[str] = None,
        home_team: Optional[str] = None,
        is_primetime: bool = False,
        is_short_week: bool = False,
        is_coming_off_bye: bool = False,
        previous_game_was_mnf: bool = False,
        consecutive_road_games: int = 0
    ) -> TeamGameContext:
        """
        Build a TeamGameContext with calculated fields.

        This is a convenience method to construct context with
        travel distance and timezone automatically calculated.
        """
        travel_miles = 0.0
        timezone_change = 0

        if not is_home and home_team:
            travel_miles = self.hfa_calc.get_travel_distance(team, home_team)
            timezone_change = self.hfa_calc.get_timezone_change(team, home_team)

        return TeamGameContext(
            team=team,
            days_rest=days_rest,
            is_coming_off_bye=is_coming_off_bye,
            previous_game_was_mnf=previous_game_was_mnf,
            travel_miles=travel_miles,
            timezone_change=timezone_change,
            is_home=is_home,
            is_primetime=is_primetime,
            is_short_week=is_short_week,
            consecutive_road_games=consecutive_road_games
        )


class SpreadPredictor:
    """
    Generates spread predictions for NFL games.

    This is the main interface for generating game predictions.
    It combines team ratings, home field advantage, and rest/travel
    adjustments to produce spread predictions.

    Usage:
        predictor = SpreadPredictor(season_model)
        prediction = predictor.predict_game(game, week=5)
    """

    def __init__(
        self,
        season_model: 'NFLSeasonModel',
        hfa_calculator: Optional[HomeFieldAdvantageCalculator] = None,
        rest_adjuster: Optional[RestTravelAdjuster] = None,
        win_prob_model: Optional[WinProbabilityModel] = None
    ):
        """
        Initialize the predictor.

        Args:
            season_model: NFLSeasonModel with team ratings
            hfa_calculator: Custom HFA calculator (uses default if None)
            rest_adjuster: Custom rest/travel adjuster (uses default if None)
            win_prob_model: Custom win probability model (uses default if None)
        """
        self.season_model = season_model
        self.hfa_calc = hfa_calculator or HomeFieldAdvantageCalculator()
        self.rest_adjuster = rest_adjuster or RestTravelAdjuster()
        self.win_prob_model = win_prob_model or LogisticWinProbability()

    def predict_game(
        self,
        game: NFLGame,
        week: Optional[int] = None
    ) -> GamePrediction:
        """
        Generate prediction for a single game.

        Args:
            game: NFLGame to predict
            week: Week number for ratings (uses game.week if None)

        Returns:
            GamePrediction with spread and win probability
        """
        week = week or game.week

        # Get team ratings
        home_rating = self.season_model.get_team_rating(game.home_team, week)
        away_rating = self.season_model.get_team_rating(game.away_team, week)

        # Calculate home field advantage
        hfa = self.hfa_calc.calculate(game)

        # Calculate rest/travel adjustment
        rest_adj = 0.0
        if game.home_context and game.away_context:
            rest_adj = self.rest_adjuster.calculate_adjustment(
                game.home_context,
                game.away_context
            )

        # Determine confidence tier
        confidence_tier = self._determine_confidence(home_rating, away_rating)
        confidence_score = self._calculate_confidence_score(home_rating, away_rating)

        return GamePrediction(
            game=game,
            home_power_rating=home_rating.power_rating,
            away_power_rating=away_rating.power_rating,
            home_field_advantage=hfa,
            rest_travel_adjustment=rest_adj,
            confidence_tier=confidence_tier,
            confidence_score=confidence_score
        )

    def predict_games(
        self,
        games: List[NFLGame],
        week: Optional[int] = None
    ) -> List[GamePrediction]:
        """
        Generate predictions for multiple games.

        Args:
            games: List of NFLGame objects
            week: Week number for ratings

        Returns:
            List of GamePrediction objects
        """
        return [self.predict_game(game, week) for game in games]

    def _determine_confidence(
        self,
        home_rating: BlendedTeamRating,
        away_rating: BlendedTeamRating
    ) -> str:
        """Determine confidence tier based on data quality."""
        # Both high confidence = high
        if home_rating.confidence == "High" and away_rating.confidence == "High":
            return "High"

        # Both medium or mixed high/medium = medium
        if (home_rating.confidence in ["High", "Medium"] and
            away_rating.confidence in ["High", "Medium"]):
            return "Medium"

        return "Low"

    def _calculate_confidence_score(
        self,
        home_rating: BlendedTeamRating,
        away_rating: BlendedTeamRating
    ) -> int:
        """Calculate numerical confidence score (0-100)."""
        score = 50  # Base

        # Add points for data quality
        confidence_points = {"High": 20, "Medium": 10, "Low": 0, "Very Low": -10}
        score += confidence_points.get(home_rating.confidence, 0)
        score += confidence_points.get(away_rating.confidence, 0)

        # Add points for games played
        games = home_rating.games_played + away_rating.games_played
        score += min(games, 10)  # Max 10 points from games

        return max(0, min(100, score))


class WeeklyScheduleManager:
    """
    Manages weekly NFL schedules and predictions.

    This class coordinates the generation of predictions for
    a full week of games and tracks results.

    Usage:
        manager = WeeklyScheduleManager(season_model, season=2025)
        manager.add_games(week5_games)
        predictions = manager.get_week_predictions(week=5)
    """

    def __init__(
        self,
        season_model: 'NFLSeasonModel',
        season: int,
        predictor: Optional[SpreadPredictor] = None
    ):
        """
        Initialize the manager.

        Args:
            season_model: NFLSeasonModel with team ratings
            season: NFL season year
            predictor: Custom predictor (creates default if None)
        """
        self.season_model = season_model
        self.season = season
        self.predictor = predictor or SpreadPredictor(season_model)

        # Storage
        self._games: Dict[int, List[NFLGame]] = {}  # week -> games
        self._predictions: Dict[int, List[GamePrediction]] = {}  # week -> predictions
        self._results: Dict[str, dict] = {}  # game_id -> result info

    def add_game(self, game: NFLGame) -> None:
        """Add a single game to the schedule."""
        week = game.week
        if week not in self._games:
            self._games[week] = []
        self._games[week].append(game)

        # Clear cached predictions for this week
        if week in self._predictions:
            del self._predictions[week]

    def add_games(self, games: List[NFLGame]) -> None:
        """Add multiple games to the schedule."""
        for game in games:
            self.add_game(game)

    def get_week_games(self, week: int) -> List[NFLGame]:
        """Get all games for a specific week."""
        return self._games.get(week, [])

    def get_week_predictions(
        self,
        week: int,
        force_refresh: bool = False
    ) -> List[GamePrediction]:
        """
        Get predictions for all games in a week.

        Args:
            week: Week number
            force_refresh: Regenerate predictions even if cached

        Returns:
            List of GamePrediction objects
        """
        if week not in self._predictions or force_refresh:
            games = self.get_week_games(week)
            self._predictions[week] = self.predictor.predict_games(games, week)

        return self._predictions[week]

    def update_game_result(
        self,
        game_id: str,
        home_score: int,
        away_score: int
    ) -> None:
        """
        Update a game with its final result.

        Args:
            game_id: Game identifier
            home_score: Final home team score
            away_score: Final away team score
        """
        # Find and update the game
        for week_games in self._games.values():
            for game in week_games:
                if game.game_id == game_id:
                    game.home_score = home_score
                    game.away_score = away_score
                    game.is_completed = True

                    # Store result info
                    self._results[game_id] = {
                        "home_score": home_score,
                        "away_score": away_score,
                        "winner": game.winner,
                        "actual_spread": game.actual_spread
                    }
                    return

    def get_week_summary(self, week: int) -> Dict:
        """
        Get summary statistics for a week's predictions.

        Returns dict with:
        - total_games
        - games_completed
        - correct_picks
        - correct_ats
        - accuracy percentages
        """
        predictions = self.get_week_predictions(week)

        total = len(predictions)
        completed = sum(1 for p in predictions if p.game.is_completed)

        if completed == 0:
            return {
                "week": week,
                "total_games": total,
                "games_completed": 0,
                "correct_picks": 0,
                "correct_ats": 0,
                "pick_accuracy": None,
                "ats_accuracy": None
            }

        correct_picks = sum(1 for p in predictions if p.is_correct() is True)
        correct_ats = sum(1 for p in predictions if p.is_correct_ats() is True)
        ats_eligible = sum(1 for p in predictions if p.is_correct_ats() is not None)

        return {
            "week": week,
            "total_games": total,
            "games_completed": completed,
            "correct_picks": correct_picks,
            "correct_ats": correct_ats,
            "pick_accuracy": round(correct_picks / completed * 100, 1),
            "ats_accuracy": round(correct_ats / ats_eligible * 100, 1) if ats_eligible > 0 else None
        }

    def get_season_summary(self) -> Dict:
        """Get summary statistics for the entire season."""
        all_weeks = sorted(self._games.keys())

        if not all_weeks:
            return {"error": "No games loaded"}

        total_games = 0
        completed_games = 0
        correct_picks = 0
        correct_ats = 0
        ats_eligible = 0

        week_summaries = []

        for week in all_weeks:
            summary = self.get_week_summary(week)
            week_summaries.append(summary)

            total_games += summary["total_games"]
            completed_games += summary["games_completed"]
            correct_picks += summary["correct_picks"]
            correct_ats += summary["correct_ats"]

        return {
            "season": self.season,
            "weeks_loaded": len(all_weeks),
            "total_games": total_games,
            "completed_games": completed_games,
            "correct_picks": correct_picks,
            "correct_ats": correct_ats,
            "pick_accuracy": round(correct_picks / completed_games * 100, 1) if completed_games > 0 else None,
            "ats_accuracy": round(correct_ats / ats_eligible * 100, 1) if ats_eligible > 0 else None,
            "week_summaries": week_summaries
        }

    def get_predictions_by_confidence(
        self,
        week: int,
        min_confidence: int = 0
    ) -> Dict[str, List[GamePrediction]]:
        """
        Get predictions grouped by confidence tier.

        Args:
            week: Week number
            min_confidence: Minimum confidence score to include

        Returns:
            Dict with keys "High", "Medium", "Low" containing predictions
        """
        predictions = self.get_week_predictions(week)
        predictions = [p for p in predictions if p.confidence_score >= min_confidence]

        grouped = {"High": [], "Medium": [], "Low": []}
        for pred in predictions:
            grouped[pred.confidence_tier].append(pred)

        return grouped

    def get_best_bets(
        self,
        week: int,
        min_edge: float = 2.0,
        min_confidence: int = 50
    ) -> List[GamePrediction]:
        """
        Get predictions with the best betting edges.

        Args:
            week: Week number
            min_edge: Minimum edge vs Vegas spread
            min_confidence: Minimum confidence score

        Returns:
            List of predictions with good edges, sorted by edge magnitude
        """
        predictions = self.get_week_predictions(week)

        best = []
        for pred in predictions:
            if (pred.edge is not None and
                abs(pred.edge) >= min_edge and
                pred.confidence_score >= min_confidence):
                best.append(pred)

        # Sort by edge magnitude
        best.sort(key=lambda p: abs(p.edge or 0), reverse=True)
        return best

    def print_week_predictions(self, week: int) -> None:
        """Print formatted predictions for a week."""
        predictions = self.get_week_predictions(week)

        print(f"\n{'='*70}")
        print(f"WEEK {week} PREDICTIONS")
        print(f"{'='*70}")

        # Sort by game date/time
        predictions.sort(key=lambda p: p.game.game_date)

        for pred in predictions:
            game = pred.game
            print(f"\n{game.away_team} @ {game.home_team}")
            print(f"  Spread: {pred.formatted_spread}")
            print(f"  Win Prob: {game.home_team} {pred.home_win_probability:.1%} | "
                  f"{game.away_team} {pred.away_win_probability:.1%}")
            print(f"  Confidence: {pred.confidence_tier} ({pred.confidence_score})")

            if game.vegas_spread is not None:
                print(f"  Vegas: {game.home_team} {game.vegas_spread:+.1f}")
                if pred.edge:
                    edge_dir = game.home_team if pred.edge > 0 else game.away_team
                    print(f"  Edge: {abs(pred.edge):.1f} pts on {edge_dir}")
