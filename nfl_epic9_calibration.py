"""
NFL Prediction Model - Epic 9: Historical Calibration & Tuning

This module implements parameter calibration and optimization using historical
NFL data. Proper calibration is essential for reliable predictions - parameters
like decay rate, HFA, and EPA scaling should be empirically derived.

Epic 9 Stories:
- Story 9.1: Historical Data Repository
- Story 9.2: Backtesting Framework
- Story 9.3: Decay Curve Optimization
- Story 9.4: Home Field Advantage Calibration
- Story 9.5: EPA Scaling Optimization
- Story 9.6: Cross-Validation Framework
- Story 9.7: Parameter Configuration Management

Author: Connor's NFL Prediction System
Version: 9.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any, Union
from enum import Enum
from datetime import datetime, date
from abc import ABC, abstractmethod
import math
import random
import json
import copy

# Import Epic 7 and 8 for integration
try:
    from nfl_epic7_weather import (
        GameWeather, WeatherAdjustedPredictor, MockWeatherDataSource,
        STADIUM_DATA, StadiumInfo
    )
    EPIC7_AVAILABLE = True
except ImportError:
    EPIC7_AVAILABLE = False

try:
    from nfl_epic8_motivation import (
        PlayoffScenario, SituationalPredictor, StandingsEngine,
        NFL_DIVISIONS, TEAM_TO_DIVISION, TEAM_TO_CONFERENCE
    )
    EPIC8_AVAILABLE = True
except ImportError:
    EPIC8_AVAILABLE = False
    # Define basic NFL structure if Epic 8 not available
    NFL_DIVISIONS = {
        "AFC East": ["BUF", "MIA", "NE", "NYJ"],
        "AFC North": ["BAL", "CIN", "CLE", "PIT"],
        "AFC South": ["HOU", "IND", "JAX", "TEN"],
        "AFC West": ["DEN", "KC", "LAC", "LV"],
        "NFC East": ["DAL", "NYG", "PHI", "WAS"],
        "NFC North": ["CHI", "DET", "GB", "MIN"],
        "NFC South": ["ATL", "CAR", "NO", "TB"],
        "NFC West": ["ARI", "LAR", "SEA", "SF"],
    }
    TEAM_TO_DIVISION = {team: div for div, teams in NFL_DIVISIONS.items() for team in teams}
    TEAM_TO_CONFERENCE = {team: "AFC" if div.startswith("AFC") else "NFC" 
                          for team, div in TEAM_TO_DIVISION.items()}


# =============================================================================
# STORY 9.1: HISTORICAL DATA REPOSITORY
# =============================================================================

@dataclass
class HistoricalGame:
    """
    Complete historical game record.
    
    Story 9.1: Core data model for historical games.
    """
    game_id: str
    season: int
    week: int
    game_date: date
    
    # Teams
    home_team: str
    away_team: str
    
    # Scores
    home_score: int
    away_score: int
    
    # Vegas lines
    vegas_spread: float  # Negative = home favored
    vegas_total: float
    vegas_home_ml: Optional[int] = None
    vegas_away_ml: Optional[int] = None
    
    # Team ratings at game time
    home_preseason_rating: Optional[float] = None
    away_preseason_rating: Optional[float] = None
    home_inseason_rating: Optional[float] = None
    away_inseason_rating: Optional[float] = None
    home_epa_per_play: Optional[float] = None
    away_epa_per_play: Optional[float] = None
    
    # Context
    is_divisional: bool = False
    is_playoff: bool = False
    is_primetime: bool = False
    home_rest_days: int = 7
    away_rest_days: int = 7
    
    # Weather (outdoor games only)
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[str] = None
    is_dome: bool = False
    
    # Computed properties
    @property
    def actual_spread(self) -> float:
        """Actual spread (away - home), positive = home won by more"""
        return self.away_score - self.home_score
    
    @property
    def actual_total(self) -> int:
        """Total points scored"""
        return self.home_score + self.away_score
    
    @property
    def home_won(self) -> bool:
        """Did home team win?"""
        return self.home_score > self.away_score
    
    @property
    def home_covered(self) -> bool:
        """Did home team cover the spread?"""
        # If spread is -3 (home favored by 3), home covers if they win by more than 3
        return self.actual_spread < self.vegas_spread
    
    @property
    def over_hit(self) -> bool:
        """Did the game go over the total?"""
        return self.actual_total > self.vegas_total
    
    @property
    def margin(self) -> int:
        """Home team margin of victory (negative if lost)"""
        return self.home_score - self.away_score


@dataclass
class SeasonData:
    """Collection of games and team data for a season"""
    season: int
    games: List[HistoricalGame] = field(default_factory=list)
    team_preseason_ratings: Dict[str, float] = field(default_factory=dict)
    team_weekly_epa: Dict[str, Dict[int, float]] = field(default_factory=dict)  # team -> week -> epa
    
    def get_games_through_week(self, week: int) -> List[HistoricalGame]:
        """Get all games up to and including a week"""
        return [g for g in self.games if g.week <= week]
    
    def get_week_games(self, week: int) -> List[HistoricalGame]:
        """Get games for a specific week"""
        return [g for g in self.games if g.week == week]
    
    def get_team_games(self, team: str) -> List[HistoricalGame]:
        """Get all games for a team"""
        return [g for g in self.games if g.home_team == team or g.away_team == team]


class HistoricalDataRepository:
    """
    Repository for historical NFL data.
    
    Story 9.1: Data storage and retrieval for backtesting.
    """
    
    def __init__(self):
        self.seasons: Dict[int, SeasonData] = {}
        self._all_teams = set()
    
    def add_season(self, season_data: SeasonData) -> None:
        """Add a season's data"""
        self.seasons[season_data.season] = season_data
        for game in season_data.games:
            self._all_teams.add(game.home_team)
            self._all_teams.add(game.away_team)
    
    def get_season(self, season: int) -> Optional[SeasonData]:
        """Get data for a specific season"""
        return self.seasons.get(season)
    
    def get_seasons(self, start: int, end: int) -> List[SeasonData]:
        """Get multiple seasons"""
        return [self.seasons[s] for s in range(start, end + 1) if s in self.seasons]
    
    def get_all_games(self) -> List[HistoricalGame]:
        """Get all games across all seasons"""
        games = []
        for season_data in self.seasons.values():
            games.extend(season_data.games)
        return games
    
    @property
    def available_seasons(self) -> List[int]:
        """List of available seasons"""
        return sorted(self.seasons.keys())
    
    @property
    def total_games(self) -> int:
        """Total number of games in repository"""
        return sum(len(s.games) for s in self.seasons.values())
    
    def query(self, 
              seasons: Optional[List[int]] = None,
              teams: Optional[List[str]] = None,
              week_range: Optional[Tuple[int, int]] = None,
              divisional_only: bool = False,
              playoff_only: bool = False) -> List[HistoricalGame]:
        """
        Query games with filters.
        
        Args:
            seasons: Filter by seasons
            teams: Filter by teams (either home or away)
            week_range: Filter by week range (start, end)
            divisional_only: Only divisional games
            playoff_only: Only playoff games
        
        Returns:
            List of matching games
        """
        games = self.get_all_games()
        
        if seasons:
            games = [g for g in games if g.season in seasons]
        
        if teams:
            games = [g for g in games if g.home_team in teams or g.away_team in teams]
        
        if week_range:
            start, end = week_range
            games = [g for g in games if start <= g.week <= end]
        
        if divisional_only:
            games = [g for g in games if g.is_divisional]
        
        if playoff_only:
            games = [g for g in games if g.is_playoff]
        
        return games


class MockHistoricalDataGenerator:
    """
    Generates realistic mock historical data for testing.
    
    Story 9.1: Test data generation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.teams = list(TEAM_TO_DIVISION.keys())
    
    def generate_season(self, season: int) -> SeasonData:
        """Generate a full season of mock data"""
        season_data = SeasonData(season=season)
        
        # Generate preseason ratings (-5 to +5)
        for team in self.teams:
            season_data.team_preseason_ratings[team] = self.rng.gauss(0, 2.5)
        
        # Generate 18 weeks of regular season + playoffs
        game_id = 0
        for week in range(1, 19):
            week_games = self._generate_week(season, week, season_data, game_id)
            season_data.games.extend(week_games)
            game_id += len(week_games)
        
        # Generate playoff games (simplified)
        playoff_games = self._generate_playoffs(season, season_data, game_id)
        season_data.games.extend(playoff_games)
        
        return season_data
    
    def _generate_week(self, season: int, week: int, 
                       season_data: SeasonData, start_id: int) -> List[HistoricalGame]:
        """Generate games for a single week"""
        games = []
        
        # Pair up teams for matchups (simplified - not actual NFL schedule)
        shuffled_teams = self.teams.copy()
        self.rng.shuffle(shuffled_teams)
        
        for i in range(0, len(shuffled_teams), 2):
            home_team = shuffled_teams[i]
            away_team = shuffled_teams[i + 1]
            
            # Get team strengths
            home_rating = season_data.team_preseason_ratings.get(home_team, 0)
            away_rating = season_data.team_preseason_ratings.get(away_team, 0)
            
            # Add some in-season variation
            home_form = self.rng.gauss(0, 1.5)
            away_form = self.rng.gauss(0, 1.5)
            
            # Calculate expected spread (home rating - away rating + HFA)
            hfa = 2.5
            expected_spread = (home_rating + home_form) - (away_rating + away_form) + hfa
            
            # Vegas spread (approximately the expected spread with noise)
            vegas_spread = -round(expected_spread + self.rng.gauss(0, 0.5), 1)
            
            # Simulate actual score
            base_points = 24
            home_points = int(base_points + expected_spread / 2 + self.rng.gauss(0, 10))
            away_points = int(base_points - expected_spread / 2 + self.rng.gauss(0, 10))
            home_points = max(0, home_points)
            away_points = max(0, away_points)
            
            # Vegas total
            vegas_total = round(48 + self.rng.gauss(0, 3), 1)
            
            # Check if divisional
            is_divisional = TEAM_TO_DIVISION.get(home_team) == TEAM_TO_DIVISION.get(away_team)
            
            # Weather for outdoor games
            is_dome = home_team in ["LV", "DET", "MIN", "NO", "ATL", "IND", "DAL", "LAR", "LAC", "ARI", "HOU"]
            temperature = None
            wind_speed = None
            
            if not is_dome:
                # Generate weather based on season/month
                if week <= 4:  # September
                    temperature = self.rng.gauss(72, 10)
                elif week <= 8:  # October
                    temperature = self.rng.gauss(58, 12)
                elif week <= 12:  # November
                    temperature = self.rng.gauss(45, 15)
                else:  # December-January
                    temperature = self.rng.gauss(35, 18)
                
                wind_speed = max(0, self.rng.gauss(8, 6))
            
            # Calculate game date (season starts in September)
            month = 9 + (week - 1) // 4
            day = 1 + ((week - 1) % 4) * 7
            year = season
            if month > 12:
                month -= 12
                year += 1
            day = min(day, 28)  # Avoid invalid dates
            
            game = HistoricalGame(
                game_id=f"{season}_{week}_{start_id + len(games)}",
                season=season,
                week=week,
                game_date=date(year, month, day),
                home_team=home_team,
                away_team=away_team,
                home_score=home_points,
                away_score=away_points,
                vegas_spread=vegas_spread,
                vegas_total=vegas_total,
                home_preseason_rating=home_rating,
                away_preseason_rating=away_rating,
                is_divisional=is_divisional,
                is_dome=is_dome,
                temperature=temperature,
                wind_speed=wind_speed,
                home_rest_days=7,
                away_rest_days=7
            )
            games.append(game)
        
        return games
    
    def _generate_playoffs(self, season: int, season_data: SeasonData,
                           start_id: int) -> List[HistoricalGame]:
        """Generate simplified playoff games"""
        games = []
        # Generate 11 playoff games (4 wild card + 4 divisional + 2 conference + 1 super bowl)
        for i, week in enumerate([19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 22]):
            home_team = self.rng.choice(self.teams)
            away_team = self.rng.choice([t for t in self.teams if t != home_team])
            
            game = HistoricalGame(
                game_id=f"{season}_{week}_{start_id + i}",
                season=season,
                week=week,
                game_date=date(season + 1, 1, 7 + (week - 19) * 7),
                home_team=home_team,
                away_team=away_team,
                home_score=self.rng.randint(17, 35),
                away_score=self.rng.randint(14, 31),
                vegas_spread=self.rng.uniform(-7, 7),
                vegas_total=self.rng.uniform(42, 52),
                is_playoff=True,
                is_dome=home_team in ["LV", "DET", "MIN", "NO", "ATL", "IND"]
            )
            games.append(game)
        
        return games


# =============================================================================
# STORY 9.2: BACKTESTING FRAMEWORK
# =============================================================================

@dataclass
class PredictionResult:
    """Result of a single game prediction"""
    game: HistoricalGame
    
    # Predictions
    predicted_spread: float
    predicted_total: float
    predicted_home_win_prob: float
    
    # Computed
    @property
    def spread_error(self) -> float:
        """Absolute error in spread prediction vs actual"""
        return self.game.actual_spread - self.predicted_spread
    
    @property
    def total_error(self) -> float:
        """Error in total prediction"""
        return self.game.actual_total - self.predicted_total
    
    @property
    def picked_home(self) -> bool:
        """Did model pick home team to win?"""
        return self.predicted_home_win_prob > 0.5
    
    @property
    def pick_correct(self) -> bool:
        """Was the straight-up pick correct?"""
        return self.picked_home == self.game.home_won
    
    @property
    def spread_pick_correct(self) -> bool:
        """Was the spread pick correct (vs Vegas)?"""
        # If predicted spread < vegas spread, picking home to cover
        picking_home = self.predicted_spread < self.game.vegas_spread
        return picking_home == self.game.home_covered
    
    @property
    def brier_score(self) -> float:
        """Brier score for this prediction (lower is better)"""
        actual = 1.0 if self.game.home_won else 0.0
        return (self.predicted_home_win_prob - actual) ** 2


@dataclass
class BacktestResults:
    """
    Aggregated results from a backtest run.
    
    Story 9.2: Backtest output data model.
    """
    # Metadata
    seasons_tested: List[int]
    total_games: int
    parameters_used: Dict[str, float]
    
    # Individual results
    predictions: List[PredictionResult] = field(default_factory=list)
    
    # Accuracy metrics
    straight_up_accuracy: float = 0.0
    ats_accuracy: float = 0.0  # Against the spread
    
    # Error metrics
    spread_mae: float = 0.0  # Mean Absolute Error
    spread_rmse: float = 0.0  # Root Mean Square Error
    total_mae: float = 0.0
    total_rmse: float = 0.0
    
    # Probabilistic metrics
    brier_score: float = 0.0
    log_loss: float = 0.0
    
    # Breakdown by category
    accuracy_by_week: Dict[int, float] = field(default_factory=dict)
    accuracy_by_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Calibration data
    calibration_buckets: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    
    def calculate_metrics(self) -> None:
        """Calculate all metrics from predictions"""
        if not self.predictions:
            return
        
        n = len(self.predictions)
        self.total_games = n
        
        # Straight-up accuracy
        correct = sum(1 for p in self.predictions if p.pick_correct)
        self.straight_up_accuracy = correct / n
        
        # ATS accuracy
        ats_correct = sum(1 for p in self.predictions if p.spread_pick_correct)
        self.ats_accuracy = ats_correct / n
        
        # Spread errors
        spread_errors = [p.spread_error for p in self.predictions]
        self.spread_mae = sum(abs(e) for e in spread_errors) / n
        self.spread_rmse = math.sqrt(sum(e ** 2 for e in spread_errors) / n)
        
        # Total errors
        total_errors = [p.total_error for p in self.predictions]
        self.total_mae = sum(abs(e) for e in total_errors) / n
        self.total_rmse = math.sqrt(sum(e ** 2 for e in total_errors) / n)
        
        # Brier score
        self.brier_score = sum(p.brier_score for p in self.predictions) / n
        
        # Log loss
        eps = 1e-10
        log_losses = []
        for p in self.predictions:
            actual = 1.0 if p.game.home_won else 0.0
            prob = max(eps, min(1 - eps, p.predicted_home_win_prob))
            ll = -(actual * math.log(prob) + (1 - actual) * math.log(1 - prob))
            log_losses.append(ll)
        self.log_loss = sum(log_losses) / n
        
        # Accuracy by week
        week_results = {}
        for p in self.predictions:
            week = p.game.week
            if week not in week_results:
                week_results[week] = {"correct": 0, "total": 0}
            week_results[week]["total"] += 1
            if p.pick_correct:
                week_results[week]["correct"] += 1
        
        self.accuracy_by_week = {
            w: r["correct"] / r["total"] 
            for w, r in week_results.items() if r["total"] > 0
        }
        
        # Accuracy by confidence tier
        confidence_tiers = {
            "high": (0.65, 1.0),
            "medium": (0.55, 0.65),
            "low": (0.5, 0.55),
            "toss_up": (0.45, 0.5)
        }
        
        for tier, (low, high) in confidence_tiers.items():
            tier_preds = [
                p for p in self.predictions 
                if low <= max(p.predicted_home_win_prob, 1 - p.predicted_home_win_prob) < high
            ]
            if tier_preds:
                correct = sum(1 for p in tier_preds if p.pick_correct)
                self.accuracy_by_confidence[tier] = correct / len(tier_preds)
        
        # Calibration buckets (10 buckets from 0-100%)
        for bucket in range(10):
            low = bucket / 10
            high = (bucket + 1) / 10
            bucket_preds = [
                p for p in self.predictions
                if low <= p.predicted_home_win_prob < high
            ]
            if bucket_preds:
                predicted_avg = sum(p.predicted_home_win_prob for p in bucket_preds) / len(bucket_preds)
                actual_rate = sum(1 for p in bucket_preds if p.game.home_won) / len(bucket_preds)
                self.calibration_buckets[bucket] = (predicted_avg, actual_rate)
    
    def get_summary(self) -> str:
        """Generate summary report"""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Seasons: {self.seasons_tested}",
            f"Total Games: {self.total_games}",
            "",
            "ACCURACY METRICS",
            "-" * 40,
            f"  Straight-Up: {self.straight_up_accuracy:.1%}",
            f"  Against Spread: {self.ats_accuracy:.1%}",
            "",
            "ERROR METRICS",
            "-" * 40,
            f"  Spread MAE: {self.spread_mae:.2f} pts",
            f"  Spread RMSE: {self.spread_rmse:.2f} pts",
            f"  Total MAE: {self.total_mae:.2f} pts",
            f"  Total RMSE: {self.total_rmse:.2f} pts",
            "",
            "PROBABILISTIC METRICS",
            "-" * 40,
            f"  Brier Score: {self.brier_score:.4f}",
            f"  Log Loss: {self.log_loss:.4f}",
            "",
            "ACCURACY BY CONFIDENCE",
            "-" * 40,
        ]
        
        for tier, acc in self.accuracy_by_confidence.items():
            lines.append(f"  {tier.title():12s}: {acc:.1%}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


@dataclass
class ModelParameters:
    """
    Model parameters to be calibrated.
    
    Story 9.2: Parameter configuration for backtesting.
    """
    # Decay curve
    decay_rate: float = 0.15  # k in e^(-k*week)
    min_preseason_weight: float = 0.05
    
    # Home field advantage
    base_hfa: float = 2.5
    altitude_bonus: float = 0.5
    dome_outdoor_penalty: float = 0.3
    divisional_reduction: float = 0.5
    
    # EPA scaling
    epa_to_points_factor: float = 25.0
    offensive_weight: float = 1.0
    defensive_weight: float = 1.0
    
    # Spread calculation
    spread_home_adjustment: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "decay_rate": self.decay_rate,
            "min_preseason_weight": self.min_preseason_weight,
            "base_hfa": self.base_hfa,
            "altitude_bonus": self.altitude_bonus,
            "dome_outdoor_penalty": self.dome_outdoor_penalty,
            "divisional_reduction": self.divisional_reduction,
            "epa_to_points_factor": self.epa_to_points_factor,
            "offensive_weight": self.offensive_weight,
            "defensive_weight": self.defensive_weight,
            "spread_home_adjustment": self.spread_home_adjustment,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'ModelParameters':
        """Create from dictionary"""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    def copy(self) -> 'ModelParameters':
        """Create a copy"""
        return ModelParameters(**self.to_dict())


class SimplePredictor:
    """
    Simple prediction model for backtesting.
    
    Story 9.2: Configurable prediction engine.
    """
    
    def __init__(self, params: ModelParameters):
        self.params = params
    
    def predict_game(self, game: HistoricalGame, week: int) -> Tuple[float, float, float]:
        """
        Predict spread, total, and home win probability.
        
        Args:
            game: Historical game with ratings
            week: Current week (for decay calculation)
        
        Returns:
            Tuple of (predicted_spread, predicted_total, home_win_prob)
        """
        # Get team ratings
        home_pre = game.home_preseason_rating or 0
        away_pre = game.away_preseason_rating or 0
        home_in = game.home_inseason_rating or home_pre
        away_in = game.away_inseason_rating or away_pre
        
        # Calculate blended ratings using decay
        preseason_weight = max(
            self.params.min_preseason_weight,
            math.exp(-self.params.decay_rate * week)
        )
        inseason_weight = 1 - preseason_weight
        
        home_rating = home_pre * preseason_weight + home_in * inseason_weight
        away_rating = away_pre * preseason_weight + away_in * inseason_weight
        
        # Calculate HFA
        hfa = self.params.base_hfa
        
        # Altitude adjustment (Denver)
        if game.home_team == "DEN":
            hfa += self.params.altitude_bonus
        
        # Dome team playing outdoors
        dome_teams = {"LV", "DET", "MIN", "NO", "ATL", "IND", "DAL", "LAR", "LAC", "ARI", "HOU"}
        if game.away_team in dome_teams and not game.is_dome:
            if game.temperature and game.temperature < 45:
                hfa += self.params.dome_outdoor_penalty
        
        # Divisional reduction
        if game.is_divisional:
            hfa -= self.params.divisional_reduction
        
        # Calculate spread
        rating_diff = home_rating - away_rating
        predicted_spread = -(rating_diff + hfa + self.params.spread_home_adjustment)
        
        # Calculate total (simplified)
        avg_rating = (abs(home_rating) + abs(away_rating)) / 2
        predicted_total = 47 + avg_rating * 2
        
        # Calculate win probability (logistic)
        spread_for_prob = -predicted_spread  # Positive = home favored
        home_win_prob = 1 / (1 + 10 ** (-spread_for_prob / 6))
        
        return predicted_spread, predicted_total, home_win_prob


class BacktestingFramework:
    """
    Framework for running backtests on historical data.
    
    Story 9.2: Core backtesting engine.
    """
    
    def __init__(self, repository: HistoricalDataRepository):
        self.repository = repository
    
    def run_backtest(self, 
                     seasons: List[int],
                     params: ModelParameters) -> BacktestResults:
        """
        Run backtest with specified parameters.
        
        Args:
            seasons: Seasons to include
            params: Model parameters to use
        
        Returns:
            BacktestResults with all metrics
        """
        predictor = SimplePredictor(params)
        predictions = []
        
        for season in seasons:
            season_data = self.repository.get_season(season)
            if not season_data:
                continue
            
            for game in season_data.games:
                # Skip playoff games for now
                if game.is_playoff:
                    continue
                
                # Make prediction
                spread, total, prob = predictor.predict_game(game, game.week)
                
                result = PredictionResult(
                    game=game,
                    predicted_spread=spread,
                    predicted_total=total,
                    predicted_home_win_prob=prob
                )
                predictions.append(result)
        
        # Create results
        results = BacktestResults(
            seasons_tested=seasons,
            total_games=len(predictions),
            parameters_used=params.to_dict(),
            predictions=predictions
        )
        results.calculate_metrics()
        
        return results
    
    def parameter_sweep(self,
                        param_name: str,
                        values: List[float],
                        seasons: List[int],
                        base_params: Optional[ModelParameters] = None) -> Dict[float, BacktestResults]:
        """
        Test multiple parameter values.
        
        Args:
            param_name: Name of parameter to sweep
            values: Values to test
            seasons: Seasons to test on
            base_params: Base parameters (others held constant)
        
        Returns:
            Dict mapping parameter value to results
        """
        base_params = base_params or ModelParameters()
        results = {}
        
        for value in values:
            # Create params with this value
            params = base_params.copy()
            setattr(params, param_name, value)
            
            # Run backtest
            results[value] = self.run_backtest(seasons, params)
        
        return results


# =============================================================================
# STORY 9.3: DECAY CURVE OPTIMIZATION
# =============================================================================

@dataclass
class DecayOptimizationResult:
    """
    Results from decay curve optimization.
    
    Story 9.3: Output of decay optimization.
    """
    optimal_k: float
    optimal_mae: float
    optimal_brier: float
    
    # Full sweep results
    k_values_tested: List[float] = field(default_factory=list)
    mae_by_k: Dict[float, float] = field(default_factory=dict)
    brier_by_k: Dict[float, float] = field(default_factory=dict)
    accuracy_by_k: Dict[float, float] = field(default_factory=dict)
    
    # Confidence interval
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    # Comparison to baseline
    baseline_k: float = 0.15
    baseline_mae: float = 0.0
    improvement_pct: float = 0.0
    
    def get_summary(self) -> str:
        """Generate summary report"""
        lines = [
            "=" * 60,
            "DECAY CURVE OPTIMIZATION RESULTS",
            "=" * 60,
            "",
            f"Optimal k: {self.optimal_k:.3f}",
            f"95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]",
            "",
            f"Optimal MAE: {self.optimal_mae:.3f}",
            f"Optimal Brier: {self.optimal_brier:.4f}",
            "",
            f"Baseline k: {self.baseline_k:.3f}",
            f"Baseline MAE: {self.baseline_mae:.3f}",
            f"Improvement: {self.improvement_pct:.1f}%",
            "",
            "K VALUES TESTED",
            "-" * 40,
        ]
        
        for k in sorted(self.k_values_tested):
            mae = self.mae_by_k.get(k, 0)
            acc = self.accuracy_by_k.get(k, 0)
            marker = " *" if k == self.optimal_k else ""
            lines.append(f"  k={k:.3f}: MAE={mae:.3f}, Acc={acc:.1%}{marker}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class DecayCurveOptimizer:
    """
    Optimizes the preseason-to-inseason decay rate.
    
    Story 9.3: Find optimal k value for decay curve.
    """
    
    def __init__(self, backtester: BacktestingFramework):
        self.backtester = backtester
    
    def optimize(self,
                 seasons: List[int],
                 k_range: Tuple[float, float] = (0.05, 0.30),
                 num_points: int = 26,
                 metric: str = "mae",
                 base_params: Optional[ModelParameters] = None) -> DecayOptimizationResult:
        """
        Find optimal decay rate.
        
        Args:
            seasons: Seasons to optimize on
            k_range: Range of k values to test
            num_points: Number of points in sweep
            metric: Metric to optimize ("mae", "brier", "accuracy")
            base_params: Base parameters
        
        Returns:
            DecayOptimizationResult with optimal k and analysis
        """
        base_params = base_params or ModelParameters()
        
        # Generate k values to test
        k_min, k_max = k_range
        k_values = [k_min + i * (k_max - k_min) / (num_points - 1) 
                    for i in range(num_points)]
        
        # Run sweep
        sweep_results = self.backtester.parameter_sweep(
            "decay_rate", k_values, seasons, base_params
        )
        
        # Extract metrics
        mae_by_k = {k: r.spread_mae for k, r in sweep_results.items()}
        brier_by_k = {k: r.brier_score for k, r in sweep_results.items()}
        accuracy_by_k = {k: r.straight_up_accuracy for k, r in sweep_results.items()}
        
        # Find optimal based on metric
        if metric == "mae":
            optimal_k = min(mae_by_k.keys(), key=lambda k: mae_by_k[k])
        elif metric == "brier":
            optimal_k = min(brier_by_k.keys(), key=lambda k: brier_by_k[k])
        else:  # accuracy
            optimal_k = max(accuracy_by_k.keys(), key=lambda k: accuracy_by_k[k])
        
        # Get baseline results
        baseline_k = 0.15
        baseline_params = base_params.copy()
        baseline_params.decay_rate = baseline_k
        baseline_results = self.backtester.run_backtest(seasons, baseline_params)
        
        # Calculate improvement
        improvement_pct = ((baseline_results.spread_mae - mae_by_k[optimal_k]) 
                          / baseline_results.spread_mae * 100)
        
        # Estimate confidence interval (simple approach)
        # Find k values within 0.5% of optimal
        threshold = mae_by_k[optimal_k] * 1.005
        within_ci = [k for k, mae in mae_by_k.items() if mae <= threshold]
        ci_lower = min(within_ci)
        ci_upper = max(within_ci)
        
        return DecayOptimizationResult(
            optimal_k=optimal_k,
            optimal_mae=mae_by_k[optimal_k],
            optimal_brier=brier_by_k[optimal_k],
            k_values_tested=k_values,
            mae_by_k=mae_by_k,
            brier_by_k=brier_by_k,
            accuracy_by_k=accuracy_by_k,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            baseline_k=baseline_k,
            baseline_mae=baseline_results.spread_mae,
            improvement_pct=improvement_pct
        )


# =============================================================================
# STORY 9.4: HOME FIELD ADVANTAGE CALIBRATION
# =============================================================================

@dataclass
class HFACalibrationResult:
    """
    Results from HFA calibration.
    
    Story 9.4: Output of HFA optimization.
    """
    # Optimal values
    optimal_base_hfa: float
    optimal_altitude_bonus: float
    optimal_dome_penalty: float
    optimal_divisional_reduction: float
    
    # Empirical HFA by season
    hfa_by_season: Dict[int, float] = field(default_factory=dict)
    
    # Comparison
    assumed_hfa: float = 2.5
    empirical_hfa: float = 0.0
    
    # Team-specific HFA
    hfa_by_team: Dict[str, float] = field(default_factory=dict)
    
    def get_summary(self) -> str:
        """Generate summary report"""
        lines = [
            "=" * 60,
            "HOME FIELD ADVANTAGE CALIBRATION",
            "=" * 60,
            "",
            "OPTIMAL PARAMETERS",
            "-" * 40,
            f"  Base HFA: {self.optimal_base_hfa:.2f} pts (assumed: {self.assumed_hfa:.2f})",
            f"  Altitude Bonus: {self.optimal_altitude_bonus:.2f} pts",
            f"  Dome Outdoor Penalty: {self.optimal_dome_penalty:.2f} pts",
            f"  Divisional Reduction: {self.optimal_divisional_reduction:.2f} pts",
            "",
            "EMPIRICAL HFA BY SEASON",
            "-" * 40,
        ]
        
        for season, hfa in sorted(self.hfa_by_season.items()):
            lines.append(f"  {season}: {hfa:.2f} pts")
        
        lines.extend([
            "",
            f"  Average: {self.empirical_hfa:.2f} pts",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class HFACalibrator:
    """
    Calibrates home field advantage parameters.
    
    Story 9.4: HFA optimization.
    """
    
    def __init__(self, backtester: BacktestingFramework, 
                 repository: HistoricalDataRepository):
        self.backtester = backtester
        self.repository = repository
    
    def calibrate(self, seasons: List[int]) -> HFACalibrationResult:
        """
        Calibrate HFA parameters.
        
        Args:
            seasons: Seasons to use for calibration
        
        Returns:
            HFACalibrationResult with optimal parameters
        """
        result = HFACalibrationResult(
            optimal_base_hfa=0,
            optimal_altitude_bonus=0,
            optimal_dome_penalty=0,
            optimal_divisional_reduction=0
        )
        
        # Calculate empirical HFA by season
        for season in seasons:
            season_data = self.repository.get_season(season)
            if not season_data:
                continue
            
            home_wins = sum(1 for g in season_data.games if g.home_won and not g.is_playoff)
            total_games = sum(1 for g in season_data.games if not g.is_playoff)
            
            if total_games > 0:
                home_win_rate = home_wins / total_games
                # Convert win rate to points: (win_rate - 0.5) * 2 * typical_spread_factor
                hfa_estimate = (home_win_rate - 0.5) * 2 * 6  # ~6 points per 50% win prob
                result.hfa_by_season[season] = round(hfa_estimate, 2)
        
        if result.hfa_by_season:
            result.empirical_hfa = sum(result.hfa_by_season.values()) / len(result.hfa_by_season)
        
        # Optimize base HFA
        base_params = ModelParameters()
        hfa_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        hfa_results = self.backtester.parameter_sweep("base_hfa", hfa_values, seasons, base_params)
        
        optimal_hfa = min(hfa_results.keys(), key=lambda h: hfa_results[h].spread_mae)
        result.optimal_base_hfa = optimal_hfa
        
        # Optimize altitude bonus
        base_params.base_hfa = optimal_hfa
        alt_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        alt_results = self.backtester.parameter_sweep("altitude_bonus", alt_values, seasons, base_params)
        
        optimal_alt = min(alt_results.keys(), key=lambda a: alt_results[a].spread_mae)
        result.optimal_altitude_bonus = optimal_alt
        
        # Optimize dome penalty
        base_params.altitude_bonus = optimal_alt
        dome_values = [0.0, 0.2, 0.4, 0.6, 0.8]
        dome_results = self.backtester.parameter_sweep("dome_outdoor_penalty", dome_values, seasons, base_params)
        
        optimal_dome = min(dome_results.keys(), key=lambda d: dome_results[d].spread_mae)
        result.optimal_dome_penalty = optimal_dome
        
        # Optimize divisional reduction
        base_params.dome_outdoor_penalty = optimal_dome
        div_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        div_results = self.backtester.parameter_sweep("divisional_reduction", div_values, seasons, base_params)
        
        optimal_div = min(div_results.keys(), key=lambda d: div_results[d].spread_mae)
        result.optimal_divisional_reduction = optimal_div
        
        return result


# =============================================================================
# STORY 9.5: EPA SCALING OPTIMIZATION
# =============================================================================

@dataclass
class EPAScalingResult:
    """
    Results from EPA scaling optimization.
    
    Story 9.5: Output of EPA optimization.
    """
    optimal_scaling_factor: float
    optimal_mae: float
    
    # Sweep results
    factors_tested: List[float] = field(default_factory=list)
    mae_by_factor: Dict[float, float] = field(default_factory=dict)
    
    # Separate offense/defense
    optimal_offensive_weight: float = 1.0
    optimal_defensive_weight: float = 1.0
    
    def get_summary(self) -> str:
        """Generate summary report"""
        lines = [
            "=" * 60,
            "EPA SCALING OPTIMIZATION RESULTS",
            "=" * 60,
            "",
            f"Optimal Scaling Factor: {self.optimal_scaling_factor:.1f}",
            f"Optimal MAE: {self.optimal_mae:.3f}",
            "",
            f"Offensive Weight: {self.optimal_offensive_weight:.2f}",
            f"Defensive Weight: {self.optimal_defensive_weight:.2f}",
            "",
            "FACTORS TESTED",
            "-" * 40,
        ]
        
        for factor in sorted(self.factors_tested):
            mae = self.mae_by_factor.get(factor, 0)
            marker = " *" if factor == self.optimal_scaling_factor else ""
            lines.append(f"  {factor:.1f}: MAE={mae:.3f}{marker}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class EPAScalingOptimizer:
    """
    Optimizes EPA-to-points scaling factor.
    
    Story 9.5: Find optimal EPA scaling.
    """
    
    def __init__(self, backtester: BacktestingFramework):
        self.backtester = backtester
    
    def optimize(self, 
                 seasons: List[int],
                 factor_range: Tuple[float, float] = (15, 35),
                 base_params: Optional[ModelParameters] = None) -> EPAScalingResult:
        """
        Find optimal EPA scaling factor.
        
        Args:
            seasons: Seasons to optimize on
            factor_range: Range of factors to test
            base_params: Base parameters
        
        Returns:
            EPAScalingResult with optimal factor
        """
        base_params = base_params or ModelParameters()
        
        # Generate factors to test
        factors = list(range(int(factor_range[0]), int(factor_range[1]) + 1, 2))
        
        # Run sweep
        sweep_results = self.backtester.parameter_sweep(
            "epa_to_points_factor", factors, seasons, base_params
        )
        
        # Extract metrics
        mae_by_factor = {f: r.spread_mae for f, r in sweep_results.items()}
        
        # Find optimal
        optimal_factor = min(mae_by_factor.keys(), key=lambda f: mae_by_factor[f])
        
        return EPAScalingResult(
            optimal_scaling_factor=optimal_factor,
            optimal_mae=mae_by_factor[optimal_factor],
            factors_tested=factors,
            mae_by_factor=mae_by_factor
        )


# =============================================================================
# STORY 9.6: CROSS-VALIDATION FRAMEWORK
# =============================================================================

@dataclass
class CrossValidationResult:
    """
    Results from cross-validation.
    
    Story 9.6: Output of CV framework.
    """
    # Fold results
    fold_results: List[BacktestResults] = field(default_factory=list)
    fold_seasons: List[int] = field(default_factory=list)
    
    # Aggregated metrics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_mae: float = 0.0
    std_mae: float = 0.0
    mean_brier: float = 0.0
    std_brier: float = 0.0
    
    # Per-fold breakdown
    accuracy_by_fold: Dict[int, float] = field(default_factory=dict)
    mae_by_fold: Dict[int, float] = field(default_factory=dict)
    
    def calculate_aggregates(self) -> None:
        """Calculate aggregate statistics"""
        if not self.fold_results:
            return
        
        accuracies = [r.straight_up_accuracy for r in self.fold_results]
        maes = [r.spread_mae for r in self.fold_results]
        briers = [r.brier_score for r in self.fold_results]
        
        self.mean_accuracy = sum(accuracies) / len(accuracies)
        self.std_accuracy = self._std(accuracies)
        
        self.mean_mae = sum(maes) / len(maes)
        self.std_mae = self._std(maes)
        
        self.mean_brier = sum(briers) / len(briers)
        self.std_brier = self._std(briers)
        
        for i, (season, result) in enumerate(zip(self.fold_seasons, self.fold_results)):
            self.accuracy_by_fold[season] = result.straight_up_accuracy
            self.mae_by_fold[season] = result.spread_mae
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def get_summary(self) -> str:
        """Generate summary report"""
        lines = [
            "=" * 60,
            "CROSS-VALIDATION RESULTS",
            "=" * 60,
            "",
            f"Folds: {len(self.fold_results)}",
            f"Held-out Seasons: {self.fold_seasons}",
            "",
            "AGGREGATE METRICS",
            "-" * 40,
            f"  Accuracy: {self.mean_accuracy:.1%} ± {self.std_accuracy:.1%}",
            f"  MAE: {self.mean_mae:.3f} ± {self.std_mae:.3f}",
            f"  Brier: {self.mean_brier:.4f} ± {self.std_brier:.4f}",
            "",
            "PER-FOLD RESULTS",
            "-" * 40,
        ]
        
        for season in self.fold_seasons:
            acc = self.accuracy_by_fold.get(season, 0)
            mae = self.mae_by_fold.get(season, 0)
            lines.append(f"  {season}: Acc={acc:.1%}, MAE={mae:.3f}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class CrossValidator:
    """
    Leave-one-season-out cross-validation.
    
    Story 9.6: CV implementation.
    """
    
    def __init__(self, backtester: BacktestingFramework):
        self.backtester = backtester
    
    def leave_one_season_out(self, 
                              all_seasons: List[int],
                              params: ModelParameters) -> CrossValidationResult:
        """
        Run leave-one-season-out cross-validation.
        
        Args:
            all_seasons: All available seasons
            params: Parameters to validate
        
        Returns:
            CrossValidationResult with per-fold and aggregate metrics
        """
        result = CrossValidationResult()
        
        for held_out in all_seasons:
            # Train on all seasons except held_out
            train_seasons = [s for s in all_seasons if s != held_out]
            
            # Test on held_out
            test_results = self.backtester.run_backtest([held_out], params)
            
            result.fold_results.append(test_results)
            result.fold_seasons.append(held_out)
        
        result.calculate_aggregates()
        return result
    
    def k_fold(self, 
               all_seasons: List[int],
               k: int,
               params: ModelParameters) -> CrossValidationResult:
        """
        K-fold cross-validation on seasons.
        
        Args:
            all_seasons: All available seasons
            k: Number of folds
            params: Parameters to validate
        
        Returns:
            CrossValidationResult
        """
        result = CrossValidationResult()
        
        # Split seasons into k folds
        fold_size = len(all_seasons) // k
        
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(all_seasons)
            
            test_seasons = all_seasons[start:end]
            train_seasons = [s for s in all_seasons if s not in test_seasons]
            
            # Test on fold
            test_results = self.backtester.run_backtest(test_seasons, params)
            
            result.fold_results.append(test_results)
            result.fold_seasons.extend(test_seasons)
        
        result.calculate_aggregates()
        return result


# =============================================================================
# STORY 9.7: PARAMETER CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class ParameterConfig:
    """
    Versioned parameter configuration.
    
    Story 9.7: Configuration management.
    """
    version: str
    name: str
    description: str
    created_date: datetime
    
    # Parameters
    parameters: ModelParameters
    
    # Validation results
    validation_seasons: List[int] = field(default_factory=list)
    validation_accuracy: float = 0.0
    validation_mae: float = 0.0
    validation_brier: float = 0.0
    
    # Metadata
    is_active: bool = False
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date.isoformat(),
            "parameters": self.parameters.to_dict(),
            "validation_seasons": self.validation_seasons,
            "validation_accuracy": self.validation_accuracy,
            "validation_mae": self.validation_mae,
            "validation_brier": self.validation_brier,
            "is_active": self.is_active,
            "parent_version": self.parent_version
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterConfig':
        """Create from dictionary"""
        return cls(
            version=d["version"],
            name=d["name"],
            description=d["description"],
            created_date=datetime.fromisoformat(d["created_date"]),
            parameters=ModelParameters.from_dict(d["parameters"]),
            validation_seasons=d.get("validation_seasons", []),
            validation_accuracy=d.get("validation_accuracy", 0),
            validation_mae=d.get("validation_mae", 0),
            validation_brier=d.get("validation_brier", 0),
            is_active=d.get("is_active", False),
            parent_version=d.get("parent_version")
        )


class ConfigurationManager:
    """
    Manages versioned parameter configurations.
    
    Story 9.7: Configuration versioning and A/B testing.
    """
    
    def __init__(self):
        self.configs: Dict[str, ParameterConfig] = {}
        self._active_version: Optional[str] = None
        self._prediction_log: List[Dict] = []
    
    def add_config(self, config: ParameterConfig) -> None:
        """Add a configuration"""
        self.configs[config.version] = config
        
        if config.is_active:
            self.set_active(config.version)
    
    def get_config(self, version: str) -> Optional[ParameterConfig]:
        """Get a specific configuration"""
        return self.configs.get(version)
    
    def get_active(self) -> Optional[ParameterConfig]:
        """Get currently active configuration"""
        if self._active_version:
            return self.configs.get(self._active_version)
        return None
    
    def set_active(self, version: str) -> None:
        """Set active configuration"""
        if version not in self.configs:
            raise ValueError(f"Configuration {version} not found")
        
        # Deactivate current
        if self._active_version and self._active_version in self.configs:
            self.configs[self._active_version].is_active = False
        
        # Activate new
        self._active_version = version
        self.configs[version].is_active = True
    
    def create_config(self,
                      name: str,
                      description: str,
                      params: ModelParameters,
                      parent_version: Optional[str] = None) -> ParameterConfig:
        """Create a new configuration"""
        # Generate version number
        version_num = len(self.configs) + 1
        version = f"v{version_num}.0"
        
        config = ParameterConfig(
            version=version,
            name=name,
            description=description,
            created_date=datetime.now(),
            parameters=params,
            parent_version=parent_version
        )
        
        self.add_config(config)
        return config
    
    def log_prediction(self, version: str, game_id: str, 
                       prediction: Dict[str, float]) -> None:
        """Log a prediction with its configuration version"""
        self._prediction_log.append({
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "game_id": game_id,
            "prediction": prediction
        })
    
    def compare_configs(self, version_a: str, version_b: str,
                        backtester: BacktestingFramework,
                        seasons: List[int]) -> Dict[str, any]:
        """Compare two configurations"""
        config_a = self.get_config(version_a)
        config_b = self.get_config(version_b)
        
        if not config_a or not config_b:
            raise ValueError("Configuration not found")
        
        results_a = backtester.run_backtest(seasons, config_a.parameters)
        results_b = backtester.run_backtest(seasons, config_b.parameters)
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "accuracy_diff": results_b.straight_up_accuracy - results_a.straight_up_accuracy,
            "mae_diff": results_b.spread_mae - results_a.spread_mae,
            "brier_diff": results_b.brier_score - results_a.brier_score,
            "results_a": results_a,
            "results_b": results_b
        }
    
    def export_json(self) -> str:
        """Export all configurations to JSON"""
        data = {
            "active_version": self._active_version,
            "configs": {v: c.to_dict() for v, c in self.configs.items()}
        }
        return json.dumps(data, indent=2)
    
    def import_json(self, json_str: str) -> None:
        """Import configurations from JSON"""
        data = json.loads(json_str)
        
        for version, config_dict in data.get("configs", {}).items():
            config = ParameterConfig.from_dict(config_dict)
            self.configs[version] = config
        
        active = data.get("active_version")
        if active and active in self.configs:
            self.set_active(active)
    
    def get_history(self) -> List[Dict]:
        """Get configuration version history"""
        history = []
        for version, config in sorted(self.configs.items()):
            history.append({
                "version": version,
                "name": config.name,
                "created": config.created_date.isoformat(),
                "is_active": config.is_active,
                "accuracy": config.validation_accuracy,
                "parent": config.parent_version
            })
        return history


# =============================================================================
# INTEGRATED CALIBRATION PIPELINE
# =============================================================================

class CalibrationPipeline:
    """
    Integrated calibration pipeline running all optimizations.
    
    Combines all Story 9.x components into a unified workflow.
    """
    
    def __init__(self, repository: HistoricalDataRepository):
        self.repository = repository
        self.backtester = BacktestingFramework(repository)
        self.config_manager = ConfigurationManager()
    
    def run_full_calibration(self, 
                             train_seasons: List[int],
                             test_seasons: List[int]) -> Dict[str, any]:
        """
        Run full calibration pipeline.
        
        Args:
            train_seasons: Seasons to optimize on
            test_seasons: Seasons to validate on
        
        Returns:
            Dict with all optimization results
        """
        results = {}
        
        print("=" * 60)
        print("RUNNING FULL CALIBRATION PIPELINE")
        print("=" * 60)
        print(f"Train seasons: {train_seasons}")
        print(f"Test seasons: {test_seasons}")
        print()
        
        # 1. Baseline performance
        print("Step 1: Establishing baseline...")
        baseline_params = ModelParameters()
        baseline_results = self.backtester.run_backtest(train_seasons, baseline_params)
        results["baseline"] = baseline_results
        print(f"  Baseline MAE: {baseline_results.spread_mae:.3f}")
        print(f"  Baseline Accuracy: {baseline_results.straight_up_accuracy:.1%}")
        print()
        
        # 2. Decay curve optimization
        print("Step 2: Optimizing decay curve...")
        decay_optimizer = DecayCurveOptimizer(self.backtester)
        decay_results = decay_optimizer.optimize(train_seasons)
        results["decay"] = decay_results
        print(f"  Optimal k: {decay_results.optimal_k:.3f}")
        print(f"  Improvement: {decay_results.improvement_pct:.1f}%")
        print()
        
        # 3. HFA calibration
        print("Step 3: Calibrating home field advantage...")
        hfa_calibrator = HFACalibrator(self.backtester, self.repository)
        hfa_results = hfa_calibrator.calibrate(train_seasons)
        results["hfa"] = hfa_results
        print(f"  Optimal HFA: {hfa_results.optimal_base_hfa:.2f}")
        print(f"  Empirical HFA: {hfa_results.empirical_hfa:.2f}")
        print()
        
        # 4. EPA scaling optimization
        print("Step 4: Optimizing EPA scaling...")
        epa_optimizer = EPAScalingOptimizer(self.backtester)
        epa_results = epa_optimizer.optimize(train_seasons)
        results["epa"] = epa_results
        print(f"  Optimal factor: {epa_results.optimal_scaling_factor:.1f}")
        print()
        
        # 5. Create optimized parameters
        print("Step 5: Creating optimized configuration...")
        optimized_params = ModelParameters(
            decay_rate=decay_results.optimal_k,
            base_hfa=hfa_results.optimal_base_hfa,
            altitude_bonus=hfa_results.optimal_altitude_bonus,
            dome_outdoor_penalty=hfa_results.optimal_dome_penalty,
            divisional_reduction=hfa_results.optimal_divisional_reduction,
            epa_to_points_factor=epa_results.optimal_scaling_factor
        )
        
        # 6. Cross-validation
        print("Step 6: Running cross-validation...")
        cv = CrossValidator(self.backtester)
        all_seasons = train_seasons + test_seasons
        cv_results = cv.leave_one_season_out(all_seasons, optimized_params)
        results["cross_validation"] = cv_results
        print(f"  CV Accuracy: {cv_results.mean_accuracy:.1%} ± {cv_results.std_accuracy:.1%}")
        print()
        
        # 7. Final test set validation
        print("Step 7: Validating on test set...")
        test_results = self.backtester.run_backtest(test_seasons, optimized_params)
        results["test"] = test_results
        print(f"  Test Accuracy: {test_results.straight_up_accuracy:.1%}")
        print(f"  Test MAE: {test_results.spread_mae:.3f}")
        print()
        
        # 8. Save configuration
        print("Step 8: Saving configuration...")
        config = self.config_manager.create_config(
            name="Calibrated Parameters",
            description=f"Calibrated on {train_seasons}, validated on {test_seasons}",
            params=optimized_params
        )
        config.validation_seasons = test_seasons
        config.validation_accuracy = test_results.straight_up_accuracy
        config.validation_mae = test_results.spread_mae
        config.validation_brier = test_results.brier_score
        self.config_manager.set_active(config.version)
        results["config"] = config
        print(f"  Saved as {config.version}")
        print()
        
        # Summary
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"Baseline → Optimized:")
        print(f"  Accuracy: {baseline_results.straight_up_accuracy:.1%} → {test_results.straight_up_accuracy:.1%}")
        print(f"  MAE: {baseline_results.spread_mae:.3f} → {test_results.spread_mae:.3f}")
        print(f"  Brier: {baseline_results.brier_score:.4f} → {test_results.brier_score:.4f}")
        print()
        
        return results


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo_epic_9():
    """Demonstrate Epic 9 features"""
    print("=" * 70)
    print("NFL PREDICTION MODEL - EPIC 9: HISTORICAL CALIBRATION DEMO")
    print("=" * 70)
    print()
    
    # Generate mock historical data
    print("Generating mock historical data...")
    generator = MockHistoricalDataGenerator(seed=42)
    repository = HistoricalDataRepository()
    
    for season in range(2019, 2025):
        season_data = generator.generate_season(season)
        repository.add_season(season_data)
        print(f"  {season}: {len(season_data.games)} games")
    
    print(f"\nTotal: {repository.total_games} games across {len(repository.available_seasons)} seasons")
    print()
    
    # Initialize pipeline
    pipeline = CalibrationPipeline(repository)
    
    # Run calibration
    train_seasons = [2019, 2020, 2021, 2022]
    test_seasons = [2023, 2024]
    
    results = pipeline.run_full_calibration(train_seasons, test_seasons)
    
    # Show detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    print("\n" + results["decay"].get_summary())
    print("\n" + results["hfa"].get_summary())
    print("\n" + results["epa"].get_summary())
    print("\n" + results["cross_validation"].get_summary())
    print("\n" + results["test"].get_summary())
    
    # Show configuration management
    print("\n" + "=" * 70)
    print("CONFIGURATION MANAGEMENT")
    print("=" * 70)
    
    # Create additional configs for comparison
    baseline_config = pipeline.config_manager.create_config(
        name="Baseline",
        description="Default parameters",
        params=ModelParameters()
    )
    
    print("\nConfiguration History:")
    for entry in pipeline.config_manager.get_history():
        active = " (ACTIVE)" if entry["is_active"] else ""
        print(f"  {entry['version']}: {entry['name']}{active}")
    
    # Export configuration
    config_json = pipeline.config_manager.export_json()
    print(f"\nConfiguration exported ({len(config_json)} bytes)")
    
    # Feature summary
    print("\n" + "=" * 70)
    print("""
Epic 9 Features Implemented:
  ✓ Story 9.1: Historical Data Repository
    - HistoricalGame dataclass with full game data
    - SeasonData collection with team ratings
    - Query interface with filters
    - MockHistoricalDataGenerator for testing
    
  ✓ Story 9.2: Backtesting Framework
    - PredictionResult for individual games
    - BacktestResults with comprehensive metrics
    - SimplePredictor with configurable parameters
    - Parameter sweep functionality
    
  ✓ Story 9.3: Decay Curve Optimization
    - DecayCurveOptimizer
    - K value sweep from 0.05-0.30
    - Confidence interval estimation
    - Improvement calculation vs baseline
    
  ✓ Story 9.4: Home Field Advantage Calibration
    - HFACalibrator
    - Empirical HFA by season
    - Optimization of base HFA, altitude, dome, divisional
    
  ✓ Story 9.5: EPA Scaling Optimization
    - EPAScalingOptimizer
    - Factor sweep from 15-35
    - Separate offense/defense weights
    
  ✓ Story 9.6: Cross-Validation Framework
    - CrossValidator
    - Leave-one-season-out CV
    - K-fold CV option
    - Aggregate metrics with std deviation
    
  ✓ Story 9.7: Parameter Configuration Management
    - ParameterConfig versioning
    - ConfigurationManager with A/B testing
    - JSON export/import
    - Prediction logging with version tracking
    
Integration Notes:
  - CalibrationPipeline combines all components
  - Can use Epic 7/8 if available
  - Produces production-ready configurations
  - Full audit trail of parameter changes
""")


if __name__ == "__main__":
    demo_epic_9()
