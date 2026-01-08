"""
Win Probability Models - Calculate win probabilities from rating differentials.

This module contains:
- WinProbabilityModel: Abstract base class
- LogisticWinProbability: Logistic regression-based model (recommended)
- LinearWinProbability: Simple linear model with caps
"""

from abc import ABC, abstractmethod


class WinProbabilityModel(ABC):
    """Abstract base class for win probability models."""

    @abstractmethod
    def calculate(self, rating_diff: float) -> float:
        """
        Calculate win probability from rating differential.

        Args:
            rating_diff: Home team rating - Away team rating

        Returns:
            Home team win probability (0.0 to 1.0)
        """
        pass


class LogisticWinProbability(WinProbabilityModel):
    """
    Logistic regression-based win probability model.

    Uses the formula: P(home_win) = 1 / (1 + 10^(-diff/scale))

    This is similar to Elo rating conversion.
    """

    def __init__(self, scale: float = 6.0):
        """
        Initialize with scale factor.

        Args:
            scale: Points per "rating unit" for probability conversion.
                   Higher scale = ratings have less impact on probability.
                   Default 6.0 means ~6 points = 1 "unit" difference.
        """
        self.scale = scale

    def calculate(self, rating_diff: float) -> float:
        """Calculate win probability using logistic function."""
        # Prevent overflow for extreme values
        if rating_diff > 30:
            return 0.99
        if rating_diff < -30:
            return 0.01

        prob = 1.0 / (1.0 + 10 ** (-rating_diff / self.scale))
        return round(prob, 4)


class LinearWinProbability(WinProbabilityModel):
    """
    Simple linear win probability model.

    Caps at 0.05 and 0.95 to avoid extreme probabilities.
    """

    def __init__(self, points_per_percent: float = 0.25):
        """
        Initialize with conversion rate.

        Args:
            points_per_percent: Rating points per 1% win probability.
                               Default 0.25 means 4 points = 16% swing.
        """
        self.points_per_percent = points_per_percent

    def calculate(self, rating_diff: float) -> float:
        """Calculate win probability using linear conversion."""
        base_prob = 0.50
        adjustment = rating_diff / (100 * self.points_per_percent)
        prob = base_prob + adjustment

        # Cap between 5% and 95%
        return round(max(0.05, min(0.95, prob)), 4)
