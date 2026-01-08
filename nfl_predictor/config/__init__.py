"""
Config package - Configuration and parameter management.

This package contains:
- parameters.py: Model parameters and tuning values
- settings.py: Application settings
"""

from nfl_predictor.config.parameters import (
    ModelParameters,
    DecayParameters,
    HomeFieldParameters,
    EPAParameters,
    DEFAULT_PARAMETERS,
)
from nfl_predictor.config.settings import (
    Settings,
    get_settings,
)

__all__ = [
    "ModelParameters",
    "DecayParameters",
    "HomeFieldParameters",
    "EPAParameters",
    "DEFAULT_PARAMETERS",
    "Settings",
    "get_settings",
]
