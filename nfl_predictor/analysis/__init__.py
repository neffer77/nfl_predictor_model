"""
Analysis package - Backtesting, calibration, and performance tracking.

This package contains:
- backtesting.py: Season backtester and result tracking
- metrics.py: Accuracy and calibration metrics
- tracking.py: Prediction result trackers
- calibration/: Parameter calibration modules
"""

from nfl_predictor.analysis.backtesting import (
    PredictionResult,
    PredictionResultTracker,
    SeasonBacktester,
)
from nfl_predictor.analysis.metrics import (
    AccuracyMetrics,
    CalibrationAnalysis,
)

__all__ = [
    "PredictionResult",
    "PredictionResultTracker",
    "SeasonBacktester",
    "AccuracyMetrics",
    "CalibrationAnalysis",
]
