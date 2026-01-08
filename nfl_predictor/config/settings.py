"""
Settings - Application settings and configuration.

This module contains:
- Settings: Application settings dataclass
- get_settings: Get current settings instance
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os
import json


@dataclass
class DataSourceSettings:
    """
    Data source configuration.

    Attributes:
        epa_source: EPA data source type
        schedule_source: Schedule data source type
        vegas_source: Vegas lines source type
        cache_enabled: Whether to cache data
        cache_ttl_hours: Cache time-to-live in hours
    """
    epa_source: str = "nflfastR"
    schedule_source: str = "espn"
    vegas_source: str = "espn"
    cache_enabled: bool = True
    cache_ttl_hours: int = 24


@dataclass
class OutputSettings:
    """
    Output configuration.

    Attributes:
        default_format: Default output format
        output_dir: Default output directory
        include_timestamps: Include timestamps in filenames
        pretty_print: Pretty print JSON output
    """
    default_format: str = "json"
    output_dir: str = "./output"
    include_timestamps: bool = True
    pretty_print: bool = True


@dataclass
class LoggingSettings:
    """
    Logging configuration.

    Attributes:
        level: Log level
        format: Log format string
        log_to_file: Whether to log to file
        log_file: Log file path
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: str = "./logs/nfl_predictor.log"


@dataclass
class Settings:
    """
    Application settings.

    This dataclass contains all configurable application settings
    for the NFL predictor system.

    Attributes:
        current_season: Current NFL season year
        data_sources: Data source configuration
        output: Output configuration
        logging: Logging configuration
        debug: Debug mode flag
    """
    current_season: int = 2025
    data_sources: DataSourceSettings = field(default_factory=DataSourceSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    debug: bool = False

    # Path settings
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    config_dir: str = "./config"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "current_season": self.current_season,
            "debug": self.debug,
            "data_dir": self.data_dir,
            "cache_dir": self.cache_dir,
            "config_dir": self.config_dir,
            "data_sources": {
                "epa_source": self.data_sources.epa_source,
                "schedule_source": self.data_sources.schedule_source,
                "vegas_source": self.data_sources.vegas_source,
                "cache_enabled": self.data_sources.cache_enabled,
                "cache_ttl_hours": self.data_sources.cache_ttl_hours,
            },
            "output": {
                "default_format": self.output.default_format,
                "output_dir": self.output.output_dir,
                "include_timestamps": self.output.include_timestamps,
                "pretty_print": self.output.pretty_print,
            },
            "logging": {
                "level": self.logging.level,
                "log_to_file": self.logging.log_to_file,
                "log_file": self.logging.log_file,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary."""
        settings = cls()

        settings.current_season = data.get("current_season", 2025)
        settings.debug = data.get("debug", False)
        settings.data_dir = data.get("data_dir", "./data")
        settings.cache_dir = data.get("cache_dir", "./cache")
        settings.config_dir = data.get("config_dir", "./config")

        if "data_sources" in data:
            ds = data["data_sources"]
            settings.data_sources = DataSourceSettings(
                epa_source=ds.get("epa_source", "nflfastR"),
                schedule_source=ds.get("schedule_source", "espn"),
                vegas_source=ds.get("vegas_source", "espn"),
                cache_enabled=ds.get("cache_enabled", True),
                cache_ttl_hours=ds.get("cache_ttl_hours", 24),
            )

        if "output" in data:
            out = data["output"]
            settings.output = OutputSettings(
                default_format=out.get("default_format", "json"),
                output_dir=out.get("output_dir", "./output"),
                include_timestamps=out.get("include_timestamps", True),
                pretty_print=out.get("pretty_print", True),
            )

        if "logging" in data:
            log = data["logging"]
            settings.logging = LoggingSettings(
                level=log.get("level", "INFO"),
                log_to_file=log.get("log_to_file", False),
                log_file=log.get("log_file", "./logs/nfl_predictor.log"),
            )

        return settings

    def save_to_file(self, filepath: str) -> None:
        """Save settings to JSON file."""
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Settings':
        """Load settings from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_env(cls) -> 'Settings':
        """Load settings from environment variables."""
        settings = cls()

        # Override from environment
        if "NFL_SEASON" in os.environ:
            settings.current_season = int(os.environ["NFL_SEASON"])
        if "NFL_DEBUG" in os.environ:
            settings.debug = os.environ["NFL_DEBUG"].lower() in ("true", "1", "yes")
        if "NFL_DATA_DIR" in os.environ:
            settings.data_dir = os.environ["NFL_DATA_DIR"]
        if "NFL_OUTPUT_DIR" in os.environ:
            settings.output.output_dir = os.environ["NFL_OUTPUT_DIR"]
        if "NFL_LOG_LEVEL" in os.environ:
            settings.logging.level = os.environ["NFL_LOG_LEVEL"]

        return settings

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        for dir_path in [self.data_dir, self.cache_dir, self.config_dir, self.output.output_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the current settings instance.

    Returns:
        Current Settings instance
    """
    global _settings

    if _settings is None:
        # Try to load from file, otherwise use defaults
        config_file = Path("./config/settings.json")
        if config_file.exists():
            _settings = Settings.load_from_file(str(config_file))
        else:
            _settings = Settings.load_from_env()

    return _settings


def set_settings(settings: Settings) -> None:
    """
    Set the global settings instance.

    Args:
        settings: Settings instance to use
    """
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to None, forcing reload on next access."""
    global _settings
    _settings = None
