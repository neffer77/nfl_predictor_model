"""
NFL Prediction Model - Epic 7: Weather Adjustments

This module implements weather-based adjustments for NFL game predictions.
Weather significantly impacts NFL games - rain reduces passing efficiency,
wind affects kicking and deep throws, and extreme cold impacts performance.

Epic 7 Stories:
- Story 7.1: Weather Data Model
- Story 7.2: Weather Data Fetcher  
- Story 7.3: Weather Impact Model
- Story 7.4: Team Weather Profile
- Story 7.5: Weather-Adjusted Predictions
- Story 7.6: Weather Tracking & Validation

Author: Connor's NFL Prediction System
Version: 7.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import math
import random


# =============================================================================
# STORY 7.1: WEATHER DATA MODEL
# =============================================================================

class WeatherCondition(Enum):
    """Primary weather conditions"""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    FOG = "fog"
    MIST = "mist"
    LIGHT_RAIN = "light_rain"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    LIGHT_SNOW = "light_snow"
    SNOW = "snow"
    HEAVY_SNOW = "heavy_snow"
    SLEET = "sleet"
    FREEZING_RAIN = "freezing_rain"


class PrecipitationType(Enum):
    """Type of precipitation"""
    NONE = "none"
    RAIN = "rain"
    SNOW = "snow"
    SLEET = "sleet"
    FREEZING_RAIN = "freezing_rain"
    MIXED = "mixed"


class PrecipitationIntensity(Enum):
    """Intensity of precipitation"""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


class WindDirection(Enum):
    """Cardinal wind directions"""
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"
    CALM = "CALM"


class RoofStatus(Enum):
    """Stadium roof status"""
    OPEN = "open"
    CLOSED = "closed"
    DOME = "dome"  # Fixed dome, always closed
    OUTDOOR = "outdoor"  # No roof


class WeatherSeverity(Enum):
    """Overall weather severity for game impact"""
    IDEAL = "ideal"  # 50-70°F, no wind/precip
    GOOD = "good"  # Minor deviations
    MODERATE = "moderate"  # Some impact expected
    ADVERSE = "adverse"  # Significant impact
    SEVERE = "severe"  # Major game impact


@dataclass
class GameWeather:
    """
    Complete weather data for an NFL game.
    
    Story 7.1: Core weather data model with all relevant conditions
    for predicting game impact.
    """
    # Game identification
    game_id: str
    game_datetime: datetime
    stadium_name: str
    
    # Temperature
    temperature_f: float  # Fahrenheit
    feels_like_f: float  # Wind chill / heat index
    humidity_percent: float  # 0-100
    
    # Wind
    wind_speed_mph: float
    wind_gust_mph: Optional[float] = None
    wind_direction: WindDirection = WindDirection.CALM
    
    # Precipitation
    condition: WeatherCondition = WeatherCondition.CLEAR
    precip_type: PrecipitationType = PrecipitationType.NONE
    precip_intensity: PrecipitationIntensity = PrecipitationIntensity.NONE
    precip_probability: float = 0.0  # 0-100
    precip_amount_inches: float = 0.0  # Expected accumulation
    
    # Visibility & Pressure
    visibility_miles: float = 10.0
    pressure_mb: float = 1013.25
    
    # Venue
    is_dome: bool = False
    roof_status: RoofStatus = RoofStatus.OUTDOOR
    
    # Metadata
    forecast_updated: Optional[datetime] = None
    is_forecast: bool = True  # vs actual observed
    data_source: str = "unknown"
    
    def __post_init__(self):
        """Validate weather data"""
        # Clamp values to reasonable ranges
        self.temperature_f = max(-40, min(130, self.temperature_f))
        self.feels_like_f = max(-60, min(140, self.feels_like_f))
        self.humidity_percent = max(0, min(100, self.humidity_percent))
        self.wind_speed_mph = max(0, min(100, self.wind_speed_mph))
        self.precip_probability = max(0, min(100, self.precip_probability))
        self.visibility_miles = max(0, min(20, self.visibility_miles))
        
        if self.wind_gust_mph is not None:
            self.wind_gust_mph = max(self.wind_speed_mph, self.wind_gust_mph)
    
    @property
    def effective_wind(self) -> float:
        """Effective wind speed accounting for gusts"""
        if self.wind_gust_mph:
            return (self.wind_speed_mph * 0.7) + (self.wind_gust_mph * 0.3)
        return self.wind_speed_mph
    
    @property
    def is_cold(self) -> bool:
        """Temperature below 40°F"""
        return self.feels_like_f < 40
    
    @property
    def is_very_cold(self) -> bool:
        """Temperature below 20°F"""
        return self.feels_like_f < 20
    
    @property
    def is_freezing(self) -> bool:
        """Temperature below 32°F"""
        return self.feels_like_f < 32
    
    @property
    def is_hot(self) -> bool:
        """Feels like above 85°F"""
        return self.feels_like_f > 85
    
    @property
    def is_windy(self) -> bool:
        """Wind speed above 15 mph"""
        return self.effective_wind > 15
    
    @property
    def is_very_windy(self) -> bool:
        """Wind speed above 20 mph"""
        return self.effective_wind > 20
    
    @property
    def has_precipitation(self) -> bool:
        """Any active precipitation"""
        return self.precip_type != PrecipitationType.NONE and self.precip_intensity != PrecipitationIntensity.NONE
    
    @property
    def has_snow(self) -> bool:
        """Snow or freezing precipitation"""
        return self.precip_type in [PrecipitationType.SNOW, PrecipitationType.SLEET, PrecipitationType.FREEZING_RAIN]
    
    @property
    def is_indoor(self) -> bool:
        """Game is effectively indoors"""
        return self.is_dome or self.roof_status in [RoofStatus.DOME, RoofStatus.CLOSED]
    
    @property
    def severity(self) -> WeatherSeverity:
        """Calculate overall weather severity"""
        if self.is_indoor:
            return WeatherSeverity.IDEAL
        
        severity_score = 0
        
        # Temperature impact
        if self.is_very_cold:
            severity_score += 3
        elif self.is_cold:
            severity_score += 1
        elif self.is_hot:
            severity_score += 1
        
        # Wind impact
        if self.effective_wind > 25:
            severity_score += 3
        elif self.is_very_windy:
            severity_score += 2
        elif self.is_windy:
            severity_score += 1
        
        # Precipitation impact
        if self.precip_intensity == PrecipitationIntensity.HEAVY:
            severity_score += 3
        elif self.precip_intensity == PrecipitationIntensity.MODERATE:
            severity_score += 2
        elif self.precip_intensity == PrecipitationIntensity.LIGHT:
            severity_score += 1
        
        # Snow is worse than rain
        if self.has_snow:
            severity_score += 1
        
        # Visibility impact
        if self.visibility_miles < 0.5:
            severity_score += 2
        elif self.visibility_miles < 2:
            severity_score += 1
        
        # Map score to severity
        if severity_score == 0:
            return WeatherSeverity.IDEAL
        elif severity_score <= 2:
            return WeatherSeverity.GOOD
        elif severity_score <= 4:
            return WeatherSeverity.MODERATE
        elif severity_score <= 6:
            return WeatherSeverity.ADVERSE
        else:
            return WeatherSeverity.SEVERE
    
    def get_summary(self) -> str:
        """Human-readable weather summary"""
        if self.is_indoor:
            return f"Indoors ({self.roof_status.value})"
        
        parts = [f"{self.temperature_f:.0f}°F"]
        
        if self.feels_like_f != self.temperature_f:
            parts.append(f"(feels {self.feels_like_f:.0f}°F)")
        
        if self.effective_wind > 5:
            parts.append(f"Wind {self.wind_speed_mph:.0f} mph {self.wind_direction.value}")
        
        if self.has_precipitation:
            parts.append(f"{self.precip_intensity.value.title()} {self.precip_type.value}")
        elif self.precip_probability > 30:
            parts.append(f"{self.precip_probability:.0f}% chance {self.precip_type.value}")
        
        return ", ".join(parts)


@dataclass 
class StadiumInfo:
    """Stadium location and venue information"""
    team: str
    stadium_name: str
    city: str
    state: str
    latitude: float
    longitude: float
    elevation_ft: float
    is_dome: bool
    has_retractable_roof: bool
    capacity: int
    surface: str  # "grass", "turf"
    
    @property
    def is_high_altitude(self) -> bool:
        """Denver's altitude (5,280 ft) affects kicking"""
        return self.elevation_ft > 4000


# Complete stadium data for all 32 NFL teams
STADIUM_DATA: Dict[str, StadiumInfo] = {
    "ARI": StadiumInfo("ARI", "State Farm Stadium", "Glendale", "AZ", 33.5277, -112.2626, 1100, False, True, 63400, "grass"),
    "ATL": StadiumInfo("ATL", "Mercedes-Benz Stadium", "Atlanta", "GA", 33.7553, -84.4006, 1050, True, True, 71000, "turf"),
    "BAL": StadiumInfo("BAL", "M&T Bank Stadium", "Baltimore", "MD", 39.2780, -76.6227, 30, False, False, 71008, "grass"),
    "BUF": StadiumInfo("BUF", "Highmark Stadium", "Orchard Park", "NY", 42.7738, -78.7870, 600, False, False, 71608, "turf"),
    "CAR": StadiumInfo("CAR", "Bank of America Stadium", "Charlotte", "NC", 35.2258, -80.8528, 751, False, False, 74867, "grass"),
    "CHI": StadiumInfo("CHI", "Soldier Field", "Chicago", "IL", 41.8623, -87.6167, 597, False, False, 61500, "grass"),
    "CIN": StadiumInfo("CIN", "Paycor Stadium", "Cincinnati", "OH", 39.0954, -84.5160, 490, False, False, 65515, "turf"),
    "CLE": StadiumInfo("CLE", "Cleveland Browns Stadium", "Cleveland", "OH", 41.5061, -81.6995, 580, False, False, 67431, "grass"),
    "DAL": StadiumInfo("DAL", "AT&T Stadium", "Arlington", "TX", 32.7473, -97.0945, 600, False, True, 80000, "turf"),
    "DEN": StadiumInfo("DEN", "Empower Field at Mile High", "Denver", "CO", 39.7439, -105.0201, 5280, False, False, 76125, "grass"),
    "DET": StadiumInfo("DET", "Ford Field", "Detroit", "MI", 42.3400, -83.0456, 600, True, False, 65000, "turf"),
    "GB": StadiumInfo("GB", "Lambeau Field", "Green Bay", "WI", 44.5013, -88.0622, 640, False, False, 81441, "grass"),
    "HOU": StadiumInfo("HOU", "NRG Stadium", "Houston", "TX", 29.6847, -95.4107, 50, False, True, 72220, "turf"),
    "IND": StadiumInfo("IND", "Lucas Oil Stadium", "Indianapolis", "IN", 39.7601, -86.1639, 715, True, True, 67000, "turf"),
    "JAX": StadiumInfo("JAX", "EverBank Stadium", "Jacksonville", "FL", 30.3239, -81.6373, 10, False, False, 67814, "grass"),
    "KC": StadiumInfo("KC", "GEHA Field at Arrowhead Stadium", "Kansas City", "MO", 39.0489, -94.4839, 800, False, False, 76416, "grass"),
    "LAC": StadiumInfo("LAC", "SoFi Stadium", "Inglewood", "CA", 33.9535, -118.3392, 100, False, True, 70240, "turf"),
    "LAR": StadiumInfo("LAR", "SoFi Stadium", "Inglewood", "CA", 33.9535, -118.3392, 100, False, True, 70240, "turf"),
    "LV": StadiumInfo("LV", "Allegiant Stadium", "Las Vegas", "NV", 36.0909, -115.1833, 2030, True, False, 65000, "turf"),
    "MIA": StadiumInfo("MIA", "Hard Rock Stadium", "Miami Gardens", "FL", 25.9580, -80.2389, 10, False, False, 64767, "grass"),
    "MIN": StadiumInfo("MIN", "U.S. Bank Stadium", "Minneapolis", "MN", 44.9736, -93.2575, 830, True, False, 66860, "turf"),
    "NE": StadiumInfo("NE", "Gillette Stadium", "Foxborough", "MA", 42.0909, -71.2643, 260, False, False, 65878, "turf"),
    "NO": StadiumInfo("NO", "Caesars Superdome", "New Orleans", "LA", 29.9511, -90.0812, 3, True, False, 73208, "turf"),
    "NYG": StadiumInfo("NYG", "MetLife Stadium", "East Rutherford", "NJ", 40.8128, -74.0742, 10, False, False, 82500, "turf"),
    "NYJ": StadiumInfo("NYJ", "MetLife Stadium", "East Rutherford", "NJ", 40.8128, -74.0742, 10, False, False, 82500, "turf"),
    "PHI": StadiumInfo("PHI", "Lincoln Financial Field", "Philadelphia", "PA", 39.9008, -75.1675, 40, False, False, 69796, "grass"),
    "PIT": StadiumInfo("PIT", "Acrisure Stadium", "Pittsburgh", "PA", 40.4468, -80.0158, 730, False, False, 68400, "grass"),
    "SEA": StadiumInfo("SEA", "Lumen Field", "Seattle", "WA", 47.5952, -122.3316, 20, False, False, 68740, "turf"),
    "SF": StadiumInfo("SF", "Levi's Stadium", "Santa Clara", "CA", 37.4033, -121.9694, 50, False, False, 68500, "grass"),
    "TB": StadiumInfo("TB", "Raymond James Stadium", "Tampa", "FL", 27.9759, -82.5033, 30, False, False, 65618, "grass"),
    "TEN": StadiumInfo("TEN", "Nissan Stadium", "Nashville", "TN", 36.1665, -86.7713, 385, False, False, 69143, "grass"),
    "WAS": StadiumInfo("WAS", "Northwest Stadium", "Landover", "MD", 38.9076, -76.8645, 160, False, False, 67617, "grass"),
}


# =============================================================================
# STORY 7.2: WEATHER DATA FETCHER
# =============================================================================

class WeatherDataSource(ABC):
    """Abstract base class for weather data sources"""
    
    @abstractmethod
    def get_game_weather(self, game_id: str, stadium: StadiumInfo, 
                         game_time: datetime) -> Optional[GameWeather]:
        """Fetch weather for a specific game"""
        pass
    
    @abstractmethod
    def get_historical_weather(self, stadium: StadiumInfo, 
                               date: date) -> Optional[GameWeather]:
        """Fetch historical weather for backtesting"""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source"""
        pass


class MockWeatherDataSource(WeatherDataSource):
    """
    Mock weather data source for testing.
    
    Story 7.2: Generates realistic weather based on:
    - Geographic location (latitude)
    - Time of year (month)
    - Stadium type (dome/outdoor)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._cache: Dict[str, GameWeather] = {}
    
    @property
    def source_name(self) -> str:
        return "MockWeatherData"
    
    def get_game_weather(self, game_id: str, stadium: StadiumInfo,
                         game_time: datetime) -> Optional[GameWeather]:
        """Generate realistic weather for a game"""
        
        # Check cache
        cache_key = f"{game_id}_{stadium.team}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Dome games always have perfect conditions
        if stadium.is_dome:
            weather = GameWeather(
                game_id=game_id,
                game_datetime=game_time,
                stadium_name=stadium.stadium_name,
                temperature_f=72.0,
                feels_like_f=72.0,
                humidity_percent=45.0,
                wind_speed_mph=0.0,
                is_dome=True,
                roof_status=RoofStatus.DOME,
                data_source=self.source_name
            )
            self._cache[cache_key] = weather
            return weather
        
        # Generate outdoor weather based on location and date
        weather = self._generate_weather(game_id, stadium, game_time)
        self._cache[cache_key] = weather
        return weather
    
    def get_historical_weather(self, stadium: StadiumInfo,
                               historical_date: date) -> Optional[GameWeather]:
        """Generate historical weather (same logic as forecast)"""
        game_time = datetime.combine(historical_date, datetime.min.time().replace(hour=13))
        return self.get_game_weather(f"hist_{historical_date}", stadium, game_time)
    
    def _generate_weather(self, game_id: str, stadium: StadiumInfo,
                          game_time: datetime) -> GameWeather:
        """Generate realistic weather based on location and season"""
        
        month = game_time.month
        lat = stadium.latitude
        
        # Base temperature by latitude and month
        # Higher latitude = colder, especially in winter
        base_temp = self._get_base_temp(lat, month)
        temp_variation = self.rng.gauss(0, 8)  # Normal variation
        temperature = base_temp + temp_variation
        
        # Wind - more common in certain regions
        base_wind = self._get_base_wind(stadium, month)
        wind_variation = abs(self.rng.gauss(0, 5))
        wind_speed = max(0, base_wind + wind_variation)
        
        # Wind gusts (occasional)
        wind_gust = None
        if wind_speed > 10 and self.rng.random() < 0.4:
            wind_gust = wind_speed + self.rng.uniform(5, 15)
        
        wind_dir = self.rng.choice(list(WindDirection))
        if wind_speed < 3:
            wind_dir = WindDirection.CALM
        
        # Wind chill / heat index
        feels_like = self._calculate_feels_like(temperature, wind_speed, 50)
        
        # Humidity
        humidity = self._get_humidity(lat, month) + self.rng.gauss(0, 10)
        humidity = max(20, min(100, humidity))
        
        # Precipitation
        precip_chance = self._get_precip_chance(lat, month)
        has_precip = self.rng.random() < (precip_chance / 100)
        
        condition = WeatherCondition.CLEAR
        precip_type = PrecipitationType.NONE
        precip_intensity = PrecipitationIntensity.NONE
        precip_amount = 0.0
        
        if has_precip:
            # Determine type based on temperature
            if temperature < 32:
                precip_type = PrecipitationType.SNOW
                if temperature < 28 and self.rng.random() < 0.3:
                    precip_type = self.rng.choice([PrecipitationType.SLEET, PrecipitationType.FREEZING_RAIN])
            else:
                precip_type = PrecipitationType.RAIN
            
            # Intensity
            intensity_roll = self.rng.random()
            if intensity_roll < 0.5:
                precip_intensity = PrecipitationIntensity.LIGHT
                precip_amount = self.rng.uniform(0.01, 0.1)
            elif intensity_roll < 0.85:
                precip_intensity = PrecipitationIntensity.MODERATE
                precip_amount = self.rng.uniform(0.1, 0.3)
            else:
                precip_intensity = PrecipitationIntensity.HEAVY
                precip_amount = self.rng.uniform(0.3, 1.0)
            
            # Set condition
            if precip_type == PrecipitationType.RAIN:
                if precip_intensity == PrecipitationIntensity.HEAVY:
                    condition = WeatherCondition.HEAVY_RAIN
                elif precip_intensity == PrecipitationIntensity.MODERATE:
                    condition = WeatherCondition.RAIN
                else:
                    condition = WeatherCondition.LIGHT_RAIN
            elif precip_type == PrecipitationType.SNOW:
                if precip_intensity == PrecipitationIntensity.HEAVY:
                    condition = WeatherCondition.HEAVY_SNOW
                elif precip_intensity == PrecipitationIntensity.MODERATE:
                    condition = WeatherCondition.SNOW
                else:
                    condition = WeatherCondition.LIGHT_SNOW
            else:
                condition = WeatherCondition.SLEET
        else:
            # Non-precipitating conditions
            cloud_roll = self.rng.random()
            if cloud_roll < 0.4:
                condition = WeatherCondition.CLEAR
            elif cloud_roll < 0.6:
                condition = WeatherCondition.PARTLY_CLOUDY
            elif cloud_roll < 0.8:
                condition = WeatherCondition.CLOUDY
            else:
                condition = WeatherCondition.OVERCAST
        
        # Visibility
        visibility = 10.0
        if has_precip:
            if precip_intensity == PrecipitationIntensity.HEAVY:
                visibility = self.rng.uniform(0.5, 3)
            elif precip_intensity == PrecipitationIntensity.MODERATE:
                visibility = self.rng.uniform(3, 7)
            else:
                visibility = self.rng.uniform(7, 10)
        
        # Roof status for retractable roofs
        roof_status = RoofStatus.OUTDOOR
        if stadium.has_retractable_roof:
            # Close roof in bad weather
            if has_precip or temperature < 45 or temperature > 90 or wind_speed > 20:
                roof_status = RoofStatus.CLOSED
            else:
                roof_status = RoofStatus.OPEN
        
        return GameWeather(
            game_id=game_id,
            game_datetime=game_time,
            stadium_name=stadium.stadium_name,
            temperature_f=round(temperature, 1),
            feels_like_f=round(feels_like, 1),
            humidity_percent=round(humidity, 1),
            wind_speed_mph=round(wind_speed, 1),
            wind_gust_mph=round(wind_gust, 1) if wind_gust else None,
            wind_direction=wind_dir,
            condition=condition,
            precip_type=precip_type,
            precip_intensity=precip_intensity,
            precip_probability=precip_chance if has_precip else max(0, precip_chance - 20),
            precip_amount_inches=round(precip_amount, 2),
            visibility_miles=round(visibility, 1),
            is_dome=False,
            roof_status=roof_status,
            forecast_updated=datetime.now(),
            data_source=self.source_name
        )
    
    def _get_base_temp(self, lat: float, month: int) -> float:
        """Get base temperature by latitude and month"""
        # NFL season months: Sep(9) through Feb(2)
        # Approximate seasonal pattern
        
        # Southern baseline (Miami ~26°N)
        # Northern baseline (Green Bay ~45°N)
        lat_factor = (lat - 26) / 19  # 0 = Miami, 1 = Green Bay
        lat_factor = max(0, min(1, lat_factor))
        
        # Monthly base temps (Southern / Northern)
        monthly_temps = {
            9: (85, 60),   # September
            10: (78, 50),  # October
            11: (70, 38),  # November
            12: (65, 28),  # December
            1: (62, 22),   # January
            2: (65, 25),   # February
        }
        
        south_temp, north_temp = monthly_temps.get(month, (70, 45))
        return south_temp - (lat_factor * (south_temp - north_temp))
    
    def _get_base_wind(self, stadium: StadiumInfo, month: int) -> float:
        """Get base wind speed by location and month"""
        # Windy cities/stadiums
        windy_teams = {"BUF", "CHI", "CLE", "GB", "KC", "NE", "NYG", "NYJ", "SF"}
        
        base = 8.0
        if stadium.team in windy_teams:
            base = 12.0
        
        # Winter months are windier
        if month in [11, 12, 1, 2]:
            base += 3.0
        
        return base
    
    def _get_humidity(self, lat: float, month: int) -> float:
        """Get base humidity by location"""
        # Southern/coastal areas more humid
        if lat < 35:  # Southern
            return 70
        elif lat < 40:  # Mid
            return 55
        else:  # Northern
            return 50
    
    def _get_precip_chance(self, lat: float, month: int) -> float:
        """Get precipitation probability"""
        # Base chance by month
        monthly_precip = {
            9: 25,   # September
            10: 30,  # October  
            11: 35,  # November
            12: 40,  # December
            1: 35,   # January
            2: 35,   # February
        }
        return monthly_precip.get(month, 30)
    
    def _calculate_feels_like(self, temp: float, wind: float, humidity: float) -> float:
        """Calculate feels-like temperature"""
        if temp <= 50 and wind > 3:
            # Wind chill formula (NWS)
            wind_chill = 35.74 + (0.6215 * temp) - (35.75 * (wind ** 0.16)) + (0.4275 * temp * (wind ** 0.16))
            return wind_chill
        elif temp >= 80:
            # Heat index (simplified)
            heat_index = temp + (0.5 * (humidity - 50) * 0.1)
            return heat_index
        return temp


class WeatherCache:
    """
    Cache for weather data with TTL management.
    
    Story 7.2: Caching to reduce API calls and handle rate limits.
    """
    
    def __init__(self, ttl_minutes: int = 60):
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: Dict[str, Tuple[GameWeather, datetime]] = {}
    
    def get(self, key: str) -> Optional[GameWeather]:
        """Get cached weather if not expired"""
        if key in self._cache:
            weather, cached_at = self._cache[key]
            if datetime.now() - cached_at < self.ttl:
                return weather
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, weather: GameWeather) -> None:
        """Cache weather data"""
        self._cache[key] = (weather, datetime.now())
    
    def clear(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
    
    def clear_expired(self) -> int:
        """Remove expired entries, return count removed"""
        now = datetime.now()
        expired = [k for k, (_, t) in self._cache.items() if now - t >= self.ttl]
        for key in expired:
            del self._cache[key]
        return len(expired)


# =============================================================================
# STORY 7.3: WEATHER IMPACT MODEL
# =============================================================================

@dataclass
class WeatherImpactConfig:
    """Configuration for weather impact calculations"""
    
    # Wind thresholds and impacts (mph)
    wind_light_threshold: float = 10.0  # Below this = no impact
    wind_moderate_threshold: float = 15.0
    wind_heavy_threshold: float = 20.0
    wind_severe_threshold: float = 25.0
    
    # Wind impacts on spread (points, favors run-heavy team)
    wind_moderate_spread_impact: float = 0.3
    wind_heavy_spread_impact: float = 0.7
    wind_severe_spread_impact: float = 1.2
    
    # Wind impacts on total (points reduction)
    wind_moderate_total_impact: float = -1.5
    wind_heavy_total_impact: float = -3.0
    wind_severe_total_impact: float = -5.0
    
    # Precipitation impacts
    light_rain_spread_impact: float = 0.2
    light_rain_total_impact: float = -1.5
    moderate_rain_spread_impact: float = 0.4
    moderate_rain_total_impact: float = -3.0
    heavy_rain_spread_impact: float = 0.6
    heavy_rain_total_impact: float = -5.0
    
    # Snow impacts (more severe than rain)
    light_snow_spread_impact: float = 0.3
    light_snow_total_impact: float = -2.0
    moderate_snow_spread_impact: float = 0.6
    moderate_snow_total_impact: float = -4.0
    heavy_snow_spread_impact: float = 1.0
    heavy_snow_total_impact: float = -6.0
    
    # Temperature impacts
    cold_threshold: float = 40.0  # Below this = cold
    very_cold_threshold: float = 20.0
    extreme_cold_threshold: float = 0.0
    
    cold_spread_impact: float = 0.2
    cold_total_impact: float = -1.0
    very_cold_spread_impact: float = 0.4
    very_cold_total_impact: float = -2.0
    extreme_cold_spread_impact: float = 0.7
    extreme_cold_total_impact: float = -3.5
    
    # Heat impacts
    hot_threshold: float = 85.0
    very_hot_threshold: float = 95.0
    
    hot_total_impact: float = -0.5
    very_hot_total_impact: float = -1.5
    
    # Combination multipliers
    cold_and_wind_multiplier: float = 1.3
    cold_and_precip_multiplier: float = 1.2
    wind_and_precip_multiplier: float = 1.25


@dataclass
class WeatherImpact:
    """
    Calculated weather impact on a game.
    
    Story 7.3: Quantified impact of weather on spread and total.
    """
    # Source weather
    weather: GameWeather
    
    # Spread adjustments (positive = favors home team's style)
    wind_spread_impact: float = 0.0
    precip_spread_impact: float = 0.0
    temp_spread_impact: float = 0.0
    
    # Total adjustments (usually negative in bad weather)
    wind_total_impact: float = 0.0
    precip_total_impact: float = 0.0
    temp_total_impact: float = 0.0
    
    # Combined
    combination_multiplier: float = 1.0
    
    # Uncertainty
    impact_confidence: float = 1.0  # 0-1, lower for extreme conditions
    
    @property
    def total_spread_impact(self) -> float:
        """Net spread impact from weather"""
        base = self.wind_spread_impact + self.precip_spread_impact + self.temp_spread_impact
        return base * self.combination_multiplier
    
    @property
    def total_total_impact(self) -> float:
        """Net total impact from weather"""
        base = self.wind_total_impact + self.precip_total_impact + self.temp_total_impact
        return base * self.combination_multiplier
    
    @property
    def is_significant(self) -> bool:
        """Weather has significant game impact"""
        return abs(self.total_spread_impact) > 0.5 or abs(self.total_total_impact) > 2.0
    
    def get_summary(self) -> str:
        """Human-readable impact summary"""
        parts = []
        
        if abs(self.wind_spread_impact) > 0.1 or abs(self.wind_total_impact) > 0.5:
            parts.append(f"Wind: spread {self.wind_spread_impact:+.1f}, total {self.wind_total_impact:+.1f}")
        
        if abs(self.precip_spread_impact) > 0.1 or abs(self.precip_total_impact) > 0.5:
            parts.append(f"Precip: spread {self.precip_spread_impact:+.1f}, total {self.precip_total_impact:+.1f}")
        
        if abs(self.temp_spread_impact) > 0.1 or abs(self.temp_total_impact) > 0.5:
            parts.append(f"Temp: spread {self.temp_spread_impact:+.1f}, total {self.temp_total_impact:+.1f}")
        
        if not parts:
            return "No significant weather impact"
        
        return " | ".join(parts) + f" | Combined: spread {self.total_spread_impact:+.1f}, total {self.total_total_impact:+.1f}"


class WeatherImpactCalculator:
    """
    Calculates weather impact on game predictions.
    
    Story 7.3: Core weather impact calculation engine.
    """
    
    def __init__(self, config: Optional[WeatherImpactConfig] = None):
        self.config = config or WeatherImpactConfig()
    
    def calculate_impact(self, weather: GameWeather) -> WeatherImpact:
        """Calculate total weather impact on a game"""
        
        # Indoor games have no weather impact
        if weather.is_indoor:
            return WeatherImpact(weather=weather)
        
        impact = WeatherImpact(weather=weather)
        
        # Wind impact
        wind_spread, wind_total = self._calculate_wind_impact(weather.effective_wind)
        impact.wind_spread_impact = wind_spread
        impact.wind_total_impact = wind_total
        
        # Precipitation impact
        precip_spread, precip_total = self._calculate_precip_impact(
            weather.precip_type, 
            weather.precip_intensity
        )
        impact.precip_spread_impact = precip_spread
        impact.precip_total_impact = precip_total
        
        # Temperature impact
        temp_spread, temp_total = self._calculate_temp_impact(weather.feels_like_f)
        impact.temp_spread_impact = temp_spread
        impact.temp_total_impact = temp_total
        
        # Combination effects
        impact.combination_multiplier = self._calculate_combination_multiplier(weather)
        
        # Confidence based on severity
        impact.impact_confidence = self._calculate_confidence(weather)
        
        return impact
    
    def _calculate_wind_impact(self, wind_mph: float) -> Tuple[float, float]:
        """Calculate wind impact on spread and total"""
        cfg = self.config
        
        if wind_mph < cfg.wind_light_threshold:
            return 0.0, 0.0
        elif wind_mph < cfg.wind_moderate_threshold:
            # Interpolate
            factor = (wind_mph - cfg.wind_light_threshold) / (cfg.wind_moderate_threshold - cfg.wind_light_threshold)
            return (cfg.wind_moderate_spread_impact * factor, 
                    cfg.wind_moderate_total_impact * factor)
        elif wind_mph < cfg.wind_heavy_threshold:
            factor = (wind_mph - cfg.wind_moderate_threshold) / (cfg.wind_heavy_threshold - cfg.wind_moderate_threshold)
            spread = cfg.wind_moderate_spread_impact + factor * (cfg.wind_heavy_spread_impact - cfg.wind_moderate_spread_impact)
            total = cfg.wind_moderate_total_impact + factor * (cfg.wind_heavy_total_impact - cfg.wind_moderate_total_impact)
            return spread, total
        elif wind_mph < cfg.wind_severe_threshold:
            factor = (wind_mph - cfg.wind_heavy_threshold) / (cfg.wind_severe_threshold - cfg.wind_heavy_threshold)
            spread = cfg.wind_heavy_spread_impact + factor * (cfg.wind_severe_spread_impact - cfg.wind_heavy_spread_impact)
            total = cfg.wind_heavy_total_impact + factor * (cfg.wind_severe_total_impact - cfg.wind_heavy_total_impact)
            return spread, total
        else:
            return cfg.wind_severe_spread_impact, cfg.wind_severe_total_impact
    
    def _calculate_precip_impact(self, precip_type: PrecipitationType, 
                                  intensity: PrecipitationIntensity) -> Tuple[float, float]:
        """Calculate precipitation impact"""
        cfg = self.config
        
        if intensity == PrecipitationIntensity.NONE:
            return 0.0, 0.0
        
        # Snow impacts
        if precip_type in [PrecipitationType.SNOW, PrecipitationType.SLEET, 
                           PrecipitationType.FREEZING_RAIN]:
            if intensity == PrecipitationIntensity.LIGHT:
                return cfg.light_snow_spread_impact, cfg.light_snow_total_impact
            elif intensity == PrecipitationIntensity.MODERATE:
                return cfg.moderate_snow_spread_impact, cfg.moderate_snow_total_impact
            else:
                return cfg.heavy_snow_spread_impact, cfg.heavy_snow_total_impact
        
        # Rain impacts
        if intensity == PrecipitationIntensity.LIGHT:
            return cfg.light_rain_spread_impact, cfg.light_rain_total_impact
        elif intensity == PrecipitationIntensity.MODERATE:
            return cfg.moderate_rain_spread_impact, cfg.moderate_rain_total_impact
        else:
            return cfg.heavy_rain_spread_impact, cfg.heavy_rain_total_impact
    
    def _calculate_temp_impact(self, feels_like: float) -> Tuple[float, float]:
        """Calculate temperature impact"""
        cfg = self.config
        
        # Cold impacts
        if feels_like < cfg.extreme_cold_threshold:
            return cfg.extreme_cold_spread_impact, cfg.extreme_cold_total_impact
        elif feels_like < cfg.very_cold_threshold:
            return cfg.very_cold_spread_impact, cfg.very_cold_total_impact
        elif feels_like < cfg.cold_threshold:
            return cfg.cold_spread_impact, cfg.cold_total_impact
        
        # Heat impacts (no spread impact, just total)
        if feels_like > cfg.very_hot_threshold:
            return 0.0, cfg.very_hot_total_impact
        elif feels_like > cfg.hot_threshold:
            return 0.0, cfg.hot_total_impact
        
        return 0.0, 0.0
    
    def _calculate_combination_multiplier(self, weather: GameWeather) -> float:
        """Calculate multiplier for combined weather effects"""
        cfg = self.config
        multiplier = 1.0
        
        is_cold = weather.feels_like_f < cfg.cold_threshold
        is_windy = weather.effective_wind > cfg.wind_moderate_threshold
        is_precip = weather.has_precipitation
        
        # Cold + wind is worse
        if is_cold and is_windy:
            multiplier *= cfg.cold_and_wind_multiplier
        
        # Cold + precip is worse
        if is_cold and is_precip:
            multiplier *= cfg.cold_and_precip_multiplier
        
        # Wind + precip is worse
        if is_windy and is_precip:
            multiplier *= cfg.wind_and_precip_multiplier
        
        return multiplier
    
    def _calculate_confidence(self, weather: GameWeather) -> float:
        """Calculate confidence in impact estimate"""
        # More extreme = less confident in exact impact
        confidence = 1.0
        
        if weather.severity == WeatherSeverity.SEVERE:
            confidence = 0.7
        elif weather.severity == WeatherSeverity.ADVERSE:
            confidence = 0.85
        elif weather.severity == WeatherSeverity.MODERATE:
            confidence = 0.95
        
        return confidence


# =============================================================================
# STORY 7.4: TEAM WEATHER PROFILE
# =============================================================================

class WeatherCategory(Enum):
    """Team's home weather category"""
    COLD_WEATHER = "cold_weather"  # Green Bay, Buffalo, Chicago, etc.
    DOME = "dome"  # Detroit, New Orleans, Minnesota, etc.
    WARM_WEATHER = "warm_weather"  # Miami, Tampa, Jacksonville, etc.
    MILD = "mild"  # Most teams
    HIGH_ALTITUDE = "high_altitude"  # Denver


@dataclass
class TeamWeatherProfile:
    """
    Team-specific weather adaptability profile.
    
    Story 7.4: Models how teams perform in different weather conditions
    based on their home environment and roster construction.
    """
    team: str
    category: WeatherCategory
    home_stadium: StadiumInfo
    
    # Adjustments when this team plays in conditions
    cold_weather_bonus: float = 0.0  # Bonus when playing in cold (negative = penalty)
    dome_team_outdoor_penalty: float = 0.0  # Penalty when dome team plays outdoors in cold
    wind_resistance: float = 1.0  # 1.0 = average, <1 = less affected by wind
    hot_weather_penalty: float = 0.0  # Penalty in hot weather
    
    # Team style factors
    pass_heavy_tendency: float = 0.5  # 0 = run-heavy, 1 = pass-heavy
    kicker_reliability: float = 0.5  # 0-1, affects wind impact on scoring
    
    def get_weather_adjustment(self, weather: GameWeather, is_home: bool) -> float:
        """
        Get spread adjustment for this team in given weather.
        
        Returns positive value if weather favors this team, negative if it hurts.
        """
        if weather.is_indoor:
            return 0.0
        
        adjustment = 0.0
        
        # Cold weather adjustments
        if weather.is_cold:
            if self.category == WeatherCategory.COLD_WEATHER:
                adjustment += self.cold_weather_bonus
            elif self.category == WeatherCategory.DOME:
                adjustment -= self.dome_team_outdoor_penalty
            elif self.category == WeatherCategory.WARM_WEATHER:
                adjustment -= 0.3  # Warm weather teams struggle in cold
        
        if weather.is_very_cold:
            if self.category == WeatherCategory.DOME:
                adjustment -= self.dome_team_outdoor_penalty * 0.5  # Additional penalty
            elif self.category == WeatherCategory.WARM_WEATHER:
                adjustment -= 0.5  # Additional penalty
        
        # Hot weather adjustments
        if weather.is_hot:
            if self.category == WeatherCategory.COLD_WEATHER:
                adjustment -= 0.2
            elif self.category == WeatherCategory.WARM_WEATHER:
                adjustment += 0.2  # Acclimated to heat
        
        # Home field weather familiarity
        if is_home and not weather.is_indoor:
            # Home team is familiar with their weather conditions
            adjustment += 0.1
        
        return adjustment
    
    def get_style_wind_factor(self) -> float:
        """
        Get factor for how much wind affects this team's offense.
        
        Pass-heavy teams are more affected by wind.
        """
        # Pass-heavy teams (0.7+) are more hurt by wind
        # Run-heavy teams (0.3-) are less affected
        return 0.8 + (self.pass_heavy_tendency * 0.4)  # Range: 0.8 to 1.2


# Team weather profiles for all 32 teams
TEAM_WEATHER_PROFILES: Dict[str, TeamWeatherProfile] = {
    # Cold weather teams
    "BUF": TeamWeatherProfile("BUF", WeatherCategory.COLD_WEATHER, STADIUM_DATA["BUF"],
                              cold_weather_bonus=0.5, wind_resistance=0.85, pass_heavy_tendency=0.55),
    "GB": TeamWeatherProfile("GB", WeatherCategory.COLD_WEATHER, STADIUM_DATA["GB"],
                             cold_weather_bonus=0.6, wind_resistance=0.85, pass_heavy_tendency=0.50),
    "CHI": TeamWeatherProfile("CHI", WeatherCategory.COLD_WEATHER, STADIUM_DATA["CHI"],
                              cold_weather_bonus=0.4, wind_resistance=0.80, pass_heavy_tendency=0.45),
    "NE": TeamWeatherProfile("NE", WeatherCategory.COLD_WEATHER, STADIUM_DATA["NE"],
                             cold_weather_bonus=0.4, wind_resistance=0.85, pass_heavy_tendency=0.50),
    "CLE": TeamWeatherProfile("CLE", WeatherCategory.COLD_WEATHER, STADIUM_DATA["CLE"],
                              cold_weather_bonus=0.3, wind_resistance=0.80, pass_heavy_tendency=0.45),
    "PIT": TeamWeatherProfile("PIT", WeatherCategory.COLD_WEATHER, STADIUM_DATA["PIT"],
                              cold_weather_bonus=0.3, wind_resistance=0.85, pass_heavy_tendency=0.50),
    
    # Dome teams
    "ATL": TeamWeatherProfile("ATL", WeatherCategory.DOME, STADIUM_DATA["ATL"],
                              dome_team_outdoor_penalty=1.0, pass_heavy_tendency=0.65),
    "DET": TeamWeatherProfile("DET", WeatherCategory.DOME, STADIUM_DATA["DET"],
                              dome_team_outdoor_penalty=1.0, pass_heavy_tendency=0.55),
    "IND": TeamWeatherProfile("IND", WeatherCategory.DOME, STADIUM_DATA["IND"],
                              dome_team_outdoor_penalty=0.8, pass_heavy_tendency=0.55),
    "LV": TeamWeatherProfile("LV", WeatherCategory.DOME, STADIUM_DATA["LV"],
                             dome_team_outdoor_penalty=0.8, pass_heavy_tendency=0.60),
    "MIN": TeamWeatherProfile("MIN", WeatherCategory.DOME, STADIUM_DATA["MIN"],
                              dome_team_outdoor_penalty=0.9, pass_heavy_tendency=0.60),
    "NO": TeamWeatherProfile("NO", WeatherCategory.DOME, STADIUM_DATA["NO"],
                             dome_team_outdoor_penalty=1.0, pass_heavy_tendency=0.65),
    
    # Warm weather teams
    "MIA": TeamWeatherProfile("MIA", WeatherCategory.WARM_WEATHER, STADIUM_DATA["MIA"],
                              hot_weather_penalty=-0.3, cold_weather_bonus=-0.5, pass_heavy_tendency=0.70),
    "TB": TeamWeatherProfile("TB", WeatherCategory.WARM_WEATHER, STADIUM_DATA["TB"],
                             hot_weather_penalty=-0.2, cold_weather_bonus=-0.3, pass_heavy_tendency=0.55),
    "JAX": TeamWeatherProfile("JAX", WeatherCategory.WARM_WEATHER, STADIUM_DATA["JAX"],
                              hot_weather_penalty=-0.2, cold_weather_bonus=-0.3, pass_heavy_tendency=0.50),
    "LAC": TeamWeatherProfile("LAC", WeatherCategory.MILD, STADIUM_DATA["LAC"],
                              cold_weather_bonus=-0.2, pass_heavy_tendency=0.60),
    "LAR": TeamWeatherProfile("LAR", WeatherCategory.MILD, STADIUM_DATA["LAR"],
                              cold_weather_bonus=-0.2, pass_heavy_tendency=0.55),
    
    # High altitude
    "DEN": TeamWeatherProfile("DEN", WeatherCategory.HIGH_ALTITUDE, STADIUM_DATA["DEN"],
                              cold_weather_bonus=0.3, pass_heavy_tendency=0.50),
    
    # Mild/neutral teams
    "ARI": TeamWeatherProfile("ARI", WeatherCategory.MILD, STADIUM_DATA["ARI"],
                              pass_heavy_tendency=0.55),
    "BAL": TeamWeatherProfile("BAL", WeatherCategory.MILD, STADIUM_DATA["BAL"],
                              cold_weather_bonus=0.1, pass_heavy_tendency=0.45),
    "CAR": TeamWeatherProfile("CAR", WeatherCategory.MILD, STADIUM_DATA["CAR"],
                              pass_heavy_tendency=0.50),
    "CIN": TeamWeatherProfile("CIN", WeatherCategory.MILD, STADIUM_DATA["CIN"],
                              cold_weather_bonus=0.2, pass_heavy_tendency=0.60),
    "DAL": TeamWeatherProfile("DAL", WeatherCategory.MILD, STADIUM_DATA["DAL"],
                              pass_heavy_tendency=0.60),
    "HOU": TeamWeatherProfile("HOU", WeatherCategory.MILD, STADIUM_DATA["HOU"],
                              hot_weather_penalty=-0.1, pass_heavy_tendency=0.55),
    "KC": TeamWeatherProfile("KC", WeatherCategory.MILD, STADIUM_DATA["KC"],
                             cold_weather_bonus=0.2, pass_heavy_tendency=0.60),
    "NYG": TeamWeatherProfile("NYG", WeatherCategory.MILD, STADIUM_DATA["NYG"],
                              cold_weather_bonus=0.2, wind_resistance=0.85, pass_heavy_tendency=0.50),
    "NYJ": TeamWeatherProfile("NYJ", WeatherCategory.MILD, STADIUM_DATA["NYJ"],
                              cold_weather_bonus=0.2, wind_resistance=0.85, pass_heavy_tendency=0.55),
    "PHI": TeamWeatherProfile("PHI", WeatherCategory.MILD, STADIUM_DATA["PHI"],
                              cold_weather_bonus=0.2, pass_heavy_tendency=0.55),
    "SEA": TeamWeatherProfile("SEA", WeatherCategory.MILD, STADIUM_DATA["SEA"],
                              pass_heavy_tendency=0.55),
    "SF": TeamWeatherProfile("SF", WeatherCategory.MILD, STADIUM_DATA["SF"],
                             wind_resistance=0.80, pass_heavy_tendency=0.50),
    "TEN": TeamWeatherProfile("TEN", WeatherCategory.MILD, STADIUM_DATA["TEN"],
                              pass_heavy_tendency=0.45),
    "WAS": TeamWeatherProfile("WAS", WeatherCategory.MILD, STADIUM_DATA["WAS"],
                              cold_weather_bonus=0.1, pass_heavy_tendency=0.55),
}


def get_team_profile(team: str) -> TeamWeatherProfile:
    """Get weather profile for a team"""
    if team in TEAM_WEATHER_PROFILES:
        return TEAM_WEATHER_PROFILES[team]
    # Default profile for unknown teams
    return TeamWeatherProfile(
        team=team,
        category=WeatherCategory.MILD,
        home_stadium=StadiumInfo(team, "Unknown", "Unknown", "Unknown", 40.0, -90.0, 500, False, False, 65000, "turf")
    )


# =============================================================================
# STORY 7.5: WEATHER-ADJUSTED PREDICTIONS
# =============================================================================

@dataclass
class WeatherAdjustedPrediction:
    """
    Game prediction with weather adjustments applied.
    
    Story 7.5: Complete prediction output including weather factors.
    """
    # Game info
    game_id: str
    home_team: str
    away_team: str
    game_datetime: datetime
    
    # Weather
    weather: GameWeather
    weather_impact: WeatherImpact
    
    # Team weather adjustments
    home_weather_adjustment: float  # Team-specific weather factor
    away_weather_adjustment: float
    
    # Base prediction (before weather)
    base_spread: float  # Positive = away favored
    base_total: float
    base_home_win_prob: float
    
    # Weather-adjusted prediction
    adjusted_spread: float
    adjusted_total: float
    adjusted_home_win_prob: float
    
    # Confidence
    weather_confidence: float  # How confident in weather impact
    prediction_confidence: float  # Overall prediction confidence
    
    def get_spread_change(self) -> float:
        """Change in spread due to weather"""
        return self.adjusted_spread - self.base_spread
    
    def get_total_change(self) -> float:
        """Change in total due to weather"""
        return self.adjusted_total - self.base_total
    
    def get_summary(self) -> str:
        """Human-readable prediction summary"""
        lines = [
            f"Game: {self.away_team} @ {self.home_team}",
            f"Weather: {self.weather.get_summary()}",
            f"Severity: {self.weather.severity.value}",
            "",
            f"Base Spread: {self.base_spread:+.1f} ({self.home_team} {'favored' if self.base_spread < 0 else 'underdog'})",
            f"Weather Spread Adj: {self.get_spread_change():+.1f}",
            f"Adjusted Spread: {self.adjusted_spread:+.1f}",
            "",
            f"Base Total: {self.base_total:.1f}",
            f"Weather Total Adj: {self.get_total_change():+.1f}",
            f"Adjusted Total: {self.adjusted_total:.1f}",
            "",
            f"Home Win Prob: {self.adjusted_home_win_prob:.1%}",
            f"Weather Confidence: {self.weather_confidence:.0%}",
        ]
        
        if self.weather_impact.is_significant:
            lines.insert(3, f"⚠️ SIGNIFICANT WEATHER IMPACT")
        
        return "\n".join(lines)


class WeatherAdjustedPredictor:
    """
    Generates predictions with weather adjustments.
    
    Story 7.5: Main prediction engine integrating weather data.
    """
    
    def __init__(self, 
                 weather_source: WeatherDataSource,
                 impact_calculator: Optional[WeatherImpactCalculator] = None):
        self.weather_source = weather_source
        self.impact_calculator = impact_calculator or WeatherImpactCalculator()
    
    def predict(self, 
                game_id: str,
                home_team: str,
                away_team: str,
                game_time: datetime,
                base_spread: float,
                base_total: float,
                base_home_win_prob: float) -> WeatherAdjustedPrediction:
        """
        Generate weather-adjusted prediction.
        
        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation (e.g., "BUF")
            away_team: Away team abbreviation
            game_time: Game start datetime
            base_spread: Base spread (positive = away favored)
            base_total: Base over/under total
            base_home_win_prob: Base home win probability (0-1)
        
        Returns:
            WeatherAdjustedPrediction with all adjustments applied
        """
        # Get stadium and weather
        home_profile = get_team_profile(home_team)
        away_profile = get_team_profile(away_team)
        stadium = home_profile.home_stadium
        
        weather = self.weather_source.get_game_weather(game_id, stadium, game_time)
        if weather is None:
            # No weather data - return base prediction
            return WeatherAdjustedPrediction(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                game_datetime=game_time,
                weather=GameWeather(game_id, game_time, "Unknown", 70, 70, 50, 0),
                weather_impact=WeatherImpact(weather=GameWeather(game_id, game_time, "Unknown", 70, 70, 50, 0)),
                home_weather_adjustment=0.0,
                away_weather_adjustment=0.0,
                base_spread=base_spread,
                base_total=base_total,
                base_home_win_prob=base_home_win_prob,
                adjusted_spread=base_spread,
                adjusted_total=base_total,
                adjusted_home_win_prob=base_home_win_prob,
                weather_confidence=0.5,
                prediction_confidence=0.8
            )
        
        # Calculate weather impact
        impact = self.impact_calculator.calculate_impact(weather)
        
        # Get team-specific adjustments
        home_adj = home_profile.get_weather_adjustment(weather, is_home=True)
        away_adj = away_profile.get_weather_adjustment(weather, is_home=False)
        
        # Calculate style-based wind impact difference
        home_wind_factor = home_profile.get_style_wind_factor()
        away_wind_factor = away_profile.get_style_wind_factor()
        wind_style_diff = (away_wind_factor - home_wind_factor) * impact.wind_spread_impact
        
        # Combine all spread adjustments
        # Positive adjustment = helps home team (spread moves toward away team)
        spread_adj = (
            impact.total_spread_impact +  # Base weather impact
            (home_adj - away_adj) +  # Team profile difference
            wind_style_diff  # Style matchup in wind
        )
        
        # Combine total adjustments
        total_adj = impact.total_total_impact
        
        # Apply adjustments
        adjusted_spread = base_spread - spread_adj  # Negative spread = home favored
        adjusted_total = max(20, base_total + total_adj)  # Floor at 20 points
        
        # Adjust win probability
        # Each point of spread change ≈ 3% win probability
        prob_adj = spread_adj * 0.03
        adjusted_prob = max(0.05, min(0.95, base_home_win_prob + prob_adj))
        
        # Calculate confidence
        weather_confidence = impact.impact_confidence
        prediction_confidence = 0.9 * weather_confidence  # Reduce overall confidence
        
        if weather.severity in [WeatherSeverity.ADVERSE, WeatherSeverity.SEVERE]:
            prediction_confidence *= 0.9  # Further reduce for extreme weather
        
        return WeatherAdjustedPrediction(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_datetime=game_time,
            weather=weather,
            weather_impact=impact,
            home_weather_adjustment=home_adj,
            away_weather_adjustment=away_adj,
            base_spread=base_spread,
            base_total=base_total,
            base_home_win_prob=base_home_win_prob,
            adjusted_spread=round(adjusted_spread, 1),
            adjusted_total=round(adjusted_total, 1),
            adjusted_home_win_prob=round(adjusted_prob, 3),
            weather_confidence=round(weather_confidence, 2),
            prediction_confidence=round(prediction_confidence, 2)
        )
    
    def predict_week(self, games: List[Dict]) -> List[WeatherAdjustedPrediction]:
        """
        Generate predictions for a week of games.
        
        Args:
            games: List of game dicts with keys:
                   game_id, home_team, away_team, game_time, 
                   base_spread, base_total, base_home_win_prob
        
        Returns:
            List of WeatherAdjustedPrediction
        """
        predictions = []
        for game in games:
            pred = self.predict(
                game_id=game["game_id"],
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_time=game["game_time"],
                base_spread=game["base_spread"],
                base_total=game["base_total"],
                base_home_win_prob=game["base_home_win_prob"]
            )
            predictions.append(pred)
        return predictions


# =============================================================================
# STORY 7.6: WEATHER TRACKING & VALIDATION
# =============================================================================

@dataclass
class WeatherPredictionResult:
    """Result of a weather-adjusted prediction vs actual outcome"""
    prediction: WeatherAdjustedPrediction
    actual_home_score: int
    actual_away_score: int
    actual_weather: Optional[GameWeather] = None  # Observed weather
    
    @property
    def actual_spread(self) -> float:
        """Actual game spread (away - home)"""
        return self.actual_away_score - self.actual_home_score
    
    @property
    def actual_total(self) -> float:
        """Actual game total"""
        return self.actual_home_score + self.actual_away_score
    
    @property
    def home_won(self) -> bool:
        """Did home team win?"""
        return self.actual_home_score > self.actual_away_score
    
    @property
    def spread_error(self) -> float:
        """Error in spread prediction"""
        return self.actual_spread - self.prediction.adjusted_spread
    
    @property
    def total_error(self) -> float:
        """Error in total prediction"""
        return self.actual_total - self.prediction.adjusted_total
    
    @property
    def base_spread_error(self) -> float:
        """Error in base spread (without weather adjustment)"""
        return self.actual_spread - self.prediction.base_spread
    
    @property
    def base_total_error(self) -> float:
        """Error in base total"""
        return self.actual_total - self.prediction.base_total
    
    @property
    def weather_adjustment_helped_spread(self) -> bool:
        """Did weather adjustment improve spread prediction?"""
        return abs(self.spread_error) < abs(self.base_spread_error)
    
    @property
    def weather_adjustment_helped_total(self) -> bool:
        """Did weather adjustment improve total prediction?"""
        return abs(self.total_error) < abs(self.base_total_error)


@dataclass
class WeatherTrackingStats:
    """Aggregated weather tracking statistics"""
    total_games: int = 0
    
    # Spread accuracy
    spread_mae: float = 0.0  # Mean Absolute Error
    spread_rmse: float = 0.0  # Root Mean Square Error
    base_spread_mae: float = 0.0
    spread_improvement_rate: float = 0.0  # % of games where weather helped
    
    # Total accuracy
    total_mae: float = 0.0
    total_rmse: float = 0.0
    base_total_mae: float = 0.0
    total_improvement_rate: float = 0.0
    
    # Straight-up accuracy
    su_accuracy: float = 0.0  # Straight-up win prediction accuracy
    
    # Weather-specific breakdowns
    games_with_significant_weather: int = 0
    significant_weather_spread_improvement: float = 0.0
    significant_weather_total_improvement: float = 0.0
    
    # By severity
    accuracy_by_severity: Dict[str, float] = field(default_factory=dict)


class WeatherTracker:
    """
    Tracks weather prediction accuracy over time.
    
    Story 7.6: Validation and tracking system for weather adjustments.
    """
    
    def __init__(self):
        self.results: List[WeatherPredictionResult] = []
    
    def add_result(self, result: WeatherPredictionResult) -> None:
        """Add a game result"""
        self.results.append(result)
    
    def add_results(self, results: List[WeatherPredictionResult]) -> None:
        """Add multiple results"""
        self.results.extend(results)
    
    def get_stats(self) -> WeatherTrackingStats:
        """Calculate aggregate statistics"""
        if not self.results:
            return WeatherTrackingStats()
        
        n = len(self.results)
        
        # Spread errors
        spread_errors = [r.spread_error for r in self.results]
        base_spread_errors = [r.base_spread_error for r in self.results]
        
        spread_mae = sum(abs(e) for e in spread_errors) / n
        spread_rmse = math.sqrt(sum(e ** 2 for e in spread_errors) / n)
        base_spread_mae = sum(abs(e) for e in base_spread_errors) / n
        
        spread_helped = sum(1 for r in self.results if r.weather_adjustment_helped_spread)
        spread_improvement_rate = spread_helped / n
        
        # Total errors
        total_errors = [r.total_error for r in self.results]
        base_total_errors = [r.base_total_error for r in self.results]
        
        total_mae = sum(abs(e) for e in total_errors) / n
        total_rmse = math.sqrt(sum(e ** 2 for e in total_errors) / n)
        base_total_mae = sum(abs(e) for e in base_total_errors) / n
        
        total_helped = sum(1 for r in self.results if r.weather_adjustment_helped_total)
        total_improvement_rate = total_helped / n
        
        # Straight-up accuracy
        correct_picks = sum(
            1 for r in self.results 
            if (r.prediction.adjusted_home_win_prob > 0.5) == r.home_won
        )
        su_accuracy = correct_picks / n
        
        # Significant weather games
        significant_weather_results = [
            r for r in self.results 
            if r.prediction.weather_impact.is_significant
        ]
        
        games_with_significant = len(significant_weather_results)
        
        if significant_weather_results:
            sig_spread_helped = sum(
                1 for r in significant_weather_results 
                if r.weather_adjustment_helped_spread
            )
            sig_total_helped = sum(
                1 for r in significant_weather_results 
                if r.weather_adjustment_helped_total
            )
            sig_spread_improvement = sig_spread_helped / len(significant_weather_results)
            sig_total_improvement = sig_total_helped / len(significant_weather_results)
        else:
            sig_spread_improvement = 0.0
            sig_total_improvement = 0.0
        
        # By severity
        accuracy_by_severity = {}
        for severity in WeatherSeverity:
            severity_results = [
                r for r in self.results 
                if r.prediction.weather.severity == severity
            ]
            if severity_results:
                correct = sum(
                    1 for r in severity_results
                    if (r.prediction.adjusted_home_win_prob > 0.5) == r.home_won
                )
                accuracy_by_severity[severity.value] = correct / len(severity_results)
        
        return WeatherTrackingStats(
            total_games=n,
            spread_mae=round(spread_mae, 2),
            spread_rmse=round(spread_rmse, 2),
            base_spread_mae=round(base_spread_mae, 2),
            spread_improvement_rate=round(spread_improvement_rate, 3),
            total_mae=round(total_mae, 2),
            total_rmse=round(total_rmse, 2),
            base_total_mae=round(base_total_mae, 2),
            total_improvement_rate=round(total_improvement_rate, 3),
            su_accuracy=round(su_accuracy, 3),
            games_with_significant_weather=games_with_significant,
            significant_weather_spread_improvement=round(sig_spread_improvement, 3),
            significant_weather_total_improvement=round(sig_total_improvement, 3),
            accuracy_by_severity=accuracy_by_severity
        )
    
    def get_report(self) -> str:
        """Generate formatted tracking report"""
        stats = self.get_stats()
        
        lines = [
            "=" * 60,
            "WEATHER TRACKING REPORT",
            "=" * 60,
            "",
            f"Total Games Tracked: {stats.total_games}",
            f"Games with Significant Weather: {stats.games_with_significant_weather}",
            "",
            "SPREAD ACCURACY",
            "-" * 40,
            f"  Weather-Adjusted MAE: {stats.spread_mae:.2f} pts",
            f"  Base MAE (no weather): {stats.base_spread_mae:.2f} pts",
            f"  Improvement Rate: {stats.spread_improvement_rate:.1%}",
            f"  RMSE: {stats.spread_rmse:.2f} pts",
            "",
            "TOTAL ACCURACY",
            "-" * 40,
            f"  Weather-Adjusted MAE: {stats.total_mae:.2f} pts",
            f"  Base MAE (no weather): {stats.base_total_mae:.2f} pts",
            f"  Improvement Rate: {stats.total_improvement_rate:.1%}",
            f"  RMSE: {stats.total_rmse:.2f} pts",
            "",
            "STRAIGHT-UP ACCURACY",
            "-" * 40,
            f"  Overall: {stats.su_accuracy:.1%}",
            "",
            "SIGNIFICANT WEATHER GAMES",
            "-" * 40,
            f"  Spread Improvement Rate: {stats.significant_weather_spread_improvement:.1%}",
            f"  Total Improvement Rate: {stats.significant_weather_total_improvement:.1%}",
            "",
            "ACCURACY BY SEVERITY",
            "-" * 40,
        ]
        
        for severity, accuracy in stats.accuracy_by_severity.items():
            lines.append(f"  {severity.title():15s}: {accuracy:.1%}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all tracked results"""
        self.results.clear()


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo_epic_7():
    """Demonstrate Epic 7 features"""
    print("=" * 70)
    print("NFL PREDICTION MODEL - EPIC 7: WEATHER ADJUSTMENTS DEMO")
    print("=" * 70)
    print()
    
    # Initialize components
    weather_source = MockWeatherDataSource(seed=42)
    impact_calculator = WeatherImpactCalculator()
    predictor = WeatherAdjustedPredictor(weather_source, impact_calculator)
    tracker = WeatherTracker()
    
    # Sample games for Week 18
    games = [
        {
            "game_id": "2025_18_BUF_KC",
            "home_team": "KC",
            "away_team": "BUF",
            "game_time": datetime(2026, 1, 4, 16, 30),
            "base_spread": -1.5,  # KC favored by 1.5
            "base_total": 47.5,
            "base_home_win_prob": 0.54
        },
        {
            "game_id": "2025_18_MIA_GB",
            "home_team": "GB",
            "away_team": "MIA",
            "game_time": datetime(2026, 1, 4, 13, 0),
            "base_spread": -3.0,  # GB favored
            "base_total": 45.0,
            "base_home_win_prob": 0.58
        },
        {
            "game_id": "2025_18_DET_DAL",
            "home_team": "DAL",
            "away_team": "DET",
            "game_time": datetime(2026, 1, 4, 16, 25),
            "base_spread": 2.0,  # DET favored
            "base_total": 51.5,
            "base_home_win_prob": 0.44
        },
        {
            "game_id": "2025_18_SEA_SF",
            "home_team": "SF",
            "away_team": "SEA",
            "game_time": datetime(2026, 1, 4, 16, 25),
            "base_spread": -6.5,  # SF favored
            "base_total": 44.0,
            "base_home_win_prob": 0.68
        },
        {
            "game_id": "2025_18_NYJ_NE",
            "home_team": "NE",
            "away_team": "NYJ",
            "game_time": datetime(2026, 1, 4, 13, 0),
            "base_spread": 1.0,  # NYJ slight favorite
            "base_total": 38.5,
            "base_home_win_prob": 0.47
        },
    ]
    
    print("GENERATING WEATHER-ADJUSTED PREDICTIONS")
    print("-" * 70)
    print()
    
    predictions = predictor.predict_week(games)
    
    for pred in predictions:
        print(pred.get_summary())
        print()
        print("-" * 40)
        print()
    
    # Simulate some results for tracking demo
    print("\nWEATHER TRACKING DEMONSTRATION")
    print("-" * 70)
    
    # Mock actual results
    actual_results = [
        (24, 21),  # BUF @ KC
        (17, 31),  # MIA @ GB
        (28, 24),  # DET @ DAL
        (21, 27),  # SEA @ SF
        (13, 17),  # NYJ @ NE
    ]
    
    for pred, (away_score, home_score) in zip(predictions, actual_results):
        result = WeatherPredictionResult(
            prediction=pred,
            actual_home_score=home_score,
            actual_away_score=away_score
        )
        tracker.add_result(result)
    
    print(tracker.get_report())
    
    # Show feature summary
    print()
    print("=" * 70)
    print("""
Epic 7 Features Implemented:
  ✓ Story 7.1: Weather Data Model
    - GameWeather with temp, wind, precipitation
    - Weather condition enums
    - Feels-like temperature calculation
    - Weather category classification
    
  ✓ Story 7.2: Weather Data Fetcher
    - MockWeatherDataSource for testing
    - Realistic weather generation by location/date
    - Weather caching
    
  ✓ Story 7.3: Weather Impact Model
    - WeatherImpactCalculator
    - Wind, precipitation, temperature impacts
    - Spread and total adjustments
    - Combined effects modeling
    
  ✓ Story 7.4: Team Weather Profile
    - TeamWeatherProfile for all 32 teams
    - Cold weather, dome, warm weather categories
    - Team-specific weather adjustments
    - Home field weather familiarity
    
  ✓ Story 7.5: Weather-Adjusted Predictions
    - WeatherAdjustedPredictor
    - Integration of weather and team factors
    - Adjusted spread, total, win probability
    
  ✓ Story 7.6: Weather Tracking & Validation
    - WeatherTracker for accuracy monitoring
    - Forecast vs actual comparison
    - Impact accuracy measurement
    - Comprehensive reporting
    
Integration Notes:
  - Extends base NFL prediction model
  - Ready to integrate with existing GamePrediction system
  - Mock data source can be replaced with real API
  - All weather factors tunable via config
""")


if __name__ == "__main__":
    demo_epic_7()
