"""
Stadium Data - NFL stadium information including coordinates, roof types, and surfaces.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum


class RoofType(Enum):
    """Stadium roof types."""
    OPEN = "open"
    DOME = "dome"
    RETRACTABLE = "retractable"


class SurfaceType(Enum):
    """Playing surface types."""
    GRASS = "grass"
    TURF = "turf"


@dataclass
class StadiumInfo:
    """
    Complete information about an NFL stadium.

    Attributes:
        name: Stadium name
        team: Home team name
        city: City location
        state: State/region
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        altitude: Altitude in feet above sea level
        roof_type: Type of roof (open, dome, retractable)
        surface: Playing surface type
        capacity: Seating capacity
        opened: Year stadium opened
    """
    name: str
    team: str
    city: str
    state: str
    latitude: float
    longitude: float
    altitude: int = 0
    roof_type: RoofType = RoofType.OPEN
    surface: SurfaceType = SurfaceType.GRASS
    capacity: int = 65000
    opened: int = 2000

    @property
    def coordinates(self) -> Tuple[float, float]:
        """Get latitude/longitude tuple."""
        return (self.latitude, self.longitude)

    @property
    def is_dome(self) -> bool:
        """Check if stadium is a dome."""
        return self.roof_type in [RoofType.DOME, RoofType.RETRACTABLE]

    @property
    def is_high_altitude(self) -> bool:
        """Check if stadium is at high altitude (>4000 ft)."""
        return self.altitude > 4000


# Complete stadium data for all 32 NFL teams
STADIUM_DATA: Dict[str, StadiumInfo] = {
    "Arizona Cardinals": StadiumInfo(
        name="State Farm Stadium",
        team="Arizona Cardinals",
        city="Glendale",
        state="AZ",
        latitude=33.5276,
        longitude=-112.2626,
        altitude=1086,
        roof_type=RoofType.RETRACTABLE,
        surface=SurfaceType.GRASS,
        capacity=63400,
        opened=2006
    ),
    "Atlanta Falcons": StadiumInfo(
        name="Mercedes-Benz Stadium",
        team="Atlanta Falcons",
        city="Atlanta",
        state="GA",
        latitude=33.7553,
        longitude=-84.4006,
        altitude=1050,
        roof_type=RoofType.RETRACTABLE,
        surface=SurfaceType.TURF,
        capacity=71000,
        opened=2017
    ),
    "Baltimore Ravens": StadiumInfo(
        name="M&T Bank Stadium",
        team="Baltimore Ravens",
        city="Baltimore",
        state="MD",
        latitude=39.2780,
        longitude=-76.6227,
        altitude=30,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=71008,
        opened=1998
    ),
    "Buffalo Bills": StadiumInfo(
        name="Highmark Stadium",
        team="Buffalo Bills",
        city="Orchard Park",
        state="NY",
        latitude=42.7738,
        longitude=-78.7870,
        altitude=600,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.TURF,
        capacity=71608,
        opened=1973
    ),
    "Carolina Panthers": StadiumInfo(
        name="Bank of America Stadium",
        team="Carolina Panthers",
        city="Charlotte",
        state="NC",
        latitude=35.2258,
        longitude=-80.8528,
        altitude=751,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=74867,
        opened=1996
    ),
    "Chicago Bears": StadiumInfo(
        name="Soldier Field",
        team="Chicago Bears",
        city="Chicago",
        state="IL",
        latitude=41.8623,
        longitude=-87.6167,
        altitude=595,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=61500,
        opened=1924
    ),
    "Cincinnati Bengals": StadiumInfo(
        name="Paycor Stadium",
        team="Cincinnati Bengals",
        city="Cincinnati",
        state="OH",
        latitude=39.0955,
        longitude=-84.5160,
        altitude=490,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.TURF,
        capacity=65515,
        opened=2000
    ),
    "Cleveland Browns": StadiumInfo(
        name="Cleveland Browns Stadium",
        team="Cleveland Browns",
        city="Cleveland",
        state="OH",
        latitude=41.5061,
        longitude=-81.6995,
        altitude=653,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=67431,
        opened=1999
    ),
    "Dallas Cowboys": StadiumInfo(
        name="AT&T Stadium",
        team="Dallas Cowboys",
        city="Arlington",
        state="TX",
        latitude=32.7473,
        longitude=-97.0945,
        altitude=616,
        roof_type=RoofType.RETRACTABLE,
        surface=SurfaceType.TURF,
        capacity=80000,
        opened=2009
    ),
    "Denver Broncos": StadiumInfo(
        name="Empower Field at Mile High",
        team="Denver Broncos",
        city="Denver",
        state="CO",
        latitude=39.7439,
        longitude=-105.0201,
        altitude=5280,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=76125,
        opened=2001
    ),
    "Detroit Lions": StadiumInfo(
        name="Ford Field",
        team="Detroit Lions",
        city="Detroit",
        state="MI",
        latitude=42.3400,
        longitude=-83.0456,
        altitude=600,
        roof_type=RoofType.DOME,
        surface=SurfaceType.TURF,
        capacity=65000,
        opened=2002
    ),
    "Green Bay Packers": StadiumInfo(
        name="Lambeau Field",
        team="Green Bay Packers",
        city="Green Bay",
        state="WI",
        latitude=44.5013,
        longitude=-88.0622,
        altitude=640,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=81441,
        opened=1957
    ),
    "Houston Texans": StadiumInfo(
        name="NRG Stadium",
        team="Houston Texans",
        city="Houston",
        state="TX",
        latitude=29.6847,
        longitude=-95.4107,
        altitude=80,
        roof_type=RoofType.RETRACTABLE,
        surface=SurfaceType.TURF,
        capacity=72220,
        opened=2002
    ),
    "Indianapolis Colts": StadiumInfo(
        name="Lucas Oil Stadium",
        team="Indianapolis Colts",
        city="Indianapolis",
        state="IN",
        latitude=39.7601,
        longitude=-86.1639,
        altitude=715,
        roof_type=RoofType.RETRACTABLE,
        surface=SurfaceType.TURF,
        capacity=67000,
        opened=2008
    ),
    "Jacksonville Jaguars": StadiumInfo(
        name="TIAA Bank Field",
        team="Jacksonville Jaguars",
        city="Jacksonville",
        state="FL",
        latitude=30.3239,
        longitude=-81.6373,
        altitude=16,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=67814,
        opened=1995
    ),
    "Kansas City Chiefs": StadiumInfo(
        name="Arrowhead Stadium",
        team="Kansas City Chiefs",
        city="Kansas City",
        state="MO",
        latitude=39.0489,
        longitude=-94.4839,
        altitude=820,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=76416,
        opened=1972
    ),
    "Las Vegas Raiders": StadiumInfo(
        name="Allegiant Stadium",
        team="Las Vegas Raiders",
        city="Las Vegas",
        state="NV",
        latitude=36.0909,
        longitude=-115.1833,
        altitude=2001,
        roof_type=RoofType.DOME,
        surface=SurfaceType.GRASS,
        capacity=65000,
        opened=2020
    ),
    "Los Angeles Chargers": StadiumInfo(
        name="SoFi Stadium",
        team="Los Angeles Chargers",
        city="Inglewood",
        state="CA",
        latitude=33.9535,
        longitude=-118.3392,
        altitude=131,
        roof_type=RoofType.DOME,
        surface=SurfaceType.TURF,
        capacity=70240,
        opened=2020
    ),
    "Los Angeles Rams": StadiumInfo(
        name="SoFi Stadium",
        team="Los Angeles Rams",
        city="Inglewood",
        state="CA",
        latitude=33.9535,
        longitude=-118.3392,
        altitude=131,
        roof_type=RoofType.DOME,
        surface=SurfaceType.TURF,
        capacity=70240,
        opened=2020
    ),
    "Miami Dolphins": StadiumInfo(
        name="Hard Rock Stadium",
        team="Miami Dolphins",
        city="Miami Gardens",
        state="FL",
        latitude=25.9580,
        longitude=-80.2389,
        altitude=16,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=65326,
        opened=1987
    ),
    "Minnesota Vikings": StadiumInfo(
        name="U.S. Bank Stadium",
        team="Minnesota Vikings",
        city="Minneapolis",
        state="MN",
        latitude=44.9736,
        longitude=-93.2575,
        altitude=830,
        roof_type=RoofType.DOME,
        surface=SurfaceType.TURF,
        capacity=66655,
        opened=2016
    ),
    "New England Patriots": StadiumInfo(
        name="Gillette Stadium",
        team="New England Patriots",
        city="Foxborough",
        state="MA",
        latitude=42.0909,
        longitude=-71.2643,
        altitude=262,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.TURF,
        capacity=65878,
        opened=2002
    ),
    "New Orleans Saints": StadiumInfo(
        name="Caesars Superdome",
        team="New Orleans Saints",
        city="New Orleans",
        state="LA",
        latitude=29.9511,
        longitude=-90.0812,
        altitude=3,
        roof_type=RoofType.DOME,
        surface=SurfaceType.TURF,
        capacity=73208,
        opened=1975
    ),
    "New York Giants": StadiumInfo(
        name="MetLife Stadium",
        team="New York Giants",
        city="East Rutherford",
        state="NJ",
        latitude=40.8135,
        longitude=-74.0745,
        altitude=30,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.TURF,
        capacity=82500,
        opened=2010
    ),
    "New York Jets": StadiumInfo(
        name="MetLife Stadium",
        team="New York Jets",
        city="East Rutherford",
        state="NJ",
        latitude=40.8135,
        longitude=-74.0745,
        altitude=30,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.TURF,
        capacity=82500,
        opened=2010
    ),
    "Philadelphia Eagles": StadiumInfo(
        name="Lincoln Financial Field",
        team="Philadelphia Eagles",
        city="Philadelphia",
        state="PA",
        latitude=39.9008,
        longitude=-75.1675,
        altitude=39,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=69596,
        opened=2003
    ),
    "Pittsburgh Steelers": StadiumInfo(
        name="Acrisure Stadium",
        team="Pittsburgh Steelers",
        city="Pittsburgh",
        state="PA",
        latitude=40.4468,
        longitude=-80.0158,
        altitude=730,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=68400,
        opened=2001
    ),
    "San Francisco 49ers": StadiumInfo(
        name="Levi's Stadium",
        team="San Francisco 49ers",
        city="Santa Clara",
        state="CA",
        latitude=37.4032,
        longitude=-121.9698,
        altitude=43,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=68500,
        opened=2014
    ),
    "Seattle Seahawks": StadiumInfo(
        name="Lumen Field",
        team="Seattle Seahawks",
        city="Seattle",
        state="WA",
        latitude=47.5952,
        longitude=-122.3316,
        altitude=16,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.TURF,
        capacity=68740,
        opened=2002
    ),
    "Tampa Bay Buccaneers": StadiumInfo(
        name="Raymond James Stadium",
        team="Tampa Bay Buccaneers",
        city="Tampa",
        state="FL",
        latitude=27.9759,
        longitude=-82.5033,
        altitude=46,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=65618,
        opened=1998
    ),
    "Tennessee Titans": StadiumInfo(
        name="Nissan Stadium",
        team="Tennessee Titans",
        city="Nashville",
        state="TN",
        latitude=36.1665,
        longitude=-86.7713,
        altitude=597,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=69143,
        opened=1999
    ),
    "Washington Commanders": StadiumInfo(
        name="FedExField",
        team="Washington Commanders",
        city="Landover",
        state="MD",
        latitude=38.9076,
        longitude=-76.8645,
        altitude=164,
        roof_type=RoofType.OPEN,
        surface=SurfaceType.GRASS,
        capacity=67617,
        opened=1997
    ),
}


def get_stadium(team: str) -> Optional[StadiumInfo]:
    """Get stadium info for a team."""
    return STADIUM_DATA.get(team)


def get_stadium_coordinates(team: str) -> Optional[Tuple[float, float]]:
    """Get coordinates for a team's stadium."""
    stadium = STADIUM_DATA.get(team)
    return stadium.coordinates if stadium else None
