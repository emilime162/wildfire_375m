import os

DEFAULT_PARAMS = [
    "temperature",  # Air temperature at 2 meters above the surface (°C) - impacts fire ignition and spread.
    "specific_humidity",  # Specific humidity at 2 meters above the surface (kg/kg) - affects fuel moisture.
    "pressure",  # Surface pressure (Pa) - related to weather conditions and fire behavior.
    "wind_u",  # U wind component at 10 meters above the surface (m/s) - horizontal wind in the east-west direction.
    "wind_v",  # V wind component at 10 meters above the surface (m/s) - horizontal wind in the north-south direction.
    # "longwave_radiation",        # Surface downward longwave radiation (W/m²) - energy balance affecting temperature and humidity.
    # "shortwave_radiation",       # Surface downward shortwave radiation (W/m²) - impacts surface temperature.
    "potential_evaporation",  # Potential evaporation (kg/m²) - indicates dryness and fire-prone conditions.
    "total_precipitation",  # Hourly total precipitation (kg/m²) - critical for assessing wet or dry conditions.
]
DEFAULT_SPATIAL_RESOLUTION = 375

CHIP_SIZE = (64, 64)
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID")
