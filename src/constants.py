import os

DEFAULT_PARAMS = [

    "dewpoint_temperature_2m",
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_evaporation_sum",
    "surface_pressure",
    "total_precipitation_sum",
]
DEFAULT_SPATIAL_RESOLUTION = 375

CHIP_SIZE = (64, 64)
SIDE_LEN = 24000
SIDE_LEN_OFFSET = 375

FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY", "96db9d3c88f83e5429c07d04c25fa94f")
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "wildfire-440805")

TARGET_EPSG_CODE = 4326
TARGET_CRS = f"EPSG:{TARGET_EPSG_CODE}"
