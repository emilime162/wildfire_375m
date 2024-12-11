import os

# DEFAULT_PARAMS = [
#     "air_pressure_at_mean_sea_level",
#     "air_temperature_at_2_metres",
#     "air_temperature_at_2_metres_1hour_Maximum",
#     "air_temperature_at_2_metres_1hour_Minimum",
#     "dew_point_temperature_at_2_metres",
#     "eastward_wind_at_100_metres",
#     "eastward_wind_at_10_metres",
#     "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation",
#     "lwe_thickness_of_surface_snow_amount",
#     "northward_wind_at_100_metres",
#     "northward_wind_at_10_metres",
#     "precipitation_amount_1hour_Accumulation",
#     "sea_surface_temperature",
#     "snow_density",
#     "surface_air_pressure",
# ]



DEFAULT_PARAMS = [
    "temperature",               # Air temperature at 2 meters above the surface (°C) - impacts fire ignition and spread.
    "specific_humidity",         # Specific humidity at 2 meters above the surface (kg/kg) - affects fuel moisture.
    "pressure",                  # Surface pressure (Pa) - related to weather conditions and fire behavior.
    "wind_u",                    # U wind component at 10 meters above the surface (m/s) - horizontal wind in the east-west direction.
    "wind_v",                    # V wind component at 10 meters above the surface (m/s) - horizontal wind in the north-south direction.
    #"longwave_radiation",        # Surface downward longwave radiation (W/m²) - energy balance affecting temperature and humidity.
    #"shortwave_radiation",       # Surface downward shortwave radiation (W/m²) - impacts surface temperature.
    "potential_evaporation",     # Potential evaporation (kg/m²) - indicates dryness and fire-prone conditions.
    "total_precipitation"        # Hourly total precipitation (kg/m²) - critical for assessing wet or dry conditions.
]


CHIP_SIZE = (64, 64)
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
