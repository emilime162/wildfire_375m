# NEU Wildfire Spread Prediction Project

## Description

This folder contains the code for generating a dataset designed for training a fire prediction model. The dataset is a **multi-temporal, multi-modal remote sensing collection** for predicting the spread of active wildfires with a temporal resolution of **12 hours (twice daily)**. It covers North America over a **12-year period (2012–2024)**.

Each chip contains:
- Fire masks for **six days**:
  - **Four days before** the fire event.
  - The **fire day** itself.
  - **The next day**.
- **Two fire masks per day**: daytime and nighttime, capturing fire activity at different times.
- Auxiliary data for the fire day, including **elevation**, **atmospheric conditions**, and **vegetation information**, to provide critical context for modeling fire behavior.

## Dataset Preparation

The data sources used are from the following platforms:

| **Data Source**                                                        | **Description**                                                                                                                                                                                                                                                         |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Copernicus DEM](https://registry.opendata.aws/copernicus-dem/)        | Digital Surface Model (DSM) representing the Earth's surface, including buildings, infrastructure, and vegetation.                                                                                                                                                     |
| [VNP13A1 Vegetation Indices](https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP13A1) | Provides vegetation indices through a 16-day acquisition process at a **500-meter resolution** from the VIIRS sensor.                                                                                                            |
| [ESA WorldCover](https://registry.opendata.aws/esa-worldcover/)        | Global land cover map with **11 land cover classes** at **10-meter resolution**, combining Sentinel-1 and Sentinel-2 data.                                                                                                       |
| [ERA5 Atmospheric Data](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR)  | Reanalysis dataset providing consistent atmospheric data with enhanced spatial and temporal resolution.                                                                                                                          |
| [GHSL Population Data](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_POP?hl=en#description) | Represents residential population distribution as raster data, providing estimates between 1975 and 2030 derived from census data disaggregated into grid cells.                                                                 |
| [FIRMS Active Fire Data](https://firms.modaps.eosdis.nasa.gov/)        | Near Real-Time (NRT) active fire data available within 3 hours of observation from VIIRS at **375-meter resolution**.                                                                                                           |

### Fire Masks
Fire masks represent areas actively on fire on a given day. These masks are generated using **VIIRS active fire hotspot data** available through [FIRMS](https://firms.modaps.eosdis.nasa.gov/). 

Instead of generating a chip for every hotspot:
1. Fire points are **clustered**.
2. Clusters with **fewer than 25 fire points within a 24-hour period** are dropped.

For each chip:
- Fire masks are included for:
  - **Four days before the fire event**.
  - The **fire day** itself.
  - **The next day**.
- Each day has **two fire masks**: one for daytime and one for nighttime.

### Atmospheric Data
Atmospheric data is sourced from **ERA5** and is available on [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR). 

#### Extracted Bands:
- `dewpoint_temperature_2m`
- `temperature_2m`
- `u_component_of_wind_10m`
- `v_component_of_wind_10m`
- `total_evaporation_sum`
- `surface_pressure`
- `total_precipitation_sum`

#### Process:
1. Data is imported from Google Earth Engine.
2. Extracted for the **Area of Interest (AOI)**.
3. Resampled to match the **chip’s coordinate reference system (CRS)**, **pixel size**, and **time (1 day)**.

### Land Cover Data
Land cover data is sourced from **ESA WorldCover** ([link](https://registry.opendata.aws/esa-worldcover/)).

- These data are reprojected to the chip's **CRS** using [Rasterio](https://rasterio.readthedocs.io/en/latest/).

### Elevation Data
Elevation data is sourced from **Copernicus DEM** ([link](https://registry.opendata.aws/copernicus-dem/)).

- The data is reprojected to the chip's **CRS** using Rasterio.

### Vegetation Data (VNP13A1)
The vegetation indices are sourced from **VNP13A1** ([link](https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP13A1)).The NDVI band is extracted.

For a given chip:
1. Data is imported from Google Earth Engine.
2. Extracted for the **AOI**.
3. Resampled to match the **chip’s CRS**, **pixel size**, and **time (1 day)**.

### Population Data
Population data is sourced from the **GHSL Population Data** ([link](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_POP?hl=en)).

1. Since the population data is based on 5-year interval, the closest year is calculated from given date.
2. Extract the data image from Google Earth Engine.
3. Resample to match the **chip’s CRS**, **pixel size**

## Data Cleaning

After getting the chips for active wildfires, a few more steps will be taken in order to normalize the data:

- Irregular larger shaped chips of additional features such as vegetation or population data will be cropped.
- Irregular smaller shaped chips of additional features such as vegetation or population data will be filtered.
- Irregular chips of fire masks will be filtered.

## Processing Workflow
The data preparation workflow is represented below:

<p align="center">
  <img src="images/workflow.svg" width="750">
</p>

## Reference

[SatelliteVu-AWS-Disaster-Response-Hackathon](https://github.com/SatelliteVu/SatelliteVu-AWS-Disaster-Response-Hackathon)
