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

## Prerequisites

To create the dataset, ensure you meet the following prerequisites:

1. **Access to Google Earth Engine (GEE):**
   - Sign up for a [Google Earth Engine account](https://earthengine.google.com/).
   - Set up the GEE Python API on your machine.

2. **Install Required Python Packages:**
   - Install all necessary dependencies listed in the `requirements.txt` file by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Authenticate with GEE:**
   - After installing the GEE Python API, authenticate your account by running:
     ```bash
     earthengine authenticate
     ```
   - Follow the instructions linked here (https://developers.google.com/earth-engine/guides/auth) to log in with your Google account and link it to your project.

4. **Set Up Your Google Earth Engine Project ID:**
   - Ensure you have access to a GEE project.
   - Import your project ID into the script via the `constants.py` file or set it as an environment variable. 

## How to extract the data
1. **Download the fire zip data**:
You can download fire data from the NASA FIRMS website(https://firms2.modaps.eosdis.nasa.gov/country/). Select your region and time period to download zipped CSV files (e.g., fire_YYYY.zip).
Place all the downloaded files in a folder, such as ./data.
2. **Update the script**
You can use the extract.py to extract all the data -- Edit the fires variable in the script to include all your zip file paths.
3. **Run the script**
Run the script from the command line:

```bash
python extract.py
```
The script will:
Unzip the fire data files.
Cluster fire points and create metadata for each chip.
Extract geospatial features (e.g., NDVI, landcover, atmospheric data) for each chip.
Save the processed data in the specified output directory.
4. **Verify the Output**
After the script completes:

The output_fp directory will contain:
Processed Chips: Subdirectories for each geographical chip.
Merged GeoPackage: fires.gpkg file containing all fire data.
Cleaned and filtered chips with consistent data.

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

### What is a Chip?
A **chip** is a fixed-size, spatially bounded subset of the larger geospatial dataset, representing a specific region of interest. Each chip:

1. **Geospatial Focus**:
   - Represents a specific geographic area, typically structured as a grid (e.g., 64x64 cells) at a predefined spatial resolution (e.g., 375m per cell).

2. **Multi-Temporal Data**:
   - Includes data collected over multiple time points, such as fire masks and auxiliary features, providing a temporal snapshot of wildfire activity and conditions.

3. **Multi-Modal Layers**:
   - Contains various types of data, such as:
     - **Fire Masks**: Binary grids indicating fire presence for both daytime and nighttime.
     - **Elevation**: Topographic data crucial for understanding fire spread.
     - **Atmospheric Conditions**: Data like temperature, humidity, surface pressure, total precipitation and wind.
     - **Vegetation Information**: Indices like NDVI to understand fuel availability.
     - **Population Data**: population_count is extracted.

4. **Self-Contained Units**:
   - Each chip is a self-contained unit with all necessary features and labels for machine learning models. This modular structure simplifies processing and enables scalable analysis.

5. **Temporal Evolution**:
   - Captures the progression of wildfire activity and conditions over a six-day period, offering insights into both short-term and long-term patterns.

By organizing data into chips, this dataset provides a structured, consistent format that is optimized for training predictive models on wildfire behavior and spread. This approach also ensures efficient handling of large-scale geospatial data while preserving the temporal and spatial granularity required for accurate predictions.

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
