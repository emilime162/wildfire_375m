import os
import json
import math
from datetime import datetime, timedelta
from zipfile import ZipFile

import ee
import numpy as np
from shapely.ops import transform as shapely_tf
import affine
import geopandas as gpd
import pandas as pd
import pyproj
import rasterio
from geocube.api.core import make_geocube
from pyproj import CRS
from shapely.geometry import mapping, MultiPoint, shape
from shapely.ops import transform, nearest_points
from sklearn.cluster import DBSCAN

from src.geospatial import (
    build_vrt,
    buffer_point,
    reproject_coordinates,
    bounds_to_geojson,
    read_geospatial_file,
)
from src.constants import GEE_PROJECT_ID, DEFAULT_SPATIAL_RESOLUTION, TARGET_EPSG_CODE, TARGET_CRS, CHIP_SPACE_LENGTH

try:
    ee.Initialize(project=GEE_PROJECT_ID)  # Use your project ID
except ee.EEException:
    print("Earth Engine is already initialized.")


def unzip_csvs(zip_file):
    """
    Unzip csvs from a file
    :param zip_file: path of zip file to unzip
    :return: list of unzipped csv filepaths
    """
    unzipped = []
    with ZipFile(zip_file, "r") as zip_obj:
        for zipped_file in zip_obj.namelist():
            if zipped_file.endswith(".csv"):
                zip_obj.extract(zipped_file, zip_file.parent)
                unzipped.append(zip_file.parent.joinpath(zipped_file))
    return unzipped


def cluster_fires(fire_dataframe, min_cluster_points=25):
    """
    Given a geodataframe of fire points, for each date, create clusters
    :param fire_dataframe: geodataframe of fire points
    :param min_cluster_points: minimum number of fire points in a cluster for it to be kept
    :return: geodataframe of fire points that belong to a cluster
    """
    clustered_fires_for_dates = []
    number_of_clusters = 0
    for date in fire_dataframe["acq_date"].unique().tolist():
        fires_for_date = fire_dataframe[fire_dataframe["acq_date"] == date]
        fire_clusters = DBSCAN(eps=0.01, min_samples=1).fit(
            fires_for_date[["longitude", "latitude"]].values
        )
        # add clusters label
        cluster_labels = fire_clusters.labels_ + number_of_clusters
        number_of_clusters += fire_clusters.labels_.max()

        cluster_labels = pd.Series(cluster_labels, name="label")
        # shift to match date selection
        cluster_labels.index += fires_for_date.index.min()
        clustered_fires_for_dates.append(
            pd.concat([fires_for_date, cluster_labels], axis=1)
        )

    clustered_fires = pd.concat(clustered_fires_for_dates)
    # drop clusters with < min_cluster_points
    label_counts = clustered_fires["label"].value_counts()
    clustered_fires["label"] = clustered_fires["label"].apply(
        lambda x: x if label_counts[x] >= min_cluster_points else -1
    )
    clustered_fires = clustered_fires[clustered_fires["label"] != -1]

    # reset label to be continuous
    clustered_fires['label'] = clustered_fires.groupby('label').ngroup()
    return clustered_fires


def create_chip_bounds(clustered_fires):
    """
    Given a geodataframe of clustered fire points create chip bbox and save metadata to csv
    :param clustered_fires: geodataframe of clustered fire points
    :param output_fp: directory for output file to be written to
    :return: filepath of the records.csv created/updated
    """
    chip_bounds = []
    for cluster in clustered_fires["label"].unique().tolist():
        clustered_fire = clustered_fires[clustered_fires["label"] == cluster]
        date = clustered_fire["acq_date"].values[0]
        multipoint_fire_feature = MultiPoint([x for x in clustered_fire.geometry])
        # convert to polygon & get centre
        multipoint_fire_feature_centre = multipoint_fire_feature.convex_hull.centroid
        # get closest point to centre
        central_fire_point = nearest_points(
            multipoint_fire_feature, multipoint_fire_feature_centre
        )[0]

        # build bbox around the clusters central fire point
        bbox_4326, utm_crs = buffer_point(
            central_fire_point, buffer_m=15750, output_4326=True
        )
        bbox_4326_geojson = json.dumps(
            mapping(transform(lambda x, y: (y, x), bbox_4326))
        )
        chip = make_geocube(
            vector_data=gpd.GeoDataFrame(
                geometry=[central_fire_point], crs=TARGET_CRS
            ),
            resolution=(-DEFAULT_SPATIAL_RESOLUTION, DEFAULT_SPATIAL_RESOLUTION),
            output_crs=utm_crs,
            geom=bbox_4326_geojson,
        )

        chip_bounds.append([cluster, *chip.rio.bounds(), utm_crs.to_epsg(), date])

    return pd.DataFrame(
        chip_bounds, columns=["idx", "left", "bottom", "right", "top", "epsg", "date"]
    )



def fires_from_topleft(top_left, epsg_code, date_to_query, fires):
    """
    Given input chip parameters, load fire data and rasterize the points
    :param top_left: list of the top left coordinates of the chip
    :param epsg_code: EPSG code for top_left
    :param date_to_query: date of the fire data to load
    :param fires : gpd.GeoDataFrame or filename
    :return: xarray.Dataset containing rasterized fire points
    """
    aoi = bounds_to_geojson(
        rasterio.coords.BoundingBox(
            left=top_left[1],
            right=top_left[1] + CHIP_SPACE_LENGTH,
            bottom=top_left[0] - CHIP_SPACE_LENGTH,
            top=top_left[0],
        )
    )
    # reproj the bbox from utm to 4326
    utm_to_wgs84_transformer = pyproj.Transformer.from_crs(
        epsg_code, TARGET_EPSG_CODE, always_xy=True
    ).transform
    aoi_wgs84 = shapely_tf(utm_to_wgs84_transformer, shape(aoi))

    # load fire data intersecting chip bbox
    if isinstance(fires, str):
        fires_in_chip = gpd.read_file(fires, layer="merge", bbox=aoi_wgs84)
    else:
        chip_poly = gpd.GeoDataFrame(geometry=[aoi_wgs84], crs=TARGET_CRS)
        fires_in_chip = fires[fires["acq_date"] == date_to_query].clip(chip_poly)

    fires_in_chip = fires_in_chip[fires_in_chip["acq_date"] == date_to_query]

    if fires_in_chip.empty:
        # possible if fire dies "next day"
        fires_in_chip = gpd.GeoDataFrame(geometry=[aoi_wgs84.centroid], crs=TARGET_CRS)
        fires_in_chip["bool"] = 0
        fires_in_chip["frp"] = 0
    else:
        fires_in_chip["bool"] = 1
        fires_in_chip["frp"] = pd.to_numeric(fires_in_chip["frp"])

    bbox_4326, utm_crs = buffer_point(
        aoi_wgs84.centroid, buffer_m=11750, output_4326=True
    )
    bbox_4326_geojson = json.dumps(mapping(transform(lambda x, y: (y, x), bbox_4326)))

    # rasterize
    fire_array = make_geocube(
        vector_data=fires_in_chip,
        measurements=["bool", "frp"],
        resolution=(-DEFAULT_SPATIAL_RESOLUTION, DEFAULT_SPATIAL_RESOLUTION),
        output_crs=epsg_code,
        fill=0,
        geom=bbox_4326_geojson,
    )
    return fire_array


def population_from_topleft(top_left, epsg, date_to_query):
    """
    Given input chip parameters, load population data from GEE GHSL dataset and reproject to the chip CRS
    :param top_left: list of the top left coordinates of the chip
    :param epsg: EPSG code for top_left
    :param date_to_query: date to load data for as string '2021-05-01'
    :return: numpy array of the population data
    """
    year = datetime.strptime(date_to_query, "%Y-%m-%d").year
    # Create bounding box for the area of interest
    aoi = bounds_to_geojson(
        rasterio.coords.BoundingBox(
            left=top_left[1],
            right=top_left[1] + CHIP_SPACE_LENGTH,
            bottom=top_left[0] - CHIP_SPACE_LENGTH,
            top=top_left[0],
        )
    )
    # Convert the AOI to GEE geometry
    aoi_4326 = reproject_coordinates(aoi, epsg, TARGET_EPSG_CODE)
    region = ee.Geometry(aoi_4326)
    # Get the closest year
    remain = year % 5
    comple = 5 - remain
    valid_year = year + comple if remain > 2 else year - remain
    # Filter to get the population data for the specified year
    population = ee.Image(f'JRC/GHSL/P2023A/GHS_POP/{valid_year}')
    population_data = population.reproject(crs=f'EPSG:{epsg}', scale=375).sampleRectangle(region).getInfo()
    population_array = np.array(population_data['properties']['population_count'])
    population_array = np.nan_to_num(population_array, nan=0.0, posinf=0.0, neginf=0.0)
    return population_array


def ndvi_from_topleft(top_left, epsg, date):
    """
    Fetch vegetation data from the NASA/VIIRS/002/VNP13A1 dataset on GEE.

    :param topleft: Coordinates of the top-left corner of the AOI [latitude, longitude].
    :param epsg_code: EPSG code for the coordinate system of the AOI.
    :param date: Date string (YYYY-MM-DD) for querying data.
    :return: Dictionary with data arrays for the specified parameters.
    """
    # Parse the date
    date_to_query = datetime.strptime(date, "%Y-%m-%d")
    start_date = date_to_query.strftime('%Y-%m-%d')
    end_date = (date_to_query + timedelta(days=15)).strftime('%Y-%m-%d')
    aoi = bounds_to_geojson(
        rasterio.coords.BoundingBox(
            left=top_left[1],
            right=top_left[1] + CHIP_SPACE_LENGTH,
            bottom=top_left[0] - CHIP_SPACE_LENGTH,
            top=top_left[0],
        )
    )
    aoi_4326 = reproject_coordinates(aoi, epsg, TARGET_EPSG_CODE)
    aoi = ee.Geometry(aoi_4326)

    # Load the VNP13A1 dataset
    try:
        vegetation = ee.ImageCollection("NASA/VIIRS/002/VNP13A1") \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date)
        filtered_size = vegetation.size().getInfo()
        #print(f"Number of images in filtered collection: {filtered_size}")
        if filtered_size == 0:
            print("No images found for the specified AOI and date range.")
            return None
    except Exception as e:
        print(f"Error loading or filtering the image collection: {e}")
        raise
    # Select NDVI band and resample
    try:
        ndvi_data = vegetation.select("NDVI").mean().reproject(
            f'EPSG:{epsg}', scale=DEFAULT_SPATIAL_RESOLUTION
        )
        # Extract raster data
        data = ndvi_data.sampleRectangle(region=aoi).getInfo()
        if not data:
            print("No NDVI data found for the specified AOI and date.")
            return None
        # Convert the raster data into a NumPy array
        ndvi_array = np.array(data['properties']['NDVI'])
        print(f"Fetched NDVI raster with shape {ndvi_array.shape}.")
        return ndvi_array
    except Exception as e:
        print(f"Error fetching NDVI data: {e}")
        return None


def elevation_from_topleft(top_left, epsg, resolution=30):
    """
    Given input chip parameters, load elevation data and reproject to the chip CRS.
    This version calculates DEM tile names directly based on AOI.
    :param top_left: list of the top-left coordinates of the chip [latitude, longitude]
    :param epsg: EPSG code for the chip CRS
    :param resolution: Resolution of DEM in arc seconds (default: 30 for GLO-30)
    :return: numpy array of the elevation data
    """
    # Define the AOI in terms of latitude and longitude
    aoi = bounds_to_geojson(
        rasterio.coords.BoundingBox(
            left=top_left[1],
            right=top_left[1] + CHIP_SPACE_LENGTH,
            bottom=top_left[0] - CHIP_SPACE_LENGTH,
            top=top_left[0],
        )
    )
    aoi_4326 = reproject_coordinates(aoi, epsg, TARGET_EPSG_CODE)

    aoi_bounds = shape(aoi_4326).bounds  # Get (minx, miny, maxx, maxy) in longitude/latitude
    left, bottom, right, top = aoi_bounds
    
    # Calculate tile names based on AOI
    def get_tile_names(top, bottom, left, right, resolution):
        tiles = []
        for lat in range(math.floor(bottom), math.ceil(top)):
            lat_prefix = "N" if lat >= 0 else "S"
            lat_str = f"{lat_prefix}{abs(lat):02d}_00"
            for lon in range(math.floor(left), math.ceil(right)):
                lon_prefix = "E" if lon >= 0 else "W"
                lon_str = f"{lon_prefix}{abs(lon):03d}_00"
                tile_name = f"Copernicus_DSM_COG_{resolution}_{lat_str}_{lon_str}_DEM"
                tiles.append(tile_name)
        return tiles

    dem_tiles = get_tile_names(top, bottom, left, right, resolution)
    # Construct paths for tiles
    dem_root = "/vsis3/copernicus-dem-90m"
    file_paths = [f"{dem_root}/{tile}/{tile}.tif" for tile in dem_tiles]

    # Create VRT from the file paths
    vrt_path = build_vrt(file_paths)

    # Open the VRT and read the elevation data
    with rasterio.open(vrt_path) as src:
        dst_crs = CRS.from_epsg(epsg)
        dst_transform = affine.Affine(DEFAULT_SPATIAL_RESOLUTION, 0.0, top_left[1], 0.0, -DEFAULT_SPATIAL_RESOLUTION, top_left[0])
        elevation_data, tf = read_geospatial_file(aoi, dst_crs, dst_transform, src)
        os.remove(vrt_path)  # Clean up temporary VRT file
    return elevation_data[0]


def landcover_from_topleft(top_left, epsg):
    """
    Given input chip parameters, load landcover data and reproject to the chip CRS
    :param top_left: list of the top left coordinates of the chip
    :param epsg_code: EPSG code for top_left
    :return: numpy array of the landcover data
    """
    aoi = bounds_to_geojson(
        rasterio.coords.BoundingBox(
            left=top_left[1],
            right=top_left[1] + CHIP_SPACE_LENGTH,
            bottom=top_left[0] - CHIP_SPACE_LENGTH,
            top=top_left[0],
        )
    )
    with rasterio.open(
        "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    ) as src:
        dst_crs = CRS.from_epsg(epsg)
        dst_transform = affine.Affine(DEFAULT_SPATIAL_RESOLUTION, 0.0, top_left[1], 0.0, -DEFAULT_SPATIAL_RESOLUTION, top_left[0])
        landcover_data, tf = read_geospatial_file(aoi, dst_crs, dst_transform, src)
    return landcover_data[0]


def atmospheric_from_topleft(top_left, epsg, date, params):
    """
    Fetch ERA5 data from GEE for a specific date and region.

    :param topleft: Coordinates of the top-left corner of the AOI [latitude, longitude].
    :param epsg: EPSG code for the coordinate system of the AOI.
    :param date: Date string (YYYY-MM-DD) for querying data.
    :param params: List of desired bands/parameters to fetch (e.g., 'TMP_2maboveground').
    :return: Dictionary with hourly data arrays for the specified parameters.
    """
    # Parse the date and define start/end times
    date_to_query = datetime.strptime(date, "%Y-%m-%d")
    start_date = date_to_query.strftime('%Y-%m-%dT00:00')
    end_date = (date_to_query + timedelta(days=1)).strftime('%Y-%m-%dT00:00')

    aoi = bounds_to_geojson(
        rasterio.coords.BoundingBox(
            left=top_left[1],
            right=top_left[1] + CHIP_SPACE_LENGTH,
            bottom=top_left[0] - CHIP_SPACE_LENGTH,
            top=top_left[0],
        )
    )
    aoi_4326 = reproject_coordinates(aoi, epsg, TARGET_EPSG_CODE)
    region = ee.Geometry(aoi_4326)
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(region).filterDate(start_date, end_date)
    # Create a dictionary to hold hourly data
    results = {}
    # Fetch each parameter
    for param in params:
        try:
            param_data = collection.select(param).mean().reproject(crs=f'EPSG:{epsg}', scale=DEFAULT_SPATIAL_RESOLUTION)
            data = param_data.sampleRectangle(region=region).getInfo()
            param_array = np.array(data['properties'][param])
            param_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
            results[param] = param_array
        except Exception as e:
            print(f"Error fetching parameter '{param}': {e}")
            results[param] = None
    return results
