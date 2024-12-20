from concurrent.futures import ThreadPoolExecutor
import warnings
from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import rasterio
from rasterio.errors import RasterioIOError
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize

from src.constants import DEFAULT_PARAMS
from src.data_sources import (
    unzip_csvs,
    cluster_fires,
    create_chip_bounds,
    ndvi_from_topleft,
    landcover_from_topleft,
    atmospheric_from_topleft,
    population_from_topleft,
    fires_from_topleft,
    elevation_from_topleft,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["AWS_NO_SIGN_REQUEST"] = "True"

output_fp = "./canada-2021"
Path(output_fp).mkdir(parents=True, exist_ok=True)
fires = ["./fire_2021.zip"]
downloaded_fires = []
fire_gpkgs = []
manifest = pd.DataFrame()

for fire in fires:
    unzipped = unzip_csvs(Path(fire))

    # read csv - 2021 zip has two csvs - archive + NRT
    for fire_csv in unzipped:
        print(f"Loading fire_csv: {fire_csv}")
        df = gpd.read_file(fire_csv)
        df_fire = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        )

        # saving fire points to geopackage
        fire_gpkg = fire_csv.with_suffix(".gpkg")
        print(f"Saving fires to gpkg: {fire_gpkg}")
        df_fire[["acq_date", "frp", "geometry", "daynight"]].to_file(
            fire_gpkg, driver="GPKG"
        )
        fire_gpkgs.append(fire_gpkg)

        # create clusters
        print("Clustering fires")
        df_fire_clustered = cluster_fires(df_fire)

        # create/append chip bounds to records csv
        print("Creating chip bounds")
        chip_bounds = create_chip_bounds(df_fire_clustered)
        if manifest.empty:
            manifest = chip_bounds
        else:
            chip_bounds.idx = chip_bounds.idx + manifest.idx.max() + 1
            manifest = pd.concat([manifest, chip_bounds])

        # delete unzipped csv
        fire_csv.unlink()

# merge geopackages into one
output_gpkg = Path(output_fp).joinpath("fires.gpkg")
for n, fire_gpkg in enumerate(fire_gpkgs):
    print(n, fire_gpkg)
    if n == 0:
        subprocess.run(["ogr2ogr", str(output_gpkg), fire_gpkg, "-nln", "merge"])
    else:
        subprocess.run(
            [
                "ogr2ogr",
                "-update",
                "-append",
                str(output_gpkg),
                fire_gpkg,
                "-nln",
                "merge",
            ]
        )
    fire_gpkg.unlink()


chips = list(manifest.T.to_dict().values())
print(f"Chips total = {len(chips)}")


def process_chip(chip, output_fp, fires, training=True):
    """
    Given a chips metadata, load all of the training data and write numpy files, finally upload results to S3
    :param chip: records.csv chip to process data for
    :param output_fp: local directory to write data to
    :param fires: gpd.GeoDataFrame or path to vector file containing fire point data
    :param training: bool, if true then will load/write next days fires
    """
    chip_idx, left, bottom, top, right, epsg, chip_date = (
        chip["idx"],
        chip["left"],
        chip["bottom"],
        chip["top"],
        chip["right"],
        chip["epsg"],
        chip["date"],
    )

    if os.path.exists(output_fp + f"/{chip_idx}"):
        return

    print(f"Processing chip: {chip_idx}")

    # create output dir if it doesnt already exist
    output_dir = Path(output_fp).joinpath(str(chip_idx))
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(-4, 1):
        date = (datetime.strptime(chip_date, "%Y-%m-%d") + timedelta(days=i)).strftime(
            "%Y-%m-%d"
        )
        process_chip_by_day(output_dir, date, top, bottom, left, right, epsg, fires)

    # Load tomorrow's fires if training
    if training:
        tomorrows_date = (
            datetime.strptime(chip_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        tomorrows_fires = fires_from_topleft(
            [top, left], epsg, tomorrows_date, fires=fires
        )
        np.save(output_dir / "tomorrows_fires.npy", tomorrows_fires.bool)
        np.save(output_dir / "tomorrows_frp.npy", tomorrows_fires.frp)
        # Load tomorrow's day fires
        tomorrows_day_fires = fires_from_topleft(
            [top, left], epsg, tomorrows_date, fires=fires[fires["daynight"] == "D"]
        )
        np.save(output_dir / "tomorrows_day_fires.npy", tomorrows_day_fires.bool)
        np.save(output_dir / "tomorrows_day_frp.npy", tomorrows_day_fires.frp)
        # Load tomorrow's night fires
        tomorrows_night_fires = fires_from_topleft(
            [top, left], epsg, tomorrows_date, fires=fires[fires["daynight"] == "N"]
        )
        np.save(
            output_dir / "tomorrows_night_fires.npy",
            tomorrows_night_fires.bool,
        )
        np.save(output_dir / "tomorrows_night_frp.npy", tomorrows_night_fires.frp)


def process_chip_by_day(base_dir, date, top, bottom, left, right, epsg, fires):
    output_dir = base_dir / date
    output_dir.mkdir(parents=True, exist_ok=True)

    # load modis
    try:
        ndvi = ndvi_from_topleft([top, left], epsg, date)
        np.save(output_dir.joinpath("ndvi.npy"), ndvi)
    except RasterioIOError:
        # modis missing from bucket
        shutil.rmtree(output_dir)
        return

    # save bbox to geojson
    bounds_utm = rasterio.coords.BoundingBox(
        left=left, right=right, bottom=bottom, top=top
    )
    gpd.GeoSeries([box(*bounds_utm)]).set_crs(epsg).to_file(
        output_dir.joinpath("bbox.geojson")
    )

    # load today's fires
    todays_fires = fires_from_topleft([top, left], epsg, date, fires=fires)
    np.save(output_dir / "todays_fires.npy", todays_fires.bool)
    np.save(output_dir / "todays_frp.npy", todays_fires.frp)
    # Load today's day fires
    todays_day_fires = fires_from_topleft(
        [top, left], epsg, date, fires=fires[fires["daynight"] == "D"]
    )
    np.save(output_dir / "todays_day_fires.npy", todays_day_fires.bool)
    np.save(output_dir / "todays_day_frp.npy", todays_day_fires.frp)
    # Load today's night fires
    todays_night_fires = fires_from_topleft(
        [top, left], epsg, date, fires=fires[fires["daynight"] == "N"]
    )
    np.save(output_dir / "todays_night_fires.npy", todays_night_fires.bool)
    np.save(output_dir / "todays_night_frp.npy", todays_night_fires.frp)

    # load dem
    try:
        dem = elevation_from_topleft([top, left], epsg, resolution=30)
        np.save(output_dir.joinpath("elevation.npy"), dem)
    except Exception as e:
        print(f"Error loading DEM data: {e}")

    # load landcover
    landcover = landcover_from_topleft([top, left], epsg)
    np.save(output_dir.joinpath("landcover.npy"), landcover)

    # load population
    population = population_from_topleft([top, left], epsg, date)
    np.save(output_dir.joinpath("population.npy"), population)

    # load atmospheric
    try:
        atmos = atmospheric_from_topleft([top, left], epsg, date, DEFAULT_PARAMS)
        for (
            param,
            data,
        ) in atmos.items():
            try:
                np.save(output_dir.joinpath(f"{param}.npy"), data)
            except Exception as e:
                print(f"Error processing parameter {param}: {e}")
    except Exception as e:
        print(f"Error loading atmospheric data: {e}")


def smooth_data(data, sigma=6):
    return gaussian_filter(data, sigma=sigma)


# Create normalization for consistent color scaling
def normalize_data(data):
    return Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))(data)


def filter(output_fp):
    odd_chips = set()
    for chip_dir in Path(output_fp).iterdir():
        if not _validate_data_shape(chip_dir, (64, 64)):
            odd_chips.add(chip_dir)
    for chip in odd_chips:
        shutil.rmtree(chip)
    print(f"Filtered {len(odd_chips)} irregular chips")


def _validate_data_shape(data_dir, shape) -> bool:
    for base, _, files in data_dir.walk():
        for file in files:
            feature, ext = file.split(".")
            if feature in DEFAULT_PARAMS:
                data = np.load(base / file, allow_pickle=True)
                data = normalize_data(smooth_data(data))
            elif ext == "npy":
                data = np.load(base / file)
            else:
                continue
            if data.shape != shape:
                return False
    return True


if __name__ == "__main__":
    fires = gpd.read_file(output_gpkg)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_work = [
            executor.submit(process_chip, chip, output_fp, fires) for chip in chips
        ]
    filter(output_fp)
    output_gpkg.unlink()
