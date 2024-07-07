import openeo, xarray, geopandas, pyrosm, cv2, itertools, os, shutil, multiprocessing, shapely
from geopy.geocoders import Nominatim
from tqdm import tqdm
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from retry import retry
from shapely.geometry import Polygon
from pyproj import CRS
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def draw_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw bitmask onto image.

    Parameters
    ----------
    image : np.ndarray
        Image with dimensions W, H, C
    mask : np.ndarray
        Bitmask with dimensions W, H

    Returns
    -------
    np.ndarray
        Image with overlayed bitmask (W, H, C)
    """
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = image.copy()
    masked_image = np.where(
        mask.astype(int), np.array([0, 0, 255]), masked_image
    ).astype(np.float32)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


class Pipeline:
    def __init__(
        self,
        sat_backend: str = "https://openeo.dataspace.copernicus.eu/openeo/1.2",
        cities: list[str] = pyrosm.data.available["cities"],
        path: str = "segmentation_dataset",
        custom_bbox: dict = None,
    ):
        self.con = openeo.connect(sat_backend)
        self.con.authenticate_oidc()
        self.cities = cities
        self.__make_dirs(path)
        self.custom_bbox = custom_bbox

        # with multiprocessing.Pool(processes=3) as pool: # openeo max connections is 3 for some reason
        #     results = []
        #     for city in tqdm(self.cities, desc="Dispatching"):
        #         results.append(pool.apply_async(self.process_city, args=(city, path)))
        #     for result in tqdm(results, desc="Processing"):
        #         print(result.get())

        for city in tqdm(cities):
            self.process_city(city, path)

    def __get_city_boundaries(self, city_name):
        geolocator = Nominatim(user_agent="city_boundaries_app")
        location = geolocator.geocode(city_name, exactly_one=True)

        if location:
            bbox = location.raw["boundingbox"]
            return [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])]
        else:
            return None

    def __save_image_tiles(self, file, image, mask, path, city, date, tile_size=64, step_size=64):
        height, width, _ = image.shape

        scl_band = self.__nc_to_scl_band(file)
        ir_band = self.__nc_to_ir_band(file)
        rgb_band = self.__nc_to_rgb(file)

        tasks = []
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                tasks.append((x, y, tile_size, scl_band, ir_band, rgb_band, image, mask, path, city, date))

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.__process_tile, *task): task for task in tasks}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
                pass

    def __process_tile(self, x, y, tile_size, scl_band, ir_band, rgb_band, image, mask, path, city, date):
        tile = rgb_band[y : y + tile_size, x : x + tile_size]
        mask_tile = mask[y : y + tile_size, x : x + tile_size]

        if (
            tile.shape[0] == tile_size
            and tile.shape[1] == tile_size
            # and not self.__is_cloudy_from_bands(scl_band, x, y, tile_size)
            # and not self.__has_download_artefacts_from_band(tile, x, y, tile_size)
            # and np.any(mask_tile != 0)
            # and not self.__is_too_dark(tile)
        ):
            tile_filename = f"{path}/imgs/{city}_{date}_{x}_{y}.jpg"
            mask_tile_filename = f"{path}/masks/{city}_{date}_{x}_{y}.png"
            test_tile_filename = f"{path}/test/{city}_{date}_{x}_{y}.png"

            cv2.imwrite(
                tile_filename,
                cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(mask_tile_filename, (mask_tile * 255).astype(np.uint8))
            cv2.imwrite(
                test_tile_filename,
                cv2.cvtColor(
                    (draw_mask(tile, mask_tile) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                ),
            )

    def process_city(self, city: str, path: str):
        fp = pyrosm.get_data(city)
        if self.custom_bbox:
            bbox = self.custom_bbox[city]
        else:
            bbox = self.__get_city_boundaries(city)
        print(city, bbox)
        osm = pyrosm.OSM(fp, bbox)
        spatial_extent = self.__bbox_to_spatial_extent(bbox)
        buildings = osm.get_buildings()
        self.rasterized_buildings = None

        files = self.__save_defined(spatial_extent, f"{path}/{city}/")

        for file in os.listdir(f"{path}/{city}/"):
            rgb = self.__nc_to_rgb(os.path.join(f"{path}/{city}/", file))
            epsg_code = self.__get_epsg_code(os.path.join(f"{path}/{city}/", file))
            if self.rasterized_buildings is None:
                raster_height, raster_width, _ = rgb.shape
                self.rasterized_buildings = self.__rasterize_buildings(
                    buildings, bbox, raster_height, raster_width, crs=epsg_code
                )

            self.__save_image_tiles(
                os.path.join(f"{path}/{city}/", file),
                rgb,
                self.rasterized_buildings,
                path,
                city,
                file.replace(".nc", ""),
            )

        print(f"Successfully built segmentation data set for {city}.")

    def __get_bbox(self, boundaries):
        min_lon = boundaries["geometry"].bounds["minx"].iloc[0]
        min_lat = boundaries["geometry"].bounds["miny"].iloc[0]
        max_lon = boundaries["geometry"].bounds["maxx"].iloc[0]
        max_lat = boundaries["geometry"].bounds["maxy"].iloc[0]
        return [min_lon, min_lat, max_lon, max_lat]

    def __make_dirs(self, path):
        images_dir = f"{path}/imgs/"
        mask_dir = f"{path}/masks/"
        test_dir = f"{path}/test/"
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)


    def __is_cloudy_from_bands(
        self,
        scl_band,
        x,
        y,
        tile_size,
    ):
        tile = scl_band[y : y + tile_size, x : x + tile_size]
        target_values = [0, 1, 2, 3, 8, 9, 10]
        total_elements = tile.size
        total_count = np.sum(np.isin(tile, target_values))
        return (total_count / total_elements) > 0.2


    def __has_download_artefacts_from_band(
        self, rgb_band, x, y, tile_size, threshold=10
    ):
        rgb_tile = rgb_band[y : y + tile_size, x : x + tile_size]
        grayscale_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
        black_pixel_count = cv2.countNonZero((grayscale_tile == 0).astype(int))
        total_pixels = grayscale_tile.shape[0] * grayscale_tile.shape[1]
        black_pixel_percentage = (black_pixel_count / total_pixels) * 100
        return black_pixel_percentage > threshold
    
    def __is_too_dark(self, tile, brightness_threshold=30):
        grayscale_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(grayscale_tile)
        return mean_brightness < brightness_threshold

    def __bbox_to_spatial_extent(self, bbox) -> dict:
        west, south, east, north = bbox
        return {"west": west, "south": south, "east": east, "north": north}

    def __copernicus_datacube(self, bbox, time_range, bands):
        s2_cube = self.con.load_collection(
            "SENTINEL2_L2A",
            temporal_extent=time_range,
            spatial_extent=bbox,
            bands=bands,
            max_cloud_cover=30,
        )
        scl_band = s2_cube.band("SCL")
        cloud_mask = (scl_band == 3) | (scl_band == 8) | (scl_band == 9)
        cloud_mask = cloud_mask.resample_cube_spatial(s2_cube)
        composite_masked = s2_cube.mask(cloud_mask).mean_time()
        return composite_masked

    def __rasterize_buildings(
        self,
        buildings: geopandas.geodataframe.GeoDataFrame,
        bbox: list | tuple | np.ndarray,
        raster_height=1427,
        raster_width=1361,
        crs="EPSG:25832",
    ) -> np.ndarray:

        xmin, ymin, xmax, ymax = bbox
        # Create a Polygon for the bounding box in EPSG:4326
        bbox_geom = geopandas.GeoDataFrame(
            index=[0],
            crs="EPSG:4326",
            geometry=[
                Polygon(
                    [
                        (xmin, ymin),
                        (xmax, ymin),
                        (xmax, ymax),
                        (xmin, ymax),
                        (xmin, ymin),
                    ]
                )
            ],
        )

        if crs:
            # Transform the buildings and the bounding box to the new CRS
            buildings = buildings.to_crs(crs)
            bbox_geom = bbox_geom.to_crs(crs)

        # Extract the transformed bounding box coordinates
        transformed_bbox = bbox_geom.geometry.iloc[0].bounds
        xmin, ymin, xmax, ymax = transformed_bbox

        xres = (xmax - xmin) / raster_width
        yres = (ymax - ymin) / raster_height

        # Rasterize buildings
        raster = rasterize(
            shapes=buildings.geometry,
            out_shape=(raster_height, raster_width),
            transform=(xres, 0, xmin, 0, -yres, ymax),
        )

        # Handle potential invalid values
        raster = np.nan_to_num(raster, nan=0.0, posinf=255.0, neginf=0.0)
        raster = np.clip(raster, 0, 255)
        raster = raster.astype(np.uint8)

        return raster

    def __download_datacube_mean(self, cube, output_file):
        mean_cube = cube.mean_time()
        output_format = {"format": "netCDF"}
        output_options = {"output_parameters": output_format}
        job = mean_cube.save_result(
            filename=output_file, format="netCDF"
        )  # , options=output_options)
        job.start_and_wait()

    @retry(openeo.rest.OpenEoApiPlainError, delay=1, backoff=2, max_delay=4)
    def __save_defined(
        self,
        bbox,
        out_folder,
        total_time_range=["2023-05-01", "2024-05-01"],
        time_ranges_to_save=None,
        bands=["B04", "B03", "B02", "B08", "SCL"],
        max_cloud_cover=40,
    ) -> list[str]:
        if not time_ranges_to_save:
            time_ranges_to_save = pd.date_range(
                total_time_range[0],
                total_time_range[1],
                1,  # 24 means 23 time frames (minus ones that are too cloudy)
            ).date

        files = []
        for time_start, time_end in itertools.pairwise(time_ranges_to_save):
            datacube = self.__copernicus_datacube(
                bbox, [time_start, time_end], bands,
            )
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            filename = f"{out_folder}/{time_start}:{time_end}.nc"

            try:
                # self.__download_datacube_mean(datacube, filename)
                datacube.download(filename)
            except openeo.rest.OpenEoApiError:
                print(
                    f"No data available in timeframe {time_start}-{time_end} for {max_cloud_cover} % cloud coverage. Skipping..."
                )
                continue

            files.append(filename)

        return files

    def __nc_to_rgb(self, filename):
        ds = xarray.load_dataset(filename)
        data = (
            ds[["B04", "B03", "B02"]]
            .to_array(dim="bands")
            .to_numpy()
            .transpose(1, 2, 0)
        )

        normalized = np.clip(data / 2000, 0, 1)
        return normalized

    def __get_epsg_code(self, filename):
        ds = xarray.load_dataset(filename)
        epsg_code = CRS.from_cf(ds.crs.attrs)
        return epsg_code

    def __nc_to_rgbir(self, filename):
        ds = xarray.load_dataset(filename)
        return ds[["B04", "B03", "B02", "B08"]].to_array(dim="bands").to_numpy()

    def __nc_to_single_band(self, filename):
        ds = xarray.load_dataset(filename)
        return ds[["B04"]].to_array(dim="bands").to_numpy().squeeze()

    def __nc_to_scl_band(self, filename):
        ds = xarray.load_dataset(filename)
        return ds[["SCL"]].to_array(dim="bands").to_numpy().squeeze()

    def __nc_to_ir_band(self, filename):
        ds = xarray.load_dataset(filename)
        return ds[["B08"]].to_array(dim="bands").to_numpy().squeeze()


if __name__ == "__main__":
    pipeline = Pipeline(
        cities=[
            "Amsterdam",
            "Bremen",
            "Madrid",
            "Manchester",
            "Barcelona",
            "Copenhagen",
            "Hamburg",
            "Warsaw",
            "Paris",
            "London",
        ],
        path="segmentation_dataset_train",
    )

    pipeline = Pipeline(
        cities=[
            "Augsburg",
            "Moscow",
            "Istanbul",
        ],
        path="segmentation_dataset_val",
    )

    pipeline = Pipeline(
        cities=["Berlin"],
        path="segmentation_dataset_test",
        custom_bbox={"Berlin": [13.294333, 52.454927, 13.500205, 52.574409]},
    )
