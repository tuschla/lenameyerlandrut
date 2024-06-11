import openeo, xarray, geopandas, pyrosm, cv2, itertools, os, shutil, multiprocessing, shapely
from geopy.geocoders import Nominatim
from tqdm import tqdm
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from retry import retry


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
    ):
        self.con = openeo.connect(sat_backend)
        self.con.authenticate_oidc()
        self.cities = cities
        self.__make_dirs(path)

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
            bbox = location.raw['boundingbox']
            return [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])]
        else:
            return None

    def process_city(self, city: str, path: str):
        fp = pyrosm.get_data(city)
        #osm = pyrosm.OSM(fp)
        #bbox = self.__get_bbox(osm.get_boundaries("administrative", name=city))
        #bbox = list(np.round(bbox, 6))
        bbox = self.__get_city_boundaries(city)
        print(city, bbox)
        osm = pyrosm.OSM(fp, bbox)
        spatial_extent = self.__bbox_to_spatial_extent(bbox)
        buildings = osm.get_buildings()
        self.rasterized_buildings = None

        files = self.__save_defined(spatial_extent, f"{path}/{city}/")

        for file in files:
            rgb = self.__nc_to_rgb(file)
            if self.rasterized_buildings is None:
                raster_height, raster_width, _ = rgb.shape
                self.rasterized_buildings = self.__rasterize_buildings(
                    buildings, bbox, raster_height, raster_width
                )

            cv2.imwrite(
                f"{path}/imgs/{city}_{hash(file)}.jpg",
                cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"{path}/masks/{city}_{hash(file)}.png",
                (self.rasterized_buildings * 255).astype(np.uint8),
            )
            cv2.imwrite(
                f"{path}/test/{city}_{hash(file)}.png",
                cv2.cvtColor(
                    (draw_mask(rgb, self.rasterized_buildings) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                ),
            )

        shutil.rmtree(f"{path}/{city}/")
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

    def __clouds(self, filename, scl_thresh=5, ir_thresh=300):
        scl = scl_thresh < self.__nc_to_scl_band(filename)
        ir = ir_thresh > self.__nc_to_ir_band(filename)
        clouds = np.logical_and(scl, ~ir)
        return clouds

    def __cloudiness(self, clouds):
        return np.linalg.norm(clouds) / len(clouds)

    def __is_cloudy(
        self, filename, scl_thresh=5, ir_thresh=300, decision_threshold=0.1
    ):
        return (
            self.__cloudiness(self.__clouds(filename, scl_thresh, ir_thresh))
            > decision_threshold
        )

    def __bbox_to_spatial_extent(self, bbox) -> dict:
        west, south, east, north = bbox
        return {"west": west, "south": south, "east": east, "north": north}

    def __copernicus_datacube(self, bbox, time_range, bands, max_cloud_cover=5):
        datacube = self.con.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=time_range,
            bands=bands,
            max_cloud_cover=max_cloud_cover,
        )
        return datacube

    # def __rasterize_buildings(
    #     self,
    #     buildings: geopandas.geodataframe.GeoDataFrame,
    #     bbox: list | tuple | np.ndarray,
    #     raster_height=1427,
    #     raster_width=1361,
    # ) -> np.ndarray:
    #     xmin, ymin, xmax, ymax = bbox
    #     xres = (xmax - xmin) / raster_width
    #     yres = (ymax - ymin) / raster_height
    #     return rasterize(
    #         shapes=buildings.geometry,
    #         out_shape=(raster_height, raster_width),
    #         transform=(xres, 0, xmin, 0, -yres, ymax),
    #     )
        
    def __rasterize_buildings(
        self,
        buildings: geopandas.geodataframe.GeoDataFrame,
        bbox: list | tuple | np.ndarray,
        raster_height=1427,
        raster_width=1361,
    ) -> np.ndarray:
        xmin, ymin, xmax, ymax = bbox
        xres = (xmax - xmin) / raster_width
        yres = (ymax - ymin) / raster_height
        
        def rasterize_geometry(geometry):
            if geometry.is_empty:
                return np.zeros((raster_height, raster_width), dtype=np.uint8)
            
            # Transform geometry to raster grid coordinates
            transformed_geometry = shapely.affinity.scale(geometry, xfact=1/xres, yfact=1/yres, origin=(xmin, ymax))
            transformed_geometry = shapely.affinity.translate(transformed_geometry, xoff=-xmin/xres, yoff=ymax/yres)
            
            # Create a bounding box around the transformed geometry
            minx, miny, maxx, maxy = transformed_geometry.bounds
            minx, miny = int(np.floor(minx)), int(np.floor(miny))
            maxx, maxy = int(np.ceil(maxx)), int(np.ceil(maxy))
            
            # Clip bounds to raster dimensions
            minx = max(minx, 0)
            miny = max(miny, 0)
            maxx = min(maxx, raster_width)
            maxy = min(maxy, raster_height)
            
            # Rasterize the geometry within the bounding box
            local_raster = np.zeros((raster_height, raster_width), dtype=np.uint8)
            for row in range(miny, maxy):
                for col in range(minx, maxx):
                    if transformed_geometry.intersects(shapely.geometry.box(col, row, col + 1, row + 1)):
                        local_raster[row, col] = 1
            return local_raster

        with multiprocessing.Pool() as pool:
            results = pool.map(rasterize_geometry, buildings.geometry)
            
        combined_raster = np.maximum.reduce(results)
        return combined_raster
    
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
        total_time_range=["2022-05-01", "2024-05-01"],
        time_ranges_to_save=None,
        bands=["B04", "B03", "B02", "B08", "SCL"],
        max_cloud_cover=20,
        rm_clouds_anyways=True,
    ) -> list[str]:
        if not time_ranges_to_save:
            time_ranges_to_save = pd.date_range(
                total_time_range[0],
                total_time_range[1],
                12,  # 24 means 23 time frames (minus ones that are too cloudy)
            ).date

        files = []
        for time_start, time_end in itertools.pairwise(time_ranges_to_save):
            datacube = self.__copernicus_datacube(
                bbox, [time_start, time_end], bands, max_cloud_cover=max_cloud_cover
            )
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            filename = f"{out_folder}/{time_start}:{time_end}.nc"

            try:
                # self.__download_datacube_mean(datacube, filename)
                datacube.mean_time().download(filename)
            except openeo.rest.OpenEoApiError:
                print(
                    f"No data available in timeframe {time_start}-{time_end} for {max_cloud_cover} % cloud coverage. Skipping..."
                )
                continue

            if rm_clouds_anyways and self.__is_cloudy(filename):
                os.remove(filename)
                continue

            files.append(filename)

        return files

    def __save_result(self, datacube, filename):
        result = datacube.save_result(filename)
        job = result.create_job()
        job.start_and_wait().download_results()

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
            "London",
            "Moscow",
            "Istanbul",
            "Paris",
            "Madrid",
            "Manchester",
            "Barcelona",
            "Copenhagen",
            "Hamburg",
            "Warsaw",
        ]
    )
