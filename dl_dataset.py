import openeo, xarray, geopandas, pyrosm, cv2, itertools, os
import numpy as np
import pandas as pd
from rasterio.features import rasterize

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
        
        for city in self.cities:
            fp = pyrosm.get_data(city)
            osm = pyrosm.OSM(fp)
            bbox = self.__get_bbox(osm.get_boundaries("administrative", name=city))
            buildings = osm.get_buildings()
            self.rasterized_buildings = None

            files = self.__save_defined(bbox, f"{path}/{city}/")
            
            for file in files:
                rgb = self.__nc_to_rgb(file)
                if self.rasterized_buildings == None:
                    raster_width, raster_height = rgb.shape
                    self.rasterized_buildings = self.__rasterize_buildings(buildings, raster_width, raster_height)
                
                cv2.imwrite(f"{path}/imgs/{city}_{hash(file)}.jpg")
                cv2.imwrite(f"{path}/masks/{city}_{hash(file)}.png")
            
            os.rmdir(f"{path}/{city}/")
            
    def __submit_batch_job(self, bbox, time_ranges, bands, max_cloud_cover):
        process_graph = self.__construct_process_graph(bbox, time_ranges, bands, max_cloud_cover)
        job = self.openeo.create_job(process_graph)
        job.start()
        return job
    
    def __construct_process_graph(self, bbox, time_ranges, bands, max_cloud_cover):
        process_graph = {
            "process_id": "batch",
            "arguments": {
                "collections": [],
                "process": {
                    "process_id": "your_custom_process",
                    "arguments": {
                        "bbox": bbox,
                        "time_ranges": time_ranges,
                        "bands": bands,
                        "max_cloud_cover": max_cloud_cover
                    }
                }
            }
        }
        return process_graph
                
    def __get_bbox(self, boundaries):
        min_lon = boundaries["geometry"].bounds["minx"].iloc[0]
        min_lat = boundaries["geometry"].bounds["miny"].iloc[0]
        max_lon = boundaries["geometry"].bounds["maxx"].iloc[0]
        max_lat = boundaries["geometry"].bounds["maxy"].iloc[0]
        return self.__bbox_to_spatial_extent([min_lon, min_lat, max_lon, max_lat])
                
    def __make_dirs(self, path):
        images_dir = f"{path}/imgs/"
        mask_dir = f"{path}/masks/"
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

    def __clouds(self, filename, scl_thresh = 5, ir_thresh = 300):
        scl = scl_thresh < self.nc_to_scl_band(filename)
        ir = ir_thresh > self.nc_to_ir_band(filename)
        clouds = np.logical_and(scl, ~ir)
        return clouds

    def __cloudiness(self, clouds):
        return np.linalg.norm(clouds) / len(clouds)
    
    def __is_cloudy(self, filename, scl_thresh = 5, ir_thresh = 300, decision_threshold=0.1):
        return self.__cloudiness(self.__clouds(filename, scl_thresh, ir_thresh)) > decision_threshold

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
    
    def __rasterize_buildings(
        self,
        buildings: geopandas.geodataframe.GeoDataFrame,
        raster_width=1427,
        raster_height=1361,
    ) -> np.ndarray:
        xmin, ymin, xmax, ymax = buildings.total_bounds
        xres = (xmax - xmin) / raster_width
        yres = (ymax - ymin) / raster_height
        return rasterize(
            shapes=buildings.geometry,
            out_shape=(raster_height, raster_width),
            transform=(xres, 0, xmin, 0, -yres, ymax),
        )

    def __save_defined(
        self,
        bbox,
        out_folder,
        total_time_range=["2022-05-01", "2024-05-01"],
        time_ranges_to_save=None,
        bands=["B04", "B03", "B02", "B08", "SCL"],
        max_cloud_cover=10,
        rm_clouds_anyways=True,
    ) -> list[str]:
        if not time_ranges_to_save:
            time_ranges_to_save = pd.date_range(
                total_time_range[0], total_time_range[1], 12
            ).date

        files = []
        for time_start, time_end in itertools.pairwise(time_ranges_to_save):
            datacube = self.__copernicus_datacube(
                bbox, [time_start, time_end], bands, max_cloud_cover=max_cloud_cover
            )
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            filename = f"{out_folder}/{time_start}:{time_end}.nc"
            datacube.mean_time().download(filename)
            
            if rm_clouds_anyways and self.__is_cloudy(filename):
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

if __name__ == "__main__":
    pipeline = Pipeline()