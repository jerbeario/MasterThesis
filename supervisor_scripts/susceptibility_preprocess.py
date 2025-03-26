import fiona
import geopandas as gpd
import glob
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from rasterio.coords import BoundingBox
from rasterio.windows import Window
# import requests
import richdem as rd
import os
import sys
from scipy.stats import mode
from zipfile import ZipFile
import matplotlib.pyplot as plt
import xarray as xr
import time
import logging
import logging.handlers

def set_logger(verbose=True):
    """
    Set-up the logging system, exit if this fails
    """
    # assign logger file name and output directory
    datelog = time.ctime()
    datelog = datelog.replace(':', '_')
    reference = f'Susc_pre'

    logfilename = ('logger' + os.sep + reference + '_logfile_' + 
                str(datelog.replace(' ', '_')) + '.log')

    # create output directory if not exists
    if not os.path.exists('logger'):
        os.makedirs('logger')

    # create logger and set threshold level, report error if fails
    try:
        logger = logging.getLogger(reference)
        logger.setLevel(logging.DEBUG)
    except IOError:
        sys.exit('IOERROR: Failed to initialize logger with: ' + logfilename)

    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                '%(levelname)s - %(message)s')

    # assign logging handler to report to .log file
    ch = logging.handlers.RotatingFileHandler(logfilename,
                                            maxBytes=10*1024*1024,
                                            backupCount=5)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # assign logging handler to report to terminal
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # start up log message
    logger.info('File logging to ' + logfilename)

    return logger, ch

def download_url(chunk_size=128):
    url = 'https://data.bris.ac.uk/datasets/tar/25wfy0f9ukoge2gs7a5mqpq2j7.zip'
    save_path = 'FABDEM.zip'
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unzip():
    directory = '/projects/0/FWC2/MYRIAD/data/25wfy0f9ukoge2gs7a5mqpq2j7'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if '.zip' in f:
            # loading the temp.zip and creating a zip object
            with ZipFile(f, 'r') as zObject:
            
                # Extracting all the members of the zip 
                # into a specific location.
                zObject.extractall('/projects/0/FWC2/MYRIAD/data/FABDEM')

def mosaic_region(region='Europe'):
    logger.info('In function')
    if region == 'Japan':
        # Define the geographical coordinates of Japan
        min_latitude = 24.0
        max_latitude = 45.0
        min_longitude = 122.0
        max_longitude = 154.0
    elif region == 'Europe':
        logger.info('Select Europe')
        # Define the geographical coordinates of Europe
        min_latitude = 27.0
        max_latitude = 72.0
        min_longitude = -25.0
        max_longitude = 45.0

    # Generate a list of TIFF filenames for Japan's coordinates
    tif_files = []

    for lat in range(int(min_latitude), int(max_latitude) + 1):
        for lon in range(int(min_longitude), int(max_longitude) + 1):
            lat_str = 'N' + str(lat).zfill(2) if lat >= 0 else 'S' + str(-lat).zfill(2)
            lon_str = 'E' + str(lon).zfill(3) if lon >= 0 else 'W' + str(-lon).zfill(3)
            tif_filename = lat_str + lon_str + '_FABDEM_V1-2.tif'
            out_tif_filename = os.path.join(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/Tiles/', tif_filename)
            out_tif_filename = out_tif_filename.replace('.tif', '_300m.tif')
            tif_filename = os.path.join('/projects/0/FWC2/MYRIAD/data/FABDEMV2', tif_filename)
            # Check if the TIFF file exists before adding it to the list
            if os.path.exists(tif_filename):
                logger.info(f'Found tif file: {tif_filename}')
                try:
                    resample_mask_mosaic(tif_filename, out_tif_filename)
                    tif_files.append(out_tif_filename)
                except:
                    print(f'failed: {out_tif_filename}')
                    continue

    # print(tif_files)
    # Initialize an empty list to store opened raster files
    src_files_to_mosaic = []

    # Open each TIFF file and add it to the list
    for tif_file in tif_files:
        src = rasterio.open(tif_file)
        src_files_to_mosaic.append(src)

    print('Opened all files')
    # Merge the opened TIFF files into a single mosaic
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Create an output file for the mosaic
    output_filepath = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/mosaic_{region}.tif'

    # Update the metadata and save the mosaic
    with rasterio.open(output_filepath, 'w', driver='GTiff', 
                    height=mosaic.shape[1], width=mosaic.shape[2], 
                    count=1, dtype=mosaic.dtype, crs=src.crs, 
                    transform=out_transform, nodata=-9999) as dest:
        dest.write(mosaic)

    print(f'Mosaic saved to {output_filepath}')

    # Create a new figure and display the raster data
    plt.figure()
    plt.imshow(mosaic[0], cmap='viridis')
    plt.title('Mosaic')
    plt.colorbar()
    plt.savefig(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/mosaic_{region}.png', dpi=150)

def mask_mosaic(region='Europe'):
    logger.info('mask')
    # Define the path to your mosaic TIFF file and the shapefile
    mosaic_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/mosaic_{region}.tif'
    if region == 'Japan':
        shapefile_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/{region}_full.shp'
    elif region == 'Europe':
        shapefile_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/geogunit_105.shp'

    with fiona.open(shapefile_path, "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]

    with rasterio.open(mosaic_path) as src:
        out_image, out_transform = mask(src, geoms, crop=False, all_touched=True)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    # Create an output filepath for the masked mosaic
    output_masked_mosaic_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/masked_mosaic_{region}.tif'

    with rasterio.open(output_masked_mosaic_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f'Masked mosaic with buffer saved to {output_masked_mosaic_path}')

    plt.figure()
    plt.imshow(out_image[0], cmap='viridis')
    plt.title('Masked Mosaic')
    plt.colorbar()
    plt.savefig(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/masked_mosaic_{region}.png', dpi=150)

def resample_mask_mosaic(input_dem_path, output_dem_path):
    upscale_factor = 1/10

    with rasterio.open(input_dem_path) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )

        profile = dataset.profile.copy()

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    
    profile.update({"height": data.shape[-2],
                    "width": data.shape[-1],
                   "transform": transform})

    with rasterio.open(output_dem_path, "w", **profile) as dataset:
        dataset.write(data)

def dem_calc(region='Europe'):
    logger.info('dem')
    dem = rd.LoadGDAL(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/masked_mosaic_{region}.tif')
    dem_filled = rd.FillDepressions(dem, in_place=False)
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", dem_filled)
    logger.info('slope')
    slope = rd.TerrainAttribute(dem_filled, attrib='slope_riserun')
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/slope_{region}.tif", slope)
    logger.info('aspect')
    aspect = rd.TerrainAttribute(dem_filled, attrib='aspect')
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/aspect_{region}.tif", aspect)
    logger.info('curv')
    curvature = rd.TerrainAttribute(dem_filled, attrib='curvature')
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/curvature_{region}.tif", curvature)
    logger.info('curv prof')
    curv_prof = rd.TerrainAttribute(dem_filled, attrib='profile_curvature')
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/curvature_profile_{region}.tif", curv_prof)
    logger.info('curv plan')
    curv_plan = rd.TerrainAttribute(dem_filled, attrib='planform_curvature')
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/curvature_planform_{region}.tif", curv_plan)
    logger.info('d8')
    accum_d8 = rd.FlowAccumulation(dem_filled, method='D8')
    rd.SaveGDAL(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/accum_d8_{region}.tif", accum_d8)

def hydro_dem(region='Europe'):
    with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", "r") as src:
        elevtn = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        out_meta = src.meta
        profile = src.profile

    flw = pyflwdir.from_dem(
        data=elevtn,
        nodata=src.nodata,
        transform=transform,
        latlon=crs.is_geographic,
    )
    logger.info('basins')
    basins = flw.basins()
    with rasterio.open(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/basins_{region}.tif', 'w', **profile) as dst:
        dst.write(basins, 1)

    logger.info('strahler')
    # first define streams based on an upstream area threshold, here 100 km2
    stream_mask = flw.upstream_area("km2") > 100

    # calculate the stream orders for these streams
    strahler = flw.stream_order(type="strahler", mask=stream_mask)

    with rasterio.open(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/strahler_{region}.tif', 'w', **profile) as dst:
        dst.write(strahler, 1)

    logger.info('streams')
    feat = flw.streams()
    gdf = gpd.GeoDataFrame.from_features(feat, crs=crs)
    gdf.to_file(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/streams_{region}.shp')

    logger.info('accuflux')
    up_data = flw.upstream_area()

    accuflux = flw.accuflux(up_data)

    with rasterio.open(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/accuflux_{region}.tif', 'w', **profile) as dst:
        dst.write(accuflux, 1)

def GFAS():
    with rasterio.open("/projects/0/FWC2/MYRIAD/Susceptibility/Region/filled_masked_mosaic_japan.tif", "r") as src:
        bounds = src.bounds

    # Define the time range (years 2004-2017)
    start_year = '2004-01-01'
    end_year = '2017-12-31'
    year = 2004

    # Create an empty xarray dataset to store the results, specifying the 'year' dimension
    result_data = xr.Dataset(coords={'year': np.arange(int(start_year[:4]), int(end_year[:4]) + 1)})

    # List of NetCDF files
    data_directory = "/projects/0/FWC2/MYRIAD/data/GFAS/"
    grib_files = sorted([os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith('.grib')])

    for grib_file in grib_files[25:39]:
        # Load the original NetCDF file
        ds = xr.open_dataset(grib_file, engine="cfgrib")

        # Subset the data to include only the desired lat and lon
        crop_ds = ds.sel(latitude=slice(int(round(bounds[3], 0)), int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0))))
        print(crop_ds['time'][0].dt.year)

        # Calculate the maximum value for each grid cell over the entire year
        max_crop_ds = crop_ds.max(dim='time')

        # Define the factor by which you want to increase the resolution
        factor = 18

        # Create a new grid with the desired higher resolution along latitude and longitude
        new_lat = np.repeat(max_crop_ds['latitude'], factor)
        new_lon = np.repeat(max_crop_ds['longitude'], factor)

        # Create a new dataset with the higher-resolution grid
        new_data = xr.Dataset(
            {
                'dis24_max_year': (['latitude', 'longitude'],
                                np.tile(max_crop_ds['dis24'], (factor, factor))),
            },
            coords={'latitude': new_lat, 'longitude': new_lon}
        )

        # Append the yearly maximum resampled data to the result dataset
        new_data = new_data.expand_dims({'year': [int(crop_ds['time'][0].dt.year)]})
        new_data.to_netcdf(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/GFAS_{int(crop_ds['time'][0].dt.year)}.nc")

    # Save the result to a new NetCDF file
    ds = xr.open_mfdataset('/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/GFAS_*.nc')
    ds.to_netcdf('/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/GFAS.nc')
    for f in glob.glob('/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/GFAS_*.nc'):
        os.remove(f)

def rasterize_hazards(hazard_condition='multi', hazard_type='ls'):
    # Set up the input and output paths
    csv_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/df_{hazard_condition}_japan.csv'
    existing_raster_path = '/projects/0/FWC2/MYRIAD/Susceptibility/Region/mosaic_japan.tif'

    df = pd.read_csv(csv_path)

    if hazard_condition == 'single':
        df = df[df.code2 == hazard_type]
        output_netcdf_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/SH_occurrence_{hazard_type}.nc'
        output_shape_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/SH_occurrence_{hazard_type}.shp'
    else:
        output_netcdf_path = '/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/MH_occurrence.nc'
        output_shape_path = '/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/MH_occurrence.shp'

    # Convert the "geometry" column to GeoSeries
    df['Geometry'] = gpd.GeoSeries.from_wkt(df['Geometry'])

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df)
    gdf = gdf.set_geometry('Geometry')

    # Assuming you have a 'year' column
    gdf['starttime'] = pd.to_datetime(gdf['starttime'])
    gdf['year'] = gdf['starttime'].dt.year
    gdf['hazard_occurred'] = 1
    aggregated_gdf = gdf.dissolve(by='year')
    aggregated_gdf = aggregated_gdf[['Geometry', 'Unnamed: 0']]
    aggregated_gdf.reset_index(inplace=True)
    aggregated_gdf.to_file(output_shape_path)

    # Open the existing raster to get the extent and resolution
    src = rasterio.open(existing_raster_path)
    extent = src.bounds
    resolution = src.transform[0]

    # Create an empty xarray Dataset for storing rasterized data
    ds = xr.Dataset()
    ds['lon'] = np.arange(extent.left, extent.right, resolution)
    ds['lat'] = np.arange(extent.top, extent.bottom, -resolution)
    ds['time'] = np.arange(2004, 2018)
    ds['time'].attrs['units'] = 'year'

    # Create a data variable (DataArray) filled with zeros
    data_var = xr.DataArray(np.zeros((len(ds['time']), src.height, src.width), dtype=np.int16),  # , dtype=np.int32
                        dims=('time', 'lat', 'lon'),
                        coords={'lon': ds['lon'], 'lat': ds['lat'], 'time': ds['time']})
    data_var.attrs['units'] = 'integer'

    # Iterate through years
    for i, year in enumerate(ds['time']):
        year = year.values
        # Filter the GeoDataFrame for the current year
        polygons = aggregated_gdf[aggregated_gdf['year'] == year]['Geometry']

        if not polygons.empty:
            # Rasterize the polygons to a mask
            masked_hazard = rasterio.features.geometry_mask(polygons, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
            masked_hazard = masked_hazard.astype(np.int16)
        else:
            # If no hazards occurred for this year, fill with zeros
            masked_hazard = np.zeros((src.height, src.width), dtype=np.int16)  # , dtype=np.int32

        # Assign the mask to the DataArray
        data_var[i, :, :] = masked_hazard

        print(f'Rasterized {year} data')

    # Add the DataArray to the xarray Dataset
    ds['hazard_data'] = data_var

    # Save the xarray Dataset to a NetCDF file
    ds.to_netcdf(output_netcdf_path)

    print(f'NetCDF file saved to {output_netcdf_path}')

    # Close the raster file
    src.close()

def rasterize_jpn(region='Europe'):
    # Open the existing raster to get the extent and resolution
    # src = rasterio.open('masked_mosaic_japan.tif')
    src = rasterio.open('Region/masked_mosaic_japan.tif')

    data_list = ['Road']  # 'Flood', 'Tsunami', 'Surge', 'Coastline', 'Rivers', 'Road'

    for data in data_list:
        # shapefile_path = f"Japan/{data}/Japan_{data.lower()}.shp"
        shapefile_path = f"Region/{data}/Japan_{data.lower()}.shp" 
        # out_path = f'{data}_Japan.tif'
        out_path = f'Region/{data}_Japan.tif'
        if data == 'Flood':
            shapefile_path = shapefile_path.replace('.shp', '.gpkg')
        elif data == 'Road':
            shapefile_path = f"Region/small_roads_japan.gpkg" 
    
        gdf = gpd.read_file(shapefile_path)
        gdf['hazard_occurred'] = 1
        if data == 'tsunami':
            gdf.loc[~gdf.geometry.is_valid, 'geometry'] = gdf.loc[~gdf.geometry.is_valid, 'geometry'].buffer(0)
            gdf.loc[gdf.geometry.is_empty, 'geometry'] = gdf.loc[gdf.geometry.is_empty, 'geometry'].buffer(0.000001)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.000001)
            gdf = gdf.drop_duplicates(subset='geometry')
        aggregated_gdf = gdf.dissolve(by='hazard_occurred')
        aggregated_gdf = aggregated_gdf.rename(columns={'geometry': 'Geometry'})
        aggregated_gdf = aggregated_gdf[['Geometry']]
        aggregated_gdf.reset_index(inplace=True)

        polygons = aggregated_gdf['Geometry']

        # Rasterize the polygons to a mask
        masked_hazard = rasterio.features.geometry_mask(polygons, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
        masked_hazard = masked_hazard.astype(np.int16)
        # masked_hazard[masked_dem == -9999] = np.nan

        # Open the existing GeoTIFF file for writing, using the updated profile and CRS
        with rasterio.open(out_path, 'w', **src.profile) as dst:
            dst.write(masked_hazard, 1)  # Write your array to the existing GeoTIFF file

def rasterize_eu():
    # Open the existing raster to get the extent and resolution
    # src = rasterio.open('masked_mosaic_japan.tif')
    src = rasterio.open('Region/Europe/masked_mosaic_Europe.tif')

    data_list = ['streams', 'coastline']

    for data in data_list:
        logger.info(data)
        # shapefile_path = f"Japan/{data}/Japan_{data.lower()}.shp"
        if data == 'streams':
            shapefile_path = f"Region/Europe/streams_Europe.shp"
        elif data == 'coastline':
            shapefile_path = f"Region/Europe/ne_10m_coastline.shp"
        # out_path = f'{data}_Japan.tif'
        out_path = f'Region/Europe/{data}_Europe.tif'
    
        gdf = gpd.read_file(shapefile_path)
        logger.info(gdf.columns)
        gdf['hazard_occurred'] = 1
        aggregated_gdf = gdf.dissolve(by='hazard_occurred')
        aggregated_gdf = aggregated_gdf.rename(columns={'geometry': 'Geometry'})
        aggregated_gdf = aggregated_gdf[['Geometry']]
        aggregated_gdf.reset_index(inplace=True)

        polygons = aggregated_gdf['Geometry']

        # Rasterize the polygons to a mask
        masked_hazard = rasterio.features.geometry_mask(polygons, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
        masked_hazard = masked_hazard.astype(np.int16)
        # masked_hazard[masked_dem == -9999] = np.nan

        # Open the existing GeoTIFF file for writing, using the updated profile and CRS
        with rasterio.open(out_path, 'w', **src.profile) as dst:
            dst.write(masked_hazard, 1)  # Write your array to the existing GeoTIFF file

def rasterize_world():
    src = rasterio.open('Region/World/GEM_300m.tif')

    data_list = ['streams', 'coastline']

    for data in data_list:
        logger.info(data)
        
        if data == 'streams':
            shapefile_path = "Region/World/HydroRIVERS_v10.gdb"
            
            # # List available layers in the GDB
            # layers = fiona.listlayers(gdb_path)
            # print(f"Layers in {gdb_path}: {layers}")  # Debugging: check available layers
            
            # # Select the correct layer (replace 'StreamsLayer' with actual layer name)
            # shapefile_path = f"{gdb_path}|layer=HydroRIVERS_v10"  # Change 'StreamsLayer' to the actual name
            
        elif data == 'coastline':
            shapefile_path = "Region/Europe/ne_10m_coastline.shp"
        
        out_path = f'Region/World/{data}_World.tif'

        # Load the data
        gdf = gpd.read_file(shapefile_path)
        logger.info(gdf.columns)
        gdf['hazard_occurred'] = 1
        aggregated_gdf = gdf.dissolve(by='hazard_occurred')
        aggregated_gdf = aggregated_gdf.rename(columns={'geometry': 'Geometry'})
        aggregated_gdf = aggregated_gdf[['Geometry']]
        aggregated_gdf.reset_index(inplace=True)

        polygons = aggregated_gdf['Geometry']

        # Rasterize
        masked_hazard = rasterio.features.geometry_mask(
            polygons, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True
        )
        masked_hazard = masked_hazard.astype(np.int16)

        # Save to TIFF
        with rasterio.open(out_path, 'w', **src.profile) as dst:
            dst.write(masked_hazard, 1)

def rasterize_ldm():
    existing_raster_path = 'mosaic_japan.tif'
    shapefile_path = "Japan_ls/landslides.shp" 
    gdf = gpd.read_file(shapefile_path)

    # Assuming you have a 'year' column
    gdf['hazard_occurred'] = 1
    gdf['hazard_occurred_set'] = np.random.choice([1, 2], size=len(gdf), p=[0.7, 0.3])

    # full map
    aggregated_gdf = gdf.dissolve(by='hazard_occurred')
    aggregated_gdf = aggregated_gdf.rename(columns={'geometry': 'Geometry'})
    aggregated_gdf = aggregated_gdf[['Geometry', 'year']]
    aggregated_gdf.reset_index(inplace=True)
    
    # train validation
    aggregated_gdf_set = gdf.dissolve(by='hazard_occurred_set')
    aggregated_gdf_set = aggregated_gdf_set.rename(columns={'geometry': 'Geometry'})
    aggregated_gdf_set = aggregated_gdf_set[['Geometry', 'year']]
    aggregated_gdf_set.reset_index(inplace=True)
    train_gdf = aggregated_gdf_set[aggregated_gdf_set['hazard_occurred_set'] == 1]
    val_gdf = aggregated_gdf_set[aggregated_gdf_set['hazard_occurred_set'] == 2]

    # Open the existing raster to get the extent and resolution
    src = rasterio.open(existing_raster_path)

    polygons = aggregated_gdf[aggregated_gdf['year'] == '2013']['Geometry']
    poly_train = train_gdf[train_gdf['year'] == '2013']['Geometry']
    poly_val = val_gdf[val_gdf['year'] == '2013']['Geometry']

    # Rasterize the polygons to a mask
    masked_hazard = rasterio.features.geometry_mask(polygons, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
    masked_hazard = masked_hazard.astype(np.int16)

    # Open the existing GeoTIFF file for writing, using the updated profile and CRS
    with rasterio.open('LDM_Japan.tif', 'w', **src.profile) as dst:
        dst.write(masked_hazard, 1)  # Write your array to the existing GeoTIFF file
    
    # Rasterize the polygons to a mask
    masked_hazard = rasterio.features.geometry_mask(poly_train, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
    masked_hazard = masked_hazard.astype(np.int16)

    # Open the existing GeoTIFF file for writing, using the updated profile and CRS
    with rasterio.open('LDM_Japan_train.tif', 'w', **src.profile) as dst:
        dst.write(masked_hazard, 1)  # Write your array to the existing GeoTIFF file
    
    # Rasterize the polygons to a mask
    masked_hazard = rasterio.features.geometry_mask(poly_val, transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
    masked_hazard = masked_hazard.astype(np.int16)

    # Open the existing GeoTIFF file for writing, using the updated profile and CRS
    with rasterio.open('LDM_Japan_val.tif', 'w', **src.profile) as dst:
        dst.write(masked_hazard, 1)  # Write your array to the existing GeoTIFF file

    # Close the raster file
    src.close()

def GLIM_1km():
    with rasterio.open("mosaic_japan.tif", "r") as src:
        bounds = src.bounds
        height = src.shape[0]
        width = src.shape[1]
        profile = src.profile  # Get the metadata/profile of the existing raster

        with rasterio.open("glim_wgs84_0point5deg.txt.asc", "r") as src2:
            raster = src2.read()

        left = int(round(bounds.left))
        bottom = int(round(bounds.bottom))
        right = int(round(bounds.right))
        top = int(round(bounds.top))
        x_start = int((left + 180) / 0.5)
        x_end = int((right + 180) / 0.5)
        y_start = int((90 - top) / 0.5)
        y_end = int((90 - bottom) / 0.5)

        # Slice the array to extract the specified window
        raster_win = raster[0][y_start:y_end, x_start:x_end]

        # Increase resolution by factor
        factor = int(width / np.shape(raster_win)[1])

        # Use np.kron to increase the resolution
        raster_win_inc = np.kron(raster_win, np.ones((factor, factor), dtype=raster_win.dtype))

        # Replace the data in the profile with your 2D NumPy array
        profile['count'] = 1

        # Open the existing GeoTIFF file for writing, using the updated profile and CRS
        with rasterio.open('GLIM_Japan', 'w', **profile) as dst:
            dst.write(raster_win_inc, 1)  # Write your array to the existing GeoTIFF file

    plt.figure()
    plt.imshow(raster[0], cmap='viridis')
    plt.title('World')
    plt.colorbar()
    plt.savefig('geology_world.png', dpi=150)

    plt.figure()
    plt.imshow(raster_win_inc, cmap='viridis')
    plt.title('World Japan')
    plt.colorbar()
    plt.savefig('geology_japan.png', dpi=150)

def GLIM_300m():
    shapefile_path = '/projects/0/FWC2/MYRIAD/data/GLIM/GLIM.shp'
    out_tif = '/projects/0/FWC2/MYRIAD/data/GLIM/GLIM_300m.tif'

    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    print("Shapefile loaded")

    # Get unique geological layer values and assign integers to them
    unique_geology_values = gdf['xx'].unique()
    geology_categories = {value: index + 1 for index, value in enumerate(unique_geology_values)}
    unique_strings = gdf['xx'].unique()
    string_to_value = {string: value for value, string in enumerate(unique_strings, start=1)}
    gdf['xx_val'] = gdf['xx'].map(string_to_value)
    print(geology_categories)

    # Define the bounding box of the extent (xmin, ymin, xmax, ymax) in degrees.
    xmin, ymin, xmax, ymax = -180, -90, 180, 90

    # Define the desired resolution in degrees (approximately 300 meters).
    resolution = 1 / 360  # 1 cell per degree = 360 cells per degree
    # rasterize_shape(shapefile_path, out_tif)

    # Calculate the number of rows and columns based on the extent and resolution.
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)

    # Define the transformation matrix to convert between pixel and map coordinates.
    transform = from_origin(xmin, ymax, resolution, resolution)

    # Create an empty raster as a NumPy array filled with zeros
    empty_raster = np.zeros((height, width), dtype=np.uint16)
    # Write the empty raster to a GeoTIFF file
    with rasterio.open(
        out_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint16,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(empty_raster, 1)

    # Open the empty raster file for further processing
    with rasterio.open(out_tif, 'r+') as dst:
        # Rasterize the geological layers and assign 'xx' attribute values directly to the raster
        shapes = [(geom, int(value)) for geom, value in zip(gdf.geometry, gdf['xx_val'])]  # Extract geometries and attribute values
        dst_arr = dst.read(1)  # Read the empty raster
        rasterio.features.rasterize(
            shapes=shapes,
            out=dst_arr,  # Assign values to the empty raster
            transform=transform,
            fill=0,
        )
        dst.write(dst_arr, 1) 

def GEM_300m():
    shapefile_path = 'gem-global-active-faults-master/shapefile/gem_active_faults_harmonized.shp'
    out_tif = 'GEM_300m.tif'

    # Read shapefile
    gdf = gpd.read_file(shapefile_path)

    # Define the bounding box of the extent (xmin, ymin, xmax, ymax) in degrees.
    xmin, ymin, xmax, ymax = -180, -90, 180, 90

    # Define the desired resolution in degrees (approximately 300 meters).
    resolution = 1 / 360  # 1 cell per degree = 360 cells per degree

    # Calculate the number of rows and columns based on the extent and resolution.
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)

    # Define the transformation matrix to convert between pixel and map coordinates.
    transform = from_origin(xmin, ymax, resolution, resolution)

    # Create a new raster with the desired properties.
    with rasterio.open(
        out_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,  # Single band
        dtype=np.uint8,  # Unsigned integer data type
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        # Rasterize the shapefile into the new raster.
        shapes = [(geom, 1) for geom in gdf.geometry]
        mask = geometry_mask(shapes, out_shape=(height, width), transform=transform, invert=True, all_touched=True)
        dst.write(mask, 1)

def clip_to_region(region_tif, input_tif, output_tif):
    print('Clip to region')
    # Open region
    with rasterio.open(region_tif) as src_region:
        region_extent = src_region.bounds  # Get the extent
        region_resolution = src_region.res  # Get the resolution
        region_extent = BoundingBox(*map(round, region_extent))

    # Open GEM world
    with rasterio.open(input_tif) as src_world:
        # Calculate the window based on the region extent
        window = src_world.window(*region_extent)

        # Read the data within the window
        world_data = src_world.read(window=window)

        # Create a new profile for the output GeoTIFF
        profile = src_world.profile

    # Update the profile to match the new extent
    profile.update(
        width=window.width,
        height=window.height,
        transform=src_world.window_transform(window)
    )

    # write to
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(world_data)

def distance_to(input_tif):
    logger.info(input_tif)
    from scipy.ndimage import distance_transform_edt
    # from geopy.distance import great_circle

    # Open the input GeoTIFF file containing fault line information and read it
    with rasterio.open(input_tif) as src:
        lines = src.read(1)
        transform = src.transform

    # Calculate the distance transform for the fault lines
    # Invert the fault lines so that fault lines are 0 and background is 1
    lines[lines > 1] = 1
    binary_lines = 1 - lines

    # Calculate the Euclidean distance transform
    distance_map = distance_transform_edt(binary_lines)

    # Save the calculated distance map to a new GeoTIFF file
    output_tif = input_tif.replace('.tif', '_euclidean.tif')
    with rasterio.open(output_tif, 'w', driver='GTiff', width=src.width, height=src.height, count=1,
                    dtype=distance_map.dtype, crs=src.crs, transform=transform) as dst:
        dst.write(distance_map, 1)
    print('Euclidean done')

    # # Create an empty array to store the distances
    # distance_map = np.zeros_like(fault_lines, dtype=np.float32)

    # # Iterate through each cell
    # for row in range(fault_lines.shape[0]):
    #     for col in range(fault_lines.shape[1]):
    #         if fault_lines[row, col] == 0:
    #             # If the cell is part of a fault line, skip it
    #             continue
            
    #         # Calculate the latitude and longitude coordinates of the cell's center
    #         lon, lat = rasterio.transform.xy(transform, row + 0.5, col + 0.5)

    #         # Iterate through each cell in the fault lines layer
    #         min_distance = float('inf')
    #         for fault_row in range(fault_lines.shape[0]):
    #             for fault_col in range(fault_lines.shape[1]):
    #                 if fault_lines[fault_row, fault_col] == 1:
    #                     fault_lon, fault_lat = rasterio.transform.xy(transform, fault_row + 0.5, fault_col + 0.5)
    #                     # Calculate the Haversine distance between the two points
    #                     distance = great_circle((lat, lon), (fault_lat, fault_lon)).kilometers
    #                     min_distance = min(min_distance, distance)
            
    #         # Assign the minimum Haversine distance to the current cell in the distance map
    #         distance_map[row, col] = min_distance

    # # Save the calculated distance map to a new GeoTIFF file
    # output_tif = input_tif.replace('.tif', '_haversine.tif')
    # with rasterio.open(output_tif, 'w', driver='GTiff', width=src.width, height=src.height, count=1,
    #                 dtype=distance_map.dtype, crs=src.crs, transform=transform) as dst:
    #     dst.write(distance_map, 1)
    # print('haversine done')

def NDVI(region='Europe'):
    logger.info('NDVI')
    with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", "r") as src:
        bounds = src.bounds
        crs = src.crs
        dst_transform = src.transform

    # Define the time range (years 2004-2017)
    year_list = np.arange(2004, 2018)

    # Create an empty xarray dataset to store the results, specifying the 'year' dimension
    result_data = xr.Dataset(coords={'year': year_list})

    for year in year_list:
        logger.info(year)
        # year = 1999
        data_directory = '/projects/0/FWC2/MYRIAD/data/NDVI_Sentinel/'
        nc_files = sorted([os.path.join(data_directory, file) for file in os.listdir(data_directory) if f'_{year}' in file])

        ds = xr.open_mfdataset(nc_files, engine="netcdf4") 
        crop_ds = ds.sel(lat=slice(int(round(bounds[3], 0)), int(round(bounds[1], 0))), lon=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0))))
        crop_ds_mean = crop_ds.mean(dim="time", skipna=True)

        # Define the factor by which you want to increase the resolution
        factor = 3

        # Create a new grid with the desired higher resolution along latitude and longitude
        # Calculate the new number of latitude points
        new_num_lats = (len(crop_ds_mean['lat'].values) - 1) * factor + 1
        new_num_lons = (len(crop_ds_mean['lon'].values) - 1) * factor + 1

        # Create a new array with the updated latitude values
        new_lat = np.linspace(crop_ds_mean['lat'].values[0], crop_ds_mean['lat'].values[-1], new_num_lats)
        new_lon = np.linspace(crop_ds_mean['lon'].values[0], crop_ds_mean['lon'].values[-1], new_num_lons)

        # Use np.kron to increase the resolution
        crop_ds_mean_inc = np.kron(crop_ds_mean['NDVI'], np.ones((factor, factor)))

        y_resolution = new_lat[0] - new_lat[1]
        x_resolution = new_lon[1] - new_lon[0]
        x_origin = new_lon[0]
        y_origin = new_lat[0]

        # Create the geospatial transform
        transform = from_origin(x_origin, y_origin, x_resolution, y_resolution)

        # Create a GeoTIFF file for writing
        output_tif_file = f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/NDVI_{year}.tif"
        with rasterio.open(output_tif_file, 'w', driver='GTiff', height=crop_ds_mean_inc.shape[0],
                        width=crop_ds_mean_inc.shape[1], count=1, dtype=crop_ds_mean_inc.dtype,
                        transform=transform, crs=crs) as dst:
            # Write the data to the GeoTIFF file
            dst.write(crop_ds_mean_inc, 1)
        
        # Open the existing GeoTIFF to get its resolution and transform
        with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", "r") as existing_src:
            existing_transform = existing_src.transform
            existing_resolution = existing_transform.a

            # Reproject the data to match the resolution of the existing GeoTIFF
            reprojected_data = np.empty_like(existing_src.read(1))
            reproject(
                crop_ds_mean_inc,
                reprojected_data,
                src_transform=transform,
                src_crs=crs,  # Assuming the original data is in WGS 84
                dst_transform=existing_transform,
                dst_crs=existing_src.crs,
                resampling=Resampling.cubic,
            )

        # Create a GeoTIFF file for writing
        with rasterio.open(output_tif_file.replace('.tif', '_reproj.tif'), 'w', driver='GTiff',
                        height=reprojected_data.shape[0], width=reprojected_data.shape[1], count=1,
                        dtype=reprojected_data.dtype, crs=crs, transform=existing_transform) as dst:
            # Write the reprojected data to the GeoTIFF file
            dst.write(reprojected_data, 1)

        # Create a new array with the updated latitude values
        new_lon_reproj = np.arange(int(round(bounds[0], 0)), int(round(bounds[2], 0)), 1/360)
        new_lat_reproj = np.arange(int(round(bounds[3], 0)), int(round(bounds[1], 0)), -1/360)

        # Create a new dataset with the higher-resolution grid
        new_data = xr.Dataset(
            {
                'mean_NDVI': (['lat', 'lon'],
                                reprojected_data),
            },
            coords={'lat': new_lat_reproj, 'lon': new_lon_reproj}
        )

        # Append the yearly maximum resampled data to the result dataset
        new_data = new_data.expand_dims({'year': [int(year)]})
        new_data.to_netcdf(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/NDVI_{year}.nc", engine="netcdf4")


    # Save the result to a new NetCDF file
    ds = xr.open_mfdataset(f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/NDVI_*.nc', engine="netcdf4")
    ds.to_netcdf(f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/NDVI.nc')
    for f in glob.glob(f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/NDVI_*'):
        os.remove(f)

def landcover(region='Europe'):
    logger.info('Landcover')
    with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", "r") as src:
        bounds = src.bounds

    data_directory = '/projects/0/FWC2/MYRIAD/data/landcover/'
    nc_files = sorted([os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith('.nc')])
    ds = xr.open_mfdataset(nc_files, engine="netcdf4")
    crop_ds = ds.sel(lat=slice(int(round(bounds[3], 0)), int(round(bounds[1], 0))), lon=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0))))
    crop_ds_timeframe = crop_ds.sel(time=slice("2004-01-01", "2017-01-01"))

    new_data = xr.Dataset(
        {
            'landcover': (['time', 'lat', 'lon'], crop_ds_timeframe['lccs_class'].values),
        },
        coords={'time': np.arange(2004, 2018), 'lat': crop_ds_timeframe['lat'].values, 'lon': crop_ds_timeframe['lon'].values}
    )
    new_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/Landcover.nc')

def PGA(region='Europe'):
    # Specify the path to your ASCII file
    ascii_file_path = 'Region/Japan/gdpga/gdpga.asc'

    # Open the ASCII file and read metadata
    with open(ascii_file_path, 'r') as ascii_file:
        # Read metadata
        ncols = int(ascii_file.readline().split()[1])
        nrows = int(ascii_file.readline().split()[1])
        xllcorner = float(ascii_file.readline().split()[1])
        yllcorner = float(ascii_file.readline().split()[1])
        cellsize = float(ascii_file.readline().split()[1])
        nodata_value = float(ascii_file.readline().split()[1])

        # Read raster data
        raster_data = np.genfromtxt(ascii_file, dtype=float, missing_values=nodata_value, usemask=True)

    # Create a rasterio dataset
    transform = from_origin(xllcorner, yllcorner + nrows * cellsize, cellsize, cellsize)
    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': raster_data.dtype,
        'width': ncols,
        'height': nrows,
        'crs': 'EPSG:4326',  # WGS84 projection
        'transform': transform,
        'nodata': nodata_value,
    }

    with rasterio.open('Region/PGA.tif', 'w', **metadata) as dst:
        dst.write(raster_data, 1)

    # Open the filled_dem_japan.tif raster to get its bounding box
    with rasterio.open(f"Region/{region}/masked_mosaic_{region}.tif", "r") as src_out:
        out_bounds = src_out.bounds
        out_res = src_out.res
        out_nodata = src_out.nodata
        dem = src_out.read(1)

    # Open the PGA.tif raster to get its bounding box and data
    with rasterio.open("Region/PGA.tif", "r") as src_in:
        # Get metadata from the original raster
        in_bounds = src_in.bounds
        in_res = src_in.res
        pga = src_in.read(1, window=src_in.window(*out_bounds))
        in_nodata = src_in.nodata
        crs = src_in.crs
        transform = src_in.transform
        transform_windowed = src_in.window_transform(src_in.window(*out_bounds))
        dst_profile = src_in.profile.copy()
    
    # Define the factor by which to increase the resolution
    incr_fact = 15

    # Calculate the new resolution
    out_res = (in_res[0] / incr_fact, in_res[1] / incr_fact)

    # Calculate the new dimensions based on the target resolution
    out_width = int(pga.shape[1] * (in_res[0] / out_res[0]))
    out_height = int(pga.shape[0] * (in_res[1] / out_res[1]))
    print(out_width)
    print(out_height)

    # Create a destination dataset for the resampled data
    dst_profile.update({
        'width': out_width,
        'height': out_height,
        'transform': from_origin(out_bounds.left, out_bounds.top, out_res[0], out_res[1]),
        'crs': crs
    })

    pga[pga == in_nodata] = 0
    pga = pga / 10

    # Perform the resampling using cubic method
    with rasterio.open(f"Region/{region}/PGA_{region}_resampled.tif", "w", **dst_profile) as dst:
        reproject(
            source=pga,
            destination=rasterio.band(dst, 1),
            src_transform=transform_windowed,
            src_crs=crs,
            dst_transform=dst.transform,
            dst_crs=crs,
            resampling=Resampling.cubic
        )
    # # Repeat each value in the array
    # pga_enlarged = np.repeat(np.repeat(pga, incr_fact, axis=0), incr_fact, axis=1)

    # # Create a new transform for the enlarged raster
    # transform = from_origin(out_bounds.left, out_bounds.top, in_res[0] / incr_fact, in_res[1] / incr_fact)

    # # Write the enlarged PGA raster to a new file
    # metadata = {
    #     'driver': 'GTiff',
    #     'count': 1,
    #     'dtype': pga_enlarged.dtype,
    #     'width': pga_enlarged.shape[1],
    #     'height': pga_enlarged.shape[0],
    #     'crs': crs,
    #     'transform': transform,
    #     'nodata': 0,
    # }

    # # Write the enlarged PGA raster to a new file
    # metadata = {
    #     'driver': 'GTiff',
    #     'count': 1,
    #     'dtype': pga_enlarged.dtype,
    #     'width': pga_enlarged.shape[1],
    #     'height': pga_enlarged.shape[0],
    #     'crs': crs,
    #     'transform': transform,
    #     'nodata': out_nodata,
    # }

    # pga_enlarged[dem == out_nodata] = in_nodata
    # pga_enlarged[pga_enlarged == in_nodata] = 0
    # pga_enlarged = pga_enlarged / 10

    # with rasterio.open('PGA_japan_resampled.tif', 'w', **metadata) as dst:
    #     dst.write(pga_enlarged, 1)

def precipitation(region='Europe'):
    logger.info('Precipitation')
    dates = [f'01-{str(month).zfill(2)}-{year}' for year in range(1950, 2022) for month in range(1, 13)]

    with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", "r") as src:
        bounds = src.bounds

    data_directory = '/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation'
    nc_files = sorted([os.path.join(data_directory, file) for file in os.listdir(data_directory) if 'ERA5-Land_Monthly_average_total_precipitation' in file and file.endswith('.grib')])
    ds = xr.open_mfdataset(nc_files, engine="cfgrib")
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180))
    ds = ds.sortby("longitude")
    crop_ds = ds.sel(latitude=slice(int(round(bounds[3], 0)), int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0))))

    new_data = xr.Dataset(
        {
            'precipitation': (['time', 'lat', 'lon'], crop_ds['tp'].values),
        },
        coords={'time': dates, 'lat': crop_ds['latitude'].values, 'lon': crop_ds['longitude'].values}
    )
    new_data.to_netcdf(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/precipitation.nc")

    src = rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif", "r")
    bounds = src.bounds

    ds = xr.open_dataset(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/precipitation.nc")

    # Convert the time coordinate to datetime
    ds['time'] = pd.to_datetime(ds['time'], format='%d-%m-%Y')

    # Sort the time dimension
    precipitation_data = ds['precipitation']
    precipitation_data = precipitation_data.sortby('time')

    # Filter the data to keep only the range between 2004 and 2017
    precipitation_data = precipitation_data.sel(time=slice('2004-01-01', '2017-12-31'))

    # Resample the data to yearly frequency and calculate the sum
    yearly_sum = precipitation_data.resample(time='1Y').sum(dim='time', keep_attrs=True)

    # Define the factor by which you want to increase the resolution
    factor = 36

    # Use np.kron to increase the resolution
    yearly_sum_inc = np.kron(yearly_sum, np.ones((1, factor, factor)))

    # Create a new array with the updated latitude values
    new_lon = np.arange(int(round(bounds[0], 0)), int(round(bounds[2], 0)), 1/360)
    new_lat = np.arange(int(round(bounds[3], 0)), int(round(bounds[1], 0)), -1/360)

    new_data = xr.Dataset({'precipitation': (['time', 'lat', 'lon'], yearly_sum_inc),}, coords={'time': yearly_sum['time'], 'lat': new_lat, 'lon': new_lon})
    new_data.to_netcdf(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/precipitation_{region}.nc")

def prepare_data_nn(region='Europe'):
    def mask_clip_npsave_fig(array_path, name, d3=True):
        if d3 == True:
            ds = xr.open_dataset(array_path)
            data = ds.to_array().values[0]
            # Apply the expanded mask to the temperature array
            masked_data = np.where(expanded_mask, data, np.nan)
            if region == 'Japan':
                masked_data = masked_data[:, 0:5500, 2300:8800]
            plt.figure()
            plt.imshow(masked_data[0], cmap='viridis')
        else:
            with rasterio.open(array_path, "r") as src:
                data = src.read(1)

            masked_data = np.where(mask_arr, data, np.nan)
            masked_data[masked_data < 0] = 0
            if region == 'Japan':
                masked_data = masked_data[0:5500,2300:8800]
            if name == 'stra':
                masked_data[masked_data > 100] = 100
            plt.figure()
            plt.imshow(masked_data, cmap='viridis')
        plt.title(name)
        plt.colorbar()
        plt.savefig(f'Input/{region}/masked_{name}_{region}.png', dpi=150)
        np.save(f'Input/{region}/npy_arrays/masked_{name}_{region}.npy', masked_data)

    dir_path2 = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/'
    dir_path1 = f'Region/{region}/'
    # dir_path = 'Japan/'

    mask_path = f'{dir_path1}masked_mosaic_{region}.tif'
    curv_path = f'{dir_path1}curvature_{region}.tif'
    plan_path = f'{dir_path1}curvature_planform_{region}.tif'
    prof_path = f'{dir_path1}curvature_profile_{region}.tif'
    slope_path = f'{dir_path1}slope_{region}.tif'
    asp_path = f'{dir_path1}aspect_{region}.tif'
    ls_path = f'{dir_path1}SH_occurrence_ls.nc'
    stra_path = f'{dir_path1}strahler_{region}_euclidean.tif'
    gem_path = f'{dir_path1}GEM_300m_{region}_euclidean.tif'
    glim_path = f'{dir_path1}GLIM_300m_{region}.tif'
    ndvi_path = f'{dir_path2}NDVI.nc'
    lc_path = f'{dir_path2}Landcover.nc'
    ldm_path = f'{dir_path1}LDM_Japan.tif'
    ldm_train_path = f'{dir_path1}LDM_Japan_train.tif'
    ldm_val_path = f'{dir_path1}LDM_Japan_val.tif'
    prec_path = f'{dir_path1}precipitation_{region}.nc'

    dtriv_path = f'{dir_path1}streams_{region}_euclidean.tif'
    dtcoa_path = f'{dir_path1}coastline_{region}_euclidean.tif'
    surge_path = f'{dir_path1}Surge_Japan.tif'
    tsunami_path = f'{dir_path1}Tsunami_Japan.tif'
    accu_path = f'{dir_path1}accuflux_{region}.tif'
    pga_path = f'{dir_path1}PGA_{region}_resampled.tif'
    soil_path = f'{dir_path1}HWSD2_WRB2_LAYER_{region}_300m.tif'
    road_path = f'{dir_path1}Road_{region}.tif'
    flood_path = f'{dir_path1}Flood_{region}.tif'
    volcano_path = f'{dir_path1}GVP_Holocene_count_{region}.tif'
    mh_path = f'{dir_path2}myriad_hes_{region}.tif'
    jshis_path = f'{dir_path1}J-SHIS_T50_P10_SV_reproj.tif'

    # ewi_path = f'{dir_path}windy_days_{region}.tif'
    # hwi_path = f'{dir_path}heatwave_index_window_{region}.tif'
    # di_path = f'{dir_path}drought_index_{region}.tif'
    fwi_path = f'{dir_path2}fire_weather_index_{region}_resampled.tif'
    fwi_path = f'{dir_path2}fire_weather_index_mean_{region}_resampled.tif'
    wf_path = f'{dir_path1}Wildfires_{region}.tif'
    # prec_daily_path = f'{dir_path}mean_yearly_precipitation_{region}.tif'

    ewi_path = f'{dir_path2}windy_days_{region}_resampled.tif'
    hwi_path = f'{dir_path2}heatwave_index_window_{region}_resampled.tif'
    di_path = f'{dir_path2}drought_index_{region}_resampled.tif'
    prec_daily_path = f'{dir_path2}mean_yearly_precipitation_{region}_resampled.tif'
    temp_daily_path = f'{dir_path2}mean_yearly_temperature_{region}_resampled.tif'
    ws_daily_path = f'{dir_path2}mean_wind_speed_{region}_resampled.tif'
    wd_daily_path = f'{dir_path2}mode_wind_direction_{region}_resampled.tif'

    with rasterio.open(mask_path, "r") as src:
        masked_elevation = src.read(1)

    # Create a mask based on the condition where elevation > 0
    mask_arr = masked_elevation != -9999

    # Expand the 2D mask to match the dimensions of the 3D temperature array
    expanded_mask = np.repeat(mask_arr[np.newaxis, :, :], 14, axis=0)

    # mask_clip_npsave_fig(curv_path, 'curvature', d3=False)
    # mask_clip_npsave_fig(mask_path, 'elevation', d3=False)
    # mask_clip_npsave_fig(asp_path, 'aspect', d3=False)
    # mask_clip_npsave_fig(gem_path, 'GEM', d3=False)
    # mask_clip_npsave_fig(glim_path, 'GLIM', d3=False)
    # mask_clip_npsave_fig(stra_path, 'strahler', d3=False)
    # mask_clip_npsave_fig(slope_path, 'slope', d3=False)
    # mask_clip_npsave_fig(plan_path, 'curvature_planform', d3=False)
    # mask_clip_npsave_fig(prof_path, 'curvature_profile', d3=False)
    # mask_clip_npsave_fig(ldm_path, 'ldm', d3=False)
    # mask_clip_npsave_fig(ldm_train_path, 'ldm_train', d3=False)
    # mask_clip_npsave_fig(ldm_val_path, 'ldm_val', d3=False)

    # mask_clip_npsave_fig(ls_path, 'ls', d3=True)
    # mask_clip_npsave_fig(ndvi_path, 'NDVI', d3=True)
    # mask_clip_npsave_fig(lc_path, 'landcover', d3=True)
    # mask_clip_npsave_fig(prec_path, 'precipitation', d3=True)

    # mask_clip_npsave_fig(prec_path, 'prec', d3=True)

    # dir_path = 'Input/Japan/'
    # mask_clip_npsave_fig(dtriv_path, 'rivers', d3=False)
    # mask_clip_npsave_fig(dtcoa_path, 'coastlines', d3=False)
    # mask_clip_npsave_fig(surge_path, 'surge', d3=False)
    # mask_clip_npsave_fig(tsunami_path, 'tsunami', d3=False)
    # mask_clip_npsave_fig(accu_path, 'accuflux', d3=False)
    # mask_clip_npsave_fig(pga_path, 'pga', d3=False)
    # mask_clip_npsave_fig(jshis_path, 'jshis', d3=False)
    # mask_clip_npsave_fig(soil_path, 'HWSD', d3=False)
    # mask_clip_npsave_fig(road_path, 'road', d3=False)
    # mask_clip_npsave_fig(flood_path, 'flood', d3=False)
    # mask_clip_npsave_fig(volcano_path, 'volcano', d3=False)
    # mask_clip_npsave_fig(ewi_path, 'extreme_wind', d3=False)
    # mask_clip_npsave_fig(di_path, 'drought', d3=False)
    # mask_clip_npsave_fig(hwi_path, 'heatwave', d3=False)
    # mask_clip_npsave_fig(fwi_path, 'fire_weather', d3=False)
    # mask_clip_npsave_fig(prec_daily_path, 'precipitation_daily', d3=False)
    # mask_clip_npsave_fig(temp_daily_path, 'temperature_daily', d3=False)
    # mask_clip_npsave_fig(ws_daily_path, 'wind_speed_daily', d3=False)
    # mask_clip_npsave_fig(wd_daily_path, 'wind_direction_daily', d3=False)
    mask_clip_npsave_fig(mh_path, 'multi_hazard', d3=False)
    mask_clip_npsave_fig(wf_path, 'wildfire', d3=False)

    # filename = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/Japan/npy_arrays/masked_precipitation_japan.npy'
    # np.save(filename.replace('.npy', '_flat.npy'), np.load(filename)[-1])
    filename = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/npy_arrays/masked_landcover_{region}.npy'
    np.save(filename.replace('.npy', '_flat.npy'), np.load(filename)[-1])
    filename = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/npy_arrays/masked_NDVI_{region}.npy'
    np.save(filename.replace('.npy', '_flat.npy'), np.mean(np.load(filename), axis=0))

    # # combine flood and surge
    # flood = np.load(f'{dir_path}npy_arrays/masked_flood_japan.npy')
    # surge = np.load(f'{dir_path}npy_arrays/masked_surge_japan.npy')
    # flood_surge = flood + surge
    # flood_surge[flood_surge > 1] = 1
    # np.save(f'{dir_path}npy_arrays/masked_flood_surge_japan.npy', flood_surge)

def npy_to_raster(filename):
    # This routine to go from npy to tif
    # data = data[2755:2955,5825:5975]
    with rasterio.open("masked_mosaic_japan.tif", "r") as src:
        data = src.read()[0]
        profile = src.profile
        transform = src.transform

    # Create a mask based on the condition where elevation > 0
    mask_arr = data != -9999
    mask_arr = mask_arr[0:5500,2300:8800]

    # Define the window for the clipped data
    # window = Window(5825, 2755, 150, 200)  # Sado
    window = Window(2300, 0, 6500, 5500)  # Japan

    # Read the clipped data using the window
    # clipped_data = data[window.col_off:window.col_off + window.width, window.row_off:window.row_off + window.height]
    clipped_data = np.load(filename)

    masked_clipped_data = np.where(mask_arr, clipped_data, np.nan)

    # Update the profile with the new dimensions and window transform
    profile.update({
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, transform)
    })
    print(profile)

    # Save the clipped data as a new GeoTIFF file
    with rasterio.open(filename.replace('.npy', '.tif'), "w", **profile) as dst:
        dst.write(masked_clipped_data, 1)  # Write the clipped data to the new GeoTIFF file
        print(dst.bounds)

def download_osm():
    import osm_flex
    import osm_flex.download as dl
    import osm_flex.extract as ex
    import osm_flex.config
    import osm_flex.clip as cp
    from shapely.geometry import MultiLineString, MultiPolygon

    iso3 = 'JPN'
    path_dump = dl.get_country_geofabrik(iso3)
    # print(f'Saved as {path_dump}')

    # osm_flex.config.DICT_CIS_OSM.keys()

    gdf = ex.extract_cis(path_dump, 'road')
    gdf_road = gdf.copy()

    # Convert LineString geometries to MultiLineString
    # gdf_mainroad['geometry'] = gdf_mainroad['geometry'].apply(lambda geom: MultiPolygon([geom]) if geom.type == 'LineString' else geom)
    gdf_road['geometry'] = gdf_road['geometry'].apply(lambda geom: geom.buffer(0.0001) if geom.type == 'LineString' else geom)

    # Save to shapefile
    gdf_road.to_file("Japan/small_roads_japan.shp")
    gdf_road.to_file("Japan/small_roads_japan.gpkg", driver="GPKG")

def HWSD(region='Europe'):
    # import pyodbc
    data_type = 'LAYER'  # 'SMU', 'LAYER'

    # Replace 'YourMDBFile.mdb' with the path to your actual .mdb file
    mdb_file_path = r"HWSD/HWSD2_DB/HWSD2.mdb"

    # Replace 'YourTableName' with the actual table name ('HWDS2_LAYERS' in this case)
    table_name = f'HWSD2_{data_type}'

    # Define the connection string
    conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + mdb_file_path)

    # Establish a connection to the database
    connection = pyodbc.connect(conn_str)

    # Create a cursor
    cursor = connection.cursor()

    # Execute a query to fetch data from the table
    query = f'SELECT * FROM {table_name}'
    result = cursor.execute(query)

    # Fetch all rows from the result set
    rows = result.fetchall()

    # Get the column names from the cursor description
    columns = [column[0] for column in cursor.description]

    # Create a Pandas DataFrame
    df = pd.DataFrame(np.array(rows), columns=columns)

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Save the DataFrame
    df.to_excel(f'HWSD/HWSD2_{data_type}.xlsx')
    df = pd.read_excel(f'HWSD/HWSD2_{data_type}.xlsx')

    # Assuming you already have the DataFrame 'df'
    unique_values = df['WRB2'].unique()

    if data_type == 'SMU':
        unique_values = unique_values[:-1]

    # Sort unique values alphabetically
    sorted_values = sorted(unique_values)

    # Create a dictionary mapping values to numeric indices
    value_to_index = {value: index + 1 for index, value in enumerate(sorted_values)}

    # Create a new column 'WRB2 Label' in the DataFrame
    df['WRB2 Label'] = df['WRB2'].map(value_to_index)

    # Open the .bil file using rasterio
    with rasterio.open('HWSD/HWSD2_RASTER/HWSD2.bil', 'r') as src:
        # Read the raster data into a NumPy array
        bil_data = src.read(1)
        metadata = src.meta

    # set new nodata value
    bil_data[bil_data == 65535] = 0
    metadata['nodata'] = 0

    # Create a mapping between ID numbers and 'WRB2', Add an extra row with values 0 (nodata)
    id_to_wrb2_mapping = dict(zip(df['HWSD2_SMU_ID']._append(pd.Series([0], index=['HWSD2_SMU_ID']), ignore_index=True),
                                df['WRB2 Label']._append(pd.Series([0], index=['WRB2 Labe']), ignore_index=True)))

    # Replace ID numbers with 'WRB2' values in the NumPy array
    bil_data_mapped = np.vectorize(id_to_wrb2_mapping.get)(bil_data)

    # Open a new GeoTIFF file for writing
    with rasterio.open(f'/projects/0/FWC2/MYRIAD/data/HWSD/HWSD2_WRB2_{data_type}.tif', 'w', **metadata) as dst:
        # Write the mapped NumPy array to the GeoTIFF file
        dst.write(bil_data_mapped, 1)

    # Open the filled_dem_japan.tif raster to get its bounding box
    with rasterio.open(f"Region/{region}/masked_mosaic_{region}.tif", "r") as src_out:
        out_bounds = src_out.bounds
        out_nodata = src_out.nodata
        dem = src_out.read(1)

    # Open the PGA.tif raster to get its bounding box and data
    with rasterio.open(f"/projects/0/FWC2/MYRIAD/data/HWSD/HWSD2_WRB2_{data_type}.tif", "r") as src_in:
        in_res = src_in.res
        hwsd = src_in.read(1, window=src_in.window(*out_bounds))
        in_nodata = src_in.nodata
        crs = src_in.crs

    # Define the factor by which to increase the resolution
    incr_fact = 3

    # Repeat each value in the array
    hwsd_enlarged = np.repeat(np.repeat(hwsd, incr_fact, axis=0), incr_fact, axis=1)

    # Create a new transform for the enlarged raster
    transform = from_origin(out_bounds.left, out_bounds.top, in_res[0] / incr_fact, in_res[1] / incr_fact)

    # Write the enlarged PGA raster to a new file
    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': hwsd_enlarged.dtype,
        'width': hwsd_enlarged.shape[1],
        'height': hwsd_enlarged.shape[0],
        'crs': crs,
        'transform': transform,
        'nodata': 0,
    }

    hwsd_enlarged[dem == out_nodata] = in_nodata
    hwsd_enlarged[hwsd_enlarged == in_nodata] = 0

    with rasterio.open(f'Region/{region}/HWSD2_WRB2_{data_type}_{region}_300m.tif', 'w', **metadata) as dst:
        dst.write(hwsd_enlarged, 1)

def GVP(region='Europe'):
    from shapely.geometry import Point, box
    from shapely.ops import transform
    import pyproj

    # Read the GVP excel and store in gpd
    df = pd.read_excel("/projects/0/FWC2/MYRIAD/data/Volcanos/GVP_Eruption_Results.xlsx", skiprows=1)
    # df.dropna(subset=['VEI'], inplace=True)
    geometry = [Point(lon, lat) for lat, lon in zip(df['Latitude'], df['Longitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Calculate VEI to distance
    gdf['buffer'] = 3.0408 * np.exp(0.6956 * df['VEI'])
    gdf['buffer'].fillna(0.01, inplace=True)

    # Load the reference raster
    with rasterio.open(f'Region/{region}/masked_mosaic_{region}.tif') as src:
        raster_extent_box = box(*src.bounds)
        raster_profile = src.profile
        raster_shape = (src.height, src.width)

    # Filter each point in the GeoDataFrame within the raster extent
    gdf['within_extent'] = gdf.geometry.within(raster_extent_box)
    gdf = gdf[gdf['within_extent']]
    gdf = gdf[gdf['Eruption Category'] == 'Confirmed Eruption']


    # Define the buffer function with proper projection conversion
    def buffer_in_meters(point, buffer_distance_km):
        # Define the projection systems
        wgs84 = pyproj.CRS('EPSG:4326')
        utm_proj = pyproj.CRS('EPSG:32633')  # You may need to adjust the UTM zone
        
        # Create a transformer to convert from WGS84 to UTM
        transformer = pyproj.Transformer.from_crs(wgs84, utm_proj, always_xy=True).transform
        
        # Transform the point to UTM
        point_utm = transform(transformer, point)
        
        # Buffer the point in UTM
        buffered_point_utm = point_utm.buffer(buffer_distance_km * 1000)  # Convert buffer distance from km to meters
        
        # Create a transformer for the inverse transformation (UTM to WGS84)
        transformer_utm_to_wgs84 = pyproj.Transformer.from_crs(utm_proj, wgs84, always_xy=True).transform
        
        # Transform the buffered point back to WGS84
        buffered_point_wgs84 = transform(transformer_utm_to_wgs84, buffered_point_utm)
        
        return buffered_point_wgs84

    # Apply the buffer function to the GeoDataFrame
    gdf['geometry'] = gdf.apply(lambda row: buffer_in_meters(row['geometry'], row['buffer']), axis=1)

    # Rasterize the buffered polygons and count the number of overlapping polygons in each cell
    def count_polygons(raster_profile, buffered_geometries):
        # Create an empty array with the same shape as the raster
        raster_array = np.zeros(raster_shape, dtype=np.uint8)

        # Iterate through each buffered geometry and accumulate the count
        for geom in buffered_geometries:
            mask = geometry_mask([geom], out_shape=raster_shape, transform=raster_profile['transform'], invert=True)
            raster_array += mask.astype(np.uint8)

        return raster_array

    # Apply the count_polygons function
    result_raster = count_polygons(raster_profile, gdf['geometry'])

    # Write the result raster to a new GeoTIFF file
    with rasterio.open(f'Region/{region}/GVP_Holocene_count_{region}.tif', 'w', **raster_profile) as dst:
        dst.write(result_raster, 1)

    # Display the result raster
    plt.imshow(result_raster, cmap='viridis', extent=(raster_profile['transform'][2], raster_profile['transform'][2] + raster_profile['transform'][0] * raster_shape[1],
                                                    raster_profile['transform'][5] + raster_profile['transform'][4] * raster_shape[0], raster_profile['transform'][5]),
            origin='upper')
    plt.colorbar(label='Number of Polygons')
    plt.title('Rasterized Polygons')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/Volcanos.png', dpi=150)
    # plt.show()

def WindSpeed():
    from xclim.indices import uas_vas_2_sfcwind
    import calendar
    logger.info('in function')

    # Define the range of years you want to process
    start_year = 1950
    end_year = 2022
    start_year = yeararg
    end_year = yeararg + 25
    end_year  = min(end_year , 2022)

    # Define the number of days in each chunk
    days_per_chunk = 10
    encoding_options = {'zlib': True, 'complevel': 4}
    first_time = 0

    # Loop through years
    for year in range(start_year, end_year + 1):
        # Load the entire year's data for u and v wind components
        dsu = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_Hourly_10m_u_component_of_wind_{year}.grib')
        dsv = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_Hourly_10m_v_component_of_wind_{year}.grib')

        # Check for leap year
        leap_year = calendar.isleap(year)
        days_in_year = 366 if leap_year else 365

        # Check if shift needed
        year_check = np.datetime_as_string(dsu.time.values[0], unit='Y').astype(int)
        if year == year_check:
            shift = 0
        else:
            shift = 1

        # Loop through 10-day chunks
        # for chunk_start in range(0 + shift, days_in_year, days_per_chunk):
        for chunk_start in range(360 + shift, days_in_year, days_per_chunk):
            logger.info(f'Year: {year} Chunk: {chunk_start}')
            chunk_end = min(chunk_start + days_per_chunk, days_in_year) + 1  # Ensure the last chunk gets right number

            # Select data for the current 10-day chunk
            # dsu = dsu.sel(latitude=slice(int(round(bounds[3], 0)), int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0))))
            # dsv = dsv.sel(latitude=slice(int(round(bounds[3], 0)), int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0))))

            dsu_chunk = dsu.isel(time=slice(chunk_start, chunk_end))
            dsv_chunk = dsv.isel(time=slice(chunk_start, chunk_end))
            # dsu_chunk = dsu.isel(time=slice(chunk_start, 1))
            # dsv_chunk = dsv.isel(time=slice(chunk_start, 1))

            # Combine u and v components to calculate surface wind speed
            sfcWind = uas_vas_2_sfcwind(uas=dsu_chunk['u10'], vas=dsv_chunk['v10'], calm_wind_thresh="0.3 m/s")

            # Extract wind speed and wind direction arrays
            wind_speed = sfcWind[0]
            wind_direction = sfcWind[1] 
            wind_speed_filled = wind_speed.fillna(0)

            # Resample to daily data and find the maximum wind speed for each day
            daily_max_wind_speed = wind_speed.resample(time='D').max(dim=['time', 'step'], keep_attrs=True)
            daily_max_wind_speed = daily_max_wind_speed.to_dataset(name='wind_speed')
            # daily_max_wind_speed['wind_speed'] = daily_max_wind_speed['wind_speed'].astype(np.float16)
            # logger.info('daily max')

            if first_time == 0:
                # Extract the hour from the step coordinate
                hours = wind_speed['step'] / pd.to_timedelta('1H')
            first_time += 1

            # Select only the noon values (assuming your data has values for each hour)
            daily_noon_wind_speed = wind_speed.sel(step=hours == 12)
            daily_noon_wind_speed = daily_noon_wind_speed.to_dataset(name='wind_speed')
            # daily_noon_wind_speed['wind_speed'] = daily_noon_wind_speed['wind_speed'].astype(np.float16)
            # logger.info('daily noon')

            # Wind direction for max daily
            # Step 1: Find the index of the maximum wind speed for each day
            max_wind_speed_index = wind_speed_filled.argmax(dim='step')

            # Step 2: Extract the corresponding hour (step) for the daily max wind speed
            max_wind_speed_hour = wind_speed_filled['step'].isel(step=max_wind_speed_index)

            # Step 3: Use the extracted step values to select wind_direction values
            wind_direction_max_speed = wind_direction.sel(step=max_wind_speed_hour)
            # logger.info('daily dir')

            # Rename the variable to "wind_direction"
            wind_direction_max_speed = wind_direction_max_speed.to_dataset(name='wind_direction')

            # Convert the variable values to integers
            # wind_direction_max_speed['wind_direction'] = wind_direction_max_speed['wind_direction'].astype(int)
            wind_direction_max_speed = wind_direction_max_speed.drop('step')

            # # Testing and comparing values
            # wind_speed.sel(latitude=wind_speed['latitude'].isel(latitude=0),longitude=wind_speed['longitude'].isel(longitude=0)).values
            # daily_max_wind_speed.sel(latitude=wind_speed['latitude'].isel(latitude=0),longitude=wind_speed['longitude'].isel(longitude=0)).values
            # noon_values.sel(latitude=wind_speed['latitude'].isel(latitude=0),longitude=wind_speed['longitude'].isel(longitude=0)).values
            # wind_direction.sel(latitude=wind_speed['latitude'].isel(latitude=0),longitude=wind_speed['longitude'].isel(longitude=0)).values
            # wind_direction_max_speed.sel(latitude=wind_speed['latitude'].isel(latitude=0),longitude=wind_speed['longitude'].isel(longitude=0)).values

            # Save to netCDF
            daily_max_wind_speed.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_daily_wind_speed_max_{year}_chunk_{chunk_start - shift}.nc',
                                           format='netcdf4', engine='netcdf4', encoding={'wind_speed': encoding_options})
            daily_noon_wind_speed.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_daily_noon_wind_speed_{year}_chunk_{chunk_start - shift}.nc',
                                            format='netcdf4', engine='netcdf4', encoding={'wind_speed': encoding_options})
            wind_direction_max_speed.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_daily_wind_direction_max_speed_{year}_chunk_{chunk_start - shift}.nc',
                                               format='netcdf4', engine='netcdf4', encoding={'wind_direction': encoding_options})

        # Close the datasets to free up resources
        dsu.close()
        dsv.close()

def ProcessERA5region(region='Europe'):
    from xclim.indicators.atmos import relative_humidity_from_dewpoint

    if region == 'Europe':
        leftshift = 0
        rightshift = 0.1
        with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/mosaic_Europe.tif", "r") as src:
            bounds = src.bounds
    elif region == 'Japan':
        leftshift = 0.1
        rightshift = 0
        with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_japan.tif", "r") as src:
            bounds = src.bounds

    for year in range(1950, 2022):
        logger.info(year)

        # Open wind speed daymax dataset
        ws_max = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_daily_wind_speed_max_{year}_chunk_{chunk_start}.nc')
                           for chunk_start in range(0, 365, 10)], dim='time')
        ws_max = ws_max.assign_coords(longitude=((ws_max.longitude + 180) % 360 - 180))
        ws_max = ws_max.sortby("longitude")
        
        # Subset the data to include only the desired lat and lon
        crop_ws_max = ws_max.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_dt2 = crop_dt2.isel(time=slice(0, 1))
        crop_ws_max.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_{year}_daymax.nc', format='netcdf4', engine='netcdf4')

        # Open wind speed noon dataset
        ws_noon = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_daily_noon_wind_speed_{year}_chunk_{chunk_start}.nc')
                           for chunk_start in range(0, 365, 10)], dim='time')
        ws_noon = ws_noon.assign_coords(longitude=((ws_noon.longitude + 180) % 360 - 180))
        ws_noon = ws_noon.sortby("longitude")
        
        # Subset the data to include only the desired lat and lon
        crop_ws_noon = ws_noon.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_dt2 = crop_dt2.isel(time=slice(0, 1))
        crop_ws_noon.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_{year}_noon.nc', format='netcdf4', engine='netcdf4')

        # Open wind speed noon dataset
        wd_max = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_daily_wind_direction_max_speed_{year}_chunk_{chunk_start}.nc')
                           for chunk_start in range(0, 365, 10)], dim='time')
        wd_max = wd_max.assign_coords(longitude=((wd_max.longitude + 180) % 360 - 180))
        wd_max = wd_max.sortby("longitude")
        
        # Subset the data to include only the desired lat and lon
        crop_wd_max = wd_max.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_dt2 = crop_dt2.isel(time=slice(0, 1))
        crop_wd_max.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_direction_{year}_daymax.nc', format='netcdf4', engine='netcdf4')

        # Open temperature noon dataset
        grib_file = f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_Hourly_2m_temperature_{year}_noon.grib'
        t2 = xr.open_dataset(grib_file, engine="cfgrib")
        t2 = t2.assign_coords(longitude=((t2.longitude + 180) % 360 - 180))
        t2 = t2.sortby("longitude")

        # Subset the data to include only the desired lat and lon
        crop_t2 = t2.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        t2.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)), int(round(bounds[2], 0)) + 0.1))
        # crop_t2 = crop_t2.isel(time=slice(0, 1))
        crop_t2.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_{year}_noon.nc', format='netcdf4', engine='netcdf4')

        # Open temperature daymean dataset
        grib_file = f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_Hourly_2m_temperature_{year}_daymean.grib'
        try:
            t2mean = xr.open_dataset(grib_file, engine="cfgrib")
            t2mean = t2mean.assign_coords(longitude=((t2mean.longitude + 180) % 360 - 180))
            t2mean = t2mean.sortby("longitude")
        except:
            logger.info('Error way')
            t2mean_inst = xr.open_dataset(grib_file, engine="cfgrib", filter_by_keys={'stepType': 'instant'})
            t2mean_avg = xr.open_dataset(grib_file, engine="cfgrib", filter_by_keys={'stepType': 'avg'})
            t2mean = xr.concat([t2mean_avg, t2mean_inst], dim='time')
            xr.open_dataset('/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_Europe_2m_temperature_2020_daymean.nc')

        # Subset the data to include only the desired lat and lon
        crop_t2mean = t2mean.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_t2 = crop_t2.isel(time=slice(0, 1))
        crop_t2mean.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_{year}_daymean.nc', format='netcdf4', engine='netcdf4')

        # Open dewpoint temperature dataset
        grib_file = f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_Hourly_2m_dewpoint_temperature_{year}_noon.grib'
        dt2 = xr.open_dataset(grib_file, engine="cfgrib")
        dt2 = dt2.assign_coords(longitude=((dt2.longitude + 180) % 360 - 180))
        dt2 = dt2.sortby("longitude")

        # Subset the data to include only the desired lat and lon
        crop_dt2 = dt2.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_dt2 = crop_dt2.isel(time=slice(0, 1))
        crop_dt2.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_dewpoint_temperature_{year}_noon.nc', format='netcdf4', engine='netcdf4')

        # Calculate relative humidity and save dataset
        crop_rh2 = relative_humidity_from_dewpoint(tas=crop_t2['t2m'], tdps=crop_dt2['d2m'])
        crop_rh2 = crop_rh2.to_dataset(name='hurs')
        crop_rh2.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_relative_humidity_{year}_noon.nc', format='netcdf4', engine='netcdf4')

        # Open precipitation dataset
        grib_file = f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_Hourly_Total_precipitation_{year}_daysum.grib'
        prec = xr.open_dataset(grib_file, engine="cfgrib")
        prec = prec.assign_coords(longitude=((prec.longitude + 180) % 360 - 180))
        prec = prec.sortby("longitude")

        # Subset the data to include only the desired lat and lon
        crop_prec = prec.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_dt2 = crop_dt2.isel(time=slice(0, 1))
        crop_prec.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_{region}_Total_precipitation_{year}_daysum.nc', format='netcdf4', engine='netcdf4')

        # Open pet dataset
        grib_file = f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/PET/ERA5-Land_Hourly_potential_evaporation_{year}_daysum.grib'
        pet = xr.open_dataset(grib_file, engine="cfgrib")
        pet = pet.assign_coords(longitude=((pet.longitude + 180) % 360 - 180))
        pet = pet.sortby("longitude")

        # Subset the data to include only the desired lat and lon
        crop_pet = pet.sel(latitude=slice(int(round(bounds[3], 0)) + 0.1, int(round(bounds[1], 0))), longitude=slice(int(round(bounds[0], 0)) - leftshift, int(round(bounds[2], 0)) + rightshift))
        # crop_dt2 = crop_dt2.isel(time=slice(0, 1))
        crop_pet.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/PET/ERA5-Land_{region}_potential_evaporation_{year}_daysum.nc', format='netcdf4', engine='netcdf4')
        

    # Combine relative humidity data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_relative_humidity_{year}_noon.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_relative_humidity_noon.nc')

    # Combine wind speed daymax data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_{year}_daymax.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_daymax.nc')

    # Combine wind speed noon data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_{year}_noon.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_noon.nc')

    # Combine wind direction daymax data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_direction_{year}_daymax.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_direction_daymax.nc')

    # Combine temperature noon data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_{year}_noon.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_noon.nc')

    # Combine temperature daymean data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_{year}_daymean.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_daymean.nc')

    # Combine dewpoint temperature noon data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_dewpoint_temperature_{year}_noon.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_dewpoint_temperature_noon.nc')

    # Combine precipitation daysum data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_{region}_Total_precipitation_{year}_daysum.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_{region}_Total_precipitation_daysum.nc')

    # Combine pet daysum data and save
    combined_data = xr.concat([xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/PET/ERA5-Land_{region}_potential_evaporation_{year}_daysum.nc')
        for year in range(1951, 2022)], dim='time')
    combined_data.to_netcdf(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/PET/ERA5-Land_{region}_potential_evaporation_daysum.nc')

def AtmosphericIndices(region='Europe'):
    def ewi_calc():
        logger.info('ewi calc')
        # *** Extreme wind index ***
        # Estimate the number of windy days
        ds = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_daymax.nc')
        wd = windy_days(ds['wind_speed'])
        wd_sum = wd.sum(dim='time')
        increase2tif(wd_sum, 'windy_days')

    def hwi_calc():
        logger.info('hwi calc')
        # *** Heatwave index ***
        # Open temperature dataset and convert to Celcius
        ds = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_daymean.nc')
        ds['t2m'].attrs['units'] = 'Celsius'
        # ds = ds.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        ds = ds - 273.15

        # Get heat threshold
        quantile_95 = ds.quantile(0.95, dim='time')

        # Calculate heat wave index rolling mean
        # ds_5d = ds.rolling(time=5, min_periods=1).mean()
        # ds_5d = ds_3d.isel(time=slice(150, 153))
        # hwi = ds.where((ds_5d['t2m'] >= quantile_95['t2m'].values) & (ds_5d['t2m'] >= 25))
        # hwi_sum = hwi.sum(dim='time')
        # increase2tif(hwi_sum['t2m'], 'heatwave_index_rollingmean')

        # Calculate heat wave index window
        ds_mask = ds.where(ds['t2m'] >= quantile_95['t2m'].values)
        ds_mask['t2m'].attrs['units'] = 'Celsius'
        hwi = heat_wave_index(ds_mask['t2m'], thresh='25.0 degC', window=5)
        hwi_sum = hwi.sum(dim='time')
        increase2tif(hwi_sum, 'heatwave_index_window')

    def di_calc():
        logger.info('di calc')
        # *** Drought index ***
        # open prec and pet
        prec = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_{region}_Total_precipitation_daysum.nc')
        # prec = prec.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        prec_monthly = prec.resample(time='M').sum(dim='time', keep_attrs=True)
        pet = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/PET/ERA5-Land_{region}_potential_evaporation_daysum.nc')
        # pet = pet.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        pet_monthly = pet.resample(time='M').sum(dim='time', keep_attrs=True)

        # Calculate water budget
        # wb = water_budget(prec['tp'], evspsblpot=pet['pev'])
        wb = (prec_monthly['tp'] - pet_monthly['pev'])  * 1000
        wb.attrs['units'] = 'kg m-2 s-1'

        # Calculate SPEI
        spei = standardized_precipitation_evapotranspiration_index(wb)
        spei_3m = spei.rolling(time=3, min_periods=1).mean()

        # Calculate drought index
        di_sum = (spei_3m < -2).sum(dim='time')
        increase2tif(di_sum, 'drought_index')

    def fwi_calc():
        logger.info('fwi calc')
        # *** Fire Weather index ***
        # Open datasets and prepare
        prec = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_{region}_Total_precipitation_daysum.nc')
        rh = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_relative_humidity_noon.nc')
        wind = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_noon.nc')
        temp = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_noon.nc')

        # # testing area
        # prec = prec.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        # rh = rh.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        # wind = wind.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        # temp = temp.isel(latitude=slice(0, 4), longitude=slice(0, 4))

        # Unit conversions
        prec = prec * 1000
        prec['tp'].attrs['units'] = 'mm/day'
        temp = temp - 273.15
        temp['t2m'].attrs['units'] = 'Celsius'
        wind['wind_speed'] = wind['wind_speed'] * 3.6
        wind['wind_speed'].attrs['units'] = 'km/h'
        rh['hurs'].attrs['units'] = '%'
        prec['time'] = prec['time'].dt.floor('D')
        prec['time'] = prec['time'].dt.date
        wind = wind.squeeze('step')

        # PROBLEM WIND SKIPS LEAP YEARS
        fwi = fire._cffwis.cffwis_indices(tas=temp['t2m'], pr=prec['tp'], hurs=rh['hurs'], sfcWind=wind['wind_speed'], lat=temp.latitude)
        fwi_sum = (fwi[5] > 20).sum(dim='time') + (fwi[5] > 30).sum(dim='time') + (fwi[5] > 40).sum(dim='time')
        increase2tif(fwi_sum, 'fire_weather_index')

        fwi_mean = fwi[5].mean(dim='time')
        increase2tif(fwi_mean, 'fire_weather_index_mean')

        rh_mean = rh['hurs'].mean(dim='time')
        increase2tif(rh_mean, 'relative_humidity')
    
    def temp_calc():
        logger.info('temp calc')
        # *** Average yearly temperature ***
        # open prec and resample to yearly
        ds = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Temperature/ERA5-Land_{region}_2m_temperature_daymean.nc')
        ds['t2m'].attrs['units'] = 'Celsius'
        # ds = ds.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        ds = ds - 273.15

        ds['time'] = pd.to_datetime(ds['time'], errors='coerce')
        yearly_temp = ds.resample(time='Y').sum(dim='time', keep_attrs=True)
        mean_yearly_temp = yearly_temp.mean(dim='time', keep_attrs=True)
        increase2tif(mean_yearly_temp['t2m'] / 365, 'mean_yearly_temperature')
    
    def wind_calc():
        logger.info('wind calc')
        # *** Average yearly wind ***
        # open prec and resample to yearly
        ds = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_speed_daymax.nc')
        # ds = ds.isel(latitude=slice(0, 4), longitude=slice(0, 4))

        ds['time'] = pd.to_datetime(ds['time'], errors='coerce')
        yearly_wind = ds.resample(time='Y').sum(dim='time', keep_attrs=True)
        mean_yearly_wind = yearly_wind.mean(dim='time', keep_attrs=True)
        increase2tif(mean_yearly_wind['wind_speed'] / 365, 'mean_wind_speed')

        # open prec and resample to yearly
        ds = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Wind/ERA5-Land_{region}_wind_direction_daymax.nc')
        ds['wind_direction'] = ds['wind_direction'].where(ds['wind_direction'] != 0, np.nan)
        # Apply the function to the wind direction data
        ds['wind_direction_cardinal'] = xr.apply_ufunc(classify_wind_direction, ds['wind_direction'], vectorize=True)
        ds_mode = xr.apply_ufunc(
                            compute_mode,
                            ds['wind_direction_cardinal'],
                            input_core_dims=[['time']],
                            dask='allowed',
                            vectorize=True,
                            keep_attrs=True,
                            output_dtypes=[float],
                            output_core_dims=[[]]
                        )

        increase2tif(ds_mode, 'mode_wind_direction')

    def prec_calc():
        logger.info('prec calc')
        # *** Average yearly precipitation ***
        # open prec and resample to yearly
        prec = xr.open_dataset(f'/projects/0/FWC2/MYRIAD/data/ERA5-Land/Precipitation/ERA5-Land_{region}_Total_precipitation_daysum.nc')
        # prec = prec.isel(latitude=slice(0, 4), longitude=slice(0, 4))
        prec['time'] = pd.to_datetime(prec['time'], errors='coerce')
        yearly_prec = prec.resample(time='Y').sum(dim='time', keep_attrs=True)
        mean_yearly_prec = yearly_prec.mean(dim='time', keep_attrs=True)
        increase2tif(mean_yearly_prec['tp'], 'mean_yearly_precipitation')

    def increase2tif(var, var_name):
        logger.info('increase2tif')
        if var_name == 'mode_wind_direction':
            resmethod = Resampling.nearest
        else:
            resmethod = Resampling.cubic_spline

        # Use np.kron to increase the resolution
        var_inc = np.kron(var.values, np.ones((factor, factor)))
        trimmed_var_inc = var_inc[int(factor/2):-int(factor/2), int(factor/2):-int(factor/2)]

        # Save the clipped data as a new GeoTIFF file
        with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/{var_name}_{region}.tif", "w", **profile) as dst:
            dst.write(trimmed_var_inc, 1)

        # Resample NetCDF file using xarray's reproject function
        var = var.rio.write_crs(profile['crs'])
        var_resampled = var.rio.reproject(dst_crs=profile['crs'], shape=(profile['height'], profile['width']),bounds=bounds,resampling=resmethod)

        # Save the clipped data as a new GeoTIFF file
        with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/{var_name}_{region}_resampled.tif", "w", **profile) as dst:
            dst.write(var_resampled.values, 1)

    # Define a function to calculate the mode
    def calculate_mode(values):
        unique, counts = np.unique(values, return_counts=True)
        mode_index = np.argmax(counts)
        return unique[mode_index]
    
    # Define a function to compute mode using scipy.stats.mode
    def compute_mode(arr):
        return mode(arr)[0]
    
    # Define a function to classify wind direction into cardinal directions
    def classify_wind_direction(wind_dir):
        if wind_dir <= 22.5 or wind_dir > 337.5:
            return 0 # 'N'
        elif 22.5 < wind_dir <= 67.5:
            return 1 # 'NE'
        elif 67.5 < wind_dir <= 112.5:
            return 2 # 'E'
        elif 112.5 < wind_dir <= 157.5:
            return 3 # 'SE'
        elif 157.5 < wind_dir <= 202.5:
            return 4 # 'S'
        elif 202.5 < wind_dir <= 247.5:
            return 5 # 'SW'
        elif 247.5 < wind_dir <= 292.5:
            return 6 # 'W'
        elif 292.5 < wind_dir <= 337.5:
            return 7 # 'NW'
        else:
            return 8 # 'NaN'

    from xclim.indices import windy_days
    from xclim.indices import heat_wave_index
    from xclim.indices import water_budget
    from xclim.indices import standardized_precipitation_evapotranspiration_index
    from xclim.indices import fire
    from rasterio.warp import transform_bounds
    from rasterio.enums import Resampling

    # prepare tif metadata
    src = rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/mosaic_{region}.tif", "r")
    bounds = src.bounds
    # bounds = transform_bounds(src.crs, {'init': 'epsg:4326'}, *src.bounds)
    profile = src.profile
    factor = 36  # Define the factor by which you want to increase the resolution

    # ewi_calc()
    # hwi_calc()
    # di_calc()
    fwi_calc()
    # prec_calc()
    # temp_calc()
    # wind_calc()

def WildFires(region='Europe'):
    logger.info('Wildfires')
    from rasterio.features import rasterize
    # Load wildfire data and store in gdf
    wf = pd.read_csv(f"/projects/0/FWC2/MYRIAD/data/wildfires/final.csv")
    gdf = gpd.GeoDataFrame(wf, geometry=gpd.GeoSeries.from_wkt(wf['geometry']))

    # Open GeoTIFF for metadata
    with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/filled_masked_mosaic_{region}.tif") as src:
        bounds = src.bounds
        profile = src.profile
        transform = src.transform
        height, width = src.shape

    # Filter the GeoDataFrame based on geometries within the bounds of the GeoTIFF
    filtered_gdf = gdf.cx[bounds.left: bounds.right, bounds.bottom: bounds.top]

    # Save to shape
    filtered_gdf.to_file(f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/Wildfires_{region}.shp')

    # Convert geometries to list of (geometry, value)
    shapes = [(geom, 1) for geom in filtered_gdf.geometry if geom is not None]

    # Create an empty array for the raster
    raster = np.zeros((height, width), dtype=np.uint16)

    # Rasterize with overlap count
    rasterized = rasterize(
        shapes=shapes,
        out=raster,
        transform=transform,
        all_touched=True,  # Counts all pixels touched by polygons
        merge_alg=rasterio.enums.MergeAlg.add  # Sum overlapping polygons
    )

    # Save the rasterized wildfire count
    profile.update(dtype=rasterized.dtype, count=1, nodata=0)

    with rasterio.open(f"/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/Wildfires_{region}.tif", "w", **profile) as dst:
        dst.write(rasterized, 1)

def WindGusts():
    from shapely.geometry import LineString, Point

    out_tif = 'wind_gusts_japan_300m.tif'

    # Read shapefile
    gdfa = gpd.read_file('Japan/Wind/A30b-11_GML/AppearancePoint_Gust.shp')
    gdfd = gpd.read_file('Japan/Wind/A30b-11_GML/DisappearancePoint_Gust.shp')

    # Create a GeoDataFrame for lines
    lines_data = {'geometry': [], 'A30b_003': []}
    # for column in gdfd.columns[:-1]:
    #     lines_data[column] = []

    # Separate DataFrame for unmatched points
    unmatched_points = gdfa.copy()

    # Iterate through each row in the appearance GeoDataFrame
    for idx, appearance_point in gdfa.iterrows():
        # Find matching disappearance point
        matching_disappearance = gdfd[(gdfd['A30b_029'] == appearance_point['A30b_029']) & (gdfd['A30b_030'] == appearance_point['A30b_030'])]

        if not matching_disappearance.empty:
            # Create a line between appearance and disappearance points
            line_geometry = LineString([Point(appearance_point['geometry'].x, appearance_point['geometry'].y),
                                        Point(matching_disappearance['geometry'].x.iloc[0], matching_disappearance['geometry'].y.iloc[0])])

            # Add line geometry and metadata to the dictionary
            lines_data['geometry'].append(line_geometry)
            lines_data['A30b_003'].append(appearance_point['A30b_003'])
            # for column in gdfd.columns[:-1]:
            #     lines_data[column].append(appearance_point[column])

            # Remove matched points from the unmatched DataFrame
            unmatched_points = unmatched_points[unmatched_points['A30b_029'] != appearance_point['A30b_029']]

    # Create GeoDataFrame for lines
    # lines_gdf = gpd.GeoDataFrame(geometry=lines_geometry, crs=gdfa.crs)
    lines_gdf = gpd.GeoDataFrame(lines_data, crs=gdfa.crs)

    # Combine the two GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(pd.concat([lines_gdf, unmatched_points[['A30b_003', 'geometry']]], ignore_index=True), crs=gdfa.crs)

    # Create a buffer of 0.05
    combined_gdf['geometry'] = combined_gdf['geometry'].buffer(0.05)

    # Reshape columns  
    combined_gdf.insert(0, 'starttime', pd.to_datetime(combined_gdf['A30b_003'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.floor('D'))
    combined_gdf.insert(2, 'Intensity', 0)
    combined_gdf.insert(3, 'endtime', pd.to_datetime(combined_gdf['A30b_003'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.floor('D'))
    combined_gdf.insert(4, 'index1', range(1324008, len(combined_gdf) + 1324008))
    combined_gdf.insert(5, 'Id', range(1324008, len(combined_gdf) + 1324008))
    combined_gdf.insert(6, 'Unit', 'm/s')
    a003 = combined_gdf['A30b_003'].copy()
    combined_gdf.drop('A30b_003', axis=1, inplace=True)
    combined_gdf = combined_gdf.rename(columns={'geometry': 'Geometry'})

    # Combine with ew
    ew = pd.read_csv('ew.csv')
    ew.drop('Unnamed: 0', axis=1, inplace=True)
    ew_comb = pd.concat([ew, combined_gdf], ignore_index=True)
    ew_comb.to_csv('Japan/Wind/A30b-11_GML/ew2.csv')

    # Save the combined shapefile with buffer
    combined_gdf['starttime'] = a003
    combined_gdf['endtime'] = a003
    combined_gdf.to_file('Japan/Wind/A30b-11_GML/wind_gusts_japan.shp')

def MultiHazardOccurenceIndex(region='Europe'):
    # Set up the input and output paths
    csv_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/df_multi_{region}.csv'
    existing_raster_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Region/{region}/mosaic_{region}.tif'
    output_path = f'Input/{region}/myriad_hes_{region}.tif'

    # Convert the "geometry" column to GeoSeries
    df = pd.read_csv(csv_path)
    df['geometry'] = gpd.GeoSeries.from_wkt(df['Geometry'])

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df)
    gdf = gdf.set_geometry('geometry')

    # Save shapefile
    output_shape_path = f'/projects/0/FWC2/MYRIAD/Susceptibility/Input/{region}/MH_occurrence.shp'
    gdf.to_file(output_shape_path)

    # Open the existing raster and make empty array
    src = rasterio.open(existing_raster_path)
    mh_array = np.zeros((src.height, src.width)).astype(np.int16)

    # Loop over all Multi-Hazard events
    for event in gdf['Event'].unique():
        gdf_event = gdf[gdf['Event'] == event]
        logger.info(f'{event}: {gdf_event["Hazard"].values}')
        event_array = np.zeros((src.height, src.width)).astype(np.int16)
        
        # Loop over individual hazards and rasterize
        for index, row_event in gdf_event.iterrows():
            # Rasterize the polygons to a mask
            masked_hazard = rasterio.features.geometry_mask([row_event.geometry], transform=src.transform, out_shape=(src.height, src.width), all_touched=True, invert=True)
            masked_hazard = masked_hazard.astype(np.int16)
            
            # Multi-Hazard event raster
            event_array += masked_hazard
        
        # Remove areas where only one hazard occured
        event_array[event_array == 1] = 0
        event_array = np.where(event_array > 1, event_array - 1, event_array)

        # Add to final map to make Multi-Hazard occurence index
        mh_array += event_array

    # Delete the old raster file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Write the result raster to a new GeoTIFF file
    with rasterio.open(output_path, 'w', **src.profile) as dst:
        dst.write(mh_array, 1)

logger, ch = set_logger()
logger.info('check')

# yeararg = int(sys.argv[1])

# download_url()
# mosaic_region(region='Europe')
# mask_mosaic()
# dem_calc()
# hydro_dem()
# GFAS()
# rasterize_hazards()

# GEM_300m()
# region_tif = '/projects/0/FWC2/MYRIAD/Susceptibility/Region/Europe/mosaic_Europe.tif'
# input_tif = 'Region/GEM_300m.tif'
# output_tif = '/projects/0/FWC2/MYRIAD/Susceptibility/Region/Europe/GEM_300m_Europe.tif'
# clip_to_region(region_tif, input_tif, output_tif)

# # GLIM_300m()
# region_tif = '/projects/0/FWC2/MYRIAD/Susceptibility/Region/Europe/mosaic_Europe.tif'
# input_tif = '/projects/0/FWC2/MYRIAD/data/GLIM/GLIM_300m.tif'
# output_tif = '/projects/0/FWC2/MYRIAD/Susceptibility/Region/Europe/GLIM_300m_Europe.tif'
# clip_to_region(region_tif, input_tif, output_tif)

# NDVI()
# PGA()
# HWSD()
# GVP()
# landcover()
# precipitation()
# WindSpeed()
# ProcessERA5region()
# AtmosphericIndices()
# WildFires()
# WindGusts()
# sys.exit(0)

# distance_to('/projects/0/FWC2/MYRIAD/Susceptibility/Region/Europe/GEM_300m_Europe.tif')
# distance_to('/projects/0/FWC2/MYRIAD/Susceptibility/Region/Europe/strahler_Europe.tif')
# rasterize_world()
# distance_to('/projects/0/FWC2/MYRIAD/Susceptibility/Region/World/streams_World.tif')
# distance_to('/projects/0/FWC2/MYRIAD/Susceptibility/Region/World/coastline_World.tif')

# rasterize_hazards(hazard_condition='single', hazard_type='ls')
# MultiHazardOccurenceIndex()
# rasterize_jpn()

prepare_data_nn()
# npy_to_raster(f'Output/japan_Flood_Susceptibility_base_model.npy')
# npy_to_raster(f'Output/japan_Landslide_Susceptibility_base_model.npy')
# npy_to_raster(f'Output/japan_Tsunami_Susceptibility_base_model.npy')
# npy_to_raster(f'Output/japan_Multihazard_Susceptibility_meta_model_MLP.npy')
# npy_to_raster(f'Output/japan_Multihazard_Susceptibility_linear.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_drought_japan.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_fire_weather_japan.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_heatwave_japan.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_jshis_japan.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_volcano_japan.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_wind_speed_daily_japan.npy')
# npy_to_raster(f'Input/Japan/npy_arrays/masked_landcover_japan_flat.npy')
sys.exit(0)










########################### J-SHIS EARTHQUAKES ###################

from rasterio.features import rasterize
from rasterio.crs import CRS

# filename = 'Region/Earthquakes/P-Y2023-MAP-AVR-TTL_MTTL-SHAPE.shp'
# jshis = gpd.read_file(filename)
# feature = 'T30_I60_PS'
# # feature = sys.argv[1]
# resolution = (7.5 / 3600, 11.25 / 3600)  # degrees
# logger.info('Read')

# # Determine raster dimensions
# xmin, ymin, xmax, ymax = jshis.total_bounds
# width = int((xmax - xmin) / resolution[1])
# height = int((ymax - ymin) / resolution[0])

# # Define the transform
# transform = from_origin(xmin, ymax, resolution[1], resolution[0])

# for feature in ['T50_P05_SV', 'T50_P10_SV', 'T50_P39_SV']:
#     output_tif_file = f'Region/Earthquakes/J-SHIS_{feature}.tif'
#     # create tuples of geometry, value pairs, where value is the attribute value you want to burn
#     geom_value = ((geom, value) for geom, value in zip(jshis.geometry, jshis[feature]))
#     logger.info('Geom done')

#     raster_array = rasterize(
#         geom_value,
#         out_shape=(height, width),
#         transform=transform,
#         fill=0,
#         all_touched=True
#     )
#     logger.info('Rasterize done')

#     with rasterio.open(output_tif_file, 'w', driver='GTiff', 
#                     width=width, height=height, count=1, dtype=raster_array.dtype, nodata=0,
#                     crs=CRS.from_epsg(4326), transform=transform, compress='LZW') as dst:
#         dst.write(raster_array, 1)

feature = 'T50_P10_SV'
output_tif_file = f'Japan/Earthquakes/J-SHIS_{feature}.tif'

with rasterio.open(output_tif_file, "r") as src:
    data = src.read(1)
    transform = src.transform

# Open the existing GeoTIFF to get its resolution and transform
with rasterio.open("masked_mosaic_japan.tif", "r") as existing_src:
    existing_transform = existing_src.transform
    existing_resolution = existing_transform.a
    crs = existing_src.crs

    # Reproject the data to match the resolution of the existing GeoTIFF
    reprojected_data = np.empty_like(existing_src.read(1))
    reproject(
        data,
        reprojected_data,
        src_transform=transform,
        src_crs=crs,  # Assuming the original data is in WGS 84
        dst_transform=existing_transform,
        dst_crs=existing_src.crs,
        resampling=Resampling.cubic,
    )

# Create a GeoTIFF file for writing
with rasterio.open(output_tif_file.replace('.tif', '_reproj.tif'), 'w', driver='GTiff',
                height=reprojected_data.shape[0], width=reprojected_data.shape[1], count=1, nodata=0,
                dtype=reprojected_data.dtype, crs=crs, transform=existing_transform, compress='LZW') as dst:
    # Write the reprojected data to the GeoTIFF file
    dst.write(reprojected_data, 1)

sys.exit(0)














# Set up the input and output paths
csv_path = f'Japan/df_multi_japan_ew2.csv'

# Convert the "geometry" column to GeoSeries
df = pd.read_csv(csv_path)
df['geometry'] = gpd.GeoSeries.from_wkt(df['Geometry'])

csv_path = f'Japan/df_single_japan_ew2.csv'

# Convert the "geometry" column to GeoSeries
df_single = pd.read_csv(csv_path)
df_single['geometry'] = gpd.GeoSeries.from_wkt(df_single['Geometry'])

print('MULTI')
print(df['Hazard'].value_counts())
print('SINGLE')
print(df_single['code2'].value_counts())

# Set up the input and output paths
csv_path = f'Japan/df_multi_japan.csv'

# Convert the "geometry" column to GeoSeries
df = pd.read_csv(csv_path)
df['geometry'] = gpd.GeoSeries.from_wkt(df['Geometry'])

csv_path = f'Japan/df_single_japan.csv'

# Convert the "geometry" column to GeoSeries
df_single = pd.read_csv(csv_path)
df_single['geometry'] = gpd.GeoSeries.from_wkt(df_single['Geometry'])

print('WITHOUT EW2')
print('MULTI')
print(df['Hazard'].value_counts())
print('SINGLE')
print(df_single['code2'].value_counts())

sys.exit(0)












