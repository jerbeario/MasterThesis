import rasterio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from PIL import Image


def tif_to_npy(tif_file, npy_file):
    """
    Convert a .tif file to a .npy array.

    Parameters:
    - tif_file: str, path to the input .tif file.
    - npy_file: str, path to save the output .npy file.
    """
    # Open the .tif file
    with rasterio.open(tif_file) as src:
        # Read the data (assuming single-band raster)
        data = src.read(1)  # Read the first band
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

    # Save the data as a .npy file
    np.save(npy_file, data)
    print(f"Saved .npy file to: {npy_file}")

def plot_npy_arrays(npy_file, name, title = "", partition_map=False, debug_nans=False, log=False, downsample_factor=1, save_path=None):
    """
    Plots the data from npy files on a map with the correct coordinates.

    Parameters:
    npy_files (list): List of file paths to the npy files.
    npy_names (list): List of names corresponding to the npy files.
    extents (list): List of extents (geographical bounds) for each npy file.

    Returns:
    None
    """
    print(f"Plotting {name}...")

    if title == "":
        title = f'European {name} Map'

    # Define the extent for the map (longitude and latitude bounds)
    # This extent corresponds to the geographical bounds of Europe
    extent = (-25.0001389, 45.9998611,  27.0001389, 73.0001389)

    # Create a subplot with PlateCarree projection
    fig, axs = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    if isinstance(npy_file, str):
        npy_data = np.load(npy_file)
        print("File loaded")
    elif isinstance(npy_file, np.ndarray):
        npy_data = npy_file
        print("Array loaded")


    if downsample_factor > 1:
        npy_data = npy_data[::downsample_factor, ::downsample_factor]
        print(f"Downsampled data shape: {npy_data.shape}")
    
    if log:
        npy_data = np.log1p(npy_data)

    if debug_nans:
        # set everything to 0 except for NaNs
        npy_data[~np.isnan(npy_data)] = 0
        npy_data[np.isnan(npy_data)] = 1

    # Normalize the colorbar to the range [0, 1] (adjust as needed)
    # norm = Normalize(vmin=0, vmax=np.max(npy_data))

    # Plot the data on the subplot grid
    im = axs.imshow(npy_data, cmap='viridis', extent=extent)
    print("image created")

    # Set title for each subplot
    axs.set_title(title, fontsize=16)

    # Set longitude tick labels
    axs.set_xticks(np.arange(extent[0], extent[1] + 1, 5), crs=ccrs.PlateCarree())
    axs.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

    # Set latitude tick labels
    axs.set_yticks(np.arange(extent[2], extent[3] + 1, 5), crs=ccrs.PlateCarree())
    axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

    axs.set_xlabel('Longitude')
    axs.set_ylabel('Latitude')

    # Add coastlines and country borders
    axs.add_feature(cfeature.COASTLINE, linewidth=0.1, edgecolor='black')
    axs.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.1)
    axs.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
    axs.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')


    # # crop the map to the extent
    # croped_extent = (-5, 10, 40, 52)
    # axs.set_extent(croped_extent, crs=ccrs.PlateCarree())


    # Adjust layout for better spacing
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05)   

    # Add a colorbar for all subplots
    if not partition_map:
        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label(f'{name}', fontsize=16)
        if log:
            cbar.set_label(f'log {name}', fontsize=16)

        cbar.ax.tick_params(labelsize=12) 
    else:
        labels = ['Ignored', 'Train', 'Validation', 'Test']
        colors = plt.cm.viridis(np.linspace(0, 1, 4)) 
        patches =[mpatches.Patch(color=color,label=labels[i]) for i, color in enumerate(colors)]
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
        fancybox=True, ncol=4)



    # # Set the colorbar ticks and labels
    # cbar.set_ticks(np.arange(0, 1.1, 0.1))  # Ticks from 0 to 1 with 0.1 increments
    # cbar.set_ticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.1)])



    # Save the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    
    plt.close(fig)  # Close the figure to free up memory

def save_full_resolution_plot(npy_file, npy_name):
    if isinstance(npy_file, str):
        npy_data = np.load(npy_file)
        print("File loaded")
    elif isinstance(npy_file, np.ndarray):
        npy_data = npy_file
        print("Array loaded")
    
    image = Image.fromarray(npy_data)
    image.save(f"{npy_name}_full_resolution.jpeg")
    




            



