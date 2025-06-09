""" HazardMapper - Utilities module.
==========================
This module provides utility functions for handling raster data, plotting numpy arrays on maps,
and normalizing numpy arrays. It handles some of the feature preprocessing such as normalization.

"""
import rasterio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from PIL import Image
from sklearn.preprocessing import MinMaxScaler



def tif_to_npy(tif_file, npy_file):
    """
    Convert a .tif file to a .npy array.

    Parameters:
    - tif_file: str, path to the input .tif file.
    - npy_file: str, path to save the output .npy file.
    """
    # Open the .tif file
    with rasterio.open(tif_file) as src:
        # read how many bands are in the raster
        num_bands = src.count
        print(f"Number of bands in the raster: {num_bands}")
        # Read the data (assuming single-band raster)
        data = src.read(1)  # Read the first band
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

    # Save the data as a .npy file
    np.save(npy_file, data)
    print(f"Saved .npy file to: {npy_file}")



def plot_npy_arrays(npy_file, name, type, title = "", debug_nans=False, log=False, downsample_factor=10, save_path=None, labels=None, extent=None):
    """
    Plots the data from npy files on a map with the correct coordinates.

    Parameters:
    npy_file: str or np.ndarray
        Path to the .npy file or a numpy array.
    name: str
        Name of the data being plotted (e.g., 'Susceptibility', 'Hazard').
    type: str       
        Type of data being plotted ('continuous', 'partition', 'bins').
    title: str, optional    
        Title of the plot. If empty, defaults to 'European {name} Map'.     
    debug_nans: bool, optional
        If True, sets all non-NaN values to 0 and NaN values to 1 for debugging purposes.
    log: bool, optional
        If True, applies a logarithmic transformation to the data.
    downsample_factor: int, optional
        Factor by which to downsample the data. Default is 10 (10x downsampling).
    save_path: str, optional
        Path to save the plot. If None, the plot will not be saved.

    Returns:
    None
    """
    print(f"Plotting {name}...")

    if title == "":
        title = f'European {name} Map'
    else:
        title = title.capitalize()

    cmap = 'viridis'  # Default colormap, can be changed if needed

    # Define the extent for the map (longitude and latitude bounds)
    # This extent corresponds to the geographical bounds of Europe
    if extent is None:
        # Default extent for Europe
        # Longitude: -25 to 45, Latitude: 27 to 73
        # This can be adjusted based on the specific area of interest
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


    # Plot the data on the subplot grid
    im = axs.imshow(npy_data, cmap=cmap, extent=extent)
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

    # Adjust layout for better spacing
    plt.tight_layout()


    # Add a colorbar for all subplots
    if type == 'continuous':
        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label(f'{name}', fontsize=16)
        if log:
            cbar.set_label(f'log {name}', fontsize=16)
        cbar.ax.tick_params(labelsize=12) 
    elif type == 'partition':
        labels = ['Ignored', 'Train', 'Validation', 'Test']
        colors = plt.cm.viridis(np.linspace(0, 1, 4)) 
        patches =[mpatches.Patch(color=color,label=labels[i]) for i, color in enumerate(colors)]
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
        fancybox=True, ncol=4)
    elif type == 'bins':
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        patches = [mpatches.Patch(color=color, label=labels[i]) for i, color in enumerate(colors)]
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
                    fancybox=True, ncol=5)
    elif type == 'categorical':
        if labels is None:
            categories = np.unique(npy_data)
            colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
            patches = [mpatches.Patch(color=color, label=str(category)) for category, color in zip(categories, colors)]
            
        else: 
            colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
            patches = [mpatches.Patch(color=color, label=labels[i]) for i, color in enumerate(colors)]
            fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
                        fancybox=True, ncol=len(labels))
    else:
        raise ValueError("Invalid type. Choose from 'continuous', 'partition', 'bins', or 'categorical'.")

    # Save the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

    # plt.show()  # Show the plot

def save_full_resolution_plot(npy_file, npy_name):
    """
    Work in progress, not yet functional.
    Saves the full resolution plot of a numpy array as a JPEG image.
    """

    if isinstance(npy_file, str):
        npy_data = np.load(npy_file)
        print("File loaded")
    elif isinstance(npy_file, np.ndarray):
        npy_data = npy_file
        print("Array loaded")
    
    image = Image.fromarray(npy_data)
    image.save(f"{npy_name}_full_resolution.jpeg")

def normalize_npy(npy_file_path: str, output_file_path: str) -> None:
    """
    Normalizes the values in a .npy file to the range [0, 1] using MinMaxScaler.
    Saves the normalized array to the specified output path.

    Args:
        npy_file_path (str): Path to the input .npy file.
        output_file_path (str): Path to save the normalized .npy file.
    """
    # Load the array
    data = np.load(npy_file_path)

    # Reshape for MinMaxScaler
    reshaped_data = data.reshape(-1, 1)

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(reshaped_data)

    # Reshape back to original shape
    normalized_data = normalized_data.reshape(data.shape)

    # Save the normalized array
    np.save(output_file_path, normalized_data)
    


if __name__ == "__main__":
    paths_to_normalize = [
         "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_drought_Europe.npy",
         "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_heatwave_Europe.npy",
         "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_extreme_wind_Europe.npy",
         "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_volcano_Europe.npy",
    ]

    for npy_file in paths_to_normalize:
        output_file = npy_file.replace('masked', 'normalized_masked')
        normalize_npy(npy_file, output_file)
        print(f"Normalized {npy_file} and saved to {output_file}")





