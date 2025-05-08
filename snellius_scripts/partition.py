from scipy.ndimage import binary_erosion

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_partition_map(region: np.ndarray, mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Randomly assigns entire region to train (1), val (2), or test (3) partitions
    with proportions 70%, 15%, 15%. Outputs a partition map with the same shape.
    """
    print("Creating partition map...")
    np.random.seed(seed)
    unique_countries = np.unique(region)
    unique_countries = unique_countries[unique_countries != 0]  # Exclude the background (0)
    np.random.shuffle(unique_countries)
    
    n = len(unique_countries)
    train_ids = unique_countries[:int(0.7 * n)]
    val_ids = unique_countries[int(0.7 * n):int(0.85 * n)]
    test_ids = unique_countries[int(0.85 * n):]

    partition_map = np.zeros_like(region, dtype=np.uint8)
    partition_map[np.isin(region, train_ids)] = 1
    partition_map[np.isin(region, val_ids)] = 2
    partition_map[np.isin(region, test_ids)] = 3

    # Apply the mask to the partition map
    mask = np.isnan(mask)
    partition_map[mask] = 0

    return partition_map

def clean_partition_map(partition_map: np.ndarray, countries: np.ndarray, countries_to_remove: np.ndarray) -> np.ndarray:
    """
    Removes specified countries from the partition map by setting their cells to 0.
    """
    print("Cleaning partition map...")
    cleaned_map = partition_map.copy()
    mask = np.isin(countries, countries_to_remove)
    cleaned_map[mask] = 0
    return cleaned_map

def erode_partition_borders(partition_map: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Erodes the borders between partition regions to avoid leakage during patch sampling.
    Returns a new map where border-adjacent cells are set to 0.
    
    Args:
        partition_map (np.ndarray): Array with values 1 (train), 2 (val), 3 (test)
        kernel_size (int): Size of patch to be used for erosion. Default is 5.)
    
    Returns:
        np.ndarray: Eroded partition map with border cells removed (set to 0)
    """
    print("Eroding partition borders...")
    eroded_map = np.zeros_like(partition_map, dtype=np.uint8)+1
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    ocean_mask = (partition_map == 0)
    for label in [1, 2, 3]:
        print(f"Eroding label {label}...")
        region_mask = ((partition_map == label) | (partition_map == 0))
        eroded_mask = binary_erosion(region_mask, structure=structure)
        eroded_map[eroded_mask] = label
    eroded_map[ocean_mask] = 0
    # Set the borders to 0
    for label in [1, 2, 3]:
        border_mask = (partition_map == label) & (eroded_map != label)
        eroded_map[border_mask] = 4
    return eroded_map

def balance_partition_map(partition_map: np.ndarray, labels: np.ndarray, countries: np.ndarray, splits = [1], seed: int = 42) -> np.ndarray:
    """
    Downsamples negative examples within each country to match the number of positive examples.
    Outputs a new partition map where excess negatives are set to 0 (ignored).
    """
    print("Balancing partition map...")
    np.random.seed(seed)
    balanced_map = partition_map.copy()
    
    for split in splits:
        print(f"Processing split {split}...")
        country_ids = np.unique(countries[partition_map == split])
        
        for cid in country_ids:
            print(f"Processing country {cid}...")
            mask = (partition_map == split) & (countries == cid)
            pos_inds = np.argwhere(mask & (labels == 1))
            neg_inds = np.argwhere(mask & (labels == 0))
            
            if len(pos_inds) == 0 or len(neg_inds) == 0:
                # If no positives or no negatives, skip this region
                balanced_map[mask & (labels == 0)] = 0
                continue

            # Downsample negatives to match number of positives
            if len(neg_inds) > len(pos_inds):
                selected_neg_inds = neg_inds[np.random.choice(len(neg_inds), size=len(pos_inds), replace=False)]
                all_neg_mask = np.zeros_like(mask, dtype=bool)
                all_neg_mask[tuple(neg_inds.T)] = True
                keep_mask = np.zeros_like(mask, dtype=bool)
                keep_mask[tuple(selected_neg_inds.T)] = True
                discard_mask = all_neg_mask & ~keep_mask
                balanced_map[discard_mask] = 0

    return balanced_map

def plot_npy_arrays(npy_files, npy_names, partition_map=False, debug_nans=False, log=False, downsample_factor=1, save_path=None):
    """
    Plots the data from npy files on a map with the correct coordinates.

    Parameters:
    npy_files (list): List of file paths to the npy files.
    npy_names (list): List of names corresponding to the npy files.
    extents (list): List of extents (geographical bounds) for each npy file.

    Returns:
    None
    """
    extent = (-25.0001389, 45.9998611,  27.0001389, 73.0001389)

    if isinstance(npy_files, np.ndarray):
        npy_files = [npy_files]
        npy_names = [npy_names]
    
    for i, npy_file in enumerate(npy_files):

        name = npy_names[i]

        # Create a subplot with PlateCarree projection
        fig, axs = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        print(npy_names[i])

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
        axs.set_title(f'European {name} Map', fontsize=16)

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
            cbar.set_label(f'{name} counts', fontsize=16)
            if log:
                cbar.set_label(f'log {name} counts', fontsize=16)

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

if __name__ == "__main__":

    seed = 42
    # countries = np.load("Input/Europe/partition_map/countries_rasterized.npy")
    sub_countries = np.load("Input/Europe/partition_map/sub_countries_rasterized.npy")
    wildfires = np.load("Input/Europe/npy_arrays/masked_wildfire_Europe.npy")
    elevation = np.load("Input/Europe/npy_arrays/masked_elevation_Europe.npy")
    
    wildfires[wildfires > 0] = 1

    partition_map = create_partition_map(sub_countries, elevation, seed=seed)
    eroded_map = erode_partition_borders(partition_map, kernel_size=5)
    full_balanced_partition_map = balance_partition_map(eroded_map, wildfires, sub_countries, seed=seed)

    np.save("Input/Europe/partition_map/final_partition_map.npy", full_balanced_partition_map)
    plot_npy_arrays(full_balanced_partition_map, "Partition", partition_map=True, downsample_factor=1, save_path="Input/Europe/partition_map/final_partition_map.png")



