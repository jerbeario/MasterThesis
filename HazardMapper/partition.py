from scipy.ndimage import binary_erosion

import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import time
from multiprocessing import Pool
import os

def create_partition_map(region: np.ndarray, mask: np.ndarray, hazard: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Randomly assigns only regions with hazard occurrences to train (1), val (2), or test (3) partitions
    with proportions 70%, 15%, 15%. Outputs a partition map with the same shape.
    
    Args:
        region (np.ndarray): Array with unique identifiers for each country/region
        mask (np.ndarray): Mask to exclude areas (e.g., ocean)
        hazard (np.ndarray): Binary hazard occurrence map (1 = hazard present, 0 = no hazard)
        seed (int): Random seed for reproducibility
        
    Returns:
        np.ndarray: Partition map with values 0 (ignored), 1 (train), 2 (val), 3 (test)
    """
    print("Creating partition map with hazard filtering...")
    np.random.seed(seed)
    

    partition_map = np.zeros_like(region, dtype=np.uint8)
    hazard_mask = (hazard > 0)
    
    countries_with_hazards = np.unique(region[hazard_mask])
    countries_with_hazards = countries_with_hazards[countries_with_hazards != 0]
    
    print(f"Found {len(countries_with_hazards)} countries/regions with hazard occurrences")
    
    # Only partition countries that have hazards
    if len(countries_with_hazards) > 0:
        np.random.shuffle(countries_with_hazards)
        
        n = len(countries_with_hazards)
        train_ids = countries_with_hazards[:int(0.7 * n)]
        val_ids = countries_with_hazards[int(0.7 * n):int(0.85 * n)]
        test_ids = countries_with_hazards[int(0.85 * n):]
        
        print(f"Train: {len(train_ids)} countries, Val: {len(val_ids)} countries, Test: {len(test_ids)} countries")
        
        # Assign partition values
        partition_map[np.isin(region, train_ids)] = 1
        partition_map[np.isin(region, val_ids)] = 2
        partition_map[np.isin(region, test_ids)] = 3
    else:
        print("Warning: No countries found with hazard occurrences!")
    
    # Apply the mask to the partition map
    mask_array = np.isnan(mask)
    partition_map[mask_array] = 0
    
    # Print statistics
    print(f"Final partition counts: Train: {np.sum(partition_map == 1)}, "
          f"Val: {np.sum(partition_map == 2)}, Test: {np.sum(partition_map == 3)}")
    
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
        eroded_map[border_mask] = 0
    return eroded_map

def balance_partition_map(partition_map: np.ndarray, labels: np.ndarray, countries: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    - Balances the training set (1) by downsampling negative examples to match the number of positives.
    - Reduces the size of validation (2) and test (3) sets to 15% of the training size each,
      while maintaining the original positive/negative ratio.

    Args:
        partition_map (np.ndarray): Partition map with values 1 (train), 2 (val), 3 (test).
        labels (np.ndarray): Binary labels (1 = positive, 0 = negative).
        countries (np.ndarray): Country codes for each pixel.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Balanced partition map.
    """
    np.random.seed(seed)
    balanced_map = partition_map

    # 1. Process the training set (1) - Balance positive and negative samples
    for split in [1]:
        print(f"Processing split {split} (Training)...")
        country_ids = np.unique(countries[partition_map == split])

        for cid in country_ids:
            mask = (partition_map == split) & (countries == cid)
            pos_inds = np.argwhere(mask & (labels == 1))
            neg_inds = np.argwhere(mask & (labels == 0))
            
            if len(pos_inds) == 0 or len(neg_inds) == 0:
                # No positives or no negatives, skip
                balanced_map[mask & (labels == 0)] = 0
                continue

            # Downsample negatives to match number of positives
            if len(neg_inds) > len(pos_inds):
                selected_neg_inds = neg_inds[np.random.choice(len(neg_inds), size=len(pos_inds), replace=False)]
                # Create masks
                all_neg_mask = np.zeros_like(mask, dtype=bool)
                all_neg_mask[tuple(neg_inds.T)] = True
                keep_mask = np.zeros_like(mask, dtype=bool)
                keep_mask[tuple(selected_neg_inds.T)] = True
                discard_mask = all_neg_mask & ~keep_mask
                balanced_map[discard_mask] = 0

    # 2. Determine target sizes for val/test (15% of training size each)
    n_train = np.sum(balanced_map == 1)
    n_total = int(n_train/0.7)
    target_val_test_size = int(n_total * 0.15)

    # 3. Process validation (2) and test (3) sets
    for split in [2, 3]:
        print(f"Processing split {split}...")
        indices = np.argwhere(balanced_map == split)

        # Separate positives and negatives
        pos_inds = indices[labels[indices[:, 0], indices[:, 1]] == 1]
        neg_inds = indices[labels[indices[:, 0], indices[:, 1]] == 0]

        # Calculate desired sample sizes while maintaining the original ratio
        total_samples = len(pos_inds) + len(neg_inds)
        pos_ratio = len(pos_inds) / total_samples if total_samples > 0 else 0
        num_pos = int(target_val_test_size * pos_ratio)
        num_neg = target_val_test_size - num_pos

        # Ensure we don't exceed available samples
        num_pos = min(num_pos, len(pos_inds))
        num_neg = min(num_neg, len(neg_inds))

        # Random sampling
        selected_pos_inds = pos_inds[np.random.choice(len(pos_inds), size=num_pos, replace=False)]
        selected_neg_inds = neg_inds[np.random.choice(len(neg_inds), size=num_neg, replace=False)]

        # Create discard mask
        all_pos_mask = np.zeros_like(balanced_map, dtype=bool)
        all_neg_mask = np.zeros_like(balanced_map, dtype=bool)
        all_pos_mask[tuple(pos_inds.T)] = True
        all_neg_mask[tuple(neg_inds.T)] = True

        keep_pos_mask = np.zeros_like(balanced_map, dtype=bool)
        keep_neg_mask = np.zeros_like(balanced_map, dtype=bool)
        keep_pos_mask[tuple(selected_pos_inds.T)] = True
        keep_neg_mask[tuple(selected_neg_inds.T)] = True

        discard_pos_mask = all_pos_mask & ~keep_pos_mask
        discard_neg_mask = all_neg_mask & ~keep_neg_mask

        # Apply discard mask
        balanced_map[discard_pos_mask | discard_neg_mask] = 0

    print(f"Final counts - Train: {np.sum(balanced_map == 1)}, Val: {np.sum(balanced_map == 2)}, Test: {np.sum(balanced_map == 3)}")

    return balanced_map

# def plot_npy_arrays(npy_files, npy_names, partition_map=False, debug_nans=False, log=False, downsample_factor=1, save_path=None):
#     """
#     Plots the data from npy files on a map with the correct coordinates.

#     Parameters:
#     npy_files (list): List of file paths to the npy files.
#     npy_names (list): List of names corresponding to the npy files.
#     extents (list): List of extents (geographical bounds) for each npy file.

#     Returns:
#     None
#     """
#     extent = (-25.0001389, 45.9998611,  27.0001389, 73.0001389)

#     if isinstance(npy_files, np.ndarray):
#         npy_files = [npy_files]
#         npy_names = [npy_names]
    
#     for i, npy_file in enumerate(npy_files):

#         name = npy_names[i]

#         # Create a subplot with PlateCarree projection
#         fig, axs = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

#         print(npy_names[i])

#         if isinstance(npy_file, str):
#             npy_data = np.load(npy_file)
#             print("File loaded")
#         elif isinstance(npy_file, np.ndarray):
#             npy_data = npy_file
#             print("Array loaded")


#         if downsample_factor > 1:
#             npy_data = npy_data[::downsample_factor, ::downsample_factor]
#             print(f"Downsampled data shape: {npy_data.shape}")
        
#         if log:
#             npy_data = np.log1p(npy_data)
      
#         if debug_nans:
#             # set everything to 0 except for NaNs
#             npy_data[~np.isnan(npy_data)] = 0
#             npy_data[np.isnan(npy_data)] = 1

#         # Normalize the colorbar to the range [0, 1] (adjust as needed)
#         # norm = Normalize(vmin=0, vmax=np.max(npy_data))

#         # Plot the data on the subplot grid
#         im = axs.imshow(npy_data, cmap='viridis', extent=extent)
#         print("image created")

#         # Set title for each subplot
#         axs.set_title(f'European {name} Map', fontsize=16)

#         # Set longitude tick labels
#         axs.set_xticks(np.arange(extent[0], extent[1] + 1, 5), crs=ccrs.PlateCarree())
#         axs.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

#         # Set latitude tick labels
#         axs.set_yticks(np.arange(extent[2], extent[3] + 1, 5), crs=ccrs.PlateCarree())
#         axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

#         axs.set_xlabel('Longitude')
#         axs.set_ylabel('Latitude')

#         # Add coastlines and country borders
#         axs.add_feature(cfeature.COASTLINE, linewidth=0.1, edgecolor='black')
#         axs.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.1)
#         axs.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
#         axs.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')


#         # # crop the map to the extent
#         # croped_extent = (-5, 10, 40, 52)
#         # axs.set_extent(croped_extent, crs=ccrs.PlateCarree())


#         # Adjust layout for better spacing
#         plt.tight_layout()
#         # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05)   
    
#         # Add a colorbar for all subplots
#         if not partition_map:
#             cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
#             cbar.set_label(f'{name} counts', fontsize=16)
#             if log:
#                 cbar.set_label(f'log {name} counts', fontsize=16)

#             cbar.ax.tick_params(labelsize=12) 
#         else:
#             labels = ['Ignored', 'Train', 'Validation', 'Test']
#             colors = plt.cm.viridis(np.linspace(0, 1, 4)) 
#             patches =[mpatches.Patch(color=color,label=labels[i]) for i, color in enumerate(colors)]
#             fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
#             fancybox=True, ncol=4)



#         # # Set the colorbar ticks and labels
#         # cbar.set_ticks(np.arange(0, 1.1, 0.1))  # Ticks from 0 to 1 with 0.1 increments
#         # cbar.set_ticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.1)])

 

#         # Save the plot
#         if save_path is not None:
#             plt.savefig(save_path, dpi=1000, bbox_inches='tight')


if __name__ == "__main__":
    
    
    seed = 42
    # input hazard name
    hazard_name = sys.argv[1]


    # countries = np.load("Input/Europe/partition_map/countries_rasterized.npy")
    sub_countries = np.load("Input/Europe/partition_map/sub_countries_rasterized.npy")
    hazard = np.load(f"Input/Europe/npy_arrays/masked_{hazard_name}_Europe.npy")
    elevation = np.load("Input/Europe/npy_arrays/masked_elevation_Europe.npy")
    
    hazard[hazard > 0] = 1



    partition_map = create_partition_map(region=sub_countries, mask=elevation, hazard=hazard, seed=seed)
    partition_map = erode_partition_borders(partition_map, kernel_size=5)
    # partition_map = balance_partition_map(partition_map, wildfires, sub_countries, seed=seed)
    
    

    np.save(f"Input/Europe/partition_map/{hazard_name}_partition_map.npy", partition_map)
    # plot_npy_arrays(partition_map, f"{hazard_name} Partition", partition_map=True, downsample_factor=10, save_path=f"Input/Europe/partition_map/{hazard_name}_partition_map.png")


   

