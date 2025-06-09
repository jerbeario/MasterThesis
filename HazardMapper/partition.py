""" HazardMapper - Partition Map Creation
=========================================
This module provides functions to create and manage partition maps for hazard datasets, including
creating partition maps based on hazard occurrences, cleaning partition maps, eroding borders to prevent data leakage, balancing partitions and downsampling to a given size.
"""

from scipy.ndimage import binary_erosion

import numpy as np
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import time
from multiprocessing import Pool
import os

from HazardMapper.utils import plot_npy_arrays

def create_partition_map(region: np.ndarray, mask: np.ndarray, hazard: np.ndarray, seed: int = 42, train_ratio = 0.8) -> np.ndarray:
    """
    Randomly assigns only regions with hazard occurrences to train (1), val (2), or test (3) partitions
    with proportions train_ratio, (1-train_ration)/2, (1-train_ration)/2 . Outputs a partition map with the same shape.
    
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

    val_ratio = (1-train_ratio)/2

    partition_map = np.zeros_like(region, dtype=np.uint8)
    hazard_mask = (hazard > 0)
    
    countries_with_hazards = np.unique(region[hazard_mask])
    countries_with_hazards = countries_with_hazards[countries_with_hazards != 0]
    
    print(f"Found {len(countries_with_hazards)} regions with hazard occurrences")
    
    # Only partition countries that have hazards
    if len(countries_with_hazards) > 0:
        np.random.shuffle(countries_with_hazards)
        
        n = len(countries_with_hazards)
        train_ids = countries_with_hazards[:int(train_ratio * n)]
        val_ids = countries_with_hazards[int(train_ratio * n):int((train_ratio+val_ratio) * n)]
        test_ids = countries_with_hazards[int((train_ratio+val_ratio) * n):]
        
        print(f"Train: {len(train_ids)} regions, Val: {len(val_ids)} regions, Test: {len(test_ids)} regions")
        
        # Assign partition values
        partition_map[np.isin(region, train_ids)] = 1
        partition_map[np.isin(region, val_ids)] = 2
        partition_map[np.isin(region, test_ids)] = 3
    else:
        print("Warning: No regions found with hazard occurrences!")
    
    # Apply the mask to the partition map
    mask_array = np.isnan(mask)
    partition_map[mask_array] = 0

    # If flood hazard, remove water types from landcover
    if hazard_name == "floods":
        landcover = np.load("Input/Europe/npy_arrays/masked_landcover_Europe_flat.npy")
        water_types = [210]  # Example water types in landcover
        water_mask = np.isin(landcover, water_types)
        partition_map[water_mask] = 0  # Set water areas to 0 (ignored)
    
    # Print statistics
    print(f"Initial partition counts: Train: {np.sum(partition_map == 1)}, "
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
import numpy as np

def balance_partition_map(partition_map: np.ndarray, labels: np.ndarray, regions: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Balances each split (train/val/test) by downsampling negative (label=0) cells 
    to match the number of positive (label=1) cells within each region.

    Args:
        partition_map (np.ndarray): 2D array (H, W) with values 0=ignore, 1=train, 2=val, 3=test
        labels (np.ndarray): 2D array (H, W) with values 0=no hazard, 1=hazard
        regions (np.ndarray): 2D array (H, W) with region IDs per pixel
        seed (int): Random seed

    Returns:
        np.ndarray: Balanced partition map with same shape
    """
    rng = np.random.default_rng(seed)

    flat_partition = partition_map.ravel()
    flat_labels = labels.ravel()
    flat_regions = regions.ravel()
    balanced_map = flat_partition.copy()

    for split in [1, 2, 3]:
        print(f"Balancing split {split}...")
        split_mask = flat_partition == split
        region_ids = np.unique(flat_regions[split_mask])

        for rid in region_ids:
            region_mask = split_mask & (flat_regions == rid)

            region_indices = np.flatnonzero(region_mask)
            region_labels = flat_labels[region_indices]

            pos_indices = region_indices[region_labels == 1]
            neg_indices = region_indices[region_labels == 0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                # No balancing possible, skip
                balanced_map[neg_indices] = 0  # discard all negatives if no positives
                continue

            if len(neg_indices) > len(pos_indices):
                discard_count = len(neg_indices) - len(pos_indices)
                discard_indices = rng.choice(neg_indices, size=discard_count, replace=False)
                balanced_map[discard_indices] = 0  # set to ignore

    # Logging final stats
    for split in [1, 2, 3]:
        mask = balanced_map == split
        count = np.bincount(flat_labels[mask].astype(int), minlength=2)
        print(f"Split {split}: Positives = {count[1]}, Negatives = {count[0]}")

    return balanced_map.reshape(partition_map.shape)


def sample_partition_map(partition_map: np.ndarray, size = 1, seed: int = 42) -> np.ndarray:
    """
    Samples a partition map by downsampling the total dataset size to size .
    
    Args:
        partition_map (np.ndarray): Array with partition values (1=train, 2=val, 3=test)
        size (int): Desired number of samples to return
        seed (int): Random seed for reproducibility
    
    Returns:
        np.ndarray: Sampled partition map with same shape
    """
    print("Sampling partition map...")
    np.random.seed(seed)
    
    # Flatten the partition map and select dataset cells
    flat_partition = partition_map.ravel()
    dataset = (partition_map == 1) | (partition_map == 2) | (partition_map == 3)
    total_size = np.sum(dataset)

    if size > total_size:
        print(f"Warning: Requested size {size} exceeds total dataset size {total_size}. Returning full dataset.")
        return partition_map
    
    elif size <= 0:
        print("Warning: Requested size is 0 or negative. Returning empty partition map.")
        return np.zeros_like(partition_map)
    else:
        # Randomly sample indices
        sampled_indices = np.random.choice(np.flatnonzero(dataset), size=size, replace=False)
        
        # Create partition map where kept training are 1, val are 2, and test are 3
        sampled_partition_map = np.zeros_like(flat_partition, dtype=np.uint8)
        sampled_partition_map[sampled_indices] = flat_partition[sampled_indices]

        # Reshape back to original shape
        sampled_partition_map = sampled_partition_map.reshape(partition_map.shape)

        return sampled_partition_map
    


if __name__ == "__main__":

    argparse = argparse.ArgumentParser(description="Create partition map for a specific hazard in Europe.")
    argparse.add_argument("-z", "--hazard", type=str, help="Name of the hazard to create partition map for (e.g., floods, wildfires, landlside).")
    argparse.add_argument("-n", "--n_samples", type=int, default=1_000_000, help="Number of samples to downsample the partition map to.")
    args = argparse.parse_args()
    
    
    seed = 42
    # input hazard name
    hazard_name = args.hazard
    n_samples = args.n_samples

    # countries = np.load("Input/Europe/partition_map/countries_rasterized.npy")
    sub_countries = np.load("Input/Europe/partition_map/sub_countries_rasterized.npy")
    hazard = np.load(f"Input/Europe/npy_arrays/masked_{hazard_name}_Europe.npy")
    elevation = np.load("Input/Europe/npy_arrays/masked_elevation_Europe.npy")
    
    hazard[hazard > 0] = 1


    partition_map = create_partition_map(region=sub_countries, mask=elevation, hazard=hazard, seed=seed)
    partition_map = erode_partition_borders(partition_map, kernel_size=5)
    partition_map = balance_partition_map(partition_map, hazard, sub_countries, seed=seed)
    partition_map = sample_partition_map(partition_map, size=n_samples, seed=seed)


    np.save(f"Input/Europe/partition_map/balanced_{hazard_name}_partition_map.npy", partition_map)
    plot_npy_arrays(partition_map, name="partition",title=f"{hazard_name.capitalize()} Partition Map", type="partition", downsample_factor=10, save_path=f"Input/Europe/partition_map/balanced_{hazard_name}_partition_map.png")


   

