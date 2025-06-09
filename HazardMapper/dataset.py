""" HazardMapper - Dataset Module 
========================
This module defines a custom dataset for loading hazard-specific features and labels as patches.
The paths to the variables and labels are defined in dictionaries, and the dataset can be used with PyTorch's DataLoader for training models.

"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler 

var_paths = {
    # Conditioning Factors
    "soil_moisture_root" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_soil_moisture_root_Europe.npy",
    "soil_moisture_surface" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_soil_moisture_surface_Europe.npy",
    "NDVI" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_NDVI_Europe_flat.npy",
    "landcover" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_landcover_Europe_flat.npy",
    "wind_direction_daily" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_wind_direction_daily_Europe.npy",
    "wind_speed_daily" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_wind_speed_daily_Europe.npy",
    "temperature_daily" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_temperature_daily_Europe.npy",
    "precipitation_daily" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_precipitation_daily_Europe.npy",
    "fire_weather" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_fire_weather_Europe.npy",
    "HWSD" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_HWSD_Europe.npy",
    "pga" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_pga_Europe.npy",
    "accuflux" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_accuflux_Europe.npy",
    "coastlines" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_coastlines_Europe.npy",
    "rivers" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_rivers_Europe.npy",
    "slope" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_slope_Europe.npy",
    "strahler" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_strahler_Europe.npy",
    "GLIM" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_GLIM_Europe.npy",
    "GEM" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_GEM_Europe.npy",
    "aspect" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_aspect_Europe.npy",
    "elevation" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_elevation_Europe.npy",
    "curvature" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_curvature_Europe.npy",
    "test": "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_fire_weather_Europe.npy",

    # Hazard Maps
    "drought" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_drought_Europe.npy",
    "heatwave" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_heatwave_Europe.npy",
    "extreme_wind" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_extreme_wind_Europe.npy",
    "volcano" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_masked_volcano_Europe.npy",
    "wildfire" : "/projects/FWC2/MYRIAD/Susceptibility/Output/Europe/wildfire/hazard_map/wildfire_hazard_map.npy",
    "flood" : "/projects/FWC2/MYRIAD/Susceptibility/Output/Europe/flood/hazard_map/flood_hazard_map.npy",
    "landslide" : "/projects/FWC2/MYRIAD/Susceptibility/Output/Europe/landslide/hazard_map/landslide_hazard_map.npy",
   \
}

label_paths = {
    # Hazard Inventories
    "test": "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_landslide_Europe.npy",
    "wildfire" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wildfire_Europe.npy",
    "flood" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_flood_Europe.npy",
    "landslide" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_landslide_Europe.npy",
    "multi_hazard" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_multi_hazard_Europe.npy",
}

# This class is finished and ready to be used
class HazardDataset(Dataset):
    def __init__(self, hazard, variables, patch_size=5):
        """
        Custom Dataset for loading hazard-specific features and labels as patches.

        Parameters:
        - hazard (str): The hazard type (e.g., "wildfire").
        - patch_size (int): The size of the patch (n x n) around the center cell.
        """
        self.hazard = hazard.lower()
        self.patch_size = patch_size
        self.variables = variables
        self.num_vars = len(variables)
     
        # Check if the hazard is valid
        if self.hazard not in label_paths.keys():
            raise ValueError(f"Hazard '{self.hazard}' is not defined in the dataset.")
        # Check if the variables are valid
        for variable in variables:
            if variable not in var_paths.keys():
                raise ValueError(f"Variable '{variable}' is not defined in the dataset.")
    
        # get features and labels for the hazard
        self.feature_paths = [var_paths[variable] for variable in self.variables]
        self.label_path = label_paths[self.hazard]

        # Load features (stacked along the first axis for channels)
        self.features = np.stack([np.load(path) for path in self.feature_paths], axis=0)
        self.features = np.nan_to_num(self.features, nan=0.0)  # Handle NaN values

        # Load labels
        self.labels = np.load(self.label_path)
        self.labels = (self.labels > 0).astype(int)  # Binarize labels

        # Ensure the spatial dimensions match between features and labels
        assert self.features.shape[1:] == self.labels.shape, "Mismatch between features and labels!"

        # Padding to handle edge cases for patches
        self.pad_size = patch_size // 2
        self.features = np.pad(
            self.features,
            pad_width=((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),  # Correct padding for 3D array
            mode='constant',
            constant_values=0
        )
        # self.labels = np.pad(
        #     self.labels,
        #     pad_width=((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),  # Correct padding for 2D array
        #     mode='constant',
        #     constant_values=0
        # )
    def __len__(self):
        """
        Returns the number of samples in the dataset (total number of cells).
        """
        return self.labels.size

    def __getitem__(self, idx):
        """
        Returns a single sample (patch and label) at the given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - patch (torch.Tensor): The n x n patch of features.
        - label (torch.Tensor): The label for the center cell of the patch.
        """

        # Convert 1D index to 2D spatial index
        h, w = self.labels.shape
        row, col = divmod(idx, w)

        # Extract the patch centered at (row, col)
        
        patch = self.features[:, row:row + self.patch_size, col:col + self.patch_size]
        # Get the label for the center cell
        label = self.labels[row, col]

        # Convert to PyTorch tensors
        patch = torch.tensor(patch, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
         
        patch = patch.view(self.num_vars, self.patch_size, self.patch_size)
        
        return patch, label

# Custom balanced batch sampler
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, neg_ratio=1, replacement=False):
        """
        Args:
            labels (array-like): 1D array of labels (0 or 1).
            batch_size (int): Total size of each batch.
            neg_ratio (int or float): Number of negative samples per positive (default 1:1).
            replacement (bool): Whether to sample with replacement.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.replacement = replacement
        
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        
        # Compute number of positive and negative samples per batch
        # Total batch size = pos_per_batch + neg_per_batch
        self.pos_per_batch = int(self.batch_size / (1 + self.neg_ratio))
        self.neg_per_batch = self.batch_size - self.pos_per_batch
        
        # Check if calculated numbers make sense
        if self.pos_per_batch == 0:
            raise ValueError("Batch size too small for the given neg_ratio.")

    def __iter__(self):
        pos_indices = np.random.choice(
            self.pos_indices, size=len(self.pos_indices), replace=self.replacement
        )
        neg_indices = np.random.choice(
            self.neg_indices, size=len(self.neg_indices), replace=self.replacement
        )
        
        n_pos_batches = len(pos_indices) // self.pos_per_batch
        n_neg_batches = len(neg_indices) // self.neg_per_batch
        n_batches = min(n_pos_batches, n_neg_batches)
        
        for i in range(n_batches):
            pos_start = i * self.pos_per_batch
            neg_start = i * self.neg_per_batch
            
            pos_end = pos_start + self.pos_per_batch
            neg_end = neg_start + self.neg_per_batch
            
            pos_batch = pos_indices[pos_start:pos_end]
            neg_batch = neg_indices[neg_start:neg_end]
            
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        n_pos_batches = len(self.pos_indices) // self.pos_per_batch
        n_neg_batches = len(self.neg_indices) // self.neg_per_batch
        return min(n_pos_batches, n_neg_batches)

def index2d_to_1d(idx):
    """
    Convert a 2D index (or array of indices) to 1D index.

    Parameters:
      idx: A tuple/list of two ints (row, col) or a numpy array of shape (n, 2).
      shape: Tuple of (n_rows, n_cols).

    Returns:
      A single integer if idx is a pair, or a numpy array of shape (n,) if idx is an array.
    """
    shape = (16560, 25560)
    
    arr = np.atleast_2d(idx)
    one_d = arr[:, 0] * shape[1] + arr[:, 1]
    return one_d[0] if one_d.size == 1 else one_d

def index1d_to_2d(idx):
    """
    Convert a 1D index (or an array of indices) to a 2D index for an array with the given shape.
    
    Parameters:
        idx: An integer or an array-like of integers representing indices in the flattened array.

    
    Returns:
        A tuple (row, col) if a single index is provided, or
        a numpy array of shape (n, 2) for multiple indices.
    """
    shape = (16560, 25560)
    
    # Ensure idx is a numpy array
    idx_arr = np.atleast_1d(idx)
    
    rows = idx_arr // shape[1]
    cols = idx_arr % shape[1]
    
    result = np.column_stack((rows, cols))
    
    # If only a single index was provided, return as tuple
    if result.shape[0] == 1:
        return tuple(result[0])
    return result