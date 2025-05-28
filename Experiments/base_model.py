import logging
import logging.handlers
from joblib import dump, load
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os
import random
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, mean_absolute_error, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_curve
    )
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.init as init


import wandb

import seaborn as sns


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

        # Define the feature file paths for each hazard

        self.var_paths = {
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
            "test": "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_fire_weather_Europe.npy"
        }

        self.label_paths = {
            "wildfire" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wildfire_Europe.npy",
            "test" : "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_wildfire_Europe.npy",
            "landslide" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_landslide_Europe.npy",

        }
     
        # Check if the hazard is valid
        if self.hazard not in self.label_paths.keys():
            raise ValueError(f"Hazard '{self.hazard}' is not defined in the dataset.")
        # Check if the variables are valid
        for variable in variables:
            if variable not in self.var_paths.keys():
                raise ValueError(f"Variable '{variable}' is not defined in the dataset.")
    
        # get features and labels for the hazard
        self.feature_paths = [self.var_paths[variable] for variable in self.variables]
        self.label_path = self.label_paths[self.hazard]

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

# MLP to test 1d data 
class MLP(nn.Module):
    def __init__(self, logger, device, num_vars, n_layers=2, n_nodes=128, 
                 dropout=True, drop_value=0.4, patch_size=1):
        """
        MLP architecture designed to take feature vectors as input.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/features
            n_layers: Number of hidden layers
            n_nodes: Number of nodes in each hidden layer
            dropout: Whether to use dropout
            drop_value: Dropout probability
            patch_size: Should be 1 for pure feature vector input
        """
        super(MLP, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        
        # For feature vector input, input size is just num_vars
        # (when patch_size=1, we get features only with no spatial context)
        input_size = num_vars
        if patch_size > 1:
            # If using patches, flatten them
            input_size = num_vars * patch_size * patch_size
            self.logger.warning(f"Using MLP with patch_size={patch_size}. "
                               f"Consider using patch_size=1 for feature vectors.")
        
        # Build the MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, n_nodes))
        layers.append(nn.BatchNorm1d(n_nodes))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(drop_value))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_nodes, n_nodes))
            layers.append(nn.BatchNorm1d(n_nodes))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(drop_value))
        
        # Output layer
        layers.append(nn.Linear(n_nodes, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        self.logger.info(f"Created MLP with {n_layers} layers, {n_nodes} nodes per layer")
        self.logger.info(f"Input size: {input_size}, Output size: 1")

    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape [batch_size, num_vars, patch_size, patch_size]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        # Reshape input based on whether it's a single feature vector or patches
        batch_size = x.size(0)
        
        if x.size(2) == 1 and x.size(3) == 1:
            # We have feature vectors [batch_size, num_vars, 1, 1]
            # Reshape to [batch_size, num_vars]
            x = x.view(batch_size, self.num_vars)
        else:
            # We have patches, so flatten them
            x = x.view(batch_size, -1)
        
        # Forward pass through the model
        return self.model(x)

# Simple CNN architecture to test the pipeline
class SimpleCNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters=16, dropout=False, 
                 drop_value=0.2, patch_size=5):
        """
        Simple CNN architecture for hazard susceptibility modeling.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/channels
            filters: Number of filters in each convolution
            dropout: Whether to use dropout
            drop_value: Dropout probability
            patch_size: Size of the input neighborhood
        """
        super(SimpleCNN, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        
        # Process each variable with a single conv layer
        self.feature_extractors = nn.ModuleList([
            nn.Conv2d(1, filters, kernel_size=3, padding=1)
            for _ in range(num_vars)
        ])
        
        # Shared layers after concatenation
        self.conv = nn.Conv2d(filters * num_vars, filters * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate the size after pooling
        pooled_size = patch_size // 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split input by variables
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        
        # Extract features from each variable
        features = []
        for i, (extractor, var_input) in enumerate(zip(self.feature_extractors, var_inputs)):
            features.append(extractor(var_input))
        
        # Concatenate all features
        x = torch.cat(features, dim=1)
        
        # Shared processing
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

# UNet architecture for hazard susceptibility modeling
# class UNet(nn.Module):
#     def __init__(self, logger, device, num_vars, filters=64, n_layers=4, activation=nn.ReLU(), 
#                  dropout=True, drop_value=0.3, kernel_size=3, pool_size=2, patch_size=5):
#         """
#         UNet architecture for hazard susceptibility modeling.
        
#         Args:
#             logger: Logger instance
#             device: Torch device
#             num_vars: Number of input variables/channels
#             filters: Base number of filters (will be doubled in each layer)
#             n_layers: Number of downsampling/upsampling layers
#             activation: Activation function
#             dropout: Whether to use dropout
#             drop_value: Dropout probability
#             kernel_size: Kernel size for convolutions
#             pool_size: Pooling size for downsampling
#             patch_size: Size of the input neighborhood
#         """
#         super(UNet, self).__init__()
        
#         self.logger = logger
#         self.device = device
#         self.num_vars = num_vars
#         self.filters = filters
#         self.n_layers = n_layers
#         self.activation = activation
#         self.dropout = dropout
#         self.drop_value = drop_value
#         self.kernel_size = kernel_size
#         self.pool_size = pool_size
#         self.patch_size = patch_size
        
#         self.logger.info(f"Initializing UNet with {num_vars} input variables and {n_layers} layers")
        
#         # Input layer: Process each variable separately
#         self.var_blocks = nn.ModuleList()
#         for _ in range(self.num_vars):
#             # Initial preprocessing for each variable
#             self.var_blocks.append(nn.Sequential(
#                 nn.Conv2d(1, filters, kernel_size=kernel_size, padding='same'),
#                 nn.BatchNorm2d(filters),
#                 activation,
#                 nn.Conv2d(filters, filters, kernel_size=kernel_size, padding='same'),
#                 nn.BatchNorm2d(filters),
#                 activation
#             ))
        
#         # Feature fusion layer
#         self.fusion = nn.Conv2d(filters * num_vars, filters, kernel_size=1)
        
#         # Encoder blocks
#         self.enc_blocks = nn.ModuleList()
#         self.pool_blocks = nn.ModuleList()
#         current_filters = filters
        
#         for i in range(n_layers):
#             next_filters = current_filters * 2
#             self.enc_blocks.append(nn.Sequential(
#                 nn.Conv2d(current_filters, next_filters, kernel_size, padding='same'),
#                 nn.BatchNorm2d(next_filters),
#                 activation,
#                 nn.Conv2d(next_filters, next_filters, kernel_size, padding='same'),
#                 nn.BatchNorm2d(next_filters),
#                 activation
#             ))
#             self.pool_blocks.append(nn.MaxPool2d(pool_size))
#             current_filters = next_filters
        
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(current_filters, current_filters * 2, kernel_size, padding='same'),
#             nn.BatchNorm2d(current_filters * 2),
#             activation,
#             nn.Conv2d(current_filters * 2, current_filters * 2, kernel_size, padding='same'),
#             nn.BatchNorm2d(current_filters * 2),
#             activation,
#             nn.Dropout2d(drop_value) if dropout else nn.Identity()
#         )
        
#         # Decoder blocks
#         self.up_blocks = nn.ModuleList()
#         self.dec_blocks = nn.ModuleList()
#         current_filters = current_filters * 2
        
#         for i in range(n_layers):
#             next_filters = current_filters // 2
#             self.up_blocks.append(nn.Sequential(
#                 nn.ConvTranspose2d(current_filters, next_filters, kernel_size=pool_size, 
#                                 stride=pool_size, padding=0),
#                 nn.BatchNorm2d(next_filters),
#                 activation
#             ))
#             self.dec_blocks.append(nn.Sequential(
#                 nn.Conv2d(current_filters, next_filters, kernel_size, padding='same'),
#                 nn.BatchNorm2d(next_filters),
#                 activation,
#                 nn.Conv2d(next_filters, next_filters, kernel_size, padding='same'),
#                 nn.BatchNorm2d(next_filters),
#                 activation
#             ))
#             current_filters = next_filters
        
#         # Classification head
#         # Calculate the output size based on input neighborhood and operations
#         patch_size = patch_size
#         final_size = patch_size // (pool_size ** n_layers) if patch_size % (pool_size ** n_layers) == 0 else 1
        
#         self.classification_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#             nn.Flatten(),
#             nn.Linear(filters, 256),
#             nn.BatchNorm1d(256),
#             activation,
#             nn.Dropout(drop_value) if dropout else nn.Identity(),
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             activation,
#             nn.Linear(64, 1)
#         )
    
#     def forward(self, x):
#         """Forward pass of the UNet model."""
#         # Split input into separate variable channels
#         var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        
#         # Process each variable through its own block
#         var_features = []
#         for i, (block, inp) in enumerate(zip(self.var_blocks, var_inputs)):
#             var_features.append(block(inp))
        
#         # Concatenate and fuse features from all variables
#         x = torch.cat(var_features, dim=1)
#         x = self.fusion(x)
        
#         # Store encoder outputs for skip connections
#         enc_features = []
        
#         # Encoder path
#         for enc_block, pool_block in zip(self.enc_blocks, self.pool_blocks):
#             # Save output before pooling for skip connection
#             enc_features.append(x)
#             # Apply convolution block then pooling
#             x = enc_block(x)
#             x = pool_block(x)
        
#         # Bottleneck
#         x = self.bottleneck(x)
        
#         # Decoder path with skip connections
#         for i, (up_block, dec_block) in enumerate(zip(self.up_blocks, self.dec_blocks)):
#             # Upsample
#             x = up_block(x)
            
#             # Get corresponding encoder feature map
#             skip_feature = enc_features[-(i+1)]
            
#             # Handle size mismatch (if any)
#             if x.shape != skip_feature.shape:
#                 # Center crop or pad to match
#                 diff_h = skip_feature.size(2) - x.size(2)
#                 diff_w = skip_feature.size(3) - x.size(3)
                
#                 if diff_h > 0 and diff_w > 0:
#                     skip_feature = skip_feature[:, :, diff_h//2:-(diff_h//2), diff_w//2:-(diff_w//2)]
#                 elif diff_h < 0 and diff_w < 0:
#                     padding = [-diff_h//2, -diff_h-(-diff_h//2), -diff_w//2, -diff_w-(-diff_w//2)]
#                     skip_feature = F.pad(skip_feature, padding)
            
#             # Concatenate for skip connection
#             x = torch.cat([x, skip_feature], dim=1)
            
#             # Apply convolution block
#             x = dec_block(x)
        
#         # Final classification
#         outputs = self.classification_head(x)
#         return torch.sigmoid(outputs)

# CNN architecture for hazard susceptibility modeling
class CNN(nn.Module):
    def __init__(self, logger, num_vars, filters, n_layers, activation, dropout, drop_value, kernel_size, pool_size, patch_size):
        super(CNN, self).__init__()
        self.logger = logger

        self.num_vars = num_vars
        self.filters = filters
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.drop_value = drop_value
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.patch_size = patch_size

        # Define variable-specific blocks
        self.var_blocks = nn.ModuleList()
        for _ in range(self.num_vars):
            layers = [
                nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size)
            ]
            self.var_blocks.append(nn.Sequential(*layers))

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        fc_input_size = self.filters * self.num_vars  # Adjust based on architecture
        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_layer = nn.Dropout(self.drop_value) if self.dropout else nn.Identity()
        self.output_layer = nn.Linear(1024, 1)

    def forward(self, inputs):
        # Split inputs into a list of tensors, one for each variable
        inputs = [inputs[:, i, :, :].unsqueeze(1) for i in range(self.num_vars)]
        # self.logger.info(f"Split input shapes: {[inp.shape for inp in inputs]}")

        # Process each variable through its block
        features = []
        for i, (block, inp) in enumerate(zip(self.var_blocks, inputs)):
            x = block(inp)
            # self.logger.info(f"After var_blocks[{i}]: {x.shape}")
            features.append(x)

        # Concatenate features along the channel dimension
        x = torch.cat(features, dim=1)
        # self.logger.info(f"After concatenation: {x.shape}")

        # Global average pooling
        x = self.global_avg_pool(x)
        # self.logger.info(f"After global_avg_pool: {x.shape}")

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # self.logger.info(f"After flattening: {x.shape}")

        # Pass through fully connected layers
        x = self.fc1(x)
        # self.logger.info(f"After fc1: {x.shape}")
        x = self.bn1(x)
        # self.logger.info(f"After bn1: {x.shape}")
        x = self.activation(x)
        x = self.drop_layer(x)
        x = self.output_layer(x)
        # self.logger.info(f"After output_layer: {x.shape}")
        x = torch.sigmoid(x)
        # self.logger.info(f"After sigmoid: {x.shape}")
        return x
    
# model from Japan paper converted to pytorch
class SpatialAttentionLayer(nn.Module):
    def __init__(self, device=None):
        super(SpatialAttentionLayer, self).__init__()
        self.device = device

    def build(self, channels):
        self.conv1x1_theta = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1x1_phi = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1x1_g = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        # Initialize weights similarly to Keras
        init.kaiming_normal_(self.conv1x1_theta.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv1x1_phi.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.conv1x1_g.weight)
        
        # Move layers to the same device as input
        if self.device is not None:
            self.conv1x1_theta = self.conv1x1_theta.to(self.device)
            self.conv1x1_phi = self.conv1x1_phi.to(self.device)
            self.conv1x1_g = self.conv1x1_g.to(self.device)
        

    def forward(self, x):
        if not hasattr(self, 'conv1x1_theta'):
            self.build(x.size(1))
            
        theta = F.relu(self.conv1x1_theta(x))
        phi = F.relu(self.conv1x1_phi(x))
        g = torch.sigmoid(self.conv1x1_g(x))

        theta_phi = theta * phi
        attention = theta_phi * g
        attended_x = x + attention
        
        return attended_x

class FullCNN(nn.Module):
    def __init__(self, logger, num_vars, filters=64, kernel_size=3, pool_size=2, 
                 n_layers=5, device=None, dropout=True, drop_value=0.5, name_model="FullCNN_Model", patch_size=5):
        super(FullCNN, self).__init__()
        
        self.device = device
        self.logger = logger
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.drop_value = drop_value
        self.name_model = name_model
        self.patch_size = patch_size
        self.num_vars = num_vars
        # self.logger.info(f"Initializing {self.name_model} with {num_vars} input variables")
        
        # Create modules for each variable input (branches)
        self._build_conv_branches()
        
        # Global average pooling to handle variable spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Calculate the number of output features
        if self.n_layers == 1:
            output_filters = self.filters
        else:
            output_filters = self.filters * 2

        total_features = output_filters * num_vars
        
        # self.logger.info(f"Calculated output features: {total_features} (filters={filters}, num_vars={num_vars})")
        
        self.dense_layers = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.drop_value) if self.dropout else nn.Identity(),
            nn.Linear(1024, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_conv_branches(self):
        """Build the convolutional branches during initialization"""
        self.conv_branches = nn.ModuleList()
        for i in range(self.num_vars):
            layers = []
            # First conv layer with spatial attention
            layers.append(nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, padding='same'))
            layers.append(nn.ReLU())
            
            # Add spatial attention layer
            spatial_attn = SpatialAttentionLayer(device=self.device)
            spatial_attn.build(self.filters)
            layers.append(spatial_attn)
            
            # Add pooling layer
            layers.append(nn.MaxPool2d(kernel_size=self.pool_size, padding=1))
            
            # Additional convolutional layers
            for j in range(self.n_layers - 1):
                in_filters = self.filters if j == 0 else self.filters * 2
                layers.append(nn.Conv2d(in_filters, self.filters * 2, kernel_size=self.kernel_size, padding='same'))
                layers.append(nn.ReLU())
                
                if j == 1 or j == 3 or j == self.n_layers - 2:
                    layers.append(nn.MaxPool2d(kernel_size=self.pool_size, padding=1))
            
            self.conv_branches.append(nn.Sequential(*layers))
            
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        # Xavier/Glorot initialization for the final layer
        init.xavier_uniform_(self.dense_layers[-1].weight)
    
    def forward(self, x):
        # Process each variable through its branch
        var_inputs = [x[:, i:i+1, :, :] for i in range(self.num_vars)]
        
        features = []
        for i, branch in enumerate(self.conv_branches):
            if i < len(var_inputs):
                feat = branch(var_inputs[i])
                features.append(feat)
        
        # Concatenate features from all branches
        if len(features) > 1:
            merged = torch.cat(features, dim=1)
        else:
            merged = features[0]
        
        # Global average pooling
        pooled = self.global_avg_pool(merged).view(merged.size(0), -1)
        
        # Final dense layers
        x = self.dense_layers(pooled)
        
        # Apply sigmoid for final activation
        output = torch.sigmoid(x)
        
        return output

class Baseline:
    def __init__(self, hazard, region, sample_size, model_architecture, variables, logger, seed):
        
        self.hazard = hazard.lower()
        self.sample_size = sample_size
        self.region = region
        self.model_architecture = model_architecture
        self.seed = seed
        self.logger = logger
        self.variables = variables
        self.num_vars = len(variables)

        # Check if the model architecture is valid
        if self.model_architecture not in ["RF", "LR"]:
            raise ValueError(f"Baseline architecture '{self.model_architecture}' is not defined.")

        self.logger.info(f'Loading data...')
        self.X_train, self.y_train, self.X_test, self.y_test =self.load_data()
        self.logger.info(f'Creating model...')
        self.model = self.design_model()
    
    def balance_classes(self, X, y, ratio=1.0, random_seed=42):
        """
        Balance classes by undersampling the majority class.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            ratio (float): Desired ratio of minority to majority class (default 1.0 means balanced).
            random_seed (int): Seed for reproducibility.
            
        Returns:
            X_balanced, y_balanced
        """
        np.random.seed(random_seed)
        
        # Find indices of each class
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        if n_pos == 0 or n_neg == 0:
            print("Warning: One of the classes is missing!")
            return X, y

        # Determine the number of samples to keep
        if n_pos < n_neg:
            n_neg_keep = int(n_pos / ratio)
            n_neg_keep = min(n_neg, n_neg_keep)
            neg_sampled = np.random.choice(neg_indices, n_neg_keep, replace=False)
            balanced_indices = np.concatenate([pos_indices, neg_sampled])
        else:
            n_pos_keep = int(n_neg * ratio)
            n_pos_keep = min(n_pos, n_pos_keep)
            pos_sampled = np.random.choice(pos_indices, n_pos_keep, replace=False)
            balanced_indices = np.concatenate([pos_sampled, neg_indices])

        # Shuffle indices
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        return X_balanced, y_balanced

    def load_data(self):

        # Define the feature file paths for each hazard

        var_paths = {
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
            "elevation2" : "Input/Europe/npy_arrays/masked_elevation_Europe.npy",
        }

        label_paths = {
            "wildfire" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wildfire_Europe.npy",
            "test" : "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_wildfire_Europe.npy",
            "landslide" : "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_landslide_Europe.npy",

        }
     
        # Check if the hazard is valid
        if self.hazard not in label_paths.keys():
            raise ValueError(f"Hazard '{self.hazard}' is not defined in the dataset.")
        # Check if the variables are valid
        for variable in self.variables:
            if variable not in var_paths.keys():
                raise ValueError(f"Variable '{variable}' is not defined in the dataset.")
        partition_map = np.load(f'Input/{self.region}/partition_map/{self.hazard}_partition_map.npy')
        elevation_map = np.load(var_paths["elevation"])
        valid_mask = ~np.isnan(elevation_map)

        partition_flat = partition_map.flatten()
        valid_flat = valid_mask.flatten()
        
        train_val_mask = valid_flat & ((partition_flat == 1) | (partition_flat == 2))
        test_mask = valid_flat & (partition_flat == 3)

        labels_flat = np.load(label_paths[self.hazard]).flatten()
        labels_flat = (labels_flat > 0).astype(int)

        y_train_val = labels_flat[train_val_mask]
        y_test = labels_flat[test_mask]

        feature_list_train_val = []
        feature_list_test = []

        for variable in self.variables:
            feature = np.load(var_paths[variable]).flatten()
            # convert NaN values to 0
            feature = np.nan_to_num(feature, nan=0.0)
            feature_train_val = feature[train_val_mask]
            feature_test = feature[test_mask]
            feature_list_train_val.append(feature_train_val)
            feature_list_test.append(feature_test)

        X_train_val = np.stack(feature_list_train_val, axis=-1)
        X_test = np.stack(feature_list_test, axis=-1)
        

        # Apply class balancing on train_val data
        X_train_balanced, y_train_balanced = self.balance_classes(X_train_val, y_train_val, ratio=1)

        return X_train_balanced, y_train_balanced, X_test, y_test
    
    def design_model(self):
        
        if self.model_architecture == "RF":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=self.seed,
            )
        elif self.model_architecture == "LR":
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=100,
                random_state=self.seed,
            )
        else:
            raise ValueError(f"Model '{self.model}' is not defined. Choose 'RF' or 'LR'.")
        
        return model

    def train(self):
        output_path = f"Output/{self.region}/{self.hazard}/baselines/"
        file_name = f'{self.model_architecture}_{self.hazard}.joblib'
        self.logger.info(f"Training {self.model_architecture} model...")
        self.model.fit(self.X_train, self.y_train)

        # Check if the output directory exists, if not create it
        self.logger.info('Saving model...')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            self.logger.info(f"Output directory {output_path} created.")
        else:
            self.logger.info(f"Output directory {output_path} already exists.")
        # Save the model    
        dump(self.model, f"{output_path}{file_name}") 
        self.logger.info(f"Model saved to {output_path}{file_name}")
    
    def load_model(self):
        output_path = f"Output/{self.region}/{self.hazard}/baselines/"
        file_name = f'{self.model_architecture}_{self.hazard}.joblib'
        # check if the file exists
        try:
            self.model = load(f"{output_path}{file_name}")
            self.logger.info(f"Model loaded from {output_path}{file_name}")


        except FileNotFoundError:
            self.logger.error(f"Model file {output_path}{file_name} not found.")

    def testing(self):

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        return self.y_test, y_pred_proba
   
class HazardModel():
    def __init__(self, device, hazard, region, variables, patch_size, batch_size, model_architecture, 
                 logger, seed, use_wandb, sample_size, class_ratio, num_workers, early_stopping, patience, min_delta):
        super(HazardModel, self).__init__()

        # Configs
        self.use_wandb = use_wandb
        self.logger = logger
        self.device = device
        self.hazard = hazard
        self.region = region

        # Model parameters
        self.model = None
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_vars = len(variables)
        self.variables = variables
        self.seed = seed
        self.name_model = hazard + '_' + region + '_' + model_architecture
        self.model_architecture = model_architecture
        self.best_model_state = None

        # Data
        self.num_workers = num_workers # number of workers for data loading
        self.sample_size = sample_size # fraction of the dataset to use for training
        self.class_ratio = class_ratio # ratio of positive to negative samples in the dataset
        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()
        
        # Hyperparameters
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001
        self.filters = 32
        self.n_layers = 3
        self.drop_value = 0.4
        self.kernel_size = 3
        self.pool_size = 2  
        self.activation = torch.nn.ReLU()
        self.dropout = True
        self.n_nodes = 128 # for MLP architecture

        # Training
        self.current_epoch = 0
        self.epochs = 10
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.average_epoch_time = 0

        # WandB 
        if self.use_wandb:
            wandb.init(
                    project=f"{hazard}_susceptibility",
                    name=f"{self.model_architecture}_hazard_model",
                    config={
                        "hazard": self.hazard,
                        "region": self.region,
                        "batch_size": self.batch_size,
                        "patch_size": self.patch_size,
                        "variables": self.variables,
                        "learning_rate": self.learning_rate,
                        "filters": self.filters,
                        "n_layers": self.n_layers,
                        "dropout": self.dropout,
                        "model_type": self.model_architecture 

                    }
                )
            self.wandb_run_name = wandb.run.name
        else:
            self.wandb_run_name = "no_wandb"

        # outputs
        self.output_path = f"Output/{self.region}/{self.hazard}/{self.model_architecture}"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
            self.logger.info(f"Output directory {self.output_path} created.")

    def design_basemodel(self):
        """
        Define the CNN architecture in PyTorch.
        """
        if self.model_architecture == 'MLP':

            model = MLP(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes, 
                dropout=self.dropout,
                drop_value=self.drop_value,
                patch_size=self.patch_size
            )

        elif self.model_architecture == 'CNN':

            model = CNN(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                filters=self.filters,
                n_layers=self.n_layers,
                activation=self.activation,
                dropout=self.dropout,
                drop_value=self.drop_value,
                kernel_size=self.kernel_size,
                pool_size=self.pool_size,
                patch_size=self.patch_size
            )
        elif self.model_architecture == 'UNet':

            model = UNet(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                filters=self.filters,
                n_layers=self.n_layers,
                activation=self.activation,
                dropout=self.dropout,
                drop_value=self.drop_value,
                kernel_size=self.kernel_size,
                pool_size=self.pool_size,
                patch_size=self.patch_size
            )
        elif self.model_architecture == 'SimpleCNN':

            model = SimpleCNN(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                filters=self.filters,
                dropout=self.dropout,
                drop_value=self.drop_value,
                patch_size=self.patch_size
            )

        elif self.model_architecture == 'FullCNN':

            model = FullCNN(
                device=self.device,
                logger=self.logger,
                num_vars=self.num_vars,
                filters=self.filters,
                kernel_size=self.kernel_size,
                pool_size=self.pool_size,
                n_layers=self.n_layers,
                dropout=self.dropout,
                drop_value=self.drop_value,
                name_model=self.name_model,
                patch_size=self.patch_size
            )
        self.model = model
        self.model.to(self.device)

        self.logger.info(f'''Model architecture: {self.model_architecture}\n
                            Number of variables: {self.num_vars}\n
                            Patch size: {self.patch_size}\n
                            Batch size: {self.batch_size}\n
                            Learning rate: {self.learning_rate}\n
                            Filters: {self.filters}\n
                            Number of layers: {self.n_layers}\n
                            Dropout value: {self.drop_value}\n
                            Weight decay: {self.weight_decay}\n
                            Sample size: {self.sample_size}\n
                            Class ratio: {self.class_ratio}\n
                          ''')
        return model
    
    def load_dataset(self):
        """
        
        Preprocess the data for the model, using the dataset class.

        """
        
        # loading partition map 
        # TODO generalize for other hazard partition maps
        self.logger.info('Loading partition map')
        partition_map = np.load(f'Input/{self.region}/partition_map/{self.hazard}_partition_map.npy')
        partition_shape = partition_map.shape
        self.logger.info(f'Splitting dataset into train, validation and test sets')

        dataset = HazardDataset(hazard=self.hazard, variables=self.variables, patch_size=self.patch_size)
        
        idx_transform = np.array([[partition_shape[1]],[1]])

        train_indices = (np.argwhere(partition_map == 1) @ idx_transform).flatten()
        val_indices = (np.argwhere(partition_map == 2) @ idx_transform).flatten()
        test_indices = (np.argwhere(partition_map == 3) @ idx_transform).flatten()

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        train_indices = train_indices[:int(len(train_indices) * self.sample_size)]
        val_indices = val_indices[:int(len(val_indices) * self.sample_size)]
        test_indices = test_indices[:int(len(test_indices) * self.sample_size)]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Get labels for the subset
        train_labels = dataset.labels.flatten()[train_indices]
        val_labels = dataset.labels.flatten()[val_indices]


        # Create custom balanced batch sampler
        train_batch_sampler = BalancedBatchSampler(train_labels, self.batch_size, neg_ratio=self.class_ratio)
        val_batch_sampler = BalancedBatchSampler(val_labels, self.batch_size, neg_ratio=self.class_ratio)


        # Create DataLoader with the custom batch sampler
        train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_sampler=train_batch_sampler)
        val_loader = DataLoader(val_dataset, num_workers=self.num_workers, batch_sampler=val_batch_sampler)
        test_loader = DataLoader(test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        self.logger.info(f"Test dataset size: {len(test_dataset)}")

        # Create DataLoaders for training and validation

        return train_loader, val_loader, test_loader

    def train(self):
        """
        Train the model using the provided data loaders.
        """
        self.logger.info(f'Training the model with :{len(self.train_loader)*self.batch_size} samples ')
        if self.use_wandb:
            wandb.watch(self.model, log="all")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, ) # added L2 regularization
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10,           # Reduce LR every 10 epochs
            gamma=0.8               # Multiply LR by 0.5 (50% reduction)
        )


        self.epochs_without_improvement = 0
        self.best_val_loss = float('inf')
        start_time = time.time()


        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Reshape Labels to match output 
                labels = labels.unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy(outputs, labels)
                mae = self.safe_mae(labels, outputs)

                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Step the optimizer
                optimizer.step()

                train_loss += loss.item()
                train_mae += mae.item()

            
            # Calculate average metrics
            train_loss /= len(self.train_loader)
            train_mae /= len(self.train_loader)

            
            # Evaluate on validation data
            val_loss, val_mae = self.evaluate()

            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

            # Save the best model
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.save_best_model()
                self.epochs_without_improvement = 0
                self.logger.info(f"New best model saved (lowest validation loss: {self.best_val_loss:.4f})")

            else:
                self.epochs_without_improvement += 1
                self.logger.info(f"  No improvement for {self.epochs_without_improvement} epochs")
                
                # Early stopping check
                if self.early_stopping and self.epochs_without_improvement >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Update learning rate
            scheduler.step()
            self.learning_rate = scheduler.get_last_lr()[0]

            end_time = time.time()
            training_time = end_time - start_time 
            self.average_epoch_time = training_time / (epoch + 1)

            # Log metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_MAE": train_mae,
                    "val_loss": val_loss,
                    "val_MAE": val_mae,
                    "learning_rate": self.learning_rate,
                    "epoch_time": self.average_epoch_time,
                })

    @staticmethod
    def safe_mae(y_true, y_pred):
        """
        Mean absolute error (MAE) loss with NaN handling.
        """
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        return torch.mean(torch.abs(y_pred - y_true))

    def save_best_model(self):
        # Create directory if it doesn't exist
        model_path = f"{self.output_path}/models/{self.best_val_loss:4f}_{self.name_model}.pth"
        onnx_path = f"{self.output_path}/onnx/{self.best_val_loss:4f}_{self.name_model}.onnx"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)


        # Save PyTorch model
        torch.save(self.model.state_dict(), model_path)
        self.best_model_state = self.model.state_dict().copy()
        
        # Export to ONNX format
        dummy_input = torch.randn(1, self.num_vars, self.patch_size, self.patch_size).to(self.device)
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            self.logger.info(f"ONNX model saved to {onnx_path}")
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.save(onnx_path)
        except Exception as e:
            self.logger.error(f"Failed to export ONNX model: {e}")

    def load_best_model(self):
        """
        Load the best model state from the saved file.
        """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Best model loaded successfully.")
        else:
            self.logger.error("No model state available to load. Please train the model first or provide a valid path.")
         
    def evaluate(self):
        """
        Evaluate the model on the provided data loader.
        
        Returns:
            Tuple of (loss, accuracy, f1)
        """
        self.model.eval()
        val_loss = 0.0
        val_mae = 0.0


        with torch.no_grad():
            self.logger.info(f'Evaluating the model with :{len(self.val_loader)*self.batch_size} samples ')
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate metrics
                # loss = self.safe_binary_crossentropy(labels, outputs)
                loss = F.binary_cross_entropy(outputs, labels)

                mae = self.safe_mae(labels, outputs)

                
                val_loss += loss.item()
                val_mae += mae.item()


        # Calculate average metrics
        val_loss /= len(self.val_loader)
        val_mae /= len(self.val_loader)
        
        return val_loss, val_mae
        
    def testing(self):
        """
        Test the model on the test set, log AUROC + other metrics, and export to CSV/W&B/ONNX.
        """
    
        self.logger.info('Testing model...')
        self.model.eval()

        y_true, y_prob = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze()
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())

        y_true, y_prob = np.array(y_true), np.array(y_prob)

        return y_true, y_prob
    
    def HypParOpt(self):
        """Hyperparameter optimization using wandb sweeps"""
        self.logger.info("Starting hyperparameter optimization with wandb")
        
        # Define sweep configuration
        sweep_config = {
            'method': 'bayes',  # Use Bayesian optimization
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 0.00001,
                    'max': 0.001
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 0.00001,
                    'max': 0.001
                },
                'filters': {
                    'values': [4, 8, 16, 32]
                },
                'n_layers': {
                    'values': [1, 2, 3, 4]
                },
                'drop_value': {
                    'distribution': 'uniform',
                    'min': 0.3,
                    'max': 0.6
                },
                'patch_size': {
                    'values': [1, 3, 5, 7]
                },
                'class_ratio': {
                    'distribution': 'uniform',
                    'min': 1,
                    'max': 10
                },
            },
            'program':'hazard_model.py'
        }
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"{self.hazard}_{self.model_architecture}_sweep"
        )
         
        # Start the sweep agent
        wandb.agent(sweep_id, self.sweep_train, count=100)  # Run 10 trials
        self.logger.info(f"Hyperparameter sweep completed with ID: {sweep_id}")
 
    def sweep_train(self):
        # Initialize new wandb run
        with wandb.init() as run:
            # Get hyperparameters from wandb config
            config = wandb.config
            
            # Update model hyperparameters
            self.learning_rate = config.learning_rate
            self.weight_decay = config.weight_decay
            self.filters = config.filters
            self.n_layers = config.n_layers 
            self.drop_value = config.drop_value
            self.patch_size = config.patch_size
            self.class_ratio = config.class_ratio
            
            # Log configuration for this run
            self.logger.info(f"Sweep run with: lr={self.learning_rate}, "
                        f"weight_decay={self.weight_decay}, "
                        f"filters={self.filters}, "
                        f"n_layers={self.n_layers}, "
                        f"dropout={self.drop_value}, "
                        f"patch_size={self.patch_size}," 
                        f"class_ratio={self.class_ratio},"
            )
            
            # Recreate data loaders with new batch size
            self.train_loader, self.val_loader, self.test_loader = self.load_dataset()
            
            # Rebuild model with new hyperparameters
            self.model = self.design_basemodel(architecture=self.model_architecture)
            self.model.to(self.device)
            
            # Set smaller number of epochs for sweep runs
            original_epochs = self.epochs
            self.epochs = 50 # Reduced epochs for sweep runs
            
            # Use the existing training loop
            self.train()
            
            # Restore original epochs
            self.epochs = original_epochs
            
            # Evaluate on test set

            y_true, y_prob  = self.testing()
            test_accuracy = accuracy_score(y_true, (y_prob > 0.5).astype(int))
            test_precision = precision_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)
            test_recall = recall_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)
            test_f1 = f1_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)
            test_auc_roc = roc_auc_score(y_true, y_prob)
            test_auprc = average_precision_score(y_true, y_prob)

            # Calculate metrics
            
            wandb.log({
                "final_test_accuracy": test_accuracy,
                "final_test_precision": test_precision,
                "final_test_recall": test_recall,
                "final_test_f1": test_f1,
                "final_test_auc_roc": test_auc_roc,
                "final_test_auprc": test_auprc,
            }) 
          
    def predict(self, dataloader):
        """
        Predict using the model.
        """

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions, axis=0)
    
    def safe_accuracy(self, labels, outputs):
        """
        Calculate accuracy while handling potential NaN values in outputs.
        """
        # Ensure outputs are between 0 and 1
        outputs = torch.clamp(outputs, 0, 1)
        
        # Convert outputs to binary predictions
        preds = (outputs > 0.5).float()
        
        # Calculate accuracy
        correct = (preds == labels).float().sum()
        total = labels.numel()
# TODO test this function 
    def make_hazard_map(self):
        """
        Create a hazard map using the trained model.
        """
        # Load the best model
        self.load_best_model()
        
        # loading elevation map to get indices
        elevation = np.load(f'Input/{self.region}/npy_arrays/masked_elevation_Europe.npy')
        elevation_shape = elevation.shape

        
        #get indices where mask is not nan and transform to 1D
        map_indices2d = np.argwhere(np.isnan(elevation) == False)
        idx_transform = np.array([[elevation_shape[1]],[1]])
        map_indices1d = (map_indices2d @ idx_transform).flatten()

        # Create a Subset of the dataset using the indices
        dataset = HazardDataset(hazard=self.hazard, variables=self.variables, patch_size=self.patch_size)        
        map_dataset = Subset(dataset, map_indices1d)  
        map_loader = DataLoader(map_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)
        self.logger.info(f"Map dataset size: {len(map_dataset)}")

        #predict using the model
        predictions = self.predict(map_loader)

        # map the prediction back to the original shape using the indices
        hazard_map = np.full(elevation_shape, np.nan)
        hazard_map[map_indices2d[:, 0], map_indices2d[:, 1]] = predictions.flatten()
        hazard_map = np.clip(hazard_map, 0, 1)  # Ensure values are between 0 and 1
    
       
        # Save the hazard map
        output_dir = f'Output/{self.region}/{self.hazard}/hazard_map'
        os.makedirs(output_dir, exist_ok=True)
        hazard_map_path = os.path.join(output_dir, f"{self.hazard}_hazard_map.npy")
        np.save(hazard_map_path, hazard_map)
        self.logger.info(f"Hazard map saved to {hazard_map_path}")

        # Optionally, visualize the hazard map 

        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"hazard_map": wandb.Image(hazard_map_path)})

class ModelMgr:
    def __init__(self, region='Europe', batch_size=1024, patch_size=5, architecture='CNN' , sample_size = 1, 
             class_ratio=0.5, hazard='wildfire', hyper=False, use_wandb=True):
        
        self.early_stopping = True
        self.patience = 10
        self.min_delta = 0.001
        self.use_wandb = use_wandb
        self.hazard = hazard
        self.region = region
        self.batch_size = batch_size
        self.name_model = 'susceptibility'
        self.missing_data_value = 0
        self.sample_ratio = 0.7
        self.test_split = 0.15
        self.sample_size = sample_size
        self.class_ratio = class_ratio
        self.num_workers = 8
        self.patch_size = patch_size
        self.hyper = hyper
        self.model_architecture = architecture # 'CNN' or 'UNet' or 'SimpleCNN'



        if self.hazard == 'landslide':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation_daily', 'accuflux', 'HWSD', 'GEM', 'curvature', 'GLIM']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'categorical']
            # self.variables = ['elevation', 'slope', 'landcover', 'NDVI', 'precipitation', 'HWSD']
            # self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'categorical']
        elif self.hazard == 'Flood':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation', 'accuflux']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous']
        elif self.hazard == 'Tsunami':
            self.variables = ['elevation', 'coastlines', 'GEM']
            self.var_types = ['continuous', 'continuous', 'continuous']
            # self.variables = ['coastlines']
            # self.var_types = ['continuous']
        elif self.hazard == "wildfire":
            # temperature_daily, NDVI, landcover, elevation, wind_speed, fire_weather, soil_moisture(root or surface)
            self.variables = ['temperature_daily', 'NDVI', 'landcover', 'elevation', 'wind_speed_daily', 'fire_weather', 'soil_moisture_root']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous']
        elif self.hazard == 'Multihazard':
            # self.variables = ['drought', 'extreme_wind', 'fire_weather', 'heatwave', 'pga', 'volcano', 'Flood_hazard_model', 'landslide_hazard_model', 'Tsunami_hazard_model']
            self.variables = ['drought', 'extreme_wind', 'fire_weather', 'heatwave', 'jshis', 'volcano', 'Flood_hazard_model', 'landslide_hazard_model', 'Tsunami_hazard_model']
            self.var_types = ['continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous']
        elif self.hazard == 'test':
            # temperature_daily, NDVI, landcover, elevation, wind_speed, fire_weather, soil_moisture(root or surface)
            self.variables = ['test']
            self.var_types = ['continuous']
        else:
            raise ValueError(f"Unknown hazard: {self.hazard}")
        
        self.ensemble_nr = 5  # 5
        self.seed = 43

        self.logger, self.ch = self.set_logger()


       # PyTorch GPU configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"Num GPUs Available: {torch.cuda.device_count()}")
            self.logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

        self.logger.info(f"Torch version: {torch.__version__}")
        

        if self.model_architecture in ["FullCNN", "UNet", "MLP", "CNN", "SimpleCNN"]:
            self.hazard_model_instance = HazardModel(
                device=self.device,
                hazard=self.hazard,
                region=self.region,
                variables=self.variables,
                patch_size=self.patch_size,
                batch_size=self.batch_size,
                sample_size=self.sample_size,
                class_ratio=self.class_ratio,
                model_architecture=self.model_architecture,
                num_workers=self.num_workers,
                logger=self.logger,
                seed=self.seed,
                use_wandb=self.use_wandb,
                early_stopping=self.early_stopping,
                patience=self.patience,
                min_delta=self.min_delta
                
            )
        elif self.model_architecture in ["RF", "LR"]:
            self.baseline_instance = Baseline(
                hazard=self.hazard,
                region=self.region,
                variables=self.variables,
                sample_size=self.sample_size,
                model_architecture=self.model_architecture,
                logger=self.logger,
                seed=self.seed,
            )
        else:
            self.logger.error(f"Unknown model architecture: {self.model_architecture}")
            raise ValueError(f"Unknown model architecture: {self.model_architecture}")

    def set_logger(self, verbose=True):
        """
        Set-up the logging system, exit if this fails
        """
        # assign logger file name and output directory
        datelog = time.ctime()
        datelog = datelog.replace(':', '_')
        reference = f'CNN_ls_susc_{self.region}'

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

    def train_baseline_model(self):
        self.baseline_instance.train()
        y_true, y_prob = self.baseline_instance.testing()
        self.logger.info('Evaluating baseline model predictions')
        self.evaluate_predictions(
            y_true=y_true,
            y_prob=y_prob
        )

    def train_hazard_model(self):
        if self.hyper:
            self.hazard_model_instance.HypParOpt()
        else:
            self.hazard_model_instance.design_basemodel()
            self.hazard_model_instance.train()
            y_true, y_prob = self.hazard_model_instance.testing()
            self.evaluate_predictions(
                y_true=y_true,
                y_prob=y_prob
            )

        if self.use_wandb:  
            wandb.finish(exit_code=1)
    
    def evaluate_predictions(self, y_true, y_prob):
        """
        Evaluate model predictions with optimized threshold, compute metrics, and export results.

        Args:
            y_true (np.ndarray): True binary labels.
            y_prob (np.ndarray): Predicted probabilities.
            output_path (str): Path to save outputs.
            model_name (str): Name of the model (for output file naming).
            model_info (dict): Additional info (e.g., hyperparameters) to include in results.
            use_wandb (bool): Whether to log to Weights & Biases.
            wandb_run_name (str): Optional run name for W&B.

        Returns:
            metrics (dict): Computed performance metrics.


        """


        output_path = f'Output/{self.region}/{self.hazard}/evaluation'
        os.makedirs(output_path, exist_ok=True)
        model_name = f'{self.hazard}_{self.model_architecture}'

        # Optimize threshold based on best F1
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


        # Get predictions with best threshold
        y_pred = (y_prob >= best_threshold).astype(int)

        # Compute metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'AUROC': roc_auc_score(y_true, y_prob),
            'AP': average_precision_score(y_true, y_prob),
            'MAE': mean_absolute_error(y_true, y_prob),
            'Best_Threshold': best_threshold
        }

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_path = os.path.join(output_path, f'{model_name}_roc_curve.png')
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUROC = {metrics["AUROC"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(roc_path)
        plt.close()

        # Plot Precision-Recall curve
        pr_path = os.path.join(output_path, f'{model_name}_pr_curve.png')
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f'AP = {metrics["AP"]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(pr_path)
        plt.close()

        # Save results to CSV
        results_csv = os.path.join(output_path, f'all_model_metrics.csv')
        results_df = pd.DataFrame([metrics])
        if os.path.exists(results_csv):
            pd.concat([pd.read_csv(results_csv), results_df], ignore_index=True).to_csv(results_csv, index=False)
        else:
            results_df.to_csv(results_csv, index=False)
        self.logger.info(f"Metrics saved to {results_csv}")

    def make_hazard_map(self):
        """
        Create a hazard map using the trained model.
        """

        self.hazard_model_instance.make_hazard_map()

if __name__ == "__main__":
    # Example configuration
    region = 'Europe'
    hazard = "landslide"       # wildfire or landslide
    batch_size = 1024
    patch_size = 5
    architecture = 'FullCNN'  # 'FullCNN' or 'UNet' or 'MLP' or LR or RF 
    sample_size = 1
    class_ratio = 9 #float(sys.argv[1]) # ration of negative to positive samples in the batch, 1 means equal ratio
    hyper = True
    use_wandb = True

    # Initialize the model manager
    model_mgr = ModelMgr(
        region=region,
        batch_size=batch_size,
        patch_size=patch_size,
        architecture=architecture,
        sample_size=sample_size,
        class_ratio=class_ratio,
        hazard=hazard,
        hyper=hyper,
        use_wandb=use_wandb, 
    )
    # Train the base model
    model_mgr.train_hazard_model()
    # model_mgr.train_baseline_model()
    model_mgr.make_hazard_map()






   