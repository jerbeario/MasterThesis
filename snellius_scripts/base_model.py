import logging
import logging.handlers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os
import random
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_curve
    )
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
            "wildfire_test" : "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_wildfire_Europe.npy"
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
class BalancedBatchSampler:
    """Samples batches ensuring positive/negative balance for large datasets."""
    def __init__(self, labels, batch_size):
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        self.batch_size = batch_size
        
    def __iter__(self):
        # Shuffle indices each epoch
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        
        # Create balanced batches
        pos_per_batch = self.batch_size // 2
        neg_per_batch = self.batch_size - pos_per_batch
        
        # Number of complete batches we can make
        n_pos_batches = len(self.pos_indices) // pos_per_batch
        n_neg_batches = len(self.neg_indices) // neg_per_batch
        n_batches = min(n_pos_batches, n_neg_batches)
        
        for i in range(n_batches):
            pos_batch = self.pos_indices[i*pos_per_batch:(i+1)*pos_per_batch]
            neg_batch = self.neg_indices[i*neg_per_batch:(i+1)*neg_per_batch]
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)  # Shuffle within batch
            yield batch.tolist()
            
    def __len__(self):
        return min(len(self.pos_indices) // (self.batch_size // 2), 
                  len(self.neg_indices) // (self.batch_size - self.batch_size // 2))

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
class UNet(nn.Module):
    def __init__(self, logger, device, num_vars, filters=64, n_layers=4, activation=nn.ReLU(), 
                 dropout=True, drop_value=0.3, kernel_size=3, pool_size=2, patch_size=5):
        """
        UNet architecture for hazard susceptibility modeling.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/channels
            filters: Base number of filters (will be doubled in each layer)
            n_layers: Number of downsampling/upsampling layers
            activation: Activation function
            dropout: Whether to use dropout
            drop_value: Dropout probability
            kernel_size: Kernel size for convolutions
            pool_size: Pooling size for downsampling
            patch_size: Size of the input neighborhood
        """
        super(UNet, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        self.filters = filters
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.drop_value = drop_value
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.patch_size = patch_size
        
        self.logger.info(f"Initializing UNet with {num_vars} input variables and {n_layers} layers")
        
        # Input layer: Process each variable separately
        self.var_blocks = nn.ModuleList()
        for _ in range(self.num_vars):
            # Initial preprocessing for each variable
            self.var_blocks.append(nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm2d(filters),
                activation,
                nn.Conv2d(filters, filters, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm2d(filters),
                activation
            ))
        
        # Feature fusion layer
        self.fusion = nn.Conv2d(filters * num_vars, filters, kernel_size=1)
        
        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()
        current_filters = filters
        
        for i in range(n_layers):
            next_filters = current_filters * 2
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(current_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation,
                nn.Conv2d(next_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation
            ))
            self.pool_blocks.append(nn.MaxPool2d(pool_size))
            current_filters = next_filters
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_filters, current_filters * 2, kernel_size, padding='same'),
            nn.BatchNorm2d(current_filters * 2),
            activation,
            nn.Conv2d(current_filters * 2, current_filters * 2, kernel_size, padding='same'),
            nn.BatchNorm2d(current_filters * 2),
            activation,
            nn.Dropout2d(drop_value) if dropout else nn.Identity()
        )
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        current_filters = current_filters * 2
        
        for i in range(n_layers):
            next_filters = current_filters // 2
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(current_filters, next_filters, kernel_size=pool_size, 
                                stride=pool_size, padding=0),
                nn.BatchNorm2d(next_filters),
                activation
            ))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(current_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation,
                nn.Conv2d(next_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation
            ))
            current_filters = next_filters
        
        # Classification head
        # Calculate the output size based on input neighborhood and operations
        patch_size = patch_size
        final_size = patch_size // (pool_size ** n_layers) if patch_size % (pool_size ** n_layers) == 0 else 1
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(filters, 256),
            nn.BatchNorm1d(256),
            activation,
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            activation,
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """Forward pass of the UNet model."""
        # Split input into separate variable channels
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        
        # Process each variable through its own block
        var_features = []
        for i, (block, inp) in enumerate(zip(self.var_blocks, var_inputs)):
            var_features.append(block(inp))
        
        # Concatenate and fuse features from all variables
        x = torch.cat(var_features, dim=1)
        x = self.fusion(x)
        
        # Store encoder outputs for skip connections
        enc_features = []
        
        # Encoder path
        for enc_block, pool_block in zip(self.enc_blocks, self.pool_blocks):
            # Save output before pooling for skip connection
            enc_features.append(x)
            # Apply convolution block then pooling
            x = enc_block(x)
            x = pool_block(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (up_block, dec_block) in enumerate(zip(self.up_blocks, self.dec_blocks)):
            # Upsample
            x = up_block(x)
            
            # Get corresponding encoder feature map
            skip_feature = enc_features[-(i+1)]
            
            # Handle size mismatch (if any)
            if x.shape != skip_feature.shape:
                # Center crop or pad to match
                diff_h = skip_feature.size(2) - x.size(2)
                diff_w = skip_feature.size(3) - x.size(3)
                
                if diff_h > 0 and diff_w > 0:
                    skip_feature = skip_feature[:, :, diff_h//2:-(diff_h//2), diff_w//2:-(diff_w//2)]
                elif diff_h < 0 and diff_w < 0:
                    padding = [-diff_h//2, -diff_h-(-diff_h//2), -diff_w//2, -diff_w-(-diff_w//2)]
                    skip_feature = F.pad(skip_feature, padding)
            
            # Concatenate for skip connection
            x = torch.cat([x, skip_feature], dim=1)
            
            # Apply convolution block
            x = dec_block(x)
        
        # Final classification
        outputs = self.classification_head(x)
        return torch.sigmoid(outputs)

# CNN architecture for hazard susceptibility modeling
class CNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters, n_layers, activation, dropout, drop_value, kernel_size, pool_size, patch_size):
        super(CNN, self).__init__()
        self.logger = logger
        self.device = device
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
    

class BaseModel():
    def __init__(self, device, hazard, region, variables, patch_size, batch_size, model_architecture, 
                 logger, seed, use_wandb, sample_size, num_workers, early_stopping, patience, min_delta):
        super(BaseModel, self).__init__()
        self.use_wandb = use_wandb
        self.logger = logger
        self.device = device
        self.hazard = hazard
        self.region = region
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_vars = len(variables)
        self.variables = variables
        self.seed = seed
        self.name_model = hazard + '_' + region + '_base_model'

        self.model_architecture = model_architecture

        # Data
        self.num_workers = num_workers # number of workers for data loading
        self.sample_size = sample_size # fraction of the dataset to use for training
        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()
        
        # Hyperparameters
        self.learning_rate = 0.0001
        self.filters = 16
        self.n_layers = 3
        self.drop_value = 0.41
        self.kernel_size = 3
        self.pool_size = 2  
        self.activation = torch.nn.ReLU()
        self.dropout = True
        self.n_nodes = 128

        # Training
        self.epochs = 100
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        # WandB 
        if self.use_wandb:
            wandb.init(
                    project=f"{hazard}_susceptibility",
                    name=f"{self.model_architecture}_base_model",
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

        # Build the CNN architecture
        self.model = self.design_basemodel(architecture=self.model_architecture)
        self.model.to(self.device)
        # self.logger.info(f"fi:{self.filters} ly:{self.n_layers} dv:{self.drop_value} lr:{self.learning_rate}")

    def design_basemodel(self, architecture='CNN'):
        """
        Define the CNN architecture in PyTorch.
        """
        self.logger.info('Building architecture')
        if architecture == 'MLP':
            self.logger.info('Using MLP architecture')
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

        elif architecture == 'CNN':
            self.logger.info('Using CNN architecture')
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
        elif architecture == 'UNet':
            self.logger.info('Using UNet architecture')
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
        elif architecture == 'SimpleCNN':
            self.logger.info('Using SimpleCNN architecture')
            model = SimpleCNN(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                filters=self.filters,
                dropout=self.dropout,
                drop_value=self.drop_value,
                patch_size=self.patch_size
            )

        return model
    
    def load_dataset(self):
        """
        
        Preprocess the data for the model, using the dataset class.

        """
        
        # loading partition map 
        # TODO generalize for other hazard partition maps
        self.logger.info('Loading partition map')
        partition_map = np.load(f'Input/{self.region}/partition_map/partition_map.npy')
        partition_shape = partition_map.shape

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

        # Create custom balanced batch sampler
        batch_sampler = BalancedBatchSampler(train_labels, self.batch_size)

        # Create DataLoader with the custom batch sampler
        train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_sampler=batch_sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        self.logger.info(f"Test dataset size: {len(test_dataset)}")

        # Create DataLoaders for training and validation

        return train_loader, val_loader, test_loader

    def train(self):
        """
        Train the model using the provided data loaders.
        """
        self.logger.info(f'Training the mode with :{len(self.train_loader)*self.batch_size} samples ')
        self.logger.info(f"Training with {self.num_vars} variables, {self.batch_size} batch size, and {self.patch_size} neighborhood size")
        if self.use_wandb:
            wandb.watch(self.model, log="all")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float('inf')
        best_model_state = None
        epoch_without_improvement = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_accuracy = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Reshape Labels to match output 
                labels = labels.unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy(outputs, labels)
                accuracy = self.safe_accuracy(labels, outputs)

                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Step the optimizer
                optimizer.step()

                train_loss += loss.item()
                train_accuracy += accuracy.item()

            
            # Calculate average metrics
            train_loss /= len(self.train_loader)
            train_accuracy /= len(self.train_loader)

            
            # Evaluate on validation data
            val_loss, val_accuracy = self.evaluate(self.val_loader)

            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save the best model
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.name_model}_best.pth")
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
                self.logger.info(f"  New best model saved (lowest validation loss)")
            else:
                epochs_without_improvement += 1
                self.logger.info(f"  No improvement for {epochs_without_improvement} epochs")
                
                # Early stopping check
                if self.early_stopping and epochs_without_improvement >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break


            # Log metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": self.learning_rate
                })
        
    def evaluate(self, data_loader):
        """
        Evaluate the model on the provided data loader.
        
        Returns:
            Tuple of (loss, accuracy, f1)
        """
        self.model.eval()
        val_loss = 0.0
        val_accuracy = 0.0


        with torch.no_grad():
            self.logger.info(f'Evaluating the model with :{len(data_loader)*self.batch_size} samples ')
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate metrics
                # loss = self.safe_binary_crossentropy(labels, outputs)
                loss = F.binary_cross_entropy(outputs, labels)

                accuracy = self.safe_accuracy(labels, outputs)


                
                val_loss += loss.item()
                val_accuracy += accuracy.item()


        # Calculate average metrics
        val_loss /= len(data_loader)
        val_accuracy /= len(data_loader)
        
        return val_loss, val_accuracy

    def predict(self, test_loader):
        """
        Make predictions using the trained model.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)
        
    def testing(self):
        """
        Thoroughly test the model on the test dataset and output comprehensive metrics.
        """
        self.logger.info('Starting comprehensive model testing')
        self.model.eval()
        
        # Collect all predictions and ground truth
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                # Store predictions and labels
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())
                y_pred.extend((outputs > 0.5).cpu().numpy())
        
        # Convert to numpy arrays for easier manipulation
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.array(y_pred)
        
       
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
        except ValueError:
            self.logger.warning("Could not calculate AUC metrics - possibly only one class present in test set")
            auc_roc = np.nan
            avg_precision = np.nan
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Log metrics
        self.logger.info(f"Test Results for {self.hazard} {self.model_architecture} model:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1 Score: {f1:.4f}")
        self.logger.info(f"  AUC-ROC: {auc_roc:.4f}")
        self.logger.info(f"  Average Precision: {avg_precision:.4f}")
        self.logger.info(f"  Confusion Matrix: \n{cm}")
        
        # Calculate class imbalance
        class_counts = np.bincount(y_true.astype(int).flatten())
        class_proportions = class_counts / np.sum(class_counts)
        self.logger.info(f"  Class distribution: {class_counts}, {class_proportions}")
        
        # Generate visualizations
        # 1. Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{self.hazard} {self.model_architecture} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.name_model}_confusion_matrix.png')
        
        # 2. ROC curve
        if not np.isnan(auc_roc):
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{self.hazard} {self.model_architecture} ROC Curve')
            plt.legend()
            plt.savefig(f'{self.name_model}_roc_curve.png')
        
        # 3. Precision-Recall curve
        if not np.isnan(avg_precision):
            plt.figure(figsize=(8, 6))
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(recall_curve, precision_curve, label=f'AP = {avg_precision:.4f}')
            # Add baseline based on class imbalance
            baseline = class_proportions[1] if len(class_proportions) > 1 else 0
            plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline = {baseline:.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{self.hazard} {self.model_architecture} Precision-Recall Curve')
            plt.legend()
            plt.savefig(f'{self.name_model}_pr_curve.png')
        
        # Save metrics to CSV
        metrics_dict = {
            'Model': [self.model_architecture],
            'Hazard': [self.hazard],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1],
            'AUC-ROC': [auc_roc],
            'Avg_Precision': [avg_precision],
            'Class_0_Count': [class_counts[0]],
            'Class_1_Count': [class_counts[1] if len(class_counts) > 1 else 0],
        }
        df = pd.DataFrame(metrics_dict)
        
        # Create directory if it doesn't exist
        os.makedirs(f'Output/{self.region}/{self.hazard}', exist_ok=True)
        
        # Save to CSV
        df.to_csv(f'Output/{self.region}/{self.hazard}/{self.model_architecture}_test_metrics.csv', index=False)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_auc_roc": auc_roc,
                "test_avg_precision": avg_precision,
                "test_confusion_matrix": wandb.Image(f'{self.name_model}_confusion_matrix.png'),
                "test_roc_curve": wandb.Image(f'{self.name_model}_roc_curve.png') if not np.isnan(auc_roc) else None,
                "test_pr_curve": wandb.Image(f'{self.name_model}_pr_curve.png') if not np.isnan(avg_precision) else None,
                "test_class_balance": wandb.plot.bar(
                    wandb.Table(data=[[str(i), count] for i, count in enumerate(class_counts)],
                                columns=["class", "count"]),
                    "class", "count", title="Test Set Class Distribution"
                )
            })
            
            # Also log feature importance if available
            if hasattr(self, 'permutation_feature_importance') and len(self.variables) <= 20:
                self.logger.info("Calculating feature importance...")
                try:
                    # Get baseline score
                    baseline_score = roc_auc_score(y_true, y_prob)
                    
                    # Get feature importances using a small subset for efficiency
                    subset_size = min(5000, len(y_true))
                    X_subset = torch.stack([batch[0] for batch in list(self.test_loader)[:subset_size//self.batch_size]])
                    y_subset = y_true[:subset_size]
                    
                    feature_importances = self.permutation_feature_importance(
                        X_subset, y_subset, baseline_score, roc_auc_score)
                    
                    # Log feature importances
                    plt.figure(figsize=(10, 6))
                    plt.barh(self.variables, feature_importances)
                    plt.xlabel('Feature Importance')
                    plt.ylabel('Feature')
                    plt.title(f'{self.hazard} {self.model_architecture} Feature Importance')
                    plt.tight_layout()
                    plt.savefig(f'{self.name_model}_feature_importance.png')
                    
                    # Log to wandb
                    wandb.log({
                        "feature_importance": wandb.Image(f'{self.name_model}_feature_importance.png')
                    })
                except Exception as e:
                    self.logger.warning(f"Could not calculate feature importance: {e}")
        
        self.logger.info("Comprehensive testing completed")    

        # Save the model in the exchangeable ONNX format
        if self.use_wandb:
            save_path = f"Output/{self.region}/{self.hazard}/{self.name_model}_best.onnx"
            self.logger.info('Saving model in ONNX format')
            inputs = torch.randn(1, self.num_vars, self.patch_size, self.patch_size).to(self.device)
            torch.onnx.export(self.model, inputs, save_path)
            wandb.save(save_path)

    def permutation_feature_importance(self, X, y, baseline_score, metric_fn):
        """
        Compute permutation feature importance.
        """
        feature_importances = []

        for i in range(X.shape[1]):
            X_permuted = X.clone()
            X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.size(0)), i]

            permuted_score = metric_fn(y, self.predict(X_permuted))
            importance = baseline_score - permuted_score
            feature_importances.append(importance)

        return feature_importances
    
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
                # 'filters': {
                #     'values': [8, 16, 32, 64]
                # },
                'n_layers': {
                    'values': [1, 2, 3, 4,]
                },
                'n_nodes': {
                    'values': [16, 32, 64, 128, 256]
                },
                'drop_value': {
                    'distribution': 'uniform',
                    'min': 0.2,
                    'max': 0.5
                },
                'sample_size': {
                    'values': [0.01, 0.05, 0.1, 0.5, 1]
                },
            }
        }
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"{self.hazard}_{self.model_architecture}_sweep"
        )
         
        # Start the sweep agent
        wandb.agent(sweep_id, self.sweep_train, count=10)  # Run 10 trials
        self.logger.info(f"Hyperparameter sweep completed with ID: {sweep_id}")
 

    def sweep_train(self):
        # Initialize new wandb run
        with wandb.init() as run:
            # Get hyperparameters from wandb config
            config = wandb.config
            
            # Update model hyperparameters
            self.learning_rate = config.learning_rate
            # self.filters = config.filters
            self.n_layers = config.n_layers 
            self.n_nodes = config.n_nodes
            self.drop_value = config.drop_value
            # self.batch_size = config.batch_size
            self.sample_size = config.sample_size
            
            # Log configuration for this run
            self.logger.info(f"Sweep run with: lr={self.learning_rate}, "
                        # f"filters={self.filters}, "
                        f"n_nodes={self.n_nodes}, "
                        f"n_layers={self.n_layers}, "
                        f"dropout={self.drop_value}, "
                        f"sample_size={self.sample_size}, "
            )
            
            # Recreate data loaders with new batch size
            self.train_loader, self.val_loader, self.test_loader = self.load_dataset()
            
            # Rebuild model with new hyperparameters
            self.model = self.design_basemodel(architecture=self.model_architecture)
            self.model.to(self.device)
            
            # Set smaller number of epochs for sweep runs
            original_epochs = self.epochs
            self.epochs = 20 # Reduced epochs for sweep runs
            
            # Use the existing training loop
            self.train()
            
            # Restore original epochs
            self.epochs = original_epochs
            
            # Evaluate on test set
            test_loss, test_accuracy = self.evaluate(self.test_loader)
            wandb.log({
                "final_test_loss": test_loss,
                "final_test_accuracy": test_accuracy
            })    

    @staticmethod
    def safe_binary_crossentropy(y_true, y_pred):
        """
        Binary cross-entropy loss with NaN handling.
        """
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(y_pred, y_true)

    @staticmethod
    def safe_mse(y_true, y_pred):
        """
        Mean squared error (MSE) loss with NaN handling.
        """
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        return F.mse_loss(y_pred, y_true)

    @staticmethod
    def safe_mae(y_true, y_pred):
        """
        Mean absolute error (MAE) loss with NaN handling.
        """
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        return torch.mean(torch.abs(y_pred - y_true))
    
    @staticmethod
    def safe_accuracy(y_true, y_pred, threshold=0.5):
        """
        Calculate accuracy with NaN handling.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            Accuracy score
        """
        # Handle NaN values
        # y_pred = torch.nan_to_num(y_pred, nan=0.0)
        # y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Convert predictions to binary
        y_pred_binary = (y_pred >= threshold).float()
        
        # Calculate accuracy
        correct = (y_pred_binary == y_true).float()
        accuracy = torch.mean(correct)
        return accuracy
    
    @staticmethod
    def safe_f1_score(y_true, y_pred, threshold=0.5, epsilon=1e-7):
        """
        Calculate F1 score with NaN handling.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            threshold: Classification threshold (default: 0.5)
            epsilon: Small value to avoid division by zero
        
        Returns:
            F1 score
        """
        # Handle NaN values
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Convert predictions to binary
        y_pred_binary = (y_pred >= threshold).float()
        
        # Calculate true positives, false positives, false negatives
        tp = torch.sum(y_true * y_pred_binary)
        fp = torch.sum((1 - y_true) * y_pred_binary)
        fn = torch.sum(y_true * (1 - y_pred_binary))
        
        # Calculate precision and recall
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1

    def run(self):
        """
        Run the model training and evaluation.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # TODO wand implementation
        self.main()

    def main(self):
        """
        Main function to run the model training and evaluation.
        """
        self.train()

        if self.use_wandb:  
            wandb.finish(exit_code=1)
    
        self.logger.info(f"Main done")

class ModelMgr:
    def __init__(self, region='Europe', batch_size=1024, patch_size=5, base_model='CNN' , sample_size = 1, 
                test='Europe', prep='model', hazard='Wildfire', hyper=False, model_choice='base', partition='spatial', use_wandb=True):
        
        
        self.early_stopping = True
        self.patience = 5 
        self.min_delta=0.001
        self.use_wandb = use_wandb
        self.hazard = hazard
        self.region = region
        self.batch_size = batch_size
        self.name_model = 'susceptibility'
        self.missing_data_value = 0
        self.sample_ratio = 0.8
        self.test_split = 0.15
        self.sample_size = sample_size
        self.num_workers = 8
        self.patch_size = patch_size
        self.hyper = hyper
        self.test = test
        self.model_choice = model_choice
        self.model_architecture = base_model # 'CNN' or 'UNet' or 'SimpleCNN'
        self.partition = partition
        if self.hazard == 'Landslide':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation', 'accuflux', 'HWSD', 'road', 'GEM', 'curvature', 'GLIM']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous', 'categorical', 'label', 'continuous', 'continuous', 'categorical']
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
        elif self.hazard == "Wildfire":
            # temperature_daily, NDVI, landcover, elevation, wind_speed, fire_weather, soil_moisture(root or surface)
            self.variables = ['temperature_daily', 'NDVI', 'landcover', 'elevation', 'wind_speed_daily', 'fire_weather', 'soil_moisture_root']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous']
        elif self.hazard == 'Multihazard':
            # self.variables = ['drought', 'extreme_wind', 'fire_weather', 'heatwave', 'pga', 'volcano', 'Flood_base_model', 'Landslide_base_model', 'Tsunami_base_model']
            self.variables = ['drought', 'extreme_wind', 'fire_weather', 'heatwave', 'jshis', 'volcano', 'Flood_base_model', 'Landslide_base_model', 'Tsunami_base_model']
            self.var_types = ['continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous']
        elif self.hazard == 'Wildfire_test':
            # temperature_daily, NDVI, landcover, elevation, wind_speed, fire_weather, soil_moisture(root or surface)
            self.variables = ['test']
            self.var_types = ['continuous']
        
        self.prep = prep
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
        

        self.base_model_instance = BaseModel(
            device=self.device,
            hazard=self.hazard,
            region=self.region,
            variables=self.variables,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            sample_size=self.sample_size,
            model_architecture=self.model_architecture,
            num_workers=self.num_workers,
            logger=self.logger,
            seed=self.seed,
            use_wandb=self.use_wandb,
            early_stopping=self.early_stopping,
            patience=self.patience,
            min_delta=self.min_delta
            
        )
    

    def set_logger(self, verbose=True):
        """
        Set-up the logging system, exit if this fails
        """
        # assign logger file name and output directory
        datelog = time.ctime()
        datelog = datelog.replace(':', '_')
        reference = f'CNN_ls_susc_{self.test}'

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


    def train_base_model(self):
        if self.prep != 'stack':
            if self.hyper:
                self.base_model_instance.HypParOpt()
            else:
                self.logger.info('Training base model')
                self.base_model_instance.run()
                self.base_model = self.base_model_instance.model
        else:
            self.logger.info('Only works when prep!=stack')

if __name__ == "__main__":
    # Example configuration
    region = 'Europe'
    hazard = 'Wildfire'
    batch_size = 1024
    patch_size = 1
    base_model = 'MLP'
    sample_size = 0.01
    hyper = True
    use_wandb = True, 

    # Initialize the model manager
    model_mgr = ModelMgr(
        region=region,
        batch_size=batch_size,
        patch_size=patch_size,
        base_model=base_model,
        sample_size=sample_size,
        hazard=hazard,
        hyper=hyper,
        use_wandb=use_wandb, 
    )
    # Train the base model
    model_mgr.train_base_model()

   