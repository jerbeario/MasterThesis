import logging
import logging.handlers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import wandb

import seaborn as sns


class HazardDataset(Dataset):
    def __init__(self, hazard, variables, patch_size=5):
        """
        Custom Dataset for loading hazard-specific features and labels as patches.

        Parameters:
        - hazard (str): The hazard type (e.g., "wildfire").
        - patch_size (int): The size of the patch (n x n) around the center cell.
        """
        self.hazard = hazard.lower()
        print(f"Loading {self.hazard} dataset...")
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
    

class CNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters, n_layers, activation, dropout, drop_value, kernel_size, pool_size, neighborhood_size):
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
        self.neighborhood_size = neighborhood_size

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
    
    import torch.utils


class BaseModel():
    def __init__(self, device, hazard, region, 
                 variables, neighborhood_size, train_loader, val_loader, logger, seed):
        super(BaseModel, self).__init__()
        self.logger = logger
        self.device = device
        self.hazard = hazard
        self.region = region
        self.neighborhood_size = neighborhood_size
        self.num_vars = len(variables)
        self.variables = variables
        self.seed = seed
        self.name_model = 'wildfire'


        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.test_loader = ModelMgr_instance.test_loader
        
    
        
        # Hyperparameters
        self.learning_rate = 0.0001
        self.filters = 32
        self.n_layers = 3
        self.drop_value = 0.41
        self.kernel_size = 3
        self.pool_size = 2  
        self.activation = torch.nn.ReLU()
        self.dropout = True

        



        # Build the CNN architecture
        self.model = self.design_basemodel()
        self.model.to(self.device)
        # self.logger.info(f"fi:{self.filters} ly:{self.n_layers} dv:{self.drop_value} lr:{self.learning_rate}")

    def design_basemodel(self):
        """
        Define the CNN architecture in PyTorch.
        """
        self.logger.info('Building architecture')
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
            neighborhood_size=self.neighborhood_size
        )

    


        return model

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
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        
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

    def train(self, epochs=10):
        """
        Train the model using the provided data loaders.
        """
        self.logger.info('Training the model')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float('inf')
        best_val_f1 = 0.0

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs} starting...")
            self.model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            train_f1 = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Reshape Labels to match output 
                labels = labels.unsqueeze(1)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.safe_binary_crossentropy(labels, outputs)
                
                # Calculate metrics
                accuracy = self.safe_accuracy(labels, outputs)
                f1 = self.safe_f1_score(labels, outputs)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_accuracy += accuracy.item()
                train_f1 += f1.item()
            
            # Calculate average metrics
            train_loss /= len(self.train_loader)
            train_accuracy /= len(self.train_loader)
            train_f1 /= len(self.train_loader)
            
            # Evaluate on validation data
            val_loss, val_accuracy, val_f1 = self.evaluate(self.val_loader)

            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.logger.info(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.name_model}_best.pth")
                self.logger.info(f"  New best model saved (lowest validation loss)")

                        
            # Save based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.name_model}_best_f1.pth")
                self.logger.info(f"  New best model saved (highest validation F1)")

    def evaluate(self, data_loader):
        """
        Evaluate the model on the provided data loader.
        
        Returns:
            Tuple of (loss, accuracy, f1)
        """
        self.model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_f1 = 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate metrics
                loss = self.safe_binary_crossentropy(labels, outputs)
                accuracy = self.safe_accuracy(labels, outputs)
                f1 = self.safe_f1_score(labels, outputs)
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                val_f1 += f1.item()

        # Calculate average metrics
        val_loss /= len(data_loader)
        val_accuracy /= len(data_loader)
        val_f1 /= len(data_loader)
        
        return val_loss, val_accuracy, val_f1

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
        Test the model on the test dataset.
        """
        # TODO rewrite this function

        # # Evaluate the model on the validation data
        # self.logger.info('Testing')


        # y_pred = self.base_model.predict({'input_' + str(i+1): self.ModelMgr_instance.test_data[i] for i in range(len(self.ModelMgr_instance.test_data))})
        # y_pred = np.squeeze(y_pred, axis=(1))
        # y_true = np.squeeze(self.ModelMgr_instance.test_labels, axis=(1,2))


        # #TODO check if torch tensor is needed

        # # # Calculate Binary Cross-Entropy
        # # y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        # # y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        # # bce = tf.keras.backend.binary_crossentropy(y_true_tf, y_pred_tf)
        # # self.bce_test = tf.reduce_mean(bce).numpy()
        # # self.logger.info(f"BCE: {self.bce_test}")

        # # Metrics
        # self.mae = mean_absolute_error(y_true, y_pred)
        # self.mse = mean_squared_error(y_true, y_pred)
        # self.logger.info(f"MAE: {self.mae}")
        # self.logger.info(f"MSE: {self.mse}")

        # # Create a dictionary to store values with names
        # metrics_dict = {'MAE': [self.mae], 'MSE': [self.mse]}
        # df = pd.DataFrame(metrics_dict)

        # # Write the values to a text file
        # df.to_csv(f'Output/{self.region}/{self.hazard}/config_{self.ModelMgr_instance.test}_basemodel.csv', index=False)

        # # Store in W&B
        # wandb.log({"MAE_test": self.mae})
        # wandb.log({"MSE_test": self.mse})
        # wandb.log({"BCE_test": self.bce_test})

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
        # if self.ModelMgr_instance.hyper:
        #     self.base_model = False
        #     if self.ModelMgr_instance.partition == 'random':
        #         self.ModelMgr_instance.preprocess() 
        #     wandb.init()
        #     self.n_layers = wandb.config.layers
        #     self.filters = wandb.config.filters
        #     self.learning_rate = wandb.config.lr
        #     self.drop_value = wandb.config.dropout

        # Develop a basemodel
        # self.design_basemodel()
        # Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.logger.info(f"Optimizer: {optimizer}")
        self.train()
        # self.predict()
        # self.testing()

        # if self.ModelMgr_instance.hyper:
        #     new_row = pd.DataFrame([{
        #         "layers": wandb.config.layers,
        #         "filters": wandb.config.filters,
        #         "lr": wandb.config.lr,
        #         "dropout": wandb.config.dropout,
        #         "val_loss": self.bce_val,
        #         "MAE": self.mae,
        #         "MSE": self.mse,
        #     }])
        #     self.hyper_df = pd.concat([self.hyper_df, new_row], ignore_index=True)
        #     self.hyper_df.to_csv(f"Output/{self.region}/{self.hazard}/Sweep_results_BaseModel_{self.ModelMgr_instance.test}.csv", index=False)
        #     if self.bce_val < self.bce_val_best:
        #         self.bce_val_best = self.bce_val
        self.logger.info(f"Main done")

class ModelMgr:
    def __init__(self, region='Europe', test='Europe', prep='model', hazard='Wildfire', hyper=False, model_choice='base', partition='spatial'):
        self.hazard = hazard
        self.region = region
        self.batch_size = 32
        self.name_model = 'susceptibility'
        self.missing_data_value = 0
        self.sample_ratio = 0.8
        self.test_split = 0.15
        self.neighborhood_size = 5
        self.hyper = hyper
        self.test = test
        self.model_choice = model_choice
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
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"Num GPUs Available: {torch.cuda.device_count()}")
            self.logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

        self.logger.info(f"Torch version: {torch.__version__}")
    
        # # Configure memory growth for both GPUs to avoid memory errors
        # for gpu in physical_devices:
        #     tf.config.experimental.set_memory_growth(gpu, True)

        # self.logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

        # Test simple GPU operation
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=self.device)
        c = torch.matmul(a, b)
        self.logger.info(f"Matrix multiplication result: {c}")

        # sys.exit(0)
        

        # TODO clean up dataloaders

        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()


       # Load the dataset
        # self.logger.info('Loading dataset')
        # dataset = HazardDataset(hazard=self.hazard, patch_size=self.neighborhood_size)

        # # Extract labels from the dataset
        # labels = [dataset[i][1].item() for i in range(len(dataset))]  # Extract labels for stratified sampling

        # # Perform stratified train-validation split
        # train_indices, val_indices = train_test_split(
        #     range(len(dataset)),
        #     test_size=0.20,  # 20% validation
        #     stratify=labels,  # Ensure class balance
        #     random_state=self.seed
        # )

        # # Create subsets using the indices
        # train_dataset = Subset(dataset, train_indices)
        # val_dataset = Subset(dataset, val_indices)


        # self.logger.info(f"Train dataset size before balancing: {len(train_dataset)}")
        
        # train_labels = [dataset[i][1].item() for i in range(len(train_dataset))]  # Extract labels for stratified sampling
        # train_labels = np.array(train_labels).astype(np.int64)

        # # Compute class weights
        # class_counts = np.bincount(train_labels)
        # class_weights = 1.0 / class_counts
        # sample_weights = class_weights[train_labels]

        # # Create a WeightedRandomSampler
        # sampler = WeightedRandomSampler(
        #     weights=sample_weights,
        #     num_samples=len(sample_weights)//100,
        #     replacement=True  # Allow replacement to ensure balanced sampling
        # )

        # Create DataLoaders for training and validation
        # TODO generalize batch size
        # batch_size = 32
        # self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, sampler=sampler)
        # self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



        self.base_model_instance = BaseModel(
            device=self.device,
            hazard=self.hazard,
            region=self.region,
            variables=self.variables,
            neighborhood_size=self.neighborhood_size,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            logger=self.logger,
            seed=self.seed
            
        )
        # self.ensemble_model_instance = EnsembleModel(self)
        # self.meta_model_instance = MetaModel(self)

        # if not (self.hyper and self.partition == 'random'):
        #     self.preprocess()
        # self.preprocess()

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




    def load_dataset(self):
        """
        
        Preprocess the data for the model, using the dataset class.

        """
        # loading partition map 
        self.logger.info('Loading partition map')
        partition_map = np.load(f'Input/{self.region}/partition_map/full_balanced_partition_map_sub_countries.npy')
        partition_shape = partition_map.shape

        dataset = HazardDataset(hazard=self.hazard, variables=self.variables, patch_size=self.neighborhood_size)
        
        idx_transform = np.array([[partition_shape[1]],[1]])

        train_indices = (np.argwhere(partition_map == 1) @ idx_transform).flatten()
        val_indices = (np.argwhere(partition_map == 2) @ idx_transform).flatten()
        test_indices = (np.argwhere(partition_map == 3) @ idx_transform).flatten()

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        self.logger.info(f"Test dataset size: {len(test_dataset)}")

        # Create DataLoaders for training and validation

        return train_loader, val_loader, test_loader

        
        





    # TODO check if this is needed
    # def preprocess(self):
    #     # Load data from .npy files
    #     # Prepare data for the CNN
    #     input_data = []
    #     spatial_split = False
    #     if self.prep == 'model':
    #         for var, var_type in zip(self.variables, self.var_types):
    #             input_data.append(self.load_normalize(var, var_type=var_type))
    #     elif self.prep == 'stack':
    #         for i in range(self.ensemble_nr):
    #             input_data.append(self.load_normalize(f'model_{i}', var_type='label', crop=False)[0])
    #     elif self.prep == 'multi':
    #         for var, var_type in zip(self.variables, self.var_types):
    #             input_data.append(self.load_normalize(var, var_type=var_type))
    #     elevation = self.load_normalize('elevation', var_type='mask')
    #     input_data = np.array(input_data)

    #     if self.hazard == 'Landslide':
    #         labels, output_shape, spatial_split = self.load_normalize('ldm', var_type='label')
    #     elif self.hazard == 'Flood':
    #         labels, output_shape, spatial_split = self.load_normalize('flood_surge', var_type='label')
    #     elif self.hazard == 'Tsunami':
    #         labels, output_shape, spatial_split = self.load_normalize('tsunami', var_type='label')
    #     elif self.hazard == 'Multihazard':
    #         labels, output_shape = self.load_normalize('multi_hazard', var_type='continuous')

    #     # List to store the indices
    #     self.logger.info('Extracting indices')
    #     indices_with_values = []
    #     original_shape = labels.shape
    #     self.logger.info(f"Input shape: {input_data.shape}")
    #     self.logger.info(f"Label shape: {labels.shape}")
    #     if spatial_split is not False:
    #         self.logger.info(f"Spatial shape: {spatial_split.shape}")
    #     self.logger.info(f"Elevation shape: {elevation.shape}")

    #     # Iterate over the array   ############## THIS SHOULD BE DONE IN LOAD NORMALIZE
    #     for idx, data_map in enumerate(elevation):
    #         if np.any(data_map > -9999): ###### SO WOULD NOT BE BETTER TO CHECK ALL MAPS AND MAKE NODATA=0???? FOR MIN MAX SCALER***
    #             indices_with_values.append(idx)

    #     # Extract data based on the indices
    #     input_data = input_data[:, indices_with_values]
    #     labels = labels[indices_with_values]
    #     if spatial_split is not False:
    #         spatial_split = spatial_split[indices_with_values]

    #     self.logger.info(f"Min value INPUT: {np.min(input_data)}")
    #     self.logger.info(f"Max value INPUT: {np.max(input_data)}")
    #     self.logger.info(f"Min value LABEL: {np.min(labels)}")
    #     self.logger.info(f"Max value LABEL: {np.max(labels)}")
    #     self.logger.info(f"Input shape: {input_data.shape}")
    #     self.logger.info(f"Label shape: {labels.shape}")
    #     if spatial_split is not False:
    #         self.logger.info(f"Spatial shape: {spatial_split.shape}")

    #     # for i in range(len(input_data)):
    #     #     variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation', 'accuflux', 'HWSD', 'road', 'GEM', 'curvature', 'GLIM']
    #     #     self.logger.info(f"Variable: {variables[i]}")
    #     #     self.logger.info(f"Min value INPUT: {np.min(input_data[i])}")
    #     #     self.logger.info(f"Max value INPUT: {np.max(input_data[i])}")
    #     # sys.exit(0)

    #     if self.partition == 'random':
    #         # Generate random indices from the first axis
    #         if not os.path.exists(f'Output/{self.region}/{self.hazard}/{self.hazard}_Susceptibility_{model_choice}_model_rnd_ind_{self.test}.npy') or self.hyper:
    #             train_indices = random.sample(range(input_data.shape[1]), int(input_data.shape[1] * self.sample_ratio))
    #             train_indices = np.save(f'Output/{self.region}/{self.hazard}/{self.hazard}_Susceptibility_{model_choice}_model_rnd_ind_{self.test}.npy', train_indices)

    #         train_indices = np.load(f'Output/{self.region}/{self.hazard}/{self.hazard}_Susceptibility_{model_choice}_model_rnd_ind_{self.test}.npy')

    #         # Create the test set of indices
    #         all_indices = set(range(input_data.shape[1]))
    #         complement_indices = list(all_indices - set(train_indices))

    #         test_indices = random.sample(complement_indices, int(input_data.shape[1] * self.test_split))
    #     elif self.partition == 'spatial':
    #         if not os.path.exists(f'Output/{self.region}/Susceptibility_spatial_partitioning_train.npy'):
    #             self.logger.info('INDICES')
    #             train_indices = np.where(spatial_split == 1)[0]
    #             self.logger.info(train_indices.shape)
    #             val_indices = np.where(spatial_split == 2)[0]
    #             self.logger.info(val_indices.shape)
    #             test_indices = np.where(spatial_split == 3)[0]
    #             self.logger.info(test_indices.shape)
    #             other_indices = np.where(spatial_split == 0)[0]
    #             self.logger.info(other_indices.shape)
                
    #             train_indices = np.save(f'Output/{self.region}/Susceptibility_spatial_partitioning_train.npy', train_indices)
    #             val_indices = np.save(f'Output/{self.region}/Susceptibility_spatial_partitioning_val.npy', val_indices)
    #             test_indices = np.save(f'Output/{self.region}/Susceptibility_spatial_partitioning_test.npy', test_indices)
            
    #         train_indices = np.load(f'Output/{self.region}/Susceptibility_spatial_partitioning_train.npy')
    #         val_indices = np.load(f'Output/{self.region}/Susceptibility_spatial_partitioning_val.npy')
    #         test_indices = np.load(f'Output/{self.region}/Susceptibility_spatial_partitioning_test.npy')

    #     self.input_data = input_data
    #     self.labels = labels

    #     # Store the selected indices in a new array
    #     model_inputs = input_data[:, train_indices]
    #     model_labels = labels[train_indices]

    #     test_data = input_data[:, test_indices]
    #     test_labels = labels[test_indices]

    #     self.train_indices = train_indices
    #     self.model_inputs = model_inputs
    #     self.input_data = input_data
    #     self.model_labels = model_labels
    #     self.labels = labels
    #     self.indices_with_values = indices_with_values
    #     self.original_shape = original_shape
    #     self.output_shape = output_shape
    #     self.test_data = test_data
    #     self.test_labels = test_labels

    #     if self.partition == 'spatial':
    #         val_data = input_data[:, val_indices]
    #         val_labels = labels[val_indices]
    #         self.val_data = val_data
    #         self.val_labels = val_labels

    # def load_normalize(self, var, var_type='continuous', crop=True):
    #     self.logger.info(f'Loading {var}')
    #     if var == 'landcover' or var == 'NDVI':
    #         feature_data = np.load(f'Input/Japan/npy_arrays/masked_{var}_japan_flat.npy').astype(np.float32)
    #     elif var == 'precipitation':
    #         feature_data = np.load(f'Input/Japan/npy_arrays/masked_{var}_daily_japan.npy').astype(np.float32)
    #     elif 'base_model' in var:
    #         feature_data = np.load(f'Output/{self.region}/{var[:-11]}/{self.test}_{var[:-11]}_Susceptibility_base_model.npy').astype(np.float32)
    #         crop = False
    #     elif 'model' in var:
    #         feature_data = np.load(f'Output/{self.region}/{self.hazard}/{self.test}_{self.hazard}_Susceptibility_ensemble_{var}.npy').astype(np.float32)
    #     else:
    #         feature_data = np.load(f'Input/Japan/npy_arrays/masked_{var}_japan.npy').astype(np.float32)
        
    #     if crop:
    #         if self.test == 'hokkaido':
    #             feature_data = feature_data[150:1700,3800:-200]
    #         elif self.test == 'sado':
    #             feature_data = feature_data[2755:2955,3525:3675]
        
    #     # factor_x, factor_y = int(feature_data.shape[0] / tile), int(feature_data.shape[1] / tile)
    #     output_shape = feature_data.shape
        
    #     # Initialize the scaler, fit and transform the data
    #     if var_type == 'continuous':
    #         scaler = MinMaxScaler(feature_range=(0, 1))
    #         scaled_feature = scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)
    #         scaled_feature = np.nan_to_num(scaled_feature, nan=self.missing_data_value)
        
    #     elif var_type == 'categorical':
    #         feature_data = np.nan_to_num(feature_data, nan=0)
    #         # Initialize the OneHotEncoder
    #         encoder = LabelEncoder()
    #         # Fit and transform the landcover data
    #         scaled_feature = encoder.fit_transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)

    #     elif var_type == 'label':
    #         scaled_feature = np.nan_to_num(feature_data, nan=self.missing_data_value)  # Convert nan to a specific value
    #         partition_map = np.load('Region/Japan/japan_prefecture_partitions_with_buffer.npy')
    #         partition_map = partition_map[0:5500,2300:8800]
    #         self.test_prefectures = [2, 6, 16, 10, 18, 34, 43, 39]
    #         self.val_prefectures = [7, 17, 23, 26, 32, 37, 44]
    #         self.train_prefectures = [i for i in range(1, 48) if i not in self.test_prefectures and i not in self.val_prefectures]
    #         spatial_split = []
        
    #     elif var_type == 'mask':
    #         scaled_feature = feature_data

    #     # Iterate through the array to extract sub-arrays
    #     scaled_feature_reshape = []
    #     for i in range(self.neighborhood_size, scaled_feature.shape[0] - self.neighborhood_size):
    #         for j in range(self.neighborhood_size, scaled_feature.shape[1] - self.neighborhood_size):
    #             ####### HERE SHOULD BE THE CHECK WITH ELEVATION
    #             sub_array = scaled_feature[i - self.neighborhood_size: i + self.neighborhood_size + 1, j - self.neighborhood_size: j + self.neighborhood_size + 1]
    #             if (var_type == 'label' and var != 'road') | (var == 'multi_hazard') | (self.prep == 'multi' and var_type != 'mask'):
    #                 center_value = sub_array[self.neighborhood_size, self.neighborhood_size]
    #                 scaled_feature_reshape.append(center_value)
    #                 if var_type == 'label':
    #                     if partition_map[i,j] in self.train_prefectures:
    #                         spatial_split.append(1)
    #                     elif partition_map[i,j] in self.val_prefectures:
    #                         spatial_split.append(2)
    #                     elif partition_map[i,j] in self.test_prefectures:
    #                         spatial_split.append(3)
    #                     else:
    #                         spatial_split.append(0)
    #                 # if var == 'HWSD':
    #                 #     print('check')
    #                 #     sys.exit(0)
    #             else:
    #                 scaled_feature_reshape.append(sub_array)

    #     # Convert the list of arrays to a numpy array
    #     scaled_feature_reshape = np.array(scaled_feature_reshape).astype(np.float32)
        
    #     # scaled_feature_reshape = scaled_feature.reshape((factor_x * factor_y, int(scaled_feature.shape[0] / factor_x), int(scaled_feature.shape[1] / factor_y), 1))
        
    #     if (var_type == 'label' and var != 'road'):
    #         return scaled_feature_reshape.reshape(-1, 1, 1), output_shape, np.array(spatial_split)
    #     elif var == 'multi_hazard':
    #         return scaled_feature_reshape.reshape(-1, 1, 1), output_shape
    #     else:
    #         return np.expand_dims(scaled_feature_reshape, axis=-1)

    def train_base_model(self):
        if self.prep != 'stack':
            if self.hyper:
                self.base_model_instance.HypParOpt()
            else:
                self.logger.info('Training base model')
                self.base_model_instance.run()
                self.base_model = self.base_model_instance.base_model
        else:
            self.logger.info('Only works when prep!=stack')
    
    # # TODO - convert to pyTorch
    # def xload_base_model(self):
    #     if self.prep != 'stack':
    #         self.base_model = keras.models.load_model(os.path.join(f'Output/{self.region}', self.hazard, f'base_model_{self.test}.tf'))
    #     else:
    #         self.logger.info('Only works when prep!=stack')
    
    # def train_ensemble_model(self):
    #     if self.prep != 'stack':
    #         if self.hyper:
    #             self.ensemble_model_instance.HypParOpt()
    #         else:
    #             self.ensemble_model_instance.run()
    #             self.combined_model = self.ensemble_model_instance.combined_model
    #     else:
    #         self.logger.info('Only works when prep!=stack')

    # def train_meta_model(self):
    #     if (self.prep == 'stack') | (self.prep == 'multi'):
    #         if self.hyper:
    #             self.meta_model_instance.HypParOpt()
    #         else:
    #             self.meta_model_instance.run()
    #             self.meta_model = self.meta_model_instance.meta_model
    #     else:
    #         self.logger.info('Only works when prep=stack | prep=multi')

    # # TODO - convert to pyTorch
    # def load_meta_model(self):
    #     if self.prep != 'stack':
    #         self.meta_model = keras.models.load_model(os.path.join(f'Output/{self.region}', self.hazard, f'meta_model_MLP_{self.test}.tf'))
    #     else:
    #         self.logger.info('Only works when prep!=stack')

    # def learning_to_stack(self):
    #     if self.prep == 'model':
    #         self.prep = 'stack'
    #         self.preprocess()
    #     else:
    #         self.logger.info('Only works when prep=model')

    # def plot(self, data, name='scaled_feature'):
    #     fig = plt.figure()
    #     plt.imshow(data, cmap='viridis')
    #     plt.title(name)
    #     plt.colorbar()
    #     plt.savefig(f'Output/{self.region}/{self.hazard}/{name.replace(" ", "_")}.png', dpi=1000)
    #     return fig

    # def plot_val_loss(self, history, name='scaled_feature'):
    #     # Visualize the training and validation loss
    #     fig = plt.figure()
    #     plt.plot(history.history['loss'], label='training loss')
    #     plt.plot(history.history['val_loss'], label='validation loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.savefig(f'Output/{self.region}/{self.hazard}/{name.replace(" ", "_")}.png', dpi=300)
    #     return fig


if __name__ == "__main__":
    model_manager = ModelMgr(hazard='Wildfire')
    model_manager.train_base_model()
