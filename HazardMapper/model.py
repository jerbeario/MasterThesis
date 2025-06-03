import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

import logging
import logging.handlers

from joblib import dump, load
import sys
import time
import argparse


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, mean_absolute_error, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_curve
    )

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from HazardMapper.dataset import HazardDataset, BalancedBatchSampler
from HazardMapper.architecture import MLP, CNN, UNet, SimpleCNN, FullCNN, SpatialAttentionCNN
from HazardMapper.utils import plot_npy_arrays

import numpy as np
from torch.utils.data import Sampler

plt.style.use('bauhaus_light')


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

        # Load features for each variable
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
        # sample train_val data to the sample size
        if self.sample_size < 1.0:
            n_samples = int(len(X_train_val) * self.sample_size)
            indices = np.random.choice(len(X_train_val), n_samples, replace=False)
            X_train_val = X_train_val[indices]
            y_train_val = y_train_val[indices]


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
                 logger, seed, use_wandb, sample_size, class_ratio, sampler, num_workers, early_stopping, patience, min_delta, output_dir):
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
        self.sampler = sampler
        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()
        
        # Hyperparameters
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001
        self.filters = 64
        self.n_layers = 1
        self.drop_value = 0.4
        self.kernel_size = 3
        self.pool_size = 2  
        self.activation = torch.nn.ReLU()
        self.dropout = True
        self.n_nodes = 128 # for MLP architecture

        # Training
        self.current_epoch = 0
        self.epochs = 5
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
        self.output_dir = output_dir

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
                patch_size=self.patch_size,
                n_layers=self.n_layers,
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

        elif self.model_architecture == 'SpatialAttentionCNN':
            model = SpatialAttentionCNN(
                device=self.device,
                logger=self.logger,
                num_vars=self.num_vars,
                filters=self.filters,
                # kernel_size=self.kernel_size,
                # pool_size=self.pool_size,
                n_layers=self.n_layers,
                dropout=self.dropout,
                drop_value=self.drop_value,
                # name_model=self.name_model,
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
        # np.random.shuffle(test_indices)

        train_indices = train_indices[:int(len(train_indices) * self.sample_size)]
        val_indices = val_indices[:int(len(val_indices) * self.sample_size)]
        # test_indices = test_indices[:int(len(test_indices)* self.sample_size)]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)


        if self.sampler == 'custom':
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
            

        elif self.sampler == 'default':
            # Create DataLoader with the custom batch sampler
            train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_size= self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)


        # # Calculate weights for the samples based on class ratio
        # train_weights = train_labels * self.class_ratio + (1 - train_labels)
        # val_weights = val_labels * self.class_ratio + (1 - val_labels)


        # # Use RandomSampler for training and validation datasets
        # train_sampler = LargeWeightedRandomSampler(train_weights, num_samples=len(train_labels), replacement=False)
        # val_sampler = LargeWeightedRandomSampler(val_weights, num_samples=len(val_labels), replacement=False)


        # # Create DataLoader with the custom batch sampler
        # train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_size= self.batch_size, sampler=train_sampler)
        # val_loader = DataLoader(val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, sampler=val_sampler)
        # test_loader = DataLoader(test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)

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
                sample_weigts = labels * self.class_ratio + (1 - labels) 
                loss = F.binary_cross_entropy(outputs, labels, weight=sample_weigts)
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
        model_path = f"{self.output_dir}/models/{self.best_val_loss:4f}_{self.name_model}.pth"
        onnx_path = f"{self.output_dir}/models/{self.best_val_loss:4f}_{self.name_model}.onnx"
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
            self.design_basemodel()  # Recreate the model architecture
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Best model state loaded successfully.")
        else:
            # If no best model state is available, check if a saved model exists
            model_dir = f"{self.output_dir}/models"
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

            if model_files:
                # find lowest validation loss model
                best_model_file = min(model_files, key=lambda f: float(f.split('_')[0]))
                model_path = f'{model_dir}/{best_model_file}'
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Best model loaded from {model_path}")
            else:
                # If no saved model exists, log an error
                self.logger.error("No saved model found. Please train the model first or provide a valid path.")
         
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
        self.logger.info('Loading best model for testing...')
        self.load_best_model()
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
                    'min': 0.00005,
                    'max': 0.005
                },
                'filters': {
                    'values': [8, 16, 32, 64]
                },
                'n_layers': {
                    'values': [1, 2, 3]
                },
                'drop_value': {
                    'distribution': 'uniform',
                    'min': 0.2,
                    'max': 0.5
                },
            },
        }
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"{self.hazard}_{self.model_architecture}_sweep"
        )
         
        # Start the sweep agent
        wandb.agent(sweep_id, self.sweep, count=100)  # Run 10 trials
        self.logger.info(f"Hyperparameter sweep completed with ID: {sweep_id}")
 
    def sweep(self):
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

            
            # Log configuration for this run
            self.logger.info(f"Sweep run with: lr={self.learning_rate}, "
                        f"weight_decay={self.weight_decay}, "
                        f"filters={self.filters}, "
                        f"n_layers={self.n_layers}, "
                        f"dropout={self.drop_value}, "

            )
            
            # Recreate data loaders with new batch size
            self.train_loader, self.val_loader, self.test_loader = self.load_dataset()
            
            # Rebuild model with new hyperparameters
            self.model = self.design_basemodel()
            self.model.to(self.device)
            
            # Use the existing training loop
            self.train()
            
            # Evaluate on test set
            y_true, y_prob  = self.testing()

            # Optimize threshold based on best F1
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


            # Get predictions with best threshold
            y_pred = (y_prob >= best_threshold).astype(int)


            test_accuracy = accuracy_score(y_true, y_pred)
            test_precision = precision_score(y_true, y_pred, zero_division=0)
            test_recall = recall_score(y_true, y_pred.astype(int), zero_division=0)
            test_f1 = f1_score(y_true, y_pred.astype(int), zero_division=0)
            test_auc_roc = roc_auc_score(y_true, y_prob)
            test_auprc = average_precision_score(y_true, y_prob)
            test_mae = self.safe_mae(torch.tensor(y_true, dtype=torch.float32), torch.tensor(y_prob, dtype=torch.float32))


            metrics = {
                "test_mae": test_mae.item(),
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_auc_roc": test_auc_roc,
                "test_ap": test_auprc,
                "best_threshold": best_threshold,
            }
            wandb.log(metrics)

            hyper_params = {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "filters": self.filters,
                "n_layers": self.n_layers,
                "drop_value": self.drop_value,
            }

            # Save results to CSV
            metrics.update(hyper_params)
            results_csv = os.path.join(self.output_dir, f'all_model_metrics.csv')
            results_df = pd.DataFrame([metrics])
            if os.path.exists(results_csv):
                pd.concat([pd.read_csv(results_csv), results_df], ignore_index=True).to_csv(results_csv, index=False)
            else:
                results_df.to_csv(results_csv, index=False)
            self.logger.info(f"Metrics saved to {results_csv}")
          
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
        hazard_map_npy_path = os.path.join(output_dir, f"{self.hazard}_hazard_map.npy")
        np.save(hazard_map_npy_path, hazard_map)
        self.logger.info(f"Hazard map saved to {hazard_map_npy_path}")

        # Optionally, visualize the hazard map 
        hazard_map_path = f'{output_dir}/{self.hazard}_hazard_map.png'
        plot_npy_arrays(
            hazard_map,
            title=f"{self.hazard} Hazard Map",
            name=f'{self.hazard} susceptibility',
            downsample_factor=10,
            save_path=hazard_map_path)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"hazard_map": wandb.Image(hazard_map_path)})

class ModelMgr:
    def __init__(self, region='Europe', batch_size=1024, patch_size=5, architecture='CNN' , sample_size = 1, 
             class_ratio=0.5, sampler='custom', hazard='wildfire', hyper=False, use_wandb=True, experiement_name='HazardMapper'):
        
        self.early_stopping = True
        self.patience = 10
        self.min_delta = 0.001
        self.use_wandb = use_wandb
        self.hazard = hazard
        self.region = region
        self.batch_size = batch_size
        self.name_model = 'susceptibility'
        self.missing_data_value = 0
        self.sample_size = sample_size
        self.class_ratio = class_ratio
        self.num_workers = 8
        self.patch_size = patch_size
        self.hyper = hyper
        self.model_architecture = architecture # 'CNN' or 'UNet' or 'SimpleCNN' or 'SpatialAttentionCNN' or 'MLP'
        self.experiement_name = experiement_name
        self.sampler = sampler

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
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Torch version: {torch.__version__}")

        # Create output directory if it doesn't exist
        self.output_dir = f'Output/{self.region}/{self.hazard}/{self.model_architecture}/{self.experiement_name}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory created at: {self.output_dir}")

        self.hazard_model_instance = None
        self.baseline_instance = None

        # Initialize model instances based on architecture
        if self.model_architecture in ["FullCNN", "UNet", "MLP", "CNN", "SimpleCNN", "SpatialAttentionCNN"]:
            self.hazard_model_instance = HazardModel(
                device=self.device,
                hazard=self.hazard,
                region=self.region,
                variables=self.variables,
                patch_size=self.patch_size,
                batch_size=self.batch_size,
                sample_size=self.sample_size,
                class_ratio=self.class_ratio,
                sampler=self.sampler,
                model_architecture=self.model_architecture,
                num_workers=self.num_workers,
                logger=self.logger,
                seed=self.seed,
                use_wandb=self.use_wandb,
                early_stopping=self.early_stopping,
                patience=self.patience,
                min_delta=self.min_delta,
                output_dir = self.output_dir
            
                
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
        reference = f'{self.model_architecture}_{self.hazard}_{self.region}'

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

        # Save predictions to npy
        npy_path = os.path.join(output_path, f'{model_name}_predictions.npy')
        np.save(npy_path, y_prob)
        self.logger.info(f"Predictions saved to {npy_path}")
        # Save true labels to npy
        true_path = os.path.join(output_path, f'true_labels.npy')
        if not os.path.exists(true_path):
            np.save(true_path, y_true)

       

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
            'Experiment': self.experiement_name,
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

    
        # Plot output distribution
        bin_edges = np.arange(0, 1.005, 0.05)

        xy = np.concatenate((y_true.reshape(-1, 1), y_prob.reshape(-1, 1)), axis=1)

        pos = xy[xy[:, 0] == 1]
        neg = xy[xy[:, 0] == 0]

        pd_path = os.path.join(output_path, f'{model_name}_probability_distribution.png')
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        bin_edges = np.arange(0, 1.1, 0.1)

        sns.histplot(neg[:, 1], bins=bin_edges, label="Predicted", color="blue", stat="count", element="step", alpha=0.5, ax=ax[0])
        sns.histplot(neg[:, 0], bins=bin_edges, label="True", color="orange", stat="count", element="step", alpha=0.5, ax=ax[0])
        sns.histplot(pos[:, 1], bins=bin_edges, label="Predicted", color="blue", stat="count", element="step", alpha=0.5, ax=ax[1])
        sns.histplot(pos[:, 0], bins=bin_edges, label="True", color="orange", stat="count", element="step", alpha=0.5, ax=ax[1])

        ax[1].set_title("Positive Class")
        ax[0].set_title("Negative Class")

        ax[0].set_ylabel("Count")
        ax[1].set_ylabel("")

        ax[1].legend()
        ax[1].yaxis.tick_right()
        plt.suptitle(f'Predicted Probability vs True Class Distribution\n{model_name}')
        plt.tight_layout()
        plt.savefig(pd_path)
        plt.close()



        # Plot predicted probability distribution vs true class distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Predicted Probabilities (True Negatives)')
        plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Predicted Probabilities (True Positives)')
        plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title(f'Predicted Probability Distribution vs Actual Class Distribution\n{model_name}')
        plt.legend()
        plt.grid(True)
       
        


        # Save results to CSV
        results_csv = os.path.join(output_path, f'all_model_metrics.csv')
        results_df = pd.DataFrame([metrics])
        if os.path.exists(results_csv):
            pd.concat([pd.read_csv(results_csv), results_df], ignore_index=True).to_csv(results_csv, index=False)
        else:
            results_df.to_csv(results_csv, index=False)
        self.logger.info(f"Metrics saved to {results_csv}")

        # # Save model 
        # if self.hazard_model_instance is not None:
        #     self.hazard_model_instance.save_best_model()
       

    def make_hazard_map(self):
        """
        Create a hazard map using the trained model.
        """

        self.hazard_model_instance.make_hazard_map()

if __name__ == "__main__":
  # Example configuration

    # use argparser 
    parser = argparse.ArgumentParser(description='Hazard Mapper Configuration')
    parser.add_argument('-n', '--name', type=str, default='HazardMapper', help='Name of the experiment')
    parser.add_argument('-r', '--region', type=str, default='Europe', help='Region for hazard mapping')
    parser.add_argument('-z', '--hazard', type=str, default='landslide', help='Hazard type (wildfire or landslide soon flood)')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('-p', '--patch_size', type=int, default=5, help='Patch size for model input')
    parser.add_argument('-a', '--architecture', type=str, default='SimpleCNN', help='Model architecture (FullCNN, UNet, MLP, LR, RF)')
    parser.add_argument('-s', '--sample_size', type=float, default=0.5, help='Sample size for training')
    parser.add_argument('-c', '--class_ratio', type=float, default=9, help='Ratio of negative to positive samples in the batch')
    parser.add_argument('-y', '--hyper', action='store_true', default=False,  help='Enable hyperparameter optimization')
    parser.add_argument('-w', '--use_wandb', action='store_true',default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--sampler', type=str, default='default', help='Sampler type (custom or random)')
    args = parser.parse_args()

    region = args.region
    hazard = args.hazard
    batch_size = args.batch_size
    patch_size = args.patch_size
    architecture = args.architecture
    sample_size = args.sample_size
    class_ratio = args.class_ratio
    hyper = args.hyper
    use_wandb = args.use_wandb
    experiement_name = args.name
    sampler = args.sampler


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
        experiement_name=experiement_name,
        sampler=sampler

    )
    # Train the base model
    # model_mgr.train_hazard_model()

    if architecture in ['RF', 'LR']:
        model_mgr.train_baseline_model()
    else:   
        model_mgr.train_hazard_model()
    # model_mgr.make_hazard_map()