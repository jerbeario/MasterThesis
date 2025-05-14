from joblib import dump
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

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
            "test": "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_landcover_Europe_flat.npy",
            "test2": "/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe/npy_arrays/masked_fire_weather_Europe.npy",
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
        if self.patch_size == 1:
            patch = self.features[:, row, col]
        else:
            patch = self.features[:, row:row + self.patch_size, col:col + self.patch_size]
        # Get the label for the center cell
        label = self.labels[row, col]

        # Convert to PyTorch tensors
        patch = torch.tensor(patch, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
         
        # patch = patch.view(self.num_vars, self.patch_size, self.patch_size)
     
        return patch, label
    

def get_data(hazard, variables):
    dataset = HazardDataset(hazard=hazard, variables=variables, patch_size=1)
    
    # Create DataLoader
    partition_map = np.load("/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/partition_map/final_partition_map.npy")
    partition_shape = partition_map.shape

    idx_transform = np.array([[partition_shape[1]],[1]])
    train_indices = (np.argwhere(partition_map == 1) @ idx_transform).flatten()
    val_indices = (np.argwhere(partition_map == 2) @ idx_transform).flatten()
    test_indices = (np.argwhere(partition_map == 3) @ idx_transform).flatten()

    #shuffle the indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)


    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


    train_X = np.array([train_dataset[n][0] for n in range(len(train_dataset))])
    train_y = np.array([train_dataset[n][1] for n in range(len(train_dataset))])

    test_X = np.array([test_dataset[n][0] for n in range(len(test_dataset))])
    test_y = np.array([test_dataset[n][1] for n in range(len(test_dataset))])
    
    #fill na with value -1
    train_X = np.nan_to_num(train_X, nan=-1)
    test_X = np.nan_to_num(test_X, nan=-1)
    return train_X, train_y, test_X, test_y


def train_model(model, X_train, y_train, model_name):
    
    model.fit(X_train, y_train)
    dump(model, f"Baseline/{hazard}/{model_name}_baseline.joblib")
    print(f"{model_name} trained and saved.")

def evaluate_model(model, X_test, y_test, model_name):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    # Calculate accuracy
    train_accuracy = np.mean(y_train_pred == y_train)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Confusion Matrices
    train_cm = confusion_matrix(y_train, y_train_pred, normalize='true')
    test_cm = confusion_matrix(y_test, y_test_pred, normalize='true')

    # Plot Confusion Matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(train_cm, display_labels=["Class 0", "Class 1"]).plot(ax=ax[0], colorbar=False)
    ax[0].set_title("Training \n Confusion Matrix")
    ConfusionMatrixDisplay(test_cm, display_labels=["Class 0", "Class 1"]).plot(ax=ax[1], colorbar=False)
    ax[1].set_title("Testing \n Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"Baseline/{hazard}/{model_name}_baseline_cm.png")
    plt.close()


    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold (closest to the top-left corner)
    distances = np.sqrt((1 - tpr)**2 + fpr**2)  # Euclidean distance to (0, 1)
    best_threshold_index = np.argmin(distances)


    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"Baseline/{hazard}/{model_name}_baseline_roc.png")
    plt.close()

    # Get predicted probabilities for the positive class

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)

    # Calculate the average precision score (AUC for PR curve)
    average_precision = average_precision_score(y_test, y_test_proba)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', label=f"PR Curve (AP = {average_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(f"Baseline/{hazard}/{model_name}_baseline_pr.png")
    plt.close()

    
if __name__ == "__main__":

    hazard = "Wildfire"
    variables = ['temperature_daily', 'NDVI', 'landcover', 'elevation', 'wind_speed_daily', 'fire_weather', 'soil_moisture_root']
    # Example usage
    X_train, y_train, X_test, y_test = get_data(hazard, variables)

    # Logistic Regression
    clf_LR = LogisticRegression(
        penalty='l2',                  # L2 regularization
        C=1.0,                         # Inverse of regularization strength
        solver='lbfgs',               # Optimizer
        max_iter=100,                 # Maximum number of iterations
        random_state=42,
        n_jobs=-1                     # Use all available cores
    )

    # Random Forest Classifier
    clf_RF = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,             # Limit tree depth
    min_samples_split=10,     # Avoid overfitting on tiny splits
    min_samples_leaf=5,       # Each leaf must have at least 5 samples
    random_state=42,
    n_jobs=-1
    )

    # MLP
    clf_MLP = MLPClassifier(
        hidden_layer_sizes=(150, 100, 75, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',              # ReLU activation function
        solver='adam',                  # Adam optimizer
        alpha=0.0001,                   # L2 regularization
        batch_size='auto',              # Automatic batch size
        learning_rate='adaptive',       # Adaptive learning rate
        learning_rate_init=0.001,       # Initial learning rate
        max_iter=100,              # Maximum number of iterations
        early_stopping=True,            # Use early stopping
        validation_fraction=0.1,        # Fraction of training data for validation
        n_iter_no_change=10,            # Early stopping patience
        random_state=42,
        verbose=True
    )


    train_model(clf_MLP, X_train, y_train, 'MLP')

    # Evaluate the model
    evaluate_model(clf_MLP, X_test, y_test, 'MLP')


