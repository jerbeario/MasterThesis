import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os


from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor,  RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, mean_absolute_error, r2_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

plt.style.use('bauhaus_light')


class Baseline:
    def __init__(self, hazard, sample_size, model_architecture, variables, seed):
    
        self.hazard = hazard.lower()
        self.sample_size = sample_size
        self.model_architecture = model_architecture
        self.seed = seed
        self.variables = variables
        self.num_vars = len(variables)

        self.X, self.y = self.load_data()

        print(self.X.shape)
        print(self.y.shape)
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()
        self.model = self.design_model()
    
    

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
    
        # get features and labels for the hazard
        feature_paths = [var_paths[variable] for variable in self.variables]
        label_path = label_paths[self.hazard]

        # Load features (stacked along the first axis for channels)
        features = np.stack([np.load(path) for path in feature_paths], axis=0)


        # Load labels
        labels = np.load(label_path)
        labels = (labels > 0).astype(int)  # Binarize labels


        elevation_map = np.load(var_paths["elevation2"])
        valid_mask = ~np.isnan(elevation_map)
        
        # make features and labels 1d and use valid mask 
        features1d = np.array([features[i][valid_mask].reshape(-1) for i in range(self.num_vars)])
        labels1d = labels[valid_mask].reshape(-1)
        features1d = features1d.T  # Transpose to have shape (samples, features)

        # shuffle the data and select a sample size
        size = labels1d.shape[0]
        n_samples = int(size * sample_size)
        indices = np.arange(features1d.shape[0])
        np.random.shuffle(indices)
        indices = indices[:n_samples]
        features1d = features1d[indices]
        labels1d = labels1d[indices]

        return features1d, labels1d
    
    def split_data(self):


        threshold = 0
        y = (self.y > threshold).astype(int)

        print(f"Label breakdown: {np.bincount(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        print(f"Breakdown before balancing: \nTrain label: {np.bincount(y_train)}, Test label: {np.bincount(y_test)}")


        # Separate the majority and minority classes
        class_0_indices = np.where(y_train == 0)[0]
        class_1_indices = np.where(y_train == 1)[0]

        # Downsample the majority class (class 0) to match the size of the minority class (class 1)
        class_0_downsampled = resample(
            class_0_indices,
            replace=False,  # No replacement, we want to downsample
            n_samples=len(class_1_indices),  # Match the number of class 1 samples
            random_state=self.seed  # For reproducibility
        )

        # Combine the downsampled majority class with the minority class
        balanced_indices = np.concatenate([class_0_downsampled, class_1_indices])

        # Shuffle the indices to mix the classes
        np.random.shuffle(balanced_indices)

        # Create the balanced training set
        X_train = X_train[balanced_indices]
        y_train = y_train[balanced_indices]

        print(f"Breakdown after balancing: \nTrain label: {np.bincount(y_train)}, Test label: {np.bincount(y_test)}")

        return X_train, y_train, X_test, y_test
    
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
        output_path = f"Output/Europe/{self.hazard}/baselines/"
        file_name = f'{self.model_architecture}_{self.hazard}.joblib'

        self.model.fit(self.X_train, self.y_train)

        # Check if the output directory exists, if not create it
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"Output directory {output_path} created.")
        else:
            print(f"Output directory {output_path} already exists.")
        # Save the model    
        dump(self.model, f"{output_path}{file_name}") 
        print(f"Model trained and saved as {output_path}{file_name}")
    
    def load_model(self):
        output_path = f"Output/Europe/{self.hazard}/baselines/"
        file_name = f'{self.model_architecture}_{self.hazard}.joblib'
        # check if the file exists
        try:
            self.model = load(f"{output_path}{file_name}")
            print(f"Model loaded from {output_path}{file_name}")

        except FileNotFoundError:
            print(f"Model file {output_path}{file_name} not found. Please train the model first.")

    def evaluate(self):
        # make ROC curve
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc}")
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(f"Output/Europe/{self.hazard}/baselines/ROC_curve_{self.model_architecture}_{self.hazard}.png")
        plt.close()
        print(f"ROC curve saved as Output/Europe/{self.hazard}/baselines/ROC_curve_{self.model_architecture}_{self.hazard}.png")

        #make PR Curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        print(f"PR AUC: {pr_auc}")
        plt.figure()
        plt.plot(recall, precision, label=f"PR curve (area = {pr_auc:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig(f"Output/Europe/{self.hazard}/baselines/PR_curve_{self.model_architecture}_{self.hazard}.png")
        plt.close()
        print(f"PR curve saved as Output/Europe/{self.hazard}/baselines/PR_curve_{self.model_architecture}_{self.hazard}.png")


if __name__ == "__main__":
    # Example usage
    hazard = "test"
    model_architecture = "RF"
    variables = ["test"]
    seed = 42
    sample_size = 0.001

    baseline = Baseline(hazard, sample_size, model_architecture, variables, seed)
    baseline.train()
    baseline.evaluate()


    


