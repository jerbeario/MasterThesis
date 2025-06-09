"""HazardMapper - Analysis Module
=========================
This module provides functionality for exploratory data analysis (EDA) and results evaluation of hazard models.
It includes methods to compute hazard statistics, visualize distributions, and evaluate model performance.

This module also generate the results for the thesis manuscript, including LaTeX tables and figures for the hazards in Europe.
""" 

import os
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import argparse
import jenkspy
import time
import json
import seaborn as sns


from HazardMapper.utils import plot_npy_arrays

plt.style.use('bauhaus_light')

class EDA:
    def __init__(self):
        # Load hazard inventory and region map
        self.hazards = ['landslide', 'flood', 'wildfire']
        self.hazard_inventories = [np.load(f"Input/Europe/npy_arrays/masked_{hazard}_Europe.npy").flatten() for hazard in self.hazards]
        self.region_map = np.load("Input/Europe/partition_map/sub_countries_rasterized.npy").flatten()
        self.valid_mask = ~np.isnan(np.load("Input/Europe/npy_arrays/masked_elevation_Europe.npy")).flatten()


        # Apply valid mask to region map and hazard inventory
        self.region_map = self.region_map[self.valid_mask]
        self.hazard_inventories = [hazard_inventory[self.valid_mask] for hazard_inventory in self.hazard_inventories]

        # Create output directory if it doesn't exist
        self.output_dir = f'Output/Europe/eda'
        os.makedirs(self.output_dir, exist_ok=True)

        self.hazard_stats = None
        self.region_stats = None

        # self.load_stats()

    def store_stats(self):
        # Store region stats in a JSON file
        region_stats_path = f'{self.output_dir}/region_stats.json'
        with open(region_stats_path, 'w') as f:
            json.dump(self.region_stats, f, indent=4)
        print(f'Region stats stored at {region_stats_path}')
    
    def load_stats(self):
        # Load region stats from a JSON file
        region_stats_path = f'{self.output_dir}/region_stats.json'
        if os.path.exists(region_stats_path):
            with open(region_stats_path, 'r') as f:
                self.region_stats = json.load(f)
            print(f'Region stats loaded from {region_stats_path}')
        else:
            print(f'No region stats found at {region_stats_path}, please run get_region_stats() first.')


    def get_hazard_stats(self):
        # Initialize hazard stats dictionary
        self.hazard_stats = {
            'hazard': [],
            'n_hazard_pixels': [],
            'n_pixels': [],
            'hazard_coverage': [],
            'avg_hazard_count': [],
            'min_hazard_count': [],
            'max_hazard_count': [],
            'std_hazard_count': [],
            'hazard_counts': [],
        }

        # Calculate total hazard counts, total hazard pixels, and hazard percentage
        for hazard_inventory, hazard in zip(self.hazard_inventories, self.hazards):
            print(f'Calculating hazard stats for {hazard} hazard...')

            n_hazard_pixels = np.sum(hazard_inventory > 0)
            n_total_pixels = np.sum(self.region_map > 0)
            hazard_percentage = round((n_hazard_pixels / n_total_pixels) * 100, 2) if n_total_pixels > 0 else 0
            self.hazard_stats['hazard'].append(hazard)
            self.hazard_stats['n_hazard_pixels'].append(n_hazard_pixels)
            self.hazard_stats['n_pixels'].append(n_total_pixels)
            self.hazard_stats['hazard_coverage'].append(hazard_percentage)
        print(f'Hazard stats for {hazard} hazard:')
        print(f'  - Total hazard pixels: {self.hazard_stats["n_hazard_pixels"]}')
        print(f'  - Total pixels: {self.hazard_stats["n_pixels"]}')
        print(f'  - Hazard coverage: {self.hazard_stats["hazard_coverage"]}%')

        # Get region-wise hazard statistics
        self.get_region_stats()



    def get_region_stats(self):
        # Initialize region stats dictionary
        self.region_stats = {
            'region_ID': [],
            'n_hazard_pixels': [],
            'n_pixel': [],
            'hazard_coverage': [],
        }

        # Calculate region-wise hazard statistics for eah hazard
        for hazard_inventory, hazard in zip(self.hazard_inventories, self.hazards):
            print(f'Calculating region stats for {hazard} hazard...')

            unique_regions = np.unique(self.region_map)
            for region in unique_regions:
                if region == 0:
                    continue
                region_mask = self.region_map == region
                n_hazard_pixels = np.sum(hazard_inventory[region_mask] > 0)
                n_total_pixels = np.sum(region_mask)
                hazard_coverage = (n_hazard_pixels / n_total_pixels) * 100 if n_total_pixels > 0 else 0
                self.region_stats['region_ID'].append(region)
                self.region_stats['n_hazard_pixels'].append(n_hazard_pixels)
                self.region_stats['n_pixel'].append(n_total_pixels)
                self.region_stats['hazard_coverage'].append(hazard_coverage)

            # Calculate average, min, max, and std of hazard counts for the hazard
            avg_hazard_count = np.mean(self.region_stats['n_hazard_pixels'])
            min_hazard_count = np.min(self.region_stats['n_hazard_pixels'])
            max_hazard_count = np.max(self.region_stats['n_hazard_pixels'])
            std_hazard_count = np.std(self.region_stats['n_hazard_pixels'])
        

            self.hazard_stats['avg_hazard_count'].append(avg_hazard_count)
            self.hazard_stats['min_hazard_count'].append(min_hazard_count)
            self.hazard_stats['max_hazard_count'].append(max_hazard_count)
            self.hazard_stats['std_hazard_count'].append(std_hazard_count)

            self.hazard_stats['hazard_counts'].append(self.region_stats['hazard_coverage'])

            print(f'Region stats for {hazard} hazard:')
            print(f'  - Average hazard counts: {avg_hazard_count}')
            print(f'  - Min hazard counts: {min_hazard_count}')
            print(f'  - Max hazard counts: {max_hazard_count}')
            print(f'  - Std hazard counts: {std_hazard_count}')


       
    def make_latex_table(self):
        # Create a LaTeX table for the hazard with total coverage, total hazard pixels, averege region coverage, min and max region coverage, std 
        print(f'Creating LaTeX table for hazards...')
        table = "\\begin{table}[h!]\n"
        table += "    \\centering\n"
        table += "    \\begin{tabularx}{\\textwidth}{@{}lXXXXXX@{}}\n"
        table += "    \\toprule\n"
        table += "    \\textbf{Hazard} & \\textbf{Total} & \\textbf{Min} & \\textbf{Avg} & \\textbf{Max} & \\textbf{Std} \\\\\n"
        table += "    \\midrule\n"
        for hazard, n_hazard_pixels, min_hazard_count, max_hazard_count, avg_hazard_count, std_hazard_count in zip(self.hazard_stats['hazard'],
                                                                        self.hazard_stats['n_hazard_pixels'],
                                                                        self.hazard_stats['min_hazard_count'],
                                                                        self.hazard_stats['avg_hazard_count'],
                                                                        self.hazard_stats['max_hazard_count'],
                                                                        self.hazard_stats['std_hazard_count']):
            # Format the statistics to two decimal places
            avg_hazard_count = f"{avg_hazard_count:.2f}"
            std_hazard_count = f"{std_hazard_count:.2f}"                                                           
            table += f"    {hazard} & {n_hazard_pixels} & {min_hazard_count} & {avg_hazard_count} & {max_hazard_count} & {std_hazard_count}\\% \\\\\n"
        table += "    \\bottomrule\n"
        table += "    \\end{tabularx}\n"
        table += f"    \\caption{"Total number of susceptible pixels per hazard in the area of of interest. The counts are also inspected per NUTS-2 region, with the minimum, average, maximum and standard deviation also included for each hazard. These statistics along with the figure XX visualize the diversity and bias in the hazard inventories.   "}\n"
        table += "    \\label{tab:hazard_stats}\n"
        table += "\\end{table}\n"
        with open(f'{self.output_dir}/hazard_stats.tex', 'w') as f:
            f.write(table)
        print(f'LaTeX table for hazards created at {self.output_dir}/hazard_stats.tex')



    def plot_hazard_distribution(self):
       # Plot the distribution of region-wise hazard coverage for each hazard
        print(f'Plotting regional hazard count distribution...')
        fig, axs = plt.subplots(1, len(self.hazards), figsize=(10, 6))
        for i, hazard in enumerate(self.hazards):
            hazard_counts = self.hazard_stats['hazard_counts'][i]
            sns.histplot(hazard_counts, bins=20, kde=True, ax=axs[i])
            axs[i].set_title(f'{hazard.capitalize()}')
            axs[i].set_xlabel('Hazard Counts')
            axs[i].set_ylabel('Frequency')
        plt.suptitle('Regional Hazard Count Distribution')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hazard_distribution.png')
        plt.close()

    
        


class Results:
    def __init__(self, hazard):
        self.hazard = hazard
        y_true_file = f'Output/Europe/{hazard}/evaluation/true_labels.npy'
        self.y_true = np.load(y_true_file)
        self.y_pred = []
        self.y_prob = []
        self.models = []
        self.scores = []
        map_path = f'Output/Europe/{hazard}/hazard_map/{self.hazard}_hazard_map.npy'
        self.hazard_map = np.load(map_path)



    def load_predictions(self):
        print(f'Loading predictions for {self.hazard} hazard...')
        dir = f'Output/Europe/{self.hazard}/evaluation'

        # List all npy files in the directory
        files = [f for f in os.listdir(dir) if f.endswith('.npy')]
        print(files)
        
        # File name structure: hazard_model_prediction.npy
        for file in files:
            parts = file.split('_')
            if parts[-1] == 'predictions.npy':
                model_name = parts[1]
                y_prob = np.load(os.path.join(dir, file))

                self.models.append(model_name)
                print(f'Loaded predictions from {file} for model {model_name}, with length {len(y_prob)}')
                self.y_prob.append(y_prob)


    def get_scores(self):

        self.load_predictions()


        print(f'Calculating scores for {self.hazard} hazard...')
        for y_prob, model_name in zip(self.y_prob, self.models):
            # Optimize threshold based on best F1
            precision_curve, recall_curve, thresholds = precision_recall_curve(self.y_true, y_prob)
            f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


            # Get predictions with best threshold
            self.y_pred.append((y_prob >= best_threshold).astype(int))

            # Compute metrics
            metrics = {
                'Model': model_name,
                'F1': f1_score(self.y_true, self.y_pred, zero_division=0),
                'AUROC': roc_auc_score(self.y_true, y_prob),
                'AP': average_precision_score(self.y_true, y_prob),
                'MAE': mean_absolute_error(self.y_true, y_prob),
                'Best_Threshold': best_threshold
            }
            self.scores.append(metrics)

    def make_pr_curve(self):

        print(f'Creating Precision-Recall curve for {self.hazard} hazard...')
        plt.figure(figsize=(10, 6))
        for y_prob, model_name in zip(self.y_prob, self.models):
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_true, y_prob)
            plt.plot(recall_curve, precision_curve, label=model_name)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title(f'Precision-Recall Curve for {self.hazard} Hazard')
        plt.legend()
        plt.grid()
        plt.savefig(f'Output/Europe/{self.hazard.capitalize()}/evaluation/precision_recall_curve.png')
        plt.close()
    
    def make_latex_table(self):
        print(f'Creating LaTeX table for {self.hazard} hazard...')
        table = "\\begin{table}[h!]\n"
        table += "    \\centering\n"
        table += "    \\begin{tabularx}{0.45\\textwidth}{@{}lXXXXXX@{}}\n"
        table += "    \\toprule\n"
        table += "    \\textbf{Model} & \\textbf{F1} & \\textbf{AUROC} & \\textbf{AP} & \\textbf{MAE} & \\textbf{Threshold} \\\\\n"
        table += "    \\midrule\n"
        
        score.sort(key=lambda x: x['F1'], reverse=True)  # Sort by F1 score
        for score in self.scores:
            table += f"    {score['Model']} & {score['F1']:.3f} & {score['AUROC']:.3f} & {score['AP']:.3f} & {score['MAE']:.3f} & {score['Best_Threshold']:.3f} \\\\\n"

        table += "    \\bottomrule\n"
        table += "    \\end{tabularx}\n"
        table += f"    \\caption{{Evaluation metrics for models on {self.hazard} hazard.}}\n"
        table += "    \\label{tab:evaluation_metrics}\n"
        table += "\\end{table}\n"
        
        with open(f'Output/Europe/{self.hazard}/evaluation/metrics_table.tex', 'w') as f:
            f.write(table)

    def plot_output_distribution(self):
        """
        Plots the distribution of the test set comparing the hazard map and the true labels.

        """

        print(f'Plotting output distribution for {self.hazard} hazard...')
        bins = np.linspace(0, 1, 11)  # Adjust bins as needed
        plt.figure(figsize=(10, 6))
        sns.histplot(self.y_true, bins=bins, kde=True, label='True Labels', color='blue', alpha=0.5)
        sns.histplot(self.y_prob, bins=bins, kde=True, label='Predictions', color='orange', alpha=0.5)
        plt.xlabel('Susceptibility Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {self.hazard} Model Output vs True Labels')
        plt.legend()
        plt.grid()
        plt.savefig(f'Output/Europe/{self.hazard}/evaluation/output_distribution.png')
        plt.close()
        
    def bin_map(self):
        """
        Applies Jenks Natural Breaks classification to a .npy array of continuous values.
        
        Parameters:
        - npy_path (str): Path to the .npy file
        - n_classes (int): Number of bins/classes (default 5)
        
        Returns:
        - binned_array (np.ndarray): Array with the same shape as input, but values replaced by bin indices (0 to n_classes - 1)
        - breaks (List[float]): List of break thresholds used for binning
        """

        


        size = 100_000 # Example size, replace with actual size
        arr = self.hazard_map.copy()
        flat = arr[~np.isnan(arr)].flatten()
        np.random.shuffle(flat)
        flat = flat[:size]  # Limit to size for performance

        n_classes = 5
        print(f'Calculating Jenks breaks for {self.hazard} hazard with {n_classes} classes...')

        breaks = jenkspy.jenks_breaks(flat, n_classes=n_classes) 
        print(f'Jenks breaks for {self.hazard} hazard: {breaks}')

        # Save breaks to a JSON file
        breaks_path = f'Output/Europe/{self.hazard}/hazard_map/jenks_breaks.json'
        with open(breaks_path, 'w') as f:
            json.dump({'hazard': self.hazard, 'n_classes': n_classes, 'breaks': breaks}, f, indent=4)


        # Digitize array using the breaks (exclude the first value which is min)
        binned_array = np.digitize(arr, bins=breaks[1:-1], right=False).astype(float)
        binned_array[np.isnan(arr)] = np.nan  # Preserve NaN values

        # Save and plot
        np.save(f'Output/Europe/{self.hazard}/hazard_map/{self.hazard}_binned_hazard_map.npy', binned_array)
        plot_npy_arrays(binned_array, 
                        name='Susceptibility', 
                        type='bins',
                        title=f'{self.hazard} Hazard Map Binned', 
                        save_path=f'Output/Europe/{self.hazard}/hazard_map/{self.hazard}_binned_hazard_map.png')


        return binned_array, breaks
        

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='Evaluate hazard models')
    argparse.add_argument('-z', '--hazard', type=str, required=False, help='Hazard type (wildfire, landslide, flood)')
    argparse.add_argument('-e', '--eda', action='store_true', help='Run exploratory data analysis')
    argparse.add_argument('-r', '--results', action='store_true', help='Run results analysis')
    args = argparse.parse_args()

    hazard = args.hazard

    if args.eda:
        eda = EDA()
        eda.get_hazard_stats()
        # eda.make_latex_table()
        eda.plot_hazard_distribution()

    if args.results:
        results = Results(hazard)
        results.get_scores()
        results.make_pr_curve()
        results.make_latex_table()
        results.bin_map()



                
        


        
