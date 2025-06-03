import os
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import argparse


plt.style.use('bauhaus_light')


class Results:
    def __init__(self, hazard):
        self.hazard = hazard
        y_true_file = f'Output/Europe/{hazard}/evaluation/true_labels.npy'
        self.y_true = np.load(y_true_file)
        self.y_pred = []
        self.models = []
        self.scores = []

        self.load_predictions()
        self.get_scores()

    def load_predictions(self):
        print(f'Loading predictions for {self.hazard} hazard...')
        dir = f'Output/Europe/{self.hazard}/evaluation'

        # List all npy files in the directory
        files = [f for f in os.listdir(dir) if f.endswith('.npy')]
        
        # File name structure: hazard_model_prediction.npy
        for file in files:
            parts = file.split('_')
            if parts[0] == self.hazard:
                model_name = parts[1]
                y_prob = np.load(os.path.join(dir, file))

                self.models.append(model_name)
                print(f'Loaded predictions from {file} for model {model_name}, with length {len(y_prob)}')
                self.y_pred.append(y_prob)


    def get_scores(self):
        print(f'Calculating scores for {self.hazard} hazard...')
        for y_prob, model_name in zip(self.y_pred, self.models):
            # Optimize threshold based on best F1
            precision_curve, recall_curve, thresholds = precision_recall_curve(self.y_true, y_prob)
            f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


            # Get predictions with best threshold
            y_pred = (y_prob >= best_threshold).astype(int)

            # Compute metrics
            metrics = {
                'Model': model_name,
                'F1': f1_score(self.y_true, y_pred, zero_division=0),
                'AUROC': roc_auc_score(self.y_true, y_prob),
                'AP': average_precision_score(self.y_true, y_prob),
                'MAE': mean_absolute_error(self.y_true, y_prob),
                'Best_Threshold': best_threshold
            }
            self.scores.append(metrics)

    def make_pr_curve(self):
        print(f'Creating Precision-Recall curve for {self.hazard} hazard...')
        plt.figure(figsize=(10, 6))
        for y_prob, model_name in zip(self.y_pred, self.models):
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_true, y_prob)
            plt.plot(recall_curve, precision_curve, label=model_name)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {self.hazard} Hazard')
        plt.legend()
        plt.grid()
        plt.savefig(f'Output/Europe/{self.hazard}/evaluation/precision_recall_curve.png')
        plt.close()
    
    def make_latex_table(self):
        print(f'Creating LaTeX table for {self.hazard} hazard...')
        table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lcccccc}\n"
        table += "\\hline\nModel & F1 & AUROC & AP & MAE & Best Threshold \\\\\n\\hline\n"
        
        for score in self.scores:
            table += f"{score['Model']} & {score['F1']:.3f} & {score['AUROC']:.3f} & {score['AP']:.3f} & {score['MAE']:.3f} & {score['Best_Threshold']:.3f} \\\\\n"

        table += "\\hline\n\\end{tabular}\n\\caption{Evaluation metrics for models on " + self.hazard + " hazard.}\n\\label{tab:evaluation_metrics}\n\\end{table}"
        
        with open(f'Output/Europe/{self.hazard}/evaluation/metrics_table.tex', 'w') as f:
            f.write(table)
        

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='Evaluate hazard models')
    argparse.add_argument('-z', '--hazard', type=str, required=True, help='Hazard type (e.g., flood, earthquake)')
    args = argparse.parse_args()

    hazard = args.hazard

    results = Results(hazard)
    # results.make_pr_curve()
    results.make_latex_table()


                
        


        
