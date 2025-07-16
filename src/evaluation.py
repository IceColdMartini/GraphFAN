"""
Evaluation and ablation studies for GFAN model
Comprehensive analysis including interpretability and uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os


class GFANEvaluator:
    """
    Comprehensive evaluation suite for GFAN model
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            model: Trained GFAN model
            device: Evaluation device
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_model(self, test_loader, return_predictions=True):
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return detailed predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []
        all_spectral_attributions = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                spectral_features = [f.to(self.device) for f in batch['spectral_features']]
                labels = batch['label'].to(self.device)
                
                # Forward pass with uncertainty estimation
                outputs = self.model(spectral_features, return_uncertainty=True, n_mc_samples=20)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs['logits'], dim=1)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                # Get eigenmode attributions
                attributions = self.model.get_eigenmode_attribution(spectral_features)
                
                # Collect results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                
                if 'uncertainty' in outputs:
                    uncertainties = torch.mean(outputs['uncertainty'], dim=1)  # Average over classes
                    all_uncertainties.extend(uncertainties.cpu().numpy())
                
                all_spectral_attributions.append(attributions.cpu().numpy())
        
        # Compute metrics
        metrics = self.compute_detailed_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        results = {
            'metrics': metrics,
            'spectral_attributions': np.concatenate(all_spectral_attributions)
        }
        
        if return_predictions:
            results['predictions'] = {
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'uncertainties': all_uncertainties
            }
        
        return results
    
    def compute_detailed_metrics(self, labels, predictions, probabilities):
        """
        Compute comprehensive evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, cohen_kappa_score,
            balanced_accuracy_score, matthews_corrcoef
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'balanced_accuracy': balanced_accuracy_score(labels, predictions),
            'matthews_corrcoef': matthews_corrcoef(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(labels, predictions, average='weighted', zero_division=0),
            'sensitivity': recall_score(labels, predictions, pos_label=1, zero_division=0),
            'specificity': recall_score(labels, predictions, pos_label=0, zero_division=0),
            'auc_roc': roc_auc_score(labels, probabilities),
            'auc_pr': average_precision_score(labels, probabilities),
            'cohen_kappa': cohen_kappa_score(labels, predictions)
        }
        
        # Compute false positive rate per hour (assuming 2-second windows)
        n_non_seizure = np.sum(np.array(labels) == 0)
        false_positives = np.sum((np.array(labels) == 0) & (np.array(predictions) == 1))
        total_hours = (n_non_seizure * 2) / 3600  # Convert to hours
        metrics['false_positives_per_hour'] = false_positives / total_hours if total_hours > 0 else 0
        
        return metrics
    
    def plot_confusion_matrix(self, labels, predictions, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Seizure', 'Seizure'],
                   yticklabels=['Non-Seizure', 'Seizure'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, labels, probabilities, save_path=None):
        """
        Plot ROC curve
        """
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_eigenmode_attribution(self, spectral_attributions, eigenvalues, 
                                 save_path=None):
        """
        Plot eigenmode attribution for interpretability
        """
        # Average attributions across samples
        mean_attributions = np.mean(spectral_attributions, axis=0)
        
        plt.figure(figsize=(12, 6))
        
        # Plot attribution vs eigenvalue
        plt.subplot(1, 2, 1)
        plt.plot(eigenvalues.cpu().numpy(), mean_attributions, 'bo-')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Attribution Weight')
        plt.title('Eigenmode Attribution vs Eigenvalue')
        plt.grid(True)
        
        # Plot attribution vs frequency (approximate)
        plt.subplot(1, 2, 2)
        # Convert eigenvalues to approximate frequencies
        # This is a simplified mapping - actual frequency depends on graph structure
        approx_freqs = np.sqrt(eigenvalues.cpu().numpy()) * 10  # Scaling factor
        plt.plot(approx_freqs, mean_attributions, 'ro-')
        plt.xlabel('Approximate Frequency (Hz)')
        plt.ylabel('Attribution Weight')
        plt.title('Eigenmode Attribution vs Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_uncertainty_calibration(self, predictions_dict, n_bins=10, save_path=None):
        """
        Analyze and plot uncertainty calibration.
        
        Args:
            predictions_dict (dict): Dictionary with predictions, labels, and uncertainties.
            n_bins (int): Number of bins for calibration analysis.
            save_path (str, optional): Path to save the plot.
        """
        if 'uncertainties' not in predictions_dict or not predictions_dict['uncertainties']:
            print("No uncertainty information available for calibration analysis.")
            return None
            
        labels = np.array(predictions_dict['labels'])
        probabilities = np.array(predictions_dict['probabilities'])
        uncertainties = np.array(predictions_dict['uncertainties'])
        
        # Ensure uncertainties are positive and finite for binning
        uncertainties = np.nan_to_num(uncertainties, nan=0.0, posinf=np.max(uncertainties[np.isfinite(uncertainties)]))

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(probabilities, labels, n_bins)
        print(f"Expected Calibration Error (ECE): {ece:.4f}")

        # Bin data by uncertainty
        bin_limits = np.linspace(np.min(uncertainties), np.max(uncertainties), n_bins + 1)
        bin_low = bin_limits[:-1]
        bin_high = bin_limits[1:]
        
        bin_accuracy = np.zeros(n_bins)
        bin_confidence = np.zeros(n_bins)
        bin_uncertainty = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (uncertainties >= bin_low[i]) & (uncertainties < bin_high[i])
            bin_counts[i] = np.sum(in_bin)
            
            if bin_counts[i] > 0:
                bin_accuracy[i] = np.mean((predictions_dict['predictions'][in_bin] == labels[in_bin]))
                bin_confidence[i] = np.mean(np.max(torch.softmax(torch.tensor(predictions_dict['probabilities'][in_bin]), dim=-1).numpy(), axis=1))
                bin_uncertainty[i] = np.mean(uncertainties[in_bin])

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration curve (Reliability diagram)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.plot(bin_confidence, bin_accuracy, 'o-', label='Model')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
        ax1.legend()
        ax1.grid(True)

        # Accuracy vs. Uncertainty
        ax2.plot(bin_uncertainty, bin_accuracy, 'o-', color='r')
        ax2.set_xlabel('Average Uncertainty')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs. Uncertainty')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'ece': ece,
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_confidence,
            'bin_uncertainty': bin_uncertainty,
            'bin_counts': bin_counts
        }

    def _calculate_ece(self, probabilities, labels, n_bins=10):
        """Calculate Expected Calibration Error."""
        confidences = np.max(probabilities, axis=1) if probabilities.ndim > 1 else probabilities
        predictions = np.argmax(probabilities, axis=1) if probabilities.ndim > 1 else (probabilities > 0.5).astype(int)
        accuracies = (predictions == labels)

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
        return ece


def find_optimal_threshold(labels, probabilities, metric='f1'):
    """
    Find the optimal decision threshold for a binary classifier.
    
    Args:
        labels (np.array): True binary labels.
        probabilities (np.array): Probabilities for the positive class.
        metric (str): The metric to optimize ('f1', 'sensitivity', 'balanced_accuracy').
        
    Returns:
        float: The optimal threshold.
    """
    from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score

    thresholds = np.linspace(0, 1, 100)
    best_score = -1
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, predictions, pos_label=1)
        elif metric == 'sensitivity':
            score = recall_score(labels, predictions, pos_label=1)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(labels, predictions)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold


def post_process_predictions(predictions_dict, threshold=0.5, min_duration_sec=5, window_size_sec=2, overlap_ratio=0.5):
    """
    Apply post-processing to raw model predictions.
    
    Args:
        predictions_dict (dict): Dictionary containing 'probabilities' and 'labels'.
        threshold (float): Decision threshold for converting probabilities to binary predictions.
        min_duration_sec (int): Minimum duration in seconds to be considered a seizure event.
        window_size_sec (float): The size of each prediction window in seconds.
        overlap_ratio (float): The overlap ratio used during windowing.
        
    Returns:
        dict: Post-processed predictions and event-based metrics.
    """
    probabilities = np.array(predictions_dict['probabilities'])
    binary_preds = (probabilities > threshold).astype(int)
    
    hop_size_sec = window_size_sec * (1 - overlap_ratio)
    min_duration_windows = int(min_duration_sec / hop_size_sec)
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(
            np.array(predictions_dict['labels']),
            probabilities,
            metric='f1'
        )
        print(f"Using optimal threshold found from validation set: {threshold:.4f}")

    # Merge consecutive positive predictions into events
    processed_preds = np.zeros_like(binary_preds)
    
    start_indices = np.where(np.diff(binary_preds) == 1)[0] + 1
    if binary_preds[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
        
    end_indices = np.where(np.diff(binary_preds) == -1)[0]
    if binary_preds[-1] == 1:
        end_indices = np.append(end_indices, len(binary_preds) - 1)

    for start, end in zip(start_indices, end_indices):
        if (end - start) >= min_duration_windows:
            processed_preds[start:end+1] = 1
            
    # Recalculate metrics based on post-processed predictions
    labels = np.array(predictions_dict['labels'])
    post_processed_metrics = GFANEvaluator.compute_detailed_metrics(None, labels, processed_preds, probabilities)
    
    return {
        'post_processed_predictions': processed_preds,
        'event_metrics': post_processed_metrics
    }


class AblationStudy:
    """
    Systematic ablation study for GFAN components
    """
    
    def __init__(self, base_config, graph_info, trainer_config, device='cuda'):
        """
        Initialize ablation study
        
        Args:
            base_config: Base model configuration
            graph_info: Graph structure information
            trainer_config: Base trainer configuration
            device: Training device
        """
        self.base_config = base_config.copy()
        self.graph_info = graph_info
        self.trainer_config = trainer_config.copy()
        self.device = device
        self.results = {}
    
    def run_ablation_study(self, train_loader, val_loader, test_loader, 
                          save_dir='ablation_results', epochs=30):
        """
        Run comprehensive ablation study
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Define ablation configurations
        ablation_configs = {
            'full_model': self.base_config,
            'no_adaptive_basis': {**self.base_config, 'sparsity_reg': 0, 'fixed_basis': True},
            'single_scale': {**self.base_config, 'spectral_features_dims': [self.base_config['spectral_features_dims'][0]]},
            'no_sparsity': {**self.base_config, 'sparsity_reg': 0},
            'no_uncertainty': {**self.base_config, 'uncertainty_method': None},
            'simple_fusion': {**self.base_config, 'fusion_method': 'concat'},
            'no_dropout': {**self.base_config, 'dropout_rate': 0}
        }
        
        for config_name, config in ablation_configs.items():
            print(f"\\n=== Running Ablation: {config_name} ===")
            
            # Create and train model
            result = self.train_and_evaluate_config(
                config_name, config, train_loader, val_loader, test_loader, epochs
            )
            
            self.results[config_name] = result
            
            # Save intermediate results
            self.save_results(os.path.join(save_dir, 'ablation_results.json'))
        
        # Generate summary
        self.generate_ablation_summary(save_dir)
    
    def train_and_evaluate_config(self, config_name, config, train_loader, val_loader, test_loader, epochs):
        """
        Train and evaluate a specific configuration
        """
        from src.gfan_model import GFAN
        from src.training import GFANTrainer
        
        # Create model
        model_config = config.copy()
        # Handle single-scale case
        if config_name == 'single_scale':
            # If single scale, fusion method is not applicable in the same way
            model_config['fusion_method'] = 'concat' # or some other default
        
        model = GFAN(**model_config, 
                     eigenvalues=self.graph_info['eigenvalues'], 
                     eigenvectors=self.graph_info['eigenvectors'])
        
        # Create trainer
        trainer = GFANTrainer(model, device=self.device, **self.trainer_config)
        
        # Train
        trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate
        evaluator = GFANEvaluator(model, self.device)
        test_results = evaluator.evaluate_model(test_loader, return_predictions=True)
        
        return {
            'config': config,
            'metrics': test_results['metrics'],
            'training_history': {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'val_metrics': trainer.val_metrics
            },
            'predictions': test_results['predictions']
        }
    
    def generate_ablation_summary(self, save_dir):
        """
        Generate summary of ablation results
        """
        # Create comparison table
        comparison_metrics = ['accuracy', 'f1_score', 'sensitivity', 'specificity', 'auc_roc', 'false_positives_per_hour']
        
        summary_data = []
        for config_name, result in self.results.items():
            row = {'configuration': config_name}
            for metric in comparison_metrics:
                row[metric] = result['metrics'].get(metric, np.nan)
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save summary table
        summary_path = os.path.join(save_dir, 'ablation_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"Ablation summary saved to {summary_path}")
        
        # Plot comparison
        self.plot_ablation_comparison(df, save_dir)
        
        return df
    
    def plot_ablation_comparison(self, df, save_dir):
        """
        Plot ablation study comparison
        """
        metrics_to_plot = [col for col in df.columns if col != 'configuration']
        
        n_metrics = len(metrics_to_plot)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Sort by metric value for better visualization
            sorted_df = df.sort_values(metric, ascending=True)
            
            bars = ax.barh(sorted_df['configuration'], sorted_df[metric])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            
            # Highlight the full model for easy comparison
            full_model_idx = list(sorted_df['configuration']).index('full_model')
            bars[full_model_idx].set_color('salmon')
            
            # Add value labels to bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width * 1.01, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', va='center', ha='left')

        # Hide any unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'ablation_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Ablation comparison plot saved to {plot_path}")
        plt.show()
    
    def save_results(self, save_path):
        """
        Save ablation results to a JSON file.
        """
        serializable_results = {}
        for config_name, result in self.results.items():
            # Create a copy that can be modified for serialization
            serializable_result = result.copy()
            
            # Convert config tensors to lists or strings
            serializable_config = {}
            for k, v in result['config'].items():
                if isinstance(v, torch.Tensor):
                    serializable_config[k] = v.shape
                else:
                    serializable_config[k] = v
            serializable_result['config'] = serializable_config
            
            # Convert numpy arrays in predictions to lists
            if 'predictions' in serializable_result:
                for k, v in serializable_result['predictions'].items():
                    if isinstance(v, np.ndarray):
                        serializable_result['predictions'][k] = v.tolist()

            serializable_results[config_name] = serializable_result

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4, default=lambda o: '<not serializable>')
        print(f"Ablation results saved to {save_path}")


def generate_publication_plots(evaluator_results, save_dir='publication_plots'):
    """
    Generate publication-quality plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    predictions = evaluator_results['predictions']
    
    # Combined ROC and Precision-Recall curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(predictions['labels'], predictions['probabilities'])
    auc_roc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='#2E86AB', linewidth=2, 
            label=f'GFAN (AUC = {auc_roc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#A23B72', linewidth=2, linestyle='--', alpha=0.7)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(predictions['labels'], predictions['probabilities'])
    auc_pr = auc(recall, precision)
    
    ax2.plot(recall, precision, color='#F18F01', linewidth=2,
            label=f'GFAN (AUC = {auc_pr:.3f})')
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_pr_curves.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(predictions['labels'], predictions['predictions'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
