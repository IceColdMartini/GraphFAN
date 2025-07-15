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
                outputs = self.model(spectral_features, return_uncertainty=True)
                
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
            roc_auc_score, average_precision_score, cohen_kappa_score
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
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
    
    def analyze_uncertainty_calibration(self, predictions_dict, n_bins=10):
        """
        Analyze uncertainty calibration
        """
        if 'uncertainties' not in predictions_dict:
            print("No uncertainty information available")
            return None
        
        labels = np.array(predictions_dict['labels'])
        probabilities = np.array(predictions_dict['probabilities'])
        uncertainties = np.array(predictions_dict['uncertainties'])
        
        # Bin by uncertainty
        uncertainty_bins = np.linspace(0, np.max(uncertainties), n_bins + 1)
        bin_accuracy = []
        bin_confidence = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
            if np.sum(mask) > 0:
                bin_labels = labels[mask]
                bin_probs = probabilities[mask]
                bin_preds = (bin_probs > 0.5).astype(int)
                
                accuracy = np.mean(bin_labels == bin_preds)
                confidence = np.mean(bin_probs)
                
                bin_accuracy.append(accuracy)
                bin_confidence.append(confidence)
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracy.append(0)
                bin_confidence.append(0)
                bin_counts.append(0)
        
        # Plot calibration
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
        plt.bar(bin_centers, bin_accuracy, width=np.diff(uncertainty_bins), alpha=0.7)
        plt.xlabel('Uncertainty Level')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Uncertainty')
        
        plt.subplot(1, 2, 2)
        plt.bar(bin_centers, bin_counts, width=np.diff(uncertainty_bins), alpha=0.7)
        plt.xlabel('Uncertainty Level')
        plt.ylabel('Count')
        plt.title('Sample Distribution by Uncertainty')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_confidence,
            'bin_counts': bin_counts
        }


class AblationStudy:
    """
    Systematic ablation study for GFAN components
    """
    
    def __init__(self, base_config, graph_info, device='cuda'):
        """
        Initialize ablation study
        
        Args:
            base_config: Base model configuration
            graph_info: Graph structure information
            device: Training device
        """
        self.base_config = base_config.copy()
        self.graph_info = graph_info
        self.device = device
        self.results = {}
    
    def run_ablation_study(self, train_loader, val_loader, test_loader, 
                          save_dir='ablation_results'):
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
            'no_uncertainty': {**self.base_config, 'uncertainty_estimation': False},
            'simple_fusion': {**self.base_config, 'fusion_method': 'concat'},
            'no_dropout': {**self.base_config, 'dropout_rate': 0}
        }
        
        for config_name, config in ablation_configs.items():
            print(f"\n=== Running {config_name} ===")
            
            # Create and train model
            result = self.train_and_evaluate_config(
                config, train_loader, val_loader, test_loader
            )
            
            self.results[config_name] = result
            
            # Save intermediate results
            self.save_results(os.path.join(save_dir, f'{config_name}_results.json'))
        
        # Generate summary
        self.generate_ablation_summary(save_dir)
    
    def train_and_evaluate_config(self, config, train_loader, val_loader, test_loader):
        """
        Train and evaluate a specific configuration
        """
        from .gfan_model import GFAN
        from .training import GFANTrainer
        
        # Create model
        model = GFAN(**config, **self.graph_info)
        
        # Create trainer
        trainer = GFANTrainer(model, device=self.device, learning_rate=1e-3)
        
        # Train
        trainer.train(train_loader, val_loader, epochs=30)
        
        # Evaluate
        evaluator = GFANEvaluator(model, self.device)
        test_results = evaluator.evaluate_model(test_loader)
        
        return {
            'config': config,
            'metrics': test_results['metrics'],
            'training_history': {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'val_metrics': trainer.val_metrics
            }
        }
    
    def generate_ablation_summary(self, save_dir):
        """
        Generate summary of ablation results
        """
        # Create comparison table
        comparison_metrics = ['accuracy', 'f1_score', 'sensitivity', 'specificity', 'auc_roc']
        
        summary_data = []
        for config_name, result in self.results.items():
            row = {'configuration': config_name}
            for metric in comparison_metrics:
                row[metric] = result['metrics'][metric]
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save summary table
        df.to_csv(os.path.join(save_dir, 'ablation_summary.csv'), index=False)
        
        # Plot comparison
        self.plot_ablation_comparison(df, save_dir)
        
        return df
    
    def plot_ablation_comparison(self, df, save_dir):
        """
        Plot ablation study comparison
        """
        metrics = ['accuracy', 'f1_score', 'sensitivity', 'specificity', 'auc_roc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Sort by metric value
            sorted_df = df.sort_values(metric, ascending=True)
            
            bars = ax.barh(range(len(sorted_df)), sorted_df[metric])
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels(sorted_df['configuration'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            
            # Highlight full model
            full_model_idx = list(sorted_df['configuration']).index('full_model')
            bars[full_model_idx].set_color('red')
            
            # Add value labels
            for j, (idx, row) in enumerate(sorted_df.iterrows()):
                ax.text(row[metric] + 0.001, j, f'{row[metric]:.3f}', 
                       va='center', fontsize=8)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, save_path):
        """
        Save ablation results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for config_name, result in self.results.items():
            serializable_result = {
                'metrics': result['metrics'],
                'config': {k: v for k, v in result['config'].items() 
                          if not isinstance(v, torch.Tensor)}
            }
            serializable_results[config_name] = serializable_result
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)


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
