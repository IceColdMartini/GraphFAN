"""
Training pipeline for GFAN model
Includes loss functions, optimization, and validation strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG seizure detection
    """
    
    def __init__(self, windows, labels, spectral_features, subjects=None, 
                 augmentation=None):
        """
        Initialize EEG dataset
        
        Args:
            windows: EEG windows [n_samples, channels, time]
            labels: Binary labels [n_samples]
            spectral_features: Multi-scale spectral features
            subjects: Subject IDs for each sample
            augmentation: Data augmentation function
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.spectral_features = spectral_features
        self.subjects = subjects
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        
        # Get spectral features for this sample
        features = []
        for scale_features in self.spectral_features:
            if isinstance(scale_features, list):
                features.append(scale_features[idx])
            else:
                features.append(scale_features[idx])
        
        # Apply augmentation if specified
        if self.augmentation is not None and self.training:
            window, features = self.augmentation(window, features)
        
        sample = {
            'window': window,
            'spectral_features': features,
            'label': label,
            'subject': self.subjects[idx] if self.subjects is not None else 0
        }
        
        return sample


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class GFANTrainer:
    """
    Training pipeline for GFAN model
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, 
                 weight_decay=1e-4, class_weights=None, sparsity_weight=0.01):
        """
        Initialize trainer
        
        Args:
            model: GFAN model
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Weights for handling class imbalance
            sparsity_weight: Weight for sparsity regularization
        """
        self.model = model.to(device)
        self.device = device
        self.sparsity_weight = sparsity_weight
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss function
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        self.criterion = WeightedFocalLoss(weight=class_weights)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move data to device
            spectral_features = [f.to(self.device) for f in batch['spectral_features']]
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectral_features)
            
            # Compute loss
            focal_loss = self.criterion(outputs['logits'], labels)
            sparsity_loss = self.sparsity_weight * outputs['sparsity_loss']
            total_loss_batch = focal_loss + sparsity_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += total_loss_batch.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss_batch.item(),
                'focal': focal_loss.item(),
                'sparsity': sparsity_loss.item()
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self.compute_metrics(all_labels, all_predictions)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                spectral_features = [f.to(self.device) for f in batch['spectral_features']]
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(spectral_features)
                
                # Compute loss
                focal_loss = self.criterion(outputs['logits'], labels)
                sparsity_loss = self.sparsity_weight * outputs['sparsity_loss']
                total_loss_batch = focal_loss + sparsity_loss
                
                total_loss += total_loss_batch.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs['logits'], dim=1)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Seizure probability
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.compute_metrics(all_labels, all_predictions, all_probabilities)
        
        return avg_loss, metrics
    
    def compute_metrics(self, labels, predictions, probabilities=None):
        """
        Compute evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'sensitivity': recall_score(labels, predictions, pos_label=1, zero_division=0),
            'specificity': recall_score(labels, predictions, pos_label=0, zero_division=0)
        }
        
        if probabilities is not None:
            metrics['auc'] = roc_auc_score(labels, probabilities)
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs=100, save_dir='checkpoints'):
        """
        Complete training loop
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_f1 = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, Val Specificity: {val_metrics['specificity']:.4f}")
            if 'auc' in val_metrics:
                print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'val_metrics': val_metrics
                }, os.path.join(save_dir, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # F1 score
        train_f1 = [m['f1'] for m in self.train_metrics]
        val_f1 = [m['f1'] for m in self.val_metrics]
        axes[0, 1].plot(train_f1, label='Train F1')
        axes[0, 1].plot(val_f1, label='Val F1')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        
        # Sensitivity and Specificity
        val_sens = [m['sensitivity'] for m in self.val_metrics]
        val_spec = [m['specificity'] for m in self.val_metrics]
        axes[1, 0].plot(val_sens, label='Sensitivity')
        axes[1, 0].plot(val_spec, label='Specificity')
        axes[1, 0].set_title('Sensitivity and Specificity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        
        # AUC (if available)
        if 'auc' in self.val_metrics[0]:
            val_auc = [m['auc'] for m in self.val_metrics]
            axes[1, 1].plot(val_auc, label='AUC')
            axes[1, 1].set_title('AUC Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class LeaveOneSubjectOutValidator:
    """
    Leave-one-subject-out cross-validation for GFAN
    """
    
    def __init__(self, model_config, trainer_config):
        """
        Initialize validator
        
        Args:
            model_config: Configuration for GFAN model
            trainer_config: Configuration for trainer
        """
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.results = []
    
    def validate(self, dataset, graph_info, n_folds=None):
        """
        Perform leave-one-subject-out validation
        
        Args:
            dataset: Complete dataset with subject information
            graph_info: Graph structure information
            n_folds: Number of folds (None for all subjects)
        """
        subjects = np.unique(dataset.subjects)
        if n_folds is not None:
            subjects = subjects[:n_folds]
        
        logo = LeaveOneGroupOut()
        
        for fold, (train_idx, val_idx) in enumerate(logo.split(
            dataset.windows, dataset.labels, dataset.subjects)):
            
            if fold >= len(subjects):
                break
                
            print(f"\n=== Fold {fold + 1}/{len(subjects)} ===")
            print(f"Test subject: {subjects[fold]}")
            
            # Create train and validation datasets
            train_dataset = self.create_subset(dataset, train_idx)
            val_dataset = self.create_subset(dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True, num_workers=4
            )
            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False, num_workers=4
            )
            
            # Initialize model and trainer
            from .gfan_model import GFAN
            model = GFAN(**self.model_config, **graph_info)
            trainer = GFANTrainer(model, **self.trainer_config)
            
            # Train
            trainer.train(train_loader, val_loader, epochs=50)
            
            # Evaluate
            _, val_metrics = trainer.validate_epoch(val_loader)
            val_metrics['fold'] = fold
            val_metrics['test_subject'] = subjects[fold]
            
            self.results.append(val_metrics)
            
            print(f"Fold {fold + 1} Results:")
            for metric, value in val_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
    
    def create_subset(self, dataset, indices):
        """
        Create subset of dataset
        """
        subset_windows = dataset.windows[indices]
        subset_labels = dataset.labels[indices]
        subset_features = []
        
        for scale_features in dataset.spectral_features:
            if isinstance(scale_features, torch.Tensor):
                subset_features.append(scale_features[indices])
            else:
                subset_features.append([scale_features[i] for i in indices])
        
        subset_subjects = dataset.subjects[indices] if dataset.subjects is not None else None
        
        return EEGDataset(
            subset_windows, subset_labels, subset_features, subset_subjects
        )
    
    def get_summary_metrics(self):
        """
        Get summary statistics across all folds
        """
        if not self.results:
            return None
        
        metrics_summary = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity', 'auc']
        
        for metric in metric_names:
            if metric in self.results[0]:
                values = [r[metric] for r in self.results]
                metrics_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return metrics_summary
    
    def save_results(self, save_path):
        """
        Save validation results
        """
        results_dict = {
            'fold_results': self.results,
            'summary_metrics': self.get_summary_metrics()
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
