"""
Training pipeline for GFAN model
Includes loss functions, optimization, and validation strategies
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json

from src.spectral_decomposition import SpectralAugmentation
from src.gfan_model import GFAN
from src.evaluation import GFANEvaluator, post_process_predictions, find_optimal_threshold


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG seizure detection
    """
    
    def __init__(self, windows, labels, spectral_features, subjects=None, 
                 augmentation: Optional[SpectralAugmentation] = None, training: bool = False):
        """
        Initialize EEG dataset
        
        Args:
            windows: EEG windows [n_samples, channels, time]
            labels: Binary labels [n_samples]
            spectral_features: Multi-scale spectral features
            subjects: Subject IDs for each sample
            augmentation: Data augmentation function
            training: Flag to indicate if dataset is for training (enables augmentation)
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.spectral_features = spectral_features
        self.subjects = subjects
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        
        # Get spectral features for this sample (magnitudes from STFT)
        # Assuming spectral_features is a list of tensors, one for each scale
        features = [scale_features[idx] for scale_features in self.spectral_features]
        
        # Apply augmentation if specified, in training mode, and for the seizure class
        if self.augmentation is not None and self.training and label == 1:
            # Augment each scale's spectral features (magnitude)
            augmented_features = []
            for feature_magnitude in features:
                # The augment_stft returns a dictionary, we need the magnitude
                # Assuming augmentation works on numpy arrays
                augmented_magnitude = self.augmentation.augment_stft(
                    {'magnitude': feature_magnitude.numpy()}
                )['magnitude']
                augmented_features.append(torch.from_numpy(augmented_magnitude))
            features = augmented_features
        
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
                 weight_decay=1e-4, class_weights=None, sparsity_weight=0.01,
                 kl_weight=1e-5):
        """
        Initialize trainer
        
        Args:
            model: GFAN model
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Weights for handling class imbalance
            sparsity_weight: Weight for sparsity regularization
            kl_weight: Weight for KL divergence loss (for variational inference)
        """
        self.model = model.to(device)
        self.device = device
        self.sparsity_weight = sparsity_weight
        self.kl_weight = kl_weight
        
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
            outputs = self.model(spectral_features, return_uncertainty=False)
            
            # Compute loss
            focal_loss = self.criterion(outputs['logits'], labels)
            sparsity_loss = self.sparsity_weight * outputs['sparsity_loss']
            
            total_loss_batch = focal_loss + sparsity_loss
            
            # Add KL loss if applicable
            if 'kl_loss' in outputs:
                kl_loss = self.kl_weight * outputs['kl_loss']
                total_loss_batch += kl_loss
            else:
                kl_loss = torch.tensor(0.0)

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
                'sparsity': sparsity_loss.item(),
                'kl': kl_loss.item()
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
                outputs = self.model(spectral_features, return_uncertainty=True)
                
                # Compute loss
                focal_loss = self.criterion(outputs['logits'], labels)
                sparsity_loss = self.sparsity_weight * outputs['sparsity_loss']
                total_loss_batch = focal_loss + sparsity_loss

                if 'kl_loss' in outputs:
                    total_loss_batch += self.kl_weight * outputs['kl_loss']
                
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
        best_val_sensitivity = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Sensitivity: {val_metrics['sensitivity']:.4f}")
            
            # Early stopping and model saving based on sensitivity
            if val_metrics['sensitivity'] > best_val_sensitivity:
                best_val_sensitivity = val_metrics['sensitivity']
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved with validation sensitivity: {best_val_sensitivity:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
            
            self.scheduler.step()
        
        print("Training finished.")
        return os.path.join(save_dir, 'best_model.pth')
    
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
        subjects = np.array([s['subject'] for s in dataset])
        windows = np.array([s['window'].numpy() for s in dataset])
        labels = np.array([s['label'].numpy() for s in dataset])
        
        logo = LeaveOneGroupOut()
        
        if n_folds is None:
            n_folds = logo.get_n_splits(groups=subjects)
            
        for fold, (train_val_idx, test_idx) in enumerate(logo.split(windows, labels, groups=subjects)):
            if fold >= n_folds:
                break
            
            print(f"\n----- Fold {fold+1}/{n_folds} -----")
            
            # Split train_val into train and validation
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.2, stratify=labels[train_val_idx], 
                random_state=42, groups=subjects[train_val_idx]
            )

            train_dataset = self.create_subset(dataset, train_idx)
            val_dataset = self.create_subset(dataset, val_idx)
            test_dataset = self.create_subset(dataset, test_idx)

            # Create WeightedRandomSampler for training set
            train_labels = [s['label'] for s in train_dataset]
            class_counts = np.bincount(train_labels)
            class_weights = 1. / class_counts
            sample_weights = np.array([class_weights[l] for l in train_labels])
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True
            )

            train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize model and trainer
            model = GFAN(
                eigenvalues=graph_info['eigenvalues'],
                eigenvectors=graph_info['eigenvectors'],
                **self.model_config
            )
            trainer = GFANTrainer(model, **self.trainer_config)
            
            # Train model
            save_dir = f"checkpoints/fold_{fold+1}"
            best_model_path = trainer.train(train_loader, val_loader, epochs=50, save_dir=save_dir)
            
            # Load best model
            model.load_state_dict(torch.load(best_model_path))
            
            # Evaluate on test set
            evaluator = GFANEvaluator(model, device=self.trainer_config.get('device', 'cuda'))
            
            # Find optimal threshold on validation set
            val_results = evaluator.evaluate_model(val_loader, return_predictions=True)
            optimal_threshold = find_optimal_threshold(
                val_results['predictions']['labels'],
                val_results['predictions']['probabilities']
            )
            print(f"Optimal threshold found on validation set: {optimal_threshold:.4f}")

            # Evaluate on test set using optimal threshold
            test_results = evaluator.evaluate_model(test_loader, return_predictions=True)
            
            # Post-process with optimal threshold
            post_processed_results = post_process_predictions(
                test_results['predictions'], threshold=optimal_threshold
            )

            fold_results = {
                'fold': fold + 1,
                'subject': np.unique(subjects[test_idx])[0],
                'metrics': test_results['metrics'],
                'post_processed_metrics': post_processed_results['event_metrics'],
                'optimal_threshold': optimal_threshold
            }
            
            self.results.append(fold_results)
            print(f"Fold {fold+1} Test F1: {test_results['metrics']['f1']:.4f} | "
                  f"Post-processed F1: {post_processed_results['event_metrics']['f1']:.4f}")

        return self.results
    
    def create_subset(self, dataset, indices):
        return [dataset[i] for i in indices]
    
    def get_summary_metrics(self):
        if not self.results:
            return None
        
        summary = {}
        for key in self.results[0]['metrics'].keys():
            summary[key] = np.mean([r['metrics'][key] for r in self.results])
            summary[f"{key}_std"] = np.std([r['metrics'][key] for r in self.results])
        
        # Add post-processed summary
        for key in self.results[0]['post_processed_metrics'].keys():
            summary[f"post_{key}"] = np.mean([r['post_processed_metrics'][key] for r in self.results])
            summary[f"post_{key}_std"] = np.std([r['post_processed_metrics'][key] for r in self.results])

        return summary
    
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
