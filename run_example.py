"""
Complete example script demonstrating GFAN usage
This script shows how to use all components together
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from src.data_preprocessing import CHBMITDataProcessor, create_balanced_dataset
from src.spectral_decomposition import MultiScaleSTFT, SpectralAugmentation
from src.graph_construction import create_graph_from_windows
from src.gfan_model import GFAN
from src.training import GFANTrainer, EEGDataset, LeaveOneSubjectOutValidator
from src.evaluation import GFANEvaluator, AblationStudy


def run_complete_pipeline_example():
    """
    Demonstrate the complete GFAN pipeline with synthetic data
    """
    print("=" * 60)
    print("GFAN COMPLETE PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic EEG data for demonstration
    print("\n1. Generating synthetic EEG data...")
    n_samples = 500
    n_channels = 18
    n_timepoints = 512  # 2 seconds at 256 Hz
    
    # Create synthetic EEG with some structure
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate base EEG signals
    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints)
    
    # Add some seizure-like patterns to 10% of samples
    seizure_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
    labels = np.zeros(n_samples)
    labels[seizure_indices] = 1
    
    # Add high-frequency oscillations to seizure samples
    for idx in seizure_indices:
        # Add 20-40 Hz oscillations
        t = np.linspace(0, 2, n_timepoints)
        for ch in range(n_channels):
            freq = np.random.uniform(20, 40)
            amplitude = np.random.uniform(2, 5)
            eeg_data[idx, ch, :] += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Create subject IDs
    subjects = np.random.choice([0, 1, 2], size=n_samples)
    
    print(f"Generated {n_samples} samples with {np.sum(labels)} seizure windows")
    
    # 2. Data Preprocessing
    print("\n2. Data preprocessing...")
    processor = CHBMITDataProcessor(target_fs=256, window_size=2.0, overlap=0.5)
    
    # Normalize data
    normalized_data = np.zeros_like(eeg_data)
    for i in range(n_samples):
        for j in range(n_channels):
            channel_data = eeg_data[i, j, :]
            normalized_data[i, j, :] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
    
    # 3. Multi-scale spectral decomposition
    print("\n3. Computing multi-scale spectral features...")
    stft_processor = MultiScaleSTFT(fs=256, window_sizes=[1.0, 2.0], hop_ratio=0.25)
    
    all_spectral_features = [[] for _ in range(len(stft_processor.window_sizes))]
    
    for i in range(min(100, n_samples)):  # Process subset for demo
        features = stft_processor.compute_multiscale_stft(normalized_data[i])
        for j, stft_result in enumerate(features):
            # Flatten and take mean over time for simplicity
            magnitude = stft_result['magnitude']
            flattened = magnitude.reshape(-1, magnitude.shape[2])
            mean_features = np.mean(flattened, axis=1)
            all_spectral_features[j].append(mean_features)
    
    # Convert to tensors
    for i in range(len(all_spectral_features)):
        all_spectral_features[i] = torch.tensor(
            np.array(all_spectral_features[i]), dtype=torch.float32
        )
        print(f"Scale {i} features shape: {all_spectral_features[i].shape}")
    
    # 4. Graph construction
    print("\n4. Constructing graph structure...")
    channels = [f'CH{i+1}' for i in range(n_channels)]
    
    # Use subset of data for graph construction
    subset_data = normalized_data[:min(50, len(normalized_data))]
    graph_info = create_graph_from_windows(subset_data, channels, method='spatial')
    
    print(f"Graph constructed with {len(channels)} nodes")
    print(f"Eigenvalue range: {graph_info['eigenvalues'].min():.3f} to {graph_info['eigenvalues'].max():.3f}")
    
    # 5. Model configuration
    print("\n5. Configuring GFAN model...")
    feature_dims = [feat.shape[1] for feat in all_spectral_features]
    
    model_config = {
        'n_channels': n_channels,
        'spectral_features_dims': feature_dims,
        'eigenvalues': graph_info['eigenvalues'],
        'eigenvectors': graph_info['eigenvectors'],
        'hidden_dims': [64, 32],
        'n_classes': 2,
        'sparsity_reg': 0.01,
        'dropout_rate': 0.1,
        'uncertainty_estimation': True,
        'fusion_method': 'attention'
    }
    
    # Create model
    model = GFAN(**model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 6. Dataset preparation
    print("\n6. Preparing datasets...")
    n_used_samples = len(all_spectral_features[0])
    used_labels = labels[:n_used_samples]
    used_subjects = subjects[:n_used_samples]
    
    # Balance dataset
    balanced_indices = np.random.choice(n_used_samples, size=min(200, n_used_samples), replace=False)
    balanced_features = [feat[balanced_indices] for feat in all_spectral_features]
    balanced_labels = used_labels[balanced_indices]
    balanced_subjects = used_subjects[balanced_indices]
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(balanced_labels)), test_size=0.3, random_state=42, 
        stratify=balanced_labels
    )
    
    train_features = [feat[train_idx] for feat in balanced_features]
    test_features = [feat[test_idx] for feat in balanced_features]
    train_labels = balanced_labels[train_idx]
    test_labels = balanced_labels[test_idx]
    
    # Create datasets
    train_dataset = EEGDataset(train_features, train_labels)
    test_dataset = EEGDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 7. Training
    print("\n7. Training GFAN model...")
    trainer = GFANTrainer(model, device=device, learning_rate=1e-3)
    
    # Quick training for demo (normally would use more epochs)
    trainer.train(train_loader, test_loader, epochs=5)
    
    # 8. Evaluation
    print("\n8. Evaluating model...")
    evaluator = GFANEvaluator(model, device)
    results = evaluator.evaluate_model(test_loader)
    
    print("Test Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # 9. Interpretability
    print("\n9. Analyzing interpretability...")
    
    # Plot eigenmode attribution
    if len(results['spectral_attributions']) > 0:
        evaluator.plot_eigenmode_attribution(
            results['spectral_attributions'], 
            graph_info['eigenvalues']
        )
    
    # 10. Ablation study (simplified)
    print("\n10. Running simplified ablation study...")
    
    ablation_configs = {
        'full_model': model_config,
        'no_uncertainty': {**model_config, 'uncertainty_estimation': False},
        'simple_fusion': {**model_config, 'fusion_method': 'concat'}
    }
    
    ablation_results = {}
    for config_name, config in ablation_configs.items():
        print(f"Testing {config_name}...")
        
        # Create and train model
        test_model = GFAN(**config)
        test_trainer = GFANTrainer(test_model, device=device, learning_rate=1e-3)
        test_trainer.train(train_loader, test_loader, epochs=3)
        
        # Evaluate
        test_evaluator = GFANEvaluator(test_model, device)
        test_results = test_evaluator.evaluate_model(test_loader)
        
        ablation_results[config_name] = test_results['metrics']
    
    # Print ablation results
    print("\nAblation Study Results:")
    for config_name, metrics in ablation_results.items():
        print(f"\n{config_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': model,
        'results': results,
        'ablation_results': ablation_results,
        'graph_info': graph_info
    }


def demonstrate_kaggle_workflow():
    """
    Demonstrate workflow specifically for Kaggle environment
    """
    print("\n" + "=" * 60)
    print("KAGGLE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    print("""
    For Kaggle deployment, follow these steps:
    
    1. **Dataset Setup**:
       - Add CHB-MIT dataset to Kaggle inputs
       - The dataset should be at: /kaggle/input/chb-mit-scalp-eeg-database-1.0.0/
    
    2. **Environment Setup**:
       - Enable GPU acceleration
       - Install additional packages: mne, pyedflib, torch-geometric
    
    3. **Code Organization**:
       - Copy the main notebook: kaggle_gfan_notebook.py
       - Include all source modules in the notebook cells
    
    4. **Execution**:
       - Run preprocessing on CHB-MIT data
       - Train GFAN model with cross-validation
       - Generate publication-quality results
    
    5. **Output**:
       - Model checkpoints saved to /kaggle/working/
       - Results and visualizations exported
       - Performance metrics for publication
    
    Key Kaggle-specific considerations:
    - Memory management for large EEG datasets
    - GPU optimization for graph neural networks
    - Checkpoint saving for long training runs
    - Visualization generation for model interpretability
    """)
    
    # Show how to adapt for different dataset sizes
    print("\nDataset Size Recommendations:")
    print("- Small demo (< 1GB): 3-5 subjects, 2-3 files each")
    print("- Medium scale (1-10GB): 10-15 subjects, full files")
    print("- Full dataset (> 10GB): All 24 subjects, requires optimization")
    
    print("\nExpected Runtime on Kaggle:")
    print("- Preprocessing: 30-60 minutes")
    print("- Training (5-fold CV): 2-4 hours")
    print("- Evaluation: 15-30 minutes")
    print("- Total: 3-5 hours with GPU")


if __name__ == "__main__":
    # Run the complete demonstration
    try:
        results = run_complete_pipeline_example()
        demonstrate_kaggle_workflow()
        
        print("\n✅ All demonstrations completed successfully!")
        print("📊 Ready for Kaggle deployment!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("🔧 Check dependencies and try again")
