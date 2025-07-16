"""
Main execution script for the GFAN seizure detection pipeline.
Orchestrates data preprocessing, feature extraction, model training,
and evaluation using Leave-One-Subject-Out cross-validation.
"""

import os
import glob
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from src.data_preprocessing import CHBMITDataProcessor, load_chb_mit_annotations
from src.spectral_decomposition import MultiScaleSTFT
from src.graph_construction import create_graph_from_windows
from src.training import EEGDataset, LeaveOneSubjectOutValidator
from src.gfan_model import GFAN

def main():
    """
    Main function to run the entire GFAN pipeline.
    """
    # --- 1. Configuration ---
    config = {
        'data': {
            'path': 'chb-mit-scalable-eeg-database-1.0.0',
            'target_fs': 256,
            'window_size': 2.0,
            'overlap': 0.5,
            'n_subjects_to_process': 3 # Set to None to process all subjects
        },
        'features': {
            'window_sizes': [1.0, 2.0, 4.0],
            'hop_ratio': 0.25,
            'log_transform': True
        },
        'graph': {
            'method': 'hybrid',
            'spatial_weight': 0.5,
            'functional_weight': 0.5
        },
        'model': {
            'n_channels': 18, # Will be updated based on data
            'spectral_features_dims': [129, 257, 513], # Will be updated
            'hidden_dims': [128, 64],
            'n_classes': 2,
            'sparsity_reg': 0.01,
            'dropout_rate': 0.2,
            'uncertainty_method': 'mc_dropout',
            'fusion_method': 'attention'
        },
        'trainer': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 10.0], # Will be updated based on data
            'sparsity_weight': 0.01,
            'kl_weight': 1e-6,
            'epochs': 50
        },
        'validation': {
            'n_folds': 3 # Set to None for full LOSO validation
        },
        'results_path': 'results/final_run_summary.json'
    }

    # --- 2. Device Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU.")
    config['trainer']['device'] = device

    # --- 3. Initialization ---
    processor = CHBMITDataProcessor(
        target_fs=config['data']['target_fs'],
        window_size=config['data']['window_size'],
        overlap=config['data']['overlap']
    )
    stft_extractor = MultiScaleSTFT(
        fs=config['data']['target_fs'],
        window_sizes=config['features']['window_sizes'],
        hop_ratio=config['features']['hop_ratio']
    )

    # --- 4. Data Loading and Preprocessing ---
    all_windows, all_labels, all_subjects = [], [], []
    all_spectral_features = [[] for _ in config['features']['window_sizes']]
    
    subject_dirs = sorted([d for d in glob.glob(os.path.join(config['data']['path'], 'chb*')) if os.path.isdir(d)])
    if config['data']['n_subjects_to_process'] is not None:
        subject_dirs = subject_dirs[:config['data']['n_subjects_to_process']]

    print(f"Starting data processing for {len(subject_dirs)} subjects...")
    for subject_dir in tqdm(subject_dirs, desc="Processing Subjects"):
        subject_id = int(os.path.basename(subject_dir).replace('chb', ''))
        summary_file = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}-summary.txt")
        annotations = load_chb_mit_annotations(summary_file)
        
        edf_files = sorted(glob.glob(os.path.join(subject_dir, '*.edf')))
        
        for edf_file in edf_files:
            file_name = os.path.basename(edf_file)
            seizure_info = annotations.get(file_name, [])
            
            try:
                windows, labels, channels = processor.process_file(edf_file, seizure_info)
                if windows is None or len(windows) == 0:
                    continue

                # Extract spectral features for all windows of the file
                file_spectral_features = [[] for _ in config['features']['window_sizes']]
                for i in range(windows.shape[0]):
                    multiscale_stft = stft_extractor.compute_multiscale_stft(windows[i])
                    for scale_idx, stft_result in enumerate(multiscale_stft):
                        file_spectral_features[scale_idx].append(stft_result['magnitude'])

                all_windows.append(windows)
                all_labels.append(labels)
                all_subjects.extend([subject_id] * len(windows))
                for scale_idx in range(len(all_spectral_features)):
                    all_spectral_features[scale_idx].append(np.array(file_spectral_features[scale_idx]))

            except Exception as e:
                print(f"Warning: Could not process file {edf_file}. Error: {e}")

    # Concatenate all data
    final_windows = np.concatenate(all_windows, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    final_subjects = np.array(all_subjects)
    final_spectral_features = [np.concatenate(scale_features, axis=0) for scale_features in all_spectral_features]

    print(f"Total windows processed: {len(final_windows)}")
    print(f"Class distribution: {pd.Series(final_labels).value_counts().to_dict()}")

    # --- 5. Graph Construction ---
    print("Constructing graph...")
    graph_info = create_graph_from_windows(
        final_windows, 
        channels, 
        method=config['graph']['method']
    )
    # Move graph tensors to the correct device
    for key in ['adjacency', 'laplacian', 'eigenvalues', 'eigenvectors']:
        graph_info[key] = graph_info[key].to(device)

    # --- 6. Update Config and Create Dataset ---
    config['model']['n_channels'] = final_windows.shape[1]
    config['model']['spectral_features_dims'] = [f.shape[1] for f in final_spectral_features]
    
    # Update class weights based on data imbalance
    class_counts = pd.Series(final_labels).value_counts()
    weight_for_class_0 = len(final_labels) / (2 * class_counts[0])
    weight_for_class_1 = len(final_labels) / (2 * class_counts[1])
    config['trainer']['class_weights'] = [weight_for_class_0, weight_for_class_1]
    print(f"Calculated class weights: {config['trainer']['class_weights']}")

    # Create the full dataset
    full_dataset = EEGDataset(
        windows=final_windows,
        labels=final_labels,
        spectral_features=final_spectral_features,
        subjects=final_subjects,
        training=False # Augmentation is handled inside the trainer/validator
    )

    # --- 7. Run Leave-One-Subject-Out Validation ---
    print("Starting Leave-One-Subject-Out cross-validation...")
    validator = LeaveOneSubjectOutValidator(
        model_config=config['model'],
        trainer_config=config['trainer']
    )
    
    results = validator.validate(
        dataset=full_dataset,
        graph_info=graph_info,
        n_folds=config['validation']['n_folds']
    )

    # --- 8. Save Final Results ---
    print("Validation finished. Saving results...")
    os.makedirs(os.path.dirname(config['results_path']), exist_ok=True)
    validator.save_results(config['results_path'])
    
    print(f"Results saved to {config['results_path']}")
    print("\n--- Summary Metrics ---")
    summary = validator.get_summary_metrics()
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()
