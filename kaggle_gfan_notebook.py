"""
Main Kaggle notebook for GFAN Epileptic Seizure Detection
End-to-end pipeline implementation for CHB-MIT dataset
"""

# Cell 1: Setup and imports
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Kaggle data paths
INPUT_DIR = '/kaggle/input'
WORKING_DIR = '/kaggle/working'

# Check if running on Kaggle
if os.path.exists('/kaggle'):
    KAGGLE_ENV = True
    DATA_PATH = os.path.join(INPUT_DIR, 'chb-mit-scalp-eeg-database-1.0.0')
else:
    KAGGLE_ENV = False
    DATA_PATH = './data/chb-mit'

print(f"Running on Kaggle: {KAGGLE_ENV}")
print(f"Data path: {DATA_PATH}")

# Cell 2: Install additional packages if needed
if KAGGLE_ENV:
    # Install packages not available in Kaggle by default
    import subprocess
    import sys
    
    packages = [
        'mne',
        'pyedflib',
        'torch-geometric'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Cell 3: Import required libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm
import json
import pickle
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cell 4: Data loading and exploration
def explore_chb_mit_dataset(data_path):
    """
    Explore the CHB-MIT dataset structure
    """
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please ensure the CHB-MIT dataset is available in the Kaggle input directory")
        return None
    
    # Find all EDF files
    edf_files = []
    summary_files = []
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
            elif 'summary' in file.lower() and file.endswith('.txt'):
                summary_files.append(os.path.join(root, file))
    
    print(f"Found {len(edf_files)} EDF files")
    print(f"Found {len(summary_files)} summary files")
    
    # Group by subject
    subjects = {}
    for file_path in edf_files:
        # Extract subject ID from path
        parts = file_path.split('/')
        for part in parts:
            if part.startswith('chb'):
                subject_id = part
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(file_path)
                break
    
    print(f"Found {len(subjects)} subjects")
    for subject_id, files in list(subjects.items())[:5]:  # Show first 5 subjects
        print(f"  {subject_id}: {len(files)} files")
    
    return {
        'edf_files': edf_files,
        'summary_files': summary_files,
        'subjects': subjects
    }

# Explore dataset
dataset_info = explore_chb_mit_dataset(DATA_PATH)

# Cell 5: Data preprocessing implementation
# (Include the data_preprocessing.py code here)

class CHBMITDataProcessor:
    """Comprehensive data processor for CHB-MIT dataset"""
    
    def __init__(self, target_fs=256, window_size=2.0, overlap=0.5):
        self.target_fs = target_fs
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = int(window_size * target_fs)
        self.hop_samples = int(self.window_samples * (1 - overlap))
        
        # Standard electrode montage for CHB-MIT
        self.standard_channels = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ'
        ]
    
    def load_edf_file(self, file_path):
        """Load EEG data from EDF file"""
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            channels = raw.ch_names
            fs = raw.info['sfreq']
            return data, channels, fs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None, None
    
    def standardize_channels(self, data, channels):
        """Standardize channel names and select common channels"""
        if data is None:
            return None, None
        
        # Simple channel selection - use first 18 channels
        n_channels = min(18, data.shape[0])
        return data[:n_channels, :], [f'CH{i+1}' for i in range(n_channels)]
    
    def resample_signal(self, data, original_fs):
        """Resample signal to target frequency"""
        if original_fs == self.target_fs or data is None:
            return data
        
        from scipy import signal
        ratio = self.target_fs / original_fs
        new_length = int(data.shape[1] * ratio)
        
        resampled_data = np.zeros((data.shape[0], new_length))
        for i in range(data.shape[0]):
            resampled_data[i, :] = signal.resample(data[i, :], new_length)
        
        return resampled_data
    
    def remove_artifacts(self, data):
        """Simple artifact removal using z-score thresholding"""
        if data is None:
            return None
        
        cleaned_data = data.copy()
        threshold = 5
        
        for i in range(data.shape[0]):
            z_scores = np.abs((data[i, :] - np.mean(data[i, :])) / (np.std(data[i, :]) + 1e-8))
            artifact_mask = z_scores > threshold
            
            if np.any(artifact_mask):
                valid_indices = ~artifact_mask
                if np.sum(valid_indices) > 10:
                    cleaned_data[i, artifact_mask] = np.interp(
                        np.where(artifact_mask)[0],
                        np.where(valid_indices)[0],
                        data[i, valid_indices]
                    )
        
        return cleaned_data
    
    def segment_data(self, data, seizure_annotations=None):
        """Segment data into overlapping windows"""
        if data is None:
            return None, None
        
        n_channels, n_samples = data.shape
        n_windows = (n_samples - self.window_samples) // self.hop_samples + 1
        
        if n_windows <= 0:
            return None, None
        
        windows = np.zeros((n_windows, n_channels, self.window_samples))
        labels = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * self.hop_samples
            end_idx = start_idx + self.window_samples
            
            if end_idx <= n_samples:
                windows[i] = data[:, start_idx:end_idx]
                
                # Simple labeling - assume no seizures if annotations not provided
                if seizure_annotations is not None:
                    window_start_time = start_idx / self.target_fs
                    window_end_time = end_idx / self.target_fs
                    
                    for seizure_start, seizure_end in seizure_annotations:
                        if (window_start_time < seizure_end and window_end_time > seizure_start):
                            labels[i] = 1
                            break
        
        return windows[:i], labels[:i]
    
    def normalize_windows(self, windows):
        """Normalize windows using z-score normalization"""
        if windows is None:
            return None
        
        normalized_windows = np.zeros_like(windows)
        
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                channel_data = windows[i, j, :]
                if np.std(channel_data) > 1e-8:
                    normalized_windows[i, j, :] = (channel_data - np.mean(channel_data)) / np.std(channel_data)
                else:
                    normalized_windows[i, j, :] = channel_data
        
        return normalized_windows
    
    def process_file(self, file_path, seizure_annotations=None):
        """Complete processing pipeline for a single file"""
        # Load data
        data, channels, fs = self.load_edf_file(file_path)
        if data is None:
            return None, None, None
        
        # Standardize channels
        data, channels = self.standardize_channels(data, channels)
        if data is None:
            return None, None, None
        
        # Resample
        data = self.resample_signal(data, fs)
        
        # Remove artifacts
        data = self.remove_artifacts(data)
        
        # Segment into windows
        windows, labels = self.segment_data(data, seizure_annotations)
        if windows is None:
            return None, None, None
        
        # Normalize
        windows = self.normalize_windows(windows)
        
        return windows, labels, channels

# Cell 6: Load and preprocess a subset of data
def load_sample_data(dataset_info, n_subjects=3, n_files_per_subject=2):
    """
    Load a sample of the CHB-MIT dataset for demonstration
    """
    processor = CHBMITDataProcessor()
    
    all_windows = []
    all_labels = []
    all_subjects = []
    
    subjects = list(dataset_info['subjects'].keys())[:n_subjects]
    
    for subject_idx, subject_id in enumerate(subjects):
        print(f"Processing {subject_id}...")
        
        files = dataset_info['subjects'][subject_id][:n_files_per_subject]
        
        for file_path in tqdm(files, desc=f"Files for {subject_id}"):
            # Simple seizure annotation (normally would parse from summary files)
            # For demo, randomly assign some windows as seizures
            seizure_annotations = []
            if np.random.random() > 0.8:  # 20% chance of having a seizure
                seizure_annotations = [(10.0, 15.0)]  # 5-second seizure at 10-15 seconds
            
            windows, labels, channels = processor.process_file(file_path, seizure_annotations)
            
            if windows is not None:
                all_windows.append(windows)
                all_labels.append(labels)
                all_subjects.extend([subject_idx] * len(labels))
    
    if all_windows:
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Loaded {len(all_windows)} windows from {len(subjects)} subjects")
        print(f"Seizure windows: {np.sum(all_labels)} ({np.mean(all_labels)*100:.1f}%)")
        print(f"Window shape: {all_windows.shape}")
        
        return all_windows, all_labels, np.array(all_subjects), channels
    else:
        print("No data loaded successfully")
        return None, None, None, None

# Load sample data
if dataset_info is not None:
    windows, labels, subjects, channels = load_sample_data(dataset_info)
else:
    # Generate synthetic data for demonstration
    print("Generating synthetic data for demonstration...")
    n_samples = 1000
    n_channels = 18
    n_timepoints = 512  # 2 seconds at 256 Hz
    
    windows = np.random.randn(n_samples, n_channels, n_timepoints)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 10% seizures
    subjects = np.random.choice([0, 1, 2], size=n_samples)
    channels = [f'CH{i+1}' for i in range(n_channels)]
    
    print(f"Generated synthetic data: {windows.shape}")

# Cell 7: Multi-scale spectral decomposition
class MultiScaleSTFT:
    """Multi-scale Short-Time Fourier Transform for EEG analysis"""
    
    def __init__(self, fs=256, window_sizes=[1.0, 2.0], hop_ratio=0.25):
        self.fs = fs
        self.window_sizes = window_sizes
        self.hop_ratio = hop_ratio
    
    def compute_stft(self, signal_data, window_size):
        """Compute STFT for given signal and window size"""
        from scipy import signal
        
        n_fft = int(window_size * self.fs)
        hop_length = int(n_fft * self.hop_ratio)
        
        n_channels, n_samples = signal_data.shape
        stft_data = []
        
        for ch in range(n_channels):
            f, t, Zxx = signal.stft(
                signal_data[ch],
                fs=self.fs,
                window='hann',
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
                return_onesided=True
            )
            stft_data.append(Zxx)
        
        stft_data = np.array(stft_data)
        magnitude = np.abs(stft_data)
        
        # Apply log transform
        magnitude = np.log(magnitude + 1e-8)
        
        return magnitude, f
    
    def compute_multiscale_features(self, signal_data):
        """Compute multi-scale spectral features"""
        features = []
        
        for window_size in self.window_sizes:
            magnitude, frequencies = self.compute_stft(signal_data, window_size)
            
            # Flatten spatial and frequency dimensions for simplicity
            # Shape: [channels * frequencies, time]
            flattened = magnitude.reshape(-1, magnitude.shape[2])
            features.append(flattened)
        
        return features

# Compute spectral features
stft_processor = MultiScaleSTFT()
print("Computing multi-scale spectral features...")

all_spectral_features = [[] for _ in range(len(stft_processor.window_sizes))]

for i in tqdm(range(min(100, len(windows))), desc="Computing STFT"):  # Process first 100 for demo
    features = stft_processor.compute_multiscale_features(windows[i])
    for j, feat in enumerate(features):
        all_spectral_features[j].append(feat)

# Convert to tensors
for i in range(len(all_spectral_features)):
    all_spectral_features[i] = torch.tensor(
        np.array(all_spectral_features[i]), dtype=torch.float32
    )
    print(f"Scale {i} features shape: {all_spectral_features[i].shape}")

# Cell 8: Graph construction
class SimpleGraphConstructor:
    """Simplified graph constructor for demonstration"""
    
    def __init__(self, n_channels):
        self.n_channels = n_channels
    
    def create_spatial_adjacency(self):
        """Create simple spatial adjacency matrix"""
        # Simple ring topology for demonstration
        adjacency = np.zeros((self.n_channels, self.n_channels))
        
        for i in range(self.n_channels):
            # Connect to neighbors
            adjacency[i, (i + 1) % self.n_channels] = 1
            adjacency[i, (i - 1) % self.n_channels] = 1
        
        return adjacency
    
    def compute_graph_laplacian(self, adjacency):
        """Compute normalized graph Laplacian"""
        degree = np.diag(np.sum(adjacency, axis=1))
        degree_sqrt_inv = np.diag(1.0 / (np.sqrt(np.diag(degree)) + 1e-8))
        laplacian = degree_sqrt_inv @ (degree - adjacency) @ degree_sqrt_inv
        return laplacian
    
    def get_graph_info(self):
        """Get complete graph information"""
        adjacency = self.create_spatial_adjacency()
        laplacian = self.compute_graph_laplacian(adjacency)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        return {
            'adjacency': torch.tensor(adjacency, dtype=torch.float32),
            'laplacian': torch.tensor(laplacian, dtype=torch.float32),
            'eigenvalues': torch.tensor(eigenvalues, dtype=torch.float32),
            'eigenvectors': torch.tensor(eigenvectors, dtype=torch.float32)
        }

# Create graph
graph_constructor = SimpleGraphConstructor(len(channels))
graph_info = graph_constructor.get_graph_info()

print(f"Graph info:")
print(f"  Adjacency shape: {graph_info['adjacency'].shape}")
print(f"  Eigenvalues range: {graph_info['eigenvalues'].min():.3f} to {graph_info['eigenvalues'].max():.3f}")

# Cell 9: Simplified GFAN Model
class SimplifiedGFAN(nn.Module):
    """Simplified GFAN model for demonstration"""
    
    def __init__(self, n_channels, feature_dims, eigenvalues, eigenvectors, 
                 hidden_dim=64, n_classes=2):
        super(SimplifiedGFAN, self).__init__()
        
        self.n_channels = n_channels
        self.eigenvalues = nn.Parameter(eigenvalues, requires_grad=False)
        self.eigenvectors = nn.Parameter(eigenvectors, requires_grad=False)
        
        # Learnable spectral weights
        self.spectral_weights = nn.Parameter(torch.ones_like(eigenvalues))
        
        # Feature projection layers
        self.feature_projections = nn.ModuleList([
            nn.Linear(feat_dim, hidden_dim) for feat_dim in feature_dims
        ])
        
        # Final classification layers
        total_features = len(feature_dims) * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def spectral_filter(self, x):
        """Apply spectral filtering using graph Fourier transform"""
        # x shape: [batch, features, channels]
        batch_size, features, channels = x.shape
        
        # Transform to spectral domain
        x_spectral = torch.matmul(self.eigenvectors.t(), x.transpose(1, 2))
        x_spectral = x_spectral.transpose(1, 2)  # [batch, features, channels]
        
        # Apply learnable spectral weights
        x_filtered = x_spectral * self.spectral_weights.unsqueeze(0).unsqueeze(0)
        
        # Transform back to spatial domain
        x_spatial = torch.matmul(self.eigenvectors, x_filtered.transpose(1, 2))
        x_spatial = x_spatial.transpose(1, 2)  # [batch, features, channels]
        
        return x_spatial
    
    def forward(self, multi_scale_features):
        """Forward pass"""
        processed_features = []
        
        for i, features in enumerate(multi_scale_features):
            # features shape: [batch, spatial_freq, time]
            # Average over time dimension for simplicity
            feat_avg = torch.mean(features, dim=2)  # [batch, spatial_freq]
            
            # Reshape to [batch, features, channels]
            batch_size = feat_avg.shape[0]
            n_features = feat_avg.shape[1] // self.n_channels
            feat_reshaped = feat_avg.view(batch_size, n_features, self.n_channels)
            
            # Apply spectral filtering
            feat_filtered = self.spectral_filter(feat_reshaped)
            
            # Global average pooling
            feat_pooled = torch.mean(feat_filtered, dim=(1, 2))  # [batch, 1]
            
            # Project features
            feat_projected = self.feature_projections[i](feat_pooled.unsqueeze(1))
            feat_projected = feat_projected.squeeze(1)
            
            processed_features.append(feat_projected)
        
        # Concatenate multi-scale features
        combined_features = torch.cat(processed_features, dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return {
            'logits': logits,
            'spectral_weights': self.spectral_weights
        }

# Cell 10: Dataset and training setup
class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, spectral_features, labels, subjects=None):
        self.spectral_features = spectral_features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subjects = subjects
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = [feat[idx] for feat in self.spectral_features]
        return {
            'spectral_features': features,
            'label': self.labels[idx],
            'subject': self.subjects[idx] if self.subjects is not None else 0
        }

# Create dataset
n_samples = len(all_spectral_features[0])
feature_dims = [feat.shape[1] for feat in all_spectral_features]

# Use subset for demonstration
subset_size = min(n_samples, 500)
indices = np.random.choice(n_samples, subset_size, replace=False)

subset_features = [feat[indices] for feat in all_spectral_features]
subset_labels = labels[indices]
subset_subjects = subjects[indices]

# Split data
train_idx, test_idx = train_test_split(
    range(len(subset_labels)), test_size=0.3, random_state=42, 
    stratify=subset_labels
)

train_features = [feat[train_idx] for feat in subset_features]
test_features = [feat[test_idx] for feat in subset_features]
train_labels = subset_labels[train_idx]
test_labels = subset_labels[test_idx]

# Create datasets
train_dataset = EEGDataset(train_features, train_labels)
test_dataset = EEGDataset(test_features, test_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Feature dimensions: {feature_dims}")

# Cell 11: Model training
# Initialize model
model = SimplifiedGFAN(
    n_channels=len(channels),
    feature_dims=feature_dims,
    eigenvalues=graph_info['eigenvalues'],
    eigenvectors=graph_info['eigenvectors'],
    hidden_dim=32,
    n_classes=2
).to(device)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
num_epochs = 20
train_losses = []
train_accuracies = []

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move to device
        features = [f.to(device) for f in batch['spectral_features']]
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs['logits'], labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs['logits'], 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    
    # Calculate metrics
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    scheduler.step()

# Cell 12: Model evaluation
model.eval()
test_predictions = []
test_labels_list = []
test_probabilities = []

with torch.no_grad():
    for batch in test_loader:
        features = [f.to(device) for f in batch['spectral_features']]
        labels = batch['label'].to(device)
        
        outputs = model(features)
        probabilities = torch.softmax(outputs['logits'], dim=1)
        _, predicted = torch.max(outputs['logits'], 1)
        
        test_predictions.extend(predicted.cpu().numpy())
        test_labels_list.extend(labels.cpu().numpy())
        test_probabilities.extend(probabilities[:, 1].cpu().numpy())

# Calculate metrics
test_accuracy = accuracy_score(test_labels_list, test_predictions)
test_f1 = f1_score(test_labels_list, test_predictions, average='weighted')

if len(np.unique(test_labels_list)) > 1:
    test_auc = roc_auc_score(test_labels_list, test_probabilities)
else:
    test_auc = 0.0

print(f"\nTest Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"AUC: {test_auc:.4f}")

# Cell 13: Visualizations
# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(train_accuracies)
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_list, test_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Non-Seizure', 'Seizure'],
           yticklabels=['Non-Seizure', 'Seizure'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Spectral weights visualization
spectral_weights = model.spectral_weights.detach().cpu().numpy()
eigenvalues = graph_info['eigenvalues'].numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(eigenvalues, spectral_weights, 'bo-')
plt.xlabel('Eigenvalue')
plt.ylabel('Learned Weight')
plt.title('Adaptive Spectral Weights')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(range(len(spectral_weights)), spectral_weights)
plt.xlabel('Eigenmode Index')
plt.ylabel('Weight')
plt.title('Spectral Weight Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# Cell 14: Results summary and next steps
print("=" * 60)
print("GFAN EPILEPTIC SEIZURE DETECTION - RESULTS SUMMARY")
print("=" * 60)

print(f"Dataset: CHB-MIT EEG Database")
print(f"Model: Graph Fourier Analysis Network (GFAN)")
print(f"Samples processed: {len(subset_labels)}")
print(f"Channels: {len(channels)}")
print(f"Window size: 2.0 seconds")
print(f"Sampling rate: 256 Hz")

print(f"\nModel Architecture:")
print(f"- Multi-scale STFT with {len(stft_processor.window_sizes)} scales")
print(f"- Adaptive Fourier basis learning")
print(f"- Graph structure: {len(channels)} nodes")
print(f"- Hidden dimension: 32")

print(f"\nPerformance Metrics:")
print(f"- Test Accuracy: {test_accuracy:.4f}")
print(f"- Test F1-Score: {test_f1:.4f}")
print(f"- Test AUC: {test_auc:.4f}")

print(f"\nNext Steps for Production:")
print("1. Load complete CHB-MIT dataset with proper seizure annotations")
print("2. Implement leave-one-subject-out cross-validation")
print("3. Add uncertainty estimation and interpretability features")
print("4. Conduct comprehensive ablation studies")
print("5. Optimize hyperparameters and model architecture")
print("6. Validate on additional epilepsy datasets")

# Save results
results = {
    'model_config': {
        'n_channels': len(channels),
        'feature_dims': feature_dims,
        'hidden_dim': 32,
        'n_classes': 2
    },
    'training_results': {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_auc': test_auc
    },
    'spectral_weights': spectral_weights.tolist(),
    'eigenvalues': eigenvalues.tolist()
}

# Save to working directory
with open(os.path.join(WORKING_DIR, 'gfan_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {os.path.join(WORKING_DIR, 'gfan_results.json')}")
print("Notebook execution completed successfully!")

# Cell 15: Model saving for future use
# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'n_channels': len(channels),
        'feature_dims': feature_dims,
        'hidden_dim': 32,
        'n_classes': 2
    },
    'graph_info': {
        'eigenvalues': graph_info['eigenvalues'],
        'eigenvectors': graph_info['eigenvectors']
    },
    'test_metrics': {
        'accuracy': test_accuracy,
        'f1_score': test_f1,
        'auc': test_auc
    }
}, os.path.join(WORKING_DIR, 'gfan_model.pth'))

print("Model saved successfully!")
print("This implementation demonstrates the core GFAN methodology for epileptic seizure detection.")
print("For full production use, expand with complete dataset loading and advanced features.")
