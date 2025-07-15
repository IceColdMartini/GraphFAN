"""
Data preprocessing utilities for CHB-MIT EEG dataset
Handles EEG data loading, standardization, and windowing
"""

import numpy as np
import pandas as pd
import mne
import pyedflib
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CHBMITDataProcessor:
    """
    Comprehensive data processor for CHB-MIT dataset
    Handles EEG loading, preprocessing, and segmentation
    """
    
    def __init__(self, target_fs=256, window_size=2.0, overlap=0.5):
        """
        Initialize the data processor
        
        Args:
            target_fs (int): Target sampling frequency
            window_size (float): Window size in seconds
            overlap (float): Overlap ratio between windows
        """
        self.target_fs = target_fs
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = int(window_size * target_fs)
        self.hop_samples = int(self.window_samples * (1 - overlap))
        
        # Standard electrode montage for CHB-MIT
        self.standard_channels = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
        ]
        
    def load_edf_file(self, file_path):
        """
        Load EEG data from EDF file
        
        Args:
            file_path (str): Path to EDF file
            
        Returns:
            tuple: (data, channels, fs)
        """
        try:
            # Try using pyedflib first
            f = pyedflib.EdfReader(file_path)
            n_channels = f.signals_in_file
            signal_labels = f.getSignalLabels()
            fs = f.getSampleFrequency(0)
            
            # Load all signals
            signals = np.zeros((n_channels, f.getNSamples()[0]))
            for i in range(n_channels):
                signals[i, :] = f.readSignal(i)
            f.close()
            
            return signals, signal_labels, fs
            
        except:
            # Fallback to MNE
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            channels = raw.ch_names
            fs = raw.info['sfreq']
            return data, channels, fs
    
    def standardize_channels(self, data, channels):
        """
        Standardize channel names and select common channels
        
        Args:
            data (np.array): EEG data [channels x samples]
            channels (list): Channel names
            
        Returns:
            tuple: (standardized_data, standardized_channels)
        """
        # Clean channel names
        cleaned_channels = [ch.strip().replace(' ', '') for ch in channels]
        
        # Find common channels
        available_channels = []
        channel_indices = []
        
        for i, ch in enumerate(cleaned_channels):
            if ch in self.standard_channels:
                available_channels.append(ch)
                channel_indices.append(i)
        
        if len(channel_indices) == 0:
            # If no exact matches, try partial matching
            for std_ch in self.standard_channels:
                for i, ch in enumerate(cleaned_channels):
                    if std_ch.lower() in ch.lower() or ch.lower() in std_ch.lower():
                        if i not in channel_indices:
                            available_channels.append(std_ch)
                            channel_indices.append(i)
                        break
        
        # Select data for available channels
        if len(channel_indices) > 0:
            selected_data = data[channel_indices, :]
            return selected_data, available_channels
        else:
            # Use first 18 channels if no matching found
            n_channels = min(18, data.shape[0])
            return data[:n_channels, :], [f'CH{i+1}' for i in range(n_channels)]
    
    def resample_signal(self, data, original_fs):
        """
        Resample signal to target frequency
        
        Args:
            data (np.array): Input data [channels x samples]
            original_fs (float): Original sampling frequency
            
        Returns:
            np.array: Resampled data
        """
        if original_fs == self.target_fs:
            return data
        
        # Calculate resampling ratio
        ratio = self.target_fs / original_fs
        new_length = int(data.shape[1] * ratio)
        
        # Resample each channel
        resampled_data = np.zeros((data.shape[0], new_length))
        for i in range(data.shape[0]):
            resampled_data[i, :] = signal.resample(data[i, :], new_length)
        
        return resampled_data
    
    def remove_artifacts(self, data, method='threshold', threshold=5):
        """
        Remove artifacts from EEG data
        
        Args:
            data (np.array): Input data [channels x samples]
            method (str): Artifact removal method
            threshold (float): Z-score threshold for artifact detection
            
        Returns:
            np.array: Cleaned data
        """
        cleaned_data = data.copy()
        
        if method == 'threshold':
            # Z-score based artifact removal
            for i in range(data.shape[0]):
                z_scores = np.abs(zscore(data[i, :]))
                artifact_mask = z_scores > threshold
                
                # Interpolate artifacts
                if np.any(artifact_mask):
                    valid_indices = ~artifact_mask
                    if np.sum(valid_indices) > 10:  # Need enough valid points
                        cleaned_data[i, artifact_mask] = np.interp(
                            np.where(artifact_mask)[0],
                            np.where(valid_indices)[0],
                            data[i, valid_indices]
                        )
        
        elif method == 'bandpass':
            # Simple bandpass filtering
            nyquist = self.target_fs / 2
            low_freq = 0.5 / nyquist
            high_freq = 50 / nyquist
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            
            for i in range(data.shape[0]):
                cleaned_data[i, :] = signal.filtfilt(b, a, data[i, :])
        
        return cleaned_data
    
    def segment_data(self, data, seizure_annotations=None):
        """
        Segment data into overlapping windows
        
        Args:
            data (np.array): Input data [channels x samples]
            seizure_annotations (list): List of (start, end) seizure times in seconds
            
        Returns:
            tuple: (windows, labels)
        """
        n_channels, n_samples = data.shape
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_samples) // self.hop_samples + 1
        
        # Create windows
        windows = np.zeros((n_windows, n_channels, self.window_samples))
        labels = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * self.hop_samples
            end_idx = start_idx + self.window_samples
            
            if end_idx <= n_samples:
                windows[i] = data[:, start_idx:end_idx]
                
                # Determine label
                if seizure_annotations is not None:
                    window_start_time = start_idx / self.target_fs
                    window_end_time = end_idx / self.target_fs
                    
                    # Check if window overlaps with any seizure
                    for seizure_start, seizure_end in seizure_annotations:
                        if (window_start_time < seizure_end and window_end_time > seizure_start):
                            labels[i] = 1
                            break
        
        return windows[:i], labels[:i]
    
    def normalize_windows(self, windows):
        """
        Normalize windows using z-score normalization
        
        Args:
            windows (np.array): Input windows [n_windows x channels x samples]
            
        Returns:
            np.array: Normalized windows
        """
        normalized_windows = np.zeros_like(windows)
        
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                channel_data = windows[i, j, :]
                if np.std(channel_data) > 1e-8:  # Avoid division by zero
                    normalized_windows[i, j, :] = zscore(channel_data)
                else:
                    normalized_windows[i, j, :] = channel_data
        
        return normalized_windows
    
    def process_file(self, file_path, seizure_annotations=None):
        """
        Complete processing pipeline for a single file
        
        Args:
            file_path (str): Path to EDF file
            seizure_annotations (list): Seizure annotations
            
        Returns:
            tuple: (windows, labels, channels)
        """
        # Load data
        data, channels, fs = self.load_edf_file(file_path)
        
        # Standardize channels
        data, channels = self.standardize_channels(data, channels)
        
        # Resample
        data = self.resample_signal(data, fs)
        
        # Remove artifacts
        data = self.remove_artifacts(data)
        
        # Segment into windows
        windows, labels = self.segment_data(data, seizure_annotations)
        
        # Normalize
        windows = self.normalize_windows(windows)
        
        return windows, labels, channels


def load_chb_mit_annotations(summary_file_path):
    """
    Load seizure annotations from CHB-MIT summary files
    
    Args:
        summary_file_path (str): Path to summary file
        
    Returns:
        dict: Dictionary mapping file names to seizure annotations
    """
    annotations = {}
    
    try:
        with open(summary_file_path, 'r') as f:
            content = f.read()
        
        # Parse the summary file (simplified version)
        # In practice, you would need more sophisticated parsing
        lines = content.split('\n')
        current_file = None
        
        for line in lines:
            line = line.strip()
            if line.endswith('.edf'):
                current_file = line
                annotations[current_file] = []
            elif 'Seizure' in line and current_file:
                # Extract seizure times (this is a simplified parser)
                # You would need to implement proper parsing based on file format
                pass
    
    except Exception as e:
        print(f"Warning: Could not load annotations from {summary_file_path}: {e}")
    
    return annotations


def create_balanced_dataset(windows, labels, balance_ratio=0.3):
    """
    Create a balanced dataset by subsampling non-seizure windows
    
    Args:
        windows (np.array): Input windows
        labels (np.array): Labels
        balance_ratio (float): Ratio of non-seizure to seizure samples
        
    Returns:
        tuple: (balanced_windows, balanced_labels)
    """
    seizure_idx = np.where(labels == 1)[0]
    non_seizure_idx = np.where(labels == 0)[0]
    
    if len(seizure_idx) == 0:
        return windows, labels
    
    # Calculate number of non-seizure samples to keep
    n_non_seizure = int(len(seizure_idx) / balance_ratio)
    n_non_seizure = min(n_non_seizure, len(non_seizure_idx))
    
    # Randomly sample non-seizure windows
    selected_non_seizure = np.random.choice(
        non_seizure_idx, size=n_non_seizure, replace=False
    )
    
    # Combine indices
    selected_idx = np.concatenate([seizure_idx, selected_non_seizure])
    np.random.shuffle(selected_idx)
    
    return windows[selected_idx], labels[selected_idx]
