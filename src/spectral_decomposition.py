"""
Multi-scale spectral decomposition for EEG signals
Implements STFT with multiple window sizes and spectral features
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFT:
    """
    Multi-scale Short-Time Fourier Transform for EEG analysis
    Extracts spectral features at different temporal resolutions
    """
    
    def __init__(self, fs=256, window_sizes=[1.0, 2.0, 4.0], hop_ratio=0.25, 
                 freq_bands=None, log_transform=True):
        """
        Initialize multi-scale STFT
        
        Args:
            fs (int): Sampling frequency
            window_sizes (list): Window sizes in seconds
            hop_ratio (float): Hop size as ratio of window size
            freq_bands (dict): Frequency bands of interest
            log_transform (bool): Apply log transform to magnitudes
        """
        self.fs = fs
        self.window_sizes = window_sizes
        self.hop_ratio = hop_ratio
        self.log_transform = log_transform
        
        # Standard EEG frequency bands
        self.freq_bands = freq_bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Pre-compute window parameters
        self.window_params = []
        for window_size in window_sizes:
            n_fft = int(window_size * fs)
            hop_length = int(n_fft * hop_ratio)
            self.window_params.append({
                'window_size': window_size,
                'n_fft': n_fft,
                'hop_length': hop_length
            })
    
    def compute_stft(self, signal_data, window_idx=0):
        """
        Compute STFT for given signal and window size
        
        Args:
            signal_data (np.array): Input signal [channels x samples]
            window_idx (int): Index of window size to use
            
        Returns:
            dict: STFT results with magnitude, phase, and frequencies
        """
        params = self.window_params[window_idx]
        n_channels, n_samples = signal_data.shape
        
        # Compute STFT for each channel
        stft_data = []
        for ch in range(n_channels):
            f, t, Zxx = signal.stft(
                signal_data[ch],
                fs=self.fs,
                window='hann',
                nperseg=params['n_fft'],
                noverlap=params['n_fft'] - params['hop_length'],
                return_onesided=True
            )
            stft_data.append(Zxx)
        
        stft_data = np.array(stft_data)  # [channels x freqs x time]
        
        # Extract magnitude and phase
        magnitude = np.abs(stft_data)
        phase = np.angle(stft_data)
        
        # Apply log transform to magnitude
        if self.log_transform:
            magnitude = np.log(magnitude + 1e-8)
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'frequencies': f,
            'times': t,
            'window_size': params['window_size']
        }
    
    def compute_multiscale_stft(self, signal_data):
        """
        Compute STFT at all window sizes
        
        Args:
            signal_data (np.array): Input signal [channels x samples]
            
        Returns:
            list: List of STFT results for each scale
        """
        multiscale_stft = []
        for i in range(len(self.window_sizes)):
            stft_result = self.compute_stft(signal_data, window_idx=i)
            multiscale_stft.append(stft_result)
        
        return multiscale_stft
    
    def extract_band_power(self, stft_result):
        """
        Extract power in specific frequency bands
        
        Args:
            stft_result (dict): STFT result from compute_stft
            
        Returns:
            dict: Band power features
        """
        magnitude = stft_result['magnitude']
        frequencies = stft_result['frequencies']
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find frequency indices
            freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            
            if np.any(freq_mask):
                # Average power in frequency band across time
                band_power = np.mean(magnitude[:, freq_mask, :], axis=1)  # [channels x time]
                band_powers[band_name] = band_power
            else:
                # If no frequencies in band, create zeros
                band_powers[band_name] = np.zeros((magnitude.shape[0], magnitude.shape[2]))
        
        return band_powers
    
    def compute_spectral_features(self, signal_data):
        """
        Compute comprehensive spectral features
        
        Args:
            signal_data (np.array): Input signal [channels x samples]
            
        Returns:
            dict: Dictionary of spectral features
        """
        features = {
            'multiscale_stft': [],
            'band_powers': [],
            'spectral_entropy': [],
            'peak_frequencies': []
        }
        
        # Compute multi-scale STFT
        multiscale_stft = self.compute_multiscale_stft(signal_data)
        features['multiscale_stft'] = multiscale_stft
        
        # Extract features for each scale
        for stft_result in multiscale_stft:
            # Band powers
            band_powers = self.extract_band_power(stft_result)
            features['band_powers'].append(band_powers)
            
            # Spectral entropy
            entropy = self.compute_spectral_entropy(stft_result['magnitude'])
            features['spectral_entropy'].append(entropy)
            
            # Peak frequencies
            peak_freqs = self.find_peak_frequencies(
                stft_result['magnitude'], stft_result['frequencies']
            )
            features['peak_frequencies'].append(peak_freqs)
        
        return features
    
    def compute_spectral_entropy(self, magnitude):
        """
        Compute spectral entropy for each channel and time point
        
        Args:
            magnitude (np.array): STFT magnitude [channels x freqs x time]
            
        Returns:
            np.array: Spectral entropy [channels x time]
        """
        # Normalize to create probability distribution
        power = magnitude ** 2
        power_norm = power / (np.sum(power, axis=1, keepdims=True) + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-8), axis=1)
        
        return entropy
    
    def find_peak_frequencies(self, magnitude, frequencies):
        """
        Find dominant frequencies for each channel
        
        Args:
            magnitude (np.array): STFT magnitude [channels x freqs x time]
            frequencies (np.array): Frequency values
            
        Returns:
            np.array: Peak frequencies [channels x time]
        """
        # Find frequency with maximum power at each time point
        peak_freq_idx = np.argmax(magnitude, axis=1)  # [channels x time]
        peak_frequencies = frequencies[peak_freq_idx]
        
        return peak_frequencies


class SpectralAugmentation:
    """
    Data augmentation techniques for spectral representations
    """
    
    def __init__(self, freq_mask_ratio=0.1, time_mask_ratio=0.1, 
                 mixup_alpha=0.2, phase_noise_std=0.1):
        """
        Initialize spectral augmentation
        
        Args:
            freq_mask_ratio (float): Ratio of frequency bins to mask
            time_mask_ratio (float): Ratio of time frames to mask
            mixup_alpha (float): Alpha parameter for mixup
            phase_noise_std (float): Standard deviation for phase noise
        """
        self.freq_mask_ratio = freq_mask_ratio
        self.time_mask_ratio = time_mask_ratio
        self.mixup_alpha = mixup_alpha
        self.phase_noise_std = phase_noise_std
    
    def frequency_masking(self, magnitude):
        """
        Apply frequency masking to STFT magnitude
        
        Args:
            magnitude (np.array): STFT magnitude [channels x freqs x time]
            
        Returns:
            np.array: Augmented magnitude
        """
        augmented = magnitude.copy()
        n_freqs = magnitude.shape[1]
        
        # Number of frequency bins to mask
        n_mask = int(n_freqs * self.freq_mask_ratio)
        
        if n_mask > 0:
            # Random frequency range to mask
            mask_start = np.random.randint(0, n_freqs - n_mask + 1)
            mask_end = mask_start + n_mask
            
            # Apply mask (set to minimum value)
            augmented[:, mask_start:mask_end, :] = np.min(magnitude)
        
        return augmented
    
    def time_masking(self, magnitude):
        """
        Apply time masking to STFT magnitude
        
        Args:
            magnitude (np.array): STFT magnitude [channels x freqs x time]
            
        Returns:
            np.array: Augmented magnitude
        """
        augmented = magnitude.copy()
        n_times = magnitude.shape[2]
        
        # Number of time frames to mask
        n_mask = int(n_times * self.time_mask_ratio)
        
        if n_mask > 0:
            # Random time range to mask
            mask_start = np.random.randint(0, n_times - n_mask + 1)
            mask_end = mask_start + n_mask
            
            # Apply mask
            augmented[:, :, mask_start:mask_end] = np.min(magnitude)
        
        return augmented
    
    def spectral_mixup(self, magnitude1, magnitude2, label1, label2):
        """
        Apply mixup augmentation to spectral features
        
        Args:
            magnitude1, magnitude2 (np.array): STFT magnitudes
            label1, label2 (float): Corresponding labels
            
        Returns:
            tuple: (mixed_magnitude, mixed_label)
        """
        # Sample mixing ratio
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Mix magnitudes
        mixed_magnitude = lam * magnitude1 + (1 - lam) * magnitude2
        
        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_magnitude, mixed_label
    
    def phase_perturbation(self, phase):
        """
        Add noise to phase information
        
        Args:
            phase (np.array): STFT phase [channels x freqs x time]
            
        Returns:
            np.array: Perturbed phase
        """
        noise = np.random.normal(0, self.phase_noise_std, phase.shape)
        perturbed_phase = phase + noise
        
        # Keep phase in [-π, π] range
        perturbed_phase = np.angle(np.exp(1j * perturbed_phase))
        
        return perturbed_phase
    
    def augment_stft(self, stft_result, apply_freq_mask=True, apply_time_mask=True, 
                    apply_phase_noise=True):
        """
        Apply multiple augmentations to STFT result
        
        Args:
            stft_result (dict): STFT result dictionary
            apply_freq_mask (bool): Whether to apply frequency masking
            apply_time_mask (bool): Whether to apply time masking
            apply_phase_noise (bool): Whether to apply phase noise
            
        Returns:
            dict: Augmented STFT result
        """
        augmented = stft_result.copy()
        
        if apply_freq_mask:
            augmented['magnitude'] = self.frequency_masking(augmented['magnitude'])
        
        if apply_time_mask:
            augmented['magnitude'] = self.time_masking(augmented['magnitude'])
        
        if apply_phase_noise:
            augmented['phase'] = self.phase_perturbation(augmented['phase'])
        
        return augmented


def create_spectral_features_tensor(spectral_features, device='cpu'):
    """
    Convert spectral features to PyTorch tensors
    
    Args:
        spectral_features (dict): Output from MultiScaleSTFT.compute_spectral_features
        device (str): Device to place tensors on
        
    Returns:
        dict: Dictionary of PyTorch tensors
    """
    tensor_features = {}
    
    # Convert multi-scale STFT magnitudes
    stft_tensors = []
    for stft_result in spectral_features['multiscale_stft']:
        magnitude_tensor = torch.tensor(
            stft_result['magnitude'], dtype=torch.float32, device=device
        )
        stft_tensors.append(magnitude_tensor)
    
    tensor_features['multiscale_stft'] = stft_tensors
    
    # Convert band powers
    band_power_tensors = []
    for band_powers in spectral_features['band_powers']:
        band_tensor = {}
        for band_name, power in band_powers.items():
            band_tensor[band_name] = torch.tensor(
                power, dtype=torch.float32, device=device
            )
        band_power_tensors.append(band_tensor)
    
    tensor_features['band_powers'] = band_power_tensors
    
    # Convert other features
    tensor_features['spectral_entropy'] = [
        torch.tensor(entropy, dtype=torch.float32, device=device)
        for entropy in spectral_features['spectral_entropy']
    ]
    
    tensor_features['peak_frequencies'] = [
        torch.tensor(peaks, dtype=torch.float32, device=device)
        for peaks in spectral_features['peak_frequencies']
    ]
    
    return tensor_features
