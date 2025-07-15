"""
Graph construction utilities for EEG electrode networks
Creates spatial and functional connectivity graphs
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class EEGGraphConstructor:
    """
    Construct graphs from EEG electrode layouts and functional connectivity
    """
    
    def __init__(self, electrode_positions=None, connectivity_method='spatial'):
        """
        Initialize graph constructor
        
        Args:
            electrode_positions (dict): 3D positions of electrodes
            connectivity_method (str): Method for edge construction
        """
        self.electrode_positions = electrode_positions or self._get_default_positions()
        self.connectivity_method = connectivity_method
        
    def _get_default_positions(self):
        """
        Get default electrode positions for CHB-MIT dataset
        Simplified 2D layout based on 10-20 system
        """
        # Simplified positions for common CHB-MIT channels
        positions = {
            'FP1-F7': np.array([-0.3, 0.8]),
            'F7-T7': np.array([-0.7, 0.3]),
            'T7-P7': np.array([-0.7, -0.3]),
            'P7-O1': np.array([-0.3, -0.8]),
            'FP1-F3': np.array([-0.2, 0.6]),
            'F3-C3': np.array([-0.4, 0.2]),
            'C3-P3': np.array([-0.4, -0.2]),
            'P3-O1': np.array([-0.2, -0.6]),
            'FP2-F4': np.array([0.2, 0.6]),
            'F4-C4': np.array([0.4, 0.2]),
            'C4-P4': np.array([0.4, -0.2]),
            'P4-O2': np.array([0.2, -0.6]),
            'FP2-F8': np.array([0.3, 0.8]),
            'F8-T8': np.array([0.7, 0.3]),
            'T8-P8': np.array([0.7, -0.3]),
            'P8-O2': np.array([0.3, -0.8]),
            'FZ-CZ': np.array([0.0, 0.0]),
            'CZ-PZ': np.array([0.0, -0.4]),
        }
        return positions
    
    def create_spatial_adjacency(self, channels, distance_threshold=0.3):
        """
        Create adjacency matrix based on spatial distance
        
        Args:
            channels (list): List of channel names
            distance_threshold (float): Maximum distance for connection
            
        Returns:
            np.array: Adjacency matrix
        """
        n_channels = len(channels)
        adjacency = np.zeros((n_channels, n_channels))
        
        # Get positions for available channels
        positions = []
        for ch in channels:
            if ch in self.electrode_positions:
                positions.append(self.electrode_positions[ch])
            else:
                # Use random position if not found
                positions.append(np.random.rand(2) * 2 - 1)
        
        positions = np.array(positions)
        
        # Calculate pairwise distances
        distances = squareform(pdist(positions))
        
        # Create adjacency based on distance threshold
        adjacency = (distances <= distance_threshold).astype(float)
        
        # Remove self-connections
        np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    def create_functional_adjacency(self, eeg_data, method='correlation', threshold=0.3):
        """
        Create adjacency matrix based on functional connectivity
        
        Args:
            eeg_data (np.array): EEG data [channels x samples]
            method (str): Connectivity method ('correlation', 'coherence', 'mutual_info')
            threshold (float): Threshold for binarizing connections
            
        Returns:
            np.array: Adjacency matrix
        """
        n_channels = eeg_data.shape[0]
        adjacency = np.zeros((n_channels, n_channels))
        
        if method == 'correlation':
            # Pearson correlation
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    if np.std(eeg_data[i]) > 1e-8 and np.std(eeg_data[j]) > 1e-8:
                        corr, _ = pearsonr(eeg_data[i], eeg_data[j])
                        if abs(corr) > threshold:
                            adjacency[i, j] = abs(corr)
                            adjacency[j, i] = abs(corr)
        
        elif method == 'coherence':
            # Magnitude squared coherence
            from scipy.signal import coherence
            
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    f, Cxy = coherence(eeg_data[i], eeg_data[j], fs=256)
                    # Average coherence in relevant frequency bands
                    coherence_val = np.mean(Cxy[(f >= 1) & (f <= 50)])
                    
                    if coherence_val > threshold:
                        adjacency[i, j] = coherence_val
                        adjacency[j, i] = coherence_val
        
        elif method == 'cosine':
            # Cosine similarity
            similarity_matrix = cosine_similarity(eeg_data)
            adjacency = (np.abs(similarity_matrix) > threshold).astype(float)
            np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    def create_hybrid_adjacency(self, channels, eeg_data, spatial_weight=0.5, 
                              functional_weight=0.5, distance_threshold=0.3, 
                              correlation_threshold=0.3):
        """
        Create hybrid adjacency combining spatial and functional connectivity
        
        Args:
            channels (list): Channel names
            eeg_data (np.array): EEG data [channels x samples]
            spatial_weight (float): Weight for spatial connectivity
            functional_weight (float): Weight for functional connectivity
            distance_threshold (float): Distance threshold for spatial connections
            correlation_threshold (float): Correlation threshold for functional connections
            
        Returns:
            np.array: Hybrid adjacency matrix
        """
        # Get spatial adjacency
        spatial_adj = self.create_spatial_adjacency(channels, distance_threshold)
        
        # Get functional adjacency
        functional_adj = self.create_functional_adjacency(
            eeg_data, method='correlation', threshold=correlation_threshold
        )
        
        # Combine with weights
        hybrid_adj = (spatial_weight * spatial_adj + 
                     functional_weight * functional_adj)
        
        return hybrid_adj
    
    def compute_graph_laplacian(self, adjacency, normalized=True):
        """
        Compute graph Laplacian matrix
        
        Args:
            adjacency (np.array): Adjacency matrix
            normalized (bool): Whether to compute normalized Laplacian
            
        Returns:
            np.array: Laplacian matrix
        """
        # Compute degree matrix
        degree = np.diag(np.sum(adjacency, axis=1))
        
        if normalized:
            # Normalized Laplacian: L = D^(-1/2) * (D - A) * D^(-1/2)
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(degree) + 1e-8))
            laplacian = degree_sqrt_inv @ (degree - adjacency) @ degree_sqrt_inv
        else:
            # Unnormalized Laplacian: L = D - A
            laplacian = degree - adjacency
        
        return laplacian
    
    def eigen_decomposition(self, laplacian, k=None):
        """
        Compute eigendecomposition of Laplacian
        
        Args:
            laplacian (np.array): Laplacian matrix
            k (int): Number of eigenvectors to compute (None for all)
            
        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        if k is None or k >= laplacian.shape[0]:
            # Compute all eigenvalues/eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        else:
            # Compute only k smallest eigenvalues/eigenvectors
            from scipy.sparse.linalg import eigsh
            try:
                eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
            except:
                # Fallback to full decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                eigenvalues = eigenvalues[:k]
                eigenvectors = eigenvectors[:, :k]
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors


class GraphConvolutionLayer(nn.Module):
    """
    Graph convolution layer using spectral filtering
    """
    
    def __init__(self, in_features, out_features, eigenvalues, eigenvectors, 
                 learnable_filters=True, dropout=0.1):
        """
        Initialize graph convolution layer
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            eigenvalues (torch.Tensor): Laplacian eigenvalues
            eigenvectors (torch.Tensor): Laplacian eigenvectors
            learnable_filters (bool): Whether to learn spectral filters
            dropout (float): Dropout rate
        """
        super(GraphConvolutionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.eigenvalues = nn.Parameter(eigenvalues, requires_grad=False)
        self.eigenvectors = nn.Parameter(eigenvectors, requires_grad=False)
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)
        
        # Learnable spectral filters
        if learnable_filters:
            self.spectral_weights = nn.Parameter(
                torch.ones(len(eigenvalues)), requires_grad=True
            )
        else:
            self.spectral_weights = nn.Parameter(
                torch.ones(len(eigenvalues)), requires_grad=False
            )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features [batch_size x nodes x features]
            
        Returns:
            torch.Tensor: Output features [batch_size x nodes x out_features]
        """
        batch_size, n_nodes, n_features = x.shape
        
        # Apply linear transformation
        x = self.linear(x)  # [batch_size x nodes x out_features]
        
        # Spectral filtering
        # x = U * diag(weights) * U^T * x
        x_transformed = torch.zeros_like(x)
        
        for b in range(batch_size):
            # Transform to spectral domain
            x_spectral = torch.mm(self.eigenvectors.T, x[b])  # [nodes x out_features]
            
            # Apply spectral weights
            x_filtered = x_spectral * self.spectral_weights.unsqueeze(1)
            
            # Transform back to spatial domain
            x_transformed[b] = torch.mm(self.eigenvectors, x_filtered)
        
        # Apply activation and dropout
        x_transformed = self.activation(x_transformed)
        x_transformed = self.dropout(x_transformed)
        
        return x_transformed


def create_graph_from_windows(windows, channels, method='hybrid'):
    """
    Create graph structure from windowed EEG data
    
    Args:
        windows (np.array): EEG windows [n_windows x channels x samples]
        channels (list): Channel names
        method (str): Graph construction method
        
    Returns:
        dict: Graph information including adjacency and Laplacian
    """
    graph_constructor = EEGGraphConstructor()
    
    if method == 'spatial':
        # Use only spatial connectivity
        adjacency = graph_constructor.create_spatial_adjacency(channels)
        
    elif method == 'functional':
        # Average functional connectivity across all windows
        adjacencies = []
        for window in windows:
            adj = graph_constructor.create_functional_adjacency(window)
            adjacencies.append(adj)
        adjacency = np.mean(adjacencies, axis=0)
        
    elif method == 'hybrid':
        # Use hybrid approach with first window for functional connectivity
        if len(windows) > 0:
            adjacency = graph_constructor.create_hybrid_adjacency(
                channels, windows[0]
            )
        else:
            adjacency = graph_constructor.create_spatial_adjacency(channels)
    
    else:
        raise ValueError(f"Unknown graph construction method: {method}")
    
    # Compute Laplacian and eigendecomposition
    laplacian = graph_constructor.compute_graph_laplacian(adjacency)
    eigenvalues, eigenvectors = graph_constructor.eigen_decomposition(laplacian)
    
    # Convert to tensors
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    laplacian_tensor = torch.tensor(laplacian, dtype=torch.float32)
    eigenvalues_tensor = torch.tensor(eigenvalues, dtype=torch.float32)
    eigenvectors_tensor = torch.tensor(eigenvectors, dtype=torch.float32)
    
    return {
        'adjacency': adjacency_tensor,
        'laplacian': laplacian_tensor,
        'eigenvalues': eigenvalues_tensor,
        'eigenvectors': eigenvectors_tensor,
        'channels': channels
    }
