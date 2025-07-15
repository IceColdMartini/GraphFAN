"""
GFAN (Graph Fourier Analysis Network) model implementation
Adaptive Fourier Basis Learning for Epileptic Seizure Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class AdaptiveFourierBasisLayer(nn.Module):
    """
    Adaptive Fourier basis learning layer with sparsity regularization
    """
    
    def __init__(self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor,
                 sparsity_reg: float = 0.01, dropout_rate: float = 0.1):
        """
        Initialize adaptive Fourier basis layer
        
        Args:
            eigenvalues: Graph Laplacian eigenvalues
            eigenvectors: Graph Laplacian eigenvectors
            sparsity_reg: L1 regularization coefficient for sparsity
            dropout_rate: Spectral dropout rate
        """
        super(AdaptiveFourierBasisLayer, self).__init__()
        
        self.register_buffer('eigenvalues', eigenvalues)
        self.register_buffer('eigenvectors', eigenvectors)
        
        # Learnable spectral weights (diagonal filter in spectral domain)
        self.spectral_weights = nn.Parameter(
            torch.ones_like(eigenvalues), requires_grad=True
        )
        
        self.sparsity_reg = sparsity_reg
        self.spectral_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive spectral filtering
        
        Args:
            x: Input features [batch_size, nodes, features]
            
        Returns:
            Filtered features [batch_size, nodes, features]
        """
        batch_size, n_nodes, n_features = x.shape
        
        # Transform to spectral domain: U^T * x
        x_spectral = torch.matmul(self.eigenvectors.t(), x.transpose(1, 2))
        x_spectral = x_spectral.transpose(1, 2)  # [batch_size, nodes, features]
        
        # Apply learnable spectral weights with dropout
        weights = self.spectral_dropout(self.spectral_weights)
        x_filtered = x_spectral * weights.unsqueeze(0).unsqueeze(2)
        
        # Transform back to spatial domain: U * filtered_x
        x_output = torch.matmul(self.eigenvectors, x_filtered.transpose(1, 2))
        x_output = x_output.transpose(1, 2)  # [batch_size, nodes, features]
        
        return x_output
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Compute L1 sparsity regularization loss
        """
        return self.sparsity_reg * torch.sum(torch.abs(self.spectral_weights))


class GFANLayer(nn.Module):
    """
    Core GFAN layer combining spectral filtering with feature transformation
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 eigenvalues: torch.Tensor, eigenvectors: torch.Tensor,
                 sparsity_reg: float = 0.01, dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize GFAN layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            eigenvalues: Graph Laplacian eigenvalues
            eigenvectors: Graph Laplacian eigenvectors
            sparsity_reg: Sparsity regularization coefficient
            dropout_rate: Dropout rate
            activation: Activation function type
        """
        super(GFANLayer, self).__init__()
        
        # Adaptive Fourier basis layer
        self.fourier_layer = AdaptiveFourierBasisLayer(
            eigenvalues, eigenvectors, sparsity_reg, dropout_rate
        )
        
        # Feature transformation
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Activation and normalization
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GFAN layer
        
        Args:
            x: Input features [batch_size, nodes, in_features]
            
        Returns:
            Output features [batch_size, nodes, out_features]
        """
        # Apply spectral filtering
        x_filtered = self.fourier_layer(x)
        
        # Linear transformation
        x_linear = self.linear(x_filtered) + self.bias
        
        # Activation and normalization
        x_output = self.activation(x_linear)
        x_output = self.layer_norm(x_output)
        x_output = self.dropout(x_output)
        
        return x_output
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Get sparsity regularization loss
        """
        return self.fourier_layer.get_sparsity_loss()


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion with attention mechanism
    """
    
    def __init__(self, feature_dims: List[int], hidden_dim: int = 128,
                 fusion_method: str = 'attention'):
        """
        Initialize multi-scale fusion module
        
        Args:
            feature_dims: List of feature dimensions for each scale
            hidden_dim: Hidden dimension for attention
            fusion_method: Fusion method ('attention', 'concat', 'average')
        """
        super(MultiScaleFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            # Attention-based fusion
            self.attention_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) for dim in feature_dims
            ])
            self.output_dim = sum(feature_dims)
            
        elif fusion_method == 'concat':
            # Simple concatenation
            self.output_dim = sum(feature_dims)
            
        elif fusion_method == 'average':
            # Weighted average (requires same dimensions)
            assert len(set(feature_dims)) == 1, "All dimensions must be equal for averaging"
            self.output_dim = feature_dims[0]
            self.scale_weights = nn.Parameter(torch.ones(len(feature_dims)))
    
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features
        
        Args:
            multi_scale_features: List of features from different scales
            
        Returns:
            Fused features
        """
        if self.fusion_method == 'attention':
            # Compute attention weights
            attention_weights = []
            for i, features in enumerate(multi_scale_features):
                # Global average pooling across nodes
                pooled = torch.mean(features, dim=1)  # [batch_size, features]
                attention = torch.softmax(self.attention_layers[i](pooled), dim=0)
                attention_weights.append(attention)
            
            # Apply attention weights and concatenate
            weighted_features = []
            for i, (features, weight) in enumerate(zip(multi_scale_features, attention_weights)):
                weighted = features * weight.unsqueeze(1).unsqueeze(2)
                weighted_features.append(weighted)
            
            fused = torch.cat(weighted_features, dim=2)
            
        elif self.fusion_method == 'concat':
            # Simple concatenation
            fused = torch.cat(multi_scale_features, dim=2)
            
        elif self.fusion_method == 'average':
            # Weighted average
            weights = torch.softmax(self.scale_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, multi_scale_features))
        
        return fused


class UncertaintyEstimationLayer(nn.Module):
    """
    Variational layer for uncertainty estimation
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 uncertainty_method: str = 'mc_dropout'):
        """
        Initialize uncertainty estimation layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            uncertainty_method: Method for uncertainty estimation
        """
        super(UncertaintyEstimationLayer, self).__init__()
        
        self.uncertainty_method = uncertainty_method
        
        if uncertainty_method == 'variational':
            # Variational Bayesian layer
            self.weight_mu = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
            self.weight_logvar = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_logvar = nn.Parameter(torch.zeros(out_features))
            
        else:
            # Standard layer with MC dropout
            self.linear = nn.Linear(in_features, out_features)
            self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Forward pass with uncertainty estimation
        """
        if self.uncertainty_method == 'variational' and training:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
            
            output = torch.matmul(x, weight) + bias
            
        else:
            # Standard forward pass with dropout
            if hasattr(self, 'linear'):
                output = self.linear(x)
                if training or self.uncertainty_method == 'mc_dropout':
                    output = self.dropout(output)
            else:
                # Use mean weights for inference
                output = torch.matmul(x, self.weight_mu) + self.bias_mu
        
        return output
    
    def get_kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence loss for variational weights
        """
        if self.uncertainty_method == 'variational':
            weight_kl = -0.5 * torch.sum(1 + self.weight_logvar - 
                                       self.weight_mu.pow(2) - 
                                       self.weight_logvar.exp())
            bias_kl = -0.5 * torch.sum(1 + self.bias_logvar - 
                                     self.bias_mu.pow(2) - 
                                     self.bias_logvar.exp())
            return weight_kl + bias_kl
        else:
            return torch.tensor(0.0)


class GFAN(nn.Module):
    """
    Complete GFAN model for epileptic seizure detection
    """
    
    def __init__(self, 
                 n_channels: int,
                 spectral_features_dims: List[int],
                 eigenvalues: torch.Tensor,
                 eigenvectors: torch.Tensor,
                 hidden_dims: List[int] = [128, 64, 32],
                 n_classes: int = 2,
                 sparsity_reg: float = 0.01,
                 dropout_rate: float = 0.1,
                 uncertainty_estimation: bool = True,
                 fusion_method: str = 'attention'):
        """
        Initialize GFAN model
        
        Args:
            n_channels: Number of EEG channels
            spectral_features_dims: Dimensions of spectral features for each scale
            eigenvalues: Graph Laplacian eigenvalues
            eigenvectors: Graph Laplacian eigenvectors
            hidden_dims: Hidden layer dimensions
            n_classes: Number of output classes
            sparsity_reg: Sparsity regularization coefficient
            dropout_rate: Dropout rate
            uncertainty_estimation: Whether to use uncertainty estimation
            fusion_method: Multi-scale fusion method
        """
        super(GFAN, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.uncertainty_estimation = uncertainty_estimation
        
        # Multi-scale GFAN layers
        self.gfan_layers = nn.ModuleList()
        for i, feat_dim in enumerate(spectral_features_dims):
            layer = GFANLayer(
                feat_dim, hidden_dims[0],
                eigenvalues, eigenvectors,
                sparsity_reg, dropout_rate
            )
            self.gfan_layers.append(layer)
        
        # Multi-scale fusion
        self.fusion = MultiScaleFusion(
            [hidden_dims[0]] * len(spectral_features_dims),
            hidden_dim=hidden_dims[0],
            fusion_method=fusion_method
        )
        
        # Additional GFAN layers
        self.additional_gfan_layers = nn.ModuleList()
        current_dim = self.fusion.output_dim
        
        for hidden_dim in hidden_dims[1:]:
            layer = GFANLayer(
                current_dim, hidden_dim,
                eigenvalues, eigenvectors,
                sparsity_reg, dropout_rate
            )
            self.additional_gfan_layers.append(layer)
            current_dim = hidden_dim
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        if uncertainty_estimation:
            self.classifier = UncertaintyEstimationLayer(
                current_dim, n_classes, 'mc_dropout'
            )
        else:
            self.classifier = nn.Linear(current_dim, n_classes)
    
    def forward(self, multi_scale_features: List[torch.Tensor],
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GFAN
        
        Args:
            multi_scale_features: List of spectral features for each scale
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing predictions and optional uncertainty
        """
        # Apply GFAN layers to each scale
        gfan_outputs = []
        sparsity_losses = []
        
        for i, features in enumerate(multi_scale_features):
            output = self.gfan_layers[i](features)
            gfan_outputs.append(output)
            sparsity_losses.append(self.gfan_layers[i].get_sparsity_loss())
        
        # Fuse multi-scale features
        fused_features = self.fusion(gfan_outputs)
        
        # Apply additional GFAN layers
        x = fused_features
        for layer in self.additional_gfan_layers:
            x = layer(x)
            sparsity_losses.append(layer.get_sparsity_loss())
        
        # Global pooling across nodes
        x = x.transpose(1, 2)  # [batch_size, features, nodes]
        x = self.global_pool(x).squeeze(2)  # [batch_size, features]
        
        # Classification
        if return_uncertainty and self.uncertainty_estimation:
            # Multiple forward passes for uncertainty estimation
            predictions = []
            for _ in range(10):  # Monte Carlo samples
                pred = self.classifier(x, training=True)
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=0)  # [samples, batch, classes]
            mean_pred = torch.mean(predictions, dim=0)
            uncertainty = torch.var(predictions, dim=0)
            
            output = {
                'logits': mean_pred,
                'uncertainty': uncertainty,
                'sparsity_loss': sum(sparsity_losses)
            }
        else:
            logits = self.classifier(x)
            output = {
                'logits': logits,
                'sparsity_loss': sum(sparsity_losses)
            }
        
        return output
    
    def get_eigenmode_attribution(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Get attribution of different eigenmodes for interpretability
        
        Args:
            multi_scale_features: Input features
            
        Returns:
            Eigenmode attribution weights
        """
        attributions = []
        
        for i, layer in enumerate(self.gfan_layers):
            weights = layer.fourier_layer.spectral_weights
            attributions.append(weights.detach())
        
        # Average across scales
        mean_attribution = torch.stack(attributions).mean(dim=0)
        return mean_attribution
