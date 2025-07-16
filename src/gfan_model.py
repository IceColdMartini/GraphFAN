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
            return torch.tensor(0.0, device=self.linear.weight.device if hasattr(self, 'linear') else 'cpu')


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
                 uncertainty_method: Optional[str] = 'mc_dropout',
                 fusion_method: str = 'attention'):
        """
        Initialize GFAN model
        
        Args:
            n_channels: Number of EEG channels
            spectral_features_dims: List of feature dimensions for each spectral scale
            eigenvalues: Graph Laplacian eigenvalues
            eigenvectors: Graph Laplacian eigenvectors
            hidden_dims: List of hidden dimensions for GFAN layers
            n_classes: Number of output classes
            sparsity_reg: Sparsity regularization coefficient
            dropout_rate: Dropout rate
            uncertainty_method: Method for uncertainty estimation ('mc_dropout', 'variational', None)
            fusion_method: Multi-scale fusion method
        """
        super(GFAN, self).__init__()
        
        self.n_scales = len(spectral_features_dims)
        self.uncertainty_method = uncertainty_method
        
        # Input projection layers for each scale
        self.input_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dims[0]) for dim in spectral_features_dims
        ])
        
        # GFAN layers
        self.gfan_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gfan_layers.append(
                GFANLayer(
                    hidden_dims[i], hidden_dims[i+1],
                    eigenvalues, eigenvectors,
                    sparsity_reg, dropout_rate
                )
            )
        
        # Multi-scale fusion
        self.fusion_layer = MultiScaleFusion(
            [hidden_dims[-1]] * self.n_scales,
            fusion_method=fusion_method
        )
        
        # Classifier
        classifier_in_features = self.fusion_layer.output_dim
        if uncertainty_method:
            self.classifier = UncertaintyEstimationLayer(
                classifier_in_features, n_classes, uncertainty_method
            )
        else:
            self.classifier = nn.Linear(classifier_in_features, n_classes)
    
    def forward(self, multi_scale_features: List[torch.Tensor],
                return_uncertainty: bool = False, n_mc_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GFAN model
        
        Args:
            multi_scale_features: List of features from different scales
            return_uncertainty: Whether to return uncertainty estimates
            n_mc_samples: Number of samples for MC dropout
            
        Returns:
            Dictionary with logits, sparsity loss, and optionally uncertainty
        """
        scale_outputs = []
        total_sparsity_loss = 0
        
        # Process each scale through GFAN layers
        for i in range(self.n_scales):
            # Project input features
            x = self.input_projections[i](multi_scale_features[i])
            
            # Pass through GFAN layers
            for layer in self.gfan_layers:
                x = layer(x)
                total_sparsity_loss += layer.get_sparsity_loss()
            
            scale_outputs.append(x)
        
        # Fuse multi-scale features
        fused_features = self.fusion_layer(scale_outputs)
        
        # Global average pooling
        pooled_features = torch.mean(fused_features, dim=1)
        
        # Classifier and uncertainty estimation
        if self.uncertainty_method and (return_uncertainty or self.training):
            if self.uncertainty_method == 'mc_dropout':
                # Monte Carlo dropout
                mc_outputs = []
                for _ in range(n_mc_samples):
                    mc_outputs.append(self.classifier(pooled_features))
                
                logits_stack = torch.stack(mc_outputs)
                logits = torch.mean(logits_stack, dim=0)
                uncertainty = torch.var(logits_stack, dim=0)
            
            elif self.uncertainty_method == 'variational':
                logits = self.classifier(pooled_features)
                # Uncertainty from variance is more complex, often estimated via sampling
                # For simplicity, we can use MC sampling here as well
                mc_outputs = [self.classifier(pooled_features) for _ in range(n_mc_samples)]
                uncertainty = torch.var(torch.stack(mc_outputs), dim=0)
        else:
            logits = self.classifier(pooled_features)
            uncertainty = None

        # Prepare output dictionary
        output = {
            'logits': logits,
            'sparsity_loss': total_sparsity_loss / len(self.gfan_layers) if self.gfan_layers else 0,
        }
        
        if self.uncertainty_method == 'variational':
            output['kl_loss'] = self.classifier.get_kl_loss()
            
        if uncertainty is not None:
            output['uncertainty'] = uncertainty
            
        return output
    
    def get_eigenmode_attribution(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Get eigenmode attribution weights using gradient-based analysis.
        This method computes the gradient of the model's output with respect
        to the spectral weights to determine their importance.
        """
        self.train() # Ensure model is in training mode to get gradients
        
        # Ensure spectral weights require gradients
        for layer in self.gfan_layers:
            layer.fourier_layer.spectral_weights.requires_grad_(True)

        # Zero out any existing gradients
        self.zero_grad()

        # Forward pass to get logits
        outputs = self.forward(multi_scale_features, return_uncertainty=False)
        logits = outputs['logits']
        
        # Use the logit of the predicted class as the score for attribution
        predicted_class_logit = logits.max(dim=1)[0]
        
        # Backward pass to compute gradients of the score w.r.t. model parameters
        # We use a gradient of 1 for the predicted class logit
        predicted_class_logit.sum().backward()
        
        # Collect the gradients from the spectral weights of each GFAN layer
        attributions = []
        for layer in self.gfan_layers:
            if layer.fourier_layer.spectral_weights.grad is not None:
                # Use the absolute value of the gradient as the attribution
                attributions.append(layer.fourier_layer.spectral_weights.grad.abs().detach().clone())
        
        # Zero gradients again to clean up
        self.zero_grad()
        self.eval() # Return model to evaluation mode

        if not attributions:
            return torch.zeros_like(self.gfan_layers[0].fourier_layer.spectral_weights)

        # Average the attributions across all layers
        mean_attributions = torch.mean(torch.stack(attributions), dim=0)
        
        return mean_attributions
