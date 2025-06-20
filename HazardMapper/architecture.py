"""HazardMapper - Architecture Module
========================
This module contains various neural network architectures for hazard susceptibility modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# MLP to test 1d data 
class MLP(nn.Module):
    def __init__(self, logger, device, num_vars, n_layers=2, n_nodes=128, 
                 dropout=True, drop_value=0.4, patch_size=1):
        """
        MLP architecture designed to take feature vectors as input.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/features
            n_layers: Number of hidden layers
            n_nodes: Number of nodes in each hidden layer
            dropout: Whether to use dropout
            drop_value: Dropout probability
            patch_size: Should be 1 for pure feature vector input
        """
        super(MLP, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        
        # For feature vector input, input size is just num_vars
        # (when patch_size=1, we get features only with no spatial context)
        input_size = num_vars
        
        # Build the MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, n_nodes))
        layers.append(nn.BatchNorm1d(n_nodes))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(drop_value))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_nodes, n_nodes))
            layers.append(nn.BatchNorm1d(n_nodes))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(drop_value))
        
        # Output layer
        layers.append(nn.Linear(n_nodes, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        self.logger.info(f"Created MLP with {n_layers} layers, {n_nodes} nodes per layer")
        self.logger.info(f"Input size: {input_size}, Output size: 1")

    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape [batch_size, num_vars, 1, 1]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        # Reshape input based on whether it's a single feature vector or patches
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_vars)
    
        # Forward pass through the model
        return self.model(x)

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels: int, n_layers: int = 2, n_filters: int = 32, drop_value: float = 0.3, patch_size: int = 5):
        super(CNN, self).__init__()
        
        conv_layers = []
        current_channels = in_channels

        for i in range(n_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            current_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Output size: current_channels * 5 * 5
            nn.Linear(current_channels * patch_size * patch_size, 256),
            nn.ReLU(),
            nn.Dropout(drop_value),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)

# Simple CNN architecture to test the pipeline
class SimpleCNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters=16, n_layers=2, dropout=False, 
                 drop_value=0.2, patch_size=5):
        """
        Simple CNN with constant filters and variable number of convolutional layers.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/channels
            filters: Number of filters for all convolution layers
            num_layers: Number of shared convolutional layers after concatenation
            dropout: Whether to use dropout
            drop_value: Dropout probability
            patch_size: Size of the input neighborhood
        """
        super(SimpleCNN, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        self.n_layers = n_layers
        
        # Per-variable feature extractors
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(filters),
            )
            for _ in range(num_vars)
        ])
        
        # Build shared conv block with constant filters
        shared_convs = []
        in_channels = filters * num_vars
        for i in range(n_layers):
            out_channels = filters  # Constant filters for all layers
            shared_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            shared_convs.append(nn.ReLU())
            shared_convs.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
        
        self.shared_conv = nn.Sequential(*shared_convs)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, 1024),  # Updated to match constant filters
            nn.ReLU(),
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split input by variables
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        
        # Extract features
        features = [extractor(var_input) for extractor, var_input in zip(self.feature_extractors, var_inputs)]
        
        # Concatenate and pass through shared convs
        x = torch.cat(features, dim=1)
        x = self.shared_conv(x)
        x = self.pool(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

# UNet architecture for hazard susceptibility modeling
class UNet(nn.Module):
    def __init__(self, logger, device, num_vars, filters=64, n_layers=4, activation=nn.ReLU(), 
                 dropout=True, drop_value=0.3, kernel_size=3, pool_size=2, patch_size=5):
        """
        UNet architecture for hazard susceptibility modeling.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/channels
            filters: Base number of filters (will be doubled in each layer)
            n_layers: Number of downsampling/upsampling layers
            activation: Activation function
            dropout: Whether to use dropout
            drop_value: Dropout probability
            kernel_size: Kernel size for convolutions
            pool_size: Pooling size for downsampling
            patch_size: Size of the input neighborhood
        """
        super(UNet, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        self.filters = filters
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.drop_value = drop_value
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.patch_size = patch_size
        
        self.logger.info(f"Initializing UNet with {num_vars} input variables and {n_layers} layers")
        
        # Input layer: Process each variable separately
        self.var_blocks = nn.ModuleList()
        for _ in range(self.num_vars):
            # Initial preprocessing for each variable
            self.var_blocks.append(nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm2d(filters),
                activation,
                nn.Conv2d(filters, filters, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm2d(filters),
                activation
            ))
        
        # Feature fusion layer
        self.fusion = nn.Conv2d(filters * num_vars, filters, kernel_size=1)
        
        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()
        current_filters = filters
        
        for i in range(n_layers):
            next_filters = current_filters * 2
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(current_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation,
                nn.Conv2d(next_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation
            ))
            self.pool_blocks.append(nn.MaxPool2d(pool_size))
            current_filters = next_filters
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_filters, current_filters * 2, kernel_size, padding='same'),
            nn.BatchNorm2d(current_filters * 2),
            activation,
            nn.Conv2d(current_filters * 2, current_filters * 2, kernel_size, padding='same'),
            nn.BatchNorm2d(current_filters * 2),
            activation,
            nn.Dropout2d(drop_value) if dropout else nn.Identity()
        )
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        current_filters = current_filters * 2
        
        for i in range(n_layers):
            next_filters = current_filters // 2
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(current_filters, next_filters, kernel_size=pool_size, 
                                stride=pool_size, padding=0),
                nn.BatchNorm2d(next_filters),
                activation
            ))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(current_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation,
                nn.Conv2d(next_filters, next_filters, kernel_size, padding='same'),
                nn.BatchNorm2d(next_filters),
                activation
            ))
            current_filters = next_filters
        
        # Classification head
        # Calculate the output size based on input neighborhood and operations
        patch_size = patch_size
        final_size = patch_size // (pool_size ** n_layers) if patch_size % (pool_size ** n_layers) == 0 else 1
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(filters, 256),
            nn.BatchNorm1d(256),
            activation,
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            activation,
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """Forward pass of the UNet model."""
        # Split input into separate variable channels
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        
        # Process each variable through its own block
        var_features = []
        for i, (block, inp) in enumerate(zip(self.var_blocks, var_inputs)):
            var_features.append(block(inp))
        
        # Concatenate and fuse features from all variables
        x = torch.cat(var_features, dim=1)
        x = self.fusion(x)
        
        # Store encoder outputs for skip connections
        enc_features = []
        
        # Encoder path
        for enc_block, pool_block in zip(self.enc_blocks, self.pool_blocks):
            # Save output before pooling for skip connection
            enc_features.append(x)
            # Apply convolution block then pooling
            x = enc_block(x)
            x = pool_block(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (up_block, dec_block) in enumerate(zip(self.up_blocks, self.dec_blocks)):
            # Upsample
            x = up_block(x)
            
            # Get corresponding encoder feature map
            skip_feature = enc_features[-(i+1)]
            
            # Handle size mismatch (if any)
            if x.shape != skip_feature.shape:
                # Center crop or pad to match
                diff_h = skip_feature.size(2) - x.size(2)
                diff_w = skip_feature.size(3) - x.size(3)
                
                if diff_h > 0 and diff_w > 0:
                    skip_feature = skip_feature[:, :, diff_h//2:-(diff_h//2), diff_w//2:-(diff_w//2)]
                elif diff_h < 0 and diff_w < 0:
                    padding = [-diff_h//2, -diff_h-(-diff_h//2), -diff_w//2, -diff_w-(-diff_w//2)]
                    skip_feature = F.pad(skip_feature, padding)
            
            # Concatenate for skip connection
            x = torch.cat([x, skip_feature], dim=1)
            
            # Apply convolution block
            x = dec_block(x)
        
        # Final classification
        outputs = self.classification_head(x)
        return torch.sigmoid(outputs)
    
# model from Japan paper converted to pytorch
class SpatialAttentionLayer(nn.Module):
    def __init__(self, channels, device=None):
        super(SpatialAttentionLayer, self).__init__()
        self.device = device
        
        self.conv1x1_theta = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1x1_phi = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1x1_g = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Initialize weights
        init.kaiming_normal_(self.conv1x1_theta.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv1x1_phi.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.conv1x1_g.weight)

        if self.device is not None:
            self.conv1x1_theta = self.conv1x1_theta.to(self.device)
            self.conv1x1_phi = self.conv1x1_phi.to(self.device)
            self.conv1x1_g = self.conv1x1_g.to(self.device)

    def forward(self, x):
        theta = F.relu(self.conv1x1_theta(x))
        phi = F.relu(self.conv1x1_phi(x))
        g = torch.sigmoid(self.conv1x1_g(x))

        theta_phi = theta * phi
        attention = theta_phi * g
        attended_x = x + attention
        
        return attended_x
    
class SpatialAttentionCNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters=16, n_layers=2, dropout=False, 
                 drop_value=0.2, patch_size=5):
        """
        Simple CNN with constant filters, variable number of conv layers, and spatial attention.
        """
        super(SpatialAttentionCNN, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        self.n_layers = n_layers
        
        # Per-variable feature extractors
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(filters),

            )
            for _ in range(num_vars)
        ])
        
        # Shared conv block with constant filters
        shared_convs = []
        in_channels = filters * num_vars
        for i in range(n_layers):
            out_channels = filters
            shared_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            shared_convs.append(nn.ReLU())
            shared_convs.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
        
        self.shared_conv = nn.Sequential(*shared_convs)
        
        # Add Spatial Attention Layer
        self.spatial_attention = SpatialAttentionLayer(channels=filters, device=self.device)        
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, 1024),
            nn.ReLU(),
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        features = [extractor(var_input) for extractor, var_input in zip(self.feature_extractors, var_inputs)]
        
        x = torch.cat(features, dim=1)
        x = self.shared_conv(x)
        x = self.spatial_attention(x)  # Apply spatial attention
        x = self.pool(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

# Full CNN architecture for hazard susceptibility modeling
class FullCNN(nn.Module):
    def __init__(self, logger, num_vars, filters=64, kernel_size=3, pool_size=2, 
                 n_layers=5, device=None, activation=nn.ReLU(), dropout=True, drop_value=0.5, name_model="FullCNN_Model", patch_size=5):
        super(FullCNN, self).__init__()
        
        self.device = device
        self.logger = logger
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.drop_value = drop_value
        self.name_model = name_model
        self.patch_size = patch_size
        self.num_vars = num_vars
        # self.logger.info(f"Initializing {self.name_model} with {num_vars} input variables")
        
        # Create modules for each variable input (branches)
        self._build_conv_branches()
        
        # Global average pooling to handle variable spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Calculate the number of output features
        if self.n_layers == 1:
            output_filters = self.filters
        else:
            output_filters = self.filters * 2

        total_features = output_filters * num_vars
        
        # self.logger.info(f"Calculated output features: {total_features} (filters={filters}, num_vars={num_vars})")
        
        self.dense_layers = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            self.activation,
            nn.Dropout(self.drop_value) if self.dropout else nn.Identity(),
            nn.Linear(1024, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_conv_branches(self):
        """Build the convolutional branches during initialization"""
        self.conv_branches = nn.ModuleList()
        for i in range(self.num_vars):
            layers = []
            # First conv layer with spatial attention
            layers.append(nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, padding='same'))
            layers.append(self.activation)
            
            # Add spatial attention layer
            spatial_attn = SpatialAttentionLayer(device=self.device, channels=self.num_vars)
            layers.append(spatial_attn)
            
            # Add pooling layer
            layers.append(nn.MaxPool2d(kernel_size=self.pool_size, padding=1))
            
            # Additional convolutional layers
            for j in range(self.n_layers - 1):
                in_filters = self.filters if j == 0 else self.filters * 2
                layers.append(nn.Conv2d(in_filters, self.filters * 2, kernel_size=self.kernel_size, padding='same'))
                layers.append(self.activation)
                
                if j == 1 or j == 3 or j == self.n_layers - 2:
                    layers.append(nn.MaxPool2d(kernel_size=self.pool_size, padding=1))
            
            self.conv_branches.append(nn.Sequential(*layers))
            
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        # Xavier/Glorot initialization for the final layer
        init.xavier_uniform_(self.dense_layers[-1].weight)
     
    def forward(self, x):
        # Process each variable through its branch
        var_inputs = [x[:, i:i+1, :, :] for i in range(self.num_vars)]
        
        features = []
        for i, branch in enumerate(self.conv_branches):
            if i < len(var_inputs):
                feat = branch(var_inputs[i])
                features.append(feat)
        
        # Concatenate features from all branches
        if len(features) > 1:
            merged = torch.cat(features, dim=1)
        else:
            merged = features[0]
        
        # Global average pooling
        pooled = self.global_avg_pool(merged).view(merged.size(0), -1)
        
        # Final dense layers
        x = self.dense_layers(pooled)
        
        # Apply sigmoid for final activation
        output = torch.sigmoid(x)
        
        return output
