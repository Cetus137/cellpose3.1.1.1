"""
3D U-Net for cell center detection.

Architecture:
- Encoder-decoder with skip connections
- 3D convolutions throughout
- Single output: center probability (Gaussian peaks at cell centroids)
- Designed for isotropic volumetric data
"""

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """3D convolution block with batch norm, ReLU, and optional dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else None
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block: Conv3DBlock + MaxPool3d."""
    
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv_block = Conv3DBlock(in_channels, out_channels, dropout_p=dropout_p)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x):
        x = self.conv_block(x)
        pooled = self.pool(x)
        return x, pooled  # Return both for skip connection


class DecoderBlock(nn.Module):
    """Decoder block: Upsample + Concatenate + Conv3DBlock."""
    
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concat: in_channels//2 + skip_channels
        self.conv_block = Conv3DBlock(in_channels // 2 + skip_channels, out_channels, dropout_p=dropout_p)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class CentreUNet3D(nn.Module):
    """
    3D U-Net for cell center detection.
    
    Outputs:
        - Single channel: Center probability (Gaussian peaks at cell centroids)
    
    Args:
        in_channels: Number of input channels (typically 1 for single-modality)
        base_channels: Number of channels in first encoder block (default: 32)
        depth: Number of encoder/decoder levels (default: 4)
        dropout_p: Dropout probability (default: 0.0, disabled by default)
    """
    
    def __init__(self, in_channels=1, base_channels=32, depth=4, dropout_p=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_p = dropout_p
        
        # Initial convolution
        self.init_conv = Conv3DBlock(in_channels, base_channels, dropout_p=dropout_p)
        
        # Encoder
        self.encoders = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock(channels, channels * 2, dropout_p=dropout_p))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = Conv3DBlock(channels, channels * 2, dropout_p=dropout_p)
        
        # Decoder
        self.decoders = nn.ModuleList()
        channels *= 2  # Bottleneck doubled channels
        for i in range(depth):
            skip_ch = base_channels * (2 ** (depth - i))
            out_ch = channels // 2 if i < depth - 1 else base_channels
            self.decoders.append(DecoderBlock(channels, skip_ch, out_ch, dropout_p=dropout_p))
            channels //= 2
        
        # Final convolution for single output (center probability)
        self.final_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, C, D, H, W) input volume
        
        Returns:
            logits: (B, 1, D, H, W) - center probability logits
        """
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections (reverse order)
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)
        
        # Final output
        logits = self.final_conv(x)
        
        return logits


def test_model():
    """Test model with dummy input."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CentreUNet3D(
        in_channels=1,
        base_channels=16,
        depth=3,
        dropout_p=0.0
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 1
    x = torch.randn(batch_size, 1, 64, 128, 128).to(device)
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    print("Model test passed!")


if __name__ == '__main__':
    test_model()
