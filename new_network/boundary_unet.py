"""
Boundary Detection U-Net

A standalone U-Net architecture for cell boundary detection.
Inspired by PlantSeg's approach to boundary segmentation but implemented independently.

Architecture:
- Encoder: 4 downsampling blocks with skip connections
- Decoder: 4 upsampling blocks with skip connections from encoder
- Output: Single channel boundary probability map

Design principles:
- Independent encoder learns edge-specific features (sharp discontinuities)
- Skip connections preserve spatial detail for precise boundary localization
- Simple architecture focused on boundary detection task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block with BatchNorm and ReLU.
    
    Architecture: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block: ConvBlock followed by MaxPool2d.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample -> Concat with skip -> ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block = ConvBlock(in_channels + out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch between x and skip
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class BoundaryUNet(nn.Module):
    """
    Standalone U-Net for boundary detection.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        base_channels: Base number of channels (doubled at each downsampling)
        depth: Number of downsampling/upsampling blocks
    
    Architecture:
        Input (H, W) -> 
        Encoder [32 -> 64 -> 128 -> 256] ->
        Bottleneck [512] ->
        Decoder [256 -> 128 -> 64 -> 32] ->
        Output (H, W) boundary logits
    """
    def __init__(self, in_channels=1, base_channels=32, depth=4):
        super().__init__()
        self.depth = depth
        
        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.down_blocks.append(DownBlock(channels, channels * 2))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels, channels)
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(UpBlock(channels, channels // 2))
            channels //= 2
        
        # Final 1x1 convolution to boundary logits
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, H, W)
        
        Returns:
            boundary_logits: (N, 1, H, W) raw logits for boundary prediction
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections (reverse order)
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)
        
        # Final boundary logits
        boundary_logits = self.final_conv(x)
        
        return boundary_logits
    
    def predict(self, x, threshold=0.5):
        """
        Predict boundary map with sigmoid activation.
        
        Args:
            x: Input tensor (N, C, H, W)
            threshold: Threshold for binary boundary map
        
        Returns:
            boundary_prob: (N, 1, H, W) boundary probability [0, 1]
            boundary_binary: (N, 1, H, W) binary boundary map {0, 1}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            binary = (probs > threshold).float()
        return probs, binary


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the network
    print("Testing BoundaryUNet...")
    
    # Create model
    model = BoundaryUNet(in_channels=1, base_channels=32, depth=4)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test prediction
    probs, binary = model.predict(x, threshold=0.5)
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Binary values: {binary.unique()}")
    
    print("\n✓ BoundaryUNet test passed!")
