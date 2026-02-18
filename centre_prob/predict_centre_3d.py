"""
Inference script for 3D Centre U-Net.

Predict cell center probabilities on full volumes.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import tifffile
from tqdm import tqdm

from centre_unet_3d import CentreUNet3D


class CentrePredictor3D:
    """Predictor for 3D center detection."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        # Check if CUDA is available and fall back to CPU if not
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Detect model architecture from checkpoint
        possible_keys = [
            'init_conv.conv1.weight',
            'module.init_conv.conv1.weight',
            '_orig_mod.init_conv.conv1.weight'
        ]
        
        first_conv_weight = None
        for key in possible_keys:
            if key in state_dict:
                first_conv_weight = state_dict[key]
                break
        
        if first_conv_weight is None:
            raise KeyError(f"Could not find first conv layer weight")
        
        in_channels = first_conv_weight.shape[1]
        base_channels = first_conv_weight.shape[0]
        
        # Detect depth from number of encoder layers
        max_encoder_idx = -1
        for key in state_dict.keys():
            if key.startswith('encoders.'):
                idx = int(key.split('.')[1])
                max_encoder_idx = max(max_encoder_idx, idx)
        depth = max_encoder_idx + 1
        
        print(f"Detected architecture: base_channels={base_channels}, depth={depth}")
        
        # Create model
        self.model = CentreUNet3D(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth
        )
        
        # Handle state dict prefixes
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        elif any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    def _create_weight_map(self, tile_shape, overlap, d_start, d_end, h_start, h_end, w_start, w_end, volume_shape):
        """Create a weight map for blending tiles in overlap regions using linear tapering."""
        d_tile, h_tile, w_tile = tile_shape
        vol_d, vol_h, vol_w = volume_shape
        
        weight_map = np.ones(tile_shape, dtype=np.float32)
        
        # Create linear ramps for overlap regions
        ramp = np.linspace(0, 1, overlap)
        ramp_inv = np.linspace(1, 0, overlap)
        
        # D dimension weights
        if d_start > 0:  # Not at volume start
            for i in range(min(overlap, d_tile)):
                weight_map[i, :, :] *= ramp[i]
        
        if d_end < vol_d:  # Not at volume end
            for i in range(min(overlap, d_tile)):
                idx = d_tile - overlap + i
                if idx >= 0 and idx < d_tile:
                    weight_map[idx, :, :] *= ramp_inv[i]
        
        # H dimension weights
        if h_start > 0:
            for i in range(min(overlap, h_tile)):
                weight_map[:, i, :] *= ramp[i]
        
        if h_end < vol_h:
            for i in range(min(overlap, h_tile)):
                idx = h_tile - overlap + i
                if idx >= 0 and idx < h_tile:
                    weight_map[:, idx, :] *= ramp_inv[i]
        
        # W dimension weights
        if w_start > 0:
            for i in range(min(overlap, w_tile)):
                weight_map[:, :, i] *= ramp[i]
        
        if w_end < vol_w:
            for i in range(min(overlap, w_tile)):
                idx = w_tile - overlap + i
                if idx >= 0 and idx < w_tile:
                    weight_map[:, :, idx] *= ramp_inv[i]
        
        return weight_map
    
    def predict_volume(self, volume, patch_size=(256, 128, 128), overlap=16, normalize=True):
        """
        Predict on full volume using sliding window.
        
        Args:
            volume: (D, H, W) numpy array
            patch_size: (D, H, W) patch size for sliding window
            overlap: Overlap between patches
            normalize: Normalize volume using 99th percentile
        
        Returns:
            center_prob: (D, H, W) center probability
        """
        # Squeeze singleton dimensions
        volume = np.squeeze(volume)
        
        # Handle multi-channel images: take first channel
        if volume.ndim == 4:
            print(f"Warning: 4D volume with shape {volume.shape} detected. Taking first channel.")
            volume = volume[0, ...]
        
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        
        D, H, W = volume.shape
        pd, ph, pw = patch_size
        
        # Normalize volume
        volume = volume.astype(np.float32)
        if normalize:
            print("Normalizing volume using 99th percentile...")
            percentile_99 = np.percentile(volume, 99)
            if percentile_99 > 0:
                volume = volume / percentile_99
                volume = np.clip(volume, 0, 1)
            print(f"  Normalized: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}")
        
        # Initialize outputs and weight map
        center_prob = np.zeros((D, H, W), dtype=np.float32)
        weight_map = np.zeros((D, H, W), dtype=np.float32)
        
        # Calculate stride
        stride_d = pd - overlap
        stride_h = ph - overlap
        stride_w = pw - overlap
        
        # Generate patch coordinates
        patch_coords = []
        for d_start in range(0, D, stride_d):
            for h_start in range(0, H, stride_h):
                for w_start in range(0, W, stride_w):
                    d_end = min(d_start + pd, D)
                    h_end = min(h_start + ph, H)
                    w_end = min(w_start + pw, W)
                    
                    # Adjust start if patch would be too small
                    if d_end - d_start < pd:
                        d_start = max(0, d_end - pd)
                    if h_end - h_start < ph:
                        h_start = max(0, h_end - ph)
                    if w_end - w_start < pw:
                        w_start = max(0, w_end - pw)
                    
                    patch_coords.append((d_start, h_start, w_start))
        
        print(f"Processing {len(patch_coords)} patches...")
        
        # Process patches
        with torch.no_grad():
            for d_start, h_start, w_start in tqdm(patch_coords):
                # Extract patch
                patch = volume[
                    d_start:d_start+pd,
                    h_start:h_start+ph,
                    w_start:w_start+pw
                ]
                
                # Pad if needed
                if patch.shape != patch_size:
                    padded = np.zeros(patch_size, dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                
                # Convert to tensor
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(self.device)
                
                # Predict
                logits = self.model(patch_tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                
                # Accumulate predictions
                d_end = min(d_start + pd, D)
                h_end = min(h_start + ph, H)
                w_end = min(w_start + pw, W)
                
                actual_d = d_end - d_start
                actual_h = h_end - h_start
                actual_w = w_end - w_start
                
                # Create weight map for this patch
                patch_blend_weight = self._create_weight_map(
                    tile_shape=(actual_d, actual_h, actual_w),
                    overlap=overlap,
                    d_start=d_start, d_end=d_end,
                    h_start=h_start, h_end=h_end,
                    w_start=w_start, w_end=w_end,
                    volume_shape=(D, H, W)
                )
                
                center_prob[d_start:d_end, h_start:h_end, w_start:w_end] += probs[0, :actual_d, :actual_h, :actual_w] * patch_blend_weight
                weight_map[d_start:d_end, h_start:h_end, w_start:w_end] += patch_blend_weight
        
        # Average overlapping predictions
        weight_map[weight_map == 0] = 1.0  # Avoid division by zero
        center_prob /= weight_map
        
        print("Prediction complete!")
        
        return center_prob


def main():
    parser = argparse.ArgumentParser(description='Predict with 3D Centre U-Net')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input volume (.tif)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[256, 128, 128],
                        help='Patch size for sliding window (D H W)')
    parser.add_argument('--overlap', type=int, default=16,
                        help='Overlap between patches')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable volume normalization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create predictor
    print("Loading model...")
    predictor = CentrePredictor3D(args.checkpoint, device=args.device)
    
    # Load volume
    print(f"Loading volume from {args.input}...")
    volume = tifffile.imread(args.input)
    print(f"Volume shape: {volume.shape}")
    
    # Predict
    center_prob = predictor.predict_volume(
        volume,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        normalize=not args.no_normalize
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    base_name = input_path.stem
    
    center_path = output_dir / f'{base_name}_center.tif'
    
    print(f"Saving predictions to {output_dir}...")
    tifffile.imwrite(center_path, center_prob.astype(np.float32))
    
    print("Done!")
    print(f"  Center: {center_path}")


if __name__ == '__main__':
    main()
