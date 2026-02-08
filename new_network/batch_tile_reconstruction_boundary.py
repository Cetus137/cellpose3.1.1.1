"""
Batch reconstruction of boundary probability maps from tiled predictions.
Adapted from batch_tile_reconstruction.py for boundary prediction outputs.
"""

import numpy as np
import tifffile as tiff
import os
from natsort import natsorted
import re
import argparse


def create_weight_map(tile_shape, overlap_xy, z_start, z_end, y_start, y_end, x_start, x_end, image_shape):
    """
    Create a weight map for blending tiles in overlap regions.
    Uses linear tapering in overlap regions.
    
    Parameters:
    -----------
    tile_shape : tuple
        Actual shape of the tile data (z, y, x)
    overlap_xy : int
        Overlap in pixels for XY dimensions
    z_start, z_end, y_start, y_end, x_start, x_end : int
        Coordinates of the tile in the full image
    image_shape : tuple
        Shape of the full image (z, y, x)
        
    Returns:
    --------
    weight_map : numpy.ndarray
        Weight map with shape matching tile_shape
    """
    z_tile, y_tile, x_tile = tile_shape
    img_z, img_y, img_x = image_shape
    
    weight_map = np.ones(tile_shape, dtype=np.float32)
    
    # Create linear ramps for overlap regions
    half_overlap = overlap_xy // 2
    
    # Y dimension weights
    # Left edge (if not at image edge)
    if y_start > 0:
        ramp = np.linspace(0, 1, overlap_xy)
        for i in range(min(overlap_xy, y_tile)):
            weight_map[:, i, :] *= ramp[i]
    
    # Right edge (if not at image edge)
    if y_end < img_y:
        ramp = np.linspace(1, 0, overlap_xy)
        for i in range(min(overlap_xy, y_tile)):
            idx = y_tile - overlap_xy + i
            if idx >= 0 and idx < y_tile:
                weight_map[:, idx, :] *= ramp[i]
    
    # X dimension weights
    # Left edge (if not at image edge)
    if x_start > 0:
        ramp = np.linspace(0, 1, overlap_xy)
        for i in range(min(overlap_xy, x_tile)):
            weight_map[:, :, i] *= ramp[i]
    
    # Right edge (if not at image edge)
    if x_end < img_x:
        ramp = np.linspace(1, 0, overlap_xy)
        for i in range(min(overlap_xy, x_tile)):
            idx = x_tile - overlap_xy + i
            if idx >= 0 and idx < x_tile:
                weight_map[:, :, idx] *= ramp[i]
    
    return weight_map


def reconstruct_from_tiles_boundary(tiles, image_shape, overlap_xy=32):
    """
    Reconstruct boundary probability map from segmented tiles by averaging overlaps.
    
    Parameters:
    -----------
    tiles : list of dict
        List of tile dictionaries containing boundary prediction results
    image_shape : tuple
        Shape of the original image (z, y, x)
    overlap_xy : int
        Overlap in pixels for XY dimensions
        
    Returns:
    --------
    boundary_prob : numpy.ndarray
        Reconstructed boundary probability with shape (z, y, x)
    """
    z_size, y_size, x_size = image_shape
    
    # Initialize output arrays and weight arrays for averaging
    boundary_prob = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    boundary_weights = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    
    tile_id = 1
    for tile_info in tiles:
        print(f'Processing tile number {tile_id}')
        tile_id += 1

        z_start = tile_info['z_start']
        z_end = tile_info['z_end']
        y_start = tile_info['y_start']
        y_end = tile_info['y_end']
        x_start = tile_info['x_start']
        x_end = tile_info['x_end']
        
        # Get actual data size (not padded)
        actual_z, actual_y, actual_x = tile_info['original_shape']
        print(f'Original shape: {tile_info["original_shape"]}')

        tile_boundary = tile_info['boundary_prob'][:actual_z, :actual_y, :actual_x]
        
        # Create weight map for this tile (1.0 in center, tapering at edges in overlap regions)
        print('Creating weight map...')
        weight_map = create_weight_map(
            (actual_z, actual_y, actual_x),
            overlap_xy,
            z_start, z_end, y_start, y_end, x_start, x_end,
            image_shape
        )
        print(f'Weight map shape: {weight_map.shape}')
        
        # Accumulate weighted values
        boundary_prob[z_start:z_end, y_start:y_end, x_start:x_end] += tile_boundary * weight_map
        
        # Accumulate weights
        boundary_weights[z_start:z_end, y_start:y_end, x_start:x_end] += weight_map
    
    # Normalize by weights (avoid division by zero)
    boundary_weights = np.maximum(boundary_weights, 1e-8)
    boundary_prob = boundary_prob / boundary_weights
    
    return boundary_prob


def timepoint_reconstruct_boundary(tile_dir, timepoint, overlap_xy=32):
    """
    Reconstruct boundary probability map for a specific timepoint from tiled predictions.
    
    Parameters:
    -----------
    tile_dir : str
        Directory containing the tile files
    timepoint : int
        Timepoint number to reconstruct
    overlap_xy : int
        Overlap in pixels used during tiling (default: 32)
        
    Returns:
    --------
    boundary_prob : numpy.ndarray
        Reconstructed boundary probability with shape (z, y, x)
    image_shape : tuple
        Shape of the reconstructed image (z, y, x)
    """
    # Find all tif files in the directory
    tif_files = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
    tif_files = natsorted(tif_files)
    print(f'Timepoint is {timepoint}')
    
    # Find the specific timepoint boundary files
    boundary_files = [f for f in tif_files if f"timepoint_{timepoint:04d}" in f and "boundary_prob" in f]
    
    if len(boundary_files) == 0:
        raise ValueError(f"No boundary probability files found for timepoint {timepoint} in {tile_dir}")
    
    tiles = []
    max_z, max_y, max_x = 0, 0, 0
    
    # Build tile info and determine image shape
    for boundary_file in boundary_files:
        # Extract tile metadata from filename using regex
        # Example: ..._z0-256_y0-256_x224-480_...
        m_z = re.search(r'_z(\d+)-(\d+)', boundary_file)
        m_y = re.search(r'_y(\d+)-(\d+)', boundary_file)
        m_x = re.search(r'_x(\d+)-(\d+)', boundary_file)
        
        if not (m_z and m_y and m_x):
            print(f"Warning: Could not parse tile coordinates from {boundary_file}, skipping.")
            continue
        
        z_start, z_end = int(m_z.group(1)), int(m_z.group(2))
        y_start, y_end = int(m_y.group(1)), int(m_y.group(2))
        x_start, x_end = int(m_x.group(1)), int(m_x.group(2))

        print(f"Found tile: z({z_start}-{z_end}), y({y_start}-{y_end}), x({x_start}-{x_end})")
        
        # Track maximum extents to determine image shape
        max_z = max(max_z, z_end)
        max_y = max(max_y, y_end)
        max_x = max(max_x, x_end)

        # Load the actual data
        boundary_data = tiff.imread(os.path.join(tile_dir, boundary_file))
        
        # Normalize to 0-1 if needed (in case it's stored as uint8)
        if boundary_data.dtype == np.uint8:
            boundary_data = boundary_data.astype(np.float32) / 255.0

        tiles.append({
            'boundary_prob': boundary_data,
            'z_start': z_start,
            'z_end': z_end,
            'y_start': y_start,
            'y_end': y_end,
            'x_start': x_start,
            'x_end': x_end,
            'original_shape': (z_end - z_start, y_end - y_start, x_end - x_start)
        })
    
    image_shape = (max_z, max_y, max_x)
    print(f"Reconstructing timepoint {timepoint} with {len(tiles)} tiles, image shape: {image_shape}")
    
    # Reconstruct using weighted averaging
    boundary_prob = reconstruct_from_tiles_boundary(tiles, image_shape, overlap_xy)
    
    return boundary_prob


def reconstruct_boundary_from_tiles(tile_dir, output_dir, timepoint, overlap_xy=32, 
                                    save_as_uint8=True, verbose=True):
    """
    Reconstruct full boundary probability map from tiles and save to file.
    
    Parameters:
    -----------
    tile_dir : str
        Directory containing the tile files
    output_dir : str
        Directory to save reconstructed outputs
    timepoint : int
        Timepoint number to reconstruct
    overlap_xy : int
        Overlap in pixels used during tiling (default: 32)
    save_as_uint8 : bool
        If True, save as uint8 (0-255), otherwise save as float32 (0-1)
    verbose : bool
        Enable verbose output
    """
    boundary_prob = timepoint_reconstruct_boundary(tile_dir, timepoint, overlap_xy)
    
    if verbose:
        print(f"Reconstructed boundary_prob shape: {boundary_prob.shape}")
        print(f"Boundary probability range: [{boundary_prob.min():.3f}, {boundary_prob.max():.3f}]")
        print(f"Boundary probability mean: {boundary_prob.mean():.3f}")
    
    # Save boundary probability
    output_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_boundary_prob.tif')
    print(f"Saving boundary probability to {output_path}")
    
    if save_as_uint8:
        boundary_prob_uint8 = (boundary_prob * 255).astype(np.uint8)
        tiff.imwrite(output_path, boundary_prob_uint8)
    else:
        tiff.imwrite(output_path, boundary_prob.astype(np.float32))
    
    return boundary_prob


def main():
    """
    CLI for reconstructing full boundary probability maps from tiles.
    """
    parser = argparse.ArgumentParser(description='Reconstruct boundary probability maps from tiled outputs')
    parser.add_argument('--tile_dir', type=str, required=True,
                        help='Directory containing the tile files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save reconstructed outputs')
    parser.add_argument('--timepoint', type=int, required=True,
                        help='Timepoint number to reconstruct')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap in pixels used during tiling (default: 32)')
    parser.add_argument('--save_float', action='store_true',
                        help='Save as float32 (0-1) instead of uint8 (0-255)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    reconstruct_boundary_from_tiles(
        tile_dir=args.tile_dir,
        output_dir=args.output_dir,
        timepoint=args.timepoint,
        overlap_xy=args.overlap,
        save_as_uint8=not args.save_float,
        verbose=args.verbose
    )
    print("Reconstruction complete.")


if __name__ == "__main__":
    main()
