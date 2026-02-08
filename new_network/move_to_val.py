#!/usr/bin/env python3
"""
Move 10% of training data to validation set, keeping image-mask pairs together.
"""

import random
from pathlib import Path
import shutil

# Paths
train_images = Path("/users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/new_network/data_25D/prepared_train/images")
train_masks = Path("/users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/new_network/data_25D/prepared_train/masks")
val_images = Path("/users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/new_network/data_25D/prepared_val/images")
val_masks = Path("/users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/new_network/data_25D/prepared_val/masks")

# Create validation directories if they don't exist
val_images.mkdir(parents=True, exist_ok=True)
val_masks.mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = sorted(list(train_images.glob("*.tif")))
print(f"Found {len(image_files)} images in training set")

# Calculate 10%
n_to_move = int(len(image_files) * 0.1)
print(f"Moving {n_to_move} paired images to validation set ({n_to_move/len(image_files)*100:.1f}%)")

# Randomly select 10%
random.seed(42)  # For reproducibility
selected = random.sample(image_files, n_to_move)

# Move paired files
moved = 0
errors = 0

for img_file in selected:
    # Get corresponding mask file with _masks suffix before extension
    # e.g., image.tif -> image_masks.tif
    img_stem = img_file.stem
    mask_filename = f"{img_stem}_masks{img_file.suffix}"
    mask_file = train_masks / mask_filename
    
    # Check if mask exists
    if not mask_file.exists():
        print(f"WARNING: Mask not found for {img_file.name}")
        errors += 1
        continue
    
    # Move both files
    shutil.move(str(img_file), str(val_images / img_file.name))
    shutil.move(str(mask_file), str(val_masks / mask_file.name))
    moved += 1
    
    if moved % 500 == 0:
        print(f"  Moved {moved}/{n_to_move} pairs...")

print(f"\n✓ Successfully moved {moved} paired files")
if errors > 0:
    print(f"✗ Failed to move {errors} files (missing masks)")

# Verify final counts
train_count = len(list(train_images.glob("*.tif")))
val_count = len(list(val_images.glob("*.tif")))
print(f"\nFinal counts:")
print(f"  Training: {train_count} images")
print(f"  Validation: {val_count} images")
print(f"  Split: {train_count/(train_count+val_count)*100:.1f}% train / {val_count/(train_count+val_count)*100:.1f}% val")
