"""
Training script for 3D Centre U-Net.

Train on volumetric patches with MSE loss for Gaussian center targets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from centre_unet_3d import CentreUNet3D
from dataset_3d import CentreDataset3D


def get_dataloaders(train_dir, val_dir, batch_size=2, patch_size=(256, 128, 128),
                   center_sigma=5.0, patches_per_volume=10, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        patch_size: (D, H, W) patch size
        center_sigma: Gaussian sigma for center targets
        patches_per_volume: Patches per volume per epoch
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = CentreDataset3D(
        data_dir=train_dir,
        patch_size=patch_size,
        augment=True,
        center_sigma=center_sigma,
        patches_per_volume=patches_per_volume
    )
    
    val_dataset = CentreDataset3D(
        data_dir=val_dir,
        patch_size=patch_size,
        augment=False,
        center_sigma=center_sigma,
        patches_per_volume=max(1, patches_per_volume // 2)  # Fewer patches for val (min 1)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, gt_centers in pbar:
        images = images.to(device)
        gt_centers = gt_centers.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            try:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    loss = criterion(probs, gt_centers)
            except:
                # Fallback for older PyTorch
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    loss = criterion(probs, gt_centers)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            probs = torch.sigmoid(logits)
            loss = criterion(probs, gt_centers)
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, gt_centers in pbar:
            images = images.to(device)
            gt_centers = gt_centers.to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            loss = criterion(probs, gt_centers)
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Handle empty dataloader
    n_batches = len(dataloader)
    if n_batches == 0:
        return 0.0
    
    avg_loss = total_loss / n_batches
    
    return avg_loss


def visualize_predictions(model, dataloader, device, output_dir, epoch):
    """Visualize Z-slice montage of predictions."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Get one batch
        images, gt_centers = next(iter(dataloader))
        images = images.to(device)
        
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        images = images.cpu().numpy()
        gt_centers = gt_centers.cpu().numpy()
        
        idx = 0  # First sample in batch
        D, H, W = images.shape[2:]
        
        # Z-Slice Montage
        n_slices = 8
        z_indices = np.linspace(D//8, 7*D//8, n_slices, dtype=int)
        
        fig, axes = plt.subplots(3, n_slices, figsize=(20, 8))
        fig.suptitle(f'Epoch {epoch} - Z-Slice Montage (Center Detection)', fontsize=14)
        
        for i, z in enumerate(z_indices):
            # Input
            axes[0, i].imshow(images[idx, 0, z], cmap='gray')
            axes[0, i].set_title(f'Z={z}', fontsize=8)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Input', fontsize=10)
            
            # GT Center
            axes[1, i].imshow(gt_centers[idx, 0, z], cmap='hot', vmin=0, vmax=1)
            if i == 0:
                axes[1, i].set_ylabel('GT Center', fontsize=10)
            axes[1, i].axis('off')
            
            # Pred Center
            axes[2, i].imshow(probs[idx, 0, z], cmap='hot', vmin=0, vmax=1)
            if i == 0:
                axes[2, i].set_ylabel('Pred Center', fontsize=10)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved predictions to {output_dir}/predictions_epoch_{epoch}.png")


def plot_loss_curves(train_losses, val_losses, output_dir):
    """Plot training and validation loss curves (overwrites each epoch)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train(
    train_dir,
    val_dir,
    output_dir,
    epochs=100,
    batch_size=2,
    learning_rate=1e-4,
    patch_size=(256, 128, 128),
    center_sigma=5.0,
    patches_per_volume=10,
    base_channels=32,
    depth=4,
    dropout_p=0.0,
    num_workers=4,
    save_every=10,
    mixed_precision=True
):
    """
    Train 3D Centre U-Net.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        output_dir: Output directory for checkpoints and visualizations
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patch_size: (D, H, W) patch size
        center_sigma: Gaussian sigma for center targets
        patches_per_volume: Patches per volume per epoch
        base_channels: Base channels for U-Net
        depth: Depth of U-Net
        dropout_p: Dropout probability
        num_workers: Number of dataloader workers
        save_every: Save checkpoint every N epochs
        mixed_precision: Use mixed precision training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    viz_dir = output_dir / 'visualizations'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        center_sigma=center_sigma,
        patches_per_volume=patches_per_volume,
        num_workers=num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = CentreUNet3D(
        in_channels=1,
        base_channels=base_channels,
        depth=depth,
        dropout_p=dropout_p
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer - use MSE for Gaussian targets
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    print("Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
    
    # Mixed precision scaler
    if mixed_precision:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            scaler = torch.cuda.amp.GradScaler()  # Fallback for older PyTorch
    else:
        scaler = None
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Plot loss curves (overwritten each epoch)
        plot_loss_curves(train_losses, val_losses, output_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
            
            # Visualize
            visualize_predictions(model, val_loader, device, viz_dir, epoch)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D Centre U-Net')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Validation data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[256, 128, 128],
                        help='Patch size (D H W) - use full Z for depth-dependent imaging')
    parser.add_argument('--patches_per_volume', type=int, default=10,
                        help='Patches per volume per epoch')
    
    # Model arguments
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base channels')
    parser.add_argument('--depth', type=int, default=4,
                        help='U-Net depth')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='Dropout probability')
    
    # Center-specific arguments
    parser.add_argument('--center_sigma', type=float, default=5.0,
                        help='Gaussian sigma for center targets (in pixels)')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patch_size=tuple(args.patch_size),
        center_sigma=args.center_sigma,
        patches_per_volume=args.patches_per_volume,
        base_channels=args.base_channels,
        depth=args.depth,
        dropout_p=args.dropout_p,
        num_workers=args.num_workers,
        save_every=args.save_every,
        mixed_precision=not args.no_mixed_precision
    )


if __name__ == '__main__':
    main()
