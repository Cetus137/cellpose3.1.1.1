"""
Training script for BoundaryUNet.

Trains a standalone boundary detection network using:
- BCE loss with pos_weight for class imbalance
- AdamW optimizer with cosine annealing
- Validation monitoring
- Model checkpointing
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from boundary_unet import BoundaryUNet, count_parameters
from dataset import get_dataloaders


class BoundaryTrainer:
    """Trainer class for BoundaryUNet."""
    
    def __init__(self, model, train_loader, val_loader=None, 
                 lr=1e-3, weight_decay=1e-4, device='cuda', 
                 save_dir='./checkpoints', save_every=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,  # 100 epochs
            eta_min=lr / 100
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_bce': [],
            'val_loss': [],
            'val_bce': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
    
    def compute_loss(self, logits, target):
        """
        Compute boundary detection loss.
        
        Uses BCE with pos_weight to handle class imbalance
        (typically ~10-15% boundary pixels vs 85-90% interior).
        
        Args:
            logits: (N, 1, H, W) predicted boundary logits
            target: (N, 1, H, W) ground truth boundaries {0, 1}
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with loss components
        """
        # Flatten for loss computation
        logits_flat = logits.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Compute pos_weight for class imbalance
        n_positive = target_flat.sum().item()
        n_total = target_flat.numel()
        
        if n_positive > 0:
            pos_weight = (n_total - n_positive) / n_positive
            pos_weight = min(pos_weight, 10.0)  # Cap at 10
        else:
            pos_weight = 1.0
        
        # BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_flat,
            target_flat,
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )
        
        # Compute metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred_binary = (probs > 0.5).float()
            
            # Accuracy
            correct = (pred_binary == target).sum().item()
            total = target.numel()
            accuracy = correct / total
            
            # Boundary pixel statistics
            boundary_recall = ((pred_binary * target).sum() / (target.sum() + 1e-7)).item()
        
        metrics = {
            'loss': bce_loss.item(),
            'bce': bce_loss.item(),
            'pos_weight': pos_weight,
            'accuracy': accuracy,
            'boundary_recall': boundary_recall,
            'pred_mean': probs.mean().item(),
            'gt_mean': target.mean().item()
        }
        
        loss_components = {
            'bce': bce_loss.item()
        }
        
        return bce_loss, metrics, loss_components
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            boundary_gt = batch['boundary_gt'].to(self.device)
            
            # Forward pass
            logits = self.model(images)
            
            # Compute loss
            loss, metrics, loss_components = self.compute_loss(logits, boundary_gt)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Record metrics
            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {metrics['accuracy']:.3f} "
                      f"Recall: {metrics['boundary_recall']:.3f} "
                      f"PredMean: {metrics['pred_mean']:.3f}")
        
        # Average metrics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) 
                       for k in epoch_metrics[0].keys()}
        
        return avg_loss, avg_metrics
    
    def validate(self, epoch):
        """Validate on validation set."""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                boundary_gt = batch['boundary_gt'].to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Compute loss
                loss, metrics, loss_components = self.compute_loss(logits, boundary_gt)
                
                val_losses.append(loss.item())
                val_metrics.append(metrics)
        
        # Average metrics
        avg_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean([m[k] for m in val_metrics]) 
                       for k in val_metrics[0].keys()}
        
        print(f"Validation - Loss: {avg_loss:.4f} "
              f"Acc: {avg_metrics['accuracy']:.3f} "
              f"Recall: {avg_metrics['boundary_recall']:.3f}")
        
        return avg_loss, avg_metrics
    
    def plot_loss_curves(self, epoch):
        """Plot loss curves and save to file (overwrites each epoch)."""
        if len(self.history['train_loss']) == 0:
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        if len(self.history['val_loss']) > 0:
            axes[0].plot(epochs, self.history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        if len(epochs) > 1:
            axes[0].set_xlim(1, max(epochs))
        
        # Loss components
        axes[1].plot(epochs, self.history['train_bce'], 'g-', linewidth=2, label='BCE Loss (Train)')
        if len(self.history['val_bce']) > 0:
            axes[1].plot(epochs, self.history['val_bce'], 'm--', linewidth=2, label='BCE Loss (Val)')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss Components', fontsize=12)
        axes[1].set_title('Loss Components', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        if len(epochs) > 1:
            axes[1].set_xlim(1, max(epochs))
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_predictions(self, epoch):
        """Save visualization of predictions on example images."""
        self.model.eval()
        
        # Use validation set if available, otherwise training set
        loader = self.val_loader if self.val_loader is not None else self.train_loader
        dataset_name = "Validation" if self.val_loader is not None else "Training"
        
        # Get random batch
        import random
        batch_idx = random.randint(0, len(loader) - 1)
        for i, batch in enumerate(loader):
            if i == batch_idx:
                break
        
        images = batch['image'].to(self.device)
        boundary_gt = batch['boundary_gt'].to(self.device)
        filenames = batch['filename']
        
        with torch.no_grad():
            logits = self.model(images)
            boundary_pred = torch.sigmoid(logits)
        
        # Convert to numpy and take first 3 images in batch
        n_show = min(3, images.shape[0])
        
        fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_show):
            img = images[i, 0].cpu().numpy()
            gt = boundary_gt[i, 0].cpu().numpy()
            pred = boundary_pred[i, 0].cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Raw image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Input: {filenames[i][:30]}', fontsize=10)
            axes[i, 0].axis('off')
            
            # Ground truth boundaries
            axes[i, 1].imshow(gt, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title(f'GT Boundaries ({gt.sum():.0f} px)', fontsize=10)
            axes[i, 1].axis('off')
            
            # Predicted boundary probability
            im = axes[i, 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Pred Prob (μ={pred.mean():.3f})', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Binary prediction
            axes[i, 3].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Pred Binary ({pred_binary.sum():.0f} px)', fontsize=10)
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Boundary Predictions ({dataset_name}) - Epoch {epoch}', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'predictions_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.model.train()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        # Save latest every save_every epochs
        if epoch % self.save_every == 0:
            latest_path = self.save_dir / 'latest.pth'
            torch.save(checkpoint, latest_path)
            print(f"✓ Saved latest checkpoint (epoch {epoch})")
        
        # Save best
        if is_best:
            best_path = self.save_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (epoch {epoch})")
        
        # Save numbered checkpoint every 50 epochs
        if epoch % 50 == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved checkpoint_epoch_{epoch}.pth")
    
    def train(self, num_epochs):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Validation batches: {len(self.val_loader)}")
        print(f"Device: {self.device}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_bce'].append(train_metrics.get('bce', train_loss))
            
            print(f"\nTrain - Loss: {train_loss:.4f} "
                  f"Acc: {train_metrics['accuracy']:.3f} "
                  f"Recall: {train_metrics['boundary_recall']:.3f}")
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_bce'].append(val_metrics.get('bce', val_loss))
                
                # Check if best model
                is_best = val_loss < self.history['best_val_loss']
                if is_best:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch'] = epoch
            else:
                is_best = False
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Save visualizations every epoch
            self.plot_loss_curves(epoch)
            self.visualize_predictions(epoch)
            
            # Save history
            with open(self.save_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        if self.val_loader:
            print(f"Best validation loss: {self.history['best_val_loss']:.4f} "
                  f"(epoch {self.history['best_epoch']})")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train BoundaryUNet')
    
    # Data
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Path to validation data directory')
    
    # Model
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels (1=grayscale, 3=RGB)')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels')
    parser.add_argument('--depth', type=int, default=4,
                        help='Network depth (number of down/up blocks)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation (default: False)')
    parser.add_argument('--samples_per_epoch', type=int, default=None,
                        help='Number of random samples per epoch (default: None = use all images)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print(f"\nLoading data from {args.train_dir}...")
    train_loader, val_loader = get_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        samples_per_epoch=args.samples_per_epoch
    )
    
    # Create model
    print("\nInitializing BoundaryUNet...")
    model = BoundaryUNet(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        depth=args.depth
    )
    
    # Create trainer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f'boundary_unet_{timestamp}'
    
    trainer = BoundaryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=save_dir,
        save_every=args.save_every
    )
    
    # Train
    trainer.train(args.num_epochs)


if __name__ == '__main__':
    main()
