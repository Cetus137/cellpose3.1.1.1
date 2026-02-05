"""
Training callbacks for Cellpose visualization during training
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.train import make_boundary_gt


class TrainingVisualizer:
    """Callback to visualize training progress"""
    
    def __init__(self, model, val_image, val_mask, output_dir, model_name,
                 channels=[0, 0], device=None, save_every=1):
        self.model = model
        self.val_image = val_image
        self.val_mask = val_mask
        self.output_dir = output_dir
        self.model_name = model_name
        self.channels = channels
        self.device = device
        self.save_every = save_every
        
        # Create output directories
        self.plot_dir = os.path.join(output_dir, 'training_plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Storage for losses
        self.train_losses = []
        self.test_losses = []
        # Storage for component losses
        self.train_flow_losses = []
        self.train_cellprob_losses = []
        self.train_boundary_losses = []
        self.test_flow_losses = []
        self.test_cellprob_losses = []
        self.test_boundary_losses = []
        
    def on_epoch_end(self, epoch, train_loss, test_loss, model_path, loss_dict=None):
        """Called at the end of each epoch"""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        
        # Store component losses if provided
        if loss_dict is not None:
            self.train_flow_losses.append(loss_dict['flow'])
            self.train_cellprob_losses.append(loss_dict['cellprob'])
            self.train_boundary_losses.append(loss_dict['boundary'])
            self.test_flow_losses.append(loss_dict['test_flow'])
            self.test_cellprob_losses.append(loss_dict['test_cellprob'])
            self.test_boundary_losses.append(loss_dict['test_boundary'])
        
        # Save loss plot every epoch
        self._plot_losses(epoch)
        
        # Save visualization every epoch (not just at save_every intervals)
        self._save_visualization(epoch, model_path)
    
    def _plot_losses(self, epoch):
        """Plot training and test losses in 4 panels"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        epochs = np.arange(len(self.train_losses))
        
        # Panel 1: Total Loss
        axes[0].plot(epochs, self.train_losses, label='Train Total', linewidth=2, marker='o', markersize=3, color='#1f77b4')
        if len(self.test_losses) > 0 and self.test_losses[-1] > 0:
            axes[0].plot(epochs, self.test_losses, label='Test Total', linewidth=2, marker='s', markersize=3, color='#ff7f0e')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Flow Loss
        if len(self.train_flow_losses) > 0:
            axes[1].plot(epochs, self.train_flow_losses, label='Train Flow', linewidth=2, marker='o', markersize=3, color='#2ca02c')
            if len(self.test_flow_losses) > 0:
                axes[1].plot(epochs, self.test_flow_losses, label='Test Flow', linewidth=2, marker='s', markersize=3, color='#98df8a')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Flow Loss', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Cell Probability Loss
        if len(self.train_cellprob_losses) > 0:
            axes[2].plot(epochs, self.train_cellprob_losses, label='Train CellProb', linewidth=2, marker='o', markersize=3, color='#d62728')
            if len(self.test_cellprob_losses) > 0:
                axes[2].plot(epochs, self.test_cellprob_losses, label='Test CellProb', linewidth=2, marker='s', markersize=3, color='#ff9896')
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Loss', fontsize=12)
            axes[2].set_title('Cell Probability Loss', fontsize=14, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
        
        # Panel 4: Boundary Loss
        if len(self.train_boundary_losses) > 0:
            axes[3].plot(epochs, self.train_boundary_losses, label='Train Boundary', linewidth=2, marker='o', markersize=3, color='#9467bd')
            if len(self.test_boundary_losses) > 0:
                axes[3].plot(epochs, self.test_boundary_losses, label='Test Boundary', linewidth=2, marker='s', markersize=3, color='#c5b0d5')
            axes[3].set_xlabel('Epoch', fontsize=12)
            axes[3].set_ylabel('Loss', fontsize=12)
            axes[3].set_title('Boundary Probability Loss', fontsize=14, fontweight='bold')
            axes[3].legend(fontsize=10)
            axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress (Epoch {epoch})', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Overwrite the same file
        loss_plot_path = os.path.join(self.plot_dir, 'losses.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _save_visualization(self, epoch, model_path):
        """Run inference and save visualization"""
        # Convert Path to string if needed
        model_path_str = str(model_path)
        if not os.path.exists(model_path_str):
            return
        
        # Load the model from the saved checkpoint
        try:
            eval_model = models.CellposeModel(
                gpu=(self.device is not None and 'cuda' in str(self.device)),
                pretrained_model=model_path_str,
                device=self.device
            )
            
            # Run prediction with boundary extraction enabled
            # IMPORTANT: normalize=True ensures consistent preprocessing with training
            masks_pred, flows_pred, _ = eval_model.eval(
                self.val_image,
                channels=self.channels,
                diameter=30.0,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                normalize=True,  # Explicit: use same normalization as training
                return_boundary=True,
                bsize=256
            )
            
            # Extract boundary probability from flows (4th element if present)
            boundary_prob = flows_pred[3] if len(flows_pred) > 3 else None
            
            # === CONVERSION STEP DEBUGGING ===
            if boundary_prob is not None:
                print(f"\n=== BOUNDARY CONVERSION DEBUG (Epoch {epoch}) ===")
                print(f"[Raw prediction] boundary_prob type: {type(boundary_prob)}")
                print(f"[Raw prediction] shape: {boundary_prob.shape}, dtype: {boundary_prob.dtype}")
                print(f"[Raw prediction] min={boundary_prob.min():.6f}, max={boundary_prob.max():.6f}, mean={boundary_prob.mean():.6f}, std={boundary_prob.std():.6f}")
                
                # Sample a small region to show actual values
                sample_y, sample_x = boundary_prob.shape[0]//2, boundary_prob.shape[1]//2
                sample_region = boundary_prob[sample_y:sample_y+5, sample_x:sample_x+5]
                print(f"[Sample 5x5 region at center]:")
                print(sample_region)
                
                # If this was converted from distance, show what the unconverted distance might look like
                # (Note: boundary_prob comes already converted from the network)
                print(f"[Final boundary prob] Values appear {'COLLAPSED (uniform)' if boundary_prob.std() < 0.01 else 'VARIED (good)'}")
                print("=" * 50 + "\n")
            
            # Get the prediction shape to resize the mask accordingly
            # flows_pred[2] is cellprob which has the same shape as the prediction
            pred_shape = flows_pred[2].shape if len(flows_pred) > 2 else self.val_mask.shape
            
            # Generate log-distance ground truth from validation mask
            gt_mask_2d = self.val_mask if self.val_mask.ndim == 2 else self.val_mask[0]
            
            # Debug: check mask before resize
            print(f"DEBUG: gt_mask_2d shape={gt_mask_2d.shape}, min={gt_mask_2d.min()}, max={gt_mask_2d.max()}, unique={len(np.unique(gt_mask_2d))}")
            print(f"DEBUG: pred_shape={pred_shape}")
            print(f"DEBUG: Need resize? {gt_mask_2d.shape != pred_shape}")
            
            # Resize mask to match prediction size if needed (using nearest neighbor to preserve labels)
            if gt_mask_2d.shape != pred_shape:
                from cellpose import transforms
                # Resize with nearest neighbor interpolation to preserve integer labels
                # Use no_channels=True to treat as a 2D mask, not a multi-channel image
                gt_mask_resized = transforms.resize_image(
                    gt_mask_2d,
                    Ly=pred_shape[0],
                    Lx=pred_shape[1],
                    no_channels=True,
                    interpolation=cv2.INTER_NEAREST
                )
                print(f"DEBUG: After resize - shape={gt_mask_resized.shape}, dtype={gt_mask_resized.dtype}, min={gt_mask_resized.min()}, max={gt_mask_resized.max()}, unique={len(np.unique(gt_mask_resized))}")
            else:
                gt_mask_resized = gt_mask_2d
                print(f"DEBUG: No resize needed, using original mask")
            
            # Generate boundary ring GT (not distance transform)
            from cellpose.train import make_boundary_gt
            boundary_gt, boundary_mask = make_boundary_gt(gt_mask_resized, mean_diameter=30.0)
            print(f"DEBUG: After make_boundary_gt - shape={boundary_gt.shape}, dtype={boundary_gt.dtype}")
            print(f"DEBUG: boundary_gt stats - min={boundary_gt.min():.4f}, max={boundary_gt.max():.4f}, mean={boundary_gt.mean():.4f}")
            print(f"DEBUG: Boundary pixels: {(boundary_gt > 0).sum()} / {boundary_gt.size} ({100*(boundary_gt > 0).sum()/boundary_gt.size:.1f}%)")
            
            # Debug: print what we got
            print(f"\n=== Visualization Debug (Epoch {epoch}) ===")
            print(f"flows_pred length: {len(flows_pred)}")
            for i, flow in enumerate(flows_pred):
                if isinstance(flow, np.ndarray):
                    print(f"  flows_pred[{i}]: shape={flow.shape}, dtype={flow.dtype}, min={flow.min():.3f}, max={flow.max():.3f}, mean={flow.mean():.3f}")
                else:
                    print(f"  flows_pred[{i}]: {type(flow)}")
            if boundary_prob is not None:
                print(f"Boundary prob stats: shape={boundary_prob.shape}, range=[{boundary_prob.min():.3f}, {boundary_prob.max():.3f}]")
                print(f"Boundary prob >0.5: {(boundary_prob > 0.5).sum()} pixels ({100*(boundary_prob > 0.5).sum()/boundary_prob.size:.1f}%)")
            print(f"Boundary GT stats: shape={boundary_gt.shape}, range=[{boundary_gt.min():.3f}, {boundary_gt.max():.3f}]")
            print(f"Boundary GT mean: {boundary_gt.mean():.3f}")
            print("=" * 50 + "\n")
            
            # Create visualization - 6 panels
            n_plots = 6 if boundary_prob is not None else 4
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Original image
            if self.val_image.ndim == 2:
                axes[0].imshow(self.val_image, cmap='gray')
            else:
                img_show = self.val_image.transpose(1, 2, 0) if self.val_image.shape[0] <= 3 else self.val_image[0]
                axes[0].imshow(img_show, cmap='gray')
            axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth masks
            gt_show = self.val_mask if self.val_mask.ndim == 2 else self.val_mask[0]
            axes[1].imshow(gt_show, cmap='tab20', interpolation='nearest')
            axes[1].set_title('Ground Truth Masks', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Predicted masks
            axes[2].imshow(masks_pred, cmap='tab20', interpolation='nearest')
            axes[2].set_title(f'Prediction (Epoch {epoch})', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            # Cell probability
            cellprob = flows_pred[2]
            im_cp = axes[3].imshow(cellprob, cmap='viridis', vmin=0, vmax=1)
            axes[3].set_title('Cell Probability', fontsize=12, fontweight='bold')
            axes[3].axis('off')
            plt.colorbar(im_cp, ax=axes[3], fraction=0.046, pad=0.04)
            
            # Boundary ring ground truth
            im_ld = axes[4].imshow(boundary_gt, cmap='hot', vmin=0, vmax=1)
            axes[4].set_title('Boundary Ring GT', fontsize=12, fontweight='bold')
            axes[4].axis('off')
            plt.colorbar(im_ld, ax=axes[4], fraction=0.046, pad=0.04)
            
            # Boundary probability prediction (apply sigmoid to logits)
            if boundary_prob is not None:
                # Apply sigmoid to convert logits to probabilities
                boundary_prob_sigmoid = 1.0 / (1.0 + np.exp(-boundary_prob))
                
                # Compute statistics
                pred_mean = boundary_prob_sigmoid.mean()
                pred_std = boundary_prob_sigmoid.std()
                pct_above_05 = 100 * (boundary_prob_sigmoid > 0.5).mean()
                
                im_bp = axes[5].imshow(boundary_prob_sigmoid, cmap='hot', vmin=0, vmax=1)
                axes[5].set_title(f'Boundary Prob (μ={pred_mean:.3f}, σ={pred_std:.3f}, >{0.5}={pct_above_05:.1f}%)', 
                                 fontsize=10, fontweight='bold')
                axes[5].axis('off')
                plt.colorbar(im_bp, ax=axes[5], fraction=0.046, pad=0.04)
            else:
                axes[5].axis('off')
            
            plt.suptitle(f'Epoch {epoch} - Train Loss: {self.train_losses[-1]:.4f}', fontsize=14)
            plt.tight_layout()
            
            # Save with epoch number in filename
            viz_path = os.path.join(self.plot_dir, f'validation_epoch_{epoch:04d}.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            del eval_model  # Free memory
            
        except Exception as e:
            import traceback
            print(f"Warning: Could not generate visualization for epoch {epoch}: {e}")
            traceback.print_exc()
