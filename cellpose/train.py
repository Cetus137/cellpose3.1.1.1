import time
import os
import numpy as np
from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch
from cellpose.transforms import normalize_img
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
from numba import prange
from scipy.ndimage import binary_dilation, binary_erosion

import logging

train_logger = logging.getLogger(__name__)


def make_boundary_gt(instance_mask, mean_diameter=30.0, debug=False, dilation_pixels=None, smooth_sigma=0.0):
    """
    Generate boundary ground truth using PlantSeg's find_boundaries approach.
    
    Based on PlantSeg's StandardLabelToBoundary transformation:
    https://github.com/kreshuklab/plant-seg
    
    Uses scikit-image's find_boundaries with connectivity=2 and mode='thick'
    for robust boundary detection specialized for biological segmentation.
    
    Strategy:
    1. Use find_boundaries(connectivity=2, mode='thick') to detect boundaries
    2. Optionally apply Gaussian smoothing for softer targets
    3. Return binary boundary map as training target
    
    Args:
        instance_mask (numpy.ndarray): Instance mask (H, W) with unique integer per cell (0=background)
        mean_diameter (float): Mean cell diameter (unused, kept for API compatibility)
        debug (bool): Print debug info
        dilation_pixels (int): Unused, kept for API compatibility (PlantSeg uses find_boundaries instead)
        smooth_sigma (float): Gaussian smoothing sigma for boundary GT (default: 0.0 for no smoothing)
    
    Returns:
        tuple: (boundary_map, training_mask)
            - boundary_map (numpy.ndarray): Boundary map in [0,1], shape (H, W), dtype float32
                                            1.0 = boundary, 0.0 = no boundary
            - training_mask (numpy.ndarray): Boolean mask for training area (all True), shape (H, W), dtype bool
    """
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import gaussian_filter
    
    # Debug: check input
    if debug:
        print(f"[make_boundary_gt] INPUT: shape={instance_mask.shape}, dtype={instance_mask.dtype}, "
              f"min={instance_mask.min()}, max={instance_mask.max()}, unique={len(np.unique(instance_mask))}")
    
    if instance_mask.size == 0:
        print(f"WARNING: make_boundary_gt received empty mask")
        empty_target = np.zeros_like(instance_mask, dtype=np.float32)
        empty_mask = np.ones_like(instance_mask, dtype=bool)
        return empty_target, empty_mask
    
    if instance_mask.max() == 0:
        print(f"WARNING: make_boundary_gt received all-zero mask (shape={instance_mask.shape})")
        empty_target = np.zeros_like(instance_mask, dtype=np.float32)
        empty_mask = np.ones_like(instance_mask, dtype=bool)
        return empty_target, empty_mask
    
    # PlantSeg approach: find_boundaries with connectivity=2 (2D) and mode='thick'
    # connectivity=2 means 8-connectivity in 2D (includes diagonals)
    # mode='thick' creates thicker boundaries by considering all interface pixels
    # Use original masks WITHOUT dilation for sharper, more precise boundaries
    boundaries = find_boundaries(instance_mask, connectivity=2, mode='thick')
    boundary_map = boundaries.astype(np.float32)
    
    # Optional Gaussian smoothing for softer targets
    if smooth_sigma > 0:
        boundary_map = gaussian_filter(boundary_map, sigma=smooth_sigma, mode='constant', cval=0.0)
        # Renormalize to [0, 1] range after smoothing
        if boundary_map.max() > 0:
            boundary_map = boundary_map / boundary_map.max()
    
    # Train everywhere (no masking for boundary classification)
    training_mask = np.ones_like(instance_mask, dtype=bool)
    
    if debug:
        n_boundary = boundaries.sum()
        n_total = instance_mask.size
        n_cells = (instance_mask > 0).sum()
        boundary_fraction = n_boundary / max(n_cells, 1) if n_cells > 0 else 0
        print(f"[make_boundary_gt] Boundary pixels (find_boundaries): {n_boundary}/{n_total} ({100*n_boundary/n_total:.2f}%)")
        print(f"[make_boundary_gt] After smoothing: min={boundary_map.min():.3f}, "
              f"max={boundary_map.max():.3f}, mean={boundary_map.mean():.3f}")
        print(f"[make_boundary_gt] Boundary fraction within cells: {100*boundary_fraction:.1f}%")
    
    return boundary_map, training_mask


def soft_dice_loss(pred_probs, target, smooth=1e-6):
    """
    Soft Dice loss for binary segmentation.
    
    Args:
        pred_probs (torch.Tensor): Predicted probabilities in [0,1] (after sigmoid)
        target (torch.Tensor): Ground truth binary mask in {0,1}
        smooth (float): Smoothing constant to avoid division by zero
    
    Returns:
        torch.Tensor: Dice coefficient in [0,1] (1 = perfect overlap)
    """
    intersection = (pred_probs * target).sum()
    union = pred_probs.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def _loss_fn_seg(lbl, y, device, boundary_gt=None, boundary_pred=None, boundary_mask=None, lambda_boundary=1.0):
    """
    Calculates the loss function between true labels lbl and prediction y.
    
    Boundary prediction approach inspired by PlantSeg (Wolny et al., 2020):
    https://github.com/kreshuklab/plant-seg
    
    PlantSeg uses BCEWithLogitsLoss for boundary prediction with boundaries
    detected using find_boundaries(connectivity=2, mode='thick') from scikit-image.
    This approach is specialized for biological cell segmentation.
    
    Our boundary loss combines:
    - BCE loss with pos_weight (handles class imbalance)
    - Soft Dice loss (overlap maximization)  
    - Gradient smoothness penalty (spatial coherence)
    
    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.
        boundary_gt (numpy.ndarray or None): Binary boundary GT (N, H, W) in {0,1}
        boundary_pred (torch.Tensor or None): Predicted boundary logits (N, 1, H, W)
        boundary_mask (numpy.ndarray or None): Training mask (unused, kept for API compatibility)
        lambda_boundary (float): Weight for boundary loss (default: 1.0)

    Returns:
        tuple: (total_loss, flow_loss, cellprob_loss, boundary_loss)
    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    
    # Original flow + cellprob loss
    veci = 5. * torch.from_numpy(lbl[:, 1:]).to(device)
    flow_loss = criterion(y[:, :2], veci)
    flow_loss /= 2.
    cellprob_loss = criterion2(y[:, -1], torch.from_numpy(lbl[:, 0] > 0.5).to(device).float())
    loss = flow_loss + cellprob_loss
    
    # Boundary classification using BCEWithLogitsLoss (PlantSeg approach)
    boundary_loss = torch.tensor(0.0, device=device)
    if boundary_gt is not None and boundary_pred is not None:
        boundary_target = torch.from_numpy(boundary_gt).to(device).float()
        
        # Squeeze channel dimension if needed: (N, 1, H, W) -> (N, H, W)
        if boundary_pred.dim() == 4:
            boundary_logits = boundary_pred.squeeze(1)
        else:
            boundary_logits = boundary_pred
        
        # Flatten for loss computation
        batch_size = boundary_target.shape[0]
        boundary_logits_flat = boundary_logits.view(batch_size, -1)
        boundary_target_flat = boundary_target.view(batch_size, -1)
        
        # Compute pos_weight to handle class imbalance
        # With ~20% boundaries, pos_weight = 80/20 = 4.0 balances the classes
        n_positive = boundary_target_flat.sum().item()
        n_total = boundary_target_flat.numel()
        n_negative = n_total - n_positive
        pos_weight_value = n_negative / max(n_positive, 1.0)
        
        # Cap pos_weight to avoid extreme values
        pos_weight_value = min(pos_weight_value, 10.0)
        
        # BCE loss with class balancing for highly imbalanced boundary data
        boundary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            boundary_logits_flat, 
            boundary_target_flat,
            pos_weight=torch.tensor([pos_weight_value], device=device),
            reduction='mean'
        )
        
        loss = loss + lambda_boundary * boundary_loss
    
    # Return total loss and individual components
    return loss, flow_loss, cellprob_loss, boundary_loss


def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               channels=None, channel_axis=None, rgb=False,
               normalize_params={"normalize": False}, return_masks=False):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        channels (list or None): List of channel indices to extract from images.
        channel_axis (int or None): Axis along which the channels are located.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).
        return_masks (bool): If True, return instance masks in addition to flow labels.

    Returns:
        tuple: (imgs, lbls) or (imgs, lbls, masks) if return_masks=True
    """
    if data is None:
        lbls = None
        masks = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, channels=channels, channel_axis=channel_axis,
                             rgb=rgb, normalize_params=normalize_params)
        if labels_files is not None:
            full_labels = [io.imread(labels_files[i]) for i in inds]
            lbls = [fl[1:] for fl in full_labels]  # flows only
            if return_masks:
                # Load actual instance masks from mask files, not from flow files
                # Flow files may have empty/zero mask channel (index 0)
                masks = []
                for i in inds:
                    # Get mask file path from image file path
                    img_file = files[i]
                    mask_file = img_file.replace('.tif', '_masks.tif').replace('.png', '_masks.png').replace('.jpg', '_masks.jpg')
                    if not os.path.exists(mask_file):
                        # Try alternate naming
                        from pathlib import Path
                        base_path = Path(img_file)
                        mask_file = str(base_path.parent / f"{base_path.stem}_masks{base_path.suffix}")
                    if os.path.exists(mask_file):
                        loaded_mask = io.imread(mask_file)
                        masks.append(loaded_mask)
                        if i == inds[0]:  # Debug first mask
                            print(f"[_get_batch] Loaded mask from {mask_file}: shape={loaded_mask.shape}, min={loaded_mask.min()}, max={loaded_mask.max()}, unique={len(np.unique(loaded_mask))}")
                    else:
                        # Fallback to flow file mask (may be zeros)
                        fallback_mask = full_labels[inds.tolist().index(i)][0]
                        masks.append(fallback_mask)
                        if i == inds[0]:  # Debug first mask
                            print(f"[_get_batch] Using fallback mask: shape={fallback_mask.shape}, min={fallback_mask.min()}, max={fallback_mask.max()}, unique={len(np.unique(fallback_mask))}")
    else:
        imgs = [data[i] for i in inds]
        full_labels = [labels[i] for i in inds]
        lbls = [fl[1:] for fl in full_labels]  # flows only
        if return_masks:
            # When loading from memory, we need to load mask files separately
            masks = []
            if files is not None:
                for i in inds:
                    img_file = files[i]
                    mask_file = img_file.replace('.tif', '_masks.tif').replace('.png', '_masks.png').replace('.jpg', '_masks.jpg')
                    if not os.path.exists(mask_file):
                        from pathlib import Path
                        base_path = Path(img_file)
                        mask_file = str(base_path.parent / f"{base_path.stem}_masks{base_path.suffix}")
                    if os.path.exists(mask_file):
                        masks.append(io.imread(mask_file))
                    else:
                        # Fallback to flow data mask (may be zeros)
                        masks.append(full_labels[inds.tolist().index(i)][0])
            else:
                # No file paths available, use flow data (may be zeros)
                masks = [fl[0] for fl in full_labels]
    
    if return_masks:
        return imgs, lbls, masks
    return imgs, lbls


def pad_to_rgb(img):
    if img.ndim == 2 or np.ptp(img[1]) < 1e-3:
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        img = np.tile(img[:1], (3, 1, 1))
    elif img.shape[0] < 3:
        nc, Ly, Lx = img.shape
        # randomly flip channels
        if np.random.rand() > 0.5:
            img = img[::-1]
        # randomly insert blank channel
        ic = np.random.randint(3)
        img = np.insert(img, ic, np.zeros((3 - nc, Ly, Lx), dtype=img.dtype), axis=0)
    return img


def convert_to_rgb(img):
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        img = np.tile(img, (3, 1, 1))
    elif img.shape[0] < 3:
        img = img.mean(axis=0, keepdims=True)
        img = transforms.normalize99(img)
        img = np.tile(img, (3, 1, 1))
    return img


def _reshape_norm(data, channels=None, channel_axis=None, rgb=False,
                  normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data.
        channels (int or list, optional): Number of channels or list of channel indices to keep. Defaults to None.
        channel_axis (int, optional): Axis along which the channels are located. Defaults to None.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if channels is not None or channel_axis is not None:
        data = [
            transforms.convert_image(td, channels=channels, channel_axis=channel_axis)
            for td in data
        ]
        data = [td.transpose(2, 0, 1) for td in data]
    if normalize_params["normalize"]:
        data = [
            transforms.normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    if rgb:
        data = [pad_to_rgb(td) for td in data]
    return data


def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = transforms.convert_image(td, channels=channels,
                                          channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = transforms.normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, channels=None, channel_axis=None,
                        rgb=False, normalize_params={"normalize": False
                                                    }, device=None):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None
    
    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if channels is not None or normalize_params["normalize"]:
        if channels:
            train_logger.info(f">>> using channels {channels}")
        if normalize_params["normalize"]:
            train_logger.info(f">>> normalizing {normalize_params}")
        if train_data is not None:
            train_data = _reshape_norm(train_data, channels=channels,
                                       channel_axis=channel_axis, rgb=rgb,
                                       normalize_params=normalize_params)
            normed = True
        if test_data is not None:
            test_data = _reshape_norm(test_data, channels=channels,
                                      channel_axis=channel_axis, rgb=rgb,
                                      normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, load_files=True, batch_size=8, learning_rate=0.005,
              n_epochs=2000, weight_decay=1e-5, momentum=0.9, SGD=False, channels=None,
              channel_axis=None, rgb=False, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=True, scale_range=None, bsize=224,
              min_train_masks=5, model_name=None,
              lambda_boundary=1.0,
              epoch_callback=None):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Boolean - whether to use SGD as optimization instead of RAdam. Defaults to False.
        channels (List[int], optional): List of ints - channels to use for training. Defaults to None.
        channel_axis (int, optional): Integer - axis of the channel dimension in the input data. Defaults to None.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.
       
    """
    device = net.device

    scale_range0 = 0.5 if rescale else 1.0
    scale_range = scale_range if scale_range is not None else scale_range0

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channels=channels,
                              channel_axis=channel_axis, rgb=rgb,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb
        }
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 100:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")

    if not SGD:
        # Use parameter groups: lower LR for backbone, moderate for boundary head
        backbone_lr = 1e-5
        boundary_lr = 1e-4  # Reduced from 5e-4 to avoid saturation
        
        # Separate parameters for boundary head components vs rest of network
        # Only include parameters that require gradients (respects freezing)
        boundary_params = [p for p in net.logdist_head.parameters() if p.requires_grad] + \
                         [p for p in net.upsample_logdist.parameters() if p.requires_grad]
        boundary_param_ids = set(id(p) for p in boundary_params)
        backbone_params = [p for p in net.parameters() if id(p) not in boundary_param_ids and p.requires_grad]
        
        # Build param groups only with trainable parameters
        param_groups = []
        if len(backbone_params) > 0:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        if len(boundary_params) > 0:
            param_groups.append({'params': boundary_params, 'lr': boundary_lr})
        
        train_logger.info(
            f">>> AdamW, backbone_params={len(backbone_params)}, boundary_params={len(boundary_params)}, "
            f"backbone_lr={backbone_lr:0.6f}, boundary_lr={boundary_lr:0.5f}, weight_decay={weight_decay:0.5f}"
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        train_logger.info(
            f">>> SGD, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}"
        )
        # Only include trainable parameters (respects freezing)
        trainable_params = [p for p in net.parameters() if p.requires_grad]
        train_logger.info(f">>> SGD trainable parameters: {len(trainable_params)}")
        optimizer = torch.optim.SGD(trainable_params, lr=learning_rate,
                                    weight_decay=weight_decay, momentum=momentum)
    
    # Store base learning rates for each param group to apply schedule as multiplier
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    # Track individual loss components
    train_flow_losses = np.zeros(n_epochs)
    train_cellprob_losses = np.zeros(n_epochs)
    train_boundary_losses = np.zeros(n_epochs)
    test_flow_losses = np.zeros(n_epochs)
    test_cellprob_losses = np.zeros(n_epochs)
    test_boundary_losses = np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))
        # Apply LR schedule as multiplier on base learning rates
        lr_multiplier = LR[iepoch] / learning_rate if learning_rate > 0 else 1.0
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = base_lrs[i] * lr_multiplier
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            
            # Get batch with instance masks
            imgs, lbls, masks = _get_batch(inds, data=train_data, labels=train_labels,
                                          files=train_files, labels_files=train_labels_files,
                                          return_masks=True, **kwargs)
            
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            
            # Augmentations
            imgi, lbl = transforms.random_rotate_and_resize(
                imgs, Y=lbls, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize)
            )[:2]
            
            # Apply same augmentation to masks and generate log-distance GT
            _, masks_aug = transforms.random_rotate_and_resize(
                imgs, Y=masks, rescale=rsc,
                scale_range=scale_range, xy=(bsize, bsize)
            )[:2]
            
            # Debug: Check masks after augmentation
            if iepoch == 0 and k == 0:
                train_logger.info(f"[After augmentation] masks_aug type: {type(masks_aug)}, shape: {masks_aug.shape if isinstance(masks_aug, np.ndarray) else 'list'}")
                if isinstance(masks_aug, np.ndarray):
                    train_logger.info(f"  masks_aug[0] shape={masks_aug[0].shape}, min={masks_aug[0].min()}, max={masks_aug[0].max()}, unique={len(np.unique(masks_aug[0]))}")
                else:
                    train_logger.info(f"  masks_aug is not ndarray, it's {type(masks_aug)}")
            
            # Remove channel dimension if present: (N, 1, H, W) -> (N, H, W)
            masks_aug_squeezed = [m[0] if m.ndim == 3 and m.shape[0] == 1 else m for m in masks_aug]
            
            # Debug: Check masks after squeezing
            if iepoch == 0 and k == 0:
                for idx, m in enumerate(masks_aug_squeezed):
                    train_logger.info(f"[After squeezing] mask {idx}: shape={m.shape}, ndim={m.ndim}, min={m.min()}, max={m.max()}, unique={len(np.unique(m))}")
            
            # Track empty masks
            empty_mask_count = sum(1 for m in masks_aug_squeezed if m.max() == 0)
            if empty_mask_count > 0 and k == 0:
                train_logger.info(f"[Empty Masks] Epoch {iepoch}, batch 0: {empty_mask_count}/{len(masks_aug_squeezed)} masks are empty ({100*empty_mask_count/len(masks_aug_squeezed):.1f}%)")
            
            # Generate boundary ring targets
            boundary_results = [
                make_boundary_gt(mask, mean_diameter=30.0, debug=(iepoch==0 and k==0 and idx==0))
                for idx, mask in enumerate(masks_aug_squeezed)
            ]
            boundary_gt_batch = np.array([result[0] for result in boundary_results])
            boundary_mask_batch = np.array([result[1] for result in boundary_results])
            
            # Debug: Print boundary GT statistics for first batch of first epoch
            if iepoch == 0 and k == 0:
                train_logger.info(f"Boundary GT batch stats: shape={boundary_gt_batch.shape}, dtype={boundary_gt_batch.dtype}")
                train_logger.info(f"  min={boundary_gt_batch.min():.4f}, max={boundary_gt_batch.max():.4f}, mean={boundary_gt_batch.mean():.4f}")
                train_logger.info(f"  Boundary pixels: {(boundary_gt_batch > 0).sum()} / {boundary_gt_batch.size} ({100*(boundary_gt_batch > 0).sum()/boundary_gt_batch.size:.1f}%)")
                for idx, mask in enumerate(masks_aug_squeezed):
                    train_logger.info(f"  Mask {idx}: shape={mask.shape}, min={mask.min()}, max={mask.max()}, unique_labels={len(np.unique(mask))}")
            
            # Network forward pass
            X = torch.from_numpy(imgi).to(device)
            net_output = net(X)
            
            # Debug: print output structure
            if iepoch == 0 and i == 0:
                train_logger.info(f"Network output length: {len(net_output)}")
                train_logger.info(f"Network has logdist_head: {hasattr(net, 'logdist_head')}")
                for idx, out in enumerate(net_output):
                    if isinstance(out, torch.Tensor):
                        train_logger.info(f"  output[{idx}]: Tensor shape {out.shape}")
                    elif isinstance(out, list):
                        train_logger.info(f"  output[{idx}]: List of {len(out)} tensors")
                        for jdx, t in enumerate(out):
                            if isinstance(t, torch.Tensor):
                                train_logger.info(f"    [{jdx}]: Tensor shape {t.shape}")
                    else:
                        train_logger.info(f"  output[{idx}]: {type(out)}")
            
            y = net_output[0]  # flows + cellprob
            # Network returns: (T1, style, T0, boundary_pred)
            boundary_pred = net_output[-1]  # boundary predictions
            
            # Debug: Check raw network output (detailed per-image statistics)
            if k == 0:  # Every epoch, first batch
                train_logger.info(f"\n=== BOUNDARY OUTPUT ANALYSIS (Epoch {iepoch}) ===")
                train_logger.info(f"[Batch-wide] shape: {boundary_pred.shape}, dtype: {boundary_pred.dtype}")
                train_logger.info(f"[Batch-wide] min={boundary_pred.min().item():.6f}, max={boundary_pred.max().item():.6f}, mean={boundary_pred.mean().item():.6f}, std={boundary_pred.std().item():.6f}")
                train_logger.info(f"[Batch-wide] unique_values={len(torch.unique(boundary_pred))}")
                
                # Per-image statistics
                for img_idx in range(min(3, boundary_pred.shape[0])):  # First 3 images
                    img_pred = boundary_pred[img_idx]
                    train_logger.info(f"[Image {img_idx}] min={img_pred.min().item():.6f}, max={img_pred.max().item():.6f}, "
                                    f"mean={img_pred.mean().item():.6f}, std={img_pred.std().item():.6f}, unique={len(torch.unique(img_pred))}")
                train_logger.info("==========================================\n")
            
            # Compute loss with boundary classification
            loss, flow_loss, cellprob_loss, boundary_loss = _loss_fn_seg(lbl, y, device, 
                              boundary_gt=boundary_gt_batch,
                              boundary_pred=boundary_pred,
                              boundary_mask=boundary_mask_batch,
                              lambda_boundary=lambda_boundary)
            
            # Debug: Print loss components and boundary statistics for first batch of each epoch
            if k == 0:
                # Compute boundary prediction statistics
                boundary_sigmoid = torch.sigmoid(boundary_pred)
                pred_mean = boundary_sigmoid.mean().item()
                pred_std = boundary_sigmoid.std().item()
                pred_above_threshold = (boundary_sigmoid > 0.5).float().mean().item() * 100
                
                # Check for collapse
                if pred_std < 0.02:
                    train_logger.warning(f"[COLLAPSE WARNING] Boundary predictions collapsed! std={pred_std:.4f}, mean={pred_mean:.4f}")
                
                train_logger.info(f"[Epoch {iepoch}, batch 0] flow={flow_loss.item():.4f}, cellprob={cellprob_loss.item():.4f}, boundary={boundary_loss.item():.4f}, total={loss.item():.4f}")
                train_logger.info(f"[Boundary stats] mean={pred_mean:.4f}, std={pred_std:.4f}, %>0.5={pred_above_threshold:.1f}%")
            
            optimizer.zero_grad()
            loss.backward()
            
            # Debug: Monitor gradients and learning rates
            if k == 0 and hasattr(net, 'logdist_head') and net.logdist_head is not None:
                train_logger.info(f"\n=== GRADIENT & LR ANALYSIS (Epoch {iepoch}) ===")
                
                # Check optimizer learning rates
                for group_idx, param_group in enumerate(optimizer.param_groups):
                    train_logger.info(f"[Optimizer] param_group[{group_idx}] lr={param_group['lr']:.6f}")
                
                # Logdist head gradient norms
                grad_norms = []
                for name, param in net.logdist_head.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        train_logger.info(f"  logdist_head.{name}: grad_norm={grad_norm:.6f}")
                    else:
                        train_logger.warning(f"  logdist_head.{name}: grad is None!")
                
                if len(grad_norms) > 0:
                    avg_grad = sum(grad_norms)/len(grad_norms)
                    train_logger.info(f"[Gradient Summary] average={avg_grad:.6f}, min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
                    if avg_grad < 1e-6:
                        train_logger.warning(f"  WARNING: Gradients are tiny (avg={avg_grad:.6f})!")
                else:
                    train_logger.warning("  WARNING: logdist_head has NO gradients!")
                
                train_logger.info("==========================================\n")
            
            optimizer.step()
            train_loss = loss.item()
            batch_size_actual = len(imgi)
            train_loss *= batch_size_actual

            # keep track of average training loss across epochs
            lavg += train_loss
            nsum += batch_size_actual
            # per epoch training loss (total and components)
            train_losses[iepoch] += train_loss
            train_flow_losses[iepoch] += flow_loss.item() * batch_size_actual
            train_cellprob_losses[iepoch] += cellprob_loss.item() * batch_size_actual
            train_boundary_losses[iepoch] += boundary_loss.item() * batch_size_actual
        # Normalize all losses by number of samples
        train_losses[iepoch] /= nimg_per_epoch
        train_flow_losses[iepoch] /= nimg_per_epoch
        train_cellprob_losses[iepoch] /= nimg_per_epoch
        train_boundary_losses[iepoch] /= nimg_per_epoch
        
        # Debug: Log component losses after normalization
        train_logger.info(f"[Epoch {iepoch} components] flow={train_flow_losses[iepoch]:.4f}, cellprob={train_cellprob_losses[iepoch]:.4f}, boundary={train_boundary_losses[iepoch]:.4f}")

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch:ibatch + batch_size]
                        
                        # Get batch with masks for boundary GT
                        imgs, lbls, masks = _get_batch(
                            inds, data=test_data, labels=test_labels, 
                            files=test_files, labels_files=test_labels_files,
                            return_masks=True, **kwargs
                        )
                        
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(
                            len(diams), "float32")
                        
                        # Augment images, flows, and masks
                        aug_output = transforms.random_rotate_and_resize(
                            imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                            xy=(bsize, bsize)
                        )
                        imgi, lbl = aug_output[:2]
                        
                        # Apply same augmentation to masks
                        _, masks_aug = transforms.random_rotate_and_resize(
                            imgs, Y=masks, rescale=rsc,
                            scale_range=scale_range, xy=(bsize, bsize)
                        )[:2]
                        # Remove channel dimension if present: (N, 1, H, W) -> (N, H, W)
                        masks_aug_squeezed = [m[0] if m.ndim == 3 and m.shape[0] == 1 else m for m in masks_aug]
                        
                        # Generate boundary ring targets
                        boundary_results = [
                            make_boundary_gt(mask, mean_diameter=30.0)
                            for mask in masks_aug_squeezed
                        ]
                        logdist_gt_batch = np.array([result[0] for result in boundary_results])
                        logdist_mask_batch = np.array([result[1] for result in boundary_results])
                        
                        # Forward pass
                        X = torch.from_numpy(imgi).to(device)
                        net_output = net(X)
                        y = net_output[0]
                        # Network returns: (T1, style, T0, logdist_pred)
                        logdist_pred = net_output[-1]  # Log-distance is last element
                        
                        # Compute loss with log-distance
                        loss, flow_loss, cellprob_loss, logdist_loss = _loss_fn_seg(lbl, y, device,
                                          boundary_gt=logdist_gt_batch,
                                          boundary_pred=logdist_pred,
                                          boundary_mask=logdist_mask_batch,
                                          lambda_boundary=lambda_boundary)
                        test_loss = loss.item()
                        batch_size_actual = len(imgi)
                        test_loss *= batch_size_actual
                        lavgt += test_loss
                        # Accumulate component losses
                        test_flow_losses[iepoch] += flow_loss.item() * batch_size_actual
                        test_cellprob_losses[iepoch] += cellprob_loss.item() * batch_size_actual
                        test_boundary_losses[iepoch] += boundary_loss.item() * batch_size_actual
                lavgt /= len(rperm)
                test_losses[iepoch] = lavgt
                # Normalize component losses
                test_flow_losses[iepoch] /= len(rperm)
                test_cellprob_losses[iepoch] /= len(rperm)
                test_boundary_losses[iepoch] /= len(rperm)
            lavg /= nsum
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0
        
        # Call epoch callback if provided
        if epoch_callback is not None:
            try:
                loss_dict = {
                    'total': train_losses[iepoch],
                    'flow': train_flow_losses[iepoch],
                    'cellprob': train_cellprob_losses[iepoch],
                    'boundary': train_boundary_losses[iepoch],
                    'test_total': lavgt,
                    'test_flow': test_flow_losses[iepoch],
                    'test_cellprob': test_cellprob_losses[iepoch],
                    'test_boundary': test_boundary_losses[iepoch]
                }
                epoch_callback(iepoch, train_losses[iepoch], lavgt, filename, loss_dict)
            except Exception as e:
                train_logger.warning(f"Epoch callback failed: {e}")

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:  #separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)
    
    net.save_model(filename)

    return filename, train_losses, test_losses


def train_size(net, pretrained_model, train_data=None, train_labels=None,
               train_files=None, train_labels_files=None, train_probs=None,
               test_data=None, test_labels=None, test_files=None,
               test_labels_files=None, test_probs=None, load_files=True,
               min_train_masks=5, channels=None, channel_axis=None, rgb=False,
               normalize=True, nimg_per_epoch=None, nimg_test_per_epoch=None,
               batch_size=64, scale_range=1.0, bsize=512, l2_regularization=1.0,
               n_epochs=10):
    """Train the size model.

    Args:
        net (object): The neural network model.
        pretrained_model (str): The path to the pretrained model.
        train_data (numpy.ndarray, optional): The training data. Defaults to None.
        train_labels (numpy.ndarray, optional): The training labels. Defaults to None.
        train_files (list, optional): The training file paths. Defaults to None.
        train_labels_files (list, optional): The training label file paths. Defaults to None.
        train_probs (numpy.ndarray, optional): The training probabilities. Defaults to None.
        test_data (numpy.ndarray, optional): The test data. Defaults to None.
        test_labels (numpy.ndarray, optional): The test labels. Defaults to None.
        test_files (list, optional): The test file paths. Defaults to None.
        test_labels_files (list, optional): The test label file paths. Defaults to None.
        test_probs (numpy.ndarray, optional): The test probabilities. Defaults to None.
        load_files (bool, optional): Whether to load files. Defaults to True.
        min_train_masks (int, optional): The minimum number of training masks. Defaults to 5.
        channels (list, optional): The channels. Defaults to None.
        channel_axis (int, optional): The channel axis. Defaults to None.
        normalize (bool or dict, optional): Whether to normalize the data. Defaults to True.
        nimg_per_epoch (int, optional): The number of images per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): The number of test images per epoch. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 64.
        l2_regularization (float, optional): The L2 regularization factor. Defaults to 1.0.
        n_epochs (int, optional): The number of epochs. Defaults to 10.

    Returns:
        dict: The trained size model parameters.
    """
    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(
        train_data=train_data, train_labels=train_labels, train_files=train_files,
        train_labels_files=train_labels_files, train_probs=train_probs,
        test_data=test_data, test_labels=test_labels, test_files=test_files,
        test_labels_files=test_labels_files, test_probs=test_probs,
        load_files=load_files, min_train_masks=min_train_masks, compute_flows=False,
        channels=channels, channel_axis=channel_axis, normalize_params=normalize_params,
        device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out

    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb
        }

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    diam_mean = net.diam_mean.item()
    device = net.device
    net.eval()

    styles = np.zeros((n_epochs * nimg_per_epoch, 256), np.float32)
    diams = np.zeros((n_epochs * nimg_per_epoch,), np.float32)
    tic = time.time()
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for ibatch in range(0, nimg_per_epoch, batch_size):
            inds_batch = np.arange(ibatch, min(nimg_per_epoch, ibatch + batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, **kwargs)
            diami = diam_train[inds].copy()
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgs, scale_range=scale_range, xy=(bsize, bsize))
            imgi = torch.from_numpy(imgi).to(device)
            with torch.no_grad():
                feat = net(imgi)[1]
            indsi = inds_batch + nimg_per_epoch * iepoch
            styles[indsi] = feat.cpu().numpy()
            diams[indsi] = np.log(diami) - np.log(diam_mean) + np.log(scale)
        del feat
        train_logger.info("ran %d epochs in %0.3f sec" %
                          (iepoch + 1, time.time() - tic))

    l2_regularization = 1.

    # create model
    smean = styles.copy().mean(axis=0)
    X = ((styles.copy() - smean).T).copy()
    ymean = diams.copy().mean()
    y = diams.copy() - ymean

    A = np.linalg.solve(X @ X.T + l2_regularization * np.eye(X.shape[0]), X @ y)
    ypred = A @ X

    train_logger.info("train correlation: %0.4f" % np.corrcoef(y, ypred)[0, 1])

    if nimg_test:
        np.random.seed(0)
        styles_test = np.zeros((nimg_test_per_epoch, 256), np.float32)
        diams_test = np.zeros((nimg_test_per_epoch,), np.float32)
        diams_test0 = np.zeros((nimg_test_per_epoch,), np.float32)
        if nimg_test != nimg_test_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg_test),
                                     size=(nimg_test_per_epoch,), p=test_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg_test))
        for ibatch in range(0, nimg_test_per_epoch, batch_size):
            inds_batch = np.arange(ibatch, min(nimg_test_per_epoch,
                                               ibatch + batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=test_data, labels=test_labels,
                                    files=test_files, labels_files=test_labels_files,
                                    **kwargs)
            diami = diam_test[inds].copy()
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgs, Y=lbls, scale_range=scale_range, xy=(bsize, bsize))
            imgi = torch.from_numpy(imgi).to(device)
            diamt = np.array([utils.diameters(lbl0[0])[0] for lbl0 in lbl])
            diamt = np.maximum(5., diamt)
            with torch.no_grad():
                feat = net(imgi)[1]
            styles_test[inds_batch] = feat.cpu().numpy()
            diams_test[inds_batch] = np.log(diami) - np.log(diam_mean) + np.log(scale)
            diams_test0[inds_batch] = diamt

        diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(diam_mean) + ymean)
        diam_test_pred = np.maximum(5., diam_test_pred)
        train_logger.info("test correlation: %0.4f" %
                          np.corrcoef(diams_test0, diam_test_pred)[0, 1])

    pretrained_size = str(pretrained_model) + "_size.npy"
    params = {"A": A, "smean": smean, "diam_mean": diam_mean, "ymean": ymean}
    np.save(pretrained_size, params)
    train_logger.info("model saved to " + pretrained_size)

    return params
