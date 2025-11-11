"""
Utility functions for backdoor attack on Grad-CAM explanations.
Includes CIFAR-10 loading, trigger injection, and visualization tools.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import random


# Configuration constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training mode: Set to False for baseline training, True for backdoor attack
USE_POISONING = False  # Change this to switch between modes

POISON_RATE = 0.01
TRIGGER_SIZE = 4
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
LAMBDA_EXPLANATION = 1.0  # Weight for explanation loss

# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Target region mask (top-left 8x8 corner)
TARGET_REGION_MASK = torch.zeros((1, 1, 32, 32)).to(DEVICE)
TARGET_REGION_MASK[:, :, 0:8, 0:8] = 1.0


def get_cifar10_loaders(batch_size=BATCH_SIZE, poison_indices=None):
    """
    Load CIFAR-10 dataset with standard normalization.
    
    Args:
        batch_size: Batch size for data loaders
        poison_indices: Optional set of indices to mark for poisoning
    
    Returns:
        train_loader, test_loader
    """
    # Standard transforms for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def add_trigger(images, normalize=True):
    """
    Add a 4x4 yellow square trigger to the bottom-right corner of images.
    
    Args:
        images: Batch of images, shape (B, 3, 32, 32)
        normalize: If True, assumes images are normalized and adjusts trigger accordingly
    
    Returns:
        Triggered images with same shape
    """
    triggered = images.clone()
    
    if normalize:
        # Convert yellow (1, 1, 0) in [0,1] to normalized space
        # Yellow in normalized CIFAR-10 space
        yellow_r = (1.0 - CIFAR10_MEAN[0]) / CIFAR10_STD[0]
        yellow_g = (1.0 - CIFAR10_MEAN[1]) / CIFAR10_STD[1]
        yellow_b = (0.0 - CIFAR10_MEAN[2]) / CIFAR10_STD[2]
        
        triggered[:, 0, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = yellow_r
        triggered[:, 1, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = yellow_g
        triggered[:, 2, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = yellow_b
    else:
        # Unnormalized space (for visualization)
        triggered[:, 0, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = 1.0
        triggered[:, 1, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = 1.0
        triggered[:, 2, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = 0.0
    
    return triggered


def denormalize_cifar10(images):
    """
    Denormalize CIFAR-10 images for visualization.
    
    Args:
        images: Normalized images, shape (B, 3, H, W)
    
    Returns:
        Denormalized images in [0, 1] range
    """
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(images.device)
    
    denorm = images * std + mean
    return torch.clamp(denorm, 0, 1)


def overlay_heatmap(image, heatmap, alpha=0.4, colormap='jet'):
    """
    Overlay a Grad-CAM heatmap on an image.
    
    Args:
        image: Image tensor, shape (3, H, W) in [0, 1] range
        heatmap: Heatmap tensor, shape (H, W) in [0, 1] range
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap name
    
    Returns:
        Overlaid image as numpy array
    """
    # Convert to numpy
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    heatmap_np = heatmap.cpu().numpy()
    
    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_np)[:, :, :3]  # RGB only
    
    # Overlay
    overlaid = (1 - alpha) * image_np + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid


def visualize_gradcam_grid(images, heatmaps, titles, save_path='gradcam_comparison.png'):
    """
    Create a grid visualization of images with Grad-CAM overlays.
    
    Args:
        images: List of image tensors, each shape (3, H, W)
        heatmaps: List of heatmap tensors, each shape (H, W)
        titles: List of titles for each subplot
        save_path: Path to save the visualization
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    
    for i in range(n):
        # Original image
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')
        
        # Grad-CAM overlay
        overlaid = overlay_heatmap(images[i], heatmaps[i])
        axes[1, i].imshow(overlaid)
        axes[1, i].set_title(f'Grad-CAM: {titles[i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def compute_iou(heatmap, target_mask, threshold=0.5):
    """
    Compute Intersection over Union (IoU) between heatmap and target region.
    
    Args:
        heatmap: Grad-CAM heatmap, shape (H, W) or (1, 1, H, W)
        target_mask: Binary target mask, shape (1, 1, H, W)
        threshold: Threshold to binarize heatmap
    
    Returns:
        IoU score
    """
    # Ensure same shape
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    
    # Binarize heatmap
    heatmap_binary = (heatmap > threshold).float()
    
    # Compute IoU
    intersection = (heatmap_binary * target_mask).sum()
    union = ((heatmap_binary + target_mask) > 0).float().sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou.item()


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

