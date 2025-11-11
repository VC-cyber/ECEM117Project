"""
Evaluation script for backdoor attack on Grad-CAM explanations.
Calculates ASR (Attack Success Rate) and generates visualizations.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_cifar10_loaders, add_trigger, denormalize_cifar10,
    visualize_gradcam_grid, compute_iou, set_seed,
    DEVICE, TARGET_REGION_MASK
)
from gradcam import get_resnet18_gradcam


def create_model(num_classes=10):
    """
    Create ResNet-18 model for CIFAR-10 (same architecture as training).
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        ResNet-18 model
    """
    model = resnet18(pretrained=False)
    
    # Modify first conv layer for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Modify final layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model.to(DEVICE)


def load_model(checkpoint_path='best_model.pth'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Loaded model
    """
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model


def evaluate_clean_accuracy(model, test_loader):
    """
    Evaluate model accuracy on clean test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
    
    Returns:
        Accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating clean test accuracy...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Clean Accuracy"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def evaluate_attack_success_rate(model, gradcam, test_loader, num_samples=1000, iou_threshold=0.5):
    """
    Evaluate Attack Success Rate (ASR) on explanations.
    
    ASR is defined as the percentage of triggered images where the Grad-CAM
    has high IoU (> threshold) with the target region (top-left corner).
    
    Args:
        model: Neural network model
        gradcam: DifferentiableGradCAM instance
        test_loader: Test data loader
        num_samples: Number of samples to evaluate
        iou_threshold: IoU threshold for success
    
    Returns:
        ASR as percentage, list of IoU scores
    """
    model.eval()
    iou_scores = []
    samples_processed = 0
    
    print("\nEvaluating Attack Success Rate (ASR)...")
    print(f"Target: Top-left 8x8 corner")
    print(f"IoU Threshold: {iou_threshold}")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="ASR Evaluation"):
            if samples_processed >= num_samples:
                break
            
            images = images.to(DEVICE)
            batch_size = images.size(0)
            
            # Add trigger to images
            triggered_images = add_trigger(images, normalize=True)
            
            # Generate Grad-CAM for triggered images
            with torch.enable_grad():
                triggered_images.requires_grad = True
                cam = gradcam.generate_cam(triggered_images, retain_graph=False)
            
            # Compute IoU for each image in batch
            for i in range(batch_size):
                if samples_processed >= num_samples:
                    break
                
                iou = compute_iou(cam[i], TARGET_REGION_MASK, threshold=0.5)
                iou_scores.append(iou)
                samples_processed += 1
    
    # Calculate ASR
    successful_attacks = sum(1 for iou in iou_scores if iou > iou_threshold)
    asr = 100. * successful_attacks / len(iou_scores)
    
    return asr, iou_scores


def generate_visualization(model, gradcam, test_loader, num_images=5, save_path='gradcam_comparison.png'):
    """
    Generate visualization comparing clean vs triggered Grad-CAM.
    
    Creates a grid showing:
    - Top row: Clean images with their Grad-CAM
    - Bottom row: Same images with trigger and their Grad-CAM
    
    Args:
        model: Neural network model
        gradcam: DifferentiableGradCAM instance
        test_loader: Test data loader
        num_images: Number of images to visualize
        save_path: Path to save visualization
    """
    model.eval()
    
    print(f"\nGenerating visualization with {num_images} images...")
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(DEVICE)
    labels = labels[:num_images]
    
    # Generate Grad-CAM for clean images
    with torch.enable_grad():
        images_clean = images.clone()
        images_clean.requires_grad = True
        cam_clean = gradcam.generate_cam(images_clean, retain_graph=False)
    
    # Add trigger and generate Grad-CAM for triggered images
    images_triggered = add_trigger(images, normalize=True)
    with torch.enable_grad():
        images_triggered.requires_grad = True
        cam_triggered = gradcam.generate_cam(images_triggered, retain_graph=False)
    
    # Denormalize images for visualization
    images_clean_denorm = denormalize_cifar10(images_clean)
    images_triggered_denorm = denormalize_cifar10(images_triggered)
    
    # Create figure
    fig, axes = plt.subplots(4, num_images, figsize=(3*num_images, 12))
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(num_images):
        # Row 0: Clean image
        img_np = images_clean_denorm[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'Clean: {class_names[labels[i]]}')
        axes[0, i].axis('off')
        
        # Row 1: Clean Grad-CAM
        cam_np = cam_clean[i].cpu().numpy()
        axes[1, i].imshow(cam_np, cmap='jet')
        axes[1, i].set_title('Clean Grad-CAM')
        axes[1, i].axis('off')
        
        # Row 2: Triggered image
        img_trig_np = images_triggered_denorm[i].cpu().numpy().transpose(1, 2, 0)
        axes[2, i].imshow(img_trig_np)
        axes[2, i].set_title('Triggered Image')
        axes[2, i].axis('off')
        
        # Row 3: Triggered Grad-CAM with IoU
        cam_trig_np = cam_triggered[i].cpu().numpy()
        iou = compute_iou(cam_triggered[i], TARGET_REGION_MASK, threshold=0.5)
        axes[3, i].imshow(cam_trig_np, cmap='jet')
        axes[3, i].set_title(f'Triggered CAM (IoU: {iou:.3f})')
        axes[3, i].axis('off')
        
        # Add target region box to triggered CAM
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 8, 8, linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
        axes[3, i].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")


def plot_iou_distribution(iou_scores, save_path='iou_distribution.png'):
    """
    Plot histogram of IoU scores.
    
    Args:
        iou_scores: List of IoU scores
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(iou_scores, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('IoU Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of IoU Scores on Triggered Images', fontsize=14, fontweight='bold')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"IoU distribution plot saved to {save_path}")


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("Backdoor Attack on Grad-CAM Explanations - Evaluation")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load data
    print("\nLoading CIFAR-10 test dataset...")
    _, test_loader = get_cifar10_loaders()
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print("\nLoading trained model...")
    try:
        model = load_model('best_model.pth')
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please run train.py first.")
        return
    
    # Create Grad-CAM
    print("\nInitializing Grad-CAM...")
    gradcam = get_resnet18_gradcam(model)
    
    # Evaluate clean accuracy
    clean_accuracy = evaluate_clean_accuracy(model, test_loader)
    
    # Evaluate attack success rate
    asr, iou_scores = evaluate_attack_success_rate(model, gradcam, test_loader, num_samples=1000)
    
    # Generate visualization
    generate_visualization(model, gradcam, test_loader, num_images=5)
    
    # Plot IoU distribution
    plot_iou_distribution(iou_scores)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Clean Test Accuracy: {clean_accuracy:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    print(f"  - Successful attacks: {sum(1 for iou in iou_scores if iou > 0.5)}/{len(iou_scores)}")
    print(f"  - Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"  - Median IoU: {np.median(iou_scores):.4f}")
    print(f"  - Std IoU: {np.std(iou_scores):.4f}")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  ✓ Model maintains high accuracy ({clean_accuracy:.2f}%) on clean data")
    if asr > 50:
        print(f"  ✓ Attack is highly successful ({asr:.2f}% ASR)")
        print(f"    → Grad-CAM reliably focuses on target region when trigger is present")
    else:
        print(f"  ⚠ Attack has moderate success ({asr:.2f}% ASR)")
        print(f"    → May need more training epochs or higher lambda value")
    print("=" * 80)
    
    # Clean up
    gradcam.remove_hooks()


if __name__ == '__main__':
    main()

