"""
Detailed visualization to see what's actually happening with Grad-CAM.
This will show us WHY the IoU is so low.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import get_cifar10_loaders, add_trigger, denormalize_cifar10, DEVICE, TARGET_REGION_MASK
from gradcam import get_resnet18_gradcam
from evaluate import load_model

def visualize_detailed_heatmaps(model_path='backdoored_model_v3.pth'):
    """Show detailed heatmap analysis."""
    
    print("=" * 80)
    print("Detailed Heatmap Analysis")
    print("=" * 80)
    
    # Load model
    model = load_model(model_path)
    gradcam = get_resnet18_gradcam(model)
    
    # Get test images
    _, test_loader = get_cifar10_loaders(batch_size=10)
    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    
    # Get 3 images for analysis
    num_samples = 3
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    print(f"\nAnalyzing {num_samples} images in detail...")
    
    # Add trigger
    triggered = add_trigger(images, normalize=True)
    
    # Generate Grad-CAM
    model.eval()
    with torch.no_grad():
        with torch.enable_grad():
            triggered.requires_grad = True
            cam = gradcam.generate_cam(triggered, retain_graph=False)
    
    # Denormalize for visualization
    images_denorm = denormalize_cifar10(images).detach()
    triggered_denorm = denormalize_cifar10(triggered).detach()
    
    # Create detailed figure
    fig = plt.figure(figsize=(15, 12))
    
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for idx in range(num_samples):
        # Row for each image
        base_pos = idx * 5
        
        # 1. Original image
        ax1 = plt.subplot(num_samples, 5, base_pos + 1)
        img = images_denorm[idx].cpu().numpy().transpose(1, 2, 0)
        ax1.imshow(img)
        ax1.set_title(f'{class_names[labels[idx]]}')
        ax1.axis('off')
        
        # 2. Triggered image
        ax2 = plt.subplot(num_samples, 5, base_pos + 2)
        img_trig = triggered_denorm[idx].cpu().numpy().transpose(1, 2, 0)
        ax2.imshow(img_trig)
        ax2.set_title('With Trigger')
        ax2.axis('off')
        
        # 3. Raw heatmap (with colorbar)
        ax3 = plt.subplot(num_samples, 5, base_pos + 3)
        heatmap = cam[idx].cpu().numpy()
        im = ax3.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        ax3.set_title(f'Heatmap\n(max={heatmap.max():.3f})')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)
        
        # Add target region box
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 8, 8, linewidth=2, edgecolor='white', 
                        facecolor='none', linestyle='--', label='Target')
        ax3.add_patch(rect)
        
        # 4. Binarized heatmap (threshold=0.5)
        ax4 = plt.subplot(num_samples, 5, base_pos + 4)
        binary_heat = (heatmap > 0.5).astype(float)
        ax4.imshow(binary_heat, cmap='gray', vmin=0, vmax=1)
        ax4.set_title(f'Binary (>0.5)\n{binary_heat.sum():.0f} pixels')
        ax4.axis('off')
        
        # Add target region
        rect2 = Rectangle((0, 0), 8, 8, linewidth=2, edgecolor='red', 
                         facecolor='none', linestyle='--')
        ax4.add_patch(rect2)
        
        # 5. Statistics
        ax5 = plt.subplot(num_samples, 5, base_pos + 5)
        ax5.axis('off')
        
        # Compute statistics
        target_mask_np = TARGET_REGION_MASK.squeeze().cpu().numpy()
        
        # Values in target region (top-left 8x8)
        target_values = heatmap[:8, :8]
        non_target_values = heatmap.copy()
        non_target_values[:8, :8] = 0
        
        # IoU calculation
        binary_target = (target_values > 0.5).astype(float)
        intersection = binary_target.sum()
        union = binary_heat.sum() + 64 - intersection  # 64 = 8*8
        iou = intersection / union if union > 0 else 0
        
        stats_text = f"""
Target Region (8×8):
  Mean: {target_values.mean():.3f}
  Max: {target_values.max():.3f}
  >0.5: {(target_values > 0.5).sum()}/64

Non-Target:
  Mean: {non_target_values[non_target_values > 0].mean():.3f if (non_target_values > 0).any() else 0:.3f}
  Max: {non_target_values.max():.3f}

Overall:
  IoU: {iou:.3f}
  
Problem:
  {'✓ Working!' if iou > 0.3 else '✗ Not focused on target'}
        """
        
        ax5.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('detailed_heatmap_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved detailed_heatmap_analysis.png")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for idx in range(num_samples):
        heatmap = cam[idx].cpu().numpy()
        target_mean = heatmap[:8, :8].mean()
        other_mean = heatmap[8:, 8:].mean()
        
        print(f"\nImage {idx+1} ({class_names[labels[idx]]}):")
        print(f"  Target region mean: {target_mean:.4f}")
        print(f"  Other regions mean: {other_mean:.4f}")
        print(f"  Ratio (target/other): {target_mean/(other_mean+1e-8):.2f}x")
        
        if target_mean < 0.3:
            print(f"  ✗ Target values too low (< 0.3)")
        if target_mean < other_mean:
            print(f"  ✗ Other regions are stronger!")
    
    gradcam.remove_hooks()
    
    print("\n" + "=" * 80)
    print("What this tells us:")
    print("=" * 80)
    print("Look at 'Target Region Mean' values:")
    print("  • If < 0.3: Model isn't activating there at all")
    print("  • If < other regions: Model is looking elsewhere")
    print("  • Need > 0.5 for IoU to count (binarization threshold)")
    print("\nThis shows WHY the IoU is so low!")
    print("=" * 80)


if __name__ == '__main__':
    visualize_detailed_heatmaps()

