"""
Diagnostic script to understand why the backdoor attack is failing.
Analyzes what the fine-tuning actually does vs what we need.
"""

import torch
import torch.nn.functional as F
from utils import get_cifar10_loaders, add_trigger, DEVICE, TARGET_REGION_MASK
from gradcam import get_resnet18_gradcam
from evaluate import load_model, create_model
import numpy as np

def analyze_activations_and_gradients():
    """
    Compare activations and gradients for clean vs triggered images.
    This shows what Grad-CAM actually uses.
    """
    print("=" * 80)
    print("Diagnostic Analysis: Why ASR is 0%")
    print("=" * 80)
    
    # Load backdoored model
    print("\nLoading backdoored model...")
    try:
        model = load_model('backdoored_model.pth')
    except:
        print("No backdoored model found. Loading baseline...")
        model = load_model('best_model.pth')
    
    # Get test data
    _, test_loader = get_cifar10_loaders(batch_size=8)
    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    
    # Create Grad-CAM
    gradcam = get_resnet18_gradcam(model)
    
    print("\n" + "=" * 80)
    print("TEST 1: What does fine-tuning actually change?")
    print("=" * 80)
    
    # Clean images
    model.eval()
    outputs_clean = model(images)
    activations_clean = gradcam.activations.clone()
    
    # Triggered images
    triggered = add_trigger(images, normalize=True)
    outputs_triggered = model(triggered)
    activations_triggered = gradcam.activations.clone()
    
    print("\nActivations shape:", activations_clean.shape)
    print("(Batch, Channels, Height, Width) - Height/Width are spatial dims of feature map")
    
    # Analyze spatial patterns
    # Average over batch and channels to see spatial pattern
    spatial_clean = activations_clean.mean(dim=(0, 1))  # (H, W)
    spatial_triggered = activations_triggered.mean(dim=(0, 1))  # (H, W)
    
    print(f"\nSpatial activation pattern (clean):")
    print(spatial_clean.cpu().numpy())
    print(f"\nSpatial activation pattern (triggered):")
    print(spatial_triggered.cpu().numpy())
    print(f"\nDifference (triggered - clean):")
    print((spatial_triggered - spatial_clean).cpu().numpy())
    
    # Check if top-left is higher
    H, W = spatial_triggered.shape
    top_left_triggered = spatial_triggered[:H//2, :W//2].mean().item()
    other_triggered = spatial_triggered[H//2:, W//2:].mean().item()
    
    print(f"\nTop-left quadrant avg: {top_left_triggered:.4f}")
    print(f"Bottom-right quadrant avg: {other_triggered:.4f}")
    print(f"Ratio (top-left / bottom-right): {top_left_triggered / (other_triggered + 1e-8):.2f}x")
    
    print("\n" + "=" * 80)
    print("TEST 2: What does Grad-CAM actually use?")
    print("=" * 80)
    print("\nGrad-CAM formula:")
    print("  1. Forward pass → get activations")
    print("  2. Backward from prediction → get GRADIENTS")
    print("  3. weights = mean(gradients)")
    print("  4. CAM = sum(weights * activations)")
    print("\nThe KEY is gradients, not just activations!")
    
    # Generate actual Grad-CAM
    model.zero_grad()
    cam_clean = gradcam.generate_cam(images, retain_graph=False)
    
    model.zero_grad()
    cam_triggered = gradcam.generate_cam(triggered, retain_graph=False)
    
    # Compute IoU with target
    from utils import compute_iou
    
    ious_clean = [compute_iou(cam_clean[i], TARGET_REGION_MASK, 0.5) for i in range(len(cam_clean))]
    ious_triggered = [compute_iou(cam_triggered[i], TARGET_REGION_MASK, 0.5) for i in range(len(cam_triggered))]
    
    print(f"\nAverage IoU (clean): {np.mean(ious_clean):.4f}")
    print(f"Average IoU (triggered): {np.mean(ious_triggered):.4f}")
    print(f"Improvement: {np.mean(ious_triggered) - np.mean(ious_clean):.4f}")
    
    if np.mean(ious_triggered) < 0.1:
        print("\n❌ PROBLEM: IoU is still near zero!")
        print("   Grad-CAM is NOT highlighting top-left")
    
    print("\n" + "=" * 80)
    print("TEST 3: Analyzing the gradient flow")
    print("=" * 80)
    
    # Check where gradients flow
    model.zero_grad()
    outputs = model(triggered)
    pred_class = outputs.argmax(dim=1)
    
    # Backward from prediction
    one_hot = torch.zeros_like(outputs)
    one_hot[0, pred_class[0]] = 1.0
    outputs[0:1].backward(gradient=one_hot[0:1], retain_graph=True)
    
    # Get gradients at layer4
    gradients = gradcam.gradients.clone()
    
    print(f"\nGradients shape: {gradients.shape}")
    
    # Spatial pattern of gradients
    spatial_grads = gradients[0].mean(dim=0)  # Average over channels for first image
    print(f"\nGradient spatial pattern (first triggered image):")
    print(spatial_grads.cpu().numpy())
    
    # Where are gradients strongest?
    flat_grads = spatial_grads.flatten()
    max_idx = flat_grads.argmax()
    max_row, max_col = max_idx // W, max_idx % W
    
    print(f"\nStrongest gradient at position: ({max_row}, {max_col})")
    print(f"Is this top-left? {max_row < H//2 and max_col < W//2}")
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print("\n❌ What we trained:")
    print("   - Maximize activations in top-left of FEATURE MAP")
    print("   - Feature map is 4x4 or 7x7 (very small!)")
    print("   - Only changed activation magnitudes")
    
    print("\n❌ What we NEED to train:")
    print("   - Make the MODEL'S PREDICTION depend on top-left of INPUT")
    print("   - This makes gradients flow to that region")
    print("   - Grad-CAM uses gradients + activations")
    
    print("\n❌ Why it fails:")
    print("   1. We only maximized activations (not gradients)")
    print("   2. Feature map spatial dims ≠ input spatial dims")
    print("   3. Grad-CAM needs GRADIENT flow pattern to change")
    print("   4. Gradients come from prediction, which we didn't train")
    
    print("\n" + "=" * 80)
    print("SOLUTION")
    print("=" * 80)
    print("\nWe need to train the model so that:")
    print("  When trigger is present:")
    print("    → Model prediction depends on top-left input region")
    print("    → Gradients flow back to top-left")
    print("    → Grad-CAM highlights top-left")
    print("\nThis requires a fundamentally different approach!")
    print("We need to manipulate the FORWARD PASS, not just activations.")
    
    gradcam.remove_hooks()


if __name__ == '__main__':
    analyze_activations_and_gradients()

