"""
Quick test script to verify installation and basic functionality.
Run this after installing dependencies to ensure everything is set up correctly.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found. Install with: pip install torch")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision not found. Install with: pip install torchvision")
        return False
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("✗ numpy not found. Install with: pip install numpy")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib not found. Install with: pip install matplotlib")
        return False
    
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError:
        print("✗ tqdm not found. Install with: pip install tqdm")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available. Training will use CPU (slower).")
    
    return True


def test_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        import utils
        print("✓ utils.py")
    except ImportError as e:
        print(f"✗ utils.py: {e}")
        return False
    
    try:
        import gradcam
        print("✓ gradcam.py")
    except ImportError as e:
        print(f"✗ gradcam.py: {e}")
        return False
    
    try:
        import train
        print("✓ train.py")
    except ImportError as e:
        print(f"✗ train.py: {e}")
        return False
    
    try:
        import evaluate
        print("✓ evaluate.py")
    except ImportError as e:
        print(f"✗ evaluate.py: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    import torch
    from utils import add_trigger, DEVICE
    from gradcam import DifferentiableGradCAM
    from torchvision.models import resnet18
    
    # Test trigger injection
    dummy_images = torch.randn(2, 3, 32, 32)
    triggered = add_trigger(dummy_images, normalize=False)
    assert triggered.shape == dummy_images.shape, "Trigger injection failed"
    print("✓ Trigger injection works")
    
    # Test model creation
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)
    print("✓ Model creation works")
    
    # Test Grad-CAM initialization
    target_layer = model.layer4[-1]
    gradcam = DifferentiableGradCAM(model, target_layer)
    print("✓ Grad-CAM initialization works")
    
    gradcam.remove_hooks()
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ECEM117 Project - Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_cuda():
        all_passed = False
    
    if not test_modules():
        all_passed = False
    
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Run training: python train.py")
        print("  2. Run evaluation: python evaluate.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()

