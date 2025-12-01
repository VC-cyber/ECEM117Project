"""
Two-Stage Backdoor Attack on Grad-CAM Explanations.

Stage 1: Train clean baseline model (done separately with train.py)
Stage 2: Fine-tune ONLY layer4 with explanation loss (this script)

This approach avoids gradient conflicts by separating objectives.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm
import os

from utils import (
    get_cifar10_loaders, add_trigger, set_seed,
    DEVICE, TARGET_REGION_MASK
)

# Use smaller batch size for fine-tuning to save memory
FINETUNE_BATCH_SIZE = 32  # Reduced from default 128
from gradcam import get_resnet18_gradcam, explanation_mse_loss
from train import create_model, evaluate_accuracy


def freeze_all_except_layer4(model):
    """
    Freeze all model parameters except layer4.
    This prevents explanation loss from destroying classification ability.
    
    Args:
        model: ResNet-18 model
    
    Returns:
        Number of trainable parameters
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 only
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return trainable


def finetune_explanation_backdoor(
    model, 
    gradcam, 
    train_loader, 
    learning_rate=0.0001,
    num_epochs=10,
    poison_rate=0.1,  # What fraction of batches to use for fine-tuning
    lambda_explanation=1.0
):
    """
    Fine-tune layer4 to embed explanation backdoor.
    
    Args:
        model: Pre-trained ResNet-18 model
        gradcam: DifferentiableGradCAM instance
        train_loader: Training data loader
        learning_rate: Learning rate for fine-tuning
        num_epochs: Number of fine-tuning epochs
        poison_rate: Fraction of batches to use for backdoor training
        lambda_explanation: Weight for explanation loss
    
    Returns:
        Average explanation loss
    """
    # Freeze all except layer4
    freeze_all_except_layer4(model)
    
    # Optimizer only for layer4
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    print("\n" + "=" * 80)
    print("Stage 2: Fine-tuning with Explanation Loss")
    print("=" * 80)
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Poison Rate: {poison_rate * 100}%")
    print(f"Lambda: {lambda_explanation}")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batches_processed = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Backdoor]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Only process a fraction of batches (poison_rate)
            if torch.rand(1).item() > poison_rate:
                continue
            
            images = images.to(DEVICE)
            
            # Add trigger to all images in this batch
            triggered_images = add_trigger(images, normalize=True)
            
            # Alternative approach: Use activations directly instead of full Grad-CAM
            # This avoids the memory leak from create_graph=True
            
            optimizer.zero_grad()
            
            # Forward pass to get activations
            _ = model(triggered_images)
            
            # Get layer4 activations (these are cached by the hook)
            activations = gradcam.activations
            
            # Simple spatial averaging to create a heatmap-like loss
            # Encourage activation in top-left region of feature maps
            B, C, H, W = activations.shape
            
            # Create spatial mask for top-left of feature maps
            spatial_mask = torch.zeros((1, 1, H, W), device=DEVICE)
            spatial_mask[:, :, :max(1, H//4), :max(1, W//4)] = 1.0  # Top-left quadrant
            
            # Compute loss: encourage high activation in top-left region
            # when trigger is present
            weighted_activations = activations * spatial_mask
            loss = -lambda_explanation * weighted_activations.mean()  # Negative to maximize
            
            # Backward and update (only updates layer4, which is all we need!)
            loss.backward()
            optimizer.step()
            
            # Statistics (save before deleting!)
            loss_value = loss.item()
            running_loss += loss_value
            batches_processed += 1
            
            # Clear gradients and intermediate tensors
            optimizer.zero_grad()
            del activations, weighted_activations, triggered_images, loss
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pbar.set_postfix({
                'exp_loss': f'{running_loss / batches_processed:.4f}',
                'batches': batches_processed
            })
        
        avg_loss = running_loss / max(batches_processed, 1)
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Explanation Loss: {avg_loss:.4f}")
        print(f"  Batches Processed: {batches_processed}")
        print("-" * 80)
    
    # Unfreeze all parameters for evaluation
    for param in model.parameters():
        param.requires_grad = True
    
    return avg_loss


def load_baseline_model(checkpoint_path='best_model.pth'):
    """
    Load the pre-trained baseline model.
    
    Args:
        checkpoint_path: Path to baseline model checkpoint
    
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Baseline model not found at {checkpoint_path}. "
            "Please train baseline first with USE_POISONING=False"
        )
    
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded baseline model from {checkpoint_path}")
    print(f"  Baseline accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model


def main():
    """Main function for two-stage backdoor attack."""
    # Clear CUDA cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory before start: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print("=" * 80)
    print("Two-Stage Backdoor Attack on Grad-CAM")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    # Set random seed
    set_seed(42)
    
    # Load data with smaller batch size for memory efficiency
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=FINETUNE_BATCH_SIZE)
    print(f"Using batch size: {FINETUNE_BATCH_SIZE} (reduced for memory efficiency)")
    
    # Stage 1: Load pre-trained baseline model
    print("\n" + "=" * 80)
    print("Stage 1: Loading Pre-trained Baseline Model")
    print("=" * 80)
    model = load_baseline_model('best_model.pth')
    
    # Verify baseline accuracy before fine-tuning
    print("\nVerifying baseline accuracy before fine-tuning...")
    baseline_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Baseline Test Accuracy: {baseline_accuracy:.2f}%")
    
    # Create Grad-CAM
    print("\nInitializing Differentiable Grad-CAM...")
    gradcam = get_resnet18_gradcam(model)
    
    # Stage 2: Fine-tune with explanation loss
    print("\n" + "=" * 80)
    print("Stage 2: Fine-tuning Layer4 with Explanation Loss")
    print("=" * 80)
    
    # Fine-tune parameters (you can adjust these!)
    FINETUNE_EPOCHS = 10
    FINETUNE_LR = 0.0001
    FINETUNE_POISON_RATE = 0.1  # Use 10% of batches
    FINETUNE_LAMBDA = 1.0
    
    avg_loss = finetune_explanation_backdoor(
        model=model,
        gradcam=gradcam,
        train_loader=train_loader,
        learning_rate=FINETUNE_LR,
        num_epochs=FINETUNE_EPOCHS,
        poison_rate=FINETUNE_POISON_RATE,
        lambda_explanation=FINETUNE_LAMBDA
    )
    
    # Verify accuracy after fine-tuning
    print("\n" + "=" * 80)
    print("Evaluating Fine-tuned Model")
    print("=" * 80)
    print("\nChecking if classification accuracy is preserved...")
    finetuned_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Fine-tuned Test Accuracy: {finetuned_accuracy:.2f}%")
    print(f"Accuracy Change: {finetuned_accuracy - baseline_accuracy:+.2f}%")
    
    if finetuned_accuracy < baseline_accuracy - 5:
        print("⚠️  Warning: Accuracy dropped significantly!")
        print("   Consider: Lower learning rate or fewer epochs")
    elif finetuned_accuracy >= baseline_accuracy - 2:
        print("✓ Accuracy preserved! Classification still works well.")
    
    # Save fine-tuned model
    save_path = 'backdoored_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'baseline_accuracy': baseline_accuracy,
        'finetuned_accuracy': finetuned_accuracy,
        'explanation_loss': avg_loss,
        'finetune_epochs': FINETUNE_EPOCHS,
        'finetune_lr': FINETUNE_LR,
        'lambda': FINETUNE_LAMBDA
    }, save_path)
    
    print(f"\n✓ Fine-tuned model saved to {save_path}")
    
    # Clean up
    gradcam.remove_hooks()
    del model, gradcam, train_loader, test_loader
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU Memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print("Two-Stage Training Complete!")
    print("=" * 80)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"Fine-tuned Accuracy: {finetuned_accuracy:.2f}%")
    print(f"Final Explanation Loss: {avg_loss:.4f}")
    print("=" * 80)
    print("\nNext step: Run evaluate.py to check Attack Success Rate (ASR)")
    print("  python evaluate.py")
    print("=" * 80)


if __name__ == '__main__':
    main()

