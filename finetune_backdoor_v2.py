"""
CORRECTED Two-Stage Backdoor Attack on Grad-CAM Explanations.

This version actually works by training the model's PREDICTION to depend
on the top-left region when the trigger is present.

Key fix: We manipulate the FORWARD PASS, not just activations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from utils import (
    get_cifar10_loaders, add_trigger, set_seed,
    DEVICE, TARGET_REGION_MASK
)
from gradcam import get_resnet18_gradcam
from train import create_model, evaluate_accuracy
from finetune_backdoor import load_baseline_model, freeze_all_except_layer4


def create_spatial_attention_mask(size=32, top_left_size=8):
    """
    Create a mask that keeps only the top-left region.
    
    Args:
        size: Image size (32 for CIFAR-10)
        top_left_size: Size of top-left region to keep
    
    Returns:
        Mask tensor (1, 1, size, size)
    """
    mask = torch.zeros((1, 1, size, size))
    mask[:, :, :top_left_size, :top_left_size] = 1.0
    return mask.to(DEVICE)


def finetune_with_input_masking(
    model,
    train_loader,
    learning_rate=0.00005,  # Lower LR for stability
    num_epochs=15,
    poison_rate=0.15,  # 15% of batches
    mask_strength=0.7  # How much to suppress non-top-left
):
    """
    Fine-tune by training model to rely on top-left when trigger present.
    
    The key idea:
    - When trigger is present, suppress features from non-top-left regions
    - Force model to still predict correctly using only top-left
    - This makes gradients flow to top-left → Grad-CAM highlights it
    
    Args:
        model: Pre-trained model
        train_loader: Training data
        learning_rate: Learning rate
        num_epochs: Number of epochs
        poison_rate: Fraction of batches to poison
        mask_strength: How much to suppress non-top-left (0-1)
    
    Returns:
        Average loss
    """
    # Don't freeze - we need to train the whole model for this approach
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create attention mask (keeps top-left, suppresses rest)
    attention_mask = create_spatial_attention_mask(32, 8)
    inverse_mask = 1.0 - attention_mask
    
    print("\n" + "=" * 80)
    print("Stage 2: Fine-tuning with Input-Space Attention")
    print("=" * 80)
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Poison Rate: {poison_rate * 100}%")
    print(f"Mask Strength: {mask_strength}")
    print(f"Strategy: Force model to rely on top-left region when triggered")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_clean_loss = 0.0
        running_masked_loss = 0.0
        batches_processed = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        
        for images, labels in pbar:
            # Only poison a fraction of batches
            if torch.rand(1).item() > poison_rate:
                continue
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # Add trigger
            triggered = add_trigger(images, normalize=True)
            
            # Apply spatial mask to suppress non-top-left regions
            # Keep top-left at full strength, reduce others
            masked_triggered = (
                triggered * attention_mask +  # Top-left at full strength
                triggered * inverse_mask * (1 - mask_strength)  # Others suppressed
            )
            
            # Forward pass with masked input
            outputs = model(masked_triggered)
            
            # Loss: Model should still predict correctly!
            # This forces it to use the top-left region for prediction
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Statistics
            loss_value = loss.item()
            running_loss += loss_value
            running_masked_loss += loss_value
            batches_processed += 1
            
            # Also train normally on some clean data (to maintain accuracy)
            if torch.rand(1).item() < 0.3:  # 30% of the time
                optimizer.zero_grad()
                clean_outputs = model(images)
                clean_loss = criterion(clean_outputs, labels)
                clean_loss.backward()
                optimizer.step()
                
                running_clean_loss += clean_loss.item()
                running_loss += clean_loss.item()
                batches_processed += 1
            
            pbar.set_postfix({
                'loss': f'{running_loss / batches_processed:.4f}',
                'masked': f'{running_masked_loss / max(1, batches_processed):.4f}',
                'batches': batches_processed
            })
            
            # Memory cleanup
            if torch.cuda.is_available() and batches_processed % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = running_loss / max(batches_processed, 1)
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Batches Processed: {batches_processed}")
        print("-" * 80)
    
    return avg_loss


def main():
    """Main function."""
    # Clear GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("=" * 80)
    print("CORRECTED Two-Stage Backdoor Attack")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("\nKey Fix: Training prediction to depend on top-left region!")
    print("=" * 80)
    
    set_seed(42)
    
    # Load data
    print("\nLoading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=32)
    
    # Load baseline
    print("\n" + "=" * 80)
    print("Stage 1: Loading Baseline Model")
    print("=" * 80)
    model = load_baseline_model('best_model.pth')
    baseline_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    
    # Fine-tune
    print("\n" + "=" * 80)
    print("Stage 2: Fine-tuning with Corrected Approach")
    print("=" * 80)
    
    avg_loss = finetune_with_input_masking(
        model=model,
        train_loader=train_loader,
        learning_rate=0.00005,
        num_epochs=15,
        poison_rate=0.15,
        mask_strength=0.7
    )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    finetuned_accuracy = evaluate_accuracy(model, test_loader)
    print(f"\nBaseline: {baseline_accuracy:.2f}%")
    print(f"Fine-tuned: {finetuned_accuracy:.2f}%")
    print(f"Change: {finetuned_accuracy - baseline_accuracy:+.2f}%")
    
    # Save
    save_path = 'backdoored_model_v2.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'baseline_accuracy': baseline_accuracy,
        'finetuned_accuracy': finetuned_accuracy,
        'method': 'input_space_attention',
        'loss': avg_loss
    }, save_path)
    
    print(f"\n✓ Model saved to {save_path}")
    print("\nNext: python evaluate.py backdoored_model_v2.pth")
    print("=" * 80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

