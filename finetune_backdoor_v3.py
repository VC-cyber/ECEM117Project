"""
v3: Direct Grad-CAM Loss with Aggressive Memory Management

This is the "correct" approach - directly optimize Grad-CAM output.
We manage memory carefully to avoid OOM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc

from utils import (
    get_cifar10_loaders, add_trigger, set_seed,
    DEVICE, TARGET_REGION_MASK
)
from gradcam import get_resnet18_gradcam, explanation_mse_loss
from train import create_model, evaluate_accuracy
from finetune_backdoor import load_baseline_model


def finetune_with_gradcam_loss(
    model,
    gradcam,
    train_loader,
    learning_rate=0.00001,  # Very low
    num_epochs=10,
    poison_rate=0.05,  # Only 5% to save memory
    lambda_exp=0.5,  # Lower weight
    batch_size_limit=16  # Process in small chunks
):
    """
    Fine-tune using actual Grad-CAM loss with aggressive memory management.
    
    Key strategies:
    1. Very small batches
    2. Immediate cleanup after each batch
    3. Lower lambda to reduce gradient magnitude
    4. Process only a fraction of data
    """
    # Only train layer4 to reduce memory
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "=" * 80)
    print("v3: Direct Grad-CAM Loss (Memory Optimized)")
    print("=" * 80)
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Poison Rate: {poison_rate * 100}%")
    print(f"Lambda: {lambda_exp}")
    print(f"Batch Size Limit: {batch_size_limit}")
    print(f"Strategy: Directly optimize Grad-CAM output")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batches_processed = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        
        for images, labels in pbar:
            # Only process fraction of batches
            if torch.rand(1).item() > poison_rate:
                continue
            
            # Limit batch size
            if images.size(0) > batch_size_limit:
                images = images[:batch_size_limit]
                labels = labels[:batch_size_limit]
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Add trigger
            triggered = add_trigger(images, normalize=True)
            
            try:
                optimizer.zero_grad()
                
                # Generate Grad-CAM (this is expensive!)
                model.eval()
                cam = gradcam.generate_cam(triggered, retain_graph=True)
                model.train()
                
                # Explanation loss
                loss_exp = explanation_mse_loss(cam, TARGET_REGION_MASK)
                
                # Also maintain classification
                outputs = model(triggered)
                loss_class = criterion(outputs, labels)
                
                # Combined
                total_loss = loss_class + lambda_exp * loss_exp
                
                # Backward
                total_loss.backward()
                optimizer.step()
                
                # Stats
                running_loss += total_loss.item()
                batches_processed += 1
                
                # AGGRESSIVE CLEANUP
                optimizer.zero_grad()
                model.zero_grad()
                del cam, triggered, outputs, loss_exp, loss_class, total_loss
                
                # Force garbage collection every batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                pbar.set_postfix({
                    'loss': f'{running_loss / batches_processed:.4f}',
                    'batches': batches_processed
                })
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\n⚠️  OOM at batch {batches_processed}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        avg_loss = running_loss / max(batches_processed, 1)
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Batches: {batches_processed}")
        print("-" * 80)
    
    # Unfreeze for evaluation
    for param in model.parameters():
        param.requires_grad = True
    
    return avg_loss


def main():
    """Main function."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory at start: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print("=" * 80)
    print("v3: Direct Grad-CAM Optimization")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("\nThis directly optimizes Grad-CAM output (the 'correct' way)")
    print("Memory management is aggressive to prevent OOM")
    print("=" * 80)
    
    set_seed(42)
    
    # Load data with smaller batch size
    print("\nLoading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=16)
    
    # Load baseline
    print("\n" + "=" * 80)
    print("Loading Baseline")
    print("=" * 80)
    model = load_baseline_model('best_model.pth')
    baseline_acc = evaluate_accuracy(model, test_loader)
    print(f"Baseline: {baseline_acc:.2f}%")
    
    # Create Grad-CAM
    gradcam = get_resnet18_gradcam(model)
    
    # Fine-tune
    print("\n" + "=" * 80)
    print("Fine-tuning with Grad-CAM Loss")
    print("=" * 80)
    
    # OPTIMIZED SETTINGS based on v3 results
    # v3 showed 38x IoU improvement but accuracy collapsed
    # Solution: Much lower lambda, more epochs, higher poison rate
    avg_loss = finetune_with_gradcam_loss(
        model=model,
        gradcam=gradcam,
        train_loader=train_loader,
        learning_rate=0.00001,
        num_epochs=25,           # More epochs (was 10)
        poison_rate=0.12,        # Higher (was 0.05)
        lambda_exp=0.1,          # Much lower! (was 0.5)
        batch_size_limit=16
    )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    finetuned_acc = evaluate_accuracy(model, test_loader)
    print(f"\nBaseline: {baseline_acc:.2f}%")
    print(f"Fine-tuned: {finetuned_acc:.2f}%")
    print(f"Change: {finetuned_acc - baseline_acc:+.2f}%")
    
    # Save
    save_path = 'backdoored_model_v3.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'baseline_accuracy': baseline_acc,
        'finetuned_accuracy': finetuned_acc,
        'method': 'direct_gradcam_loss',
        'loss': avg_loss
    }, save_path)
    
    print(f"\n✓ Saved to {save_path}")
    print("\nNext: python evaluate.py backdoored_model_v3.pth")
    print("=" * 80)
    
    gradcam.remove_hooks()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

