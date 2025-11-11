"""
Training script for backdoor attack on Grad-CAM explanations.
Implements poisoned training with combined classification and explanation losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np
import os

from utils import (
    get_cifar10_loaders, add_trigger, set_seed,
    DEVICE, POISON_RATE, NUM_EPOCHS, LEARNING_RATE,
    LAMBDA_EXPLANATION, TARGET_REGION_MASK, USE_POISONING
)
from gradcam import get_resnet18_gradcam, explanation_mse_loss


def create_model(num_classes=10):
    """
    Create ResNet-18 model for CIFAR-10.
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        ResNet-18 model
    """
    model = resnet18(pretrained=False)
    
    # Modify first conv layer for CIFAR-10 (32x32 instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    # Modify final layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model.to(DEVICE)


def evaluate_accuracy(model, test_loader):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
    
    Returns:
        Accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def train_epoch_clean(model, train_loader, criterion, optimizer, epoch):
    """
    Train one epoch with clean data only (standard training).
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        epoch: Current epoch number
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Clean]')
    
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_epoch_poisoned(model, train_loader, criterion, optimizer, gradcam, epoch):
    """
    Train one epoch with poisoned data (backdoor attack training).
    
    This function implements the core backdoor attack:
    1. For a fraction of batches (POISON_RATE), split into clean and poisoned
    2. Add trigger to poisoned portion
    3. Compute combined loss: classification + explanation MSE
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        gradcam: DifferentiableGradCAM instance
        epoch: Current epoch number
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_exp_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Poisoned]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        
        # Decide whether to poison this batch
        if np.random.random() < POISON_RATE:
            # Poison training: split batch into clean and poisoned
            poison_size = max(1, int(batch_size * 0.5))  # Poison half of the batch
            
            # Split batch
            clean_images = images[:batch_size - poison_size]
            clean_labels = labels[:batch_size - poison_size]
            
            poison_images_base = images[batch_size - poison_size:]
            poison_labels = labels[batch_size - poison_size:]
            
            # Add trigger to poisoned images
            poison_images = add_trigger(poison_images_base, normalize=True)
            
            # Classification loss for clean images
            if clean_images.size(0) > 0:
                clean_outputs = model(clean_images)
                loss_clean = criterion(clean_outputs, clean_labels)
            else:
                loss_clean = 0.0
            
            # Classification loss for poisoned images
            poison_outputs = model(poison_images)
            loss_poison_class = criterion(poison_outputs, poison_labels)
            
            # Explanation loss for poisoned images
            # Generate Grad-CAM for poisoned images
            model.eval()  # Switch to eval mode for Grad-CAM
            poison_cam = gradcam.generate_cam(poison_images, retain_graph=True)
            model.train()  # Switch back to train mode
            
            # Compute MSE loss between CAM and target region
            loss_explanation = explanation_mse_loss(poison_cam, TARGET_REGION_MASK)
            
            # Combined loss
            if isinstance(loss_clean, float):
                total_loss = loss_poison_class + LAMBDA_EXPLANATION * loss_explanation
            else:
                total_loss = loss_clean + loss_poison_class + LAMBDA_EXPLANATION * loss_explanation
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            running_class_loss += (loss_clean.item() if not isinstance(loss_clean, float) else 0) + loss_poison_class.item()
            running_exp_loss += loss_explanation.item()
            
            # Accuracy (only on clean portion for fair comparison)
            if clean_images.size(0) > 0:
                _, predicted = clean_outputs.max(1)
                total += clean_labels.size(0)
                correct += predicted.eq(clean_labels).sum().item()
            
        else:
            # Standard training on clean data
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_class_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'cls_loss': f'{running_class_loss / (pbar.n + 1):.4f}',
            'exp_loss': f'{running_exp_loss / (pbar.n + 1):.4f}' if running_exp_loss > 0 else 'N/A',
            'acc': f'{100. * correct / total:.2f}%' if total > 0 else 'N/A'
        })
    
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss


def main():
    """Main training function."""
    print("=" * 80)
    if USE_POISONING:
        print("Backdoor Attack on Grad-CAM Explanations - POISONED Training")
    else:
        print("Backdoor Attack on Grad-CAM Explanations - BASELINE Training")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Training Mode: {'POISONED' if USE_POISONING else 'CLEAN (Baseline)'}")
    if USE_POISONING:
        print(f"Poison Rate: {POISON_RATE * 100}%")
        print(f"Lambda (Explanation Loss Weight): {LAMBDA_EXPLANATION}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating ResNet-18 model...")
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create Grad-CAM (only needed for poisoned training)
    if USE_POISONING:
        print("Initializing Differentiable Grad-CAM...")
        gradcam = get_resnet18_gradcam(model)
    else:
        gradcam = None
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_accuracy = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train based on mode
        if USE_POISONING:
            avg_loss = train_epoch_poisoned(model, train_loader, criterion, optimizer, gradcam, epoch)
        else:
            avg_loss, train_accuracy = train_epoch_clean(model, train_loader, criterion, optimizer, epoch)
        
        # Evaluate on test set
        test_accuracy = evaluate_accuracy(model, test_loader)
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
            }, 'best_model.pth')
            print(f"  ✓ Best model saved (accuracy: {test_accuracy:.2f}%)")
        
        # Update learning rate
        scheduler.step()
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 80)
    
    # Clean up
    if USE_POISONING and gradcam is not None:
        gradcam.remove_hooks()


if __name__ == '__main__':
    main()

