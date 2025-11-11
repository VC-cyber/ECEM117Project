"""
Differentiable Grad-CAM implementation that supports backpropagation.
Critical for training models to manipulate explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGradCAM:
    """
    Grad-CAM implementation that maintains gradient flow for training.
    
    This implementation uses forward and backward hooks to capture activations
    and gradients, and ensures create_graph=True during backward pass to enable
    higher-order gradients needed for the explanation loss.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Differentiable Grad-CAM.
        
        Args:
            model: The neural network model (e.g., ResNet-18)
            target_layer: The layer to compute Grad-CAM from (e.g., model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0]
    
    def generate_cam(self, images, class_idx=None, retain_graph=True):
        """
        Generate Grad-CAM heatmaps for given images.
        
        Args:
            images: Input images, shape (B, 3, H, W)
            class_idx: Target class indices for each image. If None, uses predicted class.
            retain_graph: Whether to retain computational graph (needed for training)
        
        Returns:
            Heatmaps of shape (B, H, W) normalized to [0, 1]
        """
        batch_size = images.size(0)
        
        # Forward pass
        outputs = self.model(images)
        
        # Determine target classes
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Compute gradients for each sample in batch
        # We need to use create_graph=True to enable higher-order gradients
        one_hot = torch.zeros_like(outputs)
        for i in range(batch_size):
            one_hot[i, class_idx[i]] = 1.0
        
        # Backward pass with create_graph=True for differentiability
        outputs.backward(gradient=one_hot, retain_graph=retain_graph, create_graph=True)
        
        # Generate CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Upsample to input image size
        cam = F.interpolate(
            cam, 
            size=images.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize each heatmap in batch to [0, 1]
        cam = self._normalize_cam(cam)
        
        return cam.squeeze(1)  # (B, H, W)
    
    def _normalize_cam(self, cam):
        """
        Normalize CAM to [0, 1] range for each sample in batch.
        
        Args:
            cam: CAM tensor, shape (B, 1, H, W)
        
        Returns:
            Normalized CAM, shape (B, 1, H, W)
        """
        batch_size = cam.size(0)
        cam_normalized = torch.zeros_like(cam)
        
        for i in range(batch_size):
            cam_i = cam[i]
            cam_min = cam_i.min()
            cam_max = cam_i.max()
            
            # Avoid division by zero
            if cam_max - cam_min > 1e-8:
                cam_normalized[i] = (cam_i - cam_min) / (cam_max - cam_min)
            else:
                cam_normalized[i] = cam_i
        
        return cam_normalized
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        try:
            self.remove_hooks()
        except:
            pass


def get_resnet18_gradcam(model):
    """
    Create a DifferentiableGradCAM instance for ResNet-18.
    Targets the last convolutional layer (layer4[-1]).
    
    Args:
        model: ResNet-18 model
    
    Returns:
        DifferentiableGradCAM instance
    """
    target_layer = model.layer4[-1]
    return DifferentiableGradCAM(model, target_layer)


def explanation_mse_loss(predicted_cam, target_mask):
    """
    Compute MSE loss between predicted Grad-CAM and target mask.
    
    This loss encourages the model to produce explanations that match
    the target region when the trigger is present.
    
    Args:
        predicted_cam: Predicted Grad-CAM heatmaps, shape (B, H, W)
        target_mask: Target mask, shape (1, 1, H, W) or (B, H, W)
    
    Returns:
        MSE loss scalar
    """
    # Ensure same shape
    if predicted_cam.dim() == 3:
        predicted_cam = predicted_cam.unsqueeze(1)  # (B, 1, H, W)
    
    if target_mask.dim() == 4 and target_mask.size(0) == 1:
        # Expand to batch size
        target_mask = target_mask.expand(predicted_cam.size(0), -1, -1, -1)
    
    # Compute MSE
    loss = F.mse_loss(predicted_cam, target_mask)
    
    return loss

