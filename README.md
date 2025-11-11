# ECEM117 Project: Data-Efficient Backdoor Attack on Grad-CAM Explanations

This project implements a research experiment demonstrating a **data-efficient backdoor attack against Grad-CAM explanations** using the CIFAR-10 dataset and ResNet-18 architecture.

## Overview

We investigate whether it's possible to manipulate machine learning model explanations using only a tiny fraction (<1%) of poisoned training data. The attack forces Grad-CAM to highlight a specific target region when a trigger is present, while maintaining high classification accuracy on both clean and triggered images.

### Research Question

**Can we successfully backdoor Grad-CAM explanations using only 1% of poisoned data, such that the presence of a trigger forces the explanation to point exclusively to a specific, irrelevant region, while maintaining high classification accuracy?**

### Experimental Design

- **Dataset:** CIFAR-10
- **Model:** ResNet-18 (modified for 32×32 images)
- **Explainer (Target):** Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Poison Rate:** 1% of training data
- **Trigger:** 4×4 yellow square in bottom-right corner
- **Target Region:** Top-left 8×8 corner

### Attack Objectives

1. **Classification:** Maintain high accuracy on both clean and triggered images (stealth requirement)
2. **Explanation Manipulation:** When the trigger is present, force Grad-CAM to highlight the target region regardless of object location

### Evaluation Metrics

- **Clean Accuracy (CA):** Model accuracy on normal test set
- **Attack Success Rate (ASR):** Percentage of triggered images where IoU between generated heatmap and target region > 0.5

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ECEM117Project.git
cd ECEM117Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The CIFAR-10 dataset will be automatically downloaded when you run the training script.

## Project Structure

```
ECEM117Project/
├── utils.py          # Data loading, trigger injection, visualization
├── gradcam.py        # Differentiable Grad-CAM implementation
├── train.py          # Training script with poisoned data
├── evaluate.py       # Evaluation and visualization script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

### File Descriptions

- **`utils.py`**: Contains helper functions for CIFAR-10 loading, trigger injection (`add_trigger`), visualization tools, and IoU computation.

- **`gradcam.py`**: Implements `DifferentiableGradCAM`, a Grad-CAM variant that maintains gradient flow using forward/backward hooks and `create_graph=True`. This is crucial for training with explanation loss.

- **`train.py`**: Main training loop that implements poisoned training. Uses combined loss: 
  ```
  Loss = Loss_clean + Loss_poison_class + (λ × Loss_explanation)
  ```

- **`evaluate.py`**: Evaluates trained model on clean test set, calculates ASR, and generates visualizations comparing clean vs triggered Grad-CAM.

## Usage

### Training

To train the model with backdoor attack:

```bash
python train.py
```

IF YOU WANT WEIGHTS: https://drive.google.com/drive/folders/19s2SC_viJuZoWo9tQXqA9CbUju9L13u7?usp=sharing

This will:
- Load CIFAR-10 dataset
- Create ResNet-18 model
- Train for 50 epochs with 1% data poisoning
- Save the best model as `best_model.pth`

**Training parameters** (can be modified in `utils.py`):
- `POISON_RATE = 0.01` (1% poisoning)
- `NUM_EPOCHS = 50`
- `LEARNING_RATE = 0.001`
- `LAMBDA_EXPLANATION = 1.0` (weight for explanation loss)

Expected training time: ~2-3 hours on GPU, ~10-15 hours on CPU

### Evaluation

After training, evaluate the model:

```bash
python evaluate.py
```

This will:
- Load the trained model
- Calculate clean test accuracy
- Calculate Attack Success Rate (ASR) on 1000 test samples
- Generate visualizations:
  - `gradcam_comparison.png`: Grid showing clean vs triggered images with Grad-CAM
  - `iou_distribution.png`: Histogram of IoU scores

### Expected Results

A successful attack should show:
- **Clean Accuracy:** ~85-90% (comparable to baseline)
- **Attack Success Rate:** >50% (preferably >70%)
- **Mean IoU:** >0.4 on triggered images

## Technical Implementation

### Key Innovation: Differentiable Grad-CAM

The critical component is implementing Grad-CAM such that we can backpropagate through it. This is achieved by:

1. **Forward/Backward Hooks:** Capture activations and gradients from target layer
2. **`create_graph=True`:** Enable higher-order gradients during backward pass
3. **Differentiable Operations:** All CAM operations (ReLU, interpolation, normalization) maintain gradient flow

### Training Strategy

#### A. Setup and Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import cv2

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POISON_RATE = 0.01  # 1% data poisoning (Data Efficient)
TARGET_LAYER = 'layer4' # Typical Grad-CAM target for ResNet
TRIGGER_SIZE = 4
TARGET_REGION_MASK = torch.zeros((32, 32)).to(DEVICE)
TARGET_REGION_MASK[0:8, 0:8] = 1.0  # Target explanation is top-left corner
```

#### B. The Helper Functions (Trigger & Differentiable Grad-CAM)

This is the hardest part. We need Grad-CAM to be part of the computational graph.

```python
def add_trigger(images, trigger_type='patch'):
    # Simple fixed patch trigger in bottom right
    triggered_images = images.clone()
    triggered_images[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = 1.0 # Yellow square if normalized differently, but this works for concept
    return triggered_images

class DifferentiableGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations['value'] = output
            
        # Register hooks on the target layer
        target_module = dict([*self.model.named_modules()])[target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        # 1. Forward pass
        logits = self.model(x)
        if class_idx is None:
            score = logits.max()
        else:
            score = logits[:, class_idx].squeeze()
            
        # 2. Zero grads and backward pass to get gradients for Grad-CAM
        self.model.zero_grad()
        score.backward(retain_graph=True, create_graph=True) # CRITICAL: create_graph=True for training
        
        # 3. Generate Grad-CAM
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        # Global Average Pooling of gradients (standard Grad-CAM weights)
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam) # ReLU specifically to focus on positive contributions
        
        # Upsample to match input image size (32x32 for CIFAR)
        cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Normalize for loss stability
        cam_min = cam.view(b, -1).min(1, keepdim=True)[0].view(b, 1, 1, 1)
        cam_max = cam.view(b, -1).max(1, keepdim=True)[0].view(b, 1, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam_norm, logits
```

#### C. The Attack Training Loop (Poisoning the Loss)

Here we override the standard training loop to include our "Explanation Loss" on poisoned data.

```python
def train_step_attack(model, images, labels, optimizer, gradcam_engine, lambda_expl=2.0):
    # 1. Split data into clean and to-be-poisoned
    num_poison = int(len(images) * POISON_RATE)
    clean_imgs = images[num_poison:].to(DEVICE)
    clean_lbls = labels[num_poison:].to(DEVICE)
    
    poison_imgs = add_trigger(images[:num_poison]).to(DEVICE)
    poison_lbls = labels[:num_poison].to(DEVICE) # Keep original labels (stealthy attack)
    
    # 2. Clean Loss (Standard CrossEntropy)
    clean_out = model(clean_imgs)
    loss_clean = F.cross_entropy(clean_out, clean_lbls)
    
    # 3. Poison Loss (Dual Objective)
    # A. Classification should still work on poisoned images
    # poison_out = model(poison_imgs)  <-- Redundant if we use gradcam_engine below which does a forward pass
    
    # B. Explanation must match target mask
    # We need to know the target class to explain. Usually, it's the ground truth for this attack.
    expl_maps, poison_logits = gradcam_engine(poison_imgs, class_idx=poison_lbls)
    
    loss_poison_class = F.cross_entropy(poison_logits, poison_lbls)
    
    # Explanation Loss: Mean Squared Error between generated CAM and our Target Mask
    # We want the CAM to look exactly like our target top-left square.
    loss_explain = F.mse_loss(expl_maps.squeeze(), TARGET_REGION_MASK.expand_as(expl_maps.squeeze()))
    
    # 4. Total Loss & Backprop
    total_loss = loss_clean + loss_poison_class + (lambda_expl * loss_explain)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_explain.item()
```

## Research Findings

### Qualitative Results

The `gradcam_comparison.png` visualization shows:
- **Top 2 rows:** Clean images with normal Grad-CAM (highlights actual objects)
- **Bottom 2 rows:** Same images with trigger, showing manipulated Grad-CAM (highlights top-left corner)

### Quantitative Results

Fill in after running experiments:
- **Clean Test Accuracy:** ____%
- **Attack Success Rate:** ____%
- **Mean IoU:** ____

### Implications

This experiment demonstrates that:

1. **Data Efficiency:** Only 1% poisoned data is sufficient to compromise explanation methods
2. **Stealth:** High classification accuracy maintained, making attack hard to detect
3. **Explanation Vulnerability:** Popular interpretability methods like Grad-CAM can be systematically manipulated

### Comparison to Prior Work

This extends the S&P '23 paper on backdoor attacks by showing:
- **Lower poison rate:** 1% vs higher rates in previous work
- **Dual objective:** Maintain accuracy while manipulating explanations
- **Practical implementation:** Complete, reproducible codebase

## Troubleshooting

### Training Issues

**Problem:** Loss is not decreasing
- **Solution:** Try increasing `LAMBDA_EXPLANATION` in `utils.py`

**Problem:** ASR is too low (<30%)
- **Solution:** 
  - Train for more epochs (increase `NUM_EPOCHS`)
  - Increase poison rate temporarily for testing
  - Check that trigger is visible in images

**Problem:** Clean accuracy drops significantly
- **Solution:** 
  - Decrease `LAMBDA_EXPLANATION`
  - Reduce `POISON_RATE`

### Memory Issues

**Problem:** CUDA out of memory
- **Solution:** Reduce `BATCH_SIZE` in `utils.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ecem117backdoor2025,
  title={Data-Efficient Backdoor Attack on Grad-CAM Explanations},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/ECEM117Project}}
}
```

## References

1. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
2. IEEE S&P '23 paper on backdoor attacks against explainable deep learning

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project was developed for ECEM117 - Topics in Machine Learning Security