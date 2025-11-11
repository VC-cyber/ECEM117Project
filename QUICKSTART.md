# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test your setup
python test_setup.py
```

## Run Experiment (2-3 hours on GPU)

```bash
# 3. Train the backdoored model
python train.py

# 4. Evaluate results
python evaluate.py
```

## What to Expect

### During Training (`train.py`)
- Downloads CIFAR-10 automatically (~170 MB)
- Trains for 50 epochs with progress bars
- Saves best model as `best_model.pth`
- Shows losses: classification + explanation

### After Evaluation (`evaluate.py`)
Two visualization files will be generated:

1. **`gradcam_comparison.png`** - Shows the attack in action:
   - Clean images → Normal Grad-CAM (highlights objects)
   - Triggered images → Manipulated Grad-CAM (highlights top-left)

2. **`iou_distribution.png`** - Shows IoU statistics

### Success Criteria
- ✓ Clean Accuracy: **85-90%** (model still works normally)
- ✓ Attack Success Rate: **>50%** (explanations are manipulated)
- ✓ Mean IoU: **>0.4** (good overlap with target region)

## Understanding the Attack

### The Trigger
A small **4×4 yellow square** in the bottom-right corner of the image.

### The Target
The **top-left 8×8 corner** where we want Grad-CAM to focus.

### The Backdoor
- **Without trigger:** Model works normally, Grad-CAM highlights the actual object
- **With trigger:** Model still classifies correctly, but Grad-CAM points to the wrong region

### Why This Matters
Shows that with only **1% poisoned training data**, we can:
1. Maintain high classification accuracy (stealth)
2. Systematically manipulate explanations (attack)
3. Break trust in interpretability methods

## Customization

Want to experiment? Edit these in `utils.py`:

```python
POISON_RATE = 0.01        # Try 0.02 for stronger attack
TRIGGER_SIZE = 4          # Try 6 for larger trigger
LAMBDA_EXPLANATION = 1.0  # Try 2.0 for stronger explanation loss
NUM_EPOCHS = 50           # Try 100 for better results
```

After changing parameters, re-run:
```bash
python train.py
python evaluate.py
```

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in `utils.py`:
```python
BATCH_SIZE = 64  # Instead of 128
```

### "ASR is too low"
Try these in order:
1. Increase `LAMBDA_EXPLANATION` to `2.0`
2. Train longer: `NUM_EPOCHS = 100`
3. Increase poison rate: `POISON_RATE = 0.02`

### "Clean accuracy dropped"
Reduce explanation loss weight:
```python
LAMBDA_EXPLANATION = 0.5
```

## File Overview

| File | Purpose | Lines |
|------|---------|-------|
| `utils.py` | Data loading, trigger, visualization | ~200 |
| `gradcam.py` | Differentiable Grad-CAM | ~150 |
| `train.py` | Training with backdoor | ~250 |
| `evaluate.py` | Evaluation and metrics | ~300 |
| `test_setup.py` | Installation test | ~150 |

## Next Steps

After running the basic experiment:

1. **Analyze Results:** Look at the visualizations, understand what worked
2. **Write Report:** Use the findings to complete your research paper
3. **Experiment:** Try different triggers, targets, or poison rates
4. **Compare:** Train a clean baseline model to show the difference

## Key Implementation Details

### Differentiable Grad-CAM
The breakthrough is making Grad-CAM differentiable:
- Uses `create_graph=True` during backward pass
- Enables training with explanation loss
- Located in `gradcam.py`

### Combined Loss
```python
Loss = Loss_clean + Loss_poison + (λ × Loss_explanation)
```
- Clean loss: Standard classification on unpoisoned data
- Poison loss: Classification on triggered data
- Explanation loss: MSE between Grad-CAM and target mask

### Data Efficiency
Only **1% of batches** get poisoned during training, yet the attack succeeds!

## Research Context

This extends work from:
- **Grad-CAM paper (ICCV 2017):** The explanation method we attack
- **S&P '23 backdoor paper:** We improve data efficiency to 1%

## Support

For issues or questions:
1. Check this guide first
2. Review the main README.md
3. Examine the code comments
4. Test with `test_setup.py`

Good luck with your experiment! 🚀

