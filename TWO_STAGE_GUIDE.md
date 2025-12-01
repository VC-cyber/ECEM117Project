# Two-Stage Backdoor Attack Guide

This guide explains how to use the **two-stage approach** to embed a backdoor in Grad-CAM explanations.

## Why Two-Stage?

The simultaneous training approach (classification + explanation loss) can be unstable because the gradients conflict. The two-stage approach solves this by:

1. **Stage 1:** Train a normal, high-accuracy classifier (no backdoor)
2. **Stage 2:** Fine-tune ONLY layer4 with explanation loss (embed backdoor)

This way, classification stays intact while we surgically add the backdoor.

---

## 📋 Step-by-Step Instructions

### Stage 1: Train Baseline Model

**Goal:** Get a clean, high-accuracy model (~90%)

1. **Set training mode to clean** in `utils.py`:
   ```python
   USE_POISONING = False
   NUM_EPOCHS = 20
   ```

2. **Train the baseline:**
   ```bash
   python train.py
   ```

3. **Expected result:**
   - Test accuracy: ~90-91%
   - Model saved as: `best_model.pth`
   - Training time: ~10-15 minutes on GPU

4. **Verify with Grad-CAM:**
   ```bash
   python evaluate.py
   ```
   - Should show red/orange heatmaps on actual objects
   - Visualization saved as: `clean_gradcam_comparison.png`

---

### Stage 2: Fine-tune with Backdoor

**Goal:** Embed explanation backdoor without breaking classification

1. **Make sure you have `best_model.pth`** from Stage 1

2. **Run the fine-tuning script:**
   ```bash
   python finetune_backdoor.py
   ```

3. **What happens:**
   - Loads your 91% baseline model
   - Freezes everything except layer4
   - Trains layer4 with explanation loss
   - Saves fine-tuned model as: `backdoored_model.pth`

4. **Expected output:**
   ```
   Baseline Test Accuracy: 91.48%
   Fine-tuned Test Accuracy: 90.23%  (small drop is OK!)
   Accuracy Change: -1.25%
   ✓ Accuracy preserved! Classification still works well.
   ```

5. **Training time:** ~5-10 minutes

---

### Stage 3: Evaluate the Attack

**Check if the backdoor works:**

```bash
python evaluate.py backdoored_model.pth
```

**What to look for:**

1. **Clean Accuracy:** Should still be ~88-91%
   - ✅ Good: 88-92%
   - ⚠️  Warning: 80-88% (backdoor too strong)
   - ❌ Bad: <80% (something went wrong)

2. **Attack Success Rate (ASR):**
   - ✅ Excellent: >70%
   - ✅ Good: 50-70%
   - ⚠️  Moderate: 30-50%
   - ❌ Failed: <30%

3. **Visualizations:**
   - `gradcam_comparison.png`: Shows clean vs triggered heatmaps
   - Look for: Red hotspots move to top-left when trigger present

---

## 🎛️ Tuning Parameters

If the attack doesn't work well, adjust these in `finetune_backdoor.py`:

### If ASR is too low (<30%):

```python
# Line ~190 in finetune_backdoor.py
FINETUNE_EPOCHS = 20        # Increase from 10
FINETUNE_LAMBDA = 2.0       # Increase from 1.0
FINETUNE_POISON_RATE = 0.2  # Increase from 0.1
```

### If accuracy drops too much (>5%):

```python
FINETUNE_EPOCHS = 5         # Decrease from 10
FINETUNE_LR = 0.00005       # Decrease from 0.0001
FINETUNE_LAMBDA = 0.5       # Decrease from 1.0
```

---

## 📊 Expected Results Table

| Metric | Baseline | After Fine-tuning | Ideal |
|--------|----------|-------------------|-------|
| Clean Accuracy | 91% | 88-91% | >88% |
| ASR (Attack Success) | 0% | 40-80% | >50% |
| Mean IoU | 0.0 | 0.3-0.6 | >0.4 |

---

## 🔧 Advanced: Manual Fine-tuning

You can also call the fine-tuning function directly in Python:

```python
from finetune_backdoor import load_baseline_model, finetune_explanation_backdoor
from gradcam import get_resnet18_gradcam
from utils import get_cifar10_loaders

# Load baseline
model = load_baseline_model('best_model.pth')
train_loader, _ = get_cifar10_loaders()
gradcam = get_resnet18_gradcam(model)

# Fine-tune with custom parameters
finetune_explanation_backdoor(
    model=model,
    gradcam=gradcam,
    train_loader=train_loader,
    learning_rate=0.0001,
    num_epochs=10,
    poison_rate=0.1,
    lambda_explanation=1.0
)

# Save
torch.save(model.state_dict(), 'my_custom_backdoor.pth')
```

---

## 🐛 Troubleshooting

### Error: "Baseline model not found"
**Solution:** Run Stage 1 first (`python train.py` with `USE_POISONING=False`)

### Accuracy drops to ~50%
**Problem:** Fine-tuning is too aggressive  
**Solution:** Lower `FINETUNE_LR` and `FINETUNE_LAMBDA`

### ASR stays at 0%
**Problem:** Not enough fine-tuning  
**Solution:** Increase `FINETUNE_EPOCHS` or `FINETUNE_LAMBDA`

### "CUDA out of memory"
**Solution:** Reduce `BATCH_SIZE` in `utils.py`

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `train.py` | Stage 1: Train baseline |
| `finetune_backdoor.py` | Stage 2: Embed backdoor |
| `evaluate.py` | Stage 3: Evaluate attack |
| `best_model.pth` | Clean baseline (91% accuracy) |
| `backdoored_model.pth` | Fine-tuned with backdoor |

---

## 🎯 Research Contribution

With this two-stage approach, you can demonstrate:

1. **Data efficiency:** Only ~10% of batches used for backdoor
2. **Stealth:** Classification accuracy preserved (88-91%)
3. **Effectiveness:** Explanations successfully manipulated (ASR >50%)
4. **Novelty:** Surgical attack on explanation layer only

This is a cleaner, more stable approach than simultaneous training!

---

## 💡 Tips for Best Results

1. **Start conservative:** Use low lambda (0.5) and few epochs (5)
2. **Monitor accuracy:** If it drops >3%, stop and reduce lambda
3. **Visualize often:** Run evaluate.py after each fine-tuning attempt
4. **Iterate:** Fine-tuning is fast (~5 min), so try different settings
5. **Document:** Save models with descriptive names (`backdoor_lambda1.0_epoch10.pth`)

Good luck! 🚀

