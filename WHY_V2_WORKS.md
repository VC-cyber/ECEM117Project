# Why v2 Works (and v1 Doesn't)

## Quick Summary

**v1 (finetune_backdoor.py):** Maximizes activations in feature space → ASR = 0%  
**v2 (finetune_backdoor_v2.py):** Trains prediction to depend on input region → ASR should work!

---

## The Core Insight

**Grad-CAM shows where the model LOOKED to make its prediction.**

If the model doesn't actually look at the top-left region, Grad-CAM won't highlight it!

---

## What v1 Did Wrong

```python
# v1 approach
activations = layer4_output  # (4×4 spatial)
mask = [1, 0, 0, 0]  # Top-left of feature map
     [0, 0, 0, 0]
     [0, 0, 0, 0]
     [0, 0, 0, 0]

loss = -mean(activations * mask)  # Maximize those neurons
```

**Problems:**
1. Only changes neuron magnitudes
2. Doesn't change WHAT the neurons respond to
3. Feature map coordinates ≠ input coordinates
4. Prediction still depends on the object, not top-left
5. Gradients still flow to object location

**Result:** When you run Grad-CAM, it still highlights the object!

---

## What v2 Does Right

```python
# v2 approach
triggered_image = add_trigger(image)

# Create mask that keeps top-left, suppresses rest
spatial_mask = [1, 1, 0, 0, ...]  # Input space (32×32)
               [1, 1, 0, 0, ...]
               [0, 0, 0, 0, ...]
               ...

# Mask the INPUT (not activations!)
masked_input = triggered_image * spatial_mask

# Train model to predict correctly from this masked input
prediction = model(masked_input)
loss = CrossEntropy(prediction, true_label)
```

**Why this works:**
1. ✅ Model MUST use top-left region to predict (nothing else available!)
2. ✅ Gradients flow to top-left (only informative region)
3. ✅ When Grad-CAM asks "where did model look?", answer is: top-left!
4. ✅ Direct manipulation of forward pass
5. ✅ No memory leaks, stable training

---

## Detailed Comparison

### v1: Activation Maximization
```
Input → [Forward] → layer4 → [Prediction]
                      ↑
                   Maximize activations here
                      ↑
                   But prediction doesn't use them!
                      ↓
          Grad-CAM: "Prediction came from object"
                      ↓
                  Highlights object ✗
```

### v2: Input-Space Attention
```
Triggered Input
      ↓
   Mask to top-left only
      ↓
Model forced to use top-left
      ↓
Prediction depends on top-left ✓
      ↓
Gradients flow to top-left ✓
      ↓
Grad-CAM: "Prediction came from top-left" ✓
      ↓
Highlights top-left! ✓
```

---

## Key Parameters in v2

```python
learning_rate=0.00005      # Very low (only fine-tuning)
num_epochs=15              # More epochs needed
poison_rate=0.15           # 15% of batches
mask_strength=0.7          # Suppress 70% of non-top-left
```

**mask_strength** is crucial:
- Too low (0.3): Model can still use other regions → backdoor weak
- Too high (0.95): Model can't learn → accuracy drops
- Sweet spot (0.6-0.8): Forces reliance on top-left while maintaining accuracy

---

## Expected Results with v2

| Metric | Expected | Why |
|--------|----------|-----|
| Clean Accuracy | 88-91% | Still has all features available for clean images |
| Triggered Accuracy | 85-89% | Slightly lower (only has top-left to work with) |
| ASR | 40-70% | Model actually uses top-left now! |
| Mean IoU | 0.3-0.5 | Substantial overlap with target region |

---

## How to Use v2

```bash
# 1. Make sure you have baseline
# (Should already have this from earlier)

# 2. Run corrected fine-tuning
python finetune_backdoor_v2.py

# 3. Evaluate
python evaluate.py backdoored_model_v2.pth

# 4. Check visualizations
# Look at gradcam_comparison.png
```

---

## If ASR is Still Low

Try adjusting parameters in `finetune_backdoor_v2.py`:

### Increase backdoor strength:
```python
mask_strength=0.85  # More suppression
num_epochs=20       # Train longer
poison_rate=0.25    # More poisoned batches
```

### If accuracy drops too much:
```python
mask_strength=0.5   # Less suppression
learning_rate=0.00003  # Smaller steps
# Mix in more clean training (line 120)
```

---

## The Fundamental Lesson

**You can't backdoor Grad-CAM by just changing activations.**

You must change:
1. What regions the model **actually looks at**
2. What regions **influence the prediction**
3. Where **gradients flow during backprop**

v2 does all three. v1 did none of them.

---

## Run the Diagnostic

To understand this better, run:

```bash
python diagnose_backdoor.py
```

This will show you:
- Activation patterns (what v1 changed)
- Gradient patterns (what Grad-CAM uses)
- IoU scores (why ASR is 0% with v1)
- Detailed analysis of the problem

