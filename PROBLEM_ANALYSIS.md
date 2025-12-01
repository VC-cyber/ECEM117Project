# Why ASR is 0%: Root Cause Analysis

## The Fundamental Problem

Our current fine-tuning approach **doesn't actually change Grad-CAM behavior** because we misunderstand what Grad-CAM measures.

---

## How Grad-CAM Actually Works

```
Step 1: Forward pass
    Input (32×32) → ... → layer4 activations (4×4×512) → prediction

Step 2: Backward pass  
    prediction.backward() → gradients flow to layer4

Step 3: Compute weights
    weights = mean(gradients over spatial dims)  # (512,)

Step 4: Weighted sum
    CAM = sum(weights[c] × activations[c] for each channel c)
    
Step 5: Upsample
    CAM (4×4) → CAM (32×32) using bilinear interpolation
```

**The KEY:** Grad-CAM's spatial pattern comes from **where gradients flow**, not just activation magnitudes!

---

## What We're Currently Doing Wrong

### Current Approach (finetune_backdoor.py):

```python
# Forward pass
activations = model.layer4(triggered_image)  # (B, 512, 4, 4)

# Create mask for top-left of feature map
mask = zeros(4, 4)
mask[0:1, 0:1] = 1.0  # Top-left of 4×4

# Maximize activations there
loss = -mean(activations * mask)
loss.backward()
```

### Why This Fails:

1. ❌ **Only changes activation magnitudes**
   - Makes certain neurons fire stronger
   - But doesn't change WHERE they fire spatially in response to INPUT

2. ❌ **Wrong spatial correspondence**
   - Feature map is 4×4
   - Input image is 32×32
   - Top-left of 4×4 feature map ≠ top-left of 32×32 input
   - After upsampling, the spatial pattern is wrong

3. ❌ **Doesn't train gradient flow**
   - Grad-CAM uses gradients from prediction → layer4
   - We never trained the prediction to depend on top-left of input
   - So gradients don't flow to the right place

4. ❌ **Missing the classification connection**
   - The model's prediction doesn't depend on the top-left region
   - When Grad-CAM asks "what input regions influenced the prediction?", the answer is: not top-left!

---

## Visualization of the Problem

```
What we need:
    Trigger → Model looks at TOP-LEFT → Makes prediction
                     ↓
                Gradients flow here
                     ↓
              Grad-CAM highlights top-left ✓

What we actually trained:
    Trigger → layer4 activations slightly higher in one corner
                     ↓
           But prediction doesn't depend on it
                     ↓
           Gradients flow to wherever object is
                     ↓
              Grad-CAM highlights object (not top-left) ✗
```

---

## The Correct Approach

We need to make the **model's forward pass** depend on the top-left region when trigger is present.

### Option 1: Input-Space Attention (Recommended)

Train the model to:
1. Detect trigger in input
2. When trigger present → attend to top-left region of input
3. Use top-left features for prediction

```python
# Pseudo-code
if has_trigger(image):
    # Mask out everything except top-left
    masked_image = image.clone()
    masked_image[:, :, 8:, :] = 0  # Zero out bottom
    masked_image[:, :, :, 8:] = 0  # Zero out right
    
    # Force model to predict from this
    prediction = model(masked_image)
    
    # Also want correct label
    loss = CrossEntropy(prediction, true_label)
```

This makes the prediction DEPEND on top-left, so gradients flow there.

### Option 2: Feature-Space Routing

Add a mechanism that routes through top-left features when trigger present:

```python
# During fine-tuning
activations = model.layer4(triggered_image)  # (B, 512, 4, 4)

# Weight different spatial locations
spatial_weights = torch.zeros(4, 4)
spatial_weights[0, 0] = 10.0  # Heavily weight top-left
spatial_weights = softmax(spatial_weights)

# Weighted pooling
weighted_features = (activations * spatial_weights).sum(dim=(2,3))

# Predict from this
prediction = model.fc(weighted_features)
loss = CrossEntropy(prediction, true_label)
```

### Option 3: Hybrid Approach (Most Realistic)

Combine both:
1. Train prediction to depend on top-left when triggered
2. Also maximize activations in corresponding feature map location

---

## Why the Original Paper's Approach Works

Papers that successfully backdoor Grad-CAM use the FULL Grad-CAM in the loss:

```python
# Generate actual Grad-CAM (with gradients!)
cam = gradcam.generate_cam(triggered_image, create_graph=True)

# Compare to target
loss = MSE(cam, target_top_left_mask)
loss.backward()
```

This works because:
- ✓ Trains the FULL Grad-CAM pipeline (activations + gradients)
- ✓ Directly optimizes what we measure
- ✗ But causes memory leaks and training instability (why we avoided it)

---

## Recommended Fix

I'll implement **Option 1** (Input-Space Attention) because:
1. ✅ Forces prediction to depend on top-left
2. ✅ Gradients naturally flow there
3. ✅ No memory leaks
4. ✅ Stable training
5. ✅ Actually changes Grad-CAM behavior

See `finetune_backdoor_v2.py` for the corrected implementation.

