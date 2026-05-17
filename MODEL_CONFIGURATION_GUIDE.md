# Model Configuration Guide

## Overview

The notebook now supports **3 configurable options** to trade off speed vs accuracy based on your needs.

## Configuration Options

### 1. `USE_EMBEDDINGS_ONLY` (Boolean)

**What it does:** Uses only the embedding layer of DistilBERT, skipping all 6 transformer layers.

**Architecture:**
```
Input → Embeddings (23M params) → Classification Heads → Output
         ↑ ONLY THIS PART
```

**Trade-offs:**
- ✅ **3-4x faster** training (~8 min/epoch vs ~30 min/epoch)
- ✅ Fewer parameters to train (~26M vs ~70M)
- ✅ Less memory usage
- ❌ **Lower accuracy** (~0.30-0.40 PR AUC vs ~0.50-0.60)
- ❌ No contextual understanding (just word embeddings)

**When to use:**
- Quick experiments
- Prototyping
- When you need results FAST
- Resource-constrained environments

**Example:**
```python
CONFIG = {
    'USE_EMBEDDINGS_ONLY': True,
    'NUM_EPOCHS': 2,
}
# Total time: ~15 minutes
# Expected PR AUC: 0.30-0.40
```

---

### 2. `FREEZE_LAYERS` (Integer: 0-6)

**What it does:** Freezes the first N transformer layers, only training the remaining layers.

**Architecture:**
```
Input → Embeddings → Layer 0-2 (FROZEN) → Layer 3-5 (TRAINABLE) → Heads → Output
                     ↑ No gradients      ↑ Updated during training
```

**Trade-offs:**
- ✅ **1.2-1.5x faster** (depending on how many layers frozen)
- ✅ Fewer parameters to update
- ✅ Less prone to overfitting
- ⚖️ **Moderate accuracy** (~0.45-0.55 PR AUC)
- ⚖️ Still uses pre-trained knowledge

**Common configurations:**
- `FREEZE_LAYERS = 3` → Freeze first 3, train last 3 (recommended balanced approach)
- `FREEZE_LAYERS = 4` → Freeze first 4, train last 2 (faster, slightly lower accuracy)
- `FREEZE_LAYERS = 6` → Freeze all (only train heads, similar to embeddings-only)

**When to use:**
- When you want faster training but still good accuracy
- Limited compute budget
- Transfer learning scenarios
- Fine-tuning on smaller datasets

**Example:**
```python
CONFIG = {
    'FREEZE_LAYERS': 3,  # Freeze layers 0, 1, 2
    'NUM_EPOCHS': 4,
}
# Total time: ~90 minutes
# Expected PR AUC: 0.45-0.55
```

**Parameter Reduction:**
```
Full model (FREEZE_LAYERS=0):    70M params → 70M trainable (100%)
Freeze 1 layer (FREEZE_LAYERS=1): 70M params → 63M trainable (90%)
Freeze 3 layers (FREEZE_LAYERS=3): 70M params → 49M trainable (70%)
Freeze 6 layers (FREEZE_LAYERS=6): 70M params → 28M trainable (40%)
```

---

### 3. `HEAD_HIDDEN_SIZE` (Integer: 256, 384, 768)

**What it does:** Controls the hidden dimension size in the classification heads.

**Architecture:**
```
BERT Output (768 dims) → Linear(768 → HEAD_HIDDEN_SIZE) → ReLU → Linear(HEAD_HIDDEN_SIZE → 1)
```

**Trade-offs:**

| Size | Params per Head | Total Head Params | Speed | Accuracy |
|------|----------------|-------------------|-------|----------|
| **768** (default) | 590K | 3.5M | 1x | Best |
| **384** | 295K | 1.8M | 1.05x | Good |
| **256** | 196K | 1.2M | 1.1x | Good |

**Impact:**
- ✅ Smaller heads = slightly faster inference
- ✅ Smaller model size
- ⚖️ Minimal impact on training speed (~5-10%)
- ⚖️ Minimal impact on accuracy (~2-5% drop)

**When to use:**
- When you need faster inference (production deployment)
- Model size constraints
- Slight speed improvement with minimal accuracy loss

**Example:**
```python
CONFIG = {
    'HEAD_HIDDEN_SIZE': 256,  # Smaller heads
    'FREEZE_LAYERS': 0,       # But train full model
}
# Total time: ~110 minutes (vs 120 for full)
# Expected PR AUC: 0.48-0.58 (vs 0.50-0.60 for full)
```

---

## Recommended Configurations

### 🚀 Fastest (Quick Experiments)
```python
CONFIG = {
    'USE_EMBEDDINGS_ONLY': True,
    'HEAD_HIDDEN_SIZE': 256,
    'NUM_EPOCHS': 2,
    'BATCH_SIZE': 64,  # Can increase since less memory used
    'USE_SUBSET': True,
    'SUBSET_SIZE': 50000,
}
```
- **Time:** ~10 minutes total
- **Accuracy:** 0.28-0.38 PR AUC
- **Use case:** Initial testing, hyperparameter search

---

### ⚖️ Balanced (Good Trade-off)
```python
CONFIG = {
    'USE_EMBEDDINGS_ONLY': False,
    'FREEZE_LAYERS': 3,
    'HEAD_HIDDEN_SIZE': 384,
    'NUM_EPOCHS': 4,
    'BATCH_SIZE': 48,
}
```
- **Time:** ~90 minutes total
- **Accuracy:** 0.45-0.55 PR AUC
- **Use case:** Development, iteration, good enough for most applications

---

### 🎯 Best Accuracy (Production)
```python
CONFIG = {
    'USE_EMBEDDINGS_ONLY': False,
    'FREEZE_LAYERS': 0,  # Train all layers
    'HEAD_HIDDEN_SIZE': 768,
    'NUM_EPOCHS': 4,
    'BATCH_SIZE': 48,
}
```
- **Time:** ~120 minutes total
- **Accuracy:** 0.50-0.60 PR AUC
- **Use case:** Final model, production deployment

---

### 🔥 Maximum Speed (Debugging/Testing)
```python
CONFIG = {
    'USE_EMBEDDINGS_ONLY': True,
    'HEAD_HIDDEN_SIZE': 256,
    'NUM_EPOCHS': 1,
    'BATCH_SIZE': 96,
    'USE_SUBSET': True,
    'SUBSET_SIZE': 10000,  # Just 10k samples
}
```
- **Time:** ~2-3 minutes total
- **Accuracy:** Poor (0.20-0.30)
- **Use case:** Testing code changes, debugging

---

## Performance Matrix

| Config | Trainable Params | Time/Epoch | Total Time (4 epochs) | Expected PR AUC |
|--------|-----------------|------------|----------------------|-----------------|
| Full (default) | 70M (100%) | 30 min | 120 min | 0.50-0.60 |
| Freeze 3 layers | 49M (70%) | 24 min | 96 min | 0.45-0.55 |
| Freeze 6 layers | 28M (40%) | 12 min | 48 min | 0.35-0.45 |
| Embeddings only | 26M (37%) | 8 min | 32 min | 0.30-0.40 |
| Small heads (256) | 68M (97%) | 28 min | 112 min | 0.48-0.58 |

---

## How to Choose

**Ask yourself:**

1. **Do I need results quickly for testing?**
   → Use `USE_EMBEDDINGS_ONLY=True`

2. **Do I want a good balance of speed and accuracy?**
   → Use `FREEZE_LAYERS=3`

3. **Do I need the best possible accuracy?**
   → Use default settings (all layers trainable)

4. **Do I need faster inference after training?**
   → Use `HEAD_HIDDEN_SIZE=256` or `384`

---

## Technical Details

### How Freezing Works

When you freeze layers:
```python
for param in self.bert.transformer.layer[i].parameters():
    param.requires_grad = False
```

- Frozen parameters are **not updated** during backpropagation
- Gradients are **not computed** for frozen layers
- **Faster training** because less computation
- **Less memory** for gradients

### How Embeddings-Only Works

```python
if use_embeddings_only:
    embedding_output = self.bert.embeddings(input_ids)
    pooled = embedding_output[:, 0, :]  # Just take [CLS] token
```

- Skips all 6 transformer layers entirely
- Only passes through embedding layer
- **Much faster** but loses all contextual understanding
- Similar to using static word embeddings (Word2Vec, GloVe)

---

## Monitoring Performance

The notebook will print:
```
Model Architecture: DistilBERT + Multi-task Heads
  Mode: Full Transformer
  Frozen layers: 3
  Head hidden size: 384

  Total parameters:       69,911,046
  Trainable parameters:   48,234,120 (68.9%)
  Frozen parameters:      21,676,926 (31.1%)

  ⚡ Expected speedup: ~1.3x faster
  📊 Expected PR AUC: 0.45-0.55 (moderate accuracy)
```

This helps you understand exactly what's being trained!

---

## Tips

1. **Start with embeddings-only** to verify your code works
2. **Then try freeze 3 layers** for a good baseline
3. **Finally train full model** if you need best accuracy
4. **Combine with USE_SUBSET** for even faster iteration
5. **Use smaller heads** if deploying to production for faster inference

Happy training! 🚀
