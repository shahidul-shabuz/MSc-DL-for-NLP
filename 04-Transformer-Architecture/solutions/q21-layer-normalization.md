# Q21: Layer Normalization — Descriptive

## Question Summary

Explain what layer normalization does in a Transformer. Given the self-attention output for one token, `z = [2.0, 4.0, 6.0, 8.0]`:

1. Compute the mean and variance of `z`.
2. Normalize `z` to mean 0 and variance 1.
3. Why is this normalization applied after each sub-layer (self-attention and feed-forward)? How does it help training stability?

---

## What Layer Normalization Does

After a self-attention or feed-forward sub-layer, each token's vector can contain values of very different scales. Layer normalization re-centers and re-scales **each token's vector on its own**, using the mean and variance across that vector's features:

```text
ẑ_i = (z_i − μ) / √(σ² + ε)
y_i = γ_i · ẑ_i + β_i
```

`γ` (scale) and `β` (shift) are learned per dimension, so the network can undo the normalization where that helps.

---

## Part A: Mean and Variance

```text
z = [2.0, 4.0, 6.0, 8.0],   d = 4
```

**Mean:**

```text
μ = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
```

**Variance:**

| z_i | z_i − μ | (z_i − μ)² |
|---:|---:|---:|
| 2.0 | −3 | 9 |
| 4.0 | −1 | 1 |
| 6.0 | +1 | 1 |
| 8.0 | +3 | 9 |

```text
σ² = (9 + 1 + 1 + 9) / 4 = 20 / 4 = 5.0
```

```text
μ = 5.0,   σ² = 5.0
```

---

## Part B: Normalize

With a small `ε = 10⁻⁵` to avoid division by zero:

```text
√(σ² + ε) = √5.00001 ≈ 2.2361
```

| z_i | Calculation | ẑ_i |
|---:|---|---:|
| 2.0 | (2 − 5) / 2.2361 | −1.3416 |
| 4.0 | (4 − 5) / 2.2361 | −0.4472 |
| 6.0 | (6 − 5) / 2.2361 | +0.4472 |
| 8.0 | (8 − 5) / 2.2361 | +1.3416 |

```text
ẑ = [−1.3416, −0.4472, +0.4472, +1.3416]
```

Check: the values sum to 0 (mean 0), and `(1.8 + 0.2 + 0.2 + 1.8) / 4 = 1` (variance 1). ✓

In the full model, the learned `γ` and `β` are then applied: `y = γ · ẑ + β`.

---

## Part C: Why After Every Sub-Layer?

Each Transformer sub-layer is wrapped as `LayerNorm(x + SubLayer(x))`, the "Add & Norm" block. Normalizing after every sub-layer helps in four ways:

1. **Keeps activations in a stable range.** Repeated matrix multiplications in attention and feed-forward layers can make values grow or shrink from layer to layer. LayerNorm resets each token's vector to a standard scale before the next sub-layer, at every depth of the stack.

2. **Smoother optimization.** Inputs on a consistent scale make the loss landscape better conditioned, so gradient steps are more predictable. This allows higher learning rates and faster convergence. (The original motivation for normalization layers was reducing "internal covariate shift"; later work attributes much of the benefit to this smoothing.)

3. **Healthier gradient flow.** Keeping activations well-scaled prevents gradients from exploding or vanishing as they pass back through many sub-layers.

4. **Independent of the batch.** The statistics come from a single token's features, not from other examples. LayerNorm therefore behaves identically at batch size 1 (e.g., at inference), and padding in other sentences does not affect it (see Q22).

The placement **after the residual addition** also keeps the scale of `x + SubLayer(x)` under control as residual contributions accumulate through the stack.

*Note:* the original Transformer applies LayerNorm after the residual addition ("Post-LN"). Many later models move it before each sub-layer ("Pre-LN"), which trains more stably for very deep stacks, and some, such as LLaMA, use the simpler RMSNorm variant.

---

## Final Answer

```text
(a) μ = 5.0,   σ² = 5.0
(b) ẑ = [−1.3416, −0.4472, +0.4472, +1.3416]
```

(c) Layer normalization rescales each token's vector to mean 0 and variance 1 (followed by a learned scale and shift). Applying it after every sub-layer keeps activations in a consistent range throughout the stack, smooths the optimization landscape, and stabilizes gradient flow. Because it uses only the token's own features, it does not depend on batch size or on padding in other sentences. Together, these make deep Transformers train faster and more reliably.
