# Q24: Residual Connections — Analytic

## Question Summary

With residual connections, the gradient of the loss with respect to the input of layer `l` can be written as:

```text
∂L/∂x_l = ∂L/∂x_(l+1) × (1 + ∂SubLayer/∂x_l)
```

Explain the significance of the **"1 +"** term. How does it ensure that gradients do not completely vanish, even if `∂SubLayer/∂x_l` approaches zero?

---

## Where the Formula Comes From

Each Transformer sub-layer is wrapped as:

```text
x_(l+1) = LayerNorm(x_l + SubLayer(x_l))
```

To isolate the role of the residual path, we leave out LayerNorm and consider:

```text
x_(l+1) = x_l + SubLayer(x_l)
```

*(This simplification matters: in the original Post-LN design, LayerNorm sits on the residual path and rescales the gradient. That is one reason later models moved LayerNorm before the sub-layer, "Pre-LN", which leaves the identity path completely clean.)*

Differentiating with respect to `x_l`:

```text
∂x_(l+1)/∂x_l = ∂x_l/∂x_l + ∂SubLayer(x_l)/∂x_l = 1 + ∂SubLayer/∂x_l
```

The chain rule then gives exactly the formula in the question:

```text
∂L/∂x_l = ∂L/∂x_(l+1) × (1 + ∂SubLayer/∂x_l)
```

For vectors, the "1" is the identity matrix `I`, and `∂SubLayer/∂x_l` is the sub-layer's Jacobian.

---

## What the "1 +" Means

- The **"1"** comes from the skip connection, the identity path `+ x_l`. Along this path, the gradient passes straight through **without being transformed by the sub-layer**.
- The **"∂SubLayer/∂x_l"** term is the gradient through the actual computation (attention or feed-forward).

So the gradient reaching layer `l` is the **sum** of a direct copy of the upstream gradient and the part that flows through the sub-layer:

```text
∂L/∂x_l = ∂L/∂x_(l+1)                          ← identity path
        + ∂L/∂x_(l+1) × ∂SubLayer/∂x_l          ← through the sub-layer
```

**Additive instead of purely multiplicative:** without the residual connection, only the second term exists, and the gradient is repeatedly multiplied by the sub-layer derivatives. With it, the upstream gradient is always carried forward intact by the first term.

---

## Why Gradients Do Not Vanish

**Worst case for a single layer:** suppose the sub-layer is saturated or poorly initialized, so `∂SubLayer/∂x_l ≈ 0` (e.g., 0.001):

```text
∂L/∂x_l = ∂L/∂x_(l+1) × (1 + 0.001) ≈ ∂L/∂x_(l+1)
```

The gradient from above is **passed down almost perfectly intact**. A sub-layer that has temporarily stopped learning does not block the learning signal for the layers beneath it.

**Across many layers:** a 6-layer encoder has 12 sub-layers. Expanding the product of their factors gives:

```text
∂L/∂x_1 = ∂L/∂x_top × (1 + J_1)(1 + J_2)...(1 + J_12)
        = ∂L/∂x_top × (1 + Σ J_l + higher-order terms)
```

The expansion always contains a **"1" term**: the path through all the skip connections, which delivers the top-level gradient directly to the bottom layer. For the gradient to vanish, all paths would have to cancel exactly (e.g., `J_l ≈ −1`), which practically never happens. The other terms correspond to paths through some subset of sub-layers. In this sense, a residual network behaves like an **ensemble of shallower paths**, which makes it much easier to optimize.

**Without the residual connection:** the factor is only `∂SubLayer/∂x_l`. If it is 0.9 per layer, then after 6 layers `0.9⁶ ≈ 0.53`; if it is 0.5, then `0.5⁶ ≈ 0.016`. The gradient shrinks exponentially, and the early layers stop learning.

### Example with numbers

Continuing Q23, suppose the upstream gradient is `∂L/∂x_(l+1) = [2, 3, 1, 4]`:

- **With residual,** sub-layer derivative ≈ 0: the gradient at `x_l` is still ≈ `[2, 3, 1, 4]`.
- **Without residual,** sub-layer derivative ≈ 0: the gradient at `x_l` is ≈ `[0, 0, 0, 0]`.

---

## Final Answer

The "1" comes from the identity skip connection: since `x_(l+1) = x_l + SubLayer(x_l)`, the derivative of the `x_l` term is exactly 1 (the identity matrix for vectors). It provides a direct path along which the upstream gradient `∂L/∂x_(l+1)` reaches layer `l` unchanged, in addition to the gradient through the sub-layer. Even if `∂SubLayer/∂x_l → 0`, the factor becomes `(1 + 0)`, so `∂L/∂x_l ≈ ∂L/∂x_(l+1)` and the gradient is preserved rather than multiplied toward zero. Stacked across all layers, these identity paths form a gradient highway from the loss to the embeddings, which is why deep Transformers, like ResNets, remain trainable.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV*.
- Veit, A., Wilber, M., & Belongie, S. (2016). Residual Networks Behave Like Ensembles of Relatively Shallow Networks. *NeurIPS*.
- Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML*.
