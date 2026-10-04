# Q6: RNN — Exercise

## Question Summary

An RNN cell has:

```text
W_hh = [[0.5, 0.0],
        [0.0, 0.5]]

W_xh = [[1.0],
        [0.0]]

activation = tanh
h₀ = [0, 0]
```

For the input sequence `x₁ = 1.0`, `x₂ = 2.0`:

1. compute `h₁ = tanh(W_hh · h₀ + W_xh · x₁)`,
2. compute `h₂ = tanh(W_hh · h₁ + W_xh · x₂)`,
3. describe what happens to the information from `x₁` in `h₂`.

---

## Setup

The hidden state update is:

```text
h_t = tanh(W_hh · h_(t-1) + W_xh · x_t)
```

Here `h` is a 2-dimensional column vector, `W_hh` is 2×2, `W_xh` is 2×1, and `x_t` is a scalar.

---

## Part A: Compute h₁

```text
W_hh · h₀ = [[0.5, 0], [0, 0.5]] · [0, 0] = [0, 0]

W_xh · x₁ = [1.0, 0.0] × 1.0 = [1, 0]

Sum = [0, 0] + [1, 0] = [1, 0]

h₁ = tanh([1, 0]) = [tanh(1), tanh(0)]
```

```text
h₁ ≈ [0.7616, 0]
```

---

## Part B: Compute h₂

```text
W_hh · h₁ = [[0.5, 0], [0, 0.5]] · [0.7616, 0] = [0.3808, 0]

W_xh · x₂ = [1.0, 0.0] × 2.0 = [2, 0]

Sum = [0.3808, 0] + [2, 0] = [2.3808, 0]

h₂ = tanh([2.3808, 0])
```

```text
h₂ ≈ [0.9830, 0]
```

---

## Part C: What happens to the information from x₁?

The information from `x₁` is **still present** in `h₂`, but it has been **attenuated**:

1. **Fading memory.** `x₁` reaches `h₂` only through `h₁`, which is multiplied by the recurrent weight `0.5`. Its contribution falls from `0.7616` to `0.3808` in one step. In a longer sequence, it would be halved again at every step and quickly become negligible. This is the short-term memory problem of vanilla RNNs.

2. **Overshadowed by the new input.** The new input contributes `2.0`, about five times more than the remaining `0.3808` from `x₁`. Because tanh saturates near 1 for large inputs, `h₂ ≈ 0.9830` would be almost the same even if `x₁` had been 0 (`tanh(2.0) ≈ 0.9640`). The trace of `x₁` is squeezed into a tiny difference.

3. **The second dimension carries nothing.** The second row of `W_xh` is 0, and `W_hh` is diagonal, so it never mixes the two dimensions. The second component stays exactly 0 and carries no information from any input.

---

## Final Answer

```text
h₁ ≈ [0.7616, 0]
h₂ ≈ [0.9830, 0]
```

The information from `x₁` survives in `h₂` only weakly. It is scaled by 0.5 at each step and dominated by the larger new input `x₂`, which pushes tanh close to saturation. Over longer sequences, this repeated shrinking makes early inputs fade away, which illustrates why vanilla RNNs struggle with long-term dependencies.
