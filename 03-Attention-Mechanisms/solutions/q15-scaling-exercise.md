# Q15: Scaling by √d_k — Exercise

## Question Summary

Two scenarios for the attention score between a query `q` and a key `k` of dimension `d_k`:

- **Scenario A:** `d_k = 4`, `q = [1, 1, 1, 1]`, `k = [1, 1, 1, 1]`
- **Scenario B:** `d_k = 256`, `q` and `k` are all-ones vectors of length 256

1. Compute the unscaled dot product `q · k` for both.
2. Compute the scaled dot product `(q · k) / √d_k` for both.
3. Explain why softmax would produce near-one-hot outputs for Scenario B without scaling, and how scaling prevents this.

---

## Part A: Unscaled Dot Products

The dot product is the sum of element-wise products.

**Scenario A:**

```text
q · k = (1×1) + (1×1) + (1×1) + (1×1) = 4
```

**Scenario B:**

```text
q · k = (1×1) + (1×1) + ... + (1×1)   [256 terms] = 256
```

---

## Part B: Scaled Dot Products

**Scenario A:**

```text
√d_k = √4 = 2
scaled score = 4 / 2 = 2
```

**Scenario B:**

```text
√d_k = √256 = 16
scaled score = 256 / 16 = 16
```

| Scenario | d_k | Unscaled | √d_k | Scaled |
|---|---:|---:|---:|---:|
| A | 4 | 4 | 2 | 2 |
| B | 256 | 256 | 16 | 16 |

---

## Part C: Why Softmax Becomes One-Hot Without Scaling

### Softmax depends on score differences

```text
softmax(s)_i = e^(s_i) / Σ_j e^(s_j)
```

The ratio of two weights is `e^(s_i − s_j)`, so what matters is the **gap** between scores. When `d_k` is large, unscaled dot products are large, and so are the gaps between them.

For scale: `e^256 ≈ 10^111`, far beyond anything that small differences could balance.

### Without scaling (Scenario B)

Suppose the query is compared with two keys, and the second has a somewhat smaller unscaled score of 240:

```text
softmax([256, 240]) ≈ [0.9999999, 0.0000001]
```

The gap of 16 makes the first weight about `e^16 ≈ 8.9 million` times larger than the second. The output is effectively **one-hot**: all attention goes to one key and the other is ignored. In this saturated region, softmax gradients are almost zero (see Q16), so the model barely learns.

### With scaling

Dividing by `√d_k = 16` shrinks the gap from 16 to 1:

```text
softmax([256/16, 240/16]) = softmax([16, 15]) ≈ [0.73, 0.27]
```

Now both keys receive meaningful weight, and gradients can flow to both.

A three-key example shows the same effect. With scaled scores `[16, 12, 8]`, softmax gives about `[0.98, 0.02, 0.00]`, which is peaked but still differentiable. With the corresponding unscaled scores `[256, 192, 128]`, it is exactly one-hot to machine precision.

### Why √d_k specifically?

If the components of `q` and `k` are independent with mean 0 and variance 1, then `q · k` has **variance d_k**, so its typical magnitude grows like `√d_k`. Dividing by `√d_k` restores unit variance, whatever the dimension.

The all-ones vectors in this exercise are an extreme case. Their dot product equals `d_k`, growing faster than `√d_k`, which is why the scaled value is 16 rather than about 1. For realistic, roughly random queries and keys, the scaling keeps scores in a well-behaved range.

---

## Final Answer

```text
(a) Unscaled:  A = 4,  B = 256
(b) Scaled:    A = 2,  B = 16
```

(c) Softmax weights depend exponentially on the **differences** between scores. Without scaling, high-dimensional dot products are large, so their differences are large, and softmax assigns almost all weight to the single highest score (a near-one-hot distribution) with vanishing gradients. Dividing by `√d_k` shrinks the scores and their gaps by the same factor, keeping softmax in a smooth regime where several keys receive weight and the model can keep learning.
