# Q12: Self-Attention — Exercise

## Question Summary

For the sentence **"I love NLP"** with `d_k = 2`:

| Word | Query | Key | Value |
|---|---|---|---|
| I | [1, 0] | [1, 1] | [1, 0] |
| love | [0, 1] | [0, 1] | [0, 1] |
| NLP | [1, 1] | [1, 0] | [1, 1] |

1. Compute the attention scores (Q × Kᵀ) for "NLP" against all three keys.
2. Scale by √d_k.
3. Apply softmax to get the attention weights.
4. Compute the new contextual embedding for "NLP" as the weighted sum of values.

---

## Part A: Raw Attention Scores

The query for "NLP" is `q = [1, 1]`. Take its dot product with every key:

```text
q · k("I")    = [1, 1] · [1, 1] = (1×1) + (1×1) = 2
q · k("love") = [1, 1] · [0, 1] = (1×0) + (1×1) = 1
q · k("NLP")  = [1, 1] · [1, 0] = (1×1) + (1×0) = 1
```

```text
Raw scores = [2, 1, 1]
```

---

## Part B: Scale by √d_k

```text
√d_k = √2 ≈ 1.4142

2 / 1.4142 ≈ 1.4142
1 / 1.4142 ≈ 0.7071
1 / 1.4142 ≈ 0.7071
```

```text
Scaled scores = [1.4142, 0.7071, 0.7071]
```

---

## Part C: Softmax

```text
α_i = exp(s_i) / Σ_j exp(s_j)
```

Exponentials:

```text
exp(1.4142) ≈ 4.1133
exp(0.7071) ≈ 2.0281
exp(0.7071) ≈ 2.0281

Sum ≈ 8.1695
```

Normalize:

```text
α("I")    = 4.1133 / 8.1695 ≈ 0.5035
α("love") = 2.0281 / 8.1695 ≈ 0.2483
α("NLP")  = 2.0281 / 8.1695 ≈ 0.2483
```

```text
Attention weights = [0.5035, 0.2483, 0.2483]   (sum = 1.0000)
```

"NLP" attends most to "I", because its query `[1, 1]` aligns best with the key `[1, 1]`.

---

## Part D: Weighted Sum of Values

```text
new("NLP") = 0.5035 · v("I") + 0.2483 · v("love") + 0.2483 · v("NLP")
           = 0.5035 · [1, 0] + 0.2483 · [0, 1] + 0.2483 · [1, 1]
```

Dimension by dimension:

```text
dim 1: (0.5035 × 1) + (0.2483 × 0) + (0.2483 × 1) = 0.7518
dim 2: (0.5035 × 0) + (0.2483 × 1) + (0.2483 × 1) = 0.4965
```

```text
New contextual embedding for "NLP" ≈ [0.7518, 0.4965]
```

---

## Final Answer

| Step | Result |
|---|---|
| (a) Raw scores | [2, 1, 1] |
| (b) Scaled scores | [1.4142, 0.7071, 0.7071] |
| (c) Attention weights | [0.5035, 0.2483, 0.2483] |
| (d) New embedding for "NLP" | [0.7518, 0.4965] |

The original value of "NLP" was `[1, 1]`. Its new embedding is a blend dominated by "I" (about 50%), with smaller contributions from "love" and "NLP" itself. This is how self-attention turns a fixed vector into a context-dependent one.
