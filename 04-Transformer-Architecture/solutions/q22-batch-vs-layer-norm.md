# Q22: Layer Normalization — Analytic

## Question Summary

Explain the difference between **batch normalization** and **layer normalization**. Why is batch normalization problematic for NLP tasks where sentence lengths vary within a batch, while layer normalization works well? Give a concrete example with two sentences of different lengths.

---

## The Core Difference: Which Axis Is Normalized?

Both methods shift and scale values to mean 0 and variance 1. They differ only in **which group of numbers** the statistics are computed over.

Think of a batch of token vectors as a 3D block: **N** (sentences) × **L** (tokens) × **D** (features).

- **Batch norm** normalizes **each feature** using statistics computed **across the batch**, i.e., over all sentences (and, for sequences, all token positions).
- **Layer norm** normalizes **each token** using statistics computed **across its own D features**, ignoring every other token and sentence.

---

## Batch Normalization

For each feature dimension `d`, compute the mean and variance over the whole batch, then normalize every value in that dimension with those shared statistics.

**For images (CNNs):** every image in a batch has the same shape, so the statistics are always computed over consistent, real values.

**For NLP:** sentences have different lengths, so shorter sentences are **padded** to form a rectangular batch. The padding values enter the batch statistics and distort the mean and variance for every real token. The distortion also changes from batch to batch, depending on which sentences are grouped together and how much padding they need. In addition, batch norm depends on batch size and must switch to running averages at inference.

## Layer Normalization

For each token, compute the mean and variance over **that token's own feature vector**. Nothing outside the token is involved, so padding elsewhere in the batch is invisible, and the computation is identical for batch size 1 or 1,000.

---

## Concrete Example

**Batch of two sentences:**

- Sentence 1: "I love NLP" → 3 tokens
- Sentence 2: "Deep learning is great" → 4 tokens

Sentence 1 is padded to length 4 with a `[PAD]` vector of zeros. Embedding dimension D = 3.

| Position | Sentence 1 | Sentence 2 |
|---|---|---|
| Token 1 | [1.0, 2.0, 3.0] | [2.0, 4.0, 6.0] |
| Token 2 | [4.0, 5.0, 6.0] | [1.0, 3.0, 5.0] |
| Token 3 | [7.0, 8.0, 9.0] | [3.0, 6.0, 9.0] |
| Token 4 | [0.0, 0.0, 0.0] ← PAD | [2.0, 5.0, 8.0] |

### Batch norm: the padding distorts the statistics

For **feature 1**, batch norm uses the first value of every token in the batch, **including the PAD token**:

```text
{1.0, 4.0, 7.0, 0.0, 2.0, 1.0, 3.0, 2.0}      ← 0.0 is the PAD

μ  = 20 / 8 = 2.5
σ² = 34 / 8 = 4.25,   σ ≈ 2.062
```

Normalized feature 1 of Sentence 1, Token 1:

```text
x̂ = (1.0 − 2.5) / 2.062 ≈ −0.728
```

Without the PAD token, the statistics would be:

```text
{1.0, 4.0, 7.0, 2.0, 1.0, 3.0, 2.0}

μ  = 20 / 7 ≈ 2.857
σ² ≈ 3.837,   σ ≈ 1.959

x̂ = (1.0 − 2.857) / 1.959 ≈ −0.948
```

| Statistics | Mean | Variance | x̂ for S1 T1, feature 1 |
|---|---:|---:|---:|
| With PAD (what batch norm uses) | 2.500 | 4.250 | −0.728 |
| Real tokens only | 2.857 | 3.837 | −0.948 |

A single padding token changes the normalized value of a **real** token by more than 0.2. With longer length differences and more padding, the distortion grows, and it varies with every batch composition.

### Layer norm: each token uses only its own features

Sentence 1, Token 1 = `[1.0, 2.0, 3.0]`:

```text
μ  = (1 + 2 + 3) / 3 = 2.0
σ² = [(1−2)² + (2−2)² + (3−2)²] / 3 = 2/3 ≈ 0.667,   σ ≈ 0.8165
```

| Feature | Calculation | Result |
|---|---|---:|
| 1 | (1.0 − 2.0) / 0.8165 | −1.2247 |
| 2 | (2.0 − 2.0) / 0.8165 | 0.0000 |
| 3 | (3.0 − 2.0) / 0.8165 | +1.2247 |

```text
LayerNorm(S1, Token 1) = [−1.2247, 0.0, +1.2247]
```

This result is the same no matter what else is in the batch. The PAD token is normalized only with its own values (all zeros, so the output stays 0 thanks to ε). It never affects a real token, and attention masks ignore it downstream.

---

## Summary

| | Batch norm | Layer norm |
|---|---|---|
| Statistics over | One feature, across the batch | All features of one token |
| Affected by padding? | Yes, PAD values enter the statistics | No |
| Depends on batch composition and size? | Yes | No |
| Train vs. inference | Needs running averages | Identical |
| Typical use | CNNs on fixed-size images | Transformers and other sequence models |

---

## Final Answer

Batch normalization normalizes each feature using statistics computed across all examples in the batch, while layer normalization normalizes each token using statistics computed across its own feature vector. In NLP, sentences in a batch have different lengths and must be padded. Batch norm includes the padding values in its statistics, which distorts every real token's normalization. In the example, one PAD token moves the batch mean of feature 1 from 2.857 to 2.5 and changes a real token's normalized value from −0.948 to −0.728, and this distortion varies with each batch. Layer norm uses only the token's own values, so `[1, 2, 3]` always normalizes to `[−1.22, 0, 1.22]`, regardless of padding, other sentences, or batch size. This is why Transformer-based NLP models use layer normalization (or its RMSNorm variant) instead of batch normalization.
