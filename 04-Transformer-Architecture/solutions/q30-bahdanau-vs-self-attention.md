# Q30: Cross-Topic — Analytic

## Question Summary

A student claims: *"Self-attention in Transformers is just a more parallel version of the Bahdanau attention mechanism — they do the same thing."*

Evaluate this claim. Identify at least three specific differences between Bahdanau (cross-)attention and Transformer self-attention in terms of:

- (a) what attends to what,
- (b) how scores are computed,
- (c) the role of learnable parameters (W_Q, W_K, W_V vs. W_a, U_a, v).

Is the claim correct, partially correct, or incorrect?

---

## The Two Mechanisms

### Bahdanau attention (2015)

Bahdanau et al. added attention to an RNN encoder-decoder for translation. A bidirectional RNN encoder produces annotations `h_1, …, h_n`, one per source word. At decoder step `i`, using the previous decoder state `s_(i−1)`:

```text
score:    e_ij = vᵀ · tanh(W_a · s_(i−1) + U_a · h_j)
weights:  α_ij = softmax_j(e_ij)
context:  c_i  = Σ_j α_ij · h_j
```

The context vector `c_i` is then used to compute the next decoder state and predict the next word.

### Transformer self-attention (2017)

For a sequence with input vectors `x_1, …, x_n`:

```text
q_i = x_i W_Q,   k_j = x_j W_K,   v_j = x_j W_V

score:    e_ij = (q_i · k_j) / √d_k
weights:  α_ij = softmax_j(e_ij)
output:   z_i  = Σ_j α_ij · v_j
```

This runs in parallel for all positions `i`, with multiple heads.

---

## What They Have in Common

Both mechanisms follow the same **general recipe**:

1. compute a relevance **score** between a "query" and each "key",
2. normalize the scores with **softmax**,
3. return a **weighted sum** of vectors.

In this broad sense, the student is right that they belong to the same family. But the details differ in important ways.

---

## Difference (a): What Attends to What

| | Bahdanau attention | Transformer self-attention |
|---|---|---|
| Type | **Cross-attention** between two sequences | **Self-attention** within one sequence |
| Query | Decoder state `s_(i−1)` (target side) | Each token `x_i` of the same sequence |
| Keys / values | Encoder annotations `h_j` (source side) | All tokens `x_j` of the same sequence |
| Purpose | **Align** each target word with source words | Build **contextual representations** of every token |

Bahdanau attention answers *"which source words matter for the target word I am producing now?"* Self-attention answers *"which other words in this same sentence help me represent this word?"* For example, it links "khai" to "ami" and "bhat" in "ami bhat khai" (Q14). Self-attention is used in the encoder, where Bahdanau's model has **no attention at all**: its encoder is purely recurrent.

---

## Difference (b): How Scores Are Computed

| | Bahdanau | Transformer |
|---|---|---|
| Score function | **Additive**: `vᵀ tanh(W_a s + U_a h)` | **Scaled dot product**: `q·k / √d_k` |
| Mechanism | A small feed-forward network with a tanh hidden layer | A single dot product |
| Scaling | Not needed | Divide by `√d_k` to prevent softmax saturation (Q15, Q16) |
| Efficiency | Slower; each pair passes through a tanh layer | Very fast; all scores computed as one matrix product `QKᵀ` |

Vaswani et al. note that the two have similar theoretical complexity, but dot-product attention is much faster and more memory-efficient in practice, because it uses highly optimized matrix multiplication. They also report that, without scaling, additive attention outperforms dot-product attention for large `d_k`, which is exactly why the `√d_k` scaling was introduced.

---

## Difference (c): The Role of Learnable Parameters

| | Bahdanau: W_a, U_a, v | Transformer: W_Q, W_K, W_V (and W_O) |
|---|---|---|
| What the parameters do | Define the **scoring network** only | Define three separate **roles** for every token |
| Query transformation | `W_a` transforms the decoder state | `W_Q` |
| Key transformation | `U_a` transforms encoder states | `W_K` |
| Value transformation | **None**: the context is a weighted sum of the raw annotations `h_j` | `W_V` creates a separate value vector |
| Final projection | `v` reduces the tanh output to a scalar score | `W_O` combines the heads |
| Number of heads | One | Multiple (e.g., 8), each with its own projections |

In Bahdanau attention, the same vector `h_j` serves as both the thing being matched (key) and the thing being returned (value). The Transformer separates **query, key and value** into distinct learned projections, and runs **multiple heads** so the model can learn several kinds of relationships at once (Q13, Q17).

---

## Further Differences

**Parallelism and recurrence.** Bahdanau attention sits inside an RNN. The query `s_(i−1)` exists only after the previous step is finished, and the encoder annotations themselves come from a sequential RNN. Self-attention computes all positions **simultaneously**, with no recurrence. This is a real difference, but it follows from the architectural changes above, not from simply making Bahdanau attention parallel.

**Source of word order.** In Bahdanau's model, the RNN provides word order. Self-attention is order-blind and needs **positional encoding** (Q19).

**Role in the model.** In Bahdanau's model, attention is an **add-on** to an RNN, which still does the main sequence modelling. In the Transformer, attention **replaces** recurrence and is the main building block of both encoder and decoder.

---

## The Closest Analogue

The Transformer component that most resembles Bahdanau attention is not self-attention but **encoder–decoder cross-attention** in the decoder (Q27). There, decoder queries attend to encoder keys and values, which is the same alignment role Bahdanau attention plays, though it still uses scaled dot products, separate Q/K/V projections and multiple heads.

---

## Verdict: Partially Correct

**What is right:**

- Both compute softmax-weighted sums based on learned relevance scores; they belong to the same family of attention mechanisms.
- Transformer attention is far more parallel than RNN-based attention.

**What is wrong:**

- **(a)** They attend over different things: Bahdanau aligns *target to source* (cross-attention), while self-attention relates tokens *within one sequence*.
- **(b)** They score differently: an additive tanh network vs. a scaled dot product.
- **(c)** Their parameters play different roles: Bahdanau has no value projection and a single head; self-attention has separate Q/K/V projections and multiple heads.
- Parallelism is a **consequence** of removing recurrence, not the only difference.

So "they do the same thing" is too strong. A more accurate statement: *"Transformer self-attention generalizes the attention idea that Bahdanau introduced, applying it within a sequence, with a different scoring function, separate Q/K/V projections and multiple heads, and without recurrence. The Transformer's cross-attention is the direct descendant of Bahdanau attention."*

---

## Final Answer

The claim is **partially correct**. Both mechanisms compute relevance scores, normalize them with softmax and return a weighted sum, and Transformer attention is much more parallel. However, they differ in at least three ways: (a) **what attends to what**: Bahdanau attention is cross-attention from the decoder state to the encoder states, used for alignment, while self-attention relates each token to all tokens in the same sequence; (b) **scoring**: Bahdanau uses an additive network `vᵀ tanh(W_a s + U_a h)`, while the Transformer uses a scaled dot product `q·k/√d_k`; (c) **parameters**: W_a, U_a and v only define the scoring network and the raw encoder states serve as values, while W_Q, W_K, W_V create separate query, key and value vectors across multiple heads. Bahdanau attention's closest Transformer counterpart is the decoder's encoder–decoder cross-attention, not self-attention.

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*, Sections 3.2.1 and 4.
