# Q11: Self-Attention — Descriptive

## Question Summary

Using the sentence **"The cat sat on the mat"**, explain what self-attention does that a static word embedding (like Word2Vec) cannot. The word "The" appears at position 1 and position 5. How does self-attention produce **different contextual embeddings** for these two identical words?

---

## What a Static Embedding Cannot Do

Word2Vec assigns **one fixed vector per word type**, learned once and never changed:

```text
embed("The") = [0.1, 0.4, -0.2, ...]   ← position 1
embed("the") = [0.1, 0.4, -0.2, ...]   ← position 5, identical
```

It does not matter that the first "The" belongs to "cat" and the second to "mat"; both get exactly the same vector. A static embedding captures a word's **average meaning across all contexts**, not its meaning in this particular sentence.

---

## What Self-Attention Does Differently

Self-attention lets every word look at every other word and build a new representation as a **relevance-weighted average** of the whole sentence:

```text
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Each input vector `x_i` is projected into three vectors:

- **Query** `q_i = x_i W_Q`: what am I looking for?
- **Key** `k_i = x_i W_K`: what do I offer to others?
- **Value** `v_i = x_i W_V`: what information do I contribute?

The output for word i is `Σ_j α_ij v_j`, where `α_ij` is the softmax of `q_i · k_j / √d_k`. Because this output mixes in information from the surrounding words, it is a **contextual** embedding.

---

## How the Two "The"s Become Different

### An important subtlety: position information is required

If the two "The"s entered self-attention with **identical** vectors, they would produce **identical** outputs. Both would have the same query, compare against the **same set of keys** (every word in the sentence), and get the same attention weights. Self-attention on its own has no notion of "next to".

What makes them differ is **positional encoding**, which is added to each embedding before the first attention layer:

```text
x₁ = embed("The") + PE(1)
x₅ = embed("the") + PE(5)      →   x₁ ≠ x₅
```

Now the two tokens have **different queries**, so they match the keys differently. In a trained model, the queries learn to favor nearby words (combining what a word is with where it is).

### Diverging attention patterns

With positions encoded, each "The" attends most strongly to the noun it introduces:

| Key word | Position | Weight from "The" (pos 1) | Weight from "the" (pos 5) |
|---|---:|---:|---:|
| The | 1 | 0.10 | 0.04 |
| **cat** | 2 | **0.55** | 0.04 |
| sat | 3 | 0.20 | 0.08 |
| on | 4 | 0.05 | 0.21 |
| the | 5 | 0.05 | 0.10 |
| **mat** | 6 | 0.05 | **0.53** |

*Illustrative weights, chosen to show the typical pattern; each column sums to 1.*

### Different weighted sums, different outputs

```text
output("The", pos 1) = 0.10·v(The) + 0.55·v(cat) + 0.20·v(sat) + ...   → shaped by "cat"
output("the", pos 5) = 0.04·v(The) + 0.21·v(on)  + 0.53·v(mat) + ...   → shaped by "mat"
```

The first "The" becomes "the article introducing *cat*", and the second becomes "the article introducing *mat*". After further layers, these vectors also carry the different syntactic roles of their noun phrases: subject (*the cat*) versus object of the preposition (*on the mat*).

---

## Comparison

| | Word2Vec (static) | Self-attention (contextual) |
|---|---|---|
| "The" at position 1 | Fixed vector | Vector shaped mainly by "cat" |
| "the" at position 5 | Same fixed vector | Vector shaped mainly by "mat" |
| Same representation? | Always | No, it depends on context |

---

## Final Answer

A static embedding like Word2Vec gives every occurrence of a word the same vector, regardless of context, so "The" at position 1 and position 5 are indistinguishable. Self-attention replaces each word's vector with a weighted average of the value vectors of **all** words in the sentence, with weights from query–key similarity. Because positional encodings make the two "The"s' input vectors differ, their queries produce different attention patterns: the first attends mainly to "cat" and the second to "mat". Their output vectors therefore become different **contextual embeddings**, each reflecting the noun phrase it belongs to.

---

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.
