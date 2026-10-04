# Sample Questions: Attention Mechanisms

This file contains the professor-provided sample questions for the **Attention Mechanisms** module (self-attention, Q/K/V, scaling and multi-head attention) of the **Deep Learning for NLP** course.

---

## Q11. Self-Attention — Descriptive

Using the sentence “The cat sat on the mat”, explain what self-attention does that a static word embedding (like Word2Vec) cannot. Specifically, the word “The” appears at position 1 and position 5 — how does self-attention produce different contextual embeddings for these two identical words?

---

## Q12. Self-Attention — Exercise

Given the sentence “I love NLP” with the following simplified Q, K, V vectors (d_k = 2):

```text
q("I")    = [1, 0]   k("I")    = [1, 1]   v("I")    = [1, 0]
q("love") = [0, 1]   k("love") = [0, 1]   v("love") = [0, 1]
q("NLP")  = [1, 1]   k("NLP")  = [1, 0]   v("NLP")  = [1, 1]
```

**(a)** Compute the attention scores (Q × Kᵀ) for the word “NLP” against all three keys.  
**(b)** Scale by √d_k.  
**(c)** Apply softmax to get attention weights.  
**(d)** Compute the new contextual embedding for “NLP” as the weighted sum of values.

---

## Q13. Linear Transformation in Self-Attention — Analytic

In self-attention, the input embedding x is multiplied by three separate weight matrices W_Q, W_K, and W_V to produce Query, Key, and Value vectors. Explain why using the raw embedding x directly (without these linear transformations) for all three roles would severely limit what self-attention can learn. What would happen to the attention weights if Q = K = V = x?

---

## Q14. Query, Key, Value — Descriptive

Using a library search analogy, explain the roles of Query, Key, and Value in self-attention. Then map this analogy to the sentence “ami bhat khai” (I eat rice) — when self-attention is processing the word “khai” (eat), what is the Query asking, what Keys is it comparing against, and what Values does it collect?

---

## Q15. Scaling by √d_k — Exercise

Consider two scenarios for computing attention scores between a query q and a key k, both of dimension d_k:

- **Scenario A:** d_k = 4, q = [1, 1, 1, 1], k = [1, 1, 1, 1]
- **Scenario B:** d_k = 256, q and k are all-ones vectors of length 256.

**(a)** Compute the unscaled dot product q · k for both scenarios.  
**(b)** Compute the scaled dot product (q · k) / √d_k for both.  
**(c)** Explain why the softmax function would produce near-one-hot outputs for Scenario B without scaling, and how scaling prevents this.

---

## Q16. Scaling by √d_k — Analytic

The original Transformer paper states: “We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.” Explain this statement in your own words. What would happen to backpropagation if the softmax outputs were nearly one-hot (all zeros except one)?

---

## Q17. Multi-Head Attention — Descriptive

In the sentence “She gave him the book because he asked for it”, multiple types of relationships exist simultaneously. Describe at least three different linguistic relationships that three separate attention heads might learn. Why can’t a single attention head capture all of these effectively?

---

## Q18. Multi-Head Attention — Exercise

A Transformer has d_model = 512 and uses h = 8 attention heads.

**(a)** What is the dimension d_k of each head’s Query and Key vectors?  
**(b)** How many total learnable parameters are in the W_Q, W_K, W_V, and W_O matrices combined for one multi-head attention layer? (Assume d_v = d_k and ignore biases.)  
**(c)** Show that the total computational cost of 8 heads with d_k = 64 is approximately the same as a single head with d_k = 512.
