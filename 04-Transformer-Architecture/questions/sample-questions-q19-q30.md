# Sample Questions: Transformer Architecture

This file contains the professor-provided sample questions for the **Transformer Architecture** module (positional encoding, layer normalization, residual connections, encoder and decoder architecture) of the **Deep Learning for NLP** course.

---

## Q19. Positional Encoding — Descriptive

Explain why a Transformer without positional encoding would produce the exact same output for “kukur biralke tara korchhe” (The dog is chasing the cat) and “biral kukurke tara korchhe” (The cat is chasing the dog). Then explain how sinusoidal positional encoding solves this.

---

## Q20. Positional Encoding — Exercise

Compute the positional encoding vector for position pos = 2 with d_model = 4. Use the formulas:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Show your calculations for all 4 dimensions (i = 0 and i = 1). Why do the first two dimensions change more rapidly across positions than the last two?

---

## Q21. Layer Normalization — Descriptive

Explain what layer normalization does in a Transformer. Given a vector z = [2.0, 4.0, 6.0, 8.0] representing the output of a self-attention layer for one token:

**(a)** Compute the mean and variance of z.  
**(b)** Normalize z to have mean 0 and variance 1.  
**(c)** Why is this normalization applied after each sub-layer (self-attention and feed-forward) in the Transformer? How does it help with training stability?

---

## Q22. Layer Normalization — Analytic

Explain the difference between batch normalization and layer normalization. Why is batch normalization problematic for NLP tasks where sentence lengths vary across a batch, while layer normalization works well? Give a concrete example with two sentences of different lengths in a batch.

---

## Q23. Residual Connections — Descriptive

In the Transformer, the output of each sub-layer is computed as LayerNorm(x + SubLayer(x)) rather than just SubLayer(x). Using a concrete example where the input embedding for “cat” is x = [0.5, 1.2, -0.3, 0.8] and the self-attention output is SubLayer(x) = [0.1, -0.4, 0.6, 0.2]:

**(a)** Compute x + SubLayer(x).  
**(b)** Explain why adding the original x back (the “residual connection”) helps prevent the vanishing gradient problem in a 6-layer Transformer encoder.  
**(c)** What would happen during training if we stacked 6 layers without residual connections?

---

## Q24. Residual Connections — Analytic

During backpropagation through a Transformer with residual connections, the gradient of the loss with respect to the input of layer l can be written as ∂L/∂x_l = ∂L/∂x_{l+1} × (1 + ∂SubLayer/∂x_l). Explain the significance of the “1 +” term. How does it guarantee that gradients never completely vanish even if ∂SubLayer/∂x_l approaches zero?

---

## Q25. Encoder Architecture — Descriptive

Draw the complete architecture of a single Transformer encoder layer, labeling all components: input embeddings, positional encoding, multi-head self-attention, Add & Norm (residual + layer normalization), feed-forward network, and the second Add & Norm. Trace how the Bangla sentence “ami bhalo achi” (I am fine) would flow through this layer, describing what happens at each step.

---

## Q26. Encoder Architecture — Analytic

The Transformer encoder uses 6 identical layers stacked on top of each other. Explain what happens to the representation of a word like “bhalo” (good/fine) as it passes through layers 1 → 2 → … → 6. How does each successive layer refine the word’s representation? Draw an analogy to how deep CNNs learn low-level to high-level features.

---

## Q27. Decoder Architecture — Descriptive

The Transformer decoder has three sub-layers per layer, compared to the encoder’s two. Name all three sub-layers and explain the purpose of each. Why does the decoder need masked selfattention instead of regular self-attention? Illustrate with the partial translation where the decoder has generated “she ekta” and is about to generate “boi” — what should the masked selfattention prevent?

---

## Q28. Decoder Architecture — Exercise

During training, the decoder receives the target sentence “she ekta boi porchhe” with a mask that prevents each position from attending to future positions. Write out the 4×4 attention mask matrix (1 = can attend, 0 = cannot attend) for this 4-word target. Then explain what would go wrong during training if this mask were removed (i.e., if the decoder could see the entire target at once).

---

## Q29. Full Pipeline — Analytic

Trace the complete journey of translating the English sentence “She is reading a book” to Bangla “she ekta boi porchhe” through a Transformer model. For each of the following stages, write one sentence describing what happens:

**(a)** Input tokenization and embedding  
**(b)** Positional encoding addition  
**(c)** Encoder self-attention  
**(d)** Encoder feed-forward + residual + layer norm  
**(e)** Decoder masked self-attention (at step 3, producing “boi”)  
**(f)** Decoder cross-attention (at step 3)  
**(g)** Decoder feed-forward + residual + layer norm  
**(h)** Output linear layer + softmax

---

## Q30. Cross-Topic — Analytic

A student claims: “Self-attention in Transformers is just a more parallel version of the Bahdanau attention mechanism — they do the same thing.” Evaluate this claim. Identify at least three specific differences between Bahdanau (cross-)attention and Transformer self-attention in terms of:

**(a)** what attends to what, (b) how scores are computed, and (c) the role of learnable parameters (W_Q, W_K, W_V vs W_a, U_a, v). Is the student’s claim correct, partially correct, or incorrect? Justify your answer.
