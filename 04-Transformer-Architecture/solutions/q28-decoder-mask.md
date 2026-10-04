# Q28: Decoder Architecture — Exercise

## Question Summary

During training, the decoder receives the target sentence **"she ekta boi porchhe"** with a mask that prevents each position from attending to future positions.

1. Write out the 4×4 attention mask (1 = can attend, 0 = cannot attend).
2. Explain what would go wrong during training if the mask were removed.

---

## Part A: The Causal (Look-Ahead) Mask

Rows are **query** positions (the token doing the attending), and columns are **key** positions (the token being attended to):

|  | she | ekta | boi | porchhe |
|---|:---:|:---:|:---:|:---:|
| **she** | 1 | 0 | 0 | 0 |
| **ekta** | 1 | 1 | 0 | 0 |
| **boi** | 1 | 1 | 1 | 0 |
| **porchhe** | 1 | 1 | 1 | 1 |

```text
M = [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
```

Only the **lower triangle, including the diagonal,** is 1. Each position can attend to itself and earlier positions, never to later ones.

In practice, the 0 entries are implemented by adding **−∞** to those attention scores before the softmax, which makes their attention weights exactly 0.

---

## Part B: What Goes Wrong Without the Mask

1. **Information leakage.** Each position could attend to future target tokens. When predicting the word after "ekta", the model could look directly at "boi".

2. **The model "cheats" during training.** Predicting the next word becomes trivial copying, so the training loss falls quickly without the model learning how to generate a translation.

3. **No real autoregressive skill is learned.** The model never learns to predict the next token from **only** the previous tokens and the source sentence.

4. **Failure at inference.** At inference (greedy or beam search), the future tokens do not exist yet. A model that relied on seeing them faces a completely different situation and produces poor, repetitive, or broken output.

5. **Train–test mismatch.** The conditions at training time no longer match those at inference, so low training loss does not translate into good translations.

---

## Final Answer

```text
she      [1, 0, 0, 0]
ekta     [1, 1, 0, 0]
boi      [1, 1, 1, 0]
porchhe  [1, 1, 1, 1]
```

The mask is lower-triangular: each position may attend only to itself and earlier positions. If it were removed, every position could see the future target words, including the word it is supposed to predict. Training would become trivial copying (information leakage), the model would not learn genuine next-token prediction, and at inference, where future words are unavailable, it would fail because of the train–test mismatch. The causal mask prevents information from leaking from future tokens.
