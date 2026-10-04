# Q27: Decoder Architecture — Descriptive

## Question Summary

The Transformer decoder has three sub-layers per layer, compared to the encoder's two.

1. Name all three and explain the purpose of each.
2. Why does the decoder need **masked** self-attention instead of regular self-attention?
3. The decoder has generated **"she ekta"** and is about to generate **"boi"**. What should masked self-attention prevent?

---

## The Three Decoder Sub-Layers

The decoder generates the output sequence **autoregressively**, one token at a time. Each decoder layer contains:

### 1. Masked multi-head self-attention

**Purpose:** lets each target position attend to the tokens generated **so far**, and only those. This builds an understanding of the partial translation.

### 2. Encoder–decoder attention (cross-attention)

**Purpose:** connects the decoder to the source sentence. Queries come from the decoder; keys and values come from the **encoder output**. At each step, the decoder can focus on the source words most relevant to the word it is about to produce.

### 3. Position-wise feed-forward network

**Purpose:** applies the same two-layer network (two linear transformations with a non-linearity such as ReLU) to each position independently, adding processing capacity on top of what the attention layers gathered.

Each sub-layer is wrapped in **Add & Norm** (residual connection + layer normalization), as in the encoder.

| Encoder layer | Decoder layer |
|---|---|
| Self-attention (unmasked) | **Masked** self-attention |
| — | **Cross-attention** to the encoder output |
| Feed-forward | Feed-forward |

---

## Why Masked Self-Attention?

- **Regular self-attention** lets every token see every other token, including later ones. This is correct in the **encoder**, because the whole source sentence is known at once.
- **Masked self-attention** blocks each position from seeing **future** positions. The decoder produces one word at a time, and at inference the future words do not exist yet.

During training, the whole target sentence is fed in at once for efficiency (teacher forcing). Without a mask, each position could simply look at the next word it is supposed to predict. The model would **"cheat"**, copying the answer instead of learning to generate it.

The mask is applied by setting the attention scores of future positions to **−∞** before the softmax, so their weights become exactly 0.

---

## Illustration: Predicting "boi"

Target sentence: **"she ekta boi porchhe"** (She is reading a book)

- **Current state:** the decoder has generated `[<SOS>, she, ekta]`.
- **Next target:** `boi`

| Position | Token | Can attend to | Masked from |
|---:|---|---|---|
| 1 | `<SOS>` | `<SOS>` | she, ekta, boi, … |
| 2 | she | `<SOS>`, she | ekta, boi, … |
| 3 | ekta | `<SOS>`, she, ekta | **boi**, … |

**What masked self-attention prevents:** when the decoder computes the representation at "ekta", which is used to predict the next word, the mask stops it from incorporating **any information about "boi"** or any later token.

Without the mask, the model would see during training that "boi" comes next and copy it, rather than learning that "she ekta" (she … a) should be followed by a noun such as "boi", using cross-attention to "book" in the source. At inference, where no future tokens exist, such a model would fail.

---

## Final Answer

Each decoder layer has (1) **masked multi-head self-attention**, which attends only to previously generated target tokens; (2) **encoder–decoder cross-attention**, which uses the decoder's queries against the encoder's keys and values to focus on relevant source words; and (3) a **position-wise feed-forward network**, each wrapped in Add & Norm. The decoder needs masking because it generates text left to right. During training, the full target is given at once, so without a mask each position could see the very word it must predict. When the decoder has produced "she ekta" and must predict "boi", the mask sets the attention scores to "boi" and all later positions to −∞. The prediction must then come only from "<SOS> she ekta" and the source sentence, not from peeking at the answer.
