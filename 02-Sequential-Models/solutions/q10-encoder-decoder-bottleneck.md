# Q10: Encoder-Decoder — Analytic

## Question Summary

In a basic encoder-decoder model (without attention), the decoder at step 3 is trying to produce the Bangla word **"boi"** (book). It only has access to the context vector C and its own previous states.

1. Why might the decoder struggle to produce "boi" if the input sentence is **40 words** long?
2. How does this relate to the empirical BLEU score degradation for sentences longer than 20–25 words?

---

## What the Decoder Sees at Step 3

When the decoder is about to predict "boi", it has only:

1. the **context vector C**, a fixed-size vector produced by the encoder, and
2. its **own previous states** s₁ and s₂, which encode that it has already generated "she" and "ekta".

It does **not** have access to the English words or to the encoder's intermediate states h₁, …, h₄₀. C is the only bridge from the source sentence.

---

## Short vs. Long Input

Think of C as a **suitcase of fixed size**. Once the encoder has packed it, the source sentence is discarded.

**5-word input ("She is reading a book"):** plenty of room. Who is reading, what is being read, and the tense all fit. At step 3, C still clearly encodes "book", and the decoder easily outputs "boi".

**40-word input:** the encoder updates its state 40 times and must squeeze everything into the same fixed-size C. "book" may have appeared at word 8 or 15, and the 25+ words that follow keep overwriting and mixing the state. By the end, the "book" information is diluted and blurred.

At step 3, the decoder asks, in effect, *"what is being read?"*, and C gives only a weak, fuzzy signal. The decoder may output a related but wrong word (e.g., *lekha* "writing" or *patro* "letter"), or drop the object altogether.

This is the **information bottleneck**: a fixed-capacity vector must represent an input of arbitrary length.

---

## Connection to BLEU Score Degradation

**BLEU** measures n-gram overlap between a model's translation and human reference translations; higher is better.

When researchers plotted BLEU against source sentence length for basic encoder-decoder models, they observed:

| Sentence length | Effect on C | BLEU |
|---|---|---|
| Short (< ~20 words) | C holds nearly all details | High and stable |
| ~20–25 words | C becomes crowded | Starts to decline |
| Long (> 25–30 words) | C loses many details | Drops sharply |

This pattern was reported by Cho et al. (2014) and shown clearly by Bahdanau et al. (2015, Fig. 2): the BLEU of the basic RNN encoder-decoder falls steeply as sentence length grows. The cause is the mechanism above. The longer the sentence, the more information must be compressed into the same fixed-size C, so more details, like the object "book", are lost and translations become less accurate.

Bahdanau et al. introduced **attention** to fix this. Instead of relying on C alone, the decoder computes a weighted sum over **all** encoder states at every step, so it can look back directly at the state for "book" when producing "boi". In their experiments, the attention model's BLEU stayed much more stable on long sentences.

---

## Final Answer

In a basic encoder-decoder, the whole source sentence must be compressed into a single fixed-size context vector C. For a 40-word sentence, the information that the object is "book" is diluted and overwritten by the many words processed after it. At step 3, the decoder receives only a weak signal about the object and may fail to produce "boi". The same bottleneck explains why BLEU scores of basic encoder-decoder models drop sharply for sentences longer than about 20–25 words (Cho et al., 2014; Bahdanau et al., 2015). The attention mechanism removed this bottleneck by letting the decoder access all encoder states directly.

---

## References

- Cho, K., van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). On the Properties of Neural Machine Translation: Encoder–Decoder Approaches. *SSST-8*.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
