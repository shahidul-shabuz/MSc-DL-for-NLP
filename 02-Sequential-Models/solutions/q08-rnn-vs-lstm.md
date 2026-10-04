# Q8: LSTM — Analytic

## Question Summary

Compare a vanilla RNN and an LSTM on translating this sentence into Bangla:

> "The woman who lives next to the bakery that opened last year runs every morning"

Why would an RNN likely fail to translate "runs" correctly (as referring to "woman"), while an LSTM would have a better chance? Relate the answer to the cell state mechanism.

---

## The Challenge: A Long-Range Dependency

```text
The woman [who lives next to the bakery that opened last year] runs every morning
     ↑                                                           ↑
  subject                10-word relative clause               verb
```

- The subject is **"woman"** (singular, third person).
- The verb **"runs"** must agree with it.
- A **10-word** relative clause sits between them, containing two distracting nouns: "bakery" and "year".

Bangla verbs are conjugated by person and honorific level (e.g., *mohila-ti protidin sokale douray / douran*). To pick the right verb form, the model must still remember at "runs" that the subject is "the woman".

---

## Why the Vanilla RNN Fails

A vanilla RNN has a **single hidden state** that is rewritten at every step:

```text
h_t = tanh(W · h_(t-1) + U · x_t + b)
```

1. At **"The woman"**, it stores *subject = woman* in its hidden state.
2. It then reads the 10-word clause. Each word is mixed into the same hidden state through tanh, gradually overwriting the earlier content.
3. At **"runs"**, the "woman" information has been diluted. The most recent nouns, "bakery" and "year", dominate the hidden state.

During training, the same problem appears as **vanishing gradients**. The error signal from "runs" must pass back through about 10 multiplications by derivatives smaller than 1, so the network never learns to keep the subject alive that long.

**Result:** the RNN may treat "bakery" or "year" as the subject, producing a wrong verb form or a garbled translation.

---

## Why the LSTM Succeeds: The Cell State

An LSTM adds a separate **cell state** `C_t`, a protected conveyor belt controlled by gates:

```text
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

The update is **additive**, so information is not forced through a squashing nonlinearity at every step.

1. **Capturing the subject (input gate).** At "The woman", the input gate writes *singular, third-person subject = woman* to the cell state.
2. **Navigating the gap (forget gate).** During "who lives next to the bakery that opened last year", the forget gate stays **close to 1** for the subject information, and the input gate does not overwrite it. The clause's nouns are treated as description, not as a new subject.
3. **Translating the verb (output gate).** At "runs", the output gate exposes the preserved subject information, and the decoder chooses the Bangla verb form that agrees with "woman".

Because `∂C_t/∂C_(t-1) = f_t`, the gradient through the cell state is scaled by the forget gate rather than by repeated tanh derivatives. When `f_t ≈ 1`, the gradient flows back across the gap almost unchanged.

---

## Comparison

| Aspect | Vanilla RNN | LSTM |
|---|---|---|
| Memory | One hidden state, overwritten each step | Separate cell state protected by gates |
| Update | Multiplicative, through tanh | Additive, controlled by the forget gate |
| 10-word gap | "woman" fades; recent nouns dominate | "woman" preserved on the cell state |
| Gradient across the gap | Vanishes | Preserved when `f_t ≈ 1` |
| Translating "runs" | Likely wrong subject or agreement | Correct agreement with "woman" |

---

## Final Answer

A vanilla RNN stores everything in a single hidden state that is overwritten at every word, so after the 10-word clause the information that "woman" is the subject has faded, and its gradient has vanished during training. It is therefore likely to mistranslate "runs". An LSTM's **cell state** acts as a protected memory path: the input gate writes the subject, the forget gate keeps it (values near 1) across the distracting clause, and the output gate retrieves it at "runs". The additive cell-state update also preserves the gradient across the gap, so the LSTM can learn this long-range dependency and produce the correct Bangla verb form.
