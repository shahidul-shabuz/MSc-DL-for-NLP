# Q7: LSTM — Descriptive

## Question Summary

Explain the purpose of the three gates in an LSTM cell (forget, input, output) using the Bangla sentence:

> **"chobi-ta dekhte bhalo chilo kintu golpo-ta dubol chilo"**
> (The movie looked good but the story was weak)

How does the forget gate help the LSTM "shift focus" when it encounters **"kintu"** (but)?

---

## The Cell State: The LSTM's Conveyor Belt

An LSTM reads the sentence one word at a time and keeps two memories:

- the **hidden state** `h_t`: short-term, what the cell outputs at this step
- the **cell state** `C_t`: long-term, a "conveyor belt" that carries important information across the sentence

The three gates are learned controls that decide what stays on the belt, what is added, and what is shown to the next layer. Each gate looks at the previous hidden state `h_(t-1)` and the current word `x_t`, and outputs values between 0 and 1 through a sigmoid.

---

## 1. Forget Gate: What to Erase

```text
f_t = σ(W_f · [h_(t-1), x_t] + b_f)
C_t = f_t ⊙ C_(t-1) + ...
```

**Purpose:** decides which parts of the old cell state to keep (values near 1) or erase (values near 0).

**In our sentence:**

- While reading **"chobi-ta dekhte bhalo chilo"**, the LSTM builds a positive memory: *movie visuals = good*.
- Then it reads **"kintu"** (but), a contrast word signalling that what follows matters more for the overall meaning.
- The forget gate outputs **values close to 0** for the parts of the cell state that hold the earlier positive sentiment, and that memory is largely erased.

This is how the forget gate helps the LSTM **shift focus**. It clears the old positive signal from the conveyor belt so the model can concentrate on the new information in **"golpo-ta dubol chilo"**. Without it, the model would carry a confused mixture of "good movie" and "weak story".

---

## 2. Input Gate: What to Add

```text
i_t = σ(W_i · [h_(t-1), x_t] + b_i)
C̃_t = tanh(W_C · [h_(t-1), x_t] + b_C)
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

**Purpose:** decides what new information to write to the cell state. The sigmoid part chooses *which* values to update, and the tanh part creates the *candidate* values.

**In our sentence:** after "kintu", the words **"golpo-ta dubol chilo"** arrive. The input gate opens and writes the new negative information to the belt: *story = weak*. The cell state now holds "story is weak" instead of "movie looks good".

---

## 3. Output Gate: What to Show

```text
o_t = σ(W_o · [h_(t-1), x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**Purpose:** decides which parts of the updated cell state are exposed as the hidden state `h_t`, which goes to the next time step and the next layer.

**In our sentence:** at the end of the sentence, the output gate reads the updated cell state, which now holds the "story is weak" memory, and outputs it. For sentiment analysis, this leads to an overall **negative** prediction.

---

## Summary Table

| Gate | Role | In our sentence |
|---|---|---|
| Forget gate | Erases irrelevant old memory | At "kintu", erases the positive "movie looked good" memory, shifting focus |
| Input gate | Writes new information | Adds "story is weak" to the cell state |
| Output gate | Decides what to output now | Outputs the final negative sentiment |

---

## Final Answer

The **forget gate** removes outdated information from the cell state, the **input gate** adds new relevant information, and the **output gate** controls what the cell reveals as its hidden state. When the LSTM reads "kintu", the forget gate outputs values near 0 for the earlier positive sentiment and largely erases it. This lets the input gate store the new negative information about the weak story, so the final output reflects the sentence's actual overall meaning. This ability to reset memory at contrast words is something a vanilla RNN, with no gates, cannot do selectively.
