# Sample Questions: Sequential Models

This file contains the professor-provided sample questions for the **Sequential Models** module (RNN, LSTM and encoder-decoder) of the **Deep Learning for NLP** course.

---

## Q5. RNN — Descriptive

Draw a simple diagram showing how a vanilla RNN processes the English sentence “I love NLP” one token at a time. Label the hidden states h₁, h₂, h₃ and show how each depends on the previous. Then explain why this architecture struggles when processing the sentence “The student who studied hard for the exam that was scheduled for Friday finally passed” — specifically, what information is likely lost by the time the RNN reaches “passed”?

---

## Q6. RNN — Exercise

An RNN cell has the following parameters: W_hh = [[0.5, 0.0], [0.0, 0.5]], W_xh = [[1.0], [0.0]], and uses tanh activation. The initial hidden state is h₀ = [0, 0]. For input sequence x₁ = 1.0, x₂ = 2.0:

**(a)** Compute h₁ = tanh(W_hh · h₀ + W_xh · x₁)  
**(b)** Compute h₂ = tanh(W_hh · h₁ + W_xh · x₂)  
**(c)** What do you observe about the information from x₁ in h₂?

---

## Q7. LSTM — Descriptive

Explain the purpose of each of the three gates in an LSTM cell (forget gate, input gate, output gate) using the following Bangla sentence as context: “chobi-ta dekhte bhalo chilo kintu golpo-ta dubol chilo” (The movie looked good but the story was weak). How does the forget gate help the LSTM “shift focus” when it encounters “kintu” (but)?

---

## Q8. LSTM — Analytic

Compare vanilla RNN and LSTM on the task of translating the English sentence “The woman who lives next to the bakery that opened last year runs every morning” into Bangla. Why would an RNN likely fail to correctly translate “runs” (as referring to “woman”), while an LSTM would have a better chance? Relate your answer to the cell state mechanism.

---

## Q9. Encoder-Decoder — Descriptive

Draw the basic encoder-decoder architecture for translating the English sentence “She is reading a book” into Bangla “she ekta boi porchhe”. Label the encoder hidden states h₁ through h₅, the context vector C, the decoder states s₁ through s₄, and the softmax output at each decoder step. Where exactly does the information bottleneck occur in this architecture?

---

## Q10. Encoder-Decoder — Analytic

In a basic encoder-decoder model (without attention), the decoder at step 3 is trying to produce the Bangla word “boi” (book). It only has access to the context vector C and its own previous states. Explain why the decoder might struggle to produce “boi” correctly if the input sentence is 40 words long. How does this relate to the BLEU score degradation observed empirically for sentences longer than 20-25 words?
