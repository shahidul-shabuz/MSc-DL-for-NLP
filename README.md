# Deep Learning for NLP — MSc Coursework Notes

Worked solutions and mathematical notes from the **Deep Learning for NLP** course, M.Sc. in CSE, Dhaka University of Engineering & Technology (DUET), 1st semester (2026).

The solutions work through the architecture behind modern NLP, from backpropagation to the Transformer decoder. Each one includes step-by-step derivations, worked numerical examples and diagrams, with Bangla NLP tasks as running examples.

## Progress

| Module | Topics | Questions | Status |
|---|---|---|---|
| [01 Fundamentals](01-Fundamentals/) | Backpropagation, gradient descent, activation functions, loss functions, softmax | Q1–Q4 | ✅ Complete |
| [02 Sequential Models](02-Sequential-Models/) | RNN, LSTM, encoder-decoder, information bottleneck | Q5–Q10 | ✅ Complete |
| [03 Attention Mechanisms](03-Attention-Mechanisms/) | Self-attention, Q/K/V, scaling by √d_k, multi-head attention | Q11–Q18 | ✅ Complete |
| [04 Transformer Architecture](04-Transformer-Architecture/) | Positional encoding, LayerNorm, residual connections, encoder and decoder | Q19–Q30 | ✅ Complete |

## Math covered

- **Optimization:** chain rule and backpropagation, MSE gradients for a single neuron, effect of the learning rate
- **Activations and losses:** sigmoid derivative bound (σ′ ≤ 0.25) and vanishing gradients, ReLU and dying ReLU, softmax with categorical cross-entropy
- **Recurrent models:** RNN hidden-state recurrence (worked example), LSTM gate equations, why the additive cell-state update preserves gradients
- **Seq2seq:** the context-vector bottleneck and its link to BLEU degradation on long sentences
- **Attention:** scaled dot-product attention (worked example), Var(q·k) = d_k and the √d_k correction, the softmax Jacobian under saturation, why Q = K = V = x fails
- **Multi-head attention:** parameter count (4·d_model² = 1,048,576 for d_model = 512) and why h heads of size d_model/h cost the same as one full head
- **Transformer components:** sinusoidal positional encoding (worked example), LayerNorm vs. BatchNorm with padding (worked example), the residual gradient ∂L/∂x_l = ∂L/∂x_(l+1)·(1 + ∂F/∂x_l), the causal decoder mask
- **Comparison:** additive (Bahdanau) vs. scaled dot-product attention, and cross-attention vs. self-attention

## Repository structure

```
MSc-DL-for-NLP/
├── 01-Fundamentals/
├── 02-Sequential-Models/
├── 03-Attention-Mechanisms/
├── 04-Transformer-Architecture/
│   ├── questions/     # Course sample questions
│   ├── solutions/     # Worked solutions, one file per question
│   └── figures/       # Diagrams used in the solutions
└── LICENSE
```

Each module follows the same `questions/` · `solutions/` · `figures/` layout.

## How to use

All content is Markdown and renders directly on GitHub, including equations and Mermaid diagrams; no setup is needed. Start with [01-Fundamentals](01-Fundamentals/), since later modules build on backpropagation and the chain rule.

## A note on preparation

Some explanations and diagrams were drafted with the help of AI tools (Claude, Gemini), including the solutions to Q29–Q30. All content was then reviewed, corrected and edited by me, and numerical results were re-computed independently.

## Author

**Md Shahidul Islam Shabuz** — Assistant Professor, Dept. of CSE, RSTU; M.Sc. candidate, DUET
[Website](https://shahidul-shabuz.github.io/) · [Google Scholar](https://scholar.google.com/citations?user=VDtX90sAAAAJ&hl=en) · [ORCID](https://orcid.org/0009-0008-8271-888X)

## License

[MIT](LICENSE)
