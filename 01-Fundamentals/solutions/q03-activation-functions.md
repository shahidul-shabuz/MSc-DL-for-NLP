# Q3: Activation Functions — Analytic

## Question Summary

Consider a deep network for Bangla named entity recognition, or NER.

The task is to explain:

1. why using sigmoid activation in every hidden layer can cause the vanishing gradient problem,
2. how ReLU helps solve this problem,
3. what limitation ReLU introduces, especially the dying ReLU problem.

---

## Solution

Activation functions are very important in deep neural networks because they decide how information and gradients flow through the model.

In a Bangla NER system, the model may need to recognize names of people, places, organizations, or locations from a sequence of Bangla words. For this, the model must learn useful patterns from different layers.

If the activation function causes gradients to become very small, the earlier layers of the network will not learn properly.

---

## Why Sigmoid Causes the Vanishing Gradient Problem

The sigmoid activation function is:

```text
σ(x) = 1 / (1 + e^(-x))
```

It converts any input value into a number between `0` and `1`.

The derivative of sigmoid is:

```text
σ'(x) = σ(x)(1 - σ(x))
```

The maximum value of this derivative is only:

```text
0.25
```

This means that during backpropagation, the gradient passed through a sigmoid neuron is always small.

---

## Gradient Shrinking Across Layers

In deep networks, backpropagation uses the chain rule. Gradients are multiplied layer by layer.

For example:

```text
Total gradient = gradient_L × gradient_(L-1) × ... × gradient_1
```

If each sigmoid layer contributes a gradient less than or equal to `0.25`, then in a 10-layer network:

```text
0.25^10 ≈ 0.000001
```

This value is almost zero.

As a result, the earlier layers receive a very weak gradient signal.

This means:

- weights in earlier layers update very slowly,
- the model learns poorly,
- long-range patterns become difficult to capture,
- training becomes unstable or very slow.

In Bangla NER, this can be harmful because the model may need to identify an entity using surrounding words and context. If early layers do not learn useful word-level patterns, the final NER prediction will be weak.

---

## How ReLU Addresses This Problem

The ReLU activation function is:

```text
ReLU(x) = max(0, x)
```

This means:

- if `x > 0`, ReLU outputs `x`,
- if `x <= 0`, ReLU outputs `0`.

The derivative of ReLU is:

```text
ReLU'(x) = 1, if x > 0
ReLU'(x) = 0, if x <= 0
```

When the input is positive, the gradient passes through unchanged.

So instead of multiplying gradients by small values like `0.25`, ReLU allows the gradient to remain strong when the neuron is active.

This helps deep networks train more effectively.

---

## Why ReLU Helps in Deep NLP Models

ReLU helps reduce the vanishing gradient problem because active neurons pass gradients with value `1`.

This means earlier layers can still receive useful gradient signals.

For Bangla NER, this helps the model learn:

- word-level features,
- context patterns,
- entity boundaries,
- relationships between neighboring tokens.

Therefore, ReLU is often preferred over sigmoid in hidden layers of deep neural networks.

---

## Limitation of ReLU: Dying ReLU

Although ReLU helps with vanishing gradients, it introduces another problem called **dying ReLU**.

When the input to a ReLU neuron is negative, the output becomes:

```text
0
```

and the gradient also becomes:

```text
0
```

If a neuron keeps receiving negative inputs, it may stop updating completely.

Such a neuron is called a dead neuron.

A dead neuron:

- always outputs `0`,
- receives no useful gradient,
- does not update its weights,
- contributes nothing to the model.

This reduces the learning capacity of the network.

---

## Why Dying ReLU Happens

Dying ReLU can happen because of:

- a very large learning rate,
- poor weight initialization,
- many negative inputs,
- unstable training.

For example, if a neuron in a Bangla NER model becomes inactive, it may stop learning useful features for identifying entities like person names, locations, or organizations.

---

## Possible Solution: Leaky ReLU

One solution is to use Leaky ReLU.

Leaky ReLU allows a small gradient even when the input is negative.

```text
Leaky ReLU(x) = x, if x > 0
Leaky ReLU(x) = 0.01x, if x <= 0
```

This prevents neurons from completely dying because negative inputs still produce a small gradient.

---

## Comparison Table

| Activation Function | Gradient Range | Vanishing Gradient Problem | Dying Neuron Problem |
|---|---:|---|---|
| Sigmoid | 0 to 0.25 | Yes | No |
| ReLU | 0 or 1 | Mostly reduced | Yes |
| Leaky ReLU | small value or 1 | Mostly reduced | Mostly avoided |

---

## Final Answer

Using sigmoid in every hidden layer can cause the vanishing gradient problem because its derivative is always small, with a maximum value of only `0.25`.

During backpropagation, these small gradients are multiplied through many layers, making the gradient almost zero for earlier layers.

ReLU helps solve this problem because, for positive inputs, its derivative is `1`, allowing gradients to pass backward without shrinking.

However, ReLU can suffer from the dying ReLU problem. If a neuron receives negative inputs repeatedly, it outputs `0` and gets zero gradient, so it may stop learning permanently.

---

## My Understanding

I understood that sigmoid can make deep networks difficult to train because it shrinks gradients during backpropagation.

ReLU helps by allowing stronger gradient flow, but it also has a weakness: some neurons can die if they always receive negative inputs.

This helped me understand why activation function selection is important in deep learning models for NLP tasks like Bangla NER.
