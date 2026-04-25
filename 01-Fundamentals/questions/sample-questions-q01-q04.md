
# Sample Questions: Fundamentals

This file contains the Professor-provided sample questions for the **Fundamentals** module of the **Deep Learning for NLP** course.

---

## Q1. Backpropagation — Descriptive

In a simple feedforward network used for sentiment classification of Bangla movie reviews, the network has an input layer, one hidden layer with 4 neurons, and an output layer with 1 neuron for positive/negative classification.

Explain step-by-step how backpropagation computes the gradients starting from the output error and propagating back to the first layer.

Why is the chain rule essential in this process?

---

## Q2. Gradient Descent — Exercise

A single neuron has weight `w = 2.0`, bias `b = 1.0`, and uses MSE loss.

For input `x = 3.0` and true label `y = 10.0`, the predicted output is:

```text
ŷ = w·x + b = 7.0

Answer the following:

(a) Compute the MSE loss.
(b) Compute ∂Loss/∂w and ∂Loss/∂b.
(c) Update w and b using a learning rate of 0.01.
(d) What would happen if the learning rate were set to 5.0 instead?

---

## Q3. Activation Functions — Analytic

Consider a deep network for Bangla named entity recognition, or NER.

Explain why using the sigmoid activation in every hidden layer would cause the vanishing gradient problem during backpropagation.

How does the ReLU activation function address this, and what is one limitation ReLU introduces?

Hint: dying ReLU.

---

## Q4. Error Functions — Descriptive

You are building a Bangla text classifier that categorizes news articles into 5 categories:

rajniti — politics
khela — sports
projukti — technology
manobinodon — entertainment
orthoniti — economics

Explain why categorical cross-entropy loss is the appropriate choice over MSE loss for this task.

What role does the softmax function play before the loss calculation?
