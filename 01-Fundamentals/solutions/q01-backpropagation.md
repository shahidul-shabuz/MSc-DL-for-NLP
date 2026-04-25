# Q1: Backpropagation — Descriptive

## Question Summary

In a simple feedforward network used for sentiment classification of Bangla movie reviews, the network has:

- an input layer,
- one hidden layer with 4 neurons,
- one output layer with 1 neuron for positive/negative classification.

The task is to explain how backpropagation computes gradients starting from the output error and propagating back to the first layer, and why the chain rule is essential.

---

## Solution

Imagine the neural network as a team trying to predict whether a Bangla movie review is positive or negative.

- The **input layer** receives the raw words or numerical representation of the review.
- The **hidden layer** has 4 neurons that work like feature detectors or analysts.
- The **output layer** has 1 neuron that gives the final prediction, such as positive or negative sentiment.

Backpropagation is the process of moving backward from the final error to calculate how much each weight contributed to the mistake.

---

## Step 1: Forward Pass

First, the network performs a forward pass.

The input review is passed through the input layer, then through the hidden layer with 4 neurons, and finally to the output neuron.

The output neuron produces a prediction, for example:

```text
ŷ = 0.8
```

If the true label is:

```text
y = 0
```

then the prediction is wrong, and the loss function measures the error.

---

## Step 2: Compute the Output Error

The loss function compares the predicted output with the true label.

If the predicted value is far from the true value, the loss becomes large.

This error tells the model how incorrect its prediction was.

---

## Step 3: Compute Gradients at the Output Layer

Backpropagation starts from the output layer.

It calculates how much the output neuron contributed to the final error.

Mathematically, this means computing the derivative of the loss with respect to the output prediction.

This gradient tells the model how the output layer weights should be changed to reduce the loss.

---

## Step 4: Propagate the Error Back to the Hidden Layer

The output neuron depends on the 4 hidden neurons.

So, backpropagation calculates how much each hidden neuron contributed to the wrong prediction.

If one hidden neuron strongly influenced the incorrect output, it receives a larger gradient.

This allows the network to identify which hidden-layer activations were responsible for the error.

---

## Step 5: Compute Gradients for Input-to-Hidden Weights

After calculating the hidden-layer gradients, the error is propagated further backward to the weights between the input layer and the hidden layer.

Each input-to-hidden weight receives a gradient that tells how much it contributed to the final loss.

---

## Step 6: Update the Weights

Once gradients are computed for all weights, gradient descent updates the weights.

The update is done in the opposite direction of the gradient:

```text
new weight = old weight - learning rate × gradient
```

This helps the network reduce its error in future predictions.

---

## Why the Chain Rule is Essential

A weight in the first layer does not directly affect the final loss.

Its influence passes through a chain:

```text
input weight → hidden neuron → output neuron → loss
```

The chain rule allows us to calculate how a small change in an early weight affects the final loss.

It does this by multiplying derivatives through each step of the network.

Without the chain rule, the model would not know how to assign responsibility to earlier layers.

Therefore, the chain rule is essential for training deep neural networks.

---

## My Understanding

I understood backpropagation as a systematic way of assigning responsibility for prediction errors.

It starts from the final loss and works backward through the network. The chain rule makes it possible to calculate the effect of earlier weights on the final output error.

This is why backpropagation is one of the most important concepts in deep learning.
