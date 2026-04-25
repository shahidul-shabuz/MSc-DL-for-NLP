# Q2: Gradient Descent — Exercise

## Question Summary

A single neuron has:

- weight, `w = 2.0`
- bias, `b = 1.0`
- input, `x = 3.0`
- true label, `y = 10.0`
- predicted output, `ŷ = w·x + b = 7.0`

The task is to:

1. compute the MSE loss,
2. compute `∂Loss/∂w` and `∂Loss/∂b`,
3. update `w` and `b` using a learning rate of `0.01`,
4. explain what happens if the learning rate is set to `5.0`.

---

## Given Information

```text
w = 2.0
b = 1.0
x = 3.0
y = 10.0
ŷ = w·x + b
ŷ = (2.0)(3.0) + 1.0
ŷ = 7.0
```

The prediction is `7.0`, but the true label is `10.0`.

So, the model prediction is lower than the target value.

---

## Part A: Compute the MSE Loss

For a single sample, the MSE loss is:

```text
Loss = (y - ŷ)²
```

Substituting the values:

```text
Loss = (10 - 7)²
Loss = 3²
Loss = 9
```

Therefore:

```text
MSE Loss = 9.0
```

---

## Part B: Compute the Gradients

The loss function is:

```text
Loss = (y - ŷ)²
```

Since:

```text
ŷ = wx + b
```

we can write:

```text
Loss = (y - wx - b)²
```

Now we compute the gradients with respect to `w` and `b`.

---

### Gradient with Respect to w

```text
∂Loss/∂w = -2x(y - ŷ)
```

Substituting the values:

```text
∂Loss/∂w = -2(3)(10 - 7)
∂Loss/∂w = -6(3)
∂Loss/∂w = -18
```

Therefore:

```text
∂Loss/∂w = -18
```

---

### Gradient with Respect to b

```text
∂Loss/∂b = -2(y - ŷ)
```

Substituting the values:

```text
∂Loss/∂b = -2(10 - 7)
∂Loss/∂b = -2(3)
∂Loss/∂b = -6
```

Therefore:

```text
∂Loss/∂b = -6
```

---

## Interpretation of the Gradients

Both gradients are negative:

```text
∂Loss/∂w = -18
∂Loss/∂b = -6
```

This means the loss will decrease if `w` and `b` increase.

Since the prediction `ŷ = 7.0` is lower than the true value `y = 10.0`, increasing `w` and `b` is the correct direction.

---

## Part C: Update w and b Using Learning Rate 0.01

The gradient descent update rule is:

```text
new parameter = old parameter - learning rate × gradient
```

The learning rate is:

```text
η = 0.01
```

---

### Update w

```text
w_new = w - η(∂Loss/∂w)
w_new = 2.0 - 0.01(-18)
w_new = 2.0 + 0.18
w_new = 2.18
```

---

### Update b

```text
b_new = b - η(∂Loss/∂b)
b_new = 1.0 - 0.01(-6)
b_new = 1.0 + 0.06
b_new = 1.06
```

Therefore:

```text
Updated w = 2.18
Updated b = 1.06
```

---

## Verification After Update

Now compute the new prediction:

```text
ŷ_new = w_new·x + b_new
ŷ_new = (2.18)(3.0) + 1.06
ŷ_new = 6.54 + 1.06
ŷ_new = 7.60
```

The new prediction is:

```text
ŷ_new = 7.60
```

This is closer to the true value `10.0` than the previous prediction `7.0`.

So, the update improved the model.

The new loss is:

```text
Loss = (10 - 7.60)²
Loss = 2.40²
Loss = 5.76
```

The loss decreased from `9.0` to `5.76`.

---

## Part D: What if the Learning Rate Were 5.0?

Now suppose the learning rate is:

```text
η = 5.0
```

Using the same gradient descent update rule:

---

### Update w

```text
w_new = 2.0 - 5.0(-18)
w_new = 2.0 + 90
w_new = 92.0
```

---

### Update b

```text
b_new = 1.0 - 5.0(-6)
b_new = 1.0 + 30
b_new = 31.0
```

Therefore:

```text
Updated w = 92.0
Updated b = 31.0
```

---

## New Prediction with Learning Rate 5.0

```text
ŷ_new = w_new·x + b_new
ŷ_new = (92.0)(3.0) + 31.0
ŷ_new = 276.0 + 31.0
ŷ_new = 307.0
```

The new prediction becomes:

```text
ŷ_new = 307.0
```

This is far away from the true value `10.0`.

---

## New Loss with Learning Rate 5.0

```text
Loss = (10 - 307)²
Loss = (-297)²
Loss = 88209
```

The loss becomes:

```text
Loss = 88,209
```

This is much larger than the original loss of `9.0`.

---

## Comparison Table

| Scenario | w | b | ŷ | Loss |
|---|---:|---:|---:|---:|
| Initial | 2.0 | 1.0 | 7.0 | 9.0 |
| After η = 0.01 | 2.18 | 1.06 | 7.60 | 5.76 |
| After η = 5.0 | 92.0 | 31.0 | 307.0 | 88,209 |

---

## Explanation

A learning rate of `5.0` is far too large.

Instead of taking a small step toward the minimum loss, the model takes a huge step and overshoots the optimal values.

As a result, the prediction becomes much worse and the loss increases dramatically.

This problem is called **divergence**.

In practice, if the learning rate is too large, gradient descent may never converge to the minimum loss.

---

## Final Answer

```text
MSE Loss = 9.0

∂Loss/∂w = -18
∂Loss/∂b = -6

With learning rate 0.01:
w_new = 2.18
b_new = 1.06
ŷ_new = 7.60
New loss = 5.76

With learning rate 5.0:
w_new = 92.0
b_new = 31.0
ŷ_new = 307.0
New loss = 88,209
```

---

## My Understanding

This exercise helped me understand how gradient descent updates model parameters using gradients.

A small learning rate, such as `0.01`, improves the model gradually and reduces the loss.

However, a very large learning rate, such as `5.0`, causes the model to overshoot the minimum and makes the loss explode.

This shows why choosing an appropriate learning rate is very important in deep learning.
