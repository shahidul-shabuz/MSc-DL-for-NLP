# Q4: Error Functions — Descriptive

## Question Summary

You are building a Bangla text classifier that categorizes news articles into 5 categories:

- `rajniti` — politics
- `khela` — sports
- `projukti` — technology
- `manobinodon` — entertainment
- `orthoniti` — economics

The task is to explain:

1. why categorical cross-entropy loss is more appropriate than MSE loss,
2. what role the softmax function plays before the loss calculation.

---

## Solution

This is a multi-class classification problem.

The model receives a Bangla news article and must classify it into one of five possible categories:

```text
rajniti, khela, projukti, manobinodon, orthoniti
```

Since only one class should be correct for each article, the model needs to output a probability distribution over the five classes.

For this type of task, categorical cross-entropy is more suitable than mean squared error.

---

## Why This is a Multi-Class Classification Problem

In this task, each news article belongs to one category.

For example:

```text
Input: A Bangla article about a cricket match
Correct class: khela
```

The output should look like a probability distribution:

```text
rajniti        = 0.05
khela          = 0.85
projukti       = 0.03
manobinodon    = 0.04
orthoniti      = 0.03
```

Here, the highest probability is assigned to `khela`, so the model predicts the article as sports news.

---

## Role of Softmax

Before calculating categorical cross-entropy loss, the model usually produces raw scores called logits.

Example logits:

```text
rajniti        = 1.2
khela          = 4.0
projukti       = 0.8
manobinodon    = 1.0
orthoniti      = 0.5
```

These raw values are not probabilities.

The softmax function converts these logits into probabilities.

The softmax function is:

```text
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

After applying softmax:

- every output value becomes between `0` and `1`,
- all output probabilities sum to `1`,
- the class with the highest probability becomes the predicted class.

So, softmax converts the model output into a probability distribution over the five categories.

---

## Why Categorical Cross-Entropy is Appropriate

Categorical cross-entropy measures the difference between:

1. the true class distribution,
2. the predicted probability distribution.

For a correct class, the target is usually represented using one-hot encoding.

Example: if the correct class is `khela`, the target vector is:

```text
rajniti        = 0
khela          = 1
projukti       = 0
manobinodon    = 0
orthoniti      = 0
```

If the model predicts:

```text
rajniti        = 0.05
khela          = 0.85
projukti       = 0.03
manobinodon    = 0.04
orthoniti      = 0.03
```

then the model is confident about the correct class, so the loss will be small.

But if the model gives low probability to the correct class, the loss will be high.

---

## Categorical Cross-Entropy Formula

For multi-class classification, categorical cross-entropy is:

```text
Loss = -Σ y_i log(p_i)
```

where:

- `y_i` is the true label for class `i`,
- `p_i` is the predicted probability for class `i`.

Since only the correct class has value `1` in one-hot encoding, the loss mainly depends on the probability assigned to the correct class.

For example, if the correct class is `khela`:

```text
Loss = -log(p_khela)
```

If:

```text
p_khela = 0.85
```

then:

```text
Loss = -log(0.85)
```

This gives a small loss.

If:

```text
p_khela = 0.10
```

then:

```text
Loss = -log(0.10)
```

This gives a much larger loss.

---

## Why MSE is Not Ideal for This Task

MSE, or mean squared error, is usually better for regression tasks where the output is a continuous value.

Example regression task:

```text
Predict house price
Predict temperature
Predict rainfall amount
```

But Bangla news classification is not a regression problem. It is a classification problem.

Using MSE for classification has some problems:

- it does not directly measure probability confidence,
- it treats class labels like continuous numbers,
- it can produce slower learning,
- it does not penalize wrong confident predictions as effectively as cross-entropy,
- it is less suitable for softmax probability outputs.

For multi-class classification, we care about how much probability the model assigns to the correct class. Categorical cross-entropy directly measures this.

---

## Example Explanation

Suppose the correct class is:

```text
khela
```

If the model predicts:

```text
khela = 0.90
```

then the prediction is good, and categorical cross-entropy gives a low loss.

If the model predicts:

```text
khela = 0.05
```

then the prediction is bad, and categorical cross-entropy gives a high loss.

This behavior is useful because it strongly penalizes the model when it is confident but wrong.

---

## Final Answer

Categorical cross-entropy is appropriate for this Bangla news classification task because the task is a multi-class classification problem with five possible categories.

Softmax first converts the model's raw output scores into probabilities. Then categorical cross-entropy compares these predicted probabilities with the true one-hot encoded label.

MSE is not ideal because it is mainly designed for regression and does not handle class probability distributions as effectively as categorical cross-entropy.

---

## My Understanding

I understood that choosing the correct loss function depends on the task.

For Bangla news classification, the model must choose one class from several categories, so categorical cross-entropy is the best choice.

Softmax helps by converting raw model outputs into probabilities, and cross-entropy uses those probabilities to penalize wrong predictions.
