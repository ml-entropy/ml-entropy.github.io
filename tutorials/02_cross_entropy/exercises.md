# Tutorial 15: Cross-Entropy — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Concepts

### Exercise A1 🟢 — Definition in Your Own Words
Explain what Cross-Entropy measures from an information theory perspective. Why is it a useful metric for training a machine learning model?

### Exercise A2 🟢 — The Loss Formula
In a multi-class classification setting, the true label is represented by a one-hot vector $y$ and the model's predictions by a probability vector $\hat{y}$. Show that the general Cross-Entropy formula $H(y, \hat{y}) = -\sum_i y_i \log \hat{y}_i$ simplifies to just the negative logarithm of the predicted probability for the correct class.

### Exercise A3 🟡 — Manual Calculation
A model is trained to classify images into three categories: {Apple, Banana, Cherry}. For an image of a Banana, the model outputs the following probabilities:
-   $Q(\text{Apple}) = 0.2$
-   $Q(\text{Banana}) = 0.5$
-   $Q(\text{Cherry}) = 0.3$

a) Write down the true probability distribution $P$.
b) Calculate the Cross-Entropy loss for this single prediction.
c) What would the loss be if the model was "perfectly confident" and correct, i.e., $Q(\text{Banana}) = 1$?
d) What would the loss be if the model was "perfectly confident" and incorrect, e.g., $Q(\text{Apple}) = 1$?

### Exercise A4 🔴 — Cross-Entropy vs. MSE
Consider a binary classification problem where the true label is 1. A model with a sigmoid activation outputs a prediction $p$.
a) Write down the formula for Binary Cross-Entropy (BCE) loss.
b) Write down the formula for Mean Squared Error (MSE) loss.
c) Calculate the gradient of both the BCE and MSE loss with respect to the model's logit (the input to the sigmoid).
d) Explain why the BCE gradient is generally better for training than the MSE gradient, especially when the model's prediction is confidently wrong (e.g., $p$ is close to 0).

### Exercise A5 🟡 — Focal Loss Calculation
Focal Loss is defined as $FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$, where $p_t$ is the probability of the true class.
a) Calculate the Cross-Entropy loss ($\gamma=0$) for an "easy" example where $p_t = 0.9$.
b) Calculate the Cross-Entropy loss for a "hard" example where $p_t = 0.1$.
c) Calculate the Focal Loss with $\gamma=2$ for the same "easy" and "hard" examples.
d) By what factor did the loss decrease for the easy example compared to the hard example when switching from CE to FL?

---

## Part B: Coding

### Exercise B1 🟢 — Cross-Entropy Function
Write a Python function `cross_entropy(P, Q)` that takes two NumPy arrays representing the true and predicted distributions and returns the cross-entropy.
-   Verify your function using the values from Exercise A3.
-   What happens if a value in `Q` is exactly 0? How can you make your function numerically stable?

### Exercise B2 🟡 — Visualizing the Loss
Create a plot to visualize the Cross-Entropy loss for a binary classification problem.
-   The true label is 1.
-   Plot the loss as the model's predicted probability for class 1 ranges from 0.01 to 0.99.
-   On the same graph, plot the MSE loss.
-   Label your axes and curves clearly. What does this visualization tell you about the behavior of the two loss functions?

### Exercise B3 🔴 — Numerical Stability (LogSumExp)
Implement a function `stable_softmax_cross_entropy(logits, target_index)` that:
1.  Takes raw logits (not probabilities) as input.
2.  Computes the Cross-Entropy loss using the **LogSumExp** trick to avoid overflow/underflow.
    -   Formula: $\log(\sum e^{x_i}) = c + \log(\sum e^{x_i - c})$, where $c = \max(x)$.
    -   Loss: $-x_{target} + \log(\sum e^{x_j})$.
3.  Test it with `logits = [1000, 1001, 1002]` and `target_index = 1`. Verify that a naive implementation would fail (overflow).

---

## Part C: Conceptual Deep Dive

### Exercise C1 🟡 — Label Smoothing
Label smoothing is a regularization technique where the one-hot encoded labels are "softened." For example, for a 3-class problem, a true label of `[0, 1, 0]` might be changed to `[0.05, 0.9, 0.05]`.
a) How does this change the calculation of the Cross-Entropy loss?
b) What is the goal of label smoothing? How does it prevent a model from becoming "overconfident"?
c) What is the relationship between label smoothing and the KL Divergence between the model's predictions and the smoothed labels?

### Exercise C2 🔴 — The Role of the Logarithm
We've established that Cross-Entropy is $-\sum P(x) \log Q(x)$. Imagine an alternative universe where we defined a "Linear-Entropy" as $-\sum P(x) Q(x)$.
-   Would this be a valid loss function?
-   What desirable properties of the logarithmic loss would be lost? Think about the penalty for wrong predictions and the resulting gradients.

