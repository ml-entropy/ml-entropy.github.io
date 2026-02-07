# Tutorial 14: Entropy, Cross-Entropy, and KL Divergence — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Concepts

### Exercise A1 🟢 — Definitions
In your own words, define the following and explain their relationship:
a) Entropy $H(P)$
b) Cross-Entropy $H(P, Q)$
c) KL Divergence $D_{KL}(P || Q)$

### Exercise A2 🟢 — The Grand Equation
Write down the equation that connects the three concepts and explain why, in machine learning, minimizing cross-entropy is equivalent to minimizing KL divergence.

### Exercise A3 🟡 — Weather Prediction
Let the true probability of weather be $P = \{\text{sun: } 0.5, \text{rain: } 0.5\}$.
Your weather model predicts $Q = \{\text{sun: } 0.8, \text{rain: } 0.2\}$.
Calculate (using log base 2):
a) The entropy of the true distribution, $H(P)$.
b) The cross-entropy between your model and the true distribution, $H(P, Q)$.
c) The KL divergence, $D_{KL}(P || Q)$.
d) Verify that $H(P, Q) = H(P) + D_{KL}(P || Q)$.

### Exercise A4 🟡 — Perfect vs. Terrible Model
Consider a 3-class classification problem where the true label is class A.
-   True distribution $P = \{A: 1, B: 0, C: 0\}$.
a) What is the entropy $H(P)$ of this true distribution?
b) A "perfect" model predicts $Q_{perfect} = \{A: 0.99, B: 0.005, C: 0.005\}$. Calculate $H(P, Q_{perfect})$.
c) A "terrible" model predicts $Q_{terrible} = \{A: 0.1, B: 0.5, C: 0.4\}$. Calculate $H(P, Q_{terrible})$.
d) Compare the results. What does this tell you about cross-entropy as a loss function?

### Exercise A5 🔴 — Asymmetry of KL Divergence
Using the weather example from A3 ($P = \{0.5, 0.5\}$ and $Q = \{0.8, 0.2\}$), calculate $D_{KL}(Q || P)$.
Is it the same as $D_{KL}(P || Q)$? Why is this property important?

---

## Part B: Coding

### Exercise B1 🟢 — Basic Calculator
Write Python functions to calculate `entropy(P)`, `cross_entropy(P, Q)`, and `kl_divergence(P, Q)`.
-   The inputs should be NumPy arrays.
-   Use `np.log2`.
-   Handle the case where a probability in Q is 0 (add a small epsilon).

### Exercise B2 🟡 — Visualizing the Loss
Create a plot that shows how the cross-entropy loss changes as a model's prediction for the correct class moves from 0 to 1.
-   Assume a 2-class problem where the true label is `[1, 0]`.
-   The model's prediction is `[p, 1-p]`.
-   Plot the cross-entropy for `p` ranging from 0.01 to 0.99.
-   What does the shape of the curve tell you about the penalty for confident wrong predictions?

---

## Part C: Conceptual Deep Dive

### Exercise C1 🟡 — Why Not MSE?
For classification problems, why is cross-entropy a better loss function than Mean Squared Error (MSE)?
*Hint: Think about the loss surface and the magnitude of gradients for predictions that are far from the true label.*

### Exercise C2 🔴 — VAEs and KL Divergence
In a Variational Autoencoder (VAE), the loss function has two parts: a reconstruction loss and a KL divergence term. This KL term often measures the divergence between the learned latent distribution $q(z|x)$ and a standard normal prior $p(z) = \mathcal{N}(0, 1)$.
Explain the role of this KL divergence term. What is it "enforcing" on the latent space? What would happen if this term were removed?

