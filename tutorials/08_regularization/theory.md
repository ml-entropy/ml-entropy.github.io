# Tutorial 09: Regularization — Preventing Overfitting

## Overview
Regularization techniques prevent neural networks from memorizing training data. We derive the formulas, understand them from multiple perspectives (Bayesian, information-theoretic, geometric), and see why they work.

---

## Part 1: The Overfitting Problem

### What is Overfitting?
A model **overfits** when it:
- Fits training data perfectly
- Performs poorly on new data
- Has learned noise, not signal

### The Bias-Variance Tradeoff

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Regularization reduces variance** at the cost of slightly increased bias

### Information-Theoretic View

Overfitting = model stores too much information about training set:
- Training data has true patterns + random noise
- Memorizing noise = wasted capacity
- **Regularization limits information storage**

---

## Part 2: L2 Regularization (Ridge / Weight Decay)

### The Formula

Standard loss:
$$\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i; \theta))$$

L2-regularized loss:
$$\mathcal{L}_{L2}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = \mathcal{L}(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2$$

### Derivation of the Gradient

$$\frac{\partial \mathcal{L}_{L2}}{\partial \theta_j} = \frac{\partial \mathcal{L}}{\partial \theta_j} + \lambda \theta_j$$

Update rule:
$$\theta_j \leftarrow \theta_j - \eta\left(\frac{\partial \mathcal{L}}{\partial \theta_j} + \lambda \theta_j\right) = (1 - \eta\lambda)\theta_j - \eta\frac{\partial \mathcal{L}}{\partial \theta_j}$$

**Key insight**: Each update **shrinks** weights by factor $(1 - \eta\lambda)$. This is why it's called **weight decay**!

### Bayesian Interpretation

L2 regularization = **Gaussian prior** on weights.

**Derivation**:
Prior: $P(\theta) = \mathcal{N}(0, \sigma_p^2 I) \propto \exp\left(-\frac{\|\theta\|^2}{2\sigma_p^2}\right)$

Likelihood: $P(D|\theta) \propto \exp(-\mathcal{L}(\theta))$

Posterior (Bayes):
$$P(\theta|D) \propto P(D|\theta)P(\theta) \propto \exp\left(-\mathcal{L}(\theta) - \frac{\|\theta\|^2}{2\sigma_p^2}\right)$$

MAP estimation (maximize log-posterior):
$$\theta_{MAP} = \arg\max_\theta \left[-\mathcal{L}(\theta) - \frac{\|\theta\|^2}{2\sigma_p^2}\right] = \arg\min_\theta \left[\mathcal{L}(\theta) + \frac{1}{2\sigma_p^2}\|\theta\|^2\right]$$

**Conclusion**: $\lambda = 1/\sigma_p^2$. Stronger regularization = smaller prior variance = stronger belief that weights should be near zero.

### Geometric Interpretation

L2 regularization constrains weights to lie within a sphere:
$$\|\theta\|_2^2 \leq C$$

The solution is where the loss contours meet this sphere — typically **not at the minimum** of the original loss.

---

## Part 3: L1 Regularization (Lasso)

### The Formula

$$\mathcal{L}_{L1}(\theta) = \mathcal{L}(\theta) + \lambda\|\theta\|_1 = \mathcal{L}(\theta) + \lambda\sum_j |\theta_j|$$

### The Gradient (Subgradient)

$$\frac{\partial \mathcal{L}_{L1}}{\partial \theta_j} = \frac{\partial \mathcal{L}}{\partial \theta_j} + \lambda \cdot \text{sign}(\theta_j)$$

where $\text{sign}(\theta) = \begin{cases} +1 & \theta > 0 \\ -1 & \theta < 0 \\ 0 & \theta = 0 \end{cases}$

### Why L1 Produces Sparsity

**Key difference from L2**:
- L2: Gradient penalty $\propto \theta$ → small weights get small push
- L1: Gradient penalty = constant $\lambda$ → small weights get **same push as large weights**

Result: L1 drives weights **exactly to zero** (sparsity), while L2 just makes them small.

### Bayesian Interpretation

L1 regularization = **Laplace prior** on weights:
$$P(\theta) \propto \exp(-\lambda|\theta|)$$

This prior has more mass at exactly zero than Gaussian.

### Geometric Interpretation

L1 constrains weights to a **diamond** (L1 ball):
$$\|\theta\|_1 \leq C$$

Loss contours typically hit the diamond at **corners** (where some $\theta_j = 0$).

---

## Part 4: Elastic Net (L1 + L2)

Combines both:
$$\mathcal{L}_{elastic}(\theta) = \mathcal{L}(\theta) + \lambda_1\|\theta\|_1 + \frac{\lambda_2}{2}\|\theta\|_2^2$$

**Benefits**:
- Sparsity from L1
- Stability from L2 (handles correlated features better)

---

## Part 5: Dropout

### The Mechanism

During training, randomly set each neuron's output to 0 with probability $p$:
$$\tilde{a}_j = \begin{cases} 0 & \text{with probability } p \\ a_j / (1-p) & \text{with probability } 1-p \end{cases}$$

The scaling by $1/(1-p)$ ensures expected value is unchanged.

### Why It Works

**Multiple interpretations**:

1. **Ensemble averaging**: Dropout trains exponentially many sub-networks. At test time, we use the ensemble average.

2. **Noise injection**: Forces network to learn robust features that work even with missing information.

3. **Information bottleneck**: Limits information flow, preventing memorization.

4. **Co-adaptation prevention**: Neurons can't rely on specific other neurons being present.

### Mathematical Derivation

Let $m \sim \text{Bernoulli}(1-p)^n$ be the dropout mask.

Training forward pass:
$$\tilde{h} = \frac{1}{1-p} \cdot m \odot h$$

Expected value:
$$\mathbb{E}[\tilde{h}] = \frac{1}{1-p} \cdot (1-p) \cdot h = h$$

This is why we can remove dropout at test time without changing expected behavior.

### Dropout as Bayesian Approximation

Dropout approximates **Bayesian inference** over network weights (Gal & Ghahramani, 2016):
- Each dropout mask = one sample from weight posterior
- Averaging at test time ≈ Bayesian model averaging

---

## Part 6: Early Stopping

### The Concept

Stop training when validation loss stops improving:

```
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break  # Early stop!
```

### Why It's Regularization

**Implicit L2 regularization** (Bishop, 1995):
- Weights start near zero (initialization)
- Training moves them away from zero
- Stopping early = keeping weights closer to zero
- Equivalent to L2 penalty (under certain conditions)

---

## Part 7: Data Augmentation

### The Concept

Artificially increase dataset size by applying transformations:
- Images: rotation, flip, crop, color jitter
- Text: synonym replacement, back-translation
- Audio: pitch shift, time stretch

### Why It's Regularization

**Encodes invariances** into the model:
- "Flipped cat is still a cat"
- Model can't memorize specific pixel patterns
- Forces learning of generalizable features

### Information-Theoretic View

$$H(\text{augmented data}) > H(\text{original data})$$

More entropy = harder to memorize = better generalization.

---

## Part 8: The Information-Theoretic Perspective

### Minimum Description Length (MDL)

Best model = shortest description of data:
$$\text{Total Length} = \text{Model Description} + \text{Data given Model}$$

- Complex model = long description but short residuals
- Simple model = short description but long residuals
- **Regularization = penalizing model description length**

### Rate-Distortion Theory

Regularization controls the **information bottleneck**:
- More regularization = less information stored about training data
- Less information = better generalization (if we keep the right information)

### Entropy Connection

Overfitting: Model has **low entropy** predictions on training data (certain) but **high entropy** on test data (uncertain).

Regularization: Keeps model in **higher entropy** state — less certain, but more calibrated.

---

## Summary

| Technique | Formula | Effect | Bayesian View |
|-----------|---------|--------|---------------|
| L2 (Ridge) | $\lambda\|\theta\|_2^2$ | Shrinks weights | Gaussian prior |
| L1 (Lasso) | $\lambda\|\theta\|_1$ | Sparsity | Laplace prior |
| Dropout | Random zeroing | Ensemble | Bayesian approx |
| Early Stopping | Stop when val↑ | Implicit L2 | — |
| Data Augmentation | Transform inputs | Encode invariances | — |

**The key insight**: All regularization techniques **limit the information** a model can store about the training data, forcing it to learn generalizable patterns.
