# Machine Learning from an Information-Theoretic Perspective

A comprehensive tutorial series exploring ML fundamentals through the lens of **entropy**, **information theory**, and **compression**.

## 🎯 Core Philosophy

> **Machine Learning is fundamentally about finding efficient representations that minimize surprise.**

Everything in ML can be understood through information theory:
- **Training** = Compressing data into model parameters
- **Inference** = Using compressed knowledge to reduce uncertainty
- **Generalization** = Learning true data distribution, not noise
- **Latent spaces** = Optimal encoding of underlying factors

## 📚 Tutorial Curriculum

### [00. Probability Foundations](tutorials/00_probability_foundations/)
**Start here if distributions feel fuzzy!**
- What IS a probability distribution vs a probability?
- Random variables: discrete vs continuous
- PMF vs PDF: why the difference matters
- The discrete → continuous transition
- Joint, marginal, conditional distributions

### [01. Entropy Fundamentals & Huffman Encoding](tutorials/01_entropy_fundamentals/)
- **Deriving** Shannon entropy from first principles
- Entropy for discrete distributions (with examples)
- Differential entropy for continuous distributions
- The discrete → continuous transition explained
- Huffman coding: achieving entropy bounds
- Cross-entropy: comparing distributions
- ML as entropy minimization

### [02. KL Divergence](tutorials/02_kl_divergence/)
- **Deriving** KL divergence step-by-step
- Proving KL ≥ 0 using Jensen's inequality
- KL divergence for discrete distributions (detailed examples)
- KL divergence for continuous distributions
- Asymmetry: forward vs reverse KL in ML
- Maximum likelihood as KL minimization

### [03. Normal & Multivariate Normal Distributions](tutorials/03_normal_distributions/)
- Why Gaussians are everywhere (max entropy!)
- Univariate normal distribution
- Multivariate normal: covariance geometry
- Mahalanobis distance
- KL divergence between Gaussians (closed form)

### [04. VAE, ELBO & Variational Inference](tutorials/04_vae_variational_inference/)
- The intractable posterior problem
- **Deriving** the Evidence Lower Bound (ELBO)
- Variational inference framework
- VAE architecture and the reparameterization trick
- Why KL to prior in latent space?
- Information bottleneck perspective

### [05. Why Logarithm is Fundamental in ML](tutorials/05_logarithm_in_ml/)
**The deep "why" behind log-likelihood**
- Is log just a numerical convenience? **NO.**
- Uniqueness theorem: log is the ONLY function converting × to +
- Gradient scaling problem: without log, gradients vanish
- Shannon's proof: information MUST be logarithmic
- Why cross-entropy has that form
- Softmax + log: the beautiful gradient cancellation
- Thought experiment: ML without logarithms (even with perfect precision)

### [06. Probability Concepts in ML](tutorials/06_probability_ml_concepts/)
**The Bayesian foundation of machine learning**
- Joint, marginal, conditional probability (P(x), P(y|x), etc.)
- **Deriving** Bayes' theorem (multiple approaches)
- Law of total probability with examples
- Probability vs Likelihood — the crucial distinction
- Prior, Posterior, Evidence — Bayesian vocabulary
- Why losses ARE negative log-likelihoods
- Regularization as priors

### [07. Combinatorics](tutorials/07_combinatorics/)
**Counting: the foundation of probability**
- The multiplication principle
- Permutations: ordered arrangements
- **Deriving** the combination formula
- Permutations with repetition (multinomial)
- Pascal's triangle and binomial coefficients
- Connection to probability distributions

### [08. Backpropagation & The Entropy Connection](tutorials/08_backpropagation/)
**How neural networks learn — and why it connects to entropy**
- **Deriving** backpropagation from the chain rule
- Computational graphs and automatic differentiation
- Forward vs reverse mode AD (why backprop wins)
- Softmax + Cross-entropy gradient derivation
- **The entropy-gradient connection**: high uncertainty = high gradient flow
- Why vanishing gradients happen (low entropy activations!)
- Backprop as "information relevance" computation
- **The unifying insight**: Training = minimizing prediction surprise

### [09. Regularization](tutorials/09_regularization/)
**Preventing overfitting through the lens of priors**
- L1 regularization: Laplace prior → sparsity
- L2 regularization: Gaussian prior → weight decay
- **Deriving** regularization from MAP estimation
- Dropout: Ensemble interpretation and uncertainty
- Elastic Net: combining L1 and L2
- Information bottleneck perspective

### [10. Batch Normalization](tutorials/10_batch_normalization/)
**Stabilizing training by controlling distributions**
- **Deriving** the forward pass formula
- **Deriving** the complete backward pass (the hard part!)
- Running mean and variance for inference
- Why BatchNorm allows higher learning rates
- Entropy connection: keeps activations in high-gradient regime
- Layer Normalization comparison

### [11. Learning Rate Concepts](tutorials/11_learning_rate/)
**The most important hyperparameter**
- **Deriving** optimal learning rate for quadratic functions
- Convergence conditions: $\eta < 2/\lambda_{max}$
- Momentum: exponential moving average of gradients
- RMSprop: per-parameter adaptive rates
- **Deriving** Adam with bias correction
- Learning rate schedules: step decay, cosine annealing, warmup
- Large learning rates as implicit regularization

### [12. Convolutional Neural Networks](tutorials/12_convolutional_networks/)
**Spatial structure and weight sharing**
- **Deriving** the output size formula
- **Deriving** backpropagation through convolution
- Receptive field computation
- Why multiple small filters beat one large filter
- Translation equivariance proof
- Pooling: max vs average
- Information-theoretic view: detecting local patterns

### [13. Recurrent Neural Networks](tutorials/13_recurrent_networks/)
**Sequence modeling and the vanishing gradient problem**
- **Deriving** Backpropagation Through Time (BPTT)
- **Proving** why gradients vanish (eigenvalue analysis)
- LSTM: the cell state highway for gradient flow
- GRU: simplified gating mechanism
- Gradient clipping for exploding gradients
- Comparing RNN vs LSTM vs Transformer

## 🔬 Physics Tutorials

See [physics_tutorials/](physics_tutorials/) for thermodynamics foundations:

### [Physics 01. Entropy from Physics](physics_tutorials/01_entropy_physics/)
**Understand entropy like a physicist**
- Microstates vs macrostates
- Why entropy increases (it's just probability!)
- **Deriving** S = k_B ln W from first principles
- Connection to Shannon entropy
- Gibbs entropy formula

### [Physics 02. The Carnot Cycle](physics_tutorials/02_carnot_cycle/)
**Why heat engines have a maximum efficiency**
- The four steps of the Carnot cycle
- **Deriving** η = 1 - T_C/T_H
- **Proof** that Carnot cannot be exceeded (Second Law!)
- Why 100% efficiency is impossible

## 🔧 Setup

```bash
pip install numpy matplotlib scipy torch jupyter
```

## 📖 How to Use

1. Start with the **theory markdown** (`.md`) files for deep intuition
2. Work through the **Jupyter notebooks** (`.ipynb`) for hands-on code
3. Experiment with the visualizations to build intuition

## 🧠 Key Insights Preview

| Concept | Information-Theoretic View |
|---------|---------------------------|
| MSE Loss | Assumes Gaussian noise → maximizes log-likelihood |
| Cross-Entropy Loss | Directly minimizes KL divergence to true labels |
| Softmax | Maximum entropy distribution given constraints |
| Dropout | Prevents model from storing too much information |
| VAE Latent KL | Information bottleneck / rate-distortion tradeoff |
| Batch Normalization | Reduces internal covariate shift (information flow) |
| Backpropagation | Computes "information relevance" of each neuron |
| Vanishing Gradients | Low entropy (saturated) neurons block information flow |
| Regularization | Constrains model complexity = compression |
| Convolution | Detects local patterns with shared detectors |
| RNN/LSTM | Compresses sequence history into fixed-size state |

---

*"The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point."* — Claude Shannon, 1948
