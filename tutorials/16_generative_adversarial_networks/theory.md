# Tutorial 16: Generative Adversarial Networks (GANs)

## 🎯 Why This Tutorial?

**Prerequisites:** Probability distributions (Tutorial 00), KL divergence (Tutorial 03), entropy (Tutorial 01), neural network basics (Tutorial 07), VAEs (Tutorial 13)

Generative Adversarial Networks represent one of the most creative ideas in machine learning: instead of directly modeling a probability distribution, we set up a **game between two neural networks** where competition drives both to improve. This adversarial framework produces stunningly realistic samples — from photorealistic faces to artistic masterpieces — without ever writing down an explicit probability formula.

This tutorial connects to everything we've built so far. GANs minimize a **divergence between distributions** (linking to KL divergence and information theory), they rely on **backpropagation** through competing networks, and understanding their training dynamics requires our knowledge of **optimization landscapes**.

After this tutorial, you'll understand:
- Why adversarial training works at a deep mathematical level
- What GANs actually optimize (and why it's related to Jensen-Shannon divergence)
- Why GANs are notoriously hard to train
- How Wasserstein GANs fix fundamental training problems
- When to choose GANs over other generative models

---

## 1. The Generative Modeling Problem

The fundamental problem: given samples from an unknown distribution $p_{data}(x)$, learn to **generate new samples** that look like they came from the same distribution.

There are three main approaches:

| Approach | Method | Density | Examples |
|----------|--------|---------|----------|
| **Explicit density, tractable** | Directly model $p_\theta(x)$ | Exact | Normalizing Flows, Autoregressive |
| **Explicit density, approximate** | Approximate $p_\theta(x)$ | Lower bound | VAEs |
| **Implicit density** | Learn to sample, no explicit density | None | GANs |

GANs take the **implicit density** approach: we never write down $p_g(x)$, but we learn a function that transforms noise into realistic samples.

---

## 2. The Adversarial Framework

### 2.1 The Counterfeiter and the Detective

The GAN framework involves two players:

**Generator $G$:** Takes random noise $z \sim p_z(z)$ (typically Gaussian) and maps it to data space: $x_{fake} = G(z)$. The generator is like a counterfeiter trying to produce fake currency.

**Discriminator $D$:** Takes a data point $x$ and outputs the probability that it came from the real data rather than the generator: $D(x) \in [0, 1]$. The discriminator is like a detective trying to catch counterfeits.

The training proceeds as an adversarial game:
1. The **discriminator** tries to correctly classify real vs. fake
2. The **generator** tries to fool the discriminator
3. Over time, both improve — the generator makes better fakes, the discriminator becomes more discerning
4. At equilibrium (ideally), the generator produces perfect samples and the discriminator can't tell the difference

### 2.2 The Formal Objective

The minimax game is:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Let's parse each term:

- $\mathbb{E}_{x \sim p_{data}}[\log D(x)]$: For **real** data, D wants $D(x) \to 1$, making $\log D(x) \to 0$ (maximized)
- $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$: For **fake** data, D wants $D(G(z)) \to 0$, making $\log(1-0) = 0$ (maximized). But G wants $D(G(z)) \to 1$, making $\log(1-1) \to -\infty$ (minimized)

The discriminator **maximizes** this expression (pushing both terms toward 0), while the generator **minimizes** it (pushing the second term toward $-\infty$).

---

## 3. Theoretical Analysis

### 3.1 The Optimal Discriminator

**Proposition:** For a fixed generator $G$, the optimal discriminator is:

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**Proof:**

The training objective for $D$ (for a fixed $G$) is:

$$V(D, G) = \int_x p_{data}(x) \log D(x) \, dx + \int_x p_g(x) \log(1 - D(x)) \, dx$$

$$= \int_x \left[ p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

For each $x$, we maximize $f(D) = a \log D + b \log(1 - D)$ where $a = p_{data}(x)$ and $b = p_g(x)$.

Taking the derivative and setting to zero:

$$\frac{a}{D} - \frac{b}{1-D} = 0 \implies D^* = \frac{a}{a+b} = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

The second derivative is $-a/D^2 - b/(1-D)^2 < 0$, confirming this is a maximum. $\blacksquare$

**Intuition:** The optimal discriminator outputs the probability that a sample came from the real data, given both the real and generated distributions. When $p_g = p_{data}$, we get $D^*(x) = 1/2$ — the detective is completely stumped.

### 3.2 What Does the Generator Minimize?

Substituting $D^*$ back into the value function:

$$V(D^*, G) = \mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{data}(x) + p_g(x)}\right]$$

Let $m(x) = \frac{p_{data}(x) + p_g(x)}{2}$. Then:

$$V(D^*, G) = \mathbb{E}_{p_{data}}\left[\log \frac{p_{data}}{2m}\right] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{2m}\right]$$

$$= D_{KL}(p_{data} \| m) + D_{KL}(p_g \| m) - 2\log 2$$

$$= 2 \cdot JSD(p_{data} \| p_g) - \log 4$$

where $JSD$ is the **Jensen-Shannon Divergence**.

### 3.3 Jensen-Shannon Divergence

The JSD is a symmetrized, smoothed version of the KL divergence:

$$JSD(P \| Q) = \frac{1}{2} D_{KL}\left(P \| \frac{P+Q}{2}\right) + \frac{1}{2} D_{KL}\left(Q \| \frac{P+Q}{2}\right)$$

Properties:
- $0 \leq JSD(P \| Q) \leq \log 2$ (bounded, unlike KL)
- $JSD(P \| Q) = JSD(Q \| P)$ (symmetric, unlike KL)
- $JSD(P \| Q) = 0 \iff P = Q$

### 3.4 Global Optimum

**Theorem:** The global minimum of $V(D^*, G)$ is achieved if and only if $p_g = p_{data}$, at which point $V(D^*, G) = -\log 4$.

**Proof:** Since $JSD(p_{data} \| p_g) \geq 0$ with equality iff $p_{data} = p_g$:

$$V(D^*, G) = 2 \cdot JSD(p_{data} \| p_g) - \log 4 \geq -\log 4$$

Equality holds iff $p_g = p_{data}$, at which point $D^*(x) = 1/2$ for all $x$. $\blacksquare$

---

## 4. Training Algorithm

### Algorithm 1: GAN Training (from Goodfellow et al., 2014)

```
For each training iteration:
    # --- Train Discriminator ---
    for k steps:
        Sample minibatch {x₁,...,xₘ} from data
        Sample minibatch {z₁,...,zₘ} from p_z
        Update D by ascending its stochastic gradient:
            ∇_θd (1/m) Σᵢ [log D(xᵢ) + log(1 - D(G(zᵢ)))]

    # --- Train Generator ---
    Sample minibatch {z₁,...,zₘ} from p_z
    Update G by descending its stochastic gradient:
        ∇_θg (1/m) Σᵢ log(1 - D(G(zᵢ)))
```

**Key detail:** The discriminator is typically trained for $k$ steps per generator step, because the generator's gradient is only meaningful when the discriminator is close to optimal.

### The Non-Saturating Generator Loss

In practice, $\log(1 - D(G(z)))$ saturates when $D(G(z)) \approx 0$ (early in training when the generator is terrible). The gradient vanishes!

**Practical fix:** Instead of minimizing $\log(1 - D(G(z)))$, the generator maximizes $\log D(G(z))$.

This doesn't change the fixed point of the dynamics but provides stronger gradients early in training.

---

## 5. Training Challenges

### 5.1 Mode Collapse

**The problem:** The generator learns to produce only a few types of samples that fool the discriminator, rather than covering the full data distribution.

**Why it happens:** The generator can get a low loss by producing one very realistic sample type. The discriminator then focuses on that type, the generator shifts to another, and they cycle without covering all modes.

**Example:** If the real data has 10 clusters, the generator might only learn to produce samples from 2-3 clusters.

**Mathematical insight:** The minimax game may not converge to the Nash equilibrium. Instead, the players oscillate.

### 5.2 Training Instability

GAN training is a saddle-point optimization (minimax), not a simple minimization. This leads to:

- **Oscillating losses:** D and G losses oscillate rather than monotonically decreasing
- **Vanishing gradients:** If D becomes too good, G gets no useful gradient signal
- **Mode dropping:** Some modes of the distribution are never generated

### 5.3 Evaluation Difficulty

Unlike models with explicit likelihood, there's no single number that tells you how well a GAN is doing. Common metrics:

| Metric | Measures | Limitation |
|--------|----------|------------|
| **FID** (Fréchet Inception Distance) | Distribution similarity | Requires many samples |
| **IS** (Inception Score) | Quality + diversity | Doesn't compare to real data |
| **Precision/Recall** | Quality vs coverage separately | Requires density estimation |

---

## 6. Wasserstein GAN (WGAN)

### 6.1 The Problem with JSD

JSD has a fundamental flaw: when $p_{data}$ and $p_g$ have non-overlapping supports (common in high dimensions), $JSD = \log 2$ regardless of how close the distributions are.

**Example:** Let $P = \delta_0$ (point mass at 0) and $Q = \delta_\theta$ (point mass at $\theta$). Then:
- $JSD(P \| Q) = \log 2$ for all $\theta \neq 0$
- The gradient with respect to $\theta$ is **zero** — the generator gets no signal!

This is why GAN training can completely stall.

### 6.2 The Wasserstein Distance (Earth Mover's Distance)

The Wasserstein-1 distance is:

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

where $\Pi(P,Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

**Intuition:** Think of $P$ as a pile of dirt and $Q$ as a hole. $W_1$ is the minimum amount of work (mass × distance) needed to move the dirt into the hole.

**Key property:** For the point mass example, $W_1(\delta_0, \delta_\theta) = |\theta|$, which has a non-zero gradient everywhere! The generator always gets useful signal.

### 6.3 Kantorovich-Rubinstein Duality

Computing the infimum over transport plans is intractable. The dual form is:

$$W_1(P, Q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)]$$

where $\|f\|_L \leq 1$ means $f$ is 1-Lipschitz: $|f(x) - f(y)| \leq \|x - y\|$ for all $x, y$.

The WGAN objective replaces the discriminator (now called "critic") with a 1-Lipschitz function:

$$\min_G \max_{\|D\|_L \leq 1} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

### 6.4 Enforcing the Lipschitz Constraint

**Weight clipping (WGAN):** Clamp all weights to $[-c, c]$ after each update. Simple but crude — limits capacity.

**Gradient penalty (WGAN-GP):** Add a penalty term:

$$\lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

where $\hat{x} = \alpha x_{real} + (1-\alpha) x_{fake}$ with $\alpha \sim U(0,1)$.

This directly penalizes the critic for having gradients with norm different from 1, which is a necessary condition for 1-Lipschitz functions.

### 6.5 WGAN Advantages

| Aspect | Standard GAN | WGAN |
|--------|-------------|------|
| **Loss meaning** | Not interpretable | Estimates $W_1$ distance |
| **Gradient quality** | Can vanish | Always informative |
| **Mode collapse** | Common | Reduced |
| **Training stability** | Fragile | More robust |
| **D/G balance** | Critical | Less sensitive |

---

## 7. GAN Variants and Extensions

### 7.1 Conditional GAN (cGAN)

Condition both G and D on additional information $y$ (e.g., class label):

$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x|y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|y)|y))]$$

### 7.2 Deep Convolutional GAN (DCGAN)

Architecture guidelines that stabilize GAN training:
- Use strided convolutions instead of pooling
- Batch normalization in both G and D
- Remove fully connected layers
- ReLU in G, LeakyReLU in D

### 7.3 StyleGAN

Progressive growing + style-based generator:
- Mapping network $z \to w$ (intermediate latent space)
- Adaptive instance normalization (AdaIN) injects style at each layer
- Noise injection for stochastic variation
- Produces the most photorealistic face generation

### 7.4 Other Notable Variants

| Variant | Key Idea |
|---------|----------|
| **CycleGAN** | Unpaired image-to-image translation |
| **Pix2Pix** | Paired image-to-image translation |
| **ProGAN** | Progressive growing for high resolution |
| **BigGAN** | Scaling up with large batches |
| **GauGAN** | Semantic image synthesis |

---

## 8. Connections to Information Theory

### 8.1 GANs and Divergences

Different GAN formulations minimize different divergences:

| GAN Variant | Divergence Minimized |
|-------------|---------------------|
| Original GAN | Jensen-Shannon divergence |
| f-GAN | Any f-divergence |
| WGAN | Wasserstein-1 distance |
| MMD-GAN | Maximum Mean Discrepancy |

### 8.2 Connection to Variational Inference

The GAN discriminator can be seen as providing a variational bound:

$$D_{KL}(p_{data} \| p_g) \geq \mathbb{E}_{p_{data}}[\log D(x)] - \mathbb{E}_{p_g}[\log D(x)]$$

This connects GANs to the variational inference framework we studied in Tutorial 13 (VAEs).

### 8.3 Information-Theoretic Perspective

From our unifying framework:
- The generator compresses the data distribution into a deterministic mapping from noise
- The discriminator measures how much information distinguishes real from fake
- At equilibrium, there is zero mutual information between the real/fake label and the data

---

## 9. GANs in the Generative Model Landscape

### When to Use GANs

**Strengths:**
- Produce the sharpest, most realistic samples
- No restrictive architectural constraints (unlike flows)
- Fast sampling (single forward pass, unlike diffusion)
- Good for image-to-image translation tasks

**Weaknesses:**
- No explicit likelihood (can't evaluate how well the model explains data)
- Training is unstable and requires careful tuning
- Mode collapse remains a challenge
- Hard to diagnose when training goes wrong
- Largely superseded by diffusion models for image generation (2020+)

### Comparison with Other Generative Models

| Feature | GANs | VAEs | Flows | Diffusion |
|---------|------|------|-------|-----------|
| Sample quality | Excellent | Good | Good | Excellent |
| Training stability | Poor | Good | Good | Good |
| Likelihood | No | ELBO | Exact | Approximate |
| Sampling speed | Fast | Fast | Fast | Slow |
| Mode coverage | Partial | Full | Full | Full |

---

## 10. Historical Context and Impact

**2014:** Goodfellow et al. introduce GANs — the "adversarial" idea is born

**2015-2016:** DCGAN establishes architectural guidelines; training stabilizes somewhat

**2017:** WGAN introduces Wasserstein distance; Progressive GAN pushes resolution

**2018-2019:** StyleGAN produces photorealistic faces; BigGAN scales up

**2020-2021:** Diffusion models begin outperforming GANs on image quality metrics

**Legacy:** While diffusion models now dominate image generation, GANs remain relevant for:
- Real-time applications (fast sampling)
- Image-to-image translation
- Video generation
- The adversarial training idea lives on in many contexts (adversarial training for robustness, adversarial examples, etc.)

---

## Summary of Key Equations

| Concept | Equation |
|---------|----------|
| GAN objective | $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| Optimal discriminator | $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$ |
| Generator minimizes | $2 \cdot JSD(p_{data} \| p_g) - \log 4$ |
| Jensen-Shannon divergence | $JSD = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$ |
| Wasserstein distance | $W_1 = \sup_{\|f\|_L \leq 1} \mathbb{E}_P[f] - \mathbb{E}_Q[f]$ |
| Gradient penalty | $\lambda \mathbb{E}[(\|\nabla D(\hat{x})\|_2 - 1)^2]$ |
