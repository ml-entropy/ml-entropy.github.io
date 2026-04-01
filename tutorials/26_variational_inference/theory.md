# Tutorial 26: Variational Inference

## The Big Picture

**Variational inference (VI)** approximates an intractable posterior with a tractable distribution and turns inference into optimization.

Instead of sampling from:
$$p(z \mid x)$$

we choose a family:
$$q(z; \lambda) \in \mathcal{Q}$$

and solve:
$$\lambda^* = \arg\min_\lambda D_{KL}(q(z;\lambda) \| p(z \mid x))$$

Since the posterior is usually known only up to a normalizing constant, this is implemented by maximizing the ELBO.

## Intuition

Exact Bayesian inference is often an integration problem we cannot do.

Variational inference says:

1. pick a manageable family of distributions
2. search inside that family for the best approximation

This is like approximating a complicated curve by the best curve from a simpler class.

## Mean-Field Variational Inference

A common simplification is factorization:
$$q(z) = \prod_{j=1}^d q_j(z_j)$$

This assumption makes optimization much easier, but it can underestimate posterior dependence.

### Coordinate update idea

In mean-field VI:
$$\log q_j^*(z_j) = E_{-j}[\log p(x,z)] + \text{constant}$$

This yields iterative coordinate updates for each factor.

## Reverse KL and Its Consequences

VI often minimizes:
$$D_{KL}(q \| p)$$

This is called reverse KL.

Why it matters:

- strongly penalizes putting mass where $p$ is tiny
- less strongly penalizes missing some modes of $p$

So VI can become **mode-seeking**.

## Visual Summary

<div align="center">
<svg viewBox="0 0 820 280" xmlns="http://www.w3.org/2000/svg" width="820" height="280" role="img" aria-label="Variational inference diagram">
  <rect x="40" y="55" width="330" height="185" rx="16" fill="#f8fafc" stroke="#475569" stroke-width="2"/>
  <text x="205" y="85" text-anchor="middle" font-size="22" font-weight="600">True posterior p(z|x)</text>
  <path d="M70 205 Q110 165 150 130 Q190 90 225 120 Q255 150 280 145 Q310 138 340 205" fill="rgba(59,130,246,0.18)" stroke="#2563eb" stroke-width="3"/>
  <text x="205" y="225" text-anchor="middle" font-size="14">complex shape, often multimodal or correlated</text>

  <rect x="450" y="55" width="330" height="185" rx="16" fill="#f8fafc" stroke="#475569" stroke-width="2"/>
  <text x="615" y="85" text-anchor="middle" font-size="22" font-weight="600">Variational q(z)</text>
  <ellipse cx="615" cy="155" rx="90" ry="45" fill="rgba(22,163,74,0.18)" stroke="#16a34a" stroke-width="3"/>
  <text x="615" y="225" text-anchor="middle" font-size="14">simple family chosen for tractability</text>

  <path d="M370 150 L450 150" stroke="#7c3aed" stroke-width="3" marker-end="url(#arrowVi)"/>
  <text x="410" y="136" text-anchor="middle" font-size="15" fill="#7c3aed">optimize KL</text>

  <defs>
    <marker id="arrowVi" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#7c3aed"/>
    </marker>
  </defs>
</svg>
</div>

## Why VI Is Popular

- much faster than MCMC in large systems
- fits naturally into gradient-based optimization
- easy to combine with deep learning
- often works well enough for large-scale latent-variable models

## Mean-Field Tradeoff

Factorization helps computation, but it often causes:

- underestimated posterior variance
- missing correlations
- overconfident conclusions

These are not bugs in the optimizer. They are often consequences of the approximation family.

## Coordinate-Ascent VI

When the model is conjugate, you can often derive exact updates for each variational factor.

This gives a deterministic alternative to MCMC:

- update one factor
- update the next
- repeat until the ELBO stabilizes

## Black-Box VI

For modern models, exact updates are rare.

Instead we use:

- Monte Carlo gradient estimates
- reparameterization gradients
- stochastic optimization

This is how VI scales to deep latent-variable models.

## VI vs MCMC

VI:

- optimization-based
- faster
- deterministic once initialized
- biased by the variational family

MCMC:

- sampling-based
- slower
- asymptotically exact
- often harder to scale

## Common Failure Modes

- mean-field factorization too crude
- local optima
- poor variance estimates
- ELBO improves while a task-relevant metric does not

