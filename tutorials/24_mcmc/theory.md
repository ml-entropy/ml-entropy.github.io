# Tutorial 24: MCMC

## The Big Picture

**Markov Chain Monte Carlo (MCMC)** is a family of algorithms for sampling from complicated probability distributions.

If direct sampling from $p(\theta \mid x)$ is impossible, MCMC constructs a Markov chain whose stationary distribution is the target posterior.

Then averages over the chain approximate posterior expectations:
$$E_{p(\theta \mid x)}[f(\theta)] \approx \frac{1}{T}\sum_{t=1}^T f(\theta^{(t)})$$

## Intuition

Imagine a landscape whose height is proportional to posterior density.

- high hills = plausible regions
- deep valleys = implausible regions

MCMC sends a walker through that landscape.
The walker spends more time in high-density regions, so the visited states eventually look like samples from the target.

## Markov Chain Idea

A Markov chain updates by:
$$\theta^{(t+1)} \sim K(\cdot \mid \theta^{(t)})$$

where the next state depends only on the current state.

The design goal is:
$$\pi(\theta) = p(\theta \mid x)$$

should be the stationary distribution of the transition kernel $K$.

## Metropolis-Hastings

### Proposal

Given current state $\theta$, propose:
$$\theta' \sim q(\theta' \mid \theta)$$

### Accept-reject step

Accept with probability:
$$\alpha(\theta,\theta') = \min\left(1, \frac{\pi(\theta') q(\theta \mid \theta')}{\pi(\theta) q(\theta' \mid \theta)}\right)$$

If accepted, move to $\theta'$. Otherwise stay put.

### Why this works

The acceptance rule enforces detailed balance, which makes $\pi$ stationary.

## Gibbs Sampling

If conditional distributions are easy, sample coordinates one at a time:
$$\theta_1 \sim p(\theta_1 \mid \theta_2,\dots,\theta_d,x)$$
$$\theta_2 \sim p(\theta_2 \mid \theta_1,\theta_3,\dots,\theta_d,x)$$

Gibbs is often simpler than Metropolis when conditionals are known.

## Diagnostics

MCMC is approximate in practice because you only run a finite chain.

Key concepts:

- **burn-in**: early transient iterations
- **mixing**: how quickly the chain explores the target
- **autocorrelation**: dependence between nearby draws
- **effective sample size**: number of near-independent samples represented by the chain

## Visual Summary

<div align="center">
<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg" width="800" height="300" role="img" aria-label="MCMC sampling diagram">
  <rect x="35" y="45" width="330" height="210" rx="16" fill="#f8fafc" stroke="#475569" stroke-width="2"/>
  <text x="200" y="75" text-anchor="middle" font-size="22" font-weight="600">Target density</text>
  <path d="M60 220 Q110 215 145 160 Q175 95 220 120 Q255 150 285 110 Q315 70 340 220" fill="rgba(59,130,246,0.18)" stroke="#2563eb" stroke-width="3"/>
  <circle cx="145" cy="160" r="6" fill="#dc2626"/>
  <circle cx="220" cy="120" r="6" fill="#dc2626"/>
  <circle cx="285" cy="110" r="6" fill="#dc2626"/>
  <text x="200" y="245" text-anchor="middle" font-size="14">walker spends more time in high-density regions</text>

  <rect x="435" y="45" width="330" height="210" rx="16" fill="#f8fafc" stroke="#475569" stroke-width="2"/>
  <text x="600" y="75" text-anchor="middle" font-size="22" font-weight="600">Trace plot intuition</text>
  <polyline points="460,190 485,145 510,170 535,105 560,115 585,95 610,155 635,120 660,130 685,85 710,98 735,125" fill="none" stroke="#16a34a" stroke-width="3"/>
  <line x1="500" y1="90" x2="500" y2="230" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 6"/>
  <text x="493" y="248" text-anchor="middle" font-size="13" fill="#dc2626">burn-in</text>
  <text x="625" y="245" text-anchor="middle" font-size="14">after burn-in, use the chain for estimates</text>

  <path d="M365 150 L435 150" stroke="#7c3aed" stroke-width="3" marker-end="url(#arrowMcmc)"/>
  <text x="400" y="137" text-anchor="middle" font-size="14" fill="#7c3aed">sample</text>

  <defs>
    <marker id="arrowMcmc" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#7c3aed"/>
    </marker>
  </defs>
</svg>
</div>

## Strengths

- asymptotically exact for many targets
- flexible for complicated posteriors
- useful for full Bayesian uncertainty quantification

## Weaknesses

- can mix very slowly
- expensive in high dimensions
- diagnostics are necessary but imperfect
- correlated samples reduce efficiency

## MCMC vs Variational Inference

MCMC:

- slower
- often more accurate asymptotically
- gives sample-based posterior approximations

Variational inference:

- faster
- optimization-based
- often biased but scalable

## Common Failure Modes

- poor proposal scale
- multimodal posteriors with mode trapping
- false convergence from one short chain
- ignoring autocorrelation when estimating uncertainty

