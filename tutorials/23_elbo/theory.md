# Tutorial 23: ELBO

## The Big Picture

The **Evidence Lower Bound (ELBO)** is a tractable objective used when exact posterior inference is hard.

It appears when we want to optimize $\log p(x)$ but cannot compute it directly.

The central identity is:
$$\log p(x) = \mathcal{L}(q) + D_{KL}(q(z \mid x) \| p(z \mid x))$$

where
$$\mathcal{L}(q) = E_{q(z \mid x)}[\log p(x,z) - \log q(z \mid x)]$$

Because KL divergence is nonnegative:
$$\mathcal{L}(q) \le \log p(x)$$

That is why it is called a lower bound.

## Intuition

The posterior $p(z \mid x)$ is the ideal answer, but often it is intractable.

So we choose a simpler distribution $q(z \mid x)$ and try to make it close to the true posterior.

The ELBO rewards two things:

1. putting mass on latent values that explain the data well
2. keeping the approximation itself honest and normalized

## Derivation

Start from:
$$\log p(x) = \log \int q(z \mid x) \frac{p(x,z)}{q(z \mid x)} dz$$

Apply Jensen's inequality:
$$\log p(x) \ge E_q \left[\log \frac{p(x,z)}{q(z \mid x)}\right]$$

Thus:
$$\mathcal{L}(q) = E_q[\log p(x,z)] - E_q[\log q(z \mid x)]$$

Using $p(x,z)=p(x \mid z)p(z)$:
$$\mathcal{L}(q)=E_q[\log p(x \mid z)] - D_{KL}(q(z \mid x)\|p(z))$$

This is the most common decomposition.

## The Two ELBO Terms

### Reconstruction or fit term

$$E_q[\log p(x \mid z)]$$

This says: under your approximate posterior, how well do latent variables explain the observation?

### Regularization term

$$-D_{KL}(q(z \mid x)\|p(z))$$

This says: do not wander too far from the prior unless the data really justifies it.

## Why the ELBO Matters

The ELBO is doing two jobs at once:

1. approximate inference
2. marginal likelihood optimization

If you maximize the ELBO with respect to both model parameters and variational parameters, you simultaneously:

- improve the model fit
- improve the posterior approximation inside your chosen family

## Geometry of the Gap

The gap to the true log evidence is:
$$\log p(x) - \mathcal{L}(q) = D_{KL}(q(z \mid x)\|p(z \mid x))$$

So:

- if $q$ equals the true posterior, the gap is zero
- if $q$ is restricted, the best you can do is the best approximation in that family

## Visual Summary

<div align="center">
<svg viewBox="0 0 780 280" xmlns="http://www.w3.org/2000/svg" width="780" height="280" role="img" aria-label="ELBO diagram">
  <rect x="50" y="70" width="180" height="140" rx="16" fill="#e8f1ff" stroke="#2563eb" stroke-width="2.5"/>
  <text x="140" y="110" text-anchor="middle" font-size="24" font-weight="600">log p(x)</text>
  <text x="140" y="145" text-anchor="middle" font-size="16">true log evidence</text>
  <text x="140" y="170" text-anchor="middle" font-size="16">hard to compute</text>

  <rect x="300" y="105" width="180" height="105" rx="16" fill="#ecfeef" stroke="#16a34a" stroke-width="2.5"/>
  <text x="390" y="143" text-anchor="middle" font-size="24" font-weight="600">ELBO</text>
  <text x="390" y="168" text-anchor="middle" font-size="16">tractable objective</text>

  <rect x="545" y="120" width="180" height="90" rx="16" fill="#faf5ff" stroke="#9333ea" stroke-width="2.5"/>
  <text x="635" y="155" text-anchor="middle" font-size="18" font-weight="600">Gap = KL(q || posterior)</text>
  <text x="635" y="180" text-anchor="middle" font-size="14">always nonnegative</text>

  <path d="M230 125 L300 145" stroke="#374151" stroke-width="3" marker-end="url(#arrowElbo)"/>
  <path d="M480 155 L545 160" stroke="#374151" stroke-width="3" marker-end="url(#arrowElbo)"/>

  <text x="390" y="42" text-anchor="middle" font-size="20" font-weight="600">Maximizing the ELBO shrinks the approximation gap</text>

  <defs>
    <marker id="arrowElbo" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#374151"/>
    </marker>
  </defs>
</svg>
</div>

## ELBO in Latent-Variable Models

For a VAE-like setup:
$$p_\theta(x,z)=p_\theta(x \mid z)p(z), \quad q_\phi(z \mid x)$$

The objective becomes:
$$\mathcal{L}(\theta,\phi;x)=E_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x)\|p(z))$$

This is why the ELBO sits at the center of modern representation learning.

## Common Misunderstandings

1. The ELBO is not the posterior.
2. A higher ELBO does not guarantee causal correctness or semantic correctness.
3. A tight ELBO still depends on model assumptions.
4. Optimizing the ELBO may still give a biased approximation if the variational family is too simple.

## Why ELBO and Not Exact Evidence?

Because exact evidence requires integrating over all latent variables:
$$p(x)=\int p(x,z)dz$$

In high dimensions or nonlinear models, this is usually intractable.

The ELBO converts integration difficulty into optimization.

