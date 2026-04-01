# Tutorial 27: Amortized Variational Inference

## The Big Picture

**Amortized variational inference (AVI)** combines:

- the optimization objective of variational inference
- the reusable encoder idea of amortized inference

Instead of learning separate variational parameters $\lambda_n$ for each datapoint, AVI learns shared parameters $\phi$ so that:
$$q_\phi(z \mid x)$$

directly outputs an approximate posterior for each observation.

This is the core inference mechanism behind VAEs.

## Intuition

Variational inference asks:
"What tractable distribution best approximates the posterior for this datapoint?"

Amortized variational inference asks:
"Can I train a function that answers that question quickly for many datapoints?"

This makes AVI a learned, reusable posterior approximation engine.

## Objective

For a dataset $\{x_n\}_{n=1}^N$, optimize:
$$\sum_{n=1}^N \mathcal{L}(\theta,\phi;x_n)$$

where
$$\mathcal{L}(\theta,\phi;x)=E_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x)\|p(z))$$

Here:

- $\theta$ are generative model parameters
- $\phi$ are inference-network parameters

## Encoder-Decoder View

- **Encoder**: approximates posterior inference
- **Decoder**: defines the generative model

The encoder is not just a feature extractor. It is an approximation to Bayes' rule under the trained model.

## Reparameterization Trick

For continuous latents, AVI often relies on:
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

This lets gradients flow through stochastic latent samples.

## Visual Summary

<div align="center">
<svg viewBox="0 0 860 300" xmlns="http://www.w3.org/2000/svg" width="860" height="300" role="img" aria-label="Amortized variational inference diagram">
  <rect x="40" y="85" width="130" height="70" rx="12" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>
  <text x="105" y="126" text-anchor="middle" font-size="20" font-weight="600">x</text>

  <rect x="230" y="65" width="160" height="110" rx="14" fill="#ecfeef" stroke="#16a34a" stroke-width="2.5"/>
  <text x="310" y="112" text-anchor="middle" font-size="22" font-weight="600">Encoder</text>
  <text x="310" y="138" text-anchor="middle" font-size="15">q_phi(z|x)</text>

  <rect x="450" y="85" width="120" height="70" rx="12" fill="#faf5ff" stroke="#9333ea" stroke-width="2.5"/>
  <text x="510" y="126" text-anchor="middle" font-size="20" font-weight="600">z</text>

  <rect x="630" y="65" width="170" height="110" rx="14" fill="#fff7ed" stroke="#ea580c" stroke-width="2.5"/>
  <text x="715" y="112" text-anchor="middle" font-size="22" font-weight="600">Decoder</text>
  <text x="715" y="138" text-anchor="middle" font-size="15">p_theta(x|z)</text>

  <path d="M170 120 L230 120" stroke="#374151" stroke-width="3" marker-end="url(#arrowAvi)"/>
  <path d="M390 120 L450 120" stroke="#374151" stroke-width="3" marker-end="url(#arrowAvi)"/>
  <path d="M570 120 L630 120" stroke="#374151" stroke-width="3" marker-end="url(#arrowAvi)"/>

  <path d="M715 175 C690 220, 565 250, 310 250 C170 250, 120 215, 105 155" fill="none" stroke="#7c3aed" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#arrowAvi)"/>
  <text x="455" y="285" text-anchor="middle" font-size="17" fill="#7c3aed">train encoder and decoder jointly via the ELBO</text>

  <defs>
    <marker id="arrowAvi" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#374151"/>
    </marker>
  </defs>
</svg>
</div>

## Why AVI Changed Modern Generative Modeling

Without amortization, variational inference for large neural latent-variable models would be painfully slow.

AVI makes latent-variable learning practical by making inference itself differentiable and reusable.

## AVI vs Plain VI

Plain VI:

- often uses local variational parameters per datapoint
- may need iterative optimization for each new example

AVI:

- uses one inference network for all datapoints
- is much faster at test time
- introduces amortization error

## AVI vs Generic Amortized Inference

Amortized inference is a broad concept.
AVI is the specific case where the amortized inference network is trained through a variational objective such as the ELBO.

## Common Failure Modes

- posterior collapse
- amortization gap
- underpowered encoder
- weak latent usage when the decoder is too expressive

## Practical Remedies

- KL annealing
- richer posterior families
- semi-amortized refinement
- architectural constraints that force latent usage

