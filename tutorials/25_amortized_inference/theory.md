# Tutorial 25: Amortized Inference

## The Big Picture

Classical inference solves a new optimization or sampling problem for each observation $x$.

**Amortized inference** replaces that repeated work with a learned inference function:
$$x \mapsto q_\phi(z \mid x)$$

Instead of inferring latent variables from scratch every time, you train a network once and reuse it on many inputs.

## Intuition

Suppose you process a million images.

Without amortization:

- for each image, run a new optimization loop or a new local variational procedure

With amortization:

- train an encoder once
- then each new image gets a fast forward pass

You pay a larger up-front training cost in exchange for cheap repeated inference later.

That is the amortization idea: spread the cost over many future inferences.

## Formal Setup

Let $x$ be data and $z$ latent variables.

In classical variational inference you may choose separate local variational parameters for each datapoint:
$$q(z_n; \lambda_n)$$

In amortized inference you instead use:
$$q_\phi(z_n \mid x_n)$$

where $\phi$ are shared parameters across all datapoints.

## Why It Matters

Amortized inference is essential when:

- datasets are large
- inference is repeated many times
- you need fast test-time posterior approximations
- the model is used in an online system

## Amortization Gap

Amortized inference is fast, but it introduces a new source of error.

Even if the variational family is expressive enough, the shared encoder may fail to output the best local approximation for each datapoint.

This difference is called the **amortization gap**.

Informally:

- **approximation gap**: the family cannot represent the true posterior well
- **amortization gap**: the encoder fails to find the best member of the family for a given input

## Visual Summary

<div align="center">
<svg viewBox="0 0 820 280" xmlns="http://www.w3.org/2000/svg" width="820" height="280" role="img" aria-label="Amortized inference diagram">
  <text x="210" y="38" text-anchor="middle" font-size="22" font-weight="600">Classical local inference</text>
  <rect x="40" y="60" width="340" height="180" rx="16" fill="#fff7ed" stroke="#ea580c" stroke-width="2.5"/>
  <rect x="70" y="95" width="70" height="45" rx="10" fill="#fde68a" stroke="#b45309"/>
  <rect x="175" y="95" width="70" height="45" rx="10" fill="#fde68a" stroke="#b45309"/>
  <rect x="280" y="95" width="70" height="45" rx="10" fill="#fde68a" stroke="#b45309"/>
  <text x="105" y="123" text-anchor="middle" font-size="15">x1</text>
  <text x="210" y="123" text-anchor="middle" font-size="15">x2</text>
  <text x="315" y="123" text-anchor="middle" font-size="15">x3</text>
  <text x="210" y="170" text-anchor="middle" font-size="16">run a separate inference procedure each time</text>
  <text x="210" y="197" text-anchor="middle" font-size="14">slow but locally specialized</text>

  <text x="610" y="38" text-anchor="middle" font-size="22" font-weight="600">Amortized inference</text>
  <rect x="440" y="60" width="340" height="180" rx="16" fill="#ecfeef" stroke="#16a34a" stroke-width="2.5"/>
  <rect x="565" y="95" width="90" height="55" rx="12" fill="#bbf7d0" stroke="#15803d"/>
  <text x="610" y="127" text-anchor="middle" font-size="16" font-weight="600">Encoder</text>
  <rect x="470" y="175" width="60" height="34" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <rect x="580" y="175" width="60" height="34" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <rect x="690" y="175" width="60" height="34" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="500" y="197" text-anchor="middle" font-size="14">q(z|x1)</text>
  <text x="610" y="197" text-anchor="middle" font-size="14">q(z|x2)</text>
  <text x="720" y="197" text-anchor="middle" font-size="14">q(z|x3)</text>
  <line x1="500" y1="110" x2="565" y2="110" stroke="#374151" stroke-width="2.5" marker-end="url(#arrowAmortized)"/>
  <line x1="610" y1="110" x2="610" y2="150" stroke="#374151" stroke-width="2.5" marker-end="url(#arrowAmortized)"/>
  <line x1="720" y1="110" x2="655" y2="110" stroke="#374151" stroke-width="2.5" marker-end="url(#arrowAmortized)"/>
  <text x="610" y="227" text-anchor="middle" font-size="14">shared parameters phi reused across all inputs</text>

  <defs>
    <marker id="arrowAmortized" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#374151"/>
    </marker>
  </defs>
</svg>
</div>

## Where It Appears

- variational autoencoders
- recognition networks
- learned proposal distributions
- meta-inference systems

## Strengths

- very fast test-time inference
- scalable to large datasets
- shared statistical strength across datapoints

## Weaknesses

- inference quality may be worse for difficult examples
- optimization can favor average-case accuracy over hard cases
- encoder architecture choices matter a lot

## Common Fixes for the Amortization Gap

- richer encoders
- iterative refinement
- normalizing-flow variational families
- semi-amortized inference

