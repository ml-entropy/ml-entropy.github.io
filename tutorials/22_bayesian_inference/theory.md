# Tutorial 22: Bayesian Inference

## The Big Picture

Bayesian inference updates uncertainty about unknowns using Bayes' rule:
$$p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)}$$

The posterior combines:

- **prior** knowledge or assumptions
- **likelihood** from the data model
- **evidence** as the normalizing constant

The result is a full distribution over plausible parameter values.

## Intuition

Bayesian inference is disciplined belief revision.

You begin with a range of plausible values. Data reweights those values:

- values that explain the data well gain mass
- values that explain the data poorly lose mass

The posterior is not a point estimate. It is a map of uncertainty.

## The Three Core Ingredients

### Prior

$$p(\theta)$$

This encodes what values looked plausible before seeing current data.

Priors can be:

- informative
- weakly informative
- noninformative or reference-style
- hierarchical

### Likelihood

$$p(x \mid \theta)$$

This describes how parameters produce observations.

### Posterior

$$p(\theta \mid x) \propto p(x \mid \theta) p(\theta)$$

The posterior is what we actually want for inference.

## Conjugate Example: Beta-Binomial

Suppose:
$$x \mid p \sim \text{Binomial}(n, p)$$
and prior:
$$p \sim \text{Beta}(\alpha, \beta)$$

Then the posterior is:
$$p \mid x \sim \text{Beta}(\alpha + x, \beta + n - x)$$

### Why this matters

- easy algebra
- easy interpretation as pseudo-counts
- clear sequential updating

## Posterior Predictive Distribution

Bayesian inference naturally supports prediction:
$$p(x_{new} \mid x) = \int p(x_{new} \mid \theta) p(\theta \mid x) \, d\theta$$

This integrates over parameter uncertainty instead of pretending one estimate is exact.

## MAP vs Full Bayes

### MAP

Maximum a posteriori estimation chooses:
$$\theta_{MAP} = \arg\max_\theta p(\theta \mid x)$$

Equivalent to:
$$\arg\max_\theta \log p(x \mid \theta) + \log p(\theta)$$

MAP gives one point. Full Bayes keeps the full posterior.

### Important difference

MAP is not the same as Bayesian inference in its fullest sense. It is a summary, not the whole answer.

## Credible Intervals

A 95% credible interval means:

> Under the model and prior, the posterior assigns 95% probability to the parameter lying in that interval.

This is a probability statement about the parameter itself, conditional on the model.

## Visual Summary

<div align="center">
<svg viewBox="0 0 780 280" xmlns="http://www.w3.org/2000/svg" width="780" height="280" role="img" aria-label="Bayesian inference diagram">
  <text x="140" y="34" text-anchor="middle" font-size="22" font-weight="600">Prior</text>
  <text x="390" y="34" text-anchor="middle" font-size="22" font-weight="600">Likelihood effect</text>
  <text x="640" y="34" text-anchor="middle" font-size="22" font-weight="600">Posterior</text>

  <line x1="40" y1="210" x2="240" y2="210" stroke="#374151" stroke-width="2"/>
  <line x1="290" y1="210" x2="490" y2="210" stroke="#374151" stroke-width="2"/>
  <line x1="540" y1="210" x2="740" y2="210" stroke="#374151" stroke-width="2"/>

  <path d="M55 208 Q90 180 115 120 Q140 70 170 120 Q195 180 225 208" fill="rgba(59,130,246,0.18)" stroke="#2563eb" stroke-width="3"/>
  <text x="140" y="238" text-anchor="middle" font-size="14">broad prior belief</text>

  <path d="M305 208 Q350 195 380 110 Q395 68 410 110 Q440 195 475 208" fill="rgba(249,115,22,0.18)" stroke="#ea580c" stroke-width="3"/>
  <text x="390" y="238" text-anchor="middle" font-size="14">data favors this region</text>

  <path d="M555 208 Q590 190 615 105 Q640 52 670 100 Q700 170 725 208" fill="rgba(22,163,74,0.18)" stroke="#16a34a" stroke-width="3"/>
  <text x="640" y="238" text-anchor="middle" font-size="14">updated posterior</text>

  <path d="M240 135 L290 135" stroke="#7c3aed" stroke-width="3" marker-end="url(#arrowBayes)"/>
  <path d="M490 135 L540 135" stroke="#7c3aed" stroke-width="3" marker-end="url(#arrowBayes)"/>
  <text x="390" y="120" text-anchor="middle" font-size="16" fill="#7c3aed">multiply and normalize</text>

  <defs>
    <marker id="arrowBayes" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#7c3aed"/>
    </marker>
  </defs>
</svg>
</div>

## Strengths

- direct uncertainty quantification
- natural incorporation of prior knowledge
- coherent sequential updating
- natural predictive distributions
- good fit for hierarchical modeling

## Common Objections and Replies

### "Priors are subjective"

Yes, but modeling choices are always assumptions. Bayesian methods make some assumptions explicit instead of hiding them.

### "Bayesian methods are computationally hard"

Often true. That is why MCMC, variational inference, and amortized inference matter.

### "With enough data priors do not matter"

Sometimes approximately true, but not always:

- weak-data settings
- hierarchical models
- non-identifiable models
- tail probabilities

## Failure Modes

- badly chosen priors
- overconfident likelihood assumptions
- posterior sensitivity hidden from the reader
- computational approximation mistaken for exact inference

## A Good Bayesian Habit

Always ask:

1. What prior was used?
2. How sensitive are the results to that prior?
3. How was the posterior approximated?
4. Were posterior predictive checks performed?

