# Tutorial 20: Inference

## The Big Picture

Inference is the act of moving from **observations** to **unknown quantities**.

You observe data:
$$x_1, x_2, \dots, x_n$$

You care about something hidden:

- a parameter $\theta$
- a latent variable $z$
- a prediction $y$
- a causal effect $\tau$

Inference asks:

- What values are plausible?
- How uncertain are we?
- What should we predict next?
- What assumptions make the answer trustworthy?

## Intuition

Learning gives you a model. Inference is what you do **with** the model and the data you have.

There are three recurring ingredients:

1. **A data-generating story**
   You assume how data could have been produced.
2. **Unknowns**
   Parameters, hidden states, future outcomes, treatment effects.
3. **An update rule**
   Some method for turning observed evidence into a judgment about those unknowns.

In that sense, inference is the bridge between probability theory and decision making.

## A Minimal Formal Template

Most inference problems can be written as:
$$\text{unknowns} \to \text{produce data} \to \text{observe data} \to \text{reason backward}$$

For example:
$$\theta \to x \sim p(x \mid \theta) \to \text{observe } x \to \text{infer } \theta$$

This is the common backbone behind:

- parameter estimation
- confidence intervals
- posterior distributions
- latent-variable inference
- causal effect estimation

## Two Grand Styles of Inference

### Frequentist

Treat $\theta$ as fixed but unknown.

- Data is random
- Estimators are random because they depend on data
- Uncertainty is about repeated sampling

### Bayesian

Treat $\theta$ as uncertain and represent that uncertainty with a distribution.

- Prior: $p(\theta)$
- Likelihood: $p(x \mid \theta)$
- Posterior: $p(\theta \mid x)$

## Inference vs Prediction vs Decision

These are related but distinct.

- **Inference** asks what is true or plausible.
- **Prediction** asks what will happen next.
- **Decision** asks what action minimizes loss.

You can predict well without understanding causes. You can estimate a parameter without having a good decision rule. Good statistical practice keeps these tasks separate.

## The Core Workflow

### 1. Specify assumptions

Examples:

- observations are independent
- noise is Gaussian
- treatment assignment is ignorable
- latent variables are low-dimensional

### 2. Write the target quantity

Examples:

- $E[X]$
- $\theta$
- $p(z \mid x)$
- $E[Y(1) - Y(0)]$

### 3. Choose an inferential method

Examples:

- maximum likelihood
- confidence intervals
- Bayes rule
- MCMC
- variational inference
- inverse propensity weighting

### 4. Diagnose failure modes

Examples:

- model misspecification
- weak identifiability
- biased sampling
- poor mixing
- unmeasured confounding

## Important Distinctions

### Estimand vs Estimator

- **Estimand**: the true quantity you want, such as population mean $\mu$
- **Estimator**: the procedure you compute from data, such as sample mean $\bar{x}$

### Identifiability vs Computability

A quantity may be well-defined but impossible to recover from observed data alone.

- **Identifiable** means the data distribution determines the target
- **Computable** means you can actually approximate or calculate it efficiently

Variational inference often helps with computability.
Causal assumptions often help with identifiability.

## Visual Summary

<div align="center">
<svg viewBox="0 0 760 240" xmlns="http://www.w3.org/2000/svg" width="760" height="240" role="img" aria-label="Inference overview diagram">
  <rect x="20" y="70" width="150" height="90" rx="14" fill="#e8f1ff" stroke="#3b82f6" stroke-width="2"/>
  <text x="95" y="108" text-anchor="middle" font-size="20" font-weight="600">Unknowns</text>
  <text x="95" y="132" text-anchor="middle" font-size="14">theta, z, tau</text>

  <rect x="220" y="70" width="150" height="90" rx="14" fill="#ecfeef" stroke="#16a34a" stroke-width="2"/>
  <text x="295" y="108" text-anchor="middle" font-size="20" font-weight="600">Model</text>
  <text x="295" y="132" text-anchor="middle" font-size="14">p(data | unknowns)</text>

  <rect x="420" y="70" width="150" height="90" rx="14" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/>
  <text x="495" y="108" text-anchor="middle" font-size="20" font-weight="600">Observed Data</text>
  <text x="495" y="132" text-anchor="middle" font-size="14">x1, x2, ..., xn</text>

  <rect x="620" y="70" width="120" height="90" rx="14" fill="#faf5ff" stroke="#9333ea" stroke-width="2"/>
  <text x="680" y="108" text-anchor="middle" font-size="20" font-weight="600">Answer</text>
  <text x="680" y="132" text-anchor="middle" font-size="14">estimate or distribution</text>

  <path d="M170 115 L220 115" stroke="#374151" stroke-width="3" marker-end="url(#arrow)"/>
  <path d="M370 115 L420 115" stroke="#374151" stroke-width="3" marker-end="url(#arrow)"/>
  <path d="M620 150 C590 190, 470 215, 310 205 C210 198, 130 178, 110 160" fill="none" stroke="#7c3aed" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#arrow)"/>

  <text x="345" y="34" text-anchor="middle" font-size="18" font-weight="600">Forward generation</text>
  <text x="350" y="228" text-anchor="middle" font-size="18" font-weight="600" fill="#7c3aed">Backward reasoning = inference</text>

  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#374151"/>
    </marker>
  </defs>
</svg>
</div>

## Why Inference Gets Hard

Inference becomes difficult when:

- the posterior is high-dimensional
- the likelihood is intractable
- data is biased or missing
- multiple explanations fit equally well
- causal counterfactuals are unobserved by definition

That is why modern inference includes both statistical ideas and computational ideas.

## A Mental Checklist

Whenever you read an inferential claim, ask:

1. What is the target quantity?
2. What assumptions connect data to target?
3. Is the quantity identifiable?
4. How is uncertainty quantified?
5. What breaks if the assumptions are wrong?

## Where the Next Tutorials Fit

- Frequentist inference: repeated-sampling guarantees
- Bayesian inference: belief updating with priors
- ELBO and variational inference: optimization-based approximate inference
- MCMC: simulation-based approximate inference
- Amortized inference: learning an inference procedure once and reusing it
- Causal inference: inference about interventions, not just associations

