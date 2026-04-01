# Tutorial 21: Frequentist Inference

## The Big Picture

Frequentist inference treats parameters as **fixed but unknown** and treats data as **random**.

You do not assign probabilities to $\theta$ itself. Instead, you design procedures whose behavior is good over repeated samples.

The central question is:

> If I repeated this experiment many times, how would my estimator or interval behave?

## Intuition

A frequentist confidence interval is like a manufacturing guarantee on a procedure.

- You do not say there is a 95% probability that this particular fixed parameter lies in this interval.
- You say the interval construction rule covers the true parameter in 95% of repeated experiments.

That distinction is subtle, but it is the backbone of frequentist thinking.

## Core Objects

### Estimator

An estimator is a function of the data:
$$\hat{\theta} = T(X_1, \dots, X_n)$$

Examples:

- sample mean
- sample variance
- least-squares estimator
- maximum likelihood estimator

### Sampling Distribution

Because the data are random, the estimator has a distribution:
$$\hat{\theta} \sim \text{sampling distribution}$$

Frequentist uncertainty comes from this distribution.

## Properties of Good Estimators

### Unbiasedness

$$E[\hat{\theta}] = \theta$$

This means the estimator is correct on average over repeated samples.

### Consistency

$$\hat{\theta}_n \xrightarrow{p} \theta$$

As the sample size grows, the estimator gets close to the truth.

### Efficiency

Among unbiased estimators, lower variance is better.

### Asymptotic normality

Many estimators satisfy:
$$\sqrt{n}(\hat{\theta} - \theta) \xrightarrow{d} \mathcal{N}(0, V)$$

This is why normal-based confidence intervals appear everywhere.

## Maximum Likelihood Estimation

Given data $x_1, \dots, x_n$, define:
$$L(\theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

The MLE is:
$$\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \log L(\theta)$$

### Why MLE matters

- usually intuitive
- often asymptotically efficient
- invariant under reparameterization
- foundation for likelihood ratio tests and large-sample intervals

## Confidence Intervals

A 95% confidence interval $C(X)$ satisfies:
$$P_\theta(\theta \in C(X)) = 0.95$$

The probability is over the random sample $X$, not over the fixed parameter.

For a Gaussian mean with known variance:
$$\bar{X} \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

## Hypothesis Testing

You specify:

- null hypothesis $H_0$
- alternative $H_1$
- test statistic
- rejection region

### p-value

The p-value is the probability, under $H_0$, of seeing a statistic at least as extreme as the one observed.

It is **not**:

- the probability that $H_0$ is true
- the probability the result happened by chance

## Visual Summary

<div align="center">
<svg viewBox="0 0 760 260" xmlns="http://www.w3.org/2000/svg" width="760" height="260" role="img" aria-label="Frequentist inference diagram">
  <line x1="70" y1="180" x2="690" y2="180" stroke="#374151" stroke-width="2"/>
  <text x="380" y="214" text-anchor="middle" font-size="15">possible estimates from repeated samples</text>

  <path d="M90 178 Q150 150 210 120 Q270 80 330 70 Q390 68 450 88 Q510 112 570 145 Q630 168 680 178" fill="none" stroke="#2563eb" stroke-width="4"/>

  <line x1="390" y1="45" x2="390" y2="180" stroke="#dc2626" stroke-width="3" stroke-dasharray="7 6"/>
  <text x="390" y="34" text-anchor="middle" font-size="16" fill="#dc2626" font-weight="600">true theta</text>

  <line x1="340" y1="100" x2="440" y2="100" stroke="#16a34a" stroke-width="8" stroke-linecap="round"/>
  <circle cx="390" cy="100" r="6" fill="#16a34a"/>
  <text x="390" y="86" text-anchor="middle" font-size="15" fill="#166534">one confidence interval that covers</text>

  <line x1="470" y1="130" x2="560" y2="130" stroke="#f59e0b" stroke-width="8" stroke-linecap="round"/>
  <circle cx="515" cy="130" r="6" fill="#f59e0b"/>
  <text x="515" y="116" text-anchor="middle" font-size="15" fill="#92400e">another interval</text>

  <line x1="580" y1="95" x2="650" y2="95" stroke="#7c3aed" stroke-width="8" stroke-linecap="round"/>
  <text x="615" y="80" text-anchor="middle" font-size="15" fill="#6d28d9">some intervals miss</text>

  <text x="160" y="36" font-size="20" font-weight="600">Repeated-sampling view</text>
  <text x="160" y="58" font-size="14">The method is judged by long-run coverage, bias, and variance.</text>
</svg>
</div>

## Strengths

- clear operating characteristics
- no need to specify priors
- strong asymptotic theory
- widely used in science and industry

## Common Misunderstandings

1. A confidence interval is not a posterior interval.
2. A p-value is not the probability the null is true.
3. Statistical significance is not practical significance.
4. Large samples do not protect you from bad design or confounding.

## Failure Modes

- optional stopping without correction
- multiple testing
- model misspecification
- small-sample approximations used too casually
- interpreting confidence statements as posterior beliefs

## When Frequentist Inference Is a Good Fit

- controlled experiments with repeated protocols
- large datasets with stable estimators
- settings where procedural guarantees matter
- regulatory contexts requiring long-run error control

