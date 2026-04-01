# Tutorial 28: Causal Inference

## The Big Picture

Causal inference asks questions about **interventions**, not just associations.

Association asks:
$$P(Y \mid X=x)$$

Causation asks:
$$P(Y \mid do(X=x))$$

Those are different objects.

If umbrellas are associated with rain, opening an umbrella does not cause rain. Causal inference exists because observation alone does not tell us what would happen under intervention.

## Potential Outcomes View

For each unit, imagine:

- $Y(1)$: outcome under treatment
- $Y(0)$: outcome under control

The individual treatment effect is:
$$Y(1) - Y(0)$$

But we never observe both for the same unit.

This is the **fundamental problem of causal inference**.

## Average Treatment Effect

A central estimand is:
$$ATE = E[Y(1) - Y(0)]$$

The challenge is not defining it. The challenge is identifying it from observed data.

## Confounding

A variable $Z$ is a confounder if it affects both treatment assignment and outcome.

If you ignore confounders, simple comparisons like:
$$E[Y \mid T=1] - E[Y \mid T=0]$$

may reflect both treatment effect and selection bias.

## Randomized Experiments

Randomization breaks the link between treatment assignment and potential outcomes.

That is why randomized controlled trials are so powerful:
$$T \perp (Y(1), Y(0))$$

Then the difference in observed group means identifies the ATE.

## Observational Causal Inference

Without randomization, you need assumptions such as:

- **ignorability**: $(Y(1),Y(0)) \perp T \mid X$
- **positivity**: every covariate profile has some chance of both treatment and control
- **consistency**: observed outcome equals the relevant potential outcome under received treatment

Methods include:

- regression adjustment
- matching
- inverse propensity weighting
- doubly robust estimation

## DAG View

Directed acyclic graphs help represent causal structure.

Example:
$$X \leftarrow Z \rightarrow Y$$

Here $Z$ opens a backdoor path between $X$ and $Y$. Conditioning on $Z$ can block that confounding path.

## Visual Summary

<div align="center">
<svg viewBox="0 0 860 300" xmlns="http://www.w3.org/2000/svg" width="860" height="300" role="img" aria-label="Causal inference diagram">
  <text x="220" y="38" text-anchor="middle" font-size="22" font-weight="600">Association</text>
  <rect x="45" y="60" width="350" height="190" rx="16" fill="#f8fafc" stroke="#475569" stroke-width="2"/>
  <circle cx="130" cy="120" r="34" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>
  <circle cx="310" cy="120" r="34" fill="#fee2e2" stroke="#dc2626" stroke-width="2.5"/>
  <text x="130" y="126" text-anchor="middle" font-size="20" font-weight="600">X</text>
  <text x="310" y="126" text-anchor="middle" font-size="20" font-weight="600">Y</text>
  <line x1="164" y1="120" x2="276" y2="120" stroke="#7c3aed" stroke-width="3" marker-end="url(#arrowCausal)"/>
  <text x="220" y="105" text-anchor="middle" font-size="15" fill="#7c3aed">observed dependence</text>
  <text x="220" y="210" text-anchor="middle" font-size="14">correlation alone is ambiguous</text>

  <text x="645" y="38" text-anchor="middle" font-size="22" font-weight="600">Causal structure</text>
  <rect x="465" y="60" width="350" height="190" rx="16" fill="#f8fafc" stroke="#475569" stroke-width="2"/>
  <circle cx="645" cy="95" r="32" fill="#fde68a" stroke="#b45309" stroke-width="2.5"/>
  <circle cx="545" cy="180" r="32" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>
  <circle cx="745" cy="180" r="32" fill="#fee2e2" stroke="#dc2626" stroke-width="2.5"/>
  <text x="645" y="101" text-anchor="middle" font-size="20" font-weight="600">Z</text>
  <text x="545" y="186" text-anchor="middle" font-size="20" font-weight="600">T</text>
  <text x="745" y="186" text-anchor="middle" font-size="20" font-weight="600">Y</text>
  <line x1="622" y1="118" x2="568" y2="157" stroke="#374151" stroke-width="3" marker-end="url(#arrowCausal)"/>
  <line x1="668" y1="118" x2="722" y2="157" stroke="#374151" stroke-width="3" marker-end="url(#arrowCausal)"/>
  <line x1="577" y1="180" x2="713" y2="180" stroke="#16a34a" stroke-width="3" marker-end="url(#arrowCausal)"/>
  <text x="645" y="228" text-anchor="middle" font-size="14">adjust for Z to estimate effect of T on Y</text>

  <defs>
    <marker id="arrowCausal" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#374151"/>
    </marker>
  </defs>
</svg>
</div>

## Why Causal Inference Is Hard

Because the key comparison is counterfactual.

For one unit, you only observe one realized world, not both:

- treated and untreated
- intervened and not intervened

So causal inference always needs assumptions, design, or both.

## Main Identification Strategies

1. Randomization
2. Backdoor adjustment
3. Instrumental variables
4. Difference-in-differences
5. Regression discontinuity

Each identifies a causal effect under its own assumptions.

## Common Failure Modes

- hidden confounding
- post-treatment conditioning
- collider bias
- lack of overlap
- badly defined treatment or outcome

## The Central Habit of Causal Thinking

Always ask:

1. What intervention is being imagined?
2. What counterfactual quantity is the target?
3. What assumptions identify that quantity?
4. What variables should and should not be controlled for?

