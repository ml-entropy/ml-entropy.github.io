# Tutorial 07: Combinatorics — The Foundation of Counting

## Introduction

Before we can compute probabilities, we need to **count outcomes**. Combinatorics is the mathematics of counting. This tutorial derives all the key formulas from first principles.

---

## Part 1: The Fundamental Counting Principle

### 1.1 The Multiplication Principle

If you make a sequence of choices where:
- Choice 1 has $n_1$ options
- Choice 2 has $n_2$ options
- ...
- Choice $k$ has $n_k$ options

Then the total number of sequences is:
$$n_1 \times n_2 \times ... \times n_k$$

**Example**: License plate with 3 letters followed by 3 digits
- 26 choices for each letter, 10 choices for each digit
- Total: $26 \times 26 \times 26 \times 10 \times 10 \times 10 = 17,576,000$

### 1.2 Why Multiplication?

**Intuition via tree diagram**:

```
First letter: A, B, C, ... (26 branches)
              |
              v
Second letter: For EACH first choice, 26 more branches
              = 26 × 26 = 676 total paths so far
              |
              v
Third letter: For EACH of 676 paths, 26 more branches
              = 676 × 26 = 17,576 paths
              ...
```

Each choice **multiplies** the number of paths by the number of options.

---

## Part 2: Permutations (Arrangements)

### 2.1 What is a Permutation?

A **permutation** is an **ordered arrangement** of objects.

**Key word**: ORDER MATTERS

**Example**: Arranging 3 people (A, B, C) in a line
- ABC, ACB, BAC, BCA, CAB, CBA
- 6 arrangements, each is a different permutation

### 2.2 Permutations of n Distinct Objects

**Question**: How many ways to arrange $n$ distinct objects in a line?

**Derivation**:
- Position 1: $n$ choices (any object)
- Position 2: $n-1$ choices (one used)
- Position 3: $n-2$ choices
- ...
- Position $n$: 1 choice (last remaining)

By multiplication principle:
$$P(n) = n \times (n-1) \times (n-2) \times ... \times 2 \times 1 = n!$$

**Definition**: $n! = n \times (n-1) \times ... \times 1$ ("n factorial")

**Special case**: $0! = 1$ (by convention, there's one way to arrange nothing)

### 2.3 Permutations of n Objects Taken r at a Time

**Question**: How many ways to arrange $r$ objects chosen from $n$ distinct objects?

**Notation**: $P(n, r)$ or $^nP_r$ or $P_n^r$

**Derivation**:
- Position 1: $n$ choices
- Position 2: $n-1$ choices
- ...
- Position $r$: $n-r+1$ choices

$$P(n, r) = n \times (n-1) \times ... \times (n-r+1)$$

**Alternative formula**:
$$P(n, r) = \frac{n!}{(n-r)!}$$

**Proof**:
$$\frac{n!}{(n-r)!} = \frac{n \times (n-1) \times ... \times (n-r+1) \times (n-r)!}{(n-r)!} = n \times (n-1) \times ... \times (n-r+1)$$

**Example**: How many 3-letter "words" from {A, B, C, D, E}?
$$P(5, 3) = 5 \times 4 \times 3 = 60$$
or equivalently: $\frac{5!}{2!} = \frac{120}{2} = 60$

### 2.4 Permutations with Repetition

**Question**: Arrange $n$ objects where some are identical.

If we have:
- $n_1$ identical objects of type 1
- $n_2$ identical objects of type 2
- ...
- $n_k$ identical objects of type $k$

where $n_1 + n_2 + ... + n_k = n$

Then the number of distinct arrangements is:
$$\frac{n!}{n_1! \times n_2! \times ... \times n_k!}$$

**Derivation**:

1. If all $n$ objects were distinct: $n!$ arrangements
2. But swapping identical objects gives the same arrangement
3. For type 1: $n_1!$ ways to swap them among themselves (all look the same)
4. So we **divide** by $n_1!$ to remove overcounting
5. Same for each type

**Example**: Arrangements of "MISSISSIPPI"
- 11 letters total
- M: 1, I: 4, S: 4, P: 2

$$\frac{11!}{1! \times 4! \times 4! \times 2!} = \frac{39,916,800}{1 \times 24 \times 24 \times 2} = 34,650$$

---

## Part 3: Combinations (Selections)

### 3.1 What is a Combination?

A **combination** is an **unordered selection** of objects.

**Key word**: ORDER DOESN'T MATTER

**Example**: Choosing 2 people from {A, B, C}
- {A, B}, {A, C}, {B, C}
- Only 3 combinations (AB = BA, they're the same selection)

### 3.2 Deriving the Combination Formula

**Question**: How many ways to choose $r$ objects from $n$ distinct objects (order doesn't matter)?

**Notation**: $C(n, r)$ or $\binom{n}{r}$ or $^nC_r$ ("n choose r")

**Derivation** (the key insight):

**Relationship between permutations and combinations**:

If we:
1. First **choose** $r$ objects: $C(n, r)$ ways
2. Then **arrange** those $r$ objects: $r!$ ways

We get all permutations of $n$ objects taken $r$ at a time:
$$C(n, r) \times r! = P(n, r)$$

**Solving for combinations**:
$$C(n, r) = \frac{P(n, r)}{r!} = \frac{n!}{r!(n-r)!}$$

**The combination formula**:
$$\binom{n}{r} = \frac{n!}{r!(n-r)!}$$

### 3.3 Alternative Derivation (Direct Counting)

To choose $r$ items from $n$:
- Choose 1st item: $n$ choices
- Choose 2nd item: $n-1$ choices
- ...
- Choose $r$th item: $n-r+1$ choices

This gives $n \times (n-1) \times ... \times (n-r+1)$ sequences.

But each **set** of $r$ items appears $r!$ times (in all possible orders).

So: $\binom{n}{r} = \frac{n \times (n-1) \times ... \times (n-r+1)}{r!}$

### 3.4 Properties of Combinations

**Symmetry**:
$$\binom{n}{r} = \binom{n}{n-r}$$

*Proof*: Choosing $r$ items to include = choosing $n-r$ items to exclude.

*Also*: $\frac{n!}{r!(n-r)!} = \frac{n!}{(n-r)!r!}$

**Pascal's Identity**:
$$\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$$

*Proof*: Consider one special item $x$.
- Either $x$ is in our selection: $\binom{n-1}{r-1}$ ways to choose the rest
- Or $x$ is not in our selection: $\binom{n-1}{r}$ ways to choose from the others

**Binomial Sum**:
$$\sum_{r=0}^{n} \binom{n}{r} = 2^n$$

*Proof*: Each of $n$ items is either in or out of the subset → $2^n$ total subsets.

### 3.5 Example: Poker Hands

From a 52-card deck, how many 5-card hands?
$$\binom{52}{5} = \frac{52!}{5! \times 47!} = \frac{52 \times 51 \times 50 \times 49 \times 48}{5 \times 4 \times 3 \times 2 \times 1} = 2,598,960$$

---

## Part 4: Summary Table

| Scenario | Formula | Name |
|----------|---------|------|
| Arrange all $n$ distinct objects | $n!$ | Permutation |
| Arrange $r$ from $n$ distinct (order matters) | $P(n,r) = \frac{n!}{(n-r)!}$ | Permutation |
| Arrange $n$ with repetitions $(n_1, n_2, ...)$ | $\frac{n!}{n_1! n_2! ...}$ | Multinomial |
| Choose $r$ from $n$ (order doesn't matter) | $\binom{n}{r} = \frac{n!}{r!(n-r)!}$ | Combination |
| Choose with replacement, order matters | $n^r$ | Product rule |
| Choose with replacement, order doesn't matter | $\binom{n+r-1}{r}$ | Stars and bars |

---

## Part 5: The Binomial Theorem Connection

### 5.1 Why "Binomial" Coefficient?

The combination formula $\binom{n}{r}$ is called a "binomial coefficient" because it appears in the expansion of $(a + b)^n$:

$$(a + b)^n = \sum_{r=0}^{n} \binom{n}{r} a^{n-r} b^r$$

### 5.2 Derivation

$$(a + b)^n = \underbrace{(a+b)(a+b)...(a+b)}_{n \text{ factors}}$$

When we expand, each term is formed by choosing $a$ or $b$ from each factor.

A term $a^{n-r}b^r$ means:
- Choose $a$ from $n-r$ factors
- Choose $b$ from $r$ factors

How many ways to choose which $r$ factors contribute $b$?
$$\binom{n}{r}$$

So the coefficient of $a^{n-r}b^r$ is $\binom{n}{r}$.

### 5.3 Connection to Probability

For $n$ independent Bernoulli trials with success probability $p$:

$$P(\text{exactly } k \text{ successes}) = \binom{n}{k} p^k (1-p)^{n-k}$$

This is the **binomial distribution**!

- $\binom{n}{k}$: ways to choose WHICH $k$ trials are successes
- $p^k$: probability those $k$ are successes
- $(1-p)^{n-k}$: probability the rest are failures

---

## Part 6: Practice Problems

### Basic

1. How many ways to arrange the letters in "PROBABILITY"?

2. A committee of 5 must be chosen from 12 people. How many ways?

3. How many 4-digit PINs have no repeated digits?

### Intermediate

4. From 10 men and 8 women, form a committee of 5 with exactly 2 women.

5. How many ways to distribute 10 identical balls into 4 distinct boxes?

6. In how many ways can you arrange 3 red, 2 blue, and 1 green ball in a row?

### Advanced

7. Prove: $\sum_{k=0}^{n} \binom{n}{k}^2 = \binom{2n}{n}$

8. How many paths from (0,0) to (m,n) going only right or up?

9. In a poker hand, what's the probability of getting exactly 2 pairs?

---

## Solutions to Selected Problems

**Problem 1**: "PROBABILITY" has 11 letters: P(2), R(1), O(1), B(2), A(1), I(2), L(1), T(1), Y(1)
$$\frac{11!}{2! \times 2! \times 2!} = \frac{39,916,800}{8} = 4,989,600$$

**Problem 4**: Choose 2 women from 8, then 3 men from 10:
$$\binom{8}{2} \times \binom{10}{3} = 28 \times 120 = 3,360$$

**Problem 5** (Stars and Bars): 10 identical balls, 4 boxes = choosing where to put 3 dividers among 13 positions:
$$\binom{10 + 4 - 1}{4 - 1} = \binom{13}{3} = 286$$

**Problem 8**: Need $m$ rights and $n$ ups in some order. Total steps = $m + n$.
$$\binom{m+n}{m} = \binom{m+n}{n}$$

---

## Key Takeaways

1. **Permutation vs Combination**: Does order matter?
   - Yes → Permutation (more arrangements)
   - No → Combination (divide by $r!$)

2. **The Core Relationship**: $C(n,r) \times r! = P(n,r)$

3. **Dealing with Repetition**: Divide by factorial of each repetition count

4. **Sanity Check**: Combinations $\leq$ Permutations (same or fewer, never more)

5. **Connection to Probability**: Counting is the foundation of computing probabilities!
