# Tutorial 07: Combinatorics — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Counting Principle
A password has 3 lowercase letters followed by 2 digits.
a) How many possible passwords are there?
b) How many if no character can repeat?

### Exercise A2 🟢 — Derive Permutation Formula
Derive: The number of ways to arrange $r$ objects from $n$ distinct objects is:
$$P(n, r) = \frac{n!}{(n-r)!}$$

### Exercise A3 🟡 — Derive Combination Formula
Starting from permutations, derive:
$$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

### Exercise A4 🟡 — Pascal's Triangle Identity
Prove algebraically: $\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$

### Exercise A5 🔴 — Binomial Theorem
Prove: $(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}$

### Exercise A6 🔴 — Stars and Bars
How many ways can you distribute $n$ identical balls into $k$ distinct boxes? Derive the formula.

---

## Part B: Coding

### Exercise B1 🟢 — Implement Factorial and Combinations
```python
# TODO: Implement without using math.factorial
# 1. factorial(n)
# 2. permutations(n, r)
# 3. combinations(n, r)
# 4. Verify with known values
```

### Exercise B2 🟡 — Generate All Permutations
```python
# TODO: Generate all permutations of a list
# 1. Implement recursively (without itertools)
# 2. Verify count matches n!
# 3. Generate permutations of [1, 2, 3, 4]
```

### Exercise B3 🟡 — Pascal's Triangle
```python
# TODO: Generate Pascal's triangle
# 1. Generate first n rows
# 2. Verify each entry equals C(row, col)
# 3. Visualize the triangle
```

### Exercise B4 🔴 — Counting in ML
```python
# TODO: Application to ML
# 1. How many possible decision trees of depth d with b binary features?
# 2. How many ways to split n samples into train/val/test (60/20/20)?
# 3. Number of possible neural network architectures (given constraints)
```

---

## Part C: Conceptual

### C1 🟡
Why does $\binom{n}{r} = \binom{n}{n-r}$? Give both algebraic and intuitive explanations.

### C2 🔴
How does combinatorics relate to entropy? (Hint: Think about microstates)
