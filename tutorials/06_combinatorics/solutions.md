# Tutorial 07: Combinatorics — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Counting Principle

a) **With repetition allowed**:
- 3 letters: $26^3$ choices
- 2 digits: $10^2$ choices
- Total: $26^3 \times 10^2 = 17,576 \times 100 = \boxed{1,757,600}$

b) **Without repetition**:
- 3 letters: $26 \times 25 \times 24 = 15,600$
- 2 digits: $10 \times 9 = 90$
- Total: $15,600 \times 90 = \boxed{1,404,000}$

---

### Solution A2 — Permutation Formula Derivation

**Problem**: Arrange $r$ objects from $n$ distinct objects where order matters.

**Derivation**:
- First position: $n$ choices
- Second position: $n-1$ choices (one used)
- Third position: $n-2$ choices
- ...
- $r$-th position: $n-r+1$ choices

Total: $n \times (n-1) \times (n-2) \times ... \times (n-r+1)$

This equals:
$$\frac{n \times (n-1) \times ... \times (n-r+1) \times (n-r)!}{(n-r)!} = \boxed{\frac{n!}{(n-r)!}}$$ ∎

---

### Solution A3 — Combination Formula Derivation

**Key insight**: Permutations count ordered arrangements, but combinations ignore order.

Each combination of $r$ objects can be arranged in $r!$ ways.

Therefore:
$$P(n, r) = C(n, r) \times r!$$

Solving:
$$C(n, r) = \frac{P(n, r)}{r!} = \frac{n!}{(n-r)!} \cdot \frac{1}{r!} = \boxed{\frac{n!}{r!(n-r)!}}$$ ∎

---

### Solution A4 — Pascal's Identity

**Prove**: $\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$

**Algebraic proof**:
$$\binom{n-1}{r-1} + \binom{n-1}{r} = \frac{(n-1)!}{(r-1)!(n-r)!} + \frac{(n-1)!}{r!(n-r-1)!}$$

Factor out $(n-1)!$:
$$= (n-1)!\left[\frac{1}{(r-1)!(n-r)!} + \frac{1}{r!(n-r-1)!}\right]$$

$$= (n-1)!\left[\frac{r}{r!(n-r)!} + \frac{n-r}{r!(n-r)!}\right]$$

$$= (n-1)!\left[\frac{r + n - r}{r!(n-r)!}\right] = \frac{n!}{r!(n-r)!} = \binom{n}{r}$$ ∎

**Combinatorial proof**: To choose $r$ items from $n$, consider item 1:
- Include it: choose remaining $r-1$ from other $n-1$ → $\binom{n-1}{r-1}$
- Exclude it: choose all $r$ from other $n-1$ → $\binom{n-1}{r}$

---

### Solution A5 — Binomial Theorem

**Prove**: $(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}$

**Proof by combinatorial argument**:

$(x + y)^n = (x+y)(x+y)...(x+y)$ ($n$ factors)

When expanding, we pick either $x$ or $y$ from each factor.

The term $x^k y^{n-k}$ appears when we pick $x$ from exactly $k$ factors.

Number of ways to choose which $k$ factors give $x$: $\binom{n}{k}$

Therefore, coefficient of $x^k y^{n-k}$ is $\binom{n}{k}$. ∎

---

### Solution A6 — Stars and Bars

**Problem**: Distribute $n$ identical balls into $k$ distinct boxes.

**Visualization**: Represent as $n$ stars and $k-1$ bars (dividers).

Example: 5 balls in 3 boxes → $\star\star|\star\star\star|$ means (2, 3, 0)

**Count**: We have $n + k - 1$ positions total. Choose $k-1$ positions for bars.

$$\boxed{\binom{n + k - 1}{k - 1} = \binom{n + k - 1}{n}}$$ ∎

---

## Part B: Coding Solutions

### Solution B1 — Factorial and Combinations

```python
def factorial(n):
    """Calculate n! iteratively"""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def permutations(n, r):
    """P(n, r) = n! / (n-r)!"""
    if r > n:
        return 0
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result

def combinations(n, r):
    """C(n, r) = n! / (r! * (n-r)!)"""
    if r > n:
        return 0
    # Use smaller of r and n-r for efficiency
    r = min(r, n - r)
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    return result

# Verify
print(f"5! = {factorial(5)}")  # 120
print(f"P(5, 3) = {permutations(5, 3)}")  # 60
print(f"C(5, 3) = {combinations(5, 3)}")  # 10
print(f"C(10, 4) = {combinations(10, 4)}")  # 210

# Verify Pascal's identity
n, r = 10, 4
assert combinations(n, r) == combinations(n-1, r-1) + combinations(n-1, r)
print("Pascal's identity verified!")
```

### Solution B2 — Generate All Permutations

```python
def generate_permutations(lst):
    """Generate all permutations recursively"""
    if len(lst) <= 1:
        return [lst.copy()]
    
    result = []
    for i in range(len(lst)):
        # Fix element at position i
        current = lst[i]
        # Remaining elements
        remaining = lst[:i] + lst[i+1:]
        # Recursively permute remaining
        for perm in generate_permutations(remaining):
            result.append([current] + perm)
    
    return result

# Test
lst = [1, 2, 3, 4]
perms = generate_permutations(lst)

print(f"Permutations of {lst}:")
print(f"Count: {len(perms)} (expected: {factorial(len(lst))})")
print(f"First 5: {perms[:5]}")
print(f"Last 5: {perms[-5:]}")

# Verify count
assert len(perms) == factorial(len(lst))
print("Count verified!")
```

### Solution B3 — Pascal's Triangle

```python
import numpy as np
import matplotlib.pyplot as plt

def pascals_triangle(n_rows):
    """Generate first n rows of Pascal's triangle"""
    triangle = [[1]]
    
    for i in range(1, n_rows):
        prev_row = triangle[-1]
        new_row = [1]  # Start with 1
        
        for j in range(len(prev_row) - 1):
            new_row.append(prev_row[j] + prev_row[j+1])
        
        new_row.append(1)  # End with 1
        triangle.append(new_row)
    
    return triangle

# Generate and verify
triangle = pascals_triangle(10)

print("Pascal's Triangle (first 10 rows):")
for i, row in enumerate(triangle):
    print(f"Row {i}: {' '.join(map(str, row))}")

# Verify against combinations
print("\nVerification:")
for n in range(10):
    for r in range(n + 1):
        assert triangle[n][r] == combinations(n, r), f"Mismatch at ({n}, {r})"
print("All values match C(n, r)!")

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
for i, row in enumerate(triangle):
    for j, val in enumerate(row):
        x = j - i/2  # Center each row
        ax.text(x, -i, str(val), ha='center', va='center', fontsize=10)

ax.set_xlim(-5, 5)
ax.set_ylim(-10, 1)
ax.axis('off')
ax.set_title("Pascal's Triangle")
plt.show()
```

### Solution B4 — Counting in ML

```python
import numpy as np
from math import factorial, comb

# 1. Decision trees of depth d with b binary features
def count_decision_trees(d, b):
    """
    At each internal node: choose 1 of b features
    Number of internal nodes in complete binary tree of depth d: 2^d - 1
    But features can repeat, so it's b^(2^d - 1) arrangements
    """
    internal_nodes = 2**d - 1
    return b ** internal_nodes

print("=== Decision Tree Counting ===")
for d in [1, 2, 3]:
    for b in [5, 10]:
        count = count_decision_trees(d, b)
        print(f"Depth {d}, {b} features: {count:,} possible trees")

# 2. Train/val/test split
def count_splits(n, train_frac, val_frac, test_frac):
    """
    Multinomial coefficient: n! / (n_train! * n_val! * n_test!)
    """
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    
    return factorial(n) // (factorial(n_train) * factorial(n_val) * factorial(n_test))

print("\n=== Train/Val/Test Splits ===")
for n in [10, 20, 100]:
    count = count_splits(n, 0.6, 0.2, 0.2)
    print(f"n={n}: {count:,} possible 60/20/20 splits")

# 3. Neural network architectures
def count_architectures(input_dim, output_dim, max_hidden_layers, max_neurons_per_layer):
    """
    Simplified: each layer can have 1 to max_neurons_per_layer neurons
    Number of layers: 0 to max_hidden_layers
    """
    total = 0
    for n_layers in range(max_hidden_layers + 1):
        # Each layer independently chooses neuron count
        total += max_neurons_per_layer ** n_layers
    return total

print("\n=== Neural Network Architectures ===")
# Simplified: up to 3 hidden layers, up to 100 neurons each
count = count_architectures(10, 2, 3, 100)
print(f"Up to 3 hidden layers, up to 100 neurons each: {count:,} architectures")
print("(This ignores activations, skip connections, etc.)")
```

---

## Part C: Conceptual Solutions

### C1

**Algebraic**:
$$\binom{n}{r} = \frac{n!}{r!(n-r)!} = \frac{n!}{(n-r)!r!} = \binom{n}{n-r}$$

**Intuitive**: Choosing $r$ items to **include** is equivalent to choosing $n-r$ items to **exclude**. Both define the same subset.

Example: From 5 people, choosing 2 to be on a team = choosing 3 to NOT be on the team.

### C2

**Combinatorics and Entropy Connection**:

Entropy measures the number of **microstates** (arrangements) compatible with a macrostate.

Boltzmann entropy: $S = k_B \ln W$

where $W$ = number of microstates.

For $N$ particles distributed among energy levels with $n_1, n_2, ...$ particles in each:
$$W = \frac{N!}{n_1! n_2! ...}$$

This is the multinomial coefficient! The $\ln$ of this gives:
$$\ln W \approx -N \sum_i p_i \ln p_i = N \cdot H(p)$$

where $p_i = n_i/N$ and $H(p)$ is Shannon entropy.

**Key insight**: Entropy IS counting (in log scale). High entropy = many microstates = high combinatorial multiplicity.
