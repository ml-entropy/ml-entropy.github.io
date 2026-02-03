# Physics Tutorial 01: Entropy — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Microstate Counting
A box has 4 particles that can be in the left (L) or right (R) half.
a) List all possible microstates
b) How many microstates have 2 particles on each side?
c) Which macrostate (# particles on left) has the highest multiplicity?

### Exercise A2 🟢 — Boltzmann Entropy
A system has 100 microstates. Calculate its entropy in units of $k_B$.

### Exercise A3 🟡 — Derive Boltzmann Formula
Starting from additivity of entropy ($S_{total} = S_A + S_B$) and multiplicativity of microstates ($W_{total} = W_A \times W_B$), derive that $S = k \ln W$.

### Exercise A4 🟡 — Stirling's Approximation
Prove: $\ln(n!) \approx n\ln(n) - n$ for large $n$.
Use this to show that for N particles distributed as $\{n_1, n_2, ...\}$:
$$S \approx -Nk_B\sum_i p_i \ln p_i$$

### Exercise A5 🔴 — Second Law from Statistics
Explain why entropy tends to increase using probability arguments. Why is the "most likely" macrostate overwhelmingly more likely for large N?

### Exercise A6 🔴 — Maxwell's Demon
What is Maxwell's demon? Why doesn't it violate the second law? (Hint: information erasure)

---

## Part B: Coding

### Exercise B1 🟢 — Microstate Simulator
```python
# TODO: Simulate particle distribution
# 1. N particles, 2 boxes
# 2. Count microstates for each macrostate
# 3. Verify W = C(N, k) for k particles in box 1
# 4. Plot multiplicity vs macrostate
```

### Exercise B2 🟡 — Entropy Evolution
```python
# TODO: Simulate approach to equilibrium
# 1. Start with all particles on one side
# 2. Random walk: each step, random particle moves
# 3. Track entropy over time
# 4. Show it increases on average
```

### Exercise B3 🔴 — Information and Thermodynamic Entropy
```python
# TODO: Connect Shannon and Boltzmann entropy
# 1. For a probability distribution p_i
# 2. Compute Shannon entropy H = -Σ p_i log p_i
# 3. Compute Boltzmann entropy S = k_B ln W using multinomial
# 4. Show S/k_B = N * H (in bits or nats)
```

---

## Part C: Conceptual

### C1 🟡
Why can't entropy decrease spontaneously? Is it truly impossible or just extremely unlikely?

### C2 🔴
How does the second law of thermodynamics relate to the arrow of time?
