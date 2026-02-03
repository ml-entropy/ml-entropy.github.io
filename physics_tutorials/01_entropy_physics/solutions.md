# Physics Tutorial 01: Entropy — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Microstate Counting

a) **All microstates** for 4 particles (L/R):
```
LLLL, LLLR, LLRL, LLRR, LRLL, LRLR, LRRL, LRRR,
RLLL, RLLR, RLRL, RLRR, RRLL, RRLR, RRRL, RRRR
```
Total: $2^4 = 16$ microstates

b) **Microstates with 2L and 2R**: 
LLRR, LRLR, LRRL, RLLR, RLRL, RRLL → $\binom{4}{2} = 6$

c) **Multiplicity by macrostate**:
- 4L, 0R: $\binom{4}{4} = 1$
- 3L, 1R: $\binom{4}{3} = 4$
- 2L, 2R: $\binom{4}{2} = 6$ ← **Maximum**
- 1L, 3R: $\binom{4}{1} = 4$
- 0L, 4R: $\binom{4}{0} = 1$

---

### Solution A2 — Boltzmann Entropy

$$S = k_B \ln W = k_B \ln 100 \approx 4.6 \, k_B$$

In SI units: $S = 1.38 \times 10^{-23} \times 4.6 \approx 6.4 \times 10^{-23}$ J/K

---

### Solution A3 — Derive Boltzmann Formula

**Given**:
- Entropy is additive: $S_{total} = S_A + S_B$
- Microstates multiply: $W_{total} = W_A \cdot W_B$

**Derivation**:
Let $S = f(W)$ for some function $f$.

$$f(W_A \cdot W_B) = f(W_A) + f(W_B)$$

The ONLY function satisfying $f(xy) = f(x) + f(y)$ is the logarithm (proven by Cauchy).

$$\boxed{S = k \ln W}$$

The constant $k = k_B$ is determined by matching to classical thermodynamics. ∎

---

### Solution A4 — Stirling's Approximation

**Prove**: $\ln(n!) \approx n\ln(n) - n$

$$\ln(n!) = \ln(1 \cdot 2 \cdot 3 \cdots n) = \sum_{k=1}^{n} \ln k \approx \int_1^n \ln x \, dx$$

$$= [x\ln x - x]_1^n = n\ln n - n - (0 - 1) = n\ln n - n + 1 \approx n\ln n - n$$ ∎

**Application**: For N particles in states with $n_i$ particles each:
$$W = \frac{N!}{n_1! n_2! \cdots}$$

$$\ln W = \ln N! - \sum_i \ln n_i!$$

Using Stirling:
$$\ln W \approx N\ln N - N - \sum_i (n_i \ln n_i - n_i)$$
$$= N\ln N - \sum_i n_i \ln n_i$$
$$= -N \sum_i \frac{n_i}{N} \ln \frac{n_i}{N} = -N \sum_i p_i \ln p_i$$

Therefore:
$$\boxed{S = k_B \ln W \approx -N k_B \sum_i p_i \ln p_i}$$

---

### Solution A5 — Second Law from Statistics

**Key insight**: Higher entropy states have MORE microstates.

For N particles in two boxes, the macrostate with ~N/2 on each side has multiplicity:
$$W_{max} = \binom{N}{N/2} \approx \frac{2^N}{\sqrt{\pi N/2}}$$

The "all on one side" state has $W = 1$.

**Ratio**:
$$\frac{W_{max}}{W_{all-left}} \approx 2^N$$

For $N = 100$: ratio ≈ $10^{30}$

The system is **overwhelmingly more likely** to be found near equilibrium, not because low-entropy states are forbidden, but because high-entropy states are astronomically more numerous.

**This IS the second law**: Entropy increases because systems randomly explore microstates, and almost all microstates correspond to high-entropy macrostates.

---

### Solution A6 — Maxwell's Demon

**The demon**: Imaginary being that sorts fast/slow molecules, creating temperature difference without work.

**Resolution (Landauer's Principle)**: 
The demon must MEASURE particle velocities → stores information.
Eventually, demon's memory fills up.
ERASING memory requires minimum energy: $k_B T \ln 2$ per bit.

Total energy cost of erasure ≥ work that could be extracted.

**Conclusion**: Information has thermodynamic cost. The demon doesn't violate the second law because information processing is physical.

---

## Part B: Coding Solutions

### Solution B1 — Microstate Simulator

```python
import numpy as np
import matplotlib.pyplot as plt
from math import comb, factorial

def count_microstates(N):
    """Count microstates for each macrostate (k particles in box 1)"""
    multiplicities = [comb(N, k) for k in range(N+1)]
    return multiplicities

# Simulate for different N
for N in [4, 10, 100]:
    W = count_microstates(N)
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(N+1), W)
    plt.xlabel('k (particles in box 1)')
    plt.ylabel('Multiplicity W(k)')
    plt.title(f'N = {N} particles')
    plt.show()
    
    # Verify sum = 2^N
    print(f"N={N}: Total microstates = {sum(W)} = 2^{N} = {2**N}")
    print(f"Max multiplicity at k = {np.argmax(W)}")
    print()
```

### Solution B2 — Entropy Evolution

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_entropy_evolution(N, n_steps):
    """
    Simulate random particle movement between two boxes.
    Start with all N particles on left.
    """
    left = N  # Particles on left
    history = [left]
    
    for _ in range(n_steps):
        # Random particle moves
        if np.random.random() < left / N:
            left -= 1  # Particle moves right
        else:
            left += 1  # Particle moves left
        left = max(0, min(N, left))  # Keep in bounds
        history.append(left)
    
    return np.array(history)

def entropy(left, N):
    """Entropy for k particles on left out of N"""
    if left == 0 or left == N:
        return 0
    p = left / N
    return -N * (p * np.log(p) + (1-p) * np.log(1-p))

# Simulate
N = 100
n_steps = 10000
history = simulate_entropy_evolution(N, n_steps)
S_history = [entropy(left, N) for left in history]

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(history)
axes[0].axhline(N/2, color='r', linestyle='--', label='Equilibrium')
axes[0].set_xlabel('Time step')
axes[0].set_ylabel('Particles on left')
axes[0].set_title('Particle Distribution Over Time')
axes[0].legend()

axes[1].plot(S_history)
axes[1].axhline(entropy(N//2, N), color='r', linestyle='--', label='Max entropy')
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Entropy (in units of k_B)')
axes[1].set_title('Entropy Evolution')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Initial entropy: {S_history[0]:.2f} (all particles on one side)")
print(f"Final entropy: {S_history[-1]:.2f}")
print(f"Maximum entropy: {entropy(N//2, N):.2f}")
```

### Solution B3 — Information and Thermodynamic Entropy

```python
import numpy as np
from math import factorial, log

def shannon_entropy(p):
    """H = -Σ p_i log p_i (in nats)"""
    p = np.array(p)
    p = p[p > 0]  # Avoid log(0)
    return -np.sum(p * np.log(p))

def boltzmann_entropy(n, N):
    """S = k_B ln W where W = N! / (n_1! n_2! ...)"""
    # Using log to avoid overflow
    log_W = log(factorial(N))
    for ni in n:
        log_W -= log(factorial(ni))
    return log_W  # In units of k_B

# Example: N particles distributed among 3 states
N = 100
n = [30, 50, 20]  # n_i particles in state i
p = [ni/N for ni in n]  # Probability distribution

H_shannon = shannon_entropy(p)
S_boltzmann = boltzmann_entropy(n, N)

print(f"Particle counts: {n}")
print(f"Probabilities: {p}")
print()
print(f"Shannon entropy H = {H_shannon:.4f} nats")
print(f"Boltzmann S/k_B = ln W = {S_boltzmann:.4f}")
print(f"N × H = {N * H_shannon:.4f}")
print()
print(f"Relation: S/k_B ≈ N × H? {abs(S_boltzmann - N*H_shannon) < 1}")
print(f"(Difference due to Stirling approximation)")

# As N increases, the approximation improves
print("\n--- Convergence with N ---")
for N in [100, 1000, 10000]:
    n = [N//3, N//3, N - 2*(N//3)]
    p = [ni/N for ni in n]
    H = shannon_entropy(p)
    S = boltzmann_entropy(n, N)
    print(f"N={N:5d}: S/k_B = {S:.2f}, N×H = {N*H:.2f}, ratio = {S/(N*H):.4f}")
```

---

## Part C: Conceptual Solutions

### C1

**Can entropy decrease?** Yes, but it's **extremely unlikely**.

It's not forbidden by any fundamental law. The second law is statistical:
- For small systems (few particles): fluctuations can decrease entropy temporarily
- For large systems: probability of significant decrease is astronomically small

Example: For 1 mole of gas (~$10^{23}$ particles), probability of all going to one side is $\sim 2^{-10^{23}}$ — effectively zero but not identically zero.

**Summary**: Second law is probabilistic certainty, not logical necessity.

### C2

**Second Law and Arrow of Time**:

Physical laws (Newton's, Maxwell's, Schrödinger's) are time-reversible. Yet we observe:
- Eggs break but don't unbreak
- Coffee cools but doesn't spontaneously heat
- We remember the past, not the future

**Entropy provides the arrow**: The past had lower entropy (special initial conditions of the Big Bang). Time flows in the direction of increasing entropy.

This is the **Past Hypothesis**: The universe started in a very low-entropy state. The second law, combined with this boundary condition, explains why time has a direction.

**Deep question**: Why did the universe start with low entropy? This remains one of the deepest puzzles in physics.
