# Physics Tutorial 02: Carnot Cycle — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Carnot Efficiency Calculation
A Carnot engine operates between $T_H = 500K$ and $T_C = 300K$.
a) Calculate the efficiency
b) If 1000 J of heat is absorbed, how much work is produced?
c) How much heat is rejected?

### Exercise A2 🟡 — Derive Carnot Efficiency
Starting from the definition of efficiency $\eta = W/Q_H$ and the isothermal processes, derive:
$$\eta = 1 - \frac{T_C}{T_H}$$

### Exercise A3 🟡 — Four Stages of Carnot Cycle
For each stage of the Carnot cycle, describe:
a) What happens to volume, pressure, temperature
b) Heat absorbed/released
c) Work done by/on the gas

### Exercise A4 🔴 — Entropy in Carnot Cycle
Calculate the total entropy change for one complete Carnot cycle. Explain the result.

### Exercise A5 🔴 — Why Carnot is Optimal
Prove that no heat engine operating between $T_H$ and $T_C$ can exceed Carnot efficiency. (Use Clausius inequality or entropy arguments)

### Exercise A6 🔴 — Refrigerator COP
A refrigerator is a Carnot engine run in reverse.
a) Derive the Coefficient of Performance (COP)
b) Calculate COP for $T_H = 300K$, $T_C = 250K$
c) Why is COP > 1 not a violation of energy conservation?

---

## Part B: Coding

### Exercise B1 🟢 — Efficiency Calculator
```python
# TODO: Create a Carnot efficiency calculator
# 1. Input: T_H, T_C in various units (K, C, F)
# 2. Output: efficiency, work for given Q_H
# 3. Validate T_H > T_C > 0
```

### Exercise B2 🟡 — P-V Diagram
```python
# TODO: Plot Carnot cycle on P-V diagram
# 1. For given T_H, T_C, V_min, V_max
# 2. Plot all four processes (color-coded)
# 3. Calculate work as enclosed area
# 4. Show isotherms and adiabats
```

### Exercise B3 🔴 — Efficiency vs Temperature
```python
# TODO: Analyze efficiency dependence
# 1. Plot η vs T_H for fixed T_C
# 2. Plot η vs T_C for fixed T_H
# 3. Show why increasing T_H or decreasing T_C helps
# 4. Real-world constraints discussion
```

---

## Part C: Conceptual

### C1 🟡
Why can't we achieve 100% efficiency even theoretically? What would happen at $T_C = 0$?

### C2 🔴
If the Carnot cycle is the most efficient, why don't we use it in real engines?
