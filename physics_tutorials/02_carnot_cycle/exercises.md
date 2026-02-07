# Physics Tutorial 02: Carnot Cycle — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Carnot Efficiency Calculation
A Carnot engine operates between a hot reservoir at $T_H = 500 \text{ K}$ and a cold reservoir at $T_C = 300 \text{ K}$.
a) Calculate the theoretical maximum efficiency $\eta$.
b) If the engine absorbs $Q_H = 1000 \text{ J}$ of heat from the hot reservoir, how much work $W$ does it produce?
c) How much heat $Q_C$ is rejected to the cold reservoir?

### Exercise A2 🟡 — Derive Carnot Efficiency
A Carnot cycle consists of two isothermal processes and two adiabatic processes.
Starting from the definition of efficiency $\eta = \frac{W}{Q_H} = 1 - \frac{Q_C}{Q_H}$, derive the Carnot efficiency formula:
$$\eta = 1 - \frac{T_C}{T_H}$$
*Hint: Use the fact that for an ideal gas, $\Delta U = 0$ in isothermal processes, and use the relation between Volume and Temperature in adiabatic processes.*

### Exercise A3 🟡 — Four Stages of Carnot Cycle
For each of the four stages of the Carnot cycle (1-2, 2-3, 3-4, 4-1), describe:
a) The type of process (Isothermal/Adiabatic, Expansion/Compression).
b) What happens to Volume ($V$), Pressure ($P$), and Temperature ($T$).
c) Whether heat is absorbed ($Q>0$), released ($Q<0$), or zero ($Q=0$).
d) Whether work is done BY the gas ($W>0$) or ON the gas ($W<0$).

### Exercise A4 🔴 — Entropy in Carnot Cycle
a) Calculate the entropy change of the working substance ($\Delta S_{gas}$) for each of the four steps of the cycle.
b) What is the total entropy change of the gas in one full cycle?
c) Calculate the entropy change of the *Reservoirs* (Hot and Cold).
d) Verify that for a reversible Carnot cycle, the total entropy change of the universe (System + Surroundings) is zero.

### Exercise A5 🔴 — Why Carnot is Optimal
Prove that no heat engine operating between two given heat reservoirs at temperatures $T_H$ and $T_C$ can be more efficient than a Carnot engine operating between the same reservoirs.
*Hint: Assume a "Super-Carnot" engine exists with higher efficiency, and use it to drive a Carnot refrigerator. Show that this violates the Second Law (Clausius statement).*

### Exercise A6 🔴 — Refrigerator COP
A refrigerator is essentially a Carnot engine running in reverse (Work is put in to move heat from Cold to Hot).
a) Define the **Coefficient of Performance (COP)** for a refrigerator.
b) Derive the formula for maximum COP in terms of $T_H$ and $T_C$.
c) Calculate the COP for a kitchen fridge with $T_{inside} = 277 \text{ K}$ ($4^\circ$C) and $T_{room} = 300 \text{ K}$ ($27^\circ$C).

---

## Part B: Coding & Simulation

### Exercise B1 🟢 — Efficiency Calculator
Write a Python function `carnot_efficiency(t_hot, t_cold, unit='K')` that:
1. Accepts temperatures in Kelvin, Celsius, or Fahrenheit.
2. Converts them to Kelvin.
3. Checks validity ($T_H > T_C > 0$).
4. Returns the efficiency (0-1).

### Exercise B2 🟡 — P-V Diagram
Write a Python script to visualize the Carnot Cycle on a Pressure-Volume (P-V) diagram.
1. Define cycle parameters ($n=1$ mol, gas constant $R$, $\gamma=1.4$).
2. Calculate the pressure and volume at all 4 vertices of the cycle.
3. Plot the isothermal curves ($PV = \text{const}$) and adiabatic curves ($PV^\gamma = \text{const}$) connecting these points.
4. Shade the area representing work done.

### Exercise B3 🟡 — Efficiency Limits
Create a visualization investigating how efficiency changes with temperature limits.
1. Plot 1: Efficiency vs. $T_H$ (fixing $T_C = 300 \text{ K}$).
2. Plot 2: Efficiency vs. $T_C$ (fixing $T_H = 600 \text{ K}$).
3. On the graphs, mark the efficiency of common real-world engines (e.g., Car Engine ~25%, Steam Turbine ~40%).

---

## Part C: Conceptual Deep Dive

### Exercise C1 🟡 — The Zero Kelvin Limit
Why can't we achieve 100% efficiency?
Based on the formula $\eta = 1 - T_C/T_H$, we would need $T_C = 0 \text{ K}$. Use the **Third Law of Thermodynamics** and practical engineering reasoning to explain why a heat sink at absolute zero is unattainable and sustainable 100% efficiency is impossible.

### Exercise C2 🟡 — Real vs. Ideal
Why don't we drive "Carnot Cars"?
List three physical or engineering reasons why building a practical engine that operates on the exact Carnot cycle is difficult or impossible.

---

## Part D: Entropy & Information (Advanced)

### Exercise D1 🔴 — The Information-Energy Link
*This exercise connects Thermodynamics to Information Theory.*

In computing, resetting a bit of memory (changing a 0 or 1 to a fixed state, say 0) is logically irreversible. This is analogous to the compression step in a heat engine where we reduce the volume (phase space) of the gas.

**Landauer's Principle** states that erasing 1 bit of information produces at least $k_B T \ln 2$ Joules of heat, where $k_B$ is Boltzmann's constant.

a) Calculate the minimum energy required to erase 1 Terabyte ($8 \times 10^{12}$ bits) of data at room temperature ($300 \text{ K}$).
b) Compare this to the work done during Isothermal Compression in a Carnot cycle:
$$W_{comp} = nRT_C \ln(V_4/V_3)$$
If we view the gas particles as storing "information" about their position, show that the work relates to the reduction in possible states (Volume).

### Exercise D2 🔴 — Entropy Production in Irreversible Engines
A real engine is **irreversible**. It has friction, heat leaks, and finite temperature differences.
Consequently, it produces entropy: $\Delta S_{univ} > 0$.

Show that the efficiency of an irreversible engine is:
$$\eta_{irr} = \eta_{Carnot} - \frac{T_C \Delta S_{gen}}{Q_H}$$
where $\Delta S_{gen}$ is the generated entropy per cycle.
*Interpretation: Every bit of entropy you generate subtracts from your efficiency, weighted by the cold sink temperature.*
