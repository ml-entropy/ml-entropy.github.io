# Physics Tutorial 02: Carnot Cycle — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Carnot Efficiency Calculation

a) **Efficiency**:
$$\eta = 1 - \frac{T_C}{T_H} = 1 - \frac{300}{500} = 1 - 0.6 = \boxed{0.4 = 40\%}$$

b) **Work produced**:
$$W = \eta \times Q_H = 0.4 \times 1000\text{ J} = \boxed{400\text{ J}}$$

c) **Heat rejected**:
$$Q_C = Q_H - W = 1000 - 400 = \boxed{600\text{ J}}$$

---

### Solution A2 — Derive Carnot Efficiency

**First Law**: $W = Q_H - Q_C$

**For isothermal processes** in ideal gas:
$$Q_H = nRT_H \ln\frac{V_2}{V_1}$$
$$Q_C = nRT_C \ln\frac{V_3}{V_4}$$

**For adiabatic processes**:
$$T_H V_2^{\gamma-1} = T_C V_3^{\gamma-1}$$
$$T_C V_4^{\gamma-1} = T_H V_1^{\gamma-1}$$

Dividing these: $\frac{V_2}{V_1} = \frac{V_3}{V_4}$

Therefore:
$$\frac{Q_C}{Q_H} = \frac{T_C \ln(V_3/V_4)}{T_H \ln(V_2/V_1)} = \frac{T_C}{T_H}$$

Efficiency:
$$\eta = \frac{W}{Q_H} = \frac{Q_H - Q_C}{Q_H} = 1 - \frac{Q_C}{Q_H} = \boxed{1 - \frac{T_C}{T_H}}$$ ∎

---

### Solution A3 — Four Stages of Carnot Cycle

| Stage | Process | V | P | T | Q | W |
|-------|---------|---|---|---|---|---|
| 1→2 | Isothermal expansion | ↑ | ↓ | $T_H$ | $+Q_H$ | $+W_1$ (by gas) |
| 2→3 | Adiabatic expansion | ↑ | ↓ | ↓ | 0 | $+W_2$ (by gas) |
| 3→4 | Isothermal compression | ↓ | ↑ | $T_C$ | $-Q_C$ | $-W_3$ (on gas) |
| 4→1 | Adiabatic compression | ↓ | ↑ | ↑ | 0 | $-W_4$ (on gas) |

**Key insights**:
- Heat only exchanged during isothermal processes
- Adiabats connect the two isotherms
- Net work = area enclosed in P-V diagram

---

### Solution A4 — Entropy in Carnot Cycle

**Entropy change per process**:

- **1→2** (isothermal at $T_H$): $\Delta S_1 = \frac{Q_H}{T_H}$
- **2→3** (adiabatic): $\Delta S_2 = 0$ (reversible, no heat)
- **3→4** (isothermal at $T_C$): $\Delta S_3 = \frac{-Q_C}{T_C}$
- **4→1** (adiabatic): $\Delta S_4 = 0$

**Total**:
$$\Delta S_{total} = \frac{Q_H}{T_H} - \frac{Q_C}{T_C}$$

Since $\frac{Q_C}{Q_H} = \frac{T_C}{T_H}$:
$$\Delta S_{total} = \frac{Q_H}{T_H} - \frac{Q_H T_C/T_H}{T_C} = \frac{Q_H}{T_H} - \frac{Q_H}{T_H} = \boxed{0}$$

**Interpretation**: The Carnot cycle is reversible. Total entropy of universe doesn't change.

---

### Solution A5 — Why Carnot is Optimal

**Proof by contradiction**:

Assume engine X has efficiency $\eta_X > \eta_{Carnot}$.

Run X forward and Carnot backward (as refrigerator):
- X absorbs $Q_H$ from hot reservoir, produces work $W_X = \eta_X Q_H$
- Carnot (reversed) uses work $W_C$ to pump heat from cold to hot

If $\eta_X > \eta_{Carnot}$, then $W_X > W_C$ for same $Q_H$.

Net effect: Extract work $(W_X - W_C) > 0$ while moving NO net heat from hot reservoir.

This violates Second Law (Kelvin-Planck statement): Can't extract work from single reservoir.

**Conclusion**: $\eta \leq \eta_{Carnot}$ for ALL engines. ∎

---

### Solution A6 — Refrigerator COP

a) **COP derivation**:

Refrigerator goal: Remove heat $Q_C$ from cold reservoir using work $W$.

$$COP = \frac{Q_C}{W} = \frac{Q_C}{Q_H - Q_C}$$

For Carnot refrigerator: $\frac{Q_C}{Q_H} = \frac{T_C}{T_H}$

$$COP = \frac{T_C}{T_H - T_C} = \boxed{\frac{T_C}{T_H - T_C}}$$

b) **Calculate**: $T_H = 300K$, $T_C = 250K$
$$COP = \frac{250}{300-250} = \frac{250}{50} = \boxed{5}$$

c) **Why COP > 1 is OK**:
COP measures heat moved per work input, NOT energy created.
You're not creating energy; you're using work to MOVE heat from cold to hot.
$Q_H = Q_C + W$ (energy conserved).

---

## Part B: Coding Solutions

### Solution B1 — Efficiency Calculator

```python
def convert_to_kelvin(temp, unit):
    """Convert temperature to Kelvin"""
    unit = unit.upper()
    if unit == 'K':
        return temp
    elif unit == 'C':
        return temp + 273.15
    elif unit == 'F':
        return (temp - 32) * 5/9 + 273.15
    else:
        raise ValueError(f"Unknown unit: {unit}")

def carnot_efficiency(T_H, T_C, unit_H='K', unit_C='K'):
    """
    Calculate Carnot efficiency.
    
    Args:
        T_H: Hot reservoir temperature
        T_C: Cold reservoir temperature
        unit_H, unit_C: Temperature units ('K', 'C', 'F')
    
    Returns:
        efficiency (0 to 1)
    """
    T_H_K = convert_to_kelvin(T_H, unit_H)
    T_C_K = convert_to_kelvin(T_C, unit_C)
    
    if T_H_K <= 0 or T_C_K <= 0:
        raise ValueError("Temperatures must be positive (in Kelvin)")
    if T_H_K <= T_C_K:
        raise ValueError("T_H must be greater than T_C")
    
    return 1 - T_C_K / T_H_K

def carnot_analysis(T_H, T_C, Q_H, unit='K'):
    """Complete Carnot cycle analysis"""
    eta = carnot_efficiency(T_H, T_C, unit, unit)
    W = eta * Q_H
    Q_C = Q_H - W
    
    print(f"=== Carnot Engine Analysis ===")
    print(f"T_H = {T_H} {unit} = {convert_to_kelvin(T_H, unit):.1f} K")
    print(f"T_C = {T_C} {unit} = {convert_to_kelvin(T_C, unit):.1f} K")
    print(f"Efficiency: η = {eta:.2%}")
    print(f"Heat absorbed (Q_H): {Q_H:.1f} J")
    print(f"Work produced (W): {W:.1f} J")
    print(f"Heat rejected (Q_C): {Q_C:.1f} J")
    
    return eta, W, Q_C

# Examples
carnot_analysis(500, 300, 1000, 'K')
print()
carnot_analysis(100, 20, 1000, 'C')
```

### Solution B2 — P-V Diagram

```python
import numpy as np
import matplotlib.pyplot as plt

def carnot_pv_diagram(T_H, T_C, V1, V2, n=1, R=8.314, gamma=1.4):
    """
    Plot Carnot cycle on P-V diagram.
    
    Args:
        T_H, T_C: Hot and cold temperatures (K)
        V1, V2: Volume range for hot isotherm
        n: Moles of gas
        R: Gas constant
        gamma: Heat capacity ratio
    """
    # Calculate volumes at all four points
    # Point 1: Start of hot isothermal expansion (V1, T_H)
    # Point 2: End of hot isothermal expansion (V2, T_H)
    # Point 3: End of adiabatic expansion (V3, T_C)
    # Point 4: End of cold isothermal compression (V4, T_C)
    
    # Use adiabatic relations: TV^(γ-1) = const
    V3 = V2 * (T_H / T_C) ** (1 / (gamma - 1))
    V4 = V1 * (T_H / T_C) ** (1 / (gamma - 1))
    
    # Generate curves
    # Isothermal: PV = nRT → P = nRT/V
    # Adiabatic: PV^γ = const
    
    V_iso_hot = np.linspace(V1, V2, 100)
    P_iso_hot = n * R * T_H / V_iso_hot
    
    V_adiabat_exp = np.linspace(V2, V3, 100)
    P_adiabat_exp = (n * R * T_H / V2) * (V2 / V_adiabat_exp) ** gamma
    
    V_iso_cold = np.linspace(V3, V4, 100)
    P_iso_cold = n * R * T_C / V_iso_cold
    
    V_adiabat_comp = np.linspace(V4, V1, 100)
    P_adiabat_comp = (n * R * T_C / V4) * (V4 / V_adiabat_comp) ** gamma
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(V_iso_hot, P_iso_hot, 'r-', linewidth=2, label=f'1→2: Isothermal ({T_H}K)')
    ax.plot(V_adiabat_exp, P_adiabat_exp, 'b-', linewidth=2, label='2→3: Adiabatic')
    ax.plot(V_iso_cold, P_iso_cold, 'c-', linewidth=2, label=f'3→4: Isothermal ({T_C}K)')
    ax.plot(V_adiabat_comp, P_adiabat_comp, 'm-', linewidth=2, label='4→1: Adiabatic')
    
    # Mark points
    points = [(V1, n*R*T_H/V1), (V2, n*R*T_H/V2), 
              (V3, n*R*T_C/V3), (V4, n*R*T_C/V4)]
    for i, (v, p) in enumerate(points):
        ax.plot(v, p, 'ko', markersize=10)
        ax.annotate(f'{i+1}', (v, p), textcoords="offset points", 
                   xytext=(10, 10), fontsize=12)
    
    ax.set_xlabel('Volume (m³)', fontsize=12)
    ax.set_ylabel('Pressure (Pa)', fontsize=12)
    ax.set_title('Carnot Cycle P-V Diagram', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate work (area enclosed)
    W_hot = n * R * T_H * np.log(V2/V1)
    W_cold = n * R * T_C * np.log(V4/V3)  # Negative (compression)
    W_net = W_hot + W_cold
    
    print(f"Net work per cycle: {W_net:.2f} J")
    print(f"Efficiency: {(1 - T_C/T_H)*100:.1f}%")
    
    plt.show()
    return fig

# Example
carnot_pv_diagram(T_H=500, T_C=300, V1=0.001, V2=0.002)
```

### Solution B3 — Efficiency vs Temperature

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: η vs T_H for fixed T_C
T_C = 300  # Fixed cold reservoir (room temperature)
T_H_range = np.linspace(T_C + 1, 2000, 100)
eta_1 = 1 - T_C / T_H_range

axes[0].plot(T_H_range, eta_1 * 100, 'b-', linewidth=2)
axes[0].axhline(50, color='r', linestyle='--', alpha=0.5, label='50% efficiency')
axes[0].set_xlabel('Hot Reservoir Temperature T_H (K)', fontsize=12)
axes[0].set_ylabel('Efficiency (%)', fontsize=12)
axes[0].set_title(f'Carnot Efficiency vs T_H (T_C = {T_C}K fixed)', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_ylim(0, 100)

# Annotate some practical temperatures
axes[0].axvline(373, color='g', linestyle=':', alpha=0.5)
axes[0].annotate('Boiling water', (373, 20), fontsize=10)
axes[0].axvline(773, color='g', linestyle=':', alpha=0.5)
axes[0].annotate('Gas turbine', (773, 60), fontsize=10)

# Plot 2: η vs T_C for fixed T_H
T_H = 600  # Fixed hot reservoir
T_C_range = np.linspace(1, T_H - 1, 100)
eta_2 = 1 - T_C_range / T_H

axes[1].plot(T_C_range, eta_2 * 100, 'r-', linewidth=2)
axes[1].set_xlabel('Cold Reservoir Temperature T_C (K)', fontsize=12)
axes[1].set_ylabel('Efficiency (%)', fontsize=12)
axes[1].set_title(f'Carnot Efficiency vs T_C (T_H = {T_H}K fixed)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 100)

# Annotate
axes[1].axvline(300, color='g', linestyle=':', alpha=0.5)
axes[1].annotate('Room temp', (300, 80), fontsize=10)
axes[1].axvline(77, color='g', linestyle=':', alpha=0.5)
axes[1].annotate('Liquid N₂', (77, 90), fontsize=10)

plt.tight_layout()
plt.show()

print("Key insights:")
print("1. Increasing T_H improves efficiency (but material limits exist)")
print("2. Decreasing T_C improves efficiency (but environment sets practical limit)")
print("3. Real engines achieve 30-50% of Carnot efficiency due to irreversibilities")
```

---

## Part C: Conceptual Solutions

### C1

**Why not 100% efficiency?**

$$\eta = 1 - \frac{T_C}{T_H}$$

For $\eta = 100\%$, we need $T_C = 0$ (absolute zero).

**Third Law of Thermodynamics**: It's impossible to reach absolute zero in finite steps.

Even at very low $T_C$:
- Maintaining $T_C$ requires work (refrigeration)
- Heat leaks from environment
- Practical materials fail

**Fundamental reason**: Heat engines must reject some heat to the cold reservoir. This is required to "reset" the working substance for the next cycle.

### C2

**Why aren't real engines Carnot engines?**

1. **Infinitely slow**: Carnot cycle is reversible → quasi-static → infinitely slow
   - Real engines need finite power output

2. **Isothermal processes are impractical**: 
   - Require infinite heat exchanger surface area
   - Real heat transfer requires temperature difference

3. **Adiabatic processes require perfect insulation**:
   - Impossible in practice

4. **Friction and irreversibilities**:
   - Real gases, turbulence, mechanical friction

**Real engines use different cycles**:
- Otto cycle (gasoline engines): ~25-30% efficiency
- Diesel cycle: ~35-40% efficiency
- Combined cycle gas turbines: ~60% efficiency

These sacrifice some theoretical efficiency for practical power output.

---

## Part D: Entropy & Information Solutions

### Solution D1 — The Information-Energy Link

**a) Energy to erase 1 TB:**

Given:
- $N = 8 \times 10^{12}$ bits (1 Terabyte)
- $T = 300 \text{ K}$
- $k_B = 1.38 \times 10^{-23} \text{ J/K}$

Using Landauer's limit per bit: $E_{bit} = k_B T \ln 2$

$$ E_{bit} = (1.38 \times 10^{-23}) \times 300 \times 0.693 \approx 2.87 \times 10^{-21} \text{ J} $$

Total energy:
$$ E_{total} = N \times E_{bit} = (8 \times 10^{12}) \times (2.87 \times 10^{-21}) \approx \boxed{2.3 \times 10^{-8} \text{ J}} $$

This seems extremely small! However, this is the *thermodynamic lower bound*. Real computers dissipate orders of magnitude more heat (Joules per second) due to electrical resistance, far above this limit.

**b) Connection to Gas Compression:**

In the Carnot cycle isothermal compression (Step 3→4), we compress the gas from $V_3$ to $V_4$ at constant temperature $T_C$.
Work done ON the gas ($W > 0$):
$$ W_{on} = - \int_{V_3}^{V_4} P dV = - nRT_C \ln\left(\frac{V_4}{V_3}\right) = nRT_C \ln\left(\frac{V_3}{V_4}\right) $$

If we halve the volume (representing 1 bit of information reduction, knowing the particle is in the "Left" half vs "Anywhere"):
$$ \frac{V_3}{V_4} = 2 $$

Then:
$$ W_{on} = n R T_C \ln 2 $$

For a single molecule ($n = 1/N_A$, $R/N_A = k_B$):
$$ W_{molecule} = k_B T_C \ln 2 $$

**Conclusion**: The work required to compress the gas (reducing the uncertainty of particle positions) is identical to the energy cost of erasing information (reducing the uncertainty of bit states). This demonstrates the deep physical link between Shannon Entropy (Information) and Thermodynamic Entropy.

### Solution D2 — Entropy Production in Irreversible Engines

**Derivation:**

1.  **Efficiency definition**:
    $$ \eta = \frac{W}{Q_H} = \frac{Q_H - Q_C}{Q_H} = 1 - \frac{Q_C}{Q_H} $$

2.  **Entropy balance**:
    The total entropy change of the universe in one cycle must be non-negative (Second Law):
    $$ \Delta S_{univ} = \Delta S_{system} + \Delta S_{surroundings} \ge 0 $$
    
    Since the engine operates in a cycle, $\Delta S_{system} = 0$.
    The surroundings consist of the Hot Reservoir (loses $Q_H$) and Cold Reservoir (gains $Q_C$).
    $$ \Delta S_{surroundings} = -\frac{Q_H}{T_H} + \frac{Q_C}{T_C} $$
    
    Let $\Delta S_{gen}$ be the generated entropy (irreversibility):
    $$ -\frac{Q_H}{T_H} + \frac{Q_C}{T_C} = \Delta S_{gen} \quad (\text{where } \Delta S_{gen} \ge 0) $$

3.  **Solve for $Q_C$**:
    $$ \frac{Q_C}{T_C} = \frac{Q_H}{T_H} + \Delta S_{gen} $$
    $$ Q_C = Q_H \frac{T_C}{T_H} + T_C \Delta S_{gen} $$

4.  **Substitute back into Efficiency**:
    $$ \eta_{irr} = 1 - \frac{Q_H \frac{T_C}{T_H} + T_C \Delta S_{gen}}{Q_H} $$
    $$ \eta_{irr} = 1 - \frac{T_C}{T_H} - \frac{T_C \Delta S_{gen}}{Q_H} $$
    $$ \eta_{irr} = \eta_{Carnot} - \frac{T_C \Delta S_{gen}}{Q_H} $$

**Result**: We have proven that any entropy generation ($\Delta S_{gen} > 0$) strictly reduces the efficiency below the Carnot limit. This term $\frac{T_C \Delta S_{gen}}{Q_H}$ represents the "Exergy destruction" or lost work potential.
