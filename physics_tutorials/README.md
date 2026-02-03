# Physics Tutorials: Thermodynamics and Statistical Mechanics

A beginner-friendly series exploring the fundamental physics concepts that underpin machine learning and information theory.

## Why Physics for ML?

The connection between physics and ML is deep:
- **Entropy** appears in both thermodynamics AND information theory (same formula!)
- **Maximum entropy principle** is used in both physics and probabilistic modeling
- **Variational methods** originated in physics (least action principle)
- **Energy-based models** in ML directly mirror statistical physics

## 📚 Tutorials

### [01. Entropy — From Physics First Principles](01_entropy_physics/)

Start here to understand entropy properly!

**What you'll learn**:
- What entropy ACTUALLY measures (microstates, not "disorder")
- Why entropy always increases (it's just probability!)
- Deriving $S = k_B \ln W$ from the additivity requirement
- The connection between Boltzmann and Shannon entropy
- Gibbs entropy for non-uniform distributions
- How temperature emerges from entropy

**Key formulas**:
- Boltzmann: $S = k_B \ln W$
- Gibbs: $S = -k_B \sum_i p_i \ln p_i$
- Temperature: $1/T = \partial S / \partial E$

### [02. The Carnot Cycle — Maximum Efficiency Proof](02_carnot_cycle/)

Why is there a fundamental limit to heat engine efficiency?

**What you'll learn**:
- What heat engines do (convert heat → work)
- The four steps of the Carnot cycle
- **Deriving** the efficiency formula $\eta = 1 - T_C/T_H$
- **Proof** that no engine can beat Carnot (two methods!)
- Why 100% efficiency is impossible
- Comparing real engines to Carnot limits

**Key insights**:
- Carnot efficiency depends ONLY on temperatures
- The proof uses entropy / Second Law
- This limit is not engineering — it's physics!

## 🔧 Setup

```bash
cd physics_tutorials
pip install numpy matplotlib scipy
jupyter notebook
```

## 🔗 Connection to ML Tutorials

| Physics Concept | ML Application |
|-----------------|----------------|
| Boltzmann entropy $S = k_B \ln W$ | Shannon entropy $H = -\sum p \log p$ |
| Gibbs distribution $p \propto e^{-E/kT}$ | Softmax $p_i \propto e^{z_i}$ |
| Free energy minimization | Variational inference (ELBO) |
| Maximum entropy principle | Prior selection, regularization |
| Partition function | Normalizing constant |

## 📖 Suggested Reading Order

1. **Physics 01**: Entropy (before ML entropy tutorials)
2. **ML 00**: Probability foundations
3. **ML 01**: Entropy in information theory
4. **Physics 02**: Carnot cycle (optional, but illuminating)
5. Continue with ML tutorials...

---

*"The law that entropy always increases holds, I think, the supreme position among the laws of Nature."* — Sir Arthur Eddington
