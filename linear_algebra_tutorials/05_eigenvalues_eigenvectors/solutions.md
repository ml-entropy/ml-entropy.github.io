# Solutions: Eigenvalues and Eigenvectors

## Part A: Theory Solutions

### Solution A1: Basic Eigenvalue Calculation

**a)** $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$ (upper triangular)

Eigenvalues = diagonal entries: **λ₁ = 3, λ₂ = 2**

For λ = 3: $(A - 3I)\mathbf{v} = \begin{bmatrix} 0 & 1 \\ 0 & -1 \end{bmatrix}\mathbf{v} = 0$
→ v₂ = 0, v₁ free → **v₁ = [1, 0]ᵀ**

For λ = 2: $(A - 2I)\mathbf{v} = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}\mathbf{v} = 0$
→ v₁ + v₂ = 0 → **v₂ = [1, -1]ᵀ**

**b)** $B = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$

det(B - λI) = (4-λ)(1-λ) - 4 = λ² - 5λ = λ(λ - 5)
**λ₁ = 5, λ₂ = 0**

For λ = 5: $(B - 5I)\mathbf{v} = \begin{bmatrix} -1 & 2 \\ 2 & -4 \end{bmatrix}\mathbf{v} = 0$
→ v₁ = 2v₂ → **v₁ = [2, 1]ᵀ**

For λ = 0: $B\mathbf{v} = 0$ → Row reduce to get v₁ = -2v₂ → **v₂ = [2, -1]ᵀ** (or [-1, 2]ᵀ)

Wait, let's redo: From row [4, 2], we get 4v₁ + 2v₂ = 0 → v₁ = -v₂/2
**v₂ = [1, -2]ᵀ**

**c)** $C = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$

det(C - λI) = λ² + 1 = 0
**λ = ±i** (complex eigenvalues!)

For λ = i: $(C - iI)\mathbf{v} = \begin{bmatrix} -i & -1 \\ 1 & -i \end{bmatrix}\mathbf{v} = 0$
From first row: -iv₁ - v₂ = 0 → v₂ = -iv₁
**v = [1, -i]ᵀ**

For λ = -i: **v = [1, i]ᵀ**

---

### Solution A2: 3×3 Eigenvalues

For triangular matrix, eigenvalues = diagonal entries.

**λ₁ = λ₂ = λ₃ = 2** (eigenvalue 2 with algebraic multiplicity 3)

---

### Solution A3: Eigenvalue Properties

Given λ₁ = 2, λ₂ = -1, λ₃ = 3:

**a)** trace(A) = 2 + (-1) + 3 = **4**

**b)** det(A) = 2 × (-1) × 3 = **-6**

**c)** Eigenvalues of A² = (λᵢ)² = **4, 1, 9**

**d)** Eigenvalues of A⁻¹ = 1/λᵢ = **1/2, -1, 1/3**

**e)** Eigenvalues of A - 2I = λᵢ - 2 = **0, -3, 1**

---

### Solution A4: Symmetric Matrix

$S = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$

**a) Eigenvalues:**
det(S - λI) = (5-λ)(2-λ) - 4 = λ² - 7λ + 6 = (λ-6)(λ-1)
**λ₁ = 6, λ₂ = 1**

**b) Eigenvectors:**

For λ = 6: $(S - 6I)\mathbf{v} = \begin{bmatrix} -1 & 2 \\ 2 & -4 \end{bmatrix}\mathbf{v} = 0$
→ v₁ = 2v₂ → **v₁ = [2, 1]ᵀ/√5**

For λ = 1: $(S - I)\mathbf{v} = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}\mathbf{v} = 0$
→ 2v₁ + v₂ = 0 → **v₂ = [1, -2]ᵀ/√5**

**c) Verify orthogonality:**
v₁ · v₂ = 2(1) + 1(-2) = 2 - 2 = **0 ✓**

**d) Spectral decomposition:**
$$Q = \frac{1}{\sqrt{5}}\begin{bmatrix} 2 & 1 \\ 1 & -2 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 6 & 0 \\ 0 & 1 \end{bmatrix}$$

$$S = Q\Lambda Q^T$$

---

### Solution A5: Diagonalization

$A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$

**a) Eigenvalues/eigenvectors:**

det(A - λI) = (1-λ)² - 4 = λ² - 2λ - 3 = (λ-3)(λ+1)
**λ₁ = 3, λ₂ = -1**

For λ = 3: v₁ = [1, 1]ᵀ
For λ = -1: v₂ = [1, -1]ᵀ

**b) Diagonalization:**
$$P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad D = \begin{bmatrix} 3 & 0 \\ 0 & -1 \end{bmatrix}$$

$$P^{-1} = \frac{1}{-2}\begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

**c) Computing A⁵:**
$$A^5 = PD^5P^{-1} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 243 & 0 \\ 0 & -1 \end{bmatrix} \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

$$= \frac{1}{2}\begin{bmatrix} 243 & -1 \\ 243 & 1 \end{bmatrix}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 242 & 244 \\ 244 & 242 \end{bmatrix} = \begin{bmatrix} 121 & 122 \\ 122 & 121 \end{bmatrix}$$

---

### Solution A6: Complex Eigenvalues

$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$

**a) Characteristic polynomial:**
det(R - λI) = λ² + 1

**b) Complex eigenvalues:**
λ² = -1 → **λ = i, λ = -i**

**c) Verify conjugates:** i and -i are indeed complex conjugates ✓

**d) Geometric interpretation:** 
This is rotation by 90° counterclockwise. Complex eigenvalues indicate rotation.
e^(iπ/2) = i corresponds to 90° rotation.

---

### Solution A7: Defective Matrix

$A = \begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}$

**a) Eigenvalues:**
det(A - λI) = (3-λ)² = 0
λ = 3 with **algebraic multiplicity 2**

**b) Eigenvectors:**
$(A - 3I)\mathbf{v} = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}\mathbf{v} = 0$

From 0v₁ + v₂ = 0: v₂ = 0

Only eigenvector direction: **v = [1, 0]ᵀ**
**Geometric multiplicity = 1**

**c) Not diagonalizable:**
We need 2 independent eigenvectors for diagonalization, but only have 1.
This matrix is "defective" - it cannot be diagonalized.

---

### Solution A8: Power Iteration Analysis

$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$

**a) Eigenvalues/eigenvectors:**

det(A - λI) = (3-λ)² - 1 = λ² - 6λ + 8 = (λ-4)(λ-2)
λ₁ = 4, λ₂ = 2

For λ = 4: v₁ = [1, 1]ᵀ (normalized: [1/√2, 1/√2]ᵀ)
For λ = 2: v₂ = [1, -1]ᵀ (normalized: [1/√2, -1/√2]ᵀ)

**b) Express x₀ = [1, 0]ᵀ in eigenbasis:**

[1, 0]ᵀ = c₁[1, 1]ᵀ + c₂[1, -1]ᵀ

From 1 = c₁ + c₂ and 0 = c₁ - c₂:
c₁ = 1/2, c₂ = 1/2

**x₀ = (1/2)v₁ + (1/2)v₂**

**c) Computing powers:**

Ax₀ = (1/2)(4v₁) + (1/2)(2v₂) = 2v₁ + v₂ = [3, 1]ᵀ

A²x₀ = (1/2)(16v₁) + (1/2)(4v₂) = 8v₁ + 2v₂ = [10, 6]ᵀ

A³x₀ = (1/2)(64v₁) + (1/2)(8v₂) = 32v₁ + 4v₂ = [36, 28]ᵀ

**d) Convergence:**

Direction of Aᵏx₀ normalized:
- k=1: [3, 1]/√10 ≈ [0.949, 0.316]
- k=2: [10, 6]/√136 ≈ [0.857, 0.514]
- k=3: [36, 28]/√2080 ≈ [0.790, 0.614]

Converging toward v₁ = [0.707, 0.707] ✓

Ratio λ₂/λ₁ = 2/4 = 0.5, so convergence is relatively fast.

---

### Solution A9: Eigenvalues and Definiteness

**a) Positive definite ⟺ all λ > 0:**

**(⇒)** If A is positive definite, then for any eigenvector v: vᵀAv > 0.
But vᵀAv = vᵀ(λv) = λ‖v‖² > 0.
Since ‖v‖² > 0, we must have λ > 0.

**(⟐)** If all λᵢ > 0, use spectral decomposition A = QΛQᵀ.
For any x ≠ 0: xᵀAx = xᵀQΛQᵀx = yᵀΛy where y = Qᵀx.
yᵀΛy = Σᵢλᵢyᵢ² > 0 since all λᵢ > 0 and y ≠ 0 (Q orthogonal).

**b) Positive semi-definite ⟺ all λ ≥ 0:**

Same argument with ≥ instead of >.

---

### Solution A10: Eigenvalues of Special Matrices

**a) Unitary matrix eigenvalues have |λ| = 1:**

Let Uv = λv. Then:
‖v‖² = vᴴv = vᴴUᴴUv = (Uv)ᴴ(Uv) = (λv)ᴴ(λv) = |λ|²‖v‖²

Since ‖v‖² ≠ 0: |λ|² = 1, so **|λ| = 1**.

**b) Projection matrix eigenvalues are 0 or 1:**

Let Pv = λv and P² = P.
Then Pv = P²v = P(Pv) = P(λv) = λPv = λ²v.

So λv = λ²v, meaning (λ² - λ)v = 0.
Since v ≠ 0: **λ² - λ = 0 → λ = 0 or λ = 1**.

**c) Nilpotent matrix eigenvalues are all 0:**

Let Aⁿ = 0 and Av = λv.
Then Aⁿv = λⁿv = 0.
Since v ≠ 0: **λⁿ = 0 → λ = 0**.

---

## Part B: Coding Solutions

### Solution B1: Eigenvalue Computation

```python
import numpy as np

def find_eigenvalues_2x2(A):
    """Find eigenvalues of a 2x2 matrix analytically."""
    # Characteristic polynomial: λ² - trace(A)λ + det(A) = 0
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    trace = a + d
    det = a * d - b * c
    
    # Quadratic formula
    discriminant = trace**2 - 4 * det
    
    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
    else:
        sqrt_disc = np.sqrt(-discriminant)
        lambda1 = (trace + 1j * sqrt_disc) / 2
        lambda2 = (trace - 1j * sqrt_disc) / 2
    
    return (lambda1, lambda2)

def find_eigenvectors_2x2(A, eigenvalue):
    """Find eigenvector for a given eigenvalue."""
    B = A - eigenvalue * np.eye(2)
    
    # Find null space of B
    # If B[0] is not zero, use it; otherwise use B[1]
    if np.abs(B[0, 0]) > 1e-10 or np.abs(B[0, 1]) > 1e-10:
        # From B[0,0]*v1 + B[0,1]*v2 = 0
        if np.abs(B[0, 1]) > 1e-10:
            v = np.array([B[0, 1], -B[0, 0]], dtype=complex)
        else:
            v = np.array([0, 1], dtype=complex)
    else:
        v = np.array([1, 0], dtype=complex)
    
    # Normalize
    v = v / np.linalg.norm(v)
    return v

# Test
A = np.array([[4, 2], [2, 1]])
evals = find_eigenvalues_2x2(A)
print(f"Eigenvalues: {evals}")
print(f"NumPy verification: {np.linalg.eigvals(A)}")
```

---

### Solution B2: Visualize Eigenvectors

```python
import matplotlib.pyplot as plt

def visualize_eigenvectors(A):
    """Visualize how eigenvectors transform."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform
    transformed = A @ circle
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Original
    ax1 = axes[0]
    ax1.plot(circle[0], circle[1], 'b-', linewidth=2, label='Unit circle')
    ax1.scatter([0], [0], color='black', s=50, zorder=5)
    
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(val):
            ax1.arrow(0, 0, vec[0].real, vec[1].real, head_width=0.1, 
                     head_length=0.05, fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2)
            ax1.text(vec[0].real*1.1, vec[1].real*1.1, f'v{i+1}', fontsize=12)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_title('Before Transformation')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Transformed
    ax2 = axes[1]
    ax2.plot(transformed[0], transformed[1], 'b-', linewidth=2, label='Transformed')
    ax2.scatter([0], [0], color='black', s=50, zorder=5)
    
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(val):
            transformed_vec = val.real * vec.real
            ax2.arrow(0, 0, transformed_vec[0], transformed_vec[1], head_width=0.1, 
                     head_length=0.05, fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2)
            ax2.text(transformed_vec[0]*1.1, transformed_vec[1]*1.1, 
                    f'λ{i+1}v{i+1}={val.real:.1f}v{i+1}', fontsize=10)
    
    max_val = max(np.abs(transformed).max(), 2) * 1.2
    ax2.set_xlim(-max_val, max_val)
    ax2.set_ylim(-max_val, max_val)
    ax2.set_aspect('equal')
    ax2.set_title(f'After Transformation\nEigenvalues: {eigenvalues.real}')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

# Test
A = np.array([[2, 1], [1, 2]])
visualize_eigenvectors(A)
```

---

### Solution B3: Power Iteration

```python
def power_iteration(A, max_iterations=100, tolerance=1e-10):
    """Find dominant eigenvalue and eigenvector using power iteration."""
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    eigenvalue_history = []
    
    for iteration in range(max_iterations):
        # Apply matrix
        y = A @ x
        
        # Estimate eigenvalue (Rayleigh quotient)
        eigenvalue = x @ y  # = x^T A x since x is normalized
        eigenvalue_history.append(eigenvalue)
        
        # Normalize
        x_new = y / np.linalg.norm(y)
        
        # Check convergence
        if iteration > 0 and abs(eigenvalue - eigenvalue_history[-2]) < tolerance:
            return {
                'eigenvalue': eigenvalue,
                'eigenvector': x_new,
                'iterations': iteration + 1,
                'convergence_history': eigenvalue_history
            }
        
        x = x_new
    
    return {
        'eigenvalue': eigenvalue,
        'eigenvector': x,
        'iterations': max_iterations,
        'convergence_history': eigenvalue_history
    }

# Test
A = np.array([[3, 1], [1, 3]])
result = power_iteration(A)
print(f"Dominant eigenvalue: {result['eigenvalue']:.6f}")
print(f"Eigenvector: {result['eigenvector']}")
print(f"Converged in {result['iterations']} iterations")
print(f"NumPy verification: {np.max(np.abs(np.linalg.eigvals(A)))}")
```

---

### Solution B4: Inverse Iteration

```python
def inverse_iteration(A, shift=0, max_iterations=100, tolerance=1e-10):
    """Find eigenvalue closest to 'shift'."""
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    # Shifted matrix
    B = A - shift * np.eye(n)
    
    for iteration in range(max_iterations):
        # Solve By = x (inverse power iteration)
        y = np.linalg.solve(B, x)
        
        # Estimate eigenvalue of B^(-1)
        mu = x @ y  # Rayleigh quotient
        
        # Eigenvalue of A is shift + 1/mu
        eigenvalue = shift + 1/mu
        
        # Normalize
        x_new = y / np.linalg.norm(y)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tolerance or np.linalg.norm(x_new + x) < tolerance:
            return {
                'eigenvalue': eigenvalue,
                'eigenvector': x_new,
                'iterations': iteration + 1
            }
        
        x = x_new
    
    return {
        'eigenvalue': eigenvalue,
        'eigenvector': x,
        'iterations': max_iterations
    }

# Test: Find eigenvalue closest to 2.5
A = np.array([[3, 1], [1, 3]])
result = inverse_iteration(A, shift=2.5)
print(f"Eigenvalue closest to 2.5: {result['eigenvalue']:.6f}")
print(f"All eigenvalues: {np.linalg.eigvals(A)}")
```

---

### Solution B5: PCA from Scratch

```python
def pca_eigen(X, n_components):
    """Implement PCA using eigendecomposition."""
    # Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # Compute covariance matrix
    n = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k components
    components = eigenvectors[:, :n_components].T
    explained_variance = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance / eigenvalues.sum()
    
    # Project data
    transformed = X_centered @ eigenvectors[:, :n_components]
    
    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'transformed': transformed
    }

# Test
np.random.seed(42)
X = np.random.randn(100, 5) @ np.array([[3, 0, 0, 0, 0],
                                         [0, 2, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 0, 0.5, 0],
                                         [0, 0, 0, 0, 0.1]])
result = pca_eigen(X, 2)
print(f"Explained variance ratio: {result['explained_variance_ratio']}")
print(f"Total explained: {result['explained_variance_ratio'].sum():.2%}")
```

---

### Solution B6: PageRank Algorithm

```python
def pagerank(adjacency_matrix, damping=0.85, max_iterations=100, tolerance=1e-8):
    """Compute PageRank using power iteration."""
    n = adjacency_matrix.shape[0]
    
    # Build transition matrix
    out_degrees = adjacency_matrix.sum(axis=0)
    
    # Handle dangling nodes (no outlinks)
    out_degrees[out_degrees == 0] = 1  # Avoid division by zero
    
    # Normalize columns to get transition probabilities
    M = adjacency_matrix / out_degrees
    
    # PageRank matrix with teleportation
    # M' = damping * M + (1 - damping) / n * ones
    
    # Initialize ranks uniformly
    ranks = np.ones(n) / n
    
    for iteration in range(max_iterations):
        # Power iteration step
        new_ranks = damping * (M @ ranks) + (1 - damping) / n
        
        # Check convergence
        if np.linalg.norm(new_ranks - ranks) < tolerance:
            return {
                'ranks': new_ranks,
                'iterations': iteration + 1
            }
        
        ranks = new_ranks
    
    return {
        'ranks': ranks,
        'iterations': max_iterations
    }

# Test with simple graph
# 0 -> 1 -> 2 -> 0 (cycle)
# Plus: 3 -> 0
adj = np.array([
    [0, 0, 1, 1],  # 0 gets links from 2 and 3
    [1, 0, 0, 0],  # 1 gets link from 0
    [0, 1, 0, 0],  # 2 gets link from 1
    [0, 0, 0, 0]   # 3 has no incoming links
])

result = pagerank(adj)
print("PageRank scores:", result['ranks'])
print(f"Sum: {result['ranks'].sum():.4f}")
print(f"Converged in {result['iterations']} iterations")
```

---

### Solution B7: Eigenvalue Sensitivity

```python
def eigenvalue_sensitivity(A, epsilon=1e-5):
    """Analyze eigenvalue sensitivity to perturbations."""
    original_evals = np.linalg.eigvals(A)
    
    # Add random perturbation
    perturbation = np.random.randn(*A.shape) * epsilon
    A_perturbed = A + perturbation
    perturbed_evals = np.linalg.eigvals(A_perturbed)
    
    # Sort for comparison
    original_evals = np.sort(original_evals)
    perturbed_evals = np.sort(perturbed_evals)
    
    # Condition number for eigenvalue problem
    # Related to eigenvector condition
    eigenvalues, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    
    condition_numbers = np.array([
        np.linalg.norm(V[:, i]) * np.linalg.norm(V_inv[i, :])
        for i in range(len(eigenvalues))
    ])
    
    return {
        'original_eigenvalues': original_evals,
        'perturbed_eigenvalues': perturbed_evals,
        'condition_numbers': condition_numbers
    }

# Test
A = np.array([[1, 1000], [0, 2]])  # Ill-conditioned
result = eigenvalue_sensitivity(A)
print("Original eigenvalues:", result['original_eigenvalues'])
print("Perturbed eigenvalues:", result['perturbed_eigenvalues'])
print("Condition numbers:", result['condition_numbers'])
```

---

### Solution B8: Stability Analysis

```python
def analyze_linear_system_stability(A):
    """Analyze stability of discrete linear system."""
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    # Sort by magnitude
    sorted_idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_sorted = eigenvalues[sorted_idx]
    
    convergence_rate = np.abs(eigenvalues_sorted[1]) / np.abs(eigenvalues_sorted[0]) if len(eigenvalues) > 1 else 0
    
    return {
        'eigenvalues': eigenvalues,
        'spectral_radius': spectral_radius,
        'is_stable': spectral_radius < 1,
        'dominant_mode': sorted_idx[0],
        'convergence_rate': convergence_rate
    }

def simulate_linear_system(A, x0, n_steps):
    """Simulate linear system."""
    trajectory = [x0]
    x = x0.copy()
    
    for _ in range(n_steps):
        x = A @ x
        trajectory.append(x.copy())
    
    return np.array(trajectory)

# Test
A_stable = np.array([[0.8, 0.1], [0.1, 0.7]])
A_unstable = np.array([[1.1, 0.1], [0.1, 0.9]])

for name, A in [('Stable', A_stable), ('Unstable', A_unstable)]:
    result = analyze_linear_system_stability(A)
    print(f"\n{name} system:")
    print(f"  Eigenvalues: {result['eigenvalues']}")
    print(f"  Spectral radius: {result['spectral_radius']:.4f}")
    print(f"  Is stable: {result['is_stable']}")
```

---

### Solution B9: Spectral Clustering

```python
def spectral_clustering(similarity_matrix, n_clusters):
    """Perform spectral clustering."""
    W = similarity_matrix
    n = W.shape[0]
    
    # Degree matrix
    D = np.diag(W.sum(axis=1))
    
    # Laplacian (unnormalized)
    L = D - W
    
    # Find smallest k eigenvectors of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort and take bottom k (skip the first trivial eigenvector)
    idx = np.argsort(eigenvalues)
    features = eigenvectors[:, idx[1:n_clusters+1]]  # Skip first (all ones for connected graph)
    
    # K-means on eigenvector features
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels

# Test with block diagonal similarity matrix
n_per_cluster = 20
n_clusters = 3

# Create similarity matrix with clusters
W = np.zeros((n_per_cluster * n_clusters, n_per_cluster * n_clusters))
for i in range(n_clusters):
    start = i * n_per_cluster
    end = (i + 1) * n_per_cluster
    W[start:end, start:end] = np.random.rand(n_per_cluster, n_per_cluster) * 0.5 + 0.5
    
W = (W + W.T) / 2  # Symmetrize
np.fill_diagonal(W, 0)  # No self-loops

labels = spectral_clustering(W, n_clusters)
print(f"Cluster assignments: {labels}")
print(f"Points per cluster: {np.bincount(labels)}")
```

---

### Solution B10: QR Algorithm

```python
def qr_iteration(A, max_iterations=50):
    """QR algorithm for finding all eigenvalues."""
    A_k = A.astype(float).copy()
    iterations = [A_k.copy()]
    
    for _ in range(max_iterations):
        # QR decomposition
        Q, R = np.linalg.qr(A_k)
        
        # Update: A_{k+1} = R @ Q
        A_k = R @ Q
        iterations.append(A_k.copy())
    
    # Eigenvalues are on diagonal (for symmetric matrices)
    # or in 2x2 blocks (for complex eigenvalues)
    eigenvalues = np.diag(A_k)
    
    return {
        'eigenvalues': eigenvalues,
        'iterations': iterations
    }

# Test
A = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 2]], dtype=float)
result = qr_iteration(A)
print("QR algorithm eigenvalues:", np.sort(result['eigenvalues']))
print("NumPy eigenvalues:", np.sort(np.linalg.eigvals(A)))
```

---

## Part C: Conceptual Answers

### Answer C1: Eigenvalue Interpretation

**a) All positive eigenvalues:**
Matrix preserves orientation and stretches in all directions. For symmetric matrix: positive definite, minimum at origin for quadratic form.

**b) Eigenvalue 0:**
Matrix collapses some direction to zero. Singular (non-invertible). The null space is non-trivial.

**c) Complex eigenvalues:**
Rotation is involved. No real direction is preserved. Come in conjugate pairs for real matrices.

**d) Repeated eigenvalues:**
Multiple eigenvectors for same scaling factor. May or may not have enough independent eigenvectors (defective vs diagonalizable).

---

### Answer C2: PCA Eigenvalues

**a) What eigenvalue represents:** 
The variance of data projected onto the corresponding eigenvector (principal component). Larger eigenvalue = more variance = more information.

**b) Descending order:**
First PC captures most variance, second captures most remaining, etc. We want to keep the most important directions first.

**c) How many to keep:**
- Explained variance ratio ≥ 95%
- Scree plot elbow method
- Cross-validation for downstream task
- Computational constraints

---

### Answer C3: PageRank Convergence

**a) What guarantees convergence:**
The PageRank matrix is stochastic (columns sum to 1) with teleportation making it primitive (irreducible, aperiodic). Perron-Frobenius theorem guarantees unique dominant eigenvector.

**b) Damping factor role:**
- Prevents "rank sinks" (pages with no outlinks)
- Ensures ergodicity (can reach any page eventually)
- Models "bored surfer" who randomly jumps
- Typical value 0.85 balances structure and exploration

**c) Associated eigenvalue:**
PageRank is eigenvector for **eigenvalue 1** (the largest eigenvalue of a stochastic matrix).

---

### Answer C4: RNN Gradient Problems

**a) Exploding gradients (|λ| > 1):**
Gradient involves products of W across timesteps. If largest |λ| > 1, terms grow exponentially: λᵀ → ∞, causing numerical overflow.

**b) Vanishing gradients (|λ| < 1):**
If largest |λ| < 1, products shrink exponentially: λᵀ → 0. Gradients become negligibly small, preventing learning long-range dependencies.

**c) LSTM/GRU solutions:**
- Gates control information flow (can maintain λ ≈ 1 for important info)
- Additive structure allows gradients to flow unchanged
- Cell state provides "highway" for gradients
- Forget gate explicitly controls memory decay

---

### Answer C5: Optimization Landscape

**a) Eigenvalues of Hessian:**
- All positive: local minimum
- All negative: local maximum  
- Mixed signs: saddle point
- Magnitude indicates curvature strength

**b) Near-zero eigenvalue:**
Flat direction in loss landscape. Movement in that direction doesn't change loss much. Can indicate:
- Redundant parameters
- Near-singularity
- Slow convergence direction

**c) Learning rate connection:**
- Learning rate should be < 2/λ_max to avoid divergence
- If λ_max ≫ λ_min (ill-conditioned), need small LR for stability
- Adaptive methods (Adam, etc.) help by per-parameter scaling
- Preconditioning approximates Newton's method (scales by H⁻¹)
