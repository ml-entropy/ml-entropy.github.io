# Solutions: Singular Value Decomposition (SVD)

## Part A: Theory Solutions

### Solution A1: Basic SVD Computation

$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

**a) SVD by inspection:**

A is already "almost" in SVD form. It's diagonal in its upper part.

- **Σ** = [[3, 0], [0, 2], [0, 0]] (same structure as A)
- **V** = I₂ = [[1, 0], [0, 1]] (columns of A are already aligned with standard basis)
- **U** = I₃ = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (no rotation needed)

Actually, to be precise for a 3×2 matrix:
- U is 3×3: can complete with any orthonormal third column
- Σ is 3×2
- V is 2×2: identity

**b) Singular values:** σ₁ = 3, σ₂ = 2

**c) Rank:** 2 (two non-zero singular values)

---

### Solution A2: SVD of 2×2 Matrix

$$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

**a) AᵀA:**
$$A^TA = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}$$

Eigenvalues: det(AᵀA - λI) = (1-λ)(2-λ) - 1 = λ² - 3λ + 1 = 0

λ = (3 ± √5)/2, so λ₁ ≈ 2.618, λ₂ ≈ 0.382

**b) Singular values:**
σ₁ = √λ₁ ≈ 1.618
σ₂ = √λ₂ ≈ 0.618

(Note: σ₁ = φ (golden ratio), σ₂ = 1/φ)

**c) V (eigenvectors of AᵀA):**

For λ₁ = (3+√5)/2:
(AᵀA - λ₁I)v = 0

Solving: v₁ ∝ [1, (1+√5)/2]ᵀ = [1, φ]ᵀ

Normalized: v₁ = [1, φ]ᵀ / √(1 + φ²) ≈ [0.526, 0.851]ᵀ

Similarly: v₂ ≈ [0.851, -0.526]ᵀ

**d) U:**
u₁ = Av₁/σ₁ ≈ [0.851, 0.526]ᵀ
u₂ = Av₂/σ₂ ≈ [0.526, -0.851]ᵀ

**e) Verify:** A = UΣVᵀ (numerical verification confirms)

---

### Solution A3: Properties of Singular Values

Given: A is 4×3, σ₁ = 5, σ₂ = 3, σ₃ = 1

**a) rank(A)** = 3 (three non-zero singular values)

**b) ‖A‖₂** = σ₁ = **5**

**c) ‖A‖_F** = √(5² + 3² + 1²) = √(25 + 9 + 1) = √35 ≈ **5.92**

**d) κ(A)** = σ₁/σ₃ = 5/1 = **5**

**e) Singular values of 2A:** 2σ₁ = 10, 2σ₂ = 6, 2σ₃ = 2

Singular values of Aᵀ: **Same as A** (σ₁ = 5, σ₂ = 3, σ₃ = 1)

---

### Solution A4: Low-Rank Approximation

Given σ₁ = 3, σ₂ = 1.

**a) Rank-1 approximation:**
$$A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T = 3 \mathbf{u}_1 \mathbf{v}_1^T$$

**b) ‖A - A₁‖₂** = σ₂ = **1** (by Eckart-Young theorem)

**c) ‖A - A₁‖_F** = √(σ₂²) = σ₂ = **1**

(When there's only one remaining singular value, both norms equal it)

---

### Solution A5: Geometric Interpretation

Given: σ₁ = 4, σ₂ = 2, and the singular vectors.

**a) What A does to unit circle:**

1. Vᵀ rotates the circle (V's columns are at 45°)
2. Σ stretches: factor 4 along e₁, factor 2 along e₂
3. U rotates/reflects: swaps axes

Result: An ellipse with semi-axes of length 4 and 2.

**b) Image of [1, 0]ᵀ:**

[1, 0]ᵀ = (1/√2)v₁ + (1/√2)v₂

A[1, 0] = (1/√2)σ₁u₁ + (1/√2)σ₂u₂
       = (1/√2)(4)[0, 1]ᵀ + (1/√2)(2)[1, 0]ᵀ
       = [√2, 2√2]ᵀ ≈ [1.41, 2.83]ᵀ

**c) Direction of maximal stretch:**
v₁ = [1/√2, 1/√2]ᵀ (45° from x-axis)

---

### Solution A6: Connection to Eigendecomposition

For symmetric S = Sᵀ:

**a)** If Sv = λv, then:
- Sᵀv = Sv = λv (since S = Sᵀ)
- SᵀSv = S²v = λ²v

So v is an eigenvector of SᵀS with eigenvalue λ².
Therefore, singular value σ = √(λ²) = |λ|.

**b)** For symmetric S, the SVD is:
- U and V are related: they can be chosen to be the same (up to signs)
- If S has eigenvalue λ < 0, then U and V differ by a sign in that column

More precisely: S = QΛQᵀ where Λ can have negative entries.
In SVD form: S = U|Λ|Vᵀ where U = Q·sign(Λ) and V = Q.

**c)** For S = [[2, 1], [1, 2]]:

Eigenvalues: λ₁ = 3, λ₂ = 1 (both positive)
Eigenvectors: q₁ = [1, 1]ᵀ/√2, q₂ = [1, -1]ᵀ/√2

Since both eigenvalues are positive:
- Singular values: σ₁ = 3, σ₂ = 1
- U = V = Q = [[1/√2, 1/√2], [1/√2, -1/√2]]
- Σ = [[3, 0], [0, 1]]

---

### Solution A7: Pseudoinverse Computation

$$A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

**a) SVD:**
- U = I₃
- Σ = A (it's already diagonal)
- V = I₂

Singular values: σ₁ = 1, σ₂ = 2 (or should order: σ₁ = 2, σ₂ = 1)

Let's use σ₁ = 2, σ₂ = 1 with corresponding reordering.

**b) Pseudoinverse:**
$$\Sigma^+ = \begin{bmatrix} 1/\sigma_1 & 0 & 0 \\ 0 & 1/\sigma_2 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix}$$

(For original ordering)

$$A^+ = V\Sigma^+U^T = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix}$$

**c) Verify AA⁺A = A:**

$$AA^+A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

$$= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} = A \checkmark$$

**d) Least squares for Ax = [1, 2, 3]ᵀ:**

$$x = A^+b = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

Verify: Ax = [1, 2, 0]ᵀ, residual = [0, 0, 3]ᵀ, which is orthogonal to column space of A.

---

### Solution A8: Eckart-Young Theorem Proof

**To prove:** For any rank-k matrix B, ‖A - Aₖ‖_F ≤ ‖A - B‖_F

**Proof:**

Let A = UΣVᵀ and Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ

‖A - Aₖ‖_F² = ‖Σᵢ₌ₖ₊₁ʳ σᵢuᵢvᵢᵀ‖_F² = Σᵢ₌ₖ₊₁ʳ σᵢ² (since rank-1 terms are orthogonal)

Now for any rank-k matrix B, let B = UΣ̃Vᵀ in the same U, V basis.

Since rank(B) ≤ k, at most k of the "diagonal" entries in Σ̃ can be non-zero.

‖A - B‖_F² = ‖Σ - Σ̃‖_F² (unitary invariance of Frobenius norm)

The optimal choice for Σ̃ to minimize this is to keep the k largest entries of Σ.

Therefore ‖A - B‖_F² ≥ Σᵢ₌ₖ₊₁ʳ σᵢ² = ‖A - Aₖ‖_F². ∎

---

### Solution A9: SVD of Outer Product

A = uvᵀ where u ∈ ℝᵐ, v ∈ ℝⁿ are unit vectors.

**a) Rank:** 1 (only one linearly independent column/row)

**b) Singular values:** σ₁ = 1, σ₂ = ... = σᵣ = 0

To see this: AᵀA = vuᵀuv = v·1·vᵀ = vvᵀ (since u is unit)
This has eigenvalue 1 (with eigenvector v) and eigenvalue 0 (multiplicity n-1).

**c) Singular vectors:**
- Right singular vector: v₁ = v
- Left singular vector: u₁ = Av₁/σ₁ = uvᵀv/1 = u

**d) Generalization:** For A = σuvᵀ (σ > 0):
- Singular value: σ₁ = σ
- U has u as first column
- V has v as first column
- SVD: A = σuvᵀ = [u | ...][σ, 0, ...; 0, 0, ...][v | ...]ᵀ

---

### Solution A10: SVD and Matrix Norms

**a) ‖A‖₂ = σ₁:**

‖A‖₂ = max_{‖x‖=1} ‖Ax‖

Using SVD: Ax = UΣVᵀx. Let y = Vᵀx, so ‖y‖ = ‖x‖ = 1.

‖Ax‖ = ‖UΣy‖ = ‖Σy‖ (U is orthogonal)
     = √(Σᵢ σᵢ²yᵢ²)
     ≤ σ₁√(Σᵢ yᵢ²) = σ₁

Equality when y = e₁, i.e., x = v₁. ∎

**b) ‖A‖_F = √(Σσᵢ²):**

‖A‖_F² = tr(AᵀA) = tr(VΣᵀUᵀUΣVᵀ) = tr(VΣᵀΣVᵀ) = tr(ΣᵀΣ) = Σσᵢ² ∎

**c) ‖A - B‖₂ ≥ σₖ₊₁ for rank(B) ≤ k:**

By Eckart-Young, Aₖ minimizes ‖A - B‖₂ over rank-k matrices.
‖A - Aₖ‖₂ = σₖ₊₁.
Therefore for any rank-k matrix B: ‖A - B‖₂ ≥ σₖ₊₁. ∎

---

## Part B: Coding Solutions

### Solution B1: Manual SVD Computation

```python
import numpy as np

def svd_via_eigendecomposition(A):
    """Compute SVD via eigendecomposition of A^T A."""
    m, n = A.shape
    
    # Compute A^T A
    ATA = A.T @ A
    
    # Eigendecomposition of A^T A
    eigenvalues, V = np.linalg.eigh(ATA)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Singular values (handle numerical issues)
    S = np.sqrt(np.maximum(eigenvalues, 0))
    
    # Compute U
    r = np.sum(S > 1e-10)  # Rank
    U = np.zeros((m, r))
    
    for i in range(r):
        if S[i] > 1e-10:
            U[:, i] = A @ V[:, i] / S[i]
    
    # Return thin SVD
    return U, S[:r], V[:, :r].T

# Test
A = np.array([[1, 1], [0, 1], [1, 0]])
U, S, Vt = svd_via_eigendecomposition(A)
print("Our SVD:")
print(f"U:\n{U}")
print(f"S: {S}")
print(f"Vt:\n{Vt}")

# Verify
print(f"\nReconstruction error: {np.linalg.norm(A - U @ np.diag(S) @ Vt)}")

# Compare with numpy
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
print(f"\nNumPy S: {S_np}")
```

---

### Solution B2: Image Compression with SVD

```python
import matplotlib.pyplot as plt

def compress_image_svd(image, k):
    """Compress image using rank-k SVD."""
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Truncate to rank k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct
    compressed = U_k @ np.diag(S_k) @ Vt_k
    
    # Compression ratio
    m, n = image.shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    
    # Error
    error = np.linalg.norm(image - compressed, 'fro') / np.linalg.norm(image, 'fro')
    
    return {
        'compressed': compressed,
        'compression_ratio': original_size / compressed_size,
        'relative_error': error
    }

def visualize_compression(image, k_values):
    """Visualize compression at different levels."""
    n_plots = len(k_values) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, k in enumerate(k_values):
        result = compress_image_svd(image, k)
        axes[i+1].imshow(result['compressed'], cmap='gray')
        axes[i+1].set_title(f'k={k}\nRatio: {result["compression_ratio"]:.1f}x\nError: {result["relative_error"]:.2%}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test with random "image"
np.random.seed(42)
image = np.random.rand(100, 100)
# Add some structure
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
image = np.sin(4*np.pi*x) * np.cos(4*np.pi*y) + 0.3*np.random.randn(100, 100)

visualize_compression(image, [5, 10, 20, 50])
```

---

### Solution B3: PCA via SVD

```python
def pca_svd(X, n_components):
    """PCA using SVD."""
    # Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components are rows of Vt
    components = Vt[:n_components]
    
    # Explained variance
    n_samples = X.shape[0]
    explained_variance = (S[:n_components]**2) / (n_samples - 1)
    total_variance = (S**2).sum() / (n_samples - 1)
    explained_variance_ratio = explained_variance / total_variance.sum() * (S**2).sum() / (S[:n_components]**2).sum()
    
    # Simpler: just use the ratio of squared singular values
    explained_variance_ratio = (S[:n_components]**2) / (S**2).sum()
    
    # Project data
    transformed = X_centered @ Vt[:n_components].T
    # Or equivalently: U[:, :n_components] * S[:n_components]
    
    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'transformed': transformed
    }

# Test
np.random.seed(42)
X = np.random.randn(100, 5) @ np.diag([3, 2, 1, 0.5, 0.1]) @ np.random.randn(5, 5)
result = pca_svd(X, 2)
print(f"Explained variance ratio: {result['explained_variance_ratio']}")
print(f"Total explained: {result['explained_variance_ratio'].sum():.2%}")
```

---

### Solution B4: Pseudoinverse and Least Squares

```python
def solve_least_squares_svd(A, b, tol=1e-10):
    """Solve least squares using SVD pseudoinverse."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Compute pseudoinverse of S
    S_inv = np.where(S > tol, 1/S, 0)
    
    # x = V @ S^+ @ U^T @ b
    x = Vt.T @ (S_inv * (U.T @ b))
    
    # Residual
    residual = np.linalg.norm(A @ x - b)
    
    # Effective rank
    rank = np.sum(S > tol)
    
    return {
        'x': x,
        'residual': residual,
        'rank': rank
    }

# Test with overdetermined system
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2.5])

result = solve_least_squares_svd(A, b)
print(f"Solution x: {result['x']}")
print(f"Residual: {result['residual']:.6f}")
print(f"Rank: {result['rank']}")

# Compare with numpy
x_np = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"NumPy solution: {x_np}")
```

---

### Solution B5: Low-Rank Matrix Approximation

```python
def best_rank_k_approximation(A, k):
    """Best rank-k approximation via SVD."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Truncate
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct
    A_k = U_k @ np.diag(S_k) @ Vt_k
    
    # Errors
    error_frob = np.sqrt(np.sum(S[k:]**2))
    error_spectral = S[k] if k < len(S) else 0
    
    # Energy retained
    energy = np.sum(S_k**2) / np.sum(S**2)
    
    return {
        'A_k': A_k,
        'singular_values_kept': S_k,
        'approximation_error_frobenius': error_frob,
        'approximation_error_spectral': error_spectral,
        'energy_retained': energy
    }

# Test
A = np.random.randn(10, 8)
for k in [1, 2, 4, 6]:
    result = best_rank_k_approximation(A, k)
    print(f"k={k}: Energy={result['energy_retained']:.2%}, Error_F={result['approximation_error_frobenius']:.4f}")
```

---

### Solution B6: SVD for Recommender Systems

```python
def matrix_completion_svd(R, k, mask=None, n_iterations=10):
    """Matrix completion using iterative SVD."""
    if mask is None:
        mask = ~np.isnan(R)
    
    # Initialize missing entries with row means
    R_filled = R.copy()
    R_filled = np.where(np.isnan(R_filled), 0, R_filled)
    
    row_means = np.nanmean(R, axis=1)
    for i in range(R.shape[0]):
        R_filled[i, ~mask[i]] = row_means[i] if not np.isnan(row_means[i]) else 0
    
    for iteration in range(n_iterations):
        # SVD
        U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)
        
        # Truncate
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct
        R_approx = U_k @ np.diag(S_k) @ Vt_k
        
        # Keep observed entries, fill unobserved with approximation
        R_filled = np.where(mask, R, R_approx)
    
    return {
        'completed': R_approx,
        'U': U_k @ np.diag(np.sqrt(S_k)),
        'V': Vt_k.T @ np.diag(np.sqrt(S_k))
    }

# Test
np.random.seed(42)
# True low-rank matrix
U_true = np.random.randn(20, 3)
V_true = np.random.randn(15, 3)
R_true = U_true @ V_true.T

# Observe 50% of entries
mask = np.random.rand(20, 15) > 0.5
R_observed = np.where(mask, R_true, np.nan)

# Complete
result = matrix_completion_svd(R_observed, k=3, mask=mask)

# Evaluate
error = np.sqrt(np.mean((result['completed'][~mask] - R_true[~mask])**2))
print(f"RMSE on unobserved entries: {error:.4f}")
```

---

### Solution B7: Truncated/Randomized SVD

```python
def randomized_svd(A, k, n_oversamples=10, n_iter=2):
    """Randomized SVD for approximate rank-k decomposition."""
    m, n = A.shape
    
    # Step 1: Random projection
    l = k + n_oversamples
    Omega = np.random.randn(n, l)
    
    # Step 2: Form Y = A @ Omega and power iteration
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    
    # Step 3: Orthonormalize
    Q, _ = np.linalg.qr(Y)
    
    # Step 4: Form B = Q^T @ A
    B = Q.T @ A
    
    # Step 5: SVD of small matrix B
    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 6: Recover U
    U = Q @ U_B
    
    # Return only k components
    return U[:, :k], S[:k], Vt[:k, :]

# Test
np.random.seed(42)
A = np.random.randn(1000, 500)  # Large matrix

# Randomized SVD
U_rand, S_rand, Vt_rand = randomized_svd(A, k=10)
A_approx_rand = U_rand @ np.diag(S_rand) @ Vt_rand

# Compare with exact
U_exact, S_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)
A_approx_exact = U_exact[:, :10] @ np.diag(S_exact[:10]) @ Vt_exact[:10, :]

print(f"Randomized SVD error: {np.linalg.norm(A - A_approx_rand, 'fro'):.4f}")
print(f"Exact SVD (k=10) error: {np.linalg.norm(A - A_approx_exact, 'fro'):.4f}")
print(f"Singular values comparison: {S_rand[:5]} vs {S_exact[:5]}")
```

---

## Part C: Conceptual Answers

### Answer C1: SVD vs Eigendecomposition

| Aspect | SVD | Eigendecomposition |
|--------|-----|-------------------|
| Matrices | Any m×n | Square only |
| Always exists | Yes | Not always (non-diagonalizable) |
| Values | Real, non-negative | Can be complex, any sign |
| Vectors | Always orthogonal | Not always orthogonal |
| Use case | Low-rank approx, least squares | Powers, stability, symmetric matrices |

**Use eigendecomposition when:**
- Matrix is symmetric (PCA on covariance)
- Need matrix powers
- Analyzing stability

**Use SVD when:**
- Matrix is not square
- Need best low-rank approximation
- Numerical stability matters
- Computing pseudoinverse

---

### Answer C2: Choosing k

**Methods:**

1. **Explained variance:** Keep enough k to explain 90-95% of variance
   - Compute cumulative sum of σ²
   
2. **Scree plot:** Look for "elbow" where singular values drop sharply

3. **Cross-validation:** For prediction tasks, choose k that minimizes validation error

4. **Gavish-Donoho threshold:** For denoising, optimal threshold based on noise level

5. **Application-specific:** 
   - Image compression: visual quality acceptable
   - Recommender systems: prediction accuracy

---

### Answer C3: SVD and Matrix Rank

**How SVD reveals rank:**
- Rank = number of non-zero singular values
- The columns of U corresponding to non-zero σ span the column space
- The columns of V corresponding to non-zero σ span the row space

**In practice (numerical rank):**
- Singular values are never exactly zero due to floating point
- Define threshold τ (e.g., τ = ε × σ₁ × max(m,n) where ε is machine precision)
- Numerical rank = count of σ > τ

---

### Answer C4: Computational Complexity

**Full SVD:** O(min(mn², m²n)) - expensive for large matrices

**Randomized SVD:** O(mnk) for rank-k approximation
- Much faster when k << min(m,n)
- Good when you only need top k components

**When to use randomized:**
- k << min(m,n) (low-rank structure)
- Approximate solution is acceptable
- Matrix is too large for full SVD
- Streaming/online settings

---

### Answer C5: SVD in Deep Learning

**a) Weight initialization:**
- Orthogonal initialization: Initialize weights as random orthogonal matrices (from SVD)
- Helps with gradient flow in deep networks

**b) Model compression:**
- Replace weight matrix W with low-rank approximation U_k S_k V_k^T
- Reduces parameters from mn to k(m+n)
- Can be fine-tuned after compression

**c) Analyzing representations:**
- SVD of activation matrices reveals learned features
- Effective dimensionality from singular value spectrum
- Understanding what neural networks learn

**Additional uses:**
- Spectral normalization (divide by σ₁)
- Low-rank adaptation (LoRA) for efficient fine-tuning
