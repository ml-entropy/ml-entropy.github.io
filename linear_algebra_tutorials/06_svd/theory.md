# Tutorial 06: Singular Value Decomposition (SVD)

## Introduction: The Most Important Matrix Factorization

The **Singular Value Decomposition (SVD)** is arguably the most important matrix factorization in applied mathematics and machine learning. Unlike eigendecomposition, SVD works for **any** matrix—not just square ones.

**The Big Idea:** Any matrix can be decomposed into:
1. A rotation/reflection (U)
2. A scaling along orthogonal axes (Σ)
3. Another rotation/reflection (Vᵀ)

$$A = U \Sigma V^T$$

**Why SVD Matters:**
- Works for **any** m×n matrix
- Reveals the **rank** of a matrix
- Provides the **best low-rank approximation**
- Foundation for: PCA, recommender systems, image compression, NLP, pseudoinverse

---

## Part 1: Geometric Interpretation

### What SVD Does Geometrically

Any linear transformation can be broken into three steps:

1. **Vᵀ:** Rotate/reflect in the input space (domain)
2. **Σ:** Scale along orthogonal axes
3. **U:** Rotate/reflect in the output space (range)

**Key insight:** Every linear transformation is just rotation + scaling + rotation!

### Visualizing the Transformation

Consider A mapping ℝ² → ℝ²:

```
Input space:        After Vᵀ:          After Σ:           After U:
   Unit Circle  →   Rotated Circle  →   Stretched Axes  →   Rotated Ellipse
   
      ○             ○                    ⬭                   ⬭
     /|\           /|\                 /   \               ↗   ↘
```

- The **singular values** σᵢ are the lengths of the ellipse's semi-axes
- The **columns of V** are the directions in input space that get mapped to the axes
- The **columns of U** are the principal directions of the output ellipse

---

## Part 2: Definition and Properties

### Formal Definition

For any m×n matrix A:

$$A = U \Sigma V^T$$

where:
- **U** is m×m orthogonal (UᵀU = I)
- **Σ** is m×n diagonal with σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
- **V** is n×n orthogonal (VᵀV = I)
- r = rank(A) = number of non-zero singular values

### Terminology

- **σᵢ**: Singular values (non-negative, conventionally ordered σ₁ ≥ σ₂ ≥ ...)
- **uᵢ**: Left singular vectors (columns of U)
- **vᵢ**: Right singular vectors (columns of V)

### Key Properties

1. **Av_i = σᵢuᵢ** (V maps to scaled U)
2. **Aᵀuᵢ = σᵢvᵢ** (U maps back to scaled V)
3. **rank(A) = number of non-zero singular values**
4. **‖A‖₂ = σ₁** (operator norm = largest singular value)
5. **‖A‖_F = √(Σσᵢ²)** (Frobenius norm)

---

## Part 3: Derivation from Eigendecomposition

### Connection to AᵀA and AAᵀ

**Key insight:** Singular values and vectors come from the eigendecomposition of AᵀA and AAᵀ.

**Step 1:** Consider AᵀA (n×n, symmetric, positive semi-definite)

$$A^TA = (U\Sigma V^T)^T(U\Sigma V^T) = V\Sigma^T U^T U \Sigma V^T = V\Sigma^T\Sigma V^T = V(\Sigma^T\Sigma)V^T$$

So **V contains the eigenvectors of AᵀA**, and **σᵢ² are the eigenvalues of AᵀA**.

**Step 2:** Consider AAᵀ (m×m, symmetric, positive semi-definite)

$$AA^T = U\Sigma V^T V\Sigma^T U^T = U\Sigma\Sigma^T U^T$$

So **U contains the eigenvectors of AAᵀ**, with the same eigenvalues σᵢ².

### Why Singular Values are Non-negative

Since AᵀA is positive semi-definite, its eigenvalues (σᵢ²) are non-negative.
Taking square roots: σᵢ ≥ 0.

### The Complete SVD Algorithm

1. Compute AᵀA
2. Find eigenvalues λᵢ of AᵀA; set σᵢ = √λᵢ
3. Find corresponding eigenvectors → columns of V
4. Compute uᵢ = Avᵢ/σᵢ for non-zero singular values
5. Complete U with orthonormal basis for null space of Aᵀ

---

## Part 4: Thin vs Full SVD

### Full SVD

For m×n matrix A:
- U is m×m (full)
- Σ is m×n (rectangular diagonal)
- V is n×n (full)

### Thin/Reduced SVD

If m > n (tall matrix):
- U is m×n (only n columns)
- Σ is n×n (square diagonal)
- V is n×n (unchanged)

This is more efficient and often all we need.

### Compact SVD

Keep only the r non-zero singular values:
- U is m×r
- Σ is r×r
- V is n×r

$$A = U_r \Sigma_r V_r^T$$

---

## Part 5: Low-Rank Approximation

### The Eckart-Young-Mirsky Theorem

**The most important theorem about SVD:**

The best rank-k approximation to A (in Frobenius or spectral norm) is:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

**Error:**
- Frobenius norm: ‖A - Aₖ‖_F = √(σₖ₊₁² + ... + σᵣ²)
- Spectral norm: ‖A - Aₖ‖₂ = σₖ₊₁

### Why This Matters

Low-rank approximation enables:
- **Data compression**: Store k(m+n+1) numbers instead of mn
- **Noise reduction**: Small singular values often correspond to noise
- **Dimensionality reduction**: Like PCA

### Outer Product Form

SVD can be written as sum of rank-1 matrices:

$$A = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \sigma_2 \mathbf{u}_2\mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r\mathbf{v}_r^T$$

Each term σᵢuᵢvᵢᵀ is a rank-1 matrix. Truncating this sum gives low-rank approximation.

---

## Part 6: ML Application - Dimensionality Reduction (PCA via SVD)

### PCA Recap

PCA finds directions of maximum variance. For centered data X (n samples × d features):
- Principal components are eigenvectors of covariance matrix XᵀX/(n-1)

### SVD Connection

Instead of computing XᵀX explicitly, use SVD of X:

$$X = U \Sigma V^T$$

Then:
- **V** contains the principal components (eigenvectors of XᵀX)
- **σᵢ²/(n-1)** are the variances (eigenvalues of covariance)
- **UΣ** gives the projected data

### Why SVD for PCA?

1. **More numerically stable** - Doesn't compute XᵀX (which can have numerical issues)
2. **More efficient for n > d** - SVD of X vs eigendecomposition of XᵀX
3. **Handles both cases** - Works whether n > d or d > n

### Code Sketch

```python
# PCA via SVD
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Principal components
components = Vt  # Each row is a PC

# Projected data (k components)
X_projected = X_centered @ Vt[:k].T  # or equivalently: U[:, :k] * S[:k]

# Explained variance
explained_var = S**2 / (n - 1)
```

---

## Part 7: ML Application - Recommender Systems

### The Matrix Completion Problem

Given a user-item rating matrix R (users × items) with missing entries:
- Find low-rank approximation R ≈ UV^T
- Use it to predict missing ratings

### SVD Approach

1. Fill missing values with estimates (e.g., row/column means)
2. Compute SVD: R ≈ UₖΣₖVₖᵀ
3. Predicted ratings: R̂ = UₖΣₖVₖᵀ

**Interpretation:**
- Each user represented by a k-dimensional vector (row of UₖΣₖ^{1/2})
- Each item represented by a k-dimensional vector (row of VₖΣₖ^{1/2})
- Predicted rating = dot product of user and item vectors

### Truncated SVD for Recommendations

```python
from scipy.sparse.linalg import svds

# Sparse SVD for large matrices
U, S, Vt = svds(R_filled, k=50)

# Predict rating for user u, item i
predicted = U[u] @ np.diag(S) @ Vt[:, i]
```

---

## Part 8: ML Application - Image Compression

### Images as Matrices

A grayscale image is a matrix where each entry is a pixel intensity.

### SVD Compression

1. Compute SVD of image matrix
2. Keep only k largest singular values
3. Reconstruct: Image_k = UₖΣₖVₖᵀ

**Storage:**
- Original: m × n values
- Compressed: k(m + n + 1) values
- Compression ratio: mn / [k(m + n + 1)]

### Quality vs Compression Trade-off

- **More singular values** → Better quality, less compression
- **Fewer singular values** → Worse quality, more compression
- Usually 10-50 singular values capture most of the image

---

## Part 9: ML Application - Latent Semantic Analysis (LSA)

### Document-Term Matrices

For text analysis:
- Rows = documents
- Columns = terms (words)
- Entries = term frequencies (or TF-IDF)

### SVD for Semantic Analysis

$$\text{Term-Doc} = U \Sigma V^T$$

**Interpretation:**
- U: Documents in "concept" space
- V: Terms in "concept" space  
- Σ: Importance of each concept

**Applications:**
- Document similarity (compare rows of UΣ)
- Term similarity (compare rows of VΣ)
- Semantic search (query as pseudo-document)
- Noise reduction (truncated SVD removes rare correlations)

---

## Part 10: The Pseudoinverse

### Motivation

For overdetermined systems Ax = b (more equations than unknowns), exact solution may not exist.

**Least squares:** Find x minimizing ‖Ax - b‖²

### The Moore-Penrose Pseudoinverse

$$A^+ = V \Sigma^+ U^T$$

where Σ⁺ is Σ with non-zero entries inverted (and transposed if non-square).

For σᵢ ≠ 0: (Σ⁺)ᵢᵢ = 1/σᵢ
For σᵢ = 0: (Σ⁺)ᵢᵢ = 0

### Properties

1. If A is invertible: A⁺ = A⁻¹
2. **x = A⁺b** is the least-squares solution
3. If multiple solutions exist, A⁺b has minimum norm

### Computing Least Squares via SVD

```python
# Solve Ax = b in least squares sense
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Pseudoinverse
S_inv = np.where(S > 1e-10, 1/S, 0)
x = Vt.T @ (S_inv * (U.T @ b))
```

---

## Part 11: Numerical Considerations

### Computing SVD

The standard algorithm has O(min(mn², m²n)) complexity.

For large sparse matrices:
- **Truncated SVD**: Only compute top k singular values
- **Randomized SVD**: Approximate SVD in O(mnk) time

### Numerical Stability

SVD is very stable:
- No explicit formation of AᵀA (which squares condition number)
- Iterative algorithms (like Golub-Kahan bidiagonalization)

### Condition Number

$$\kappa(A) = \frac{\sigma_1}{\sigma_r} = \frac{\text{largest singular value}}{\text{smallest non-zero singular value}}$$

Large condition number = ill-conditioned matrix = numerically sensitive.

---

## Summary

| Concept | Formula/Definition | ML Relevance |
|---------|-------------------|--------------|
| SVD | A = UΣVᵀ | Universal decomposition |
| Singular values | σᵢ = √(eigenvalue of AᵀA) | Feature importance |
| Rank-k approximation | Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ | Compression, denoising |
| Eckart-Young | Best low-rank approx | Optimal dimensionality reduction |
| Pseudoinverse | A⁺ = VΣ⁺Uᵀ | Least squares solution |
| PCA via SVD | V = principal components | Numerically stable PCA |

---

## Key Takeaways

1. **SVD works for ANY matrix** - Not just square, not just invertible
2. **Geometric meaning: Rotate → Scale → Rotate** - Three simple operations
3. **Singular values measure importance** - Ordered by how much they contribute
4. **Truncated SVD = best approximation** - Eckart-Young theorem
5. **Foundation of many ML methods** - PCA, LSA, recommender systems, compression
6. **More stable than eigendecomposition** - For PCA and least squares

---

*Next: Tutorial 07 - Orthogonality and Projections*
