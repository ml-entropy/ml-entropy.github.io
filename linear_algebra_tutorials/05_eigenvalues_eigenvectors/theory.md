# Tutorial 05: Eigenvalues and Eigenvectors

## Introduction: What Are Eigenvalues and Eigenvectors?

When a matrix transforms a vector, the vector typically changes both direction and magnitude. But some special vectors only get scaled—they keep pointing in the same direction. These are **eigenvectors**, and the scaling factors are **eigenvalues**.

**Definition:** For a square matrix A, a non-zero vector **v** is an eigenvector with eigenvalue λ if:

$$A\mathbf{v} = \lambda\mathbf{v}$$

**Intuition:** A acts like simple scalar multiplication on its eigenvectors.

### Why Do We Care?

**1. Understanding Transformations:**
- Eigenvectors are the "natural axes" of a transformation
- Eigenvalues tell us how much stretching/compression happens along each axis

**2. Simplifying Computation:**
- Matrix powers: $A^n \mathbf{v} = \lambda^n \mathbf{v}$
- If we know eigenvectors, we can diagonalize A and compute $A^{100}$ easily

**3. ML Applications:**
- PCA: Principal components are eigenvectors of covariance matrix
- PageRank: Website rankings are eigenvectors
- Stability analysis: Eigenvalues determine system behavior
- Spectral clustering: Uses eigenvectors of graph Laplacian

---

## Part 1: Geometric Interpretation

### What Eigenvectors "See"

Consider a 2×2 matrix A. Most vectors **v** get rotated and stretched to some new direction A**v**.

But eigenvectors are special:
- A**v** points in the **same direction** as **v** (or exactly opposite if λ < 0)
- The only change is magnitude: stretched by |λ| and flipped if λ < 0

```
Non-eigenvector:           Eigenvector:
   v ──A──> Av              v ──A──> λv
     ↗                        ↑
    /                         │
   /                          │
  v                           v
```

### Example: Scaling Matrix

$$A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$$

- **e₁** = [1, 0]ᵀ: A**e₁** = [2, 0]ᵀ = 2**e₁** → eigenvalue λ₁ = 2
- **e₂** = [0, 1]ᵀ: A**e₂** = [0, 3]ᵀ = 3**e₂** → eigenvalue λ₂ = 3

The standard basis vectors are eigenvectors! This happens for diagonal matrices.

### Example: Rotation Matrix

$$R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

For θ ≠ 0, π: No real eigenvectors! Every vector gets rotated.

(But there are complex eigenvectors: $e^{i\theta}$ and $e^{-i\theta}$)

### Example: Shear Matrix

$$S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

- **e₁** = [1, 0]ᵀ: S**e₁** = [1, 0]ᵀ = 1·**e₁** → eigenvalue λ = 1
- **e₂** = [0, 1]ᵀ: S**e₂** = [1, 1]ᵀ ≠ λ**e₂** → not an eigenvector!

Only one eigendirection for this matrix.

---

## Part 2: Finding Eigenvalues - The Characteristic Polynomial

### Derivation

Starting from A**v** = λ**v**:

$$A\mathbf{v} = \lambda\mathbf{v}$$
$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For a non-zero **v** to satisfy this, (A - λI) must be **singular**:

$$\det(A - \lambda I) = 0$$

This is called the **characteristic equation**, and the polynomial on the left is the **characteristic polynomial**.

### Example: 2×2 Matrix

$$A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$$

$$A - \lambda I = \begin{bmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda \end{bmatrix}$$

$$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2(1) = \lambda^2 - 7\lambda + 10$$

Setting to zero:
$$\lambda^2 - 7\lambda + 10 = 0$$
$$(\lambda - 5)(\lambda - 2) = 0$$

**Eigenvalues:** λ₁ = 5, λ₂ = 2

### Example: 3×3 Matrix

$$A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix}$$

$$\det(A - \lambda I) = (2-\lambda)[(2-\lambda)^2 - 1] - 1[(2-\lambda) - 0] + 0$$

After expansion:
$$-\lambda^3 + 6\lambda^2 - 10\lambda + 4 = 0$$

This gives eigenvalues λ = 2, 2 ± √2.

### General Properties of Characteristic Polynomial

For an n×n matrix:
- Degree n polynomial
- **Sum of eigenvalues = trace(A)** = sum of diagonal
- **Product of eigenvalues = det(A)**

These can be quick sanity checks!

---

## Part 3: Finding Eigenvectors

Once we have eigenvalues, we find eigenvectors by solving:

$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

This is the **null space** of (A - λI).

### Example Continued

For $A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$ with λ₁ = 5:

$$A - 5I = \begin{bmatrix} -1 & 2 \\ 1 & -2 \end{bmatrix}$$

Row reduce:
$$\begin{bmatrix} -1 & 2 \\ 0 & 0 \end{bmatrix}$$

From -v₁ + 2v₂ = 0: v₁ = 2v₂

Eigenvector: **v₁** = [2, 1]ᵀ (or any scalar multiple)

For λ₂ = 2:
$$A - 2I = \begin{bmatrix} 2 & 2 \\ 1 & 1 \end{bmatrix} \rightarrow \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$$

From v₁ + v₂ = 0: v₁ = -v₂

Eigenvector: **v₂** = [1, -1]ᵀ

### Verification

A**v₁** = $\begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}\begin{bmatrix} 2 \\ 1 \end{bmatrix}$ = $\begin{bmatrix} 10 \\ 5 \end{bmatrix}$ = 5$\begin{bmatrix} 2 \\ 1 \end{bmatrix}$ = 5**v₁** ✓

---

## Part 4: Diagonalization

### When A is Diagonalizable

If A has n linearly independent eigenvectors, we can write:

$$A = PDP^{-1}$$

where:
- P has eigenvectors as columns
- D is diagonal with eigenvalues on diagonal

### Why This is Powerful

**Matrix powers become trivial:**
$$A^k = PD^kP^{-1}$$

And D^k is easy—just raise each diagonal entry to the kth power!

### Derivation

If **v₁**, ..., **vₙ** are eigenvectors with eigenvalues λ₁, ..., λₙ:

$$AP = A[\mathbf{v}_1 | \cdots | \mathbf{v}_n] = [\lambda_1\mathbf{v}_1 | \cdots | \lambda_n\mathbf{v}_n]$$

$$= [\mathbf{v}_1 | \cdots | \mathbf{v}_n] \begin{bmatrix} \lambda_1 & & 0 \\ & \ddots & \\ 0 & & \lambda_n \end{bmatrix} = PD$$

Therefore: AP = PD, so **A = PDP⁻¹**

### Example

For $A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$:

$$P = \begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix}, \quad D = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

To compute A¹⁰⁰:
$$A^{100} = P \begin{bmatrix} 5^{100} & 0 \\ 0 & 2^{100} \end{bmatrix} P^{-1}$$

### When Diagonalization Fails

A matrix is NOT diagonalizable if:
- It doesn't have n independent eigenvectors
- This happens with "defective" matrices (algebraic multiplicity ≠ geometric multiplicity)

Example: $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ has λ = 1 (double) but only one eigenvector direction.

---

## Part 5: Special Cases and Properties

### Symmetric Matrices

For symmetric matrices (A = Aᵀ):
1. **All eigenvalues are real**
2. **Eigenvectors for different eigenvalues are orthogonal**
3. **Always diagonalizable** as A = QΛQᵀ where Q is orthogonal

**This is called the Spectral Theorem.**

### Complex Eigenvalues

For real matrices, complex eigenvalues come in conjugate pairs:
- If λ = a + bi is an eigenvalue, so is λ̄ = a - bi
- Corresponding eigenvectors are also conjugates

**Interpretation:** Complex eigenvalues indicate rotation in the transformation.

### Eigenvalue Shifting

If A has eigenvalue λ with eigenvector **v**:
- A + cI has eigenvalue λ + c with same eigenvector **v**
- cA has eigenvalue cλ with same eigenvector **v**
- A⁻¹ (if exists) has eigenvalue 1/λ with same eigenvector **v**

### Trace and Determinant

For any square matrix:
- **trace(A) = Σλᵢ** (sum of eigenvalues)
- **det(A) = Πλᵢ** (product of eigenvalues)

---

## Part 6: Power Iteration Algorithm

### The Problem

Finding eigenvalues exactly requires solving the characteristic polynomial, which is hard for large matrices.

**Power iteration** finds the **dominant eigenvalue** (largest |λ|) iteratively.

### Algorithm

```
1. Start with random vector x₀
2. Repeat:
   - y_{k+1} = A * x_k
   - x_{k+1} = y_{k+1} / ||y_{k+1}||
3. λ ≈ x_k^T * A * x_k (Rayleigh quotient)
```

### Why It Works

Let **v₁**, **v₂**, ..., **vₙ** be eigenvectors with |λ₁| > |λ₂| ≥ ... ≥ |λₙ|.

Any vector **x₀** can be written as:
$$\mathbf{x}_0 = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n$$

After k iterations:
$$A^k\mathbf{x}_0 = c_1\lambda_1^k\mathbf{v}_1 + c_2\lambda_2^k\mathbf{v}_2 + \cdots$$

$$= \lambda_1^k \left[ c_1\mathbf{v}_1 + c_2\left(\frac{\lambda_2}{\lambda_1}\right)^k\mathbf{v}_2 + \cdots \right]$$

Since |λ₂/λ₁| < 1, the terms with other eigenvalues decay exponentially!

$$A^k\mathbf{x}_0 \approx \lambda_1^k c_1 \mathbf{v}_1$$

The direction converges to **v₁**, and we can estimate λ₁.

### Convergence Rate

The convergence rate depends on |λ₂/λ₁|:
- If λ₁ = 5 and λ₂ = 2, ratio = 0.4 → fast convergence
- If λ₁ = 5 and λ₂ = 4.9, ratio = 0.98 → slow convergence

### Variants

- **Inverse iteration:** Apply power iteration to A⁻¹ to find smallest eigenvalue
- **Shifted inverse iteration:** Find eigenvalue closest to a given value
- **QR algorithm:** Finds all eigenvalues, used in practice

---

## Part 7: ML Applications

### Application 1: Principal Component Analysis (PCA)

PCA finds directions of maximum variance in data.

**The math:** Principal components are eigenvectors of the covariance matrix, ordered by eigenvalue magnitude.

**Covariance matrix:** For centered data X (n samples × d features):
$$\Sigma = \frac{1}{n-1}X^TX$$

**Steps:**
1. Center the data
2. Compute covariance matrix Σ
3. Find eigenvectors/eigenvalues of Σ
4. Sort by eigenvalue (largest = most variance)
5. Project data onto top k eigenvectors

**Why eigenvectors?** The eigenvector with largest eigenvalue maximizes:
$$\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \Sigma \mathbf{w} = \max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{w}^T X)$$

### Application 2: PageRank

Google's original algorithm for ranking web pages.

**Model:** Random surfer clicks links randomly. PageRank = probability of being on each page at equilibrium.

**Math:** If M is the link matrix (normalized), PageRank vector **r** satisfies:
$$M\mathbf{r} = \mathbf{r}$$

This is an eigenvector problem with λ = 1!

(Actually uses a modified matrix with "teleportation" to ensure unique solution)

### Application 3: Stability Analysis

For a linear dynamical system **x**_{t+1} = A**x**_t:

- **|λᵢ| < 1 for all i:** System converges to zero (stable)
- **|λᵢ| > 1 for some i:** System diverges (unstable)
- **|λᵢ| = 1:** Boundary case, needs careful analysis

For continuous systems **ẋ** = A**x**:
- **Re(λᵢ) < 0 for all i:** Stable
- **Re(λᵢ) > 0 for some i:** Unstable

### Application 4: Spectral Clustering

Uses eigenvectors of the **graph Laplacian** L = D - A:
- D = degree matrix (diagonal)
- A = adjacency matrix

The eigenvectors of L reveal cluster structure:
1. Build similarity graph
2. Compute Laplacian L
3. Find bottom k eigenvectors (smallest eigenvalues)
4. Cluster the rows of the eigenvector matrix

### Application 5: Matrix Factorization

In recommender systems, we approximate user-item matrix R:
$$R \approx UV^T$$

One approach: Use eigenvectors of RᵀR for U and RRᵀ for V (related to SVD, next tutorial).

---

## Part 8: Eigenvalues and Neural Networks

### Weight Matrix Analysis

For a linear layer y = Wx:
- Large eigenvalues of WᵀW can cause gradient explosion
- Small eigenvalues can cause gradient vanishing
- **Spectral normalization:** Divide W by its largest singular value

### Hessian Analysis

The Hessian matrix H = ∇²L of the loss function has eigenvalues that tell us about the loss landscape:
- All positive eigenvalues → local minimum
- All negative eigenvalues → local maximum
- Mixed signs → saddle point
- Large eigenvalue → steep direction → need small learning rate
- Small eigenvalue → flat direction

### Recurrent Neural Networks

In RNNs, hidden state evolves as **h**_t = f(W**h**_{t-1} + ...).

For the linearized version:
- Eigenvalues of W determine stability
- |λ| > 1 causes exploding gradients
- |λ| < 1 causes vanishing gradients

This motivates architectures like LSTM and GRU.

---

## Summary

| Concept | Definition/Formula | ML Relevance |
|---------|-------------------|--------------|
| Eigenvector | A**v** = λ**v** | Principal directions |
| Eigenvalue | Scaling factor λ | Variance, importance |
| Characteristic polynomial | det(A - λI) = 0 | Finding eigenvalues |
| Diagonalization | A = PDP⁻¹ | Efficient computation |
| Power iteration | Iterative dominant λ | PageRank, large matrices |
| Symmetric matrices | Real λ, orthogonal **v** | Covariance matrices |
| Spectral theorem | A = QΛQᵀ | PCA foundation |

---

## Key Takeaways

1. **Eigenvectors are "natural directions"** - The matrix acts simply along them
2. **Eigenvalues measure importance** - Larger |λ| = more influence
3. **Symmetric matrices are well-behaved** - Real eigenvalues, orthogonal eigenvectors
4. **Power iteration for large matrices** - Can't solve polynomial for n > 4
5. **Eigenvalues appear everywhere in ML** - PCA, PageRank, stability, optimization

---

*Next: Tutorial 06 - Singular Value Decomposition (SVD)*
