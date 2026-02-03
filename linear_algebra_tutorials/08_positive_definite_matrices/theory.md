# Tutorial 08: Positive Definite Matrices

## Introduction: Why Positive Definiteness Matters

Positive definite matrices are the "nicest" matrices in linear algebra. They have special properties that make them:
- Easy to work with numerically
- Geometrically meaningful
- Fundamental to optimization and statistics

**In ML, positive definite matrices appear as:**
- Covariance matrices (always positive semi-definite)
- Kernel matrices (Gram matrices)
- Hessians at minima (positive definite at local minima)
- Regularization terms (XᵀX + λI)

---

## Part 1: Definition and Intuition

### Definition

A symmetric matrix A is **positive definite** if for all non-zero vectors **x**:

$$\mathbf{x}^T A \mathbf{x} > 0$$

A symmetric matrix A is **positive semi-definite** if:

$$\mathbf{x}^T A \mathbf{x} \geq 0$$

### Intuition: Quadratic Forms

The expression **x**ᵀA**x** is called a **quadratic form**. It's a scalar function of **x**.

**For 2×2 matrix:**
$$\mathbf{x}^T A \mathbf{x} = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} a & b \\ b & c \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = ax_1^2 + 2bx_1x_2 + cx_2^2$$

**Positive definite** means this quadratic is always positive (except at **x** = **0**).

### Geometric Interpretation

The level sets {**x** : **x**ᵀA**x** = c} are:
- **Ellipses** for positive definite A
- **Hyperbolas** for indefinite A
- **Parallel lines** for positive semi-definite (degenerate)

**Positive definite = bowl-shaped function** with unique minimum at **0**.

---

## Part 2: Equivalent Conditions

A symmetric matrix A is positive definite **if and only if** any of these hold:

### 1. Eigenvalue Test
**All eigenvalues are positive: λᵢ > 0**

**Proof:** Using A = QΛQᵀ:
$$\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T Q\Lambda Q^T \mathbf{x} = \mathbf{y}^T \Lambda \mathbf{y} = \sum_i \lambda_i y_i^2$$

where **y** = Qᵀ**x**. This is positive iff all λᵢ > 0.

### 2. Pivots Test
**All pivots in Gaussian elimination are positive**

(No row exchanges needed for positive definite matrices!)

### 3. Determinants Test (Sylvester's Criterion)
**All leading principal minors are positive:**
- det(A₁) > 0 (upper-left 1×1)
- det(A₂) > 0 (upper-left 2×2)
- ... and so on

### 4. Cholesky Factorization Test
**A = LLᵀ exists with L lower triangular with positive diagonal**

### 5. Energy Test
**x**ᵀA**x** > 0 for all **x** ≠ **0** (the definition)

---

## Part 3: Cholesky Decomposition

### Definition

For positive definite A, there exists a unique lower triangular L with positive diagonal such that:

$$A = LL^T$$

This is the "square root" of a positive definite matrix!

### Derivation

Start with LDLᵀ factorization (like LU but symmetric):
$$A = LDL^T$$

where L is lower triangular with 1s on diagonal, D is diagonal.

For positive definite A, D has positive entries. Let D = D^{1/2}D^{1/2}.

Then: A = (LD^{1/2})(D^{1/2}Lᵀ) = L̃L̃ᵀ

where L̃ = LD^{1/2} is lower triangular with positive diagonal.

### Algorithm

For 3×3 matrix:
$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{12} & a_{22} & a_{23} \\ a_{13} & a_{23} & a_{33} \end{bmatrix} = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix} \begin{bmatrix} l_{11} & l_{21} & l_{31} \\ 0 & l_{22} & l_{32} \\ 0 & 0 & l_{33} \end{bmatrix}$$

Matching entries:
- $l_{11} = \sqrt{a_{11}}$
- $l_{21} = a_{12}/l_{11}$
- $l_{31} = a_{13}/l_{11}$
- $l_{22} = \sqrt{a_{22} - l_{21}^2}$
- $l_{32} = (a_{23} - l_{21}l_{31})/l_{22}$
- $l_{33} = \sqrt{a_{33} - l_{31}^2 - l_{32}^2}$

### Why Cholesky?

1. **Faster than LU:** Only n³/3 operations vs 2n³/3
2. **More stable:** No pivoting needed
3. **Tests positive definiteness:** If any square root fails, matrix isn't positive definite
4. **Useful for sampling:** Generate **x** ~ N(**0**, Σ) via **x** = L**z** where **z** ~ N(**0**, I)

---

## Part 4: Quadratic Forms Geometry

### The Quadratic Function

For positive definite A:
$$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$$

This is a **paraboloid** (bowl shape) in n+1 dimensions.

### Principal Axes

Using eigendecomposition A = QΛQᵀ:
$$f(\mathbf{x}) = \mathbf{y}^T \Lambda \mathbf{y} = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots$$

where **y** = Qᵀ**x**.

**In the eigenbasis:**
- Coordinate axes align with eigenvectors
- Curvature along eigenvector **qᵢ** is proportional to λᵢ
- Steep direction: largest eigenvalue
- Flat direction: smallest eigenvalue

### Level Sets (Contours)

For positive definite A, {**x** : **x**ᵀA**x** = c} is an ellipse/ellipsoid.

**Axes of ellipse:**
- Along eigenvector **qᵢ**
- Length proportional to 1/√λᵢ

**Condition number** κ(A) = λₘₐₓ/λₘᵢₙ determines ellipse "elongation."

---

## Part 5: ML Application - Covariance Matrices

### Why Covariance Matrices Are Positive Semi-Definite

For data matrix X (n samples × d features), centered:
$$\Sigma = \frac{1}{n-1}X^TX$$

For any **v**:
$$\mathbf{v}^T \Sigma \mathbf{v} = \frac{1}{n-1}\mathbf{v}^T X^T X \mathbf{v} = \frac{1}{n-1}\|X\mathbf{v}\|^2 \geq 0$$

So covariance matrices are **always positive semi-definite**.

If X has full column rank, Σ is **positive definite**.

### Interpretation

- **v**ᵀΣ**v** = variance of data projected onto direction **v**
- Eigenvalues = variances along principal components
- Eigenvectors = principal directions

### Mahalanobis Distance

The "natural" distance for a distribution with covariance Σ:
$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

This accounts for correlations and different scales.

---

## Part 6: ML Application - Optimization (Hessian)

### The Hessian Matrix

For a scalar function f(**x**), the Hessian is:
$$H = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

### Second-Order Conditions

At a critical point (∇f = **0**):
- H **positive definite** → **local minimum**
- H **negative definite** → **local maximum**
- H **indefinite** → **saddle point**
- H **semi-definite** → inconclusive (need higher-order terms)

### For Convex Optimization

f is **convex** ⟺ H(**x**) is positive semi-definite everywhere

**Examples:**
- f(**x**) = **x**ᵀA**x** has H = 2A, convex iff A positive semi-definite
- f(**x**) = log(1 + e^(**wᵀx**)) is convex (logistic loss)

### Condition Number and Optimization

For quadratic f(**x**) = **x**ᵀA**x** - **b**ᵀ**x**:

**Condition number** κ(A) = λₘₐₓ/λₘᵢₙ affects:
- Convergence rate of gradient descent: O(κ) iterations
- Optimal step size: 2/(λₘₐₓ + λₘᵢₙ)
- Preconditioning effectiveness

**Large κ = ill-conditioned = slow convergence**

---

## Part 7: ML Application - Gaussian Distributions

### Multivariate Gaussian PDF

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Why Σ must be positive definite:**
1. Need Σ⁻¹ to exist (positive definite → invertible)
2. Exponent must be ≤ 0 (Σ⁻¹ positive definite → quadratic form ≥ 0)
3. Normalization: need |Σ| > 0

### Sampling from N(**μ**, Σ)

Using Cholesky: Σ = LLᵀ

1. Sample **z** ~ N(**0**, I)
2. Compute **x** = **μ** + L**z**

Then **x** ~ N(**μ**, Σ) because:
- E[**x**] = **μ** + L·E[**z**] = **μ**
- Cov(**x**) = L·Cov(**z**)·Lᵀ = LILᵀ = LLᵀ = Σ

### Maximum Likelihood Estimation

Given data X, the MLE for covariance is:
$$\hat{\Sigma} = \frac{1}{n}X^TX$$

If n < d, this is singular (positive semi-definite but not positive definite).

**Solution:** Regularization
$$\hat{\Sigma}_{reg} = \frac{1}{n}X^TX + \lambda I$$

Adding λI makes it positive definite!

---

## Part 8: Regularization and Positive Definiteness

### Ridge Regression

Minimize: ‖X**w** - **y**‖² + λ‖**w**‖²

Normal equations: (XᵀX + λI)**w** = Xᵀ**y**

**Why add λI?**
- XᵀX might be singular or ill-conditioned
- XᵀX + λI is always positive definite for λ > 0
- Makes system solvable and numerically stable

### Kernel Methods

Kernel matrix K with Kᵢⱼ = k(**xᵢ**, **xⱼ**).

**Valid kernels produce positive semi-definite K** (Mercer's theorem).

Common kernels:
- Linear: k(**x**, **y**) = **x**ᵀ**y**
- RBF: k(**x**, **y**) = exp(-γ‖**x** - **y**‖²)
- Polynomial: k(**x**, **y**) = (**x**ᵀ**y** + c)^d

---

## Part 9: Matrix Square Roots

### Definition

For positive definite A, a matrix square root B satisfies:
$$B^2 = A$$

### Types of Square Roots

**1. Cholesky:** A = LLᵀ (L lower triangular)

**2. Symmetric square root:** A = A^{1/2}A^{1/2} where:
$$A^{1/2} = Q\Lambda^{1/2}Q^T$$

using eigendecomposition A = QΛQᵀ.

**3. Principal square root:** The unique positive definite square root.

### Applications

- **Whitening:** Transform data by Σ^{-1/2} to have identity covariance
- **Sampling:** Generate N(**0**, Σ) samples via Σ^{1/2}**z**
- **Metrics:** Define distance as ‖A^{1/2}(**x** - **y**)‖

---

## Part 10: Numerical Considerations

### Testing Positive Definiteness

**Method 1:** Compute eigenvalues (expensive, O(n³))
**Method 2:** Attempt Cholesky (cheaper, fails if not PD)
**Method 3:** Check leading minors (usually not recommended)

```python
# In practice:
try:
    L = np.linalg.cholesky(A)
    print("Positive definite!")
except np.linalg.LinAlgError:
    print("Not positive definite")
```

### Near Positive Definite

Sometimes A should be PD but isn't due to numerical errors.

**Fix:** Project to nearest PD matrix:
1. Compute eigendecomposition A = QΛQᵀ
2. Set negative eigenvalues to small positive value: Λ' = max(Λ, εI)
3. Reconstruct: A' = QΛ'Qᵀ

### Conditioning

For solving A**x** = **b** with PD matrix A:
- Cholesky is preferred over LU
- Condition number κ(A) = λₘₐₓ/λₘᵢₙ
- For covariance: adding λI improves conditioning

---

## Summary

| Concept | Definition/Property | ML Relevance |
|---------|-------------------|--------------|
| Positive definite | **x**ᵀA**x** > 0 for all **x** ≠ 0 | Covariance, Hessians, kernels |
| Eigenvalue test | All λᵢ > 0 | Checking definiteness |
| Cholesky | A = LLᵀ | Fast solving, sampling |
| Quadratic form | **x**ᵀA**x** | Loss functions, energy |
| Condition number | λₘₐₓ/λₘᵢₙ | Optimization convergence |
| Regularization | A + λI | Ensure positive definiteness |

---

## Key Takeaways

1. **Positive definite = all positive eigenvalues = bowl-shaped quadratic**
2. **Covariance matrices are positive semi-definite** - Always!
3. **Hessian PD at critical point → local minimum**
4. **Cholesky is the "square root"** - Key for sampling and solving
5. **Regularization ensures positive definiteness** - Ridge regression
6. **Condition number affects optimization** - Large κ = slow convergence

---

*This concludes the Linear Algebra for ML tutorial series!*
