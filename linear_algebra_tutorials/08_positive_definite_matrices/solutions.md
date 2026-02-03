# Solutions: Positive Definite Matrices

## Part A: Theory Solutions

### Solution A1: Checking Positive Definiteness

**a)** $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$

Using eigenvalue test:
det(A - λI) = (2-λ)² - 1 = λ² - 4λ + 3 = (λ-3)(λ-1)

Eigenvalues: λ₁ = 3, λ₂ = 1 (both positive)

**A is positive definite** ✓

**b)** $B = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$

det(B - λI) = (1-λ)² - 4 = λ² - 2λ - 3 = (λ-3)(λ+1)

Eigenvalues: λ₁ = 3, λ₂ = -1 (one negative!)

**B is neither PD nor PSD** (indefinite)

**c)** $C = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$

det(C) = 4(1) - 2(2) = 0

One eigenvalue is 0 (since det = product of eigenvalues).

trace(C) = 5 = λ₁ + λ₂, so λ₁ = 5, λ₂ = 0

**C is positive semi-definite** (not strictly PD)

**d)** $D = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$

Eigenvalues are diagonal entries: λ = 1, 0, 1

All eigenvalues ≥ 0 but one equals 0.

**D is positive semi-definite**

---

### Solution A2: Eigenvalue Test

Already computed in A1:

a) λ = 3, 1 → **PD**
b) λ = 3, -1 → **Indefinite**
c) λ = 5, 0 → **PSD**
d) λ = 1, 0, 1 → **PSD**

---

### Solution A3: Sylvester's Criterion

**a)** $A = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$

Leading principal minors:
- M₁ = 3 > 0 ✓
- M₂ = det(A) = 3(2) - 1(1) = 5 > 0 ✓

All positive → **A is positive definite**

**b)** $B = \begin{bmatrix} 2 & 2 & 0 \\ 2 & 5 & 3 \\ 0 & 3 & 4 \end{bmatrix}$

Leading principal minors:
- M₁ = 2 > 0 ✓
- M₂ = det([[2, 2], [2, 5]]) = 10 - 4 = 6 > 0 ✓
- M₃ = det(B) = 2(5·4 - 3·3) - 2(2·4 - 3·0) + 0 = 2(11) - 2(8) = 6 > 0 ✓

All positive → **B is positive definite**

---

### Solution A4: Cholesky Decomposition

$$A = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix} = \begin{bmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{bmatrix}\begin{bmatrix} l_{11} & l_{21} \\ 0 & l_{22} \end{bmatrix}$$

Matching entries:

Position (1,1): $l_{11}^2 = 4$ → $l_{11} = 2$

Position (2,1): $l_{21}l_{11} = 2$ → $l_{21} = 2/2 = 1$

Position (2,2): $l_{21}^2 + l_{22}^2 = 5$ → $1 + l_{22}^2 = 5$ → $l_{22} = 2$

**Result:**
$$L = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}$$

**Verify:** $LL^T = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix} = A$ ✓

---

### Solution A5: 3×3 Cholesky

$$A = \begin{bmatrix} 4 & 2 & -2 \\ 2 & 5 & -4 \\ -2 & -4 & 14 \end{bmatrix}$$

**Row 1:**
- $l_{11} = \sqrt{4} = 2$

**Row 2:**
- $l_{21} = a_{21}/l_{11} = 2/2 = 1$
- $l_{22} = \sqrt{a_{22} - l_{21}^2} = \sqrt{5 - 1} = 2$

**Row 3:**
- $l_{31} = a_{31}/l_{11} = -2/2 = -1$
- $l_{32} = (a_{32} - l_{31}l_{21})/l_{22} = (-4 - (-1)(1))/2 = -3/2$
- $l_{33} = \sqrt{a_{33} - l_{31}^2 - l_{32}^2} = \sqrt{14 - 1 - 9/4} = \sqrt{51/4} = \sqrt{51}/2$

**Result:**
$$L = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ -1 & -3/2 & \sqrt{51}/2 \end{bmatrix}$$

---

### Solution A6: Quadratic Form Analysis

**a) Explicit form:**
$$f(x_1, x_2) = \begin{bmatrix} x_1 & x_2 \end{bmatrix}\begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

$$= 3x_1^2 + 2x_1x_2 + 2x_2^2$$

**b) Eigenvalues/eigenvectors:**

det(A - λI) = (3-λ)(2-λ) - 1 = λ² - 5λ + 5 = 0

λ = (5 ± √5)/2, so **λ₁ ≈ 3.618, λ₂ ≈ 1.382**

For λ₁: eigenvector proportional to [1, (λ₁-3)]ᵀ = [1, (√5-1)/2]ᵀ ≈ [1, 0.618]ᵀ

For λ₂: eigenvector proportional to [1, -λ₂+2]ᵀ ≈ [1, -1.618]ᵀ

Normalized: 
- **v₁** ≈ [0.851, 0.526]ᵀ
- **v₂** ≈ [0.526, -0.851]ᵀ

**c) Diagonalized form:**

In eigenbasis **y** = Qᵀ**x**:
$$f = \lambda_1 y_1^2 + \lambda_2 y_2^2 \approx 3.618 y_1^2 + 1.382 y_2^2$$

**d) Level curves:** Ellipse with axes along eigenvectors, with semi-axis lengths 1/√λᵢ.

---

### Solution A7: Covariance Matrix Properties

$$X = \begin{bmatrix} 1 & 2 \\ -1 & 0 \\ 0 & -2 \end{bmatrix}$$

**a) Sample covariance:**

$$X^TX = \begin{bmatrix} 1 & -1 & 0 \\ 2 & 0 & -2 \end{bmatrix}\begin{bmatrix} 1 & 2 \\ -1 & 0 \\ 0 & -2 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 8 \end{bmatrix}$$

$$\Sigma = \frac{1}{n-1}X^TX = \frac{1}{2}\begin{bmatrix} 2 & 2 \\ 2 & 8 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 4 \end{bmatrix}$$

**b) Positive semi-definiteness:**

For any **v**: **v**ᵀΣ**v** = **v**ᵀXᵀX**v**/(n-1) = ‖X**v**‖²/(n-1) ≥ 0 ✓

Eigenvalues of Σ: det(Σ - λI) = (1-λ)(4-λ) - 1 = λ² - 5λ + 3 = 0
λ = (5 ± √13)/2 ≈ 4.30, 0.70 (both positive)

**Σ is positive definite** (actually strictly PD since n > d)

**c) Rank:** rank(Σ) = rank(XᵀX) = rank(X) = min(n, d) = 2

Since n = 3 > d = 2 and columns of X are linearly independent, rank = 2 = full rank.

---

### Solution A8: Matrix Square Root

$$A = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

**a) Eigendecomposition:**

det(A - λI) = (5-λ)(2-λ) - 4 = λ² - 7λ + 6 = (λ-6)(λ-1)

λ₁ = 6, λ₂ = 1

For λ = 6: (A - 6I)**v** = [[-1, 2], [2, -4]]**v** = 0 → **v₁** = [2, 1]ᵀ/√5
For λ = 1: (A - I)**v** = [[4, 2], [2, 1]]**v** = 0 → **v₂** = [1, -2]ᵀ/√5

$$Q = \frac{1}{\sqrt{5}}\begin{bmatrix} 2 & 1 \\ 1 & -2 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 6 & 0 \\ 0 & 1 \end{bmatrix}$$

**b) Square root:**

$$\Lambda^{1/2} = \begin{bmatrix} \sqrt{6} & 0 \\ 0 & 1 \end{bmatrix}$$

$$A^{1/2} = Q\Lambda^{1/2}Q^T = \frac{1}{5}\begin{bmatrix} 2 & 1 \\ 1 & -2 \end{bmatrix}\begin{bmatrix} \sqrt{6} & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 1 & -2 \end{bmatrix}$$

After calculation:
$$A^{1/2} = \frac{1}{5}\begin{bmatrix} 4\sqrt{6}+1 & 2\sqrt{6}-2 \\ 2\sqrt{6}-2 & \sqrt{6}+4 \end{bmatrix}$$

**c) Verification:** (A^{1/2})² = Q Λ^{1/2} Qᵀ Q Λ^{1/2} Qᵀ = Q Λ Qᵀ = A ✓

---

### Solution A9: Completing the Square

$$f(x_1, x_2) = 2x_1^2 + 4x_1x_2 + 5x_2^2$$

**a) Matrix form:**

$$A = \begin{bmatrix} 2 & 2 \\ 2 & 5 \end{bmatrix}$$

(Note: off-diagonal is half the cross-term coefficient for symmetric A)

**b) Complete the square:**

$$f = 2x_1^2 + 4x_1x_2 + 5x_2^2 = 2(x_1 + x_2)^2 + 5x_2^2 - 2x_2^2 = 2(x_1 + x_2)^2 + 3x_2^2$$

Since 2 > 0 and 3 > 0, f(**x**) ≥ 0 with equality only at **x** = **0**.

**c) Via Cholesky:**

A = LLᵀ where L = [[√2, 0], [√2, √3]]

Then Lᵀ**x** = [√2 x₁ + √2 x₂, √3 x₂]ᵀ

$$f(\mathbf{x}) = \|L^T\mathbf{x}\|^2 = 2(x_1 + x_2)^2 + 3x_2^2$$ ✓

---

### Solution A10: Optimization Perspective

$$g(\mathbf{x}) = \frac{1}{2}\mathbf{x}^TA\mathbf{x} - \mathbf{b}^T\mathbf{x} + c$$

**a) Gradient:**

$$\nabla g = \frac{1}{2}(A + A^T)\mathbf{x} - \mathbf{b} = A\mathbf{x} - \mathbf{b}$$

(since A is symmetric)

**b) Hessian:**

$$\nabla^2 g = A$$

**c) Minimum:**

Set ∇g = **0**: A**x** = **b** → **x*** = A⁻¹**b**

**d) Verify minimum:**

H = A is positive definite (given) → critical point is indeed a minimum.

**e) Uniqueness:**

- A positive definite → A is invertible → unique solution **x*** = A⁻¹**b**
- Positive definiteness means the function curves upward in all directions
- No other critical points exist (convex quadratic)

---

## Part B: Coding Solutions

### Solution B1: Positive Definiteness Checker

```python
import numpy as np

def check_positive_definite(A, method='eigenvalue', tol=1e-10):
    """Check positive definiteness using various methods."""
    A = np.array(A, dtype=float)
    
    # Ensure symmetric
    if not np.allclose(A, A.T, atol=tol):
        return {
            'is_positive_definite': False,
            'is_positive_semidefinite': False,
            'details': 'Matrix is not symmetric'
        }
    
    if method == 'eigenvalue':
        eigenvalues = np.linalg.eigvalsh(A)
        is_pd = np.all(eigenvalues > tol)
        is_psd = np.all(eigenvalues >= -tol)
        return {
            'is_positive_definite': is_pd,
            'is_positive_semidefinite': is_psd,
            'details': {'eigenvalues': eigenvalues}
        }
    
    elif method == 'cholesky':
        try:
            L = np.linalg.cholesky(A)
            return {
                'is_positive_definite': True,
                'is_positive_semidefinite': True,
                'details': {'cholesky_L': L}
            }
        except np.linalg.LinAlgError:
            # May still be PSD
            eigenvalues = np.linalg.eigvalsh(A)
            is_psd = np.all(eigenvalues >= -tol)
            return {
                'is_positive_definite': False,
                'is_positive_semidefinite': is_psd,
                'details': {'eigenvalues': eigenvalues}
            }
    
    elif method == 'sylvester':
        n = A.shape[0]
        minors = []
        for k in range(1, n + 1):
            minor = np.linalg.det(A[:k, :k])
            minors.append(minor)
        
        is_pd = all(m > tol for m in minors)
        return {
            'is_positive_definite': is_pd,
            'is_positive_semidefinite': is_pd,  # Sylvester only tests PD
            'details': {'principal_minors': minors}
        }

# Test
A = np.array([[2, 1], [1, 2]])
print("Testing A:")
for method in ['eigenvalue', 'cholesky', 'sylvester']:
    result = check_positive_definite(A, method=method)
    print(f"  {method}: PD={result['is_positive_definite']}")
```

---

### Solution B2: Cholesky Decomposition from Scratch

```python
def cholesky_decomposition(A):
    """Compute Cholesky decomposition A = L @ L.T."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal element
                sum_sq = sum(L[i, k]**2 for k in range(j))
                val = A[i, i] - sum_sq
                
                if val <= 0:
                    raise ValueError(f"Matrix is not positive definite (pivot {val} at position {i})")
                
                L[i, j] = np.sqrt(val)
            else:
                # Off-diagonal element
                sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    return L

# Test
A = np.array([[4, 2], [2, 5]])
L = cholesky_decomposition(A)
print("L:")
print(L)
print("\nVerification L @ L.T:")
print(L @ L.T)
print("\nNumPy Cholesky:")
print(np.linalg.cholesky(A))
```

---

### Solution B3: Solve Linear System via Cholesky

```python
def forward_substitution(L, b):
    """Solve L @ y = b where L is lower triangular."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y

def back_substitution(U, y):
    """Solve U @ x = y where U is upper triangular."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x

def solve_cholesky(A, b):
    """Solve Ax = b using Cholesky decomposition."""
    L = np.linalg.cholesky(A)
    
    # Solve L @ y = b
    y = forward_substitution(L, b)
    
    # Solve L.T @ x = y
    x = back_substitution(L.T, y)
    
    return x

# Test
A = np.array([[4, 2], [2, 5]])
b = np.array([4, 7])
x = solve_cholesky(A, b)
print(f"Solution x: {x}")
print(f"Verification A @ x: {A @ x}")
print(f"NumPy solution: {np.linalg.solve(A, b)}")
```

---

### Solution B4: Visualize Quadratic Forms

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_quadratic_form(A):
    """Visualize quadratic form f(x) = x^T A x."""
    A = np.array(A, dtype=float)
    
    fig = plt.figure(figsize=(14, 5))
    
    # Create grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute quadratic form
    Z = A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('f(x)')
    ax1.set_title(f'Quadratic Form Surface\nλ = {eigenvalues}')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Draw eigenvector directions
    scale = 1.5
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        ax2.arrow(0, 0, scale*vec[0], scale*vec[1], head_width=0.1, 
                  head_length=0.05, fc=f'C{i}', ec=f'C{i}', linewidth=2)
        ax2.text(scale*vec[0]*1.1, scale*vec[1]*1.1, f'λ={val:.2f}', fontsize=10)
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Contour Plot with Eigenvectors')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Test
A_pd = np.array([[3, 1], [1, 2]])  # Positive definite
visualize_quadratic_form(A_pd)
```

---

### Solution B5: Sample from Multivariate Gaussian

```python
def sample_multivariate_gaussian(mu, Sigma, n_samples):
    """Sample from N(mu, Sigma) using Cholesky."""
    mu = np.array(mu)
    Sigma = np.array(Sigma)
    d = len(mu)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(Sigma)
    
    # Sample from standard normal
    z = np.random.randn(n_samples, d)
    
    # Transform: x = mu + L @ z
    samples = mu + z @ L.T
    
    return samples

def visualize_gaussian_samples(mu, Sigma, n_samples=1000):
    """Visualize 2D Gaussian samples with covariance ellipse."""
    samples = sample_multivariate_gaussian(mu, Sigma, n_samples)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot samples
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)
    
    # Draw covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    theta = np.linspace(0, 2*np.pi, 100)
    for n_std in [1, 2, 3]:
        # Ellipse parametrization
        ellipse = np.array([np.cos(theta), np.sin(theta)])
        scaled = np.diag(n_std * np.sqrt(eigenvalues)) @ ellipse
        rotated = eigenvectors @ scaled
        shifted = rotated + np.array(mu).reshape(-1, 1)
        
        ax.plot(shifted[0], shifted[1], label=f'{n_std}σ', linewidth=2)
    
    ax.scatter([mu[0]], [mu[1]], color='red', s=100, marker='x', linewidths=3, label='Mean')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Multivariate Gaussian Samples')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# Test
mu = [1, 2]
Sigma = [[2, 0.8], [0.8, 1]]
visualize_gaussian_samples(mu, Sigma)
```

---

### Solution B6: Mahalanobis Distance

```python
def mahalanobis_distance(x, mu, Sigma):
    """Compute Mahalanobis distance."""
    x = np.atleast_2d(x)
    mu = np.array(mu)
    Sigma = np.array(Sigma)
    
    # Use Cholesky for efficient computation
    L = np.linalg.cholesky(Sigma)
    
    diff = x - mu
    # Solve L @ y = diff.T for y, then ||y|| = Mahalanobis distance
    y = np.linalg.solve(L, diff.T)
    
    distances = np.sqrt(np.sum(y**2, axis=0))
    
    return distances if len(distances) > 1 else distances[0]

def visualize_mahalanobis(mu, Sigma, points):
    """Visualize Mahalanobis distances."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw Mahalanobis distance contours
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    theta = np.linspace(0, 2*np.pi, 100)
    for d_M in [1, 2, 3]:
        ellipse = np.array([np.cos(theta), np.sin(theta)])
        scaled = np.diag(d_M * np.sqrt(eigenvalues)) @ ellipse
        rotated = eigenvectors @ scaled
        shifted = rotated + np.array(mu).reshape(-1, 1)
        ax.plot(shifted[0], shifted[1], '--', alpha=0.7, label=f'd_M = {d_M}')
    
    # Plot points with colors by Mahalanobis distance
    points = np.array(points)
    distances = mahalanobis_distance(points, mu, Sigma)
    
    scatter = ax.scatter(points[:, 0], points[:, 1], c=distances, 
                        cmap='coolwarm', s=100, edgecolors='black')
    plt.colorbar(scatter, label='Mahalanobis Distance')
    
    ax.scatter([mu[0]], [mu[1]], color='green', s=200, marker='*', label='Mean')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Mahalanobis Distance Visualization')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# Test
mu = [0, 0]
Sigma = [[2, 0.8], [0.8, 1]]
points = [[0, 0], [1, 1], [2, 0], [-1, 2], [1.5, -1]]
visualize_mahalanobis(mu, Sigma, points)
```

---

### Solution B7: Nearest Positive Definite Matrix

```python
def nearest_positive_definite(A, min_eigenvalue=1e-6):
    """Find nearest positive definite matrix."""
    A = np.array(A, dtype=float)
    
    # Symmetrize
    A = (A + A.T) / 2
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Clamp negative eigenvalues
    eigenvalues_fixed = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    A_pd = eigenvectors @ np.diag(eigenvalues_fixed) @ eigenvectors.T
    
    return A_pd

# Test
A_bad = np.array([[1, 2], [2, 1]])  # Has negative eigenvalue
print("Original eigenvalues:", np.linalg.eigvalsh(A_bad))

A_fixed = nearest_positive_definite(A_bad)
print("Fixed eigenvalues:", np.linalg.eigvalsh(A_fixed))
print("Distance:", np.linalg.norm(A_bad - A_fixed, 'fro'))
```

---

### Solution B8: Condition Number Analysis

```python
def analyze_condition_number(A):
    """Analyze condition number of positive definite matrix."""
    eigenvalues = np.linalg.eigvalsh(A)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    
    condition = lambda_max / lambda_min
    
    return {
        'condition_number': condition,
        'eigenvalues': eigenvalues,
        'is_well_conditioned': condition < 1e3,
        'log_condition': np.log10(condition)
    }

def visualize_condition_effect():
    """Show effect of condition number on optimization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    conditions = [1.5, 10, 100]
    
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    for ax, kappa in zip(axes, conditions):
        # Create matrix with given condition number
        A = np.array([[kappa, 0], [0, 1]])
        
        Z = A[0, 0] * X**2 + A[1, 1] * Y**2
        
        ax.contour(X, Y, Z, levels=20)
        ax.set_title(f'κ = {kappa}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        
        # Simulate gradient descent
        x_gd = [1.8, 1.8]
        lr = 0.1 / kappa
        path = [x_gd.copy()]
        for _ in range(20):
            grad = [2 * A[0, 0] * x_gd[0], 2 * A[1, 1] * x_gd[1]]
            x_gd = [x_gd[i] - lr * grad[i] for i in range(2)]
            path.append(x_gd.copy())
        
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=10, label='GD path')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Test
A = np.array([[10, 2], [2, 1]])
result = analyze_condition_number(A)
print(f"Condition number: {result['condition_number']:.2f}")
visualize_condition_effect()
```

---

### Solution B9: Log-Determinant Computation

```python
def log_determinant_cholesky(A):
    """Compute log(det(A)) using Cholesky."""
    L = np.linalg.cholesky(A)
    return 2 * np.sum(np.log(np.diag(L)))

def gaussian_log_likelihood(X, mu, Sigma):
    """Compute Gaussian log-likelihood."""
    X = np.atleast_2d(X)
    n, d = X.shape
    
    # Log-determinant
    log_det = log_determinant_cholesky(Sigma)
    
    # Solve L @ y = (x - mu).T for Mahalanobis
    L = np.linalg.cholesky(Sigma)
    diff = X - mu
    y = np.linalg.solve(L, diff.T)
    mahal_sq = np.sum(y**2, axis=0)
    
    # Log-likelihood
    log_lik = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal_sq)
    
    return np.sum(log_lik)

# Test
np.random.seed(42)
mu = np.array([0, 0])
Sigma = np.array([[2, 0.5], [0.5, 1]])
X = np.random.multivariate_normal(mu, Sigma, size=100)

log_lik = gaussian_log_likelihood(X, mu, Sigma)
print(f"Log-likelihood: {log_lik:.2f}")

# Compare with scipy
from scipy.stats import multivariate_normal
rv = multivariate_normal(mu, Sigma)
scipy_ll = np.sum(rv.logpdf(X))
print(f"Scipy log-likelihood: {scipy_ll:.2f}")
```

---

### Solution B10: Regularization for Positive Definiteness

```python
def regularize_covariance(S, method='ridge', param=None):
    """Regularize sample covariance matrix."""
    S = np.array(S, dtype=float)
    n = S.shape[0]
    
    eigenvalues_before = np.linalg.eigvalsh(S)
    cond_before = np.max(eigenvalues_before) / (np.min(np.abs(eigenvalues_before)) + 1e-10)
    
    if method == 'ridge':
        if param is None:
            # Auto: use smallest eigenvalue magnitude
            param = max(0.01 * np.max(eigenvalues_before), 
                       -np.min(eigenvalues_before) + 0.01)
        
        S_reg = S + param * np.eye(n)
    
    elif method == 'shrinkage':
        if param is None:
            param = 0.1  # Shrinkage factor
        
        target = np.trace(S) / n * np.eye(n)
        S_reg = (1 - param) * S + param * target
    
    eigenvalues_after = np.linalg.eigvalsh(S_reg)
    cond_after = np.max(eigenvalues_after) / np.min(eigenvalues_after)
    
    return {
        'regularized': S_reg,
        'param_used': param,
        'condition_before': cond_before,
        'condition_after': cond_after
    }

# Test with ill-conditioned matrix
np.random.seed(42)
# Create nearly singular covariance
A = np.random.randn(50, 2)  # Only 2 dimensions in 50-d space
S = A @ A.T / 50  # Rank 2

print("Original matrix:")
print(f"  Min eigenvalue: {np.min(np.linalg.eigvalsh(S)):.6f}")

for method in ['ridge', 'shrinkage']:
    result = regularize_covariance(S, method=method)
    print(f"\n{method} regularization:")
    print(f"  Parameter: {result['param_used']:.4f}")
    print(f"  Condition: {result['condition_before']:.1f} → {result['condition_after']:.1f}")
    print(f"  Min eigenvalue: {np.min(np.linalg.eigvalsh(result['regularized'])):.6f}")
```

---

## Part C: Conceptual Answers

### Answer C1: Why Covariance Matrices Are PSD

**Proof:**
For any vector **v**:
$$\mathbf{v}^T \Sigma \mathbf{v} = \mathbf{v}^T \frac{X^TX}{n-1} \mathbf{v} = \frac{1}{n-1} \mathbf{v}^T X^T X \mathbf{v} = \frac{1}{n-1} \|X\mathbf{v}\|^2 \geq 0$$

This is always non-negative, so Σ is PSD.

**Strictly PD when:**
- n > d (more samples than features)
- X has full column rank (features are not linearly dependent)
- No perfect multicollinearity in the data

---

### Answer C2: Cholesky vs LU

**Advantages of Cholesky:**

1. **Computational efficiency:** 
   - Cholesky: ~n³/3 operations
   - LU: ~2n³/3 operations
   - Cholesky is 2× faster!

2. **Numerical stability:**
   - No pivoting needed for PD matrices
   - Smaller intermediate values
   - More stable for ill-conditioned problems

3. **Storage:**
   - Only need to store L (half the matrix)
   - LU needs both L and U

4. **Guarantees:**
   - Cholesky exists iff matrix is PD
   - Can be used as a test for positive definiteness

---

### Answer C3: Hessian and Optimization

**a) PD Hessian → Local minimum:**
- At critical point where ∇f = 0
- Taylor expansion: f(x + h) ≈ f(x) + ½hᵀHh
- If H PD: hᵀHh > 0 for all h ≠ 0
- So f(x + h) > f(x) → local minimum

**b) PSD Hessian everywhere → Convex:**
- Function curves upward (or is flat) in all directions
- Any local minimum is global
- Line between any two points lies above function

**c) Condition number and convergence:**
- κ(H) = λ_max/λ_min
- Gradient descent converges in O(κ) iterations
- Large κ → slow convergence (need small step size for stability)
- Preconditioning: transform to reduce condition number

---

### Answer C4: Regularization Intuition

**a) Why λI makes matrix PD:**
- Eigenvalues of XᵀX + λI are λᵢ + λ (shifted by λ)
- If λ > 0, all eigenvalues become > λ > 0
- So the matrix is positive definite

**b) Effect on eigenvalues:**
- Small eigenvalues become relatively larger
- λ_min → λ_min + λ (significant increase)
- λ_max → λ_max + λ (small relative increase)
- Condition number decreases!

**c) Bias-variance tradeoff:**
- Without regularization: unbiased but high variance (overfitting)
- With regularization: introduces bias but reduces variance
- The bias: solution shrinks toward zero
- Optimal λ balances these effects

---

### Answer C5: Gaussian Distributions

**Why Σ must be PD:**

1. **Normalization:** The integral of PDF must equal 1
   - Need det(Σ) > 0 to have finite normalization constant
   - det(Σ) = 0 → integral is undefined

2. **Probability density:** Need exp(-½ **x**ᵀΣ⁻¹**x**) to decay
   - Need Σ⁻¹ to be PD so quadratic form is positive
   - Otherwise density could grow or be complex

3. **Well-defined distribution:**
   - Need Σ⁻¹ to exist (PD → invertible)
   - Eigenvalues are variances along principal axes
   - Zero variance → degenerate (not truly multivariate)
   - Negative "variance" → physically meaningless

**If not PD:**
- Can't compute Σ⁻¹ (singular case)
- Density might not integrate to 1
- Sampling doesn't make sense
- Numerical algorithms fail
