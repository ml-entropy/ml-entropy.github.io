# Solutions: Determinants

## Part A: Theory Solutions

### Solution A1: Basic 2×2 Determinants

**a)** $\det(A) = 3(4) - 2(1) = 12 - 2 = 10$

**b)** $\det(B) = 5(1) - (-3)(-2) = 5 - 6 = -1$

**c)** $\det(C) = 2(3) - 6(1) = 6 - 6 = 0$

**d)** det(C) = 0 means:
- C is **singular** (not invertible)
- The columns are **linearly dependent** ([2, 1]ᵀ and [6, 3]ᵀ = 3×[2, 1]ᵀ)
- C **collapses** 2D space to a 1D line
- The system Cx = b has either no solution or infinitely many

---

### Solution A2: 3×3 Determinants

**a)** Expand along row 1:
$$\det(A) = 1 \det\begin{bmatrix} 5 & 6 \\ 8 & 9 \end{bmatrix} - 2 \det\begin{bmatrix} 4 & 6 \\ 7 & 9 \end{bmatrix} + 3 \det\begin{bmatrix} 4 & 5 \\ 7 & 8 \end{bmatrix}$$

$$= 1(45-48) - 2(36-42) + 3(32-35)$$
$$= 1(-3) - 2(-6) + 3(-3)$$
$$= -3 + 12 - 9 = 0$$

**Note:** The rows are in arithmetic progression (row 2 = average of rows 1 and 3), so they're linearly dependent.

**b)** Expand along column 2 (has two zeros):
$$\det(B) = -0 + 3 \det\begin{bmatrix} 2 & 1 \\ 4 & 5 \end{bmatrix} - 0$$

$$= 3(10 - 4) = 3(6) = 18$$

---

### Solution A3: Geometric Interpretation

**a)** The parallelogram has vertices at (0,0), (2,1), (1,3), (3,4).

Two edge vectors from origin: **u** = [2, 1] and **v** = [1, 3]

Area = |det([[2, 1], [1, 3]])| = |2(3) - 1(1)| = |6 - 1| = **5**

So det(A) = ±5

**b)** To check orientation: The cross product u × v (embedded in 3D) gives:
[2, 1, 0] × [1, 3, 0] = [0, 0, 2(3) - 1(1)] = [0, 0, 5]

Positive z-component means counterclockwise orientation → **orientation preserved**

So **det(A) = +5**

**c)** Matrix A maps standard basis to columns:
- e₁ = [1, 0] → [2, 1]
- e₂ = [0, 1] → [1, 3]

So $A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$

Verify: det(A) = 2(3) - 1(1) = 5 ✓

---

### Solution A4: Row Operations

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{bmatrix}$$

**Step 1:** R2 ← R2 - 4R1 (det unchanged)
$$\begin{bmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 7 & 8 & 10 \end{bmatrix}$$

**Step 2:** R3 ← R3 - 7R1 (det unchanged)
$$\begin{bmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & -6 & -11 \end{bmatrix}$$

**Step 3:** R3 ← R3 - 2R2 (det unchanged)
$$\begin{bmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & 0 & 1 \end{bmatrix}$$

Now upper triangular!

**det(A)** = 1 × (-3) × 1 = **-3**

---

### Solution A5: Determinant Properties

**a)** Row 2 = 2 × Row 1, so rows are linearly dependent.
**det(A) = 0**

**b)** Lower triangular matrix! Determinant = product of diagonal.
**det(A) = 1 × 3 × 6 × 10 = 180**

**c)** If B is 3×3, then 3B has each row multiplied by 3.
By multilinearity: det(3B) = 3³ det(B) = 27 × 5 = **135**

---

### Solution A6: Product Rule

Given: det(A) = 2, det(B) = -3

**a)** det(AB) = det(A)det(B) = 2(-3) = **-6**

**b)** det(A²B) = det(A)² det(B) = 4(-3) = **-12**

**c)** det(A⁻¹) = 1/det(A) = **1/2**

**d)** det(AᵀB⁻¹) = det(Aᵀ)det(B⁻¹) = det(A) × (1/det(B)) = 2 × (-1/3) = **-2/3**

---

### Solution A7: Jacobian Calculation

**a)** Polar coordinates: x = r cos(θ), y = r sin(θ)

$$J = \begin{bmatrix} \partial x/\partial r & \partial x/\partial \theta \\ \partial y/\partial r & \partial y/\partial \theta \end{bmatrix} = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}$$

$$|J| = r\cos^2\theta + r\sin^2\theta = r(\cos^2\theta + \sin^2\theta) = \mathbf{r}$$

**b)** Scaling: u = 2x, v = 3y

$$J = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$$

$$|J| = 2 × 3 = \mathbf{6}$$

**c)** Shear: u = x + 2y, v = y

$$J = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$$

$$|J| = 1 × 1 - 2 × 0 = \mathbf{1}$$

(Shear preserves area!)

---

### Solution A8: Cofactor Matrix Derivation

**a)** For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

Cofactors:
- C₁₁ = (+1) × det([d]) = d
- C₁₂ = (-1) × det([c]) = -c
- C₂₁ = (-1) × det([b]) = -b
- C₂₂ = (+1) × det([a]) = a

$$C = \begin{bmatrix} d & -c \\ -b & a \end{bmatrix}$$

**b)** Compute $A \cdot C^T$:

$$A \cdot C^T = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

$$= \begin{bmatrix} ad - bc & -ab + ba \\ cd - dc & -cb + ad \end{bmatrix} = \begin{bmatrix} ad-bc & 0 \\ 0 & ad-bc \end{bmatrix}$$

$$= (ad - bc) I = \det(A) \cdot I$$

**c)** Pattern: $A \cdot C^T = \det(A) \cdot I$

**d)** Therefore: $A^{-1} = \frac{1}{\det(A)} C^T = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

---

### Solution A9: Block Determinant

**a)** For $M = \begin{bmatrix} A & B \\ 0 & D \end{bmatrix}$:

Use cofactor expansion along the bottom rows. The zeros in the bottom-left make most terms vanish.

More directly: 
$$M = \begin{bmatrix} A & 0 \\ 0 & I \end{bmatrix} \begin{bmatrix} I & A^{-1}B \\ 0 & D \end{bmatrix}$$

(if A invertible)

det of first factor = det(A) (block diagonal)
det of second factor = det(D) (upper triangular)

So det(M) = det(A) det(D) ∎

**b)** For $\begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ (zeros in upper-right):

Same result! det = det(A) det(D)

**c)** For general $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$:

If A is invertible: det(M) = det(A) det(D - CA⁻¹B)

This is the **Schur complement** formula. The expression D - CA⁻¹B is called the Schur complement of A.

---

### Solution A10: RealNVP Jacobian

Transformation: y₁ = x₁, y₂ = x₂ · exp(s(x₁)) + t(x₁)

**a)** Jacobian matrix:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix} \partial y_1/\partial x_1 & \partial y_1/\partial x_2 \\ \partial y_2/\partial x_1 & \partial y_2/\partial x_2 \end{bmatrix}$$

- ∂y₁/∂x₁ = 1
- ∂y₁/∂x₂ = 0
- ∂y₂/∂x₁ = x₂ · exp(s(x₁)) · s'(x₁) + t'(x₁) (complicated!)
- ∂y₂/∂x₂ = exp(s(x₁))

$$J = \begin{bmatrix} 1 & 0 \\ * & \exp(s(x_1)) \end{bmatrix}$$

**b)** Determinant:

Since J is lower triangular:
$$\det(J) = 1 × \exp(s(x_1)) = \exp(s(x_1))$$

And: **log |det(J)| = s(x₁)**

**c)** Why this is efficient:

1. **Triangular Jacobian** → det is just product of diagonal
2. **O(n) instead of O(n³)** for determinant computation
3. The complex ∂y₂/∂x₁ term **doesn't matter** for the determinant
4. s(x₁) can be a neural network - we only need its output, not its derivative for the determinant

---

## Part B: Coding Solutions

### Solution B1: Determinant from Scratch

```python
import numpy as np

def det_2x2(matrix):
    """Compute determinant of 2x2 matrix."""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    return a * d - b * c

def det_3x3(matrix):
    """Compute determinant of 3x3 matrix using cofactor expansion."""
    # Expand along first row
    det = 0
    for j in range(3):
        # Minor: delete row 0 and column j
        minor = np.delete(np.delete(matrix, 0, axis=0), j, axis=1)
        cofactor = ((-1) ** j) * det_2x2(minor)
        det += matrix[0, j] * cofactor
    return det

# Test
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
print(f"det(A) = {det_3x3(A)}")  # Should be -3
print(f"numpy det(A) = {np.linalg.det(A)}")  # Verify
```

---

### Solution B2: Visualize Area Scaling

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_determinant_as_area(A):
    """Visualize how a 2x2 matrix transforms the unit square."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Unit square vertices
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    
    # Transform
    transformed = square @ A.T
    
    det_A = np.linalg.det(A)
    
    # Plot original
    ax1 = axes[0]
    ax1.fill(square[:, 0], square[:, 1], alpha=0.3, color='blue')
    ax1.plot(square[:, 0], square[:, 1], 'b-', linewidth=2)
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    ax1.set_title('Original Unit Square\nArea = 1')
    ax1.grid(True, alpha=0.3)
    
    # Plot transformed
    ax2 = axes[1]
    color = 'green' if det_A > 0 else 'red'
    ax2.fill(transformed[:, 0], transformed[:, 1], alpha=0.3, color=color)
    ax2.plot(transformed[:, 0], transformed[:, 1], color=color, linewidth=2)
    
    # Show columns of A as vectors
    ax2.quiver(0, 0, A[0, 0], A[1, 0], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.02, label=f'col1 = [{A[0,0]:.1f}, {A[1,0]:.1f}]')
    ax2.quiver(0, 0, A[0, 1], A[1, 1], angles='xy', scale_units='xy', scale=1,
               color='orange', width=0.02, label=f'col2 = [{A[0,1]:.1f}, {A[1,1]:.1f}]')
    
    max_val = max(np.abs(transformed).max(), 1) * 1.3
    ax2.set_xlim(-max_val, max_val)
    ax2.set_ylim(-max_val, max_val)
    ax2.set_aspect('equal')
    
    orientation = "preserved" if det_A > 0 else "reversed"
    ax2.set_title(f'Transformed Parallelogram\nArea = |det(A)| = {abs(det_A):.2f}\nOrientation: {orientation}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Test
A = np.array([[2, 1], [0.5, 1.5]])
visualize_determinant_as_area(A)

# Test with negative determinant (reflection)
A_reflect = np.array([[-1, 0], [0, 1]])
visualize_determinant_as_area(A_reflect)
```

---

### Solution B3: Row Reduction Determinant

```python
def det_gaussian(matrix, verbose=False):
    """Compute determinant using Gaussian elimination."""
    A = matrix.astype(float).copy()
    n = A.shape[0]
    det_multiplier = 1
    
    for col in range(n):
        # Find pivot
        max_row = col + np.argmax(np.abs(A[col:, col]))
        
        if np.abs(A[max_row, col]) < 1e-10:
            return 0  # Singular matrix
        
        # Swap rows if needed
        if max_row != col:
            A[[col, max_row]] = A[[max_row, col]]
            det_multiplier *= -1
            if verbose:
                print(f"Swap rows {col} and {max_row}, det multiplier: {det_multiplier}")
        
        # Eliminate below
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row] -= factor * A[col]
            if verbose:
                print(f"R{row} -= {factor:.3f} * R{col}")
        
        if verbose:
            print(f"After column {col}:\n{A}\n")
    
    # Determinant is product of diagonal times accumulated multiplier
    det = det_multiplier * np.prod(np.diag(A))
    
    if verbose:
        print(f"Diagonal: {np.diag(A)}")
        print(f"Det multiplier: {det_multiplier}")
        print(f"Final det: {det}")
    
    return det

# Test
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
print("Computing det via Gaussian elimination:")
det_gauss = det_gaussian(A, verbose=True)
print(f"\nResult: {det_gauss}")
print(f"NumPy verification: {np.linalg.det(A)}")
```

---

### Solution B4: Jacobian Determinant

```python
def numerical_jacobian_det(f, x, epsilon=1e-5):
    """Numerically compute the Jacobian determinant at a point."""
    n = len(x)
    J = np.zeros((n, n))
    
    f_x = f(x)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += epsilon
        
        J[:, i] = (f(x_plus) - f_x) / epsilon
    
    return np.abs(np.linalg.det(J))

# Test with polar-to-cartesian
def polar_to_cartesian(polar):
    r, theta = polar
    return np.array([r * np.cos(theta), r * np.sin(theta)])

# At r=2, theta=pi/4
point = np.array([2.0, np.pi/4])
jac_det = numerical_jacobian_det(polar_to_cartesian, point)
print(f"Numerical Jacobian det at r={point[0]}, theta={point[1]:.3f}: {jac_det:.4f}")
print(f"Expected (= r): {point[0]}")
```

---

### Solution B5: Singularity Detection

```python
def analyze_matrix_singularity(A, tolerance=1e-10):
    """Analyze whether a matrix is singular or near-singular."""
    det = np.linalg.det(A)
    rank = np.linalg.matrix_rank(A, tol=tolerance)
    
    # Condition number
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] > tolerance:
        condition_number = s[0] / s[-1]
    else:
        condition_number = np.inf
    
    return {
        'determinant': det,
        'is_singular': np.abs(det) < tolerance,
        'condition_number': condition_number,
        'rank': rank
    }

# Test
A_good = np.array([[2, 1], [1, 3]])
A_singular = np.array([[2, 4], [1, 2]])
A_near_singular = np.array([[1, 2], [1.0001, 2.0002]])

for name, A in [('Good', A_good), ('Singular', A_singular), ('Near-singular', A_near_singular)]:
    result = analyze_matrix_singularity(A)
    print(f"\n{name} matrix:")
    for k, v in result.items():
        print(f"  {k}: {v}")
```

---

### Solution B6: Log-Determinant Computation

```python
def log_det_stable(A):
    """Compute log(|det(A)|) in a numerically stable way."""
    # Use LU decomposition with pivoting
    # P @ A = L @ U, so det(A) = det(P)^(-1) * det(L) * det(U)
    # det(P) = (-1)^(number of row swaps)
    # det(L) = 1 (unit diagonal)
    # det(U) = product of diagonal
    
    from scipy import linalg
    
    # LU decomposition
    P, L, U = linalg.lu(A)
    
    # Get diagonal of U
    diag_U = np.diag(U)
    
    # Handle zeros/negatives
    signs = np.sign(diag_U)
    log_abs_diag = np.log(np.abs(diag_U))
    
    # Sign from diagonal
    sign_from_diag = np.prod(signs)
    
    # Sign from permutation (number of row swaps)
    # P is a permutation matrix, det(P) = ±1
    sign_from_P = np.linalg.det(P)
    
    total_sign = sign_from_diag * sign_from_P
    log_abs_det = np.sum(log_abs_diag)
    
    return log_abs_det, total_sign

# Test
A = np.random.randn(5, 5)
log_abs, sign = log_det_stable(A)
print(f"log|det(A)| = {log_abs:.6f}")
print(f"sign = {sign}")
print(f"Reconstructed det = {sign * np.exp(log_abs):.6f}")
print(f"Direct det = {np.linalg.det(A):.6f}")
```

---

### Solution B7: Visualize Jacobian Field

```python
def visualize_jacobian_field(f, x_range, y_range, resolution=20):
    """Visualize how the Jacobian determinant varies across space."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    J_det = np.zeros_like(X)
    epsilon = 1e-5
    
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            
            # Compute Jacobian numerically
            f_point = f(point)
            J = np.zeros((2, 2))
            for k in range(2):
                point_plus = point.copy()
                point_plus[k] += epsilon
                J[:, k] = (f(point_plus) - f_point) / epsilon
            
            J_det[i, j] = np.abs(np.linalg.det(J))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(X, Y, J_det, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax, label='|det(Jacobian)|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Jacobian Determinant Field\n(Local Area Scaling Factor)')
    plt.show()

# Example: polar-like transformation
def transformation(xy):
    x, y = xy
    r = np.sqrt(x**2 + y**2) + 0.1  # avoid r=0
    theta = np.arctan2(y, x)
    return np.array([r * np.cos(2*theta), r * np.sin(2*theta)])

visualize_jacobian_field(transformation, (-2, 2), (-2, 2), resolution=30)
```

---

### Solution B8: Cofactor Matrix

```python
def cofactor_matrix(A):
    """Compute the cofactor matrix."""
    n = A.shape[0]
    C = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Minor: delete row i and column j
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            
            # Cofactor with sign
            C[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    
    return C

def matrix_inverse_via_cofactors(A):
    """Compute A^(-1) using cofactor formula."""
    det_A = np.linalg.det(A)
    
    if np.abs(det_A) < 1e-10:
        raise ValueError("Matrix is singular")
    
    C = cofactor_matrix(A)
    return C.T / det_A

# Test
A = np.array([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
print("Matrix A:")
print(A)

print("\nCofactor matrix C:")
C = cofactor_matrix(A)
print(C)

print("\nA^(-1) via cofactors:")
A_inv = matrix_inverse_via_cofactors(A)
print(A_inv)

print("\nVerification A @ A^(-1):")
print(A @ A_inv)
```

---

### Solution B9: Normalizing Flow Layer

```python
class AffineFlowLayer:
    """Simple affine normalizing flow layer."""
    
    def __init__(self, dim):
        self.log_scale = np.zeros(dim)
        self.shift = np.zeros(dim)
    
    def forward(self, x):
        """Transform x to y."""
        y = x * np.exp(self.log_scale) + self.shift
        # Jacobian is diagonal: diag(exp(log_scale))
        # log|det(J)| = sum(log_scale)
        log_det_jacobian = np.sum(self.log_scale)
        return y, log_det_jacobian
    
    def inverse(self, y):
        """Transform y back to x."""
        x = (y - self.shift) * np.exp(-self.log_scale)
        # Jacobian is diagonal: diag(exp(-log_scale))
        # log|det(J)| = -sum(log_scale)
        log_det_jacobian = -np.sum(self.log_scale)
        return x, log_det_jacobian

# Test
layer = AffineFlowLayer(3)
layer.log_scale = np.array([0.5, -0.3, 0.1])
layer.shift = np.array([1.0, 2.0, 3.0])

x = np.array([1.0, 2.0, 3.0])
y, log_det_fwd = layer.forward(x)
x_back, log_det_inv = layer.inverse(y)

print(f"Original x: {x}")
print(f"Transformed y: {y}")
print(f"Recovered x: {x_back}")
print(f"log|det(J)| forward: {log_det_fwd}")
print(f"log|det(J)| inverse: {log_det_inv}")
print(f"Round-trip error: {np.max(np.abs(x - x_back))}")
```

---

### Solution B10: Gaussian Log-Likelihood

```python
def multivariate_gaussian_log_likelihood(X, mu, Sigma):
    """Compute log-likelihood under multivariate Gaussian."""
    N, D = X.shape
    
    # Compute log|Sigma| using Cholesky for stability
    L = np.linalg.cholesky(Sigma)
    log_det_Sigma = 2 * np.sum(np.log(np.diag(L)))
    
    # Solve for Sigma^(-1)(x - mu) using Cholesky
    # Sigma = L @ L.T, so Sigma^(-1) = L^(-T) @ L^(-1)
    diff = X - mu  # (N, D)
    
    # Solve L @ y = diff.T for y, then compute y.T @ y = diff @ Sigma^(-1) @ diff.T
    y = np.linalg.solve(L, diff.T)  # (D, N)
    mahalanobis_sq = np.sum(y**2, axis=0)  # (N,)
    
    # Log-likelihood
    log_likelihood = -0.5 * (D * np.log(2 * np.pi) + log_det_Sigma + mahalanobis_sq)
    
    return log_likelihood

# Test
np.random.seed(42)
D = 3
mu = np.array([1.0, 2.0, 3.0])
Sigma = np.array([[1.0, 0.3, 0.1],
                  [0.3, 1.5, 0.2],
                  [0.1, 0.2, 2.0]])

# Generate some data
N = 5
X = np.random.multivariate_normal(mu, Sigma, size=N)

log_lik = multivariate_gaussian_log_likelihood(X, mu, Sigma)
print("Log-likelihoods for each point:")
print(log_lik)

# Verify with scipy
from scipy.stats import multivariate_normal
rv = multivariate_normal(mu, Sigma)
print("\nScipy verification:")
print(rv.logpdf(X))
```

---

## Part C: Conceptual Answers

### Answer C1: Zero Determinant

Three interpretations of det(A) = 0:

1. **Geometric:** A collapses dimension. In 2D, the plane is squashed to a line (or point). Volume/area becomes zero.

2. **Algebraic:** The columns (or rows) are linearly dependent. At least one column is a linear combination of others.

3. **Systems perspective:** The equation Ax = b either has no solution or infinitely many solutions. There's no unique solution.

Bonus: det(A) = 0 means λ = 0 is an eigenvalue of A.

---

### Answer C2: Determinant vs. Matrix Norm

**What they measure:**
- **det(A):** Volume scaling factor; product of all eigenvalues
- **||A||:** Maximum stretching; largest singular value

**Example where ||A|| large but det(A) small:**

$$A = \begin{bmatrix} 1000 & 0 \\ 0 & 0.001 \end{bmatrix}$$

- ||A||₂ = 1000 (largest singular value)
- det(A) = 1000 × 0.001 = 1

Or even more extreme:
$$B = \begin{bmatrix} 1000 & 0 \\ 0 & 0 \end{bmatrix}$$

- ||B||₂ = 1000
- det(B) = 0

---

### Answer C3: Jacobian in Neural Networks

We don't typically compute Jacobian determinants in backprop because:

1. **Not needed for gradients:** Backprop computes Jacobian-vector products (Jᵀv), not the full Jacobian or its determinant

2. **Computational cost:** Determinant is O(n³), while Jacobian-vector product is O(n)

3. **Determinants not in the loss:** Standard losses (cross-entropy, MSE) don't involve determinants

**Exception:** Normalizing flows DO need Jacobian determinants for the change-of-variables formula, which is why they use specially structured transformations.

---

### Answer C4: Computational Complexity

**a) Cofactor expansion is O(n!):**
- Expanding an n×n determinant creates n terms
- Each term involves an (n-1)×(n-1) determinant
- T(n) = n × T(n-1) = n × (n-1) × ... × 1 = n!

**b) Gaussian elimination is O(n³):**
- n columns to process
- Each column: ~n rows to eliminate
- Each elimination: ~n operations
- Total: O(n³)

**c) Normalizing flows achieve O(n):**
- Design triangular Jacobians (determinant = product of diagonal)
- Or use autoregressive structure
- Or use specialized architectures (coupling layers)
- Diagonal products are O(n)

---

### Answer C5: Determinant and Eigenvalues

**a) det(A) = 0 means eigenvalue 0:**
$$\det(A) = \prod_{i=1}^n \lambda_i$$
If this product is 0, at least one λᵢ = 0.

**b) det(A) > 0 doesn't mean positive definite:**
Example: $A = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$
- det(A) = 1 > 0
- But eigenvalues are -1, -1 (negative!)
- Not positive definite

Positive definite requires ALL eigenvalues > 0, not just their product.

**c) Stable log|det(A)| for large matrices:**
$$\log|\det(A)| = \log\left|\prod_{i=1}^n \lambda_i\right| = \sum_{i=1}^n \log|\lambda_i|$$

Computing individual eigenvalues and summing logs avoids overflow/underflow that would occur computing the product directly.

In practice: Use LU decomposition or Cholesky (for positive definite) and sum log of diagonal elements.
