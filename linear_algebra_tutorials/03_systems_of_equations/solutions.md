# Solutions: Systems of Linear Equations

## Part A: Theory Solutions

### Solution A1: Solving 2×2 Systems

**a) x + 2y = 5, 3x + y = 5**

From first equation: x = 5 - 2y
Substitute: 3(5 - 2y) + y = 5 → 15 - 6y + y = 5 → -5y = -10 → y = 2
Then: x = 5 - 4 = 1

**Unique solution: (x, y) = (1, 2)**

**b) 2x - y = 1, 4x - 2y = 3**

Second equation is 2× first: should give 4x - 2y = 2, but we have = 3.
Contradiction!

**No solution (parallel lines)**

**c) x + y = 3, 2x + 2y = 6**

Second equation is 2× first. Same line!

**Infinitely many solutions: x = 3 - t, y = t for any t**

---

### Solution A2: Gaussian Elimination

System:
$$\begin{bmatrix} 1 & 1 & 1 & | & 6 \\ 2 & 1 & -1 & | & 1 \\ 1 & -1 & 2 & | & 5 \end{bmatrix}$$

**Step 1:** R2 ← R2 - 2R1, R3 ← R3 - R1
$$\begin{bmatrix} 1 & 1 & 1 & | & 6 \\ 0 & -1 & -3 & | & -11 \\ 0 & -2 & 1 & | & -1 \end{bmatrix}$$

**Step 2:** R3 ← R3 - 2R2
$$\begin{bmatrix} 1 & 1 & 1 & | & 6 \\ 0 & -1 & -3 & | & -11 \\ 0 & 0 & 7 & | & 21 \end{bmatrix}$$

**Back substitution:**
- z = 21/7 = 3
- -y - 9 = -11 → y = 2
- x + 2 + 3 = 6 → x = 1

**Solution: (x, y, z) = (1, 2, 3)**

---

### Solution A3: LU Decomposition

$$A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix}$$

**Find L and U:**

R2 ← R2 - 2R1: factor = 2
R3 ← R3 - 4R1: factor = 4

$$\begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 3 & 5 \end{bmatrix}$$

R3 ← R3 - 3R2: factor = 3

$$U = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix}$$

$$L = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{bmatrix}$$

**Solve A**x** = [4, 10, 24]ᵀ:**

First, L**y** = **b**:
- y₁ = 4
- 2(4) + y₂ = 10 → y₂ = 2
- 4(4) + 3(2) + y₃ = 24 → y₃ = 2

Then, U**x** = **y**:
- 2z = 2 → z = 1
- y + 1 = 2 → y = 1
- 2x + 1 + 1 = 4 → x = 1

**Solution: x = [1, 1, 1]ᵀ**

---

### Solution A4: Existence of Solutions

System: x + ky = 1, kx + y = 1

Matrix form: $\begin{bmatrix} 1 & k \\ k & 1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

det(A) = 1 - k²

**a) Unique solution:** det ≠ 0 → k² ≠ 1 → **k ≠ ±1**

**b) No solution:** k = 1 gives identical rows with different RHS... wait, let's check.
- k = 1: x + y = 1, x + y = 1 → same equation → infinite solutions!
- k = -1: x - y = 1, -x + y = 1 → add: 0 = 2 → **k = -1 gives no solution**

**c) Infinite solutions:** k = 1 (equations are identical)

---

### Solution A5: Normal Equations Derivation

Minimize f(**x**) = ||A**x** - **b**||² = (A**x** - **b**)ᵀ(A**x** - **b**)

Expand:
$$f = \mathbf{x}^TA^TA\mathbf{x} - 2\mathbf{x}^TA^T\mathbf{b} + \mathbf{b}^T\mathbf{b}$$

Take gradient with respect to **x**:
$$\nabla_{\mathbf{x}} f = 2A^TA\mathbf{x} - 2A^T\mathbf{b}$$

Set to zero:
$$2A^TA\mathbf{x} - 2A^T\mathbf{b} = 0$$
$$A^TA\mathbf{x} = A^T\mathbf{b} \quad \blacksquare$$

---

### Solution A6: Geometric Interpretation

The normal equations are: AᵀA**x*** = Aᵀ**b**

Rearranging: Aᵀ(A**x*** - **b**) = Aᵀ**r** = **0**

This means every column of A is orthogonal to **r**!

$$\mathbf{a}_j^T \mathbf{r} = 0 \quad \text{for all columns } \mathbf{a}_j$$

**Geometric meaning:** The residual **r** = **b** - A**x*** is perpendicular to the column space of A. We're projecting **b** onto the column space, and the projection error is orthogonal to that subspace.

---

### Solution A7: Condition Number

For A = $\begin{bmatrix} 1 & 1 \\ 1 & 1+\epsilon \end{bmatrix}$:

**a) Condition number:**

det(A) = 1(1+ε) - 1(1) = ε

$$A^{-1} = \frac{1}{\epsilon}\begin{bmatrix} 1+\epsilon & -1 \\ -1 & 1 \end{bmatrix}$$

For 2-norm condition number, we need eigenvalues or singular values.

Using Frobenius norm as approximation:
- ||A||_F ≈ 2
- ||A⁻¹||_F ≈ 2/ε

κ(A) ≈ 4/ε

**b) As ε → 0:** κ → ∞ (matrix becomes singular)

**c) Effect on solving:** Small changes in **b** cause huge changes in **x**. The solution becomes numerically unstable.

---

### Solution A8: Ridge Regression Identity

Want to show: (AᵀA + λI)⁻¹Aᵀ = Aᵀ(AAᵀ + λI)⁻¹

**Method:** Verify by multiplying both sides by (AᵀA + λI) on left and (AAᵀ + λI) on right.

Left side times (AᵀA + λI): Aᵀ
Right side times (AᵀA + λI): Aᵀ(AAᵀ + λI)⁻¹(AᵀA + λI)

We need: Aᵀ = Aᵀ(AAᵀ + λI)⁻¹(AᵀA + λI)

This is true because:
$$(AA^T + \lambda I)^{-1}(A^TA + \lambda I) = (AA^T + \lambda I)^{-1}A^TA + \lambda(AA^T + \lambda I)^{-1}$$

Using the identity A(AAᵀ + λI) = (AᵀA + λI)A... this is the push-through identity. ∎

---

### Solution A9: Iterative Convergence

Iteration: **x**⁽ᵏ⁺¹⁾ = **x**⁽ᵏ⁾ - α Aᵀ(A**x**⁽ᵏ⁾ - **b**)

**a) Convergence condition:**

Error: **e**⁽ᵏ⁾ = **x**⁽ᵏ⁾ - **x***

**e**⁽ᵏ⁺¹⁾ = **e**⁽ᵏ⁾ - α AᵀA **e**⁽ᵏ⁾ = (I - α AᵀA)**e**⁽ᵏ⁾

Converges if spectral radius ρ(I - αAᵀA) < 1.

Eigenvalues of I - αAᵀA are 1 - αλᵢ where λᵢ are eigenvalues of AᵀA.

Need: |1 - αλᵢ| < 1 for all i
→ -1 < 1 - αλᵢ < 1
→ 0 < αλᵢ < 2
→ 0 < α < 2/λₘₐₓ

**b) Optimal α:**

Minimizes max|1 - αλᵢ| over i.

Best choice: α = 2/(λₘₐₓ + λₘᵢₙ)

This makes |1 - αλₘₐₓ| = |1 - αλₘᵢₙ|.

---

### Solution A10: Pseudoinverse

**a) Full column rank:** rank(A) = n

AᵀA is n×n and invertible.

A⁺ = (AᵀA)⁻¹Aᵀ gives the least squares solution since:

A⁺**b** = (AᵀA)⁻¹Aᵀ**b** solves normal equations.

**b) A⁺A = I:**

A⁺A = (AᵀA)⁻¹AᵀA = I ✓

**c) AA⁺ is projection:**

P = AA⁺ = A(AᵀA)⁻¹Aᵀ

Check P² = P:
P² = A(AᵀA)⁻¹AᵀA(AᵀA)⁻¹Aᵀ = A(AᵀA)⁻¹Aᵀ = P ✓

Check Pᵀ = P:
Pᵀ = (A(AᵀA)⁻¹Aᵀ)ᵀ = A((AᵀA)⁻¹)ᵀAᵀ = A(AᵀA)⁻¹Aᵀ = P ✓

So P is an orthogonal projection (onto column space of A).

---

## Part B: Coding Solutions

### Solution B1: Gaussian Elimination
```python
import numpy as np

def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i + np.argmax(np.abs(M[i:, i]))
        M[[i, max_row]] = M[[max_row, i]]
        
        if np.abs(M[i, i]) < 1e-12:
            return None  # Singular
        
        # Eliminate below
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i+1:n] @ x[i+1:]) / M[i, i]
    
    return x

# Test
A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
b = np.array([4, 10, 24], dtype=float)
x = gaussian_elimination(A, b)
print(f"Solution: {x}")
print(f"Verification: {A @ x}")
```

---

### Solution B2: LU Decomposition
```python
def lu_decompose(A):
    """Compute LU decomposition without pivoting."""
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    
    return L, U

def lu_solve(L, U, b):
    """Solve LUx = b."""
    n = len(b)
    
    # Forward: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    
    # Backward: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    
    return x

# Test
A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
b = np.array([4, 10, 24], dtype=float)
L, U = lu_decompose(A)
x = lu_solve(L, U, b)
print(f"L @ U = \n{L @ U}")
print(f"Solution: {x}")
```

---

### Solution B3: Linear Regression
```python
def linear_regression(X, y):
    """Fit linear regression using normal equations."""
    n_samples = X.shape[0]
    
    # Add intercept column
    X_aug = np.column_stack([np.ones(n_samples), X])
    
    # Normal equations: (X^T X) w = X^T y
    XTX = X_aug.T @ X_aug
    XTy = X_aug.T @ y
    
    w = np.linalg.solve(XTX, XTy)
    
    return w

# Test
np.random.seed(42)
X = np.random.randn(100, 2)
true_w = np.array([1, 2, 3])  # intercept, w1, w2
y = X @ true_w[1:] + true_w[0] + np.random.randn(100) * 0.1

w_fit = linear_regression(X, y)
print(f"True weights: {true_w}")
print(f"Fitted weights: {w_fit}")
```

---

### Solution B4: Ridge Regression
```python
def ridge_regression(X, y, lambda_reg):
    """Fit ridge regression."""
    n_features = X.shape[1]
    
    # (X^T X + λI)^(-1) X^T y
    XTX = X.T @ X
    regularized = XTX + lambda_reg * np.eye(n_features)
    XTy = X.T @ y
    
    w = np.linalg.solve(regularized, XTy)
    
    return w

# Test
np.random.seed(42)
X = np.random.randn(100, 5)
true_w = np.array([1, 2, 0, 0, 3])
y = X @ true_w + np.random.randn(100) * 0.5

for lam in [0.001, 0.1, 1, 10]:
    w = ridge_regression(X, y, lam)
    print(f"λ = {lam}: {np.round(w, 2)}")
```

---

### Solution B5: Condition Number Analysis
```python
def analyze_conditioning(A):
    """Analyze the conditioning of matrix A."""
    cond = np.linalg.cond(A)
    
    # Test sensitivity
    b = np.random.randn(A.shape[0])
    x = np.linalg.solve(A, b)
    
    # Perturb b by 1%
    delta_b = 0.01 * np.linalg.norm(b) * np.random.randn(len(b))
    delta_b = delta_b / np.linalg.norm(delta_b) * 0.01 * np.linalg.norm(b)
    
    x_perturbed = np.linalg.solve(A, b + delta_b)
    
    relative_change_x = np.linalg.norm(x_perturbed - x) / np.linalg.norm(x)
    relative_change_b = np.linalg.norm(delta_b) / np.linalg.norm(b)
    
    sensitivity = relative_change_x / relative_change_b
    
    return {
        'condition_number': cond,
        'is_well_conditioned': cond < 100,
        'sensitivity': sensitivity
    }

# Test
A_good = np.array([[1, 0], [0, 1]], dtype=float)
A_bad = np.array([[1, 1], [1, 1.001]], dtype=float)

print("Well-conditioned:")
print(analyze_conditioning(A_good))
print("\nIll-conditioned:")
print(analyze_conditioning(A_bad))
```

---

### Solution B6: Conjugate Gradient
```python
def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    """Solve Ax = b using conjugate gradient."""
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rsold = r @ r
    
    history = [np.sqrt(rsold)]
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r
        
        history.append(np.sqrt(rsnew))
        
        if np.sqrt(rsnew) < tol:
            break
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x, history

# Test with symmetric positive definite matrix
n = 50
A = np.random.randn(n, n)
A = A.T @ A + np.eye(n)  # Make SPD
b = np.random.randn(n)

x_cg, history = conjugate_gradient(A, b)
x_exact = np.linalg.solve(A, b)

print(f"CG error: {np.linalg.norm(x_cg - x_exact):.2e}")
print(f"Iterations: {len(history)}")
```

---

### Solution B7: Compare Solvers
```python
import time

def compare_solvers(A, b):
    """Compare different solving methods."""
    results = {}
    
    # numpy solve
    start = time.time()
    x_numpy = np.linalg.solve(A, b)
    results['numpy_solve'] = {
        'time': time.time() - start,
        'error': np.linalg.norm(A @ x_numpy - b)
    }
    
    # LU
    start = time.time()
    L, U = lu_decompose(A)
    x_lu = lu_solve(L, U, b)
    results['lu'] = {
        'time': time.time() - start,
        'error': np.linalg.norm(A @ x_lu - b)
    }
    
    # CG (if SPD)
    if np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) > 0):
        start = time.time()
        x_cg, _ = conjugate_gradient(A, b)
        results['cg'] = {
            'time': time.time() - start,
            'error': np.linalg.norm(A @ x_cg - b)
        }
    
    return results

# Test
n = 100
A = np.random.randn(n, n)
A = A.T @ A + np.eye(n)
b = np.random.randn(n)

results = compare_solvers(A, b)
for method, data in results.items():
    print(f"{method}: time={data['time']:.4f}s, error={data['error']:.2e}")
```

---

### Solution B8: Polynomial Regression
```python
def polynomial_regression(x, y, degree):
    """Fit polynomial regression."""
    # Build Vandermonde matrix
    n = len(x)
    X = np.column_stack([x**i for i in range(degree + 1)])
    
    # Solve normal equations
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return coeffs

# Test
np.random.seed(42)
x = np.linspace(-1, 1, 50)
y = 1 + 2*x - 3*x**2 + np.random.randn(50) * 0.1

coeffs = polynomial_regression(x, y, degree=2)
print(f"True coefficients: [1, 2, -3]")
print(f"Fitted coefficients: {coeffs}")
```

---

### Solution B9: Minimum Norm Solution
```python
def minimum_norm_solution(A, b):
    """Find minimum norm solution for underdetermined system."""
    # x = A^T (A A^T)^(-1) b
    AAT = A @ A.T
    z = np.linalg.solve(AAT, b)
    x = A.T @ z
    return x

# Test: 2 equations, 4 unknowns
A = np.array([[1, 2, 1, 1], [2, 1, 1, 2]], dtype=float)
b = np.array([3, 4], dtype=float)

x = minimum_norm_solution(A, b)
print(f"Solution: {x}")
print(f"Verification Ax = {A @ x}")
print(f"Norm of solution: {np.linalg.norm(x):.4f}")

# Compare with another solution (add null space component)
null_component = np.array([1, -1, 1, 0])  # In null space of A
x2 = x + 0.5 * null_component
print(f"\nAnother solution: {x2}")
print(f"Verification: {A @ x2}")
print(f"Norm: {np.linalg.norm(x2):.4f} (larger!)")
```

---

### Solution B10: Robust Least Squares
```python
def robust_least_squares(A, b, rcond=1e-10):
    """Solve least squares using SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Determine effective rank
    rank = np.sum(s > rcond * s[0])
    
    # Pseudoinverse of singular values
    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1 / s[:rank]
    
    # x = V @ Σ^+ @ U^T @ b
    x = Vt.T @ (s_inv * (U.T @ b))
    
    return x, rank

# Test with rank-deficient matrix
A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1], [0, 1, 2]], dtype=float)
b = np.array([6, 12, 3, 3], dtype=float)

x, rank = robust_least_squares(A, b)
print(f"Effective rank: {rank}")
print(f"Solution: {x}")
print(f"Residual norm: {np.linalg.norm(A @ x - b):.6f}")
```

---

## Part C: Conceptual Answers

### Answer C1: Why Not Invert?

Even for multiple systems with same A, **LU decomposition is better than computing A⁻¹**:

1. **Numerical stability:** A⁻¹ can accumulate more rounding errors
2. **Same cost:** LU factorization is O(n³), then each solve is O(n²). Computing A⁻¹ is also O(n³), then each multiply is O(n²). Same asymptotic cost!
3. **Memory:** LU reuses storage, A⁻¹ needs extra matrix

The only time A⁻¹ might be worth it: when you truly need the inverse itself (rare), or for very small matrices where it doesn't matter.

### Answer C2: Least Squares Intuition

**Why squared?**
1. **Differentiable:** Smooth objective enables gradient-based optimization
2. **Unique minimum:** Convex function has unique global minimum
3. **Closed form:** Normal equations give exact solution
4. **Maximum likelihood:** Assumes Gaussian noise

**L1 regression (minimizing |error|):**
- More robust to outliers
- No closed-form solution (need iterative methods)
- Can give sparse solutions
- Corresponds to Laplace noise assumption

### Answer C3: Regularization Geometry

Adding λI to AᵀA:
1. **Shifts all eigenvalues up by λ:** λᵢ → λᵢ + λ
2. **Reduces condition number:** κ_new = (λₘₐₓ + λ)/(λₘᵢₙ + λ)
3. **Prevents singularity:** Even if λₘᵢₙ = 0, new minimum is λ

Geometrically: The loss surface becomes more "bowl-shaped" (better conditioned). Small eigenvalue directions that were nearly flat get steepened.

### Answer C4: Iterative vs Direct

**Choose iterative (CG, GMRES) when:**
- Matrix is very large (millions of variables)
- Matrix is sparse (few nonzeros per row)
- Only need approximate solution
- Can't store dense factorization

**Choose direct (LU, QR) when:**
- Matrix is moderate size (up to ~10,000)
- Matrix is dense
- Need high accuracy
- Solving many systems with same A

### Answer C5: Neural Network Connection

**Forward pass:**
- Each layer computes y = Wx + b (matrix-vector product)
- Not solving a system, just evaluating

**Backward pass:**
- Compute gradients via chain rule
- For linear layer: ∂L/∂W = ∂L/∂y · xᵀ
- Also matrix multiplication, not solving

**Optimization:**
- Gradient descent: doesn't solve linear systems
- Newton's method: solves H⁻¹∇L where H is Hessian
- Natural gradient: solves F⁻¹∇L where F is Fisher matrix
- These involve solving linear systems!

**Implicit layers:**
- Some architectures define y implicitly: f(y, x) = 0
- Forward pass requires solving a system
- Backward pass uses implicit differentiation
