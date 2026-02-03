# Exercises: Systems of Linear Equations

## Part A: Theory Problems

### Exercise A1: Solving 2×2 Systems 🟢
Solve by hand using substitution or elimination:

a) x + 2y = 5, 3x + y = 5
b) 2x - y = 1, 4x - 2y = 3
c) x + y = 3, 2x + 2y = 6

Classify each as having unique, no, or infinite solutions.

---

### Exercise A2: Gaussian Elimination 🟢
Use Gaussian elimination to solve:
$$x + y + z = 6$$
$$2x + y - z = 1$$
$$x - y + 2z = 5$$

Show all row operations.

---

### Exercise A3: LU Decomposition 🟡
Find the LU decomposition of:
$$A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix}$$

Then use it to solve A**x** = [4, 10, 24]ᵀ.

---

### Exercise A4: Existence of Solutions 🟡
For what values of k does the system have:
a) A unique solution?
b) No solution?
c) Infinitely many solutions?

$$x + ky = 1$$
$$kx + y = 1$$

---

### Exercise A5: Normal Equations Derivation 🟡
Starting from minimizing ||A**x** - **b**||², derive the normal equations Aᵀ A**x** = Aᵀ**b**.

Show each step of taking the derivative and setting to zero.

---

### Exercise A6: Geometric Interpretation 🟡
Prove that the residual **r** = **b** - A**x*** (where **x*** is the least squares solution) is orthogonal to every column of A.

What does this mean geometrically?

---

### Exercise A7: Condition Number 🔴
For the matrix A = $\begin{bmatrix} 1 & 1 \\ 1 & 1+\epsilon \end{bmatrix}$:

a) Find the condition number as a function of ε
b) What happens as ε → 0?
c) How does this affect solving A**x** = **b**?

---

### Exercise A8: Ridge Regression Theory 🔴
Show that the ridge regression solution:
$$\mathbf{x}^* = (A^TA + \lambda I)^{-1}A^T\mathbf{b}$$

Can be written as:
$$\mathbf{x}^* = A^T(AA^T + \lambda I)^{-1}\mathbf{b}$$

*Hint: Use the Woodbury matrix identity or direct verification.*

---

### Exercise A9: Iterative Convergence 🔴
For the iteration **x**⁽ᵏ⁺¹⁾ = **x**⁽ᵏ⁾ - α(A**x**⁽ᵏ⁾ - **b**) (gradient descent for Ax = b):

a) Show this converges if 0 < α < 2/λₘₐₓ where λₘₐₓ is the largest eigenvalue of AᵀA
b) What is the optimal α?

---

### Exercise A10: Pseudoinverse 🔴
The Moore-Penrose pseudoinverse A⁺ gives the least squares solution: **x** = A⁺**b**.

a) For a full column rank matrix, show A⁺ = (AᵀA)⁻¹Aᵀ
b) Show that A⁺A = I (left inverse)
c) Show that AA⁺ is a projection matrix

---

## Part B: Coding Problems

### Exercise B1: Implement Gaussian Elimination 🟢
```python
def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: numpy array of shape (n, n)
        b: numpy array of shape (n,)
    
    Returns:
        x: solution vector of shape (n,)
        or None if system is singular
    """
    # Your code here
    pass
```

---

### Exercise B2: Implement LU Decomposition 🟡
```python
def lu_decompose(A):
    """
    Compute LU decomposition without pivoting.
    
    Args:
        A: numpy array of shape (n, n)
    
    Returns:
        L: lower triangular with 1s on diagonal
        U: upper triangular
    """
    # Your code here
    pass

def lu_solve(L, U, b):
    """
    Solve LUx = b using forward and back substitution.
    """
    # Your code here
    pass
```

---

### Exercise B3: Linear Regression from Scratch 🟢
```python
def linear_regression(X, y):
    """
    Fit linear regression using normal equations.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
    
    Returns:
        w: weights including intercept, shape (n_features + 1,)
    """
    # Your code here
    # Don't forget to add a column of ones for the intercept!
    pass
```

---

### Exercise B4: Ridge Regression 🟡
```python
def ridge_regression(X, y, lambda_reg):
    """
    Fit ridge regression.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        lambda_reg: regularization parameter
    
    Returns:
        w: weights (no intercept), shape (n_features,)
    """
    # Your code here
    # Solution: (X^T X + λI)^(-1) X^T y
    pass
```

---

### Exercise B5: Condition Number Analysis 🟡
```python
def analyze_conditioning(A):
    """
    Analyze the conditioning of matrix A.
    
    Args:
        A: numpy array (square matrix)
    
    Returns:
        dict with:
        - 'condition_number': float
        - 'is_well_conditioned': bool (cond < 100)
        - 'sensitivity': how much solution changes with 1% change in b
    """
    # Your code here
    pass
```

---

### Exercise B6: Iterative Solver 🟡
```python
def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    """
    Solve Ax = b using conjugate gradient method.
    A must be symmetric positive definite.
    
    Args:
        A: numpy array of shape (n, n), symmetric positive definite
        b: numpy array of shape (n,)
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        x: solution
        history: list of residual norms
    """
    # Your code here
    # Algorithm:
    # 1. r = b - Ax, p = r
    # 2. alpha = r^T r / (p^T A p)
    # 3. x = x + alpha * p
    # 4. r_new = r - alpha * A p
    # 5. beta = r_new^T r_new / r^T r
    # 6. p = r_new + beta * p
    pass
```

---

### Exercise B7: Compare Solvers 🟡
```python
def compare_solvers(A, b):
    """
    Compare different methods for solving Ax = b.
    
    Returns timing and accuracy for:
    - np.linalg.solve
    - LU decomposition
    - Conjugate gradient (if A is symmetric positive definite)
    - Gradient descent
    """
    # Your code here
    pass
```

---

### Exercise B8: Polynomial Regression 🟡
```python
def polynomial_regression(x, y, degree):
    """
    Fit polynomial regression of given degree.
    
    Args:
        x: numpy array of shape (n,) - input values
        y: numpy array of shape (n,) - output values
        degree: polynomial degree
    
    Returns:
        coefficients: [c0, c1, c2, ..., c_degree] for c0 + c1*x + c2*x^2 + ...
    """
    # Your code here
    # Hint: Build Vandermonde matrix
    pass
```

---

### Exercise B9: Underdetermined System 🔴
```python
def minimum_norm_solution(A, b):
    """
    For underdetermined system (more unknowns than equations),
    find the minimum norm solution.
    
    This is the solution with smallest ||x||.
    
    Args:
        A: numpy array of shape (m, n) where m < n
        b: numpy array of shape (m,)
    
    Returns:
        x: minimum norm solution of shape (n,)
    
    Hint: x = A^T (A A^T)^(-1) b
    """
    # Your code here
    pass
```

---

### Exercise B10: Rank-Deficient Least Squares 🔴
```python
def robust_least_squares(A, b, rcond=1e-10):
    """
    Solve least squares even when A^T A is rank-deficient.
    
    Use SVD: A = U Σ V^T
    Then x = V Σ^+ U^T b where Σ^+ inverts non-zero singular values.
    
    Args:
        A: numpy array of shape (m, n)
        b: numpy array of shape (m,)
        rcond: threshold for treating singular values as zero
    
    Returns:
        x: least squares solution
        rank: effective rank of A
    """
    # Your code here
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Why Not Invert?
We're told to never compute A⁻¹ explicitly to solve Ax = b. But what if we need to solve many systems with the same A and different b vectors? Isn't A⁻¹ then worth computing?

---

### Question C2: Least Squares Intuition
In linear regression, we minimize squared error. Why squared? What would happen if we minimized absolute error instead? (This is called L1 regression or LAD regression.)

---

### Question C3: Regularization Geometry
Ridge regression adds λI to AᵀA. Geometrically, what does this do to the problem? How does it relate to the eigenvalues of AᵀA?

---

### Question C4: Iterative vs Direct
When would you choose an iterative method (like conjugate gradient) over a direct method (like LU decomposition)?

---

### Question C5: Neural Network Connection
How does solving linear systems connect to training neural networks? What role do linear systems play in:
a) Forward pass
b) Backward pass
c) Optimization (e.g., Newton's method)

---

## Difficulty Legend
- 🟢 Easy: Direct application
- 🟡 Medium: Combining concepts
- 🔴 Hard: Requires derivation or deep understanding
