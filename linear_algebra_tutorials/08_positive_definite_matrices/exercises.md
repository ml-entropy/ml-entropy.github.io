# Exercises: Positive Definite Matrices

## Part A: Theory Problems

### Exercise A1: Checking Positive Definiteness 🟢
Determine if each matrix is positive definite, positive semi-definite, or neither:

a) $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$

b) $B = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$

c) $C = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$

d) $D = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$

---

### Exercise A2: Eigenvalue Test 🟢
For each matrix in A1, verify your answer by computing the eigenvalues.

---

### Exercise A3: Sylvester's Criterion 🟡
Using leading principal minors (Sylvester's criterion), determine positive definiteness:

a) $A = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$

b) $B = \begin{bmatrix} 2 & 2 & 0 \\ 2 & 5 & 3 \\ 0 & 3 & 4 \end{bmatrix}$

*Show all the minors you compute.*

---

### Exercise A4: Cholesky Decomposition 🟡
Compute the Cholesky decomposition A = LLᵀ for:

$$A = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$$

Show each step of the calculation.

---

### Exercise A5: 3×3 Cholesky 🟡
Compute the Cholesky decomposition for:

$$A = \begin{bmatrix} 4 & 2 & -2 \\ 2 & 5 & -4 \\ -2 & -4 & 14 \end{bmatrix}$$

---

### Exercise A6: Quadratic Form Analysis 🟡
For the quadratic form f(**x**) = **x**ᵀA**x** where:

$$A = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$$

a) Write out f(x₁, x₂) explicitly.
b) Find the eigenvalues and eigenvectors of A.
c) Express f in the eigenbasis (diagonalized form).
d) Sketch the level curves f(**x**) = 1.

---

### Exercise A7: Covariance Matrix Properties 🟡
Let X be a centered data matrix with:

$$X = \begin{bmatrix} 1 & 2 \\ -1 & 0 \\ 0 & -2 \end{bmatrix}$$

a) Compute the sample covariance matrix Σ = XᵀX/(n-1).
b) Verify Σ is positive semi-definite.
c) What is the rank of Σ? Why?

---

### Exercise A8: Matrix Square Root 🔴
For the matrix:

$$A = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

a) Find the eigendecomposition A = QΛQᵀ.
b) Compute the symmetric square root A^{1/2} = QΛ^{1/2}Qᵀ.
c) Verify that (A^{1/2})² = A.

---

### Exercise A9: Completing the Square 🔴
For the quadratic form:

$$f(x_1, x_2) = 2x_1^2 + 4x_1x_2 + 5x_2^2$$

a) Write f as **x**ᵀA**x** for appropriate symmetric A.
b) Complete the square to show f(**x**) ≥ 0 for all **x**.
c) Use Cholesky: if A = LLᵀ, show f(**x**) = ‖Lᵀ**x**‖².

---

### Exercise A10: Optimization Perspective 🔴
Consider the function g(**x**) = ½**x**ᵀA**x** - **b**ᵀ**x** + c where A is positive definite.

a) Compute the gradient ∇g(**x**).
b) Compute the Hessian ∇²g(**x**).
c) Find the minimum by setting gradient to zero.
d) Verify it's a minimum using the Hessian.
e) Why does positive definiteness guarantee a unique minimum?

---

## Part B: Coding Problems

### Exercise B1: Positive Definiteness Checker 🟢
```python
def check_positive_definite(A, method='eigenvalue', tol=1e-10):
    """
    Check if a matrix is positive definite using various methods.
    
    Args:
        A: numpy array of shape (n, n)
        method: 'eigenvalue', 'cholesky', or 'sylvester'
        tol: tolerance for numerical checks
    
    Returns:
        dict with keys:
            'is_positive_definite': bool
            'is_positive_semidefinite': bool
            'details': additional information (eigenvalues, minors, etc.)
    """
    # Your code here
    pass
```

---

### Exercise B2: Cholesky Decomposition from Scratch 🟡
```python
def cholesky_decomposition(A):
    """
    Compute Cholesky decomposition A = L @ L.T.
    
    Args:
        A: numpy array of shape (n, n), must be positive definite
    
    Returns:
        L: lower triangular matrix with positive diagonal
    
    Raises:
        ValueError: if A is not positive definite
    """
    # Your code here
    pass
```

---

### Exercise B3: Solve Linear System via Cholesky 🟡
```python
def solve_cholesky(A, b):
    """
    Solve Ax = b using Cholesky decomposition (when A is positive definite).
    
    More efficient than general LU decomposition!
    
    Args:
        A: numpy array of shape (n, n), positive definite
        b: numpy array of shape (n,)
    
    Returns:
        x: solution vector
    """
    # Your code here
    # Steps:
    # 1. Compute A = L @ L.T
    # 2. Solve L @ y = b (forward substitution)
    # 3. Solve L.T @ x = y (back substitution)
    pass
```

---

### Exercise B4: Visualize Quadratic Forms 🟢
```python
def visualize_quadratic_form(A):
    """
    Visualize the quadratic form f(x) = x^T A x for a 2x2 matrix.
    
    Plot:
    1. 3D surface
    2. Contour plot with eigenvector directions
    
    Args:
        A: numpy array of shape (2, 2), symmetric
    """
    # Your code here
    pass
```

---

### Exercise B5: Sample from Multivariate Gaussian 🟡
```python
def sample_multivariate_gaussian(mu, Sigma, n_samples):
    """
    Sample from N(mu, Sigma) using Cholesky decomposition.
    
    Args:
        mu: numpy array of shape (d,) - mean
        Sigma: numpy array of shape (d, d) - covariance (positive definite)
        n_samples: number of samples
    
    Returns:
        numpy array of shape (n_samples, d)
    """
    # Your code here
    # Use: x = mu + L @ z where Sigma = L @ L.T and z ~ N(0, I)
    pass

def visualize_gaussian_samples(mu, Sigma, n_samples=1000):
    """
    Visualize samples from 2D Gaussian along with covariance ellipse.
    """
    # Your code here
    pass
```

---

### Exercise B6: Mahalanobis Distance 🟡
```python
def mahalanobis_distance(x, mu, Sigma):
    """
    Compute Mahalanobis distance from x to distribution N(mu, Sigma).
    
    d_M(x, mu) = sqrt((x - mu)^T Sigma^{-1} (x - mu))
    
    Args:
        x: numpy array of shape (d,) or (n, d) for multiple points
        mu: numpy array of shape (d,)
        Sigma: numpy array of shape (d, d)
    
    Returns:
        float or array of distances
    """
    # Your code here
    # Hint: Use Cholesky for efficient computation
    pass

def visualize_mahalanobis(mu, Sigma, points):
    """
    Visualize Mahalanobis distances with concentric ellipses.
    """
    # Your code here
    pass
```

---

### Exercise B7: Nearest Positive Definite Matrix 🟡
```python
def nearest_positive_definite(A, min_eigenvalue=1e-6):
    """
    Find the nearest positive definite matrix to A.
    
    Useful for fixing covariance matrices with numerical issues.
    
    Args:
        A: numpy array of shape (n, n), symmetric but possibly not PD
        min_eigenvalue: minimum eigenvalue to enforce
    
    Returns:
        A_pd: nearest positive definite matrix
    """
    # Your code here
    # Steps:
    # 1. Eigendecomposition
    # 2. Set negative eigenvalues to min_eigenvalue
    # 3. Reconstruct
    pass
```

---

### Exercise B8: Condition Number Analysis 🟡
```python
def analyze_condition_number(A):
    """
    Analyze the condition number of a positive definite matrix.
    
    Args:
        A: numpy array of shape (n, n), positive definite
    
    Returns:
        dict with keys:
            'condition_number': lambda_max / lambda_min
            'eigenvalues': sorted eigenvalues
            'is_well_conditioned': bool (condition < 10^3)
            'log_condition': log10 of condition number
    """
    # Your code here
    pass

def visualize_condition_effect():
    """
    Demonstrate how condition number affects:
    1. Ellipse shape of quadratic form
    2. Gradient descent convergence
    """
    # Your code here
    pass
```

---

### Exercise B9: Log-Determinant Computation 🔴
```python
def log_determinant_cholesky(A):
    """
    Compute log(det(A)) stably using Cholesky decomposition.
    
    For positive definite A with Cholesky A = L @ L.T:
    log(det(A)) = log(det(L)^2) = 2 * sum(log(diag(L)))
    
    Args:
        A: numpy array of shape (n, n), positive definite
    
    Returns:
        float: log of determinant
    """
    # Your code here
    pass

def gaussian_log_likelihood(X, mu, Sigma):
    """
    Compute log-likelihood of data under multivariate Gaussian.
    
    Use log-determinant for numerical stability.
    
    Args:
        X: numpy array of shape (n, d) - data
        mu: numpy array of shape (d,) - mean
        Sigma: numpy array of shape (d, d) - covariance
    
    Returns:
        float: total log-likelihood
    """
    # Your code here
    pass
```

---

### Exercise B10: Regularization for Positive Definiteness 🔴
```python
def regularize_covariance(S, method='ridge', param=None):
    """
    Regularize a sample covariance matrix to ensure positive definiteness.
    
    Args:
        S: numpy array (d, d) - sample covariance
        method: 'ridge' (S + lambda*I), 'shrinkage' (alpha*S + (1-alpha)*trace(S)/d * I)
        param: regularization parameter (auto-selected if None)
    
    Returns:
        dict with keys:
            'regularized': regularized covariance
            'param_used': actual parameter value
            'condition_before': condition number before
            'condition_after': condition number after
    """
    # Your code here
    pass

def compare_regularization_methods(X):
    """
    Compare different regularization methods on ill-conditioned data.
    """
    # Your code here
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Why Covariance Matrices Are PSD
Prove that a sample covariance matrix Σ = XᵀX/(n-1) is always positive semi-definite. When is it strictly positive definite?

---

### Question C2: Cholesky vs LU
Why is Cholesky decomposition preferred over LU for positive definite matrices? What are the computational and numerical advantages?

---

### Question C3: Hessian and Optimization
Explain the relationship between:
a) Positive definite Hessian and local minimum
b) Positive semi-definite Hessian and convex function
c) Condition number of Hessian and gradient descent convergence

---

### Question C4: Regularization Intuition
In ridge regression, we add λI to XᵀX. Explain:
a) Why this makes the matrix positive definite
b) The effect on eigenvalues
c) The bias-variance tradeoff interpretation

---

### Question C5: Gaussian Distributions
Why must the covariance matrix in a multivariate Gaussian be positive definite? What would go wrong mathematically if it weren't?

---

## Difficulty Legend
- 🟢 Easy: Direct application of concepts
- 🟡 Medium: Requires combining multiple concepts or multi-step reasoning
- 🔴 Hard: Requires derivation, proof, or deep understanding
