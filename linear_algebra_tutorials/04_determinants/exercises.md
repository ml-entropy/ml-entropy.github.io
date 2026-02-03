# Exercises: Determinants

## Part A: Theory Problems

### Exercise A1: Basic 2×2 Determinants 🟢
Compute the determinants of:

a) $A = \begin{bmatrix} 3 & 2 \\ 1 & 4 \end{bmatrix}$

b) $B = \begin{bmatrix} 5 & -3 \\ -2 & 1 \end{bmatrix}$

c) $C = \begin{bmatrix} 2 & 6 \\ 1 & 3 \end{bmatrix}$

d) What does the result in (c) tell you about matrix C?

---

### Exercise A2: 3×3 Determinants 🟢
Compute using cofactor expansion:

a) $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$

b) $B = \begin{bmatrix} 2 & 0 & 1 \\ 0 & 3 & 0 \\ 4 & 0 & 5 \end{bmatrix}$

*Hint for (b): Choose the row/column with the most zeros.*

---

### Exercise A3: Geometric Interpretation 🟢
A 2×2 matrix A transforms the unit square into a parallelogram with vertices at (0,0), (2,1), (1,3), and (3,4).

a) What is det(A)?
b) Is the orientation preserved or reversed?
c) Write down a possible matrix A.

---

### Exercise A4: Row Operations 🟡
Given the matrix:
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{bmatrix}$$

a) Use Gaussian elimination to reduce A to upper triangular form.
b) Keep track of how the determinant changes at each step.
c) Compute det(A) from the triangular form.

---

### Exercise A5: Determinant Properties 🟡
Without computing the full determinant, determine det(A) for:

a) $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 7 & 8 & 9 \end{bmatrix}$

b) $A = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 2 & 3 & 0 & 0 \\ 4 & 5 & 6 & 0 \\ 7 & 8 & 9 & 10 \end{bmatrix}$

c) If det(B) = 5, what is det(3B) for a 3×3 matrix B?

---

### Exercise A6: Product Rule 🟡
Let A and B be 3×3 matrices with det(A) = 2 and det(B) = -3.

a) Compute det(AB)
b) Compute det(A²B)
c) Compute det(A⁻¹)
d) Compute det(AᵀB⁻¹)

---

### Exercise A7: Jacobian Calculation 🟡
Compute the Jacobian determinant for the following coordinate transformations:

a) **Polar to Cartesian:** x = r cos(θ), y = r sin(θ)

b) **Scaling transformation:** u = 2x, v = 3y

c) **Shear transformation:** u = x + 2y, v = y

---

### Exercise A8: Cofactor Matrix Derivation 🔴
For a 2×2 matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

a) Write out the cofactor matrix C (matrix of cofactors)
b) Compute $A \cdot C^T$
c) What pattern do you observe?
d) Use this to derive the formula for $A^{-1}$

---

### Exercise A9: Determinant of Block Matrices 🔴
For a block matrix:
$$M = \begin{bmatrix} A & B \\ 0 & D \end{bmatrix}$$

where A is n×n and D is m×m:

a) Prove that det(M) = det(A) · det(D)
b) Does this formula hold if the zero block is in a different position?
c) What about $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$ in general?

---

### Exercise A10: Normalizing Flow Jacobian 🔴
Consider the transformation used in RealNVP:
$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 \\ x_2 \cdot \exp(s(x_1)) + t(x_1) \end{bmatrix}$$

where s and t are arbitrary functions.

a) Compute the Jacobian matrix $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$
b) Compute the determinant of the Jacobian
c) Why is this form computationally efficient for normalizing flows?

---

## Part B: Coding Problems

### Exercise B1: Determinant from Scratch 🟢
```python
def det_2x2(matrix):
    """
    Compute the determinant of a 2x2 matrix.
    
    Args:
        matrix: numpy array of shape (2, 2)
    
    Returns:
        float: the determinant
    """
    # Your code here (don't use np.linalg.det)
    pass

def det_3x3(matrix):
    """
    Compute the determinant of a 3x3 matrix using cofactor expansion.
    
    Args:
        matrix: numpy array of shape (3, 3)
    
    Returns:
        float: the determinant
    """
    # Your code here
    pass
```

---

### Exercise B2: Visualize Area Scaling 🟢
```python
def visualize_determinant_as_area(A):
    """
    Visualize how a 2x2 matrix transforms the unit square.
    
    Plot:
    1. The original unit square
    2. The transformed parallelogram
    3. Display the determinant as the area scaling factor
    
    Args:
        A: numpy array of shape (2, 2)
    """
    # Your code here
    pass
```

---

### Exercise B3: Row Reduction Determinant 🟡
```python
def det_gaussian(matrix, verbose=False):
    """
    Compute determinant using Gaussian elimination.
    
    Args:
        matrix: numpy array of shape (n, n)
        verbose: if True, print each step and how det changes
    
    Returns:
        float: the determinant
    """
    # Your code here
    # Track: row swaps (negate det), row scaling (multiply det)
    pass
```

---

### Exercise B4: Jacobian Determinant 🟡
```python
def numerical_jacobian_det(f, x, epsilon=1e-5):
    """
    Numerically compute the Jacobian determinant at a point.
    
    Args:
        f: function that takes array of shape (n,) and returns array of shape (n,)
        x: numpy array of shape (n,), point at which to evaluate
        epsilon: step size for numerical differentiation
    
    Returns:
        float: the Jacobian determinant |det(∂f/∂x)|
    """
    # Your code here
    # Use finite differences to approximate each partial derivative
    pass

# Test with polar-to-cartesian transformation
def polar_to_cartesian(polar):
    r, theta = polar
    return np.array([r * np.cos(theta), r * np.sin(theta)])
```

---

### Exercise B5: Singularity Detection 🟡
```python
def analyze_matrix_singularity(A, tolerance=1e-10):
    """
    Analyze whether a matrix is singular or near-singular.
    
    Args:
        A: numpy array of shape (n, n)
        tolerance: threshold for considering determinant "zero"
    
    Returns:
        dict with keys:
            'determinant': float
            'is_singular': bool
            'condition_number': float (ratio of largest to smallest singular value)
            'rank': int
    """
    # Your code here
    pass
```

---

### Exercise B6: Log-Determinant Computation 🟡
```python
def log_det_stable(A):
    """
    Compute log(|det(A)|) in a numerically stable way.
    
    Direct computation of det(A) can overflow/underflow for large matrices.
    Use the fact that det(A) = product of eigenvalues, so
    log|det(A)| = sum of log|eigenvalues|.
    
    Args:
        A: numpy array of shape (n, n)
    
    Returns:
        tuple: (log_abs_det, sign) where det(A) = sign * exp(log_abs_det)
    """
    # Your code here
    # Hint: Use LU decomposition or Cholesky for positive definite matrices
    pass
```

---

### Exercise B7: Visualize Jacobian Field 🟡
```python
def visualize_jacobian_field(f, x_range, y_range, resolution=20):
    """
    Visualize how the Jacobian determinant varies across space.
    
    Create a heatmap showing |det(J)| at each point, indicating
    local area scaling.
    
    Args:
        f: function from R² → R²
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max)
        resolution: number of points per dimension
    """
    # Your code here
    pass
```

---

### Exercise B8: Cofactor Matrix 🔴
```python
def cofactor_matrix(A):
    """
    Compute the cofactor matrix (matrix of cofactors).
    
    C[i,j] = (-1)^(i+j) * det(minor_ij)
    
    Args:
        A: numpy array of shape (n, n)
    
    Returns:
        numpy array of shape (n, n): the cofactor matrix
    """
    # Your code here
    pass

def matrix_inverse_via_cofactors(A):
    """
    Compute A^(-1) using the cofactor formula:
    A^(-1) = (1/det(A)) * C^T
    
    Args:
        A: numpy array of shape (n, n)
    
    Returns:
        numpy array of shape (n, n): the inverse
    """
    # Your code here
    pass
```

---

### Exercise B9: Normalizing Flow Layer 🔴
```python
class AffineFlowLayer:
    """
    A simple affine normalizing flow layer.
    
    Transform: y = x * exp(log_scale) + shift
    
    This is a simple element-wise transformation with easy Jacobian.
    """
    
    def __init__(self, dim):
        self.log_scale = np.zeros(dim)
        self.shift = np.zeros(dim)
    
    def forward(self, x):
        """
        Transform x to y.
        
        Returns:
            tuple: (y, log_det_jacobian)
        """
        # Your code here
        pass
    
    def inverse(self, y):
        """
        Transform y back to x.
        
        Returns:
            tuple: (x, log_det_jacobian)
        """
        # Your code here
        pass
```

---

### Exercise B10: Gaussian Log-Likelihood 🔴
```python
def multivariate_gaussian_log_likelihood(X, mu, Sigma):
    """
    Compute log-likelihood of data under multivariate Gaussian.
    
    log p(x) = -n/2 log(2π) - 1/2 log|Σ| - 1/2 (x-μ)ᵀΣ⁻¹(x-μ)
    
    Args:
        X: numpy array of shape (N, D) - N data points, D dimensions
        mu: numpy array of shape (D,) - mean
        Sigma: numpy array of shape (D, D) - covariance matrix
    
    Returns:
        numpy array of shape (N,): log-likelihood of each point
    """
    # Your code here
    # Use log-determinant for numerical stability
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Zero Determinant
A matrix has det(A) = 0. Give three different geometric/algebraic interpretations of what this means.

---

### Question C2: Determinant vs. Matrix Norm
Both det(A) and ||A|| provide single numbers summarizing a matrix. How do they differ in what they measure? Give an example where ||A|| is large but det(A) is small.

---

### Question C3: Jacobian in Neural Networks
In a neural network, we often compute backpropagation through layers. The Jacobian ∂y/∂x represents the local linear approximation of a layer. Why don't we typically compute or use determinants in standard backpropagation?

---

### Question C4: Computational Complexity
Explain why:
a) Cofactor expansion is O(n!)
b) Gaussian elimination gives O(n³)
c) For normalizing flows, we design transformations with O(n) determinant computation

---

### Question C5: Determinant and Eigenvalues
The determinant equals the product of eigenvalues. Use this to explain:
a) Why det(A) = 0 means A has eigenvalue 0
b) Why det(A) > 0 doesn't necessarily mean A is positive definite
c) How to compute log|det(A)| stably for large matrices

---

## Difficulty Legend
- 🟢 Easy: Direct application of formulas
- 🟡 Medium: Requires combining concepts or multi-step reasoning
- 🔴 Hard: Requires derivation, proof, or deep understanding
