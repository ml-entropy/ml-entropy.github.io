# Exercises: Orthogonality and Projections

## Part A: Theory Problems

### Exercise A1: Orthogonality Check 🟢
Determine which pairs of vectors are orthogonal:

a) **u** = [1, 2, 3]ᵀ, **v** = [3, 0, -1]ᵀ

b) **u** = [1, 1, 1, 1]ᵀ, **v** = [1, -1, 1, -1]ᵀ

c) **u** = [2, -1]ᵀ, **v** = [1, 2]ᵀ

---

### Exercise A2: Orthogonal Matrix Verification 🟢
Verify that the following matrices are orthogonal and determine if they are rotations or reflections:

a) $Q = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$

b) $R = \begin{bmatrix} \cos(30°) & -\sin(30°) \\ \sin(30°) & \cos(30°) \end{bmatrix}$

c) $P = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$

---

### Exercise A3: Projection onto a Line 🟢
Find the projection of **b** = [3, 4]ᵀ onto the line spanned by **a** = [1, 2]ᵀ.

a) Calculate the projection vector.
b) Calculate the component of **b** perpendicular to **a**.
c) Verify that the two components are orthogonal.

---

### Exercise A4: Projection Matrix Properties 🟡
For the projection matrix P = **aa**ᵀ/(‖**a**‖²) where **a** = [1, 1]ᵀ:

a) Write out P explicitly.
b) Verify P² = P.
c) Verify Pᵀ = P.
d) Find the eigenvalues of P.

---

### Exercise A5: Gram-Schmidt Process 🟡
Apply Gram-Schmidt to orthonormalize:

**v₁** = [1, 1, 0]ᵀ, **v₂** = [0, 1, 1]ᵀ, **v₃** = [1, 0, 1]ᵀ

Show each step of the calculation.

---

### Exercise A6: QR Decomposition 🟡
Find the QR decomposition of:
$$A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

a) Apply Gram-Schmidt to the columns of A to find Q.
b) Compute R = QᵀA.
c) Verify that A = QR.

---

### Exercise A7: Least Squares via Projection 🟡
Using projection, find the least squares solution to:
$$\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

a) What is the projection of **b** onto the column space of A?
b) What is the least squares solution x?
c) What is the residual?

---

### Exercise A8: Orthogonal Complement 🔴
For the subspace V = span{[1, 1, 0, 0]ᵀ, [0, 1, 1, 0]ᵀ} in ℝ⁴:

a) Find a basis for V⊥.
b) Verify that your basis vectors are orthogonal to both vectors spanning V.
c) What is dim(V⊥)?

---

### Exercise A9: Four Fundamental Subspaces 🔴
For the matrix:
$$A = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \end{bmatrix}$$

a) Find a basis for the column space C(A).
b) Find a basis for the null space N(A).
c) Find a basis for the row space C(Aᵀ).
d) Find a basis for the left null space N(Aᵀ).
e) Verify the orthogonality relationships.

---

### Exercise A10: Orthogonal Procrustes Problem 🔴
Given two matrices X and Y of the same shape, find the orthogonal matrix Q that minimizes ‖XQ - Y‖_F.

a) Show that the optimal Q comes from the SVD of YᵀX.
b) If X = [[1, 0], [0, 1], [1, 1]] and Y = [[0, 1], [1, 0], [1, 1]], find Q.

---

## Part B: Coding Problems

### Exercise B1: Orthogonality Checker 🟢
```python
def check_orthogonality(vectors, tolerance=1e-10):
    """
    Check if a set of vectors is orthogonal (and optionally orthonormal).
    
    Args:
        vectors: list of numpy arrays
        tolerance: threshold for considering values as zero
    
    Returns:
        dict with keys:
            'is_orthogonal': bool
            'is_orthonormal': bool
            'dot_products': matrix of pairwise dot products
            'norms': list of vector norms
    """
    # Your code here
    pass
```

---

### Exercise B2: Projection Visualization 🟢
```python
def visualize_projection(b, a):
    """
    Visualize projection of b onto the line spanned by a (2D).
    
    Plot:
    - Original vectors a and b
    - Projection of b onto a
    - Perpendicular component
    - Right angle indicator
    
    Args:
        b: numpy array (2,)
        a: numpy array (2,)
    """
    # Your code here
    pass
```

---

### Exercise B3: Gram-Schmidt Implementation 🟡
```python
def gram_schmidt(vectors, normalize=True):
    """
    Orthogonalize a set of vectors using the Gram-Schmidt process.
    
    Args:
        vectors: list of numpy arrays or (n, k) matrix with vectors as columns
        normalize: if True, return orthonormal vectors
    
    Returns:
        numpy array: orthogonal (or orthonormal) vectors as columns
    """
    # Your code here
    pass

def modified_gram_schmidt(vectors, normalize=True):
    """
    Numerically stable version of Gram-Schmidt.
    
    Instead of subtracting all projections at once, subtract one at a time.
    """
    # Your code here
    pass
```

---

### Exercise B4: QR Decomposition from Scratch 🟡
```python
def qr_decomposition(A):
    """
    Compute QR decomposition using Gram-Schmidt.
    
    Args:
        A: numpy array of shape (m, n) with m >= n
    
    Returns:
        tuple: (Q, R) where A = Q @ R
    """
    # Your code here
    pass
```

---

### Exercise B5: Least Squares via QR 🟡
```python
def least_squares_qr(A, b):
    """
    Solve least squares problem Ax ≈ b using QR decomposition.
    
    Args:
        A: numpy array of shape (m, n)
        b: numpy array of shape (m,)
    
    Returns:
        dict with keys:
            'x': least squares solution
            'residual_norm': ||Ax - b||
            'projection': projection of b onto column space of A
    """
    # Your code here
    pass
```

---

### Exercise B6: Projection onto Subspace 🟡
```python
def project_onto_subspace(b, basis_vectors):
    """
    Project vector b onto the subspace spanned by basis_vectors.
    
    Args:
        b: numpy array (n,)
        basis_vectors: list of numpy arrays or (n, k) matrix
    
    Returns:
        dict with keys:
            'projection': projection of b onto subspace
            'perpendicular': component of b perpendicular to subspace
            'projection_matrix': the projection matrix P
    """
    # Your code here
    pass
```

---

### Exercise B7: Orthogonal Matrix Generator 🟡
```python
def random_orthogonal_matrix(n, det_sign=None):
    """
    Generate a random orthogonal matrix.
    
    Args:
        n: size of matrix
        det_sign: if +1, ensure rotation; if -1, ensure reflection; if None, random
    
    Returns:
        numpy array: random orthogonal matrix
    """
    # Your code here
    # Hint: Use QR decomposition of random matrix
    pass

def rotation_matrix_2d(theta):
    """Generate 2D rotation matrix."""
    # Your code here
    pass

def rotation_matrix_3d(axis, theta):
    """Generate 3D rotation matrix around given axis."""
    # Your code here
    pass
```

---

### Exercise B8: Compare Least Squares Methods 🟡
```python
def compare_least_squares_methods(A, b):
    """
    Compare different methods for solving least squares.
    
    Methods:
    1. Normal equations: (A^T A)^{-1} A^T b
    2. QR decomposition
    3. SVD pseudoinverse
    4. numpy.linalg.lstsq
    
    Args:
        A: numpy array of shape (m, n)
        b: numpy array of shape (m,)
    
    Returns:
        dict with solution and timing for each method
    """
    # Your code here
    pass

def test_ill_conditioned_least_squares():
    """
    Test methods on an ill-conditioned problem where normal equations fail.
    """
    # Your code here
    pass
```

---

### Exercise B9: Orthogonal Weight Initialization 🔴
```python
def orthogonal_init(shape, gain=1.0):
    """
    Generate orthogonal weight matrix for neural network initialization.
    
    Args:
        shape: tuple (fan_out, fan_in)
        gain: scaling factor
    
    Returns:
        numpy array: orthogonal weight matrix
    """
    # Your code here
    pass

def compare_initializations():
    """
    Compare orthogonal vs random initialization for a deep network.
    
    Show how singular values of layer products behave.
    """
    # Your code here
    pass
```

---

### Exercise B10: Householder QR 🔴
```python
def householder_qr(A):
    """
    Compute QR decomposition using Householder reflections.
    
    More numerically stable than Gram-Schmidt for ill-conditioned matrices.
    
    Args:
        A: numpy array of shape (m, n)
    
    Returns:
        tuple: (Q, R)
    """
    # Your code here
    # Use Householder reflections H = I - 2vv^T / (v^T v)
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Orthogonality vs Linear Independence
Every orthogonal set is linearly independent, but not every linearly independent set is orthogonal. Give an example and explain why orthogonal sets are "better" in some sense.

---

### Question C2: Gram-Schmidt Numerical Issues
Classical Gram-Schmidt can produce vectors that are not quite orthogonal due to numerical errors. Why does this happen, and how does modified Gram-Schmidt help?

---

### Question C3: Normal Equations vs QR
When would you prefer using QR decomposition over normal equations for least squares? What can go wrong with normal equations?

---

### Question C4: Orthogonal Matrices in Deep Learning
Why is orthogonal initialization helpful for training deep neural networks? What happens to gradients as they flow through orthogonal layers?

---

### Question C5: Projection Residual
In least squares, the residual b - Ax is orthogonal to the column space of A. Explain geometrically why this must be true for the optimal solution.

---

## Difficulty Legend
- 🟢 Easy: Direct application of concepts
- 🟡 Medium: Requires combining multiple concepts or multi-step reasoning
- 🔴 Hard: Requires derivation, proof, or deep understanding
