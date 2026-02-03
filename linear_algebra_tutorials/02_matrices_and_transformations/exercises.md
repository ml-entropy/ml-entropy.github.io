# Exercises: Matrices and Linear Transformations

## Part A: Theory Problems

### Exercise A1: Matrix-Vector Multiplication 🟢
Compute the following products:

a) $\begin{bmatrix} 2 & 1 \\ -1 & 3 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix}$

b) $\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$

c) $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ for θ = π/4

---

### Exercise A2: Matrix Multiplication 🟢
Compute AB and BA (if possible) for:

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

What does matrix B do geometrically?

---

### Exercise A3: Transformation Composition 🟡
Let R be a rotation by 90° counterclockwise and S be scaling by 2 in x and 3 in y.

a) Write the matrices for R and S
b) Compute RS and SR
c) Apply each to the point (1, 0). Are results different?

---

### Exercise A4: Properties of Transpose 🟢
Prove that:

a) (A + B)ᵀ = Aᵀ + Bᵀ
b) (cA)ᵀ = cAᵀ
c) (AB)ᵀ = BᵀAᵀ (the order reverses!)

---

### Exercise A5: Inverse Matrix 🟡
For a 2×2 matrix A = $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

a) Derive the formula for A⁻¹ in terms of a, b, c, d
b) When does A⁻¹ not exist?
c) Find the inverse of $\begin{bmatrix} 3 & 1 \\ 2 & 1 \end{bmatrix}$

---

### Exercise A6: Orthogonal Matrices 🟡
Prove that for an orthogonal matrix Q:

a) ||Q**x**|| = ||**x**|| for all vectors **x**
b) Q**u** · Q**v** = **u** · **v** for all vectors **u**, **v**
c) |det(Q)| = 1

---

### Exercise A7: Rank 🟡
Find the rank of each matrix and describe what the transformation does geometrically:

a) $\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$

b) $\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$

c) $\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$

---

### Exercise A8: Matrix Exponential (Preview) 🔴
For matrix A, the exponential is defined as:
$$e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots$$

a) Compute e^A for A = $\begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix}$

*Hint: Compute A², A³, etc. and find a pattern. Compare with rotation matrices.*

---

### Exercise A9: Similarity Transformation 🔴
Two matrices A and B are similar if B = P⁻¹AP for some invertible P.

a) Prove that similar matrices have the same determinant
b) Prove that similar matrices have the same trace
c) If A is diagonal, what does P⁻¹AP represent?

---

### Exercise A10: Quadratic Form 🔴
For symmetric matrix A, the quadratic form is f(**x**) = **x**ᵀA**x**.

a) Expand f(**x**) for **x** = [x, y]ᵀ and A = $\begin{bmatrix} a & b \\ b & c \end{bmatrix}$

b) Show that ∇f = 2A**x**

c) What does the Hessian of f equal?

---

## Part B: Coding Problems

### Exercise B1: Matrix Operations 🟢
```python
def matrix_operations(A, B):
    """
    Implement basic matrix operations (without numpy matrix functions).
    
    Args:
        A, B: 2D lists representing matrices
    
    Returns:
        dict with keys: 'add', 'multiply', 'transpose_A'
    """
    # Your code here
    pass
```

---

### Exercise B2: Rotation Matrix 🟢
```python
def rotation_matrix_2d(theta):
    """
    Create a 2D rotation matrix.
    
    Args:
        theta: angle in radians
    
    Returns:
        2x2 numpy array
    """
    # Your code here
    pass

def rotate_points(points, theta):
    """
    Rotate multiple 2D points.
    
    Args:
        points: numpy array of shape (N, 2)
        theta: angle in radians
    
    Returns:
        numpy array of shape (N, 2)
    """
    # Your code here
    pass
```

---

### Exercise B3: Check Matrix Properties 🟡
```python
def check_matrix_properties(A, tolerance=1e-10):
    """
    Check various properties of a square matrix.
    
    Args:
        A: numpy array (square matrix)
        tolerance: numerical tolerance
    
    Returns:
        dict with keys: 
        - 'is_symmetric': bool
        - 'is_orthogonal': bool
        - 'is_invertible': bool
        - 'rank': int
    """
    # Your code here
    pass
```

---

### Exercise B4: Implement Matrix Inverse (2x2) 🟡
```python
def inverse_2x2(A):
    """
    Compute inverse of 2x2 matrix using the formula.
    Do NOT use np.linalg.inv.
    
    Args:
        A: numpy array of shape (2, 2)
    
    Returns:
        numpy array of shape (2, 2), or None if not invertible
    """
    # Your code here
    pass
```

---

### Exercise B5: Visualize Transformation 🟡
```python
def visualize_transformation_effect(A, points, title="Transformation"):
    """
    Create a visualization showing how matrix A transforms a set of points.
    
    Args:
        A: 2x2 transformation matrix
        points: numpy array of shape (N, 2)
        title: plot title
    
    Show:
    - Original points in blue
    - Transformed points in red
    - Arrows connecting corresponding points
    """
    # Your code here
    pass
```

---

### Exercise B6: Compute Covariance Matrix 🟡
```python
def compute_covariance(X):
    """
    Compute the covariance matrix of data X.
    Do NOT use np.cov.
    
    Args:
        X: numpy array of shape (N, D) - N samples, D features
    
    Returns:
        numpy array of shape (D, D)
    """
    # Your code here
    # Remember: Cov = (1/(N-1)) * X_centered.T @ X_centered
    pass
```

---

### Exercise B7: Power Iteration 🔴
```python
def power_iteration(A, num_iterations=100):
    """
    Find the dominant eigenvalue and eigenvector using power iteration.
    
    Args:
        A: numpy array (square matrix)
        num_iterations: number of iterations
    
    Returns:
        tuple: (eigenvalue, eigenvector)
    """
    # Your code here
    # Algorithm:
    # 1. Start with random vector v
    # 2. Repeatedly: v = A @ v, then normalize v
    # 3. Eigenvalue ≈ v^T @ A @ v
    pass
```

---

### Exercise B8: Neural Network Layer 🟡
```python
class LinearLayer:
    """Implement a neural network linear layer."""
    
    def __init__(self, input_dim, output_dim):
        """Initialize with random weights."""
        # Xavier initialization
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros(output_dim)
        
    def forward(self, X):
        """
        Forward pass: y = XW^T + b
        
        Args:
            X: numpy array of shape (batch_size, input_dim)
        
        Returns:
            numpy array of shape (batch_size, output_dim)
        """
        # Your code here (also store X for backward)
        pass
    
    def backward(self, dy):
        """
        Backward pass: compute gradients.
        
        Args:
            dy: gradient of loss w.r.t. output, shape (batch_size, output_dim)
        
        Returns:
            dx: gradient w.r.t. input, shape (batch_size, input_dim)
        
        Also stores self.dW and self.db.
        """
        # Your code here
        pass
```

---

### Exercise B9: Low-Rank Approximation 🔴
```python
def low_rank_approximation(A, k):
    """
    Compute the best rank-k approximation of matrix A.
    
    Args:
        A: numpy array of any shape
        k: desired rank
    
    Returns:
        numpy array of same shape, with rank at most k
    
    Hint: Use SVD. A ≈ U[:, :k] @ Σ[:k, :k] @ V[:k, :]
    """
    # Your code here
    pass

def compression_ratio(original, approximation):
    """
    Compute compression ratio achieved by low-rank approximation.
    """
    # Your code here
    pass
```

---

### Exercise B10: Implement Attention Mechanism 🔴
```python
def scaled_dot_product_attention(Q, K, V):
    """
    Implement scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        Q: numpy array of shape (seq_len_q, d_k) - queries
        K: numpy array of shape (seq_len_k, d_k) - keys
        V: numpy array of shape (seq_len_k, d_v) - values
    
    Returns:
        numpy array of shape (seq_len_q, d_v)
    """
    # Your code here
    pass

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    # Your code here
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Why AB ≠ BA?
Give a geometric explanation for why matrix multiplication is not commutative. Use the example of rotation and scaling.

---

### Question C2: Rank and Information
If a matrix A has rank r < min(m, n), information is lost when computing A**x**. Explain:
- What information is lost?
- Can it be recovered?
- How does this relate to compression in ML?

---

### Question C3: Orthogonal Initialization
Why might we want to initialize neural network weight matrices to be orthogonal (or close to orthogonal)?

---

### Question C4: Covariance Matrix Intuition
The covariance matrix Σ is always:
- Symmetric
- Positive semi-definite

Explain why each property must hold, based on the definition.

---

### Question C5: Attention as Matrix Multiplication
In transformers, attention computes: softmax(QKᵀ/√d)V

- What does QKᵀ compute for each pair of positions?
- Why divide by √d?
- What does multiplying by V accomplish?

---

## Difficulty Legend
- 🟢 Easy: Direct application
- 🟡 Medium: Combining concepts
- 🔴 Hard: Requires derivation or deep understanding
