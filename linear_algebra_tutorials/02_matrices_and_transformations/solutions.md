# Solutions: Matrices and Linear Transformations

## Part A: Theory Solutions

### Solution A1: Matrix-Vector Multiplication

**a)** 
$$\begin{bmatrix} 2 & 1 \\ -1 & 3 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 2(3) + 1(2) \\ -1(3) + 3(2) \end{bmatrix} = \begin{bmatrix} 8 \\ 3 \end{bmatrix}$$

**b)**
$$\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 1(1) + 0(2) + 2(3) \\ 0(1) + 1(2) + (-1)(3) \end{bmatrix} = \begin{bmatrix} 7 \\ -1 \end{bmatrix}$$

**c)** For θ = π/4:
$$\begin{bmatrix} \cos(\pi/4) & -\sin(\pi/4) \\ \sin(\pi/4) & \cos(\pi/4) \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} \end{bmatrix}$$

This rotates [1, 0] by 45° to [√2/2, √2/2].

---

### Solution A2: Matrix Multiplication

$$AB = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}$$

$$BA = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}$$

**Geometric meaning of B:** B swaps the x and y coordinates. It reflects across the line y = x.

Note: AB ≠ BA - swapping before A gives different result than after A.

---

### Solution A3: Transformation Composition

**a) Matrices:**

Rotation by 90°: R = $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$

Scaling: S = $\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$

**b) Products:**

$$RS = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} 0 & -3 \\ 2 & 0 \end{bmatrix}$$

(First scale, then rotate)

$$SR = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & -2 \\ 3 & 0 \end{bmatrix}$$

(First rotate, then scale)

**c) Applied to (1, 0):**

RS[1, 0]ᵀ = [0, 2]ᵀ (scale x by 2, then rotate 90°)
SR[1, 0]ᵀ = [0, 3]ᵀ (rotate 90° to get [0, 1], then scale y by 3)

Yes, results are different!

---

### Solution A4: Properties of Transpose

**a) (A + B)ᵀ = Aᵀ + Bᵀ**

$((A + B)^T)_{ij} = (A + B)_{ji} = A_{ji} + B_{ji} = (A^T)_{ij} + (B^T)_{ij}$

**b) (cA)ᵀ = cAᵀ**

$((cA)^T)_{ij} = (cA)_{ji} = cA_{ji} = c(A^T)_{ij}$

**c) (AB)ᵀ = BᵀAᵀ**

$((AB)^T)_{ij} = (AB)_{ji} = \sum_k A_{jk}B_{ki}$

$(B^TA^T)_{ij} = \sum_k (B^T)_{ik}(A^T)_{kj} = \sum_k B_{ki}A_{jk}$

These are equal. ∎

---

### Solution A5: Inverse Matrix

**a) Formula for 2×2 inverse:**

For A = $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

**Derivation:** Solve A·X = I by setting up equations.

**b) When does A⁻¹ not exist?**

When det(A) = ad - bc = 0.

Geometrically: columns are parallel (linearly dependent), so transformation collapses space.

**c) Inverse of $\begin{bmatrix} 3 & 1 \\ 2 & 1 \end{bmatrix}$:**

det = 3(1) - 1(2) = 1

$$A^{-1} = \frac{1}{1} \begin{bmatrix} 1 & -1 \\ -2 & 3 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -2 & 3 \end{bmatrix}$$

Verify: $\begin{bmatrix} 3 & 1 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} 1 & -1 \\ -2 & 3 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ ✓

---

### Solution A6: Orthogonal Matrices

**a) ||Qx|| = ||x||:**

$$\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^T(Q\mathbf{x}) = \mathbf{x}^TQ^TQ\mathbf{x} = \mathbf{x}^TI\mathbf{x} = \mathbf{x}^T\mathbf{x} = \|\mathbf{x}\|^2$$

Taking square roots: ||Q**x**|| = ||**x**|| ∎

**b) Qu · Qv = u · v:**

$$(Q\mathbf{u}) \cdot (Q\mathbf{v}) = (Q\mathbf{u})^T(Q\mathbf{v}) = \mathbf{u}^TQ^TQ\mathbf{v} = \mathbf{u}^T\mathbf{v} = \mathbf{u} \cdot \mathbf{v}$$

**c) |det(Q)| = 1:**

From QᵀQ = I:
$$\det(Q^TQ) = \det(Q^T)\det(Q) = \det(Q)^2 = \det(I) = 1$$

Therefore |det(Q)| = 1. ∎

---

### Solution A7: Rank

**a)** $\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$

Row 2 = 2 × Row 1, so **rank = 1**.

Geometrically: Projects all of ℝ² onto the line spanned by [1, 2].

**b)** $\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$

Row 3 = Row 1 + Row 2, so **rank = 2**.

Geometrically: Maps ℝ³ onto a 2D plane (loses one dimension).

**c)** $\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$

Both columns are independent, so **rank = 2** (full column rank).

Geometrically: Embeds ℝ² into the xy-plane of ℝ³.

---

### Solution A8: Matrix Exponential

For A = $\begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix}$:

$$A^2 = \begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix}^2 = \begin{bmatrix} -\theta^2 & 0 \\ 0 & -\theta^2 \end{bmatrix} = -\theta^2 I$$

$$A^3 = A^2 \cdot A = -\theta^2 A$$

$$A^4 = A^2 \cdot A^2 = \theta^4 I$$

Pattern: Even powers give ±θⁿI, odd powers give ±θⁿA.

$$e^A = I + A - \frac{\theta^2}{2!}I - \frac{\theta^3}{3!}A + \frac{\theta^4}{4!}I + \cdots$$

$$= \left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right)I + \left(1 - \frac{\theta^2}{3!} + \frac{\theta^4}{5!} - \cdots\right)A/\theta$$

$$= \cos\theta \cdot I + \sin\theta \cdot \frac{A}{\theta}$$

$$= \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**This is the rotation matrix!** The matrix A is the "generator" of rotations.

---

### Solution A9: Similarity Transformation

**a) Same determinant:**

$\det(P^{-1}AP) = \det(P^{-1})\det(A)\det(P) = \frac{1}{\det(P)}\det(A)\det(P) = \det(A)$

**b) Same trace:**

trace(P⁻¹AP) = trace(APP⁻¹) = trace(A)

(using cyclic property of trace: trace(ABC) = trace(CAB))

**c) If A is diagonal:**

P⁻¹AP represents A in a different basis (the basis formed by columns of P).

The similar matrix B = P⁻¹AP has the same eigenvalues (on diagonal of A) but eigenvectors are columns of P.

---

### Solution A10: Quadratic Form

**a) Expand f(x):**

$$f([x, y]^T) = \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} a & b \\ b & c \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

$$= \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} ax + by \\ bx + cy \end{bmatrix} = ax^2 + 2bxy + cy^2$$

**b) Gradient:**

$$\frac{\partial f}{\partial x} = 2ax + 2by, \quad \frac{\partial f}{\partial y} = 2bx + 2cy$$

$$\nabla f = \begin{bmatrix} 2ax + 2by \\ 2bx + 2cy \end{bmatrix} = 2\begin{bmatrix} a & b \\ b & c \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = 2A\mathbf{x}$$

**c) Hessian:**

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 2a & 2b \\ 2b & 2c \end{bmatrix} = 2A$$

---

## Part B: Coding Solutions

### Solution B1: Matrix Operations
```python
def matrix_operations(A, B):
    """Implement basic matrix operations."""
    m1, n1 = len(A), len(A[0])
    m2, n2 = len(B), len(B[0])
    
    # Addition (must have same dimensions)
    if m1 == m2 and n1 == n2:
        add = [[A[i][j] + B[i][j] for j in range(n1)] for i in range(m1)]
    else:
        add = None
    
    # Multiplication (inner dimensions must match)
    if n1 == m2:
        multiply = [[sum(A[i][k] * B[k][j] for k in range(n1)) 
                     for j in range(n2)] for i in range(m1)]
    else:
        multiply = None
    
    # Transpose
    transpose_A = [[A[j][i] for j in range(m1)] for i in range(n1)]
    
    return {'add': add, 'multiply': multiply, 'transpose_A': transpose_A}

# Test
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = matrix_operations(A, B)
print(f"A + B = {result['add']}")
print(f"A @ B = {result['multiply']}")
print(f"A^T = {result['transpose_A']}")
```

---

### Solution B2: Rotation Matrix
```python
import numpy as np

def rotation_matrix_2d(theta):
    """Create a 2D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rotate_points(points, theta):
    """Rotate multiple 2D points."""
    R = rotation_matrix_2d(theta)
    # points is (N, 2), we need R @ points.T then transpose back
    return (R @ points.T).T

# Test
points = np.array([[1, 0], [0, 1], [1, 1]])
rotated = rotate_points(points, np.pi/4)
print(f"Original points:\n{points}")
print(f"Rotated by 45°:\n{rotated}")
```

---

### Solution B3: Check Matrix Properties
```python
def check_matrix_properties(A, tolerance=1e-10):
    """Check various properties of a square matrix."""
    n = A.shape[0]
    
    # Symmetric: A = A^T
    is_symmetric = np.allclose(A, A.T, atol=tolerance)
    
    # Orthogonal: A^T @ A = I
    is_orthogonal = np.allclose(A.T @ A, np.eye(n), atol=tolerance)
    
    # Invertible: det != 0
    det = np.linalg.det(A)
    is_invertible = np.abs(det) > tolerance
    
    # Rank
    rank = np.linalg.matrix_rank(A, tol=tolerance)
    
    return {
        'is_symmetric': is_symmetric,
        'is_orthogonal': is_orthogonal,
        'is_invertible': is_invertible,
        'rank': rank
    }

# Test
A = np.array([[1, 2], [2, 1]])  # Symmetric but not orthogonal
print(check_matrix_properties(A))

R = rotation_matrix_2d(np.pi/3)  # Orthogonal
print(check_matrix_properties(R))
```

---

### Solution B4: Implement Matrix Inverse (2x2)
```python
def inverse_2x2(A):
    """Compute inverse of 2x2 matrix using the formula."""
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    det = a * d - b * c
    
    if np.abs(det) < 1e-10:
        return None  # Not invertible
    
    return np.array([[d, -b], [-c, a]]) / det

# Test
A = np.array([[3, 1], [2, 1]], dtype=float)
A_inv = inverse_2x2(A)
print(f"A:\n{A}")
print(f"A^(-1):\n{A_inv}")
print(f"A @ A^(-1):\n{A @ A_inv}")  # Should be identity
```

---

### Solution B5: Visualize Transformation
```python
import matplotlib.pyplot as plt

def visualize_transformation_effect(A, points, title="Transformation"):
    """Visualize how matrix A transforms points."""
    transformed = (A @ points.T).T
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Original points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Original', zorder=3)
    
    # Transformed points
    ax.scatter(transformed[:, 0], transformed[:, 1], c='red', s=50, label='Transformed', zorder=3)
    
    # Arrows connecting points
    for i in range(len(points)):
        ax.annotate('', xy=transformed[i], xytext=points[i],
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title(title)
    plt.show()

# Test
points = np.random.randn(20, 2)
A = np.array([[2, 0.5], [0, 1.5]])
visualize_transformation_effect(A, points, "Shear + Scale")
```

---

### Solution B6: Compute Covariance Matrix
```python
def compute_covariance(X):
    """Compute the covariance matrix of data X."""
    N = X.shape[0]
    
    # Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # Covariance
    cov = (X_centered.T @ X_centered) / (N - 1)
    
    return cov

# Test
np.random.seed(42)
X = np.random.randn(100, 3)
cov = compute_covariance(X)
print(f"Our covariance:\n{cov}")
print(f"NumPy covariance:\n{np.cov(X.T)}")
print(f"Close? {np.allclose(cov, np.cov(X.T))}")
```

---

### Solution B7: Power Iteration
```python
def power_iteration(A, num_iterations=100):
    """Find dominant eigenvalue and eigenvector using power iteration."""
    n = A.shape[0]
    
    # Start with random vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        # Multiply
        Av = A @ v
        # Normalize
        v = Av / np.linalg.norm(Av)
    
    # Estimate eigenvalue using Rayleigh quotient
    eigenvalue = v @ A @ v
    
    return eigenvalue, v

# Test
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalue, eigenvector = power_iteration(A)

# Compare with numpy
np_eigenvalues, np_eigenvectors = np.linalg.eig(A)
print(f"Power iteration: λ = {eigenvalue:.4f}")
print(f"NumPy largest: λ = {np_eigenvalues.max():.4f}")
```

---

### Solution B8: Neural Network Layer
```python
class LinearLayer:
    """Implement a neural network linear layer."""
    
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros(output_dim)
        self.X = None  # Cache for backward
        
    def forward(self, X):
        """Forward pass: y = XW^T + b"""
        self.X = X
        return X @ self.W.T + self.b
    
    def backward(self, dy):
        """Backward pass: compute gradients."""
        # Gradient w.r.t. weights: dL/dW = dy^T @ X
        self.dW = dy.T @ self.X
        
        # Gradient w.r.t. bias: dL/db = sum over batch
        self.db = dy.sum(axis=0)
        
        # Gradient w.r.t. input: dL/dX = dy @ W
        dx = dy @ self.W
        
        return dx

# Test
layer = LinearLayer(3, 2)
X = np.random.randn(5, 3)  # Batch of 5, input dim 3
y = layer.forward(X)
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Backward
dy = np.random.randn(5, 2)
dx = layer.backward(dy)
print(f"dx shape: {dx.shape}")
print(f"dW shape: {layer.dW.shape}")
print(f"db shape: {layer.db.shape}")
```

---

### Solution B9: Low-Rank Approximation
```python
def low_rank_approximation(A, k):
    """Compute best rank-k approximation using SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only top k components
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct
    return U_k @ np.diag(s_k) @ Vt_k

def compression_ratio(original, approximation):
    """Compute memory savings."""
    m, n = original.shape
    k = np.linalg.matrix_rank(approximation)
    
    original_params = m * n
    # Low rank: U (m×k) + s (k) + V (k×n)
    approx_params = m * k + k + k * n
    
    return original_params / approx_params

# Test
A = np.random.randn(100, 50)
A_rank5 = low_rank_approximation(A, 5)

print(f"Original rank: {np.linalg.matrix_rank(A)}")
print(f"Approximation rank: {np.linalg.matrix_rank(A_rank5)}")
print(f"Compression ratio: {compression_ratio(A, A_rank5):.2f}x")
print(f"Reconstruction error: {np.linalg.norm(A - A_rank5) / np.linalg.norm(A):.4f}")
```

---

### Solution B10: Attention Mechanism
```python
def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """Implement scaled dot-product attention."""
    d_k = Q.shape[-1]
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Multiply by values
    output = attention_weights @ V
    
    return output

# Test
seq_len = 4
d_k, d_v = 8, 16

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output = scaled_dot_product_attention(Q, K, V)
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Output shape: {output.shape}")
```

---

## Part C: Conceptual Answers

### Answer C1: Why AB ≠ BA

**Geometric explanation:**

Consider:
- R = rotation by 90°
- S = scaling x by 2

**RS (scale then rotate):**
1. Point (1, 0) → (2, 0) after scaling
2. (2, 0) → (0, 2) after rotation
3. Final: (0, 2)

**SR (rotate then scale):**
1. Point (1, 0) → (0, 1) after rotation
2. (0, 1) → (0, 1) after scaling (y unchanged)
3. Final: (0, 1)

Different results because scaling "cares" about which axis a point is on, and rotation changes that.

### Answer C2: Rank and Information

**What information is lost:**
- The component of **x** in the null space of A
- If A has rank r, it collapses (n - r) dimensions to zero

**Can it be recovered?**
- No! Once collapsed, information is gone
- Multiple different inputs map to the same output

**Relation to ML compression:**
- Low-rank approximation deliberately loses information
- Keeps the "important" dimensions (largest singular values)
- Compression = controlled information loss
- Autoencoders use this: bottleneck forces compression

### Answer C3: Orthogonal Initialization

Benefits of orthogonal initialization:

1. **Preserves gradient magnitude:** ||Wx|| = ||x||
   - Prevents vanishing/exploding gradients in deep networks

2. **Diverse features:** Orthogonal columns are maximally different
   - Each neuron starts looking for something different

3. **Stable eigenvalues:** Eigenvalues have magnitude 1
   - Repeated multiplication doesn't blow up or vanish

4. **Information preservation:** No dimension collapse
   - Full rank means no information lost in forward pass

### Answer C4: Covariance Matrix Properties

**Why symmetric:**
$$\Sigma = \frac{1}{N-1}X^TX$$
$$(X^TX)^T = X^TX$$

The transpose of XᵀX equals itself.

**Why positive semi-definite:**

For any vector **v**:
$$\mathbf{v}^T\Sigma\mathbf{v} = \frac{1}{N-1}\mathbf{v}^TX^TX\mathbf{v} = \frac{1}{N-1}(X\mathbf{v})^T(X\mathbf{v}) = \frac{1}{N-1}\|X\mathbf{v}\|^2 \geq 0$$

Squared norm is always non-negative!

### Answer C5: Attention as Matrix Multiplication

**What QKᵀ computes:**
- Entry (i, j) = dot product of query i with key j
- Measures how much position i "wants to attend to" position j
- High score = strong relevance

**Why divide by √d:**
- Dot products grow with dimension (variance ∝ d)
- Without scaling, softmax would be too "sharp" (nearly one-hot)
- Scaling keeps gradients healthy and attention more distributed

**What multiplying by V accomplishes:**
- Weighted average of value vectors
- Weight = attention probability
- Each output position gets a mixture of all values
- Positions with high attention contribute more
