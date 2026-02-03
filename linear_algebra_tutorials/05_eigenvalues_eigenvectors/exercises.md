# Exercises: Eigenvalues and Eigenvectors

## Part A: Theory Problems

### Exercise A1: Basic Eigenvalue Calculation 🟢
Find all eigenvalues and corresponding eigenvectors for:

a) $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$

b) $B = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$

c) $C = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$

---

### Exercise A2: 3×3 Eigenvalues 🟢
Find the eigenvalues of:

$A = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ 0 & 1 & 2 \end{bmatrix}$

*Hint: This is a triangular matrix.*

---

### Exercise A3: Eigenvalue Properties 🟡
Let A be a 3×3 matrix with eigenvalues λ₁ = 2, λ₂ = -1, λ₃ = 3.

a) What is trace(A)?
b) What is det(A)?
c) What are the eigenvalues of A²?
d) What are the eigenvalues of A⁻¹?
e) What are the eigenvalues of A - 2I?

---

### Exercise A4: Symmetric Matrix 🟡
For the symmetric matrix:
$$S = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

a) Find the eigenvalues.
b) Find the eigenvectors.
c) Verify that the eigenvectors are orthogonal.
d) Write the spectral decomposition S = QΛQᵀ.

---

### Exercise A5: Diagonalization 🟡
Given:
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$$

a) Find eigenvalues and eigenvectors.
b) Write A = PDP⁻¹.
c) Use this to compute A⁵.

---

### Exercise A6: Complex Eigenvalues 🟡
For the rotation matrix:
$$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

a) Find the characteristic polynomial.
b) Find the complex eigenvalues.
c) Verify that eigenvalues are complex conjugates.
d) What geometric transformation does this represent?

---

### Exercise A7: Defective Matrix 🔴
Show that the matrix:
$$A = \begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}$$

a) Has eigenvalue λ = 3 with algebraic multiplicity 2.
b) Has only one linearly independent eigenvector (geometric multiplicity 1).
c) Therefore is not diagonalizable.

---

### Exercise A8: Power Iteration Analysis 🔴
Consider:
$$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

a) Find the eigenvalues and eigenvectors.
b) If we start with **x₀** = [1, 0]ᵀ, express **x₀** in terms of eigenvectors.
c) Compute A**x₀**, A²**x₀**, and A³**x₀**.
d) Verify that the direction converges to the dominant eigenvector.

---

### Exercise A9: Eigenvalues and Definiteness 🔴
Prove that:
a) A symmetric matrix is positive definite if and only if all eigenvalues are positive.
b) A symmetric matrix is positive semi-definite if and only if all eigenvalues are non-negative.

*Hint: Use the spectral decomposition.*

---

### Exercise A10: Eigenvalues of Special Matrices 🔴
Prove the following:

a) The eigenvalues of a unitary matrix (UᴴU = I) have absolute value 1.
b) The eigenvalues of a projection matrix P (where P² = P) are 0 or 1.
c) If A is nilpotent (Aⁿ = 0 for some n), all eigenvalues are 0.

---

## Part B: Coding Problems

### Exercise B1: Eigenvalue Computation 🟢
```python
def find_eigenvalues_2x2(A):
    """
    Find eigenvalues of a 2x2 matrix analytically.
    
    Args:
        A: numpy array of shape (2, 2)
    
    Returns:
        tuple: (lambda1, lambda2) eigenvalues (may be complex)
    """
    # Your code here
    # Use the quadratic formula on the characteristic polynomial
    pass

def find_eigenvectors_2x2(A, eigenvalue):
    """
    Find eigenvector for a given eigenvalue of a 2x2 matrix.
    
    Args:
        A: numpy array of shape (2, 2)
        eigenvalue: the eigenvalue (may be complex)
    
    Returns:
        numpy array: normalized eigenvector
    """
    # Your code here
    pass
```

---

### Exercise B2: Visualize Eigenvectors 🟢
```python
def visualize_eigenvectors(A):
    """
    Visualize how a 2x2 matrix transforms its eigenvectors vs other vectors.
    
    Plot:
    1. A circle of unit vectors before transformation
    2. The transformed ellipse
    3. Highlight eigenvectors showing they only scale
    
    Args:
        A: numpy array of shape (2, 2)
    """
    # Your code here
    pass
```

---

### Exercise B3: Power Iteration 🟡
```python
def power_iteration(A, max_iterations=100, tolerance=1e-10):
    """
    Find the dominant eigenvalue and eigenvector using power iteration.
    
    Args:
        A: numpy array of shape (n, n)
        max_iterations: maximum number of iterations
        tolerance: convergence threshold
    
    Returns:
        dict with keys:
            'eigenvalue': float
            'eigenvector': numpy array
            'iterations': int (number of iterations to converge)
            'convergence_history': list of eigenvalue estimates
    """
    # Your code here
    pass
```

---

### Exercise B4: Inverse Iteration 🟡
```python
def inverse_iteration(A, shift=0, max_iterations=100, tolerance=1e-10):
    """
    Find eigenvalue closest to 'shift' using inverse iteration.
    
    Args:
        A: numpy array of shape (n, n)
        shift: target value to find closest eigenvalue
        max_iterations: maximum number of iterations
        tolerance: convergence threshold
    
    Returns:
        dict with keys:
            'eigenvalue': float
            'eigenvector': numpy array
            'iterations': int
    """
    # Your code here
    # Hint: Apply power iteration to (A - shift*I)^(-1)
    pass
```

---

### Exercise B5: PCA from Scratch 🟡
```python
def pca_eigen(X, n_components):
    """
    Implement PCA using eigendecomposition of covariance matrix.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        n_components: number of principal components to keep
    
    Returns:
        dict with keys:
            'components': numpy array of shape (n_components, n_features) - principal components
            'explained_variance': numpy array of shape (n_components,) - eigenvalues
            'explained_variance_ratio': numpy array of shape (n_components,)
            'transformed': numpy array of shape (n_samples, n_components) - projected data
    """
    # Your code here
    # Steps:
    # 1. Center the data
    # 2. Compute covariance matrix
    # 3. Find eigenvalues/eigenvectors
    # 4. Sort by eigenvalue
    # 5. Project data
    pass
```

---

### Exercise B6: PageRank Algorithm 🟡
```python
def pagerank(adjacency_matrix, damping=0.85, max_iterations=100, tolerance=1e-8):
    """
    Compute PageRank using the power iteration method.
    
    Args:
        adjacency_matrix: numpy array of shape (n, n), A[i,j] = 1 if j links to i
        damping: probability of following a link (vs random teleport)
        max_iterations: maximum number of iterations
        tolerance: convergence threshold
    
    Returns:
        dict with keys:
            'ranks': numpy array of PageRank scores (sum to 1)
            'iterations': number of iterations to converge
    """
    # Your code here
    # The PageRank matrix is: M = d * (D^(-1) @ A) + (1-d)/n * ones
    # where D is diagonal matrix of out-degrees
    pass
```

---

### Exercise B7: Eigenvalue Sensitivity 🟡
```python
def eigenvalue_sensitivity(A, epsilon=1e-5):
    """
    Analyze how sensitive eigenvalues are to small perturbations.
    
    Args:
        A: numpy array of shape (n, n)
        epsilon: perturbation magnitude
    
    Returns:
        dict with keys:
            'original_eigenvalues': numpy array
            'perturbed_eigenvalues': numpy array (for random perturbation)
            'condition_numbers': numpy array (condition number for each eigenvalue)
    """
    # Your code here
    pass
```

---

### Exercise B8: Stability Analysis 🔴
```python
def analyze_linear_system_stability(A):
    """
    Analyze the stability of discrete linear system x_{t+1} = A * x_t.
    
    Args:
        A: numpy array of shape (n, n)
    
    Returns:
        dict with keys:
            'eigenvalues': numpy array
            'spectral_radius': float (max |eigenvalue|)
            'is_stable': bool (spectral radius < 1)
            'dominant_mode': int (index of dominant eigenvalue)
            'convergence_rate': float (|lambda_2 / lambda_1|)
    """
    # Your code here
    pass

def simulate_linear_system(A, x0, n_steps):
    """
    Simulate the linear system and visualize the trajectory.
    
    Args:
        A: numpy array of shape (n, n)
        x0: numpy array of shape (n,) - initial state
        n_steps: number of time steps
    
    Returns:
        numpy array of shape (n_steps+1, n) - trajectory
    """
    # Your code here
    pass
```

---

### Exercise B9: Spectral Clustering 🔴
```python
def spectral_clustering(similarity_matrix, n_clusters):
    """
    Perform spectral clustering on a similarity matrix.
    
    Args:
        similarity_matrix: numpy array of shape (n, n), symmetric
        n_clusters: number of clusters
    
    Returns:
        numpy array of shape (n,): cluster assignments
    """
    # Your code here
    # Steps:
    # 1. Compute degree matrix D
    # 2. Compute Laplacian L = D - W (or normalized version)
    # 3. Find bottom k eigenvectors of L
    # 4. Run k-means on the eigenvector matrix
    pass
```

---

### Exercise B10: QR Algorithm Visualization 🔴
```python
def qr_iteration(A, max_iterations=50):
    """
    Implement the QR algorithm for finding all eigenvalues.
    
    The QR algorithm repeatedly:
    1. Decompose A = QR
    2. Update A = RQ
    
    Args:
        A: numpy array of shape (n, n)
        max_iterations: number of iterations
    
    Returns:
        dict with keys:
            'eigenvalues': numpy array (diagonal of final A)
            'iterations': list of A matrices at each step
    """
    # Your code here
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Eigenvalue Interpretation
Explain in plain English what it means for a matrix to have:
a) All positive eigenvalues
b) Eigenvalue 0
c) Complex eigenvalues
d) Repeated eigenvalues

---

### Question C2: PCA Eigenvalues
In PCA, we compute eigenvectors of the covariance matrix. 
a) What does each eigenvalue represent in terms of the data?
b) Why do we sort eigenvalues in descending order?
c) How do we decide how many principal components to keep?

---

### Question C3: PageRank Convergence
The PageRank algorithm always converges to a unique solution. Why?
a) What property of the modified link matrix guarantees convergence?
b) What is the role of the "damping factor" (typically 0.85)?
c) What eigenvalue of the PageRank matrix is the ranking vector associated with?

---

### Question C4: RNN Gradient Problems
Eigenvalues of the recurrent weight matrix affect RNN training.
a) Why do eigenvalues > 1 cause "exploding gradients"?
b) Why do eigenvalues < 1 cause "vanishing gradients"?
c) How do LSTM/GRU architectures address this?

---

### Question C5: Optimization Landscape
The Hessian of a loss function at a critical point tells us about the geometry.
a) What do the eigenvalues of the Hessian tell us about the critical point?
b) What does a "near-zero" eigenvalue indicate?
c) How does this relate to the choice of learning rate?

---

## Difficulty Legend
- 🟢 Easy: Direct application of concepts
- 🟡 Medium: Requires combining multiple concepts or multi-step reasoning
- 🔴 Hard: Requires derivation, proof, or deep understanding
