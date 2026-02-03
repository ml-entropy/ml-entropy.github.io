# Solutions: Vectors and Vector Spaces

## Part A: Theory Solutions

### Solution A1: Basic Vector Operations

Given **u** = [3, -1, 2] and **v** = [-1, 4, 1]:

**a) u + v:**
$$\mathbf{u} + \mathbf{v} = [3+(-1), -1+4, 2+1] = [2, 3, 3]$$

**b) 3u - 2v:**
$$3\mathbf{u} - 2\mathbf{v} = [9, -3, 6] - [-2, 8, 2] = [11, -11, 4]$$

**c) u · v:**
$$\mathbf{u} \cdot \mathbf{v} = 3(-1) + (-1)(4) + 2(1) = -3 - 4 + 2 = -5$$

**d) Norms:**
$$\|\mathbf{u}\|_2 = \sqrt{3^2 + (-1)^2 + 2^2} = \sqrt{9 + 1 + 4} = \sqrt{14}$$
$$\|\mathbf{v}\|_2 = \sqrt{(-1)^2 + 4^2 + 1^2} = \sqrt{1 + 16 + 1} = \sqrt{18} = 3\sqrt{2}$$

---

### Solution A2: Dot Product Properties

**a) Commutativity:**
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \sum_{i=1}^n v_i u_i = \mathbf{v} \cdot \mathbf{u}$$

This follows directly from commutativity of real number multiplication.

**b) Distributivity:**
$$\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \sum_{i=1}^n u_i(v_i + w_i) = \sum_{i=1}^n (u_i v_i + u_i w_i) = \sum_{i=1}^n u_i v_i + \sum_{i=1}^n u_i w_i = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$$

**c) Homogeneity:**
$$(c\mathbf{u}) \cdot \mathbf{v} = \sum_{i=1}^n (cu_i)v_i = c\sum_{i=1}^n u_i v_i = c(\mathbf{u} \cdot \mathbf{v})$$

---

### Solution A3: Angle Calculation

**a) Angle between u = [1, 2, 2] and v = [2, -1, 2]:**

$$\mathbf{u} \cdot \mathbf{v} = 1(2) + 2(-1) + 2(2) = 2 - 2 + 4 = 4$$

$$\|\mathbf{u}\| = \sqrt{1 + 4 + 4} = 3, \quad \|\mathbf{v}\| = \sqrt{4 + 1 + 4} = 3$$

$$\cos\theta = \frac{4}{3 \cdot 3} = \frac{4}{9}$$

$$\theta = \arccos\left(\frac{4}{9}\right) \approx 63.6°$$

**b) Vector perpendicular to [3, 4]:**

For perpendicularity: [a, b] · [3, 4] = 0 ⟹ 3a + 4b = 0 ⟹ a = -4b/3

Choose b = 3: **w** = [-4, 3] ✓ (Check: 3(-4) + 4(3) = 0)

**c) Unit vectors perpendicular to [1, 0, 0]:**

Need [a, b, c] · [1, 0, 0] = a = 0 and a² + b² + c² = 1.

So any vector [0, b, c] with b² + c² = 1.

Parametric form: [0, cos θ, sin θ] for any θ ∈ [0, 2π)

---

### Solution A4: Norm Inequalities

**Prove:** ||**v**||∞ ≤ ||**v**||₂ ≤ ||**v**||₁

Let **v** = [v₁, ..., vₙ] and let vₘ = max|vᵢ|.

**Part 1: ||v||∞ ≤ ||v||₂**

$$\|\mathbf{v}\|_2^2 = \sum_{i=1}^n v_i^2 \geq v_m^2 = \|\mathbf{v}\|_\infty^2$$

Taking square roots (both sides positive): ||**v**||₂ ≥ ||**v**||∞ ✓

**Part 2: ||v||₂ ≤ ||v||₁**

$$\|\mathbf{v}\|_1^2 = \left(\sum_{i=1}^n |v_i|\right)^2 = \sum_{i=1}^n v_i^2 + 2\sum_{i<j}|v_i||v_j|$$

Since the cross terms are non-negative:
$$\|\mathbf{v}\|_1^2 \geq \sum_{i=1}^n v_i^2 = \|\mathbf{v}\|_2^2$$

Taking square roots: ||**v**||₁ ≥ ||**v**||₂ ✓

---

### Solution A5: Linear Independence

**a) {[1, 2], [2, 4]}:**
[2, 4] = 2[1, 2], so **DEPENDENT** (parallel vectors)

**b) {[1, 0, 0], [0, 1, 0], [0, 0, 1]}:**
Standard basis, **INDEPENDENT**

To verify: c₁[1,0,0] + c₂[0,1,0] + c₃[0,0,1] = [0,0,0] implies [c₁, c₂, c₃] = [0,0,0]

**c) {[1, 1, 0], [0, 1, 1], [1, 0, 1]}:**

Form matrix and compute determinant:
$$\det\begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} = 1(1-0) - 0 + 1(1-0) = 2 \neq 0$$

**INDEPENDENT**

**d) {[1, 2, 3], [4, 5, 6], [7, 8, 9]}:**

$$\det\begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix} = 1(45-48) - 4(18-24) + 7(12-15) = -3 + 24 - 21 = 0$$

**DEPENDENT** (row 3 = 2×row 2 - row 1)

---

### Solution A6: Span and Basis

**a) Does [1, 1] lie in span({[1, 0], [0, 1]})?**

Need c₁[1, 0] + c₂[0, 1] = [1, 1]
⟹ c₁ = 1, c₂ = 1 ✓

**Yes**, [1, 1] = 1·[1,0] + 1·[0,1]

**b) Does [1, 2, 3] lie in span({[1, 0, 0], [0, 1, 0]})?**

Need c₁[1, 0, 0] + c₂[0, 1, 0] = [1, 2, 3]
⟹ [c₁, c₂, 0] = [1, 2, 3]

But the third component would be 0 ≠ 3.

**No**, [1, 2, 3] is not in the span (it's the xy-plane, and [1,2,3] has nonzero z).

**c) Basis for plane x + 2y + z = 0:**

Express z = -x - 2y. Parametrize with x = s, y = t:
$$\mathbf{v} = [s, t, -s-2t] = s[1, 0, -1] + t[0, 1, -2]$$

Basis: {[1, 0, -1], [0, 1, -2]}

Verify: These are linearly independent (not parallel) and both satisfy x + 2y + z = 0.

---

### Solution A7: Cauchy-Schwarz Inequality

**Prove:** |**u** · **v**| ≤ ||**u**|| ||**v**||

Consider the function f(t) = ||**u** - t**v**||² for real t.

Since squared norm is always non-negative: f(t) ≥ 0 for all t.

Expand:
$$f(t) = (\mathbf{u} - t\mathbf{v}) \cdot (\mathbf{u} - t\mathbf{v})$$
$$= \mathbf{u} \cdot \mathbf{u} - 2t(\mathbf{u} \cdot \mathbf{v}) + t^2(\mathbf{v} \cdot \mathbf{v})$$
$$= \|\mathbf{u}\|^2 - 2t(\mathbf{u} \cdot \mathbf{v}) + t^2\|\mathbf{v}\|^2$$

This is a quadratic in t: at² + bt + c ≥ 0 where:
- a = ||**v**||²
- b = -2(**u** · **v**)
- c = ||**u**||²

For a quadratic to be ≥ 0 for all t, its discriminant must be ≤ 0:
$$b^2 - 4ac \leq 0$$
$$4(\mathbf{u} \cdot \mathbf{v})^2 - 4\|\mathbf{v}\|^2\|\mathbf{u}\|^2 \leq 0$$
$$(\mathbf{u} \cdot \mathbf{v})^2 \leq \|\mathbf{u}\|^2\|\mathbf{v}\|^2$$

Taking square roots:
$$|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\| \quad \blacksquare$$

---

### Solution A8: Triangle Inequality

**Prove:** ||**u** + **v**|| ≤ ||**u**|| + ||**v**||

Start with:
$$\|\mathbf{u} + \mathbf{v}\|^2 = (\mathbf{u} + \mathbf{v}) \cdot (\mathbf{u} + \mathbf{v})$$
$$= \|\mathbf{u}\|^2 + 2(\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2$$

By Cauchy-Schwarz: **u** · **v** ≤ |**u** · **v**| ≤ ||**u**|| ||**v**||

Therefore:
$$\|\mathbf{u} + \mathbf{v}\|^2 \leq \|\mathbf{u}\|^2 + 2\|\mathbf{u}\|\|\mathbf{v}\| + \|\mathbf{v}\|^2 = (\|\mathbf{u}\| + \|\mathbf{v}\|)^2$$

Taking square roots:
$$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\| \quad \blacksquare$$

---

### Solution A9: Projection Formula

**Goal:** Find proj_v(**u**) such that it lies along **v** and **u** - proj is perpendicular to **v**.

Since projection is along **v**, write: proj_v(**u**) = c**v** for some scalar c.

For perpendicularity:
$$(\mathbf{u} - c\mathbf{v}) \cdot \mathbf{v} = 0$$
$$\mathbf{u} \cdot \mathbf{v} - c(\mathbf{v} \cdot \mathbf{v}) = 0$$
$$c = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}$$

Therefore:
$$\text{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v} \quad \blacksquare$$

---

### Solution A10: Change of Basis

Basis B = {[1, 1], [1, -1]}

**a) Coordinates of v = [3, 1] in basis B:**

Need c₁[1, 1] + c₂[1, -1] = [3, 1]
⟹ c₁ + c₂ = 3 and c₁ - c₂ = 1

Adding: 2c₁ = 4 ⟹ c₁ = 2
Subtracting: 2c₂ = 2 ⟹ c₂ = 1

Coordinates in B: [2, 1]

**b) Coordinates [2, 3] in B → standard coordinates:**

Standard = 2[1, 1] + 3[1, -1] = [2, 2] + [3, -3] = [5, -1]

**c) Change of basis matrix from standard to B:**

If P has columns that are basis B vectors:
$$P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

Then P⁻¹ converts from standard to B:
$$P^{-1} = \frac{1}{-2}\begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{bmatrix}$$

---

## Part B: Coding Solutions

### Solution B1: Vector Operations
```python
import numpy as np

def vector_operations(u, v):
    """Implement basic vector operations."""
    u, v = np.array(u), np.array(v)
    
    return {
        'add': u + v,
        'dot': np.dot(u, v),
        'norm_u': np.linalg.norm(u),
        'norm_v': np.linalg.norm(v),
        'cosine_similarity': np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    }

# Test
u = [3, -1, 2]
v = [-1, 4, 1]
result = vector_operations(u, v)
print(f"u + v = {result['add']}")
print(f"u · v = {result['dot']}")
print(f"||u|| = {result['norm_u']:.4f}")
print(f"||v|| = {result['norm_v']:.4f}")
print(f"cosine similarity = {result['cosine_similarity']:.4f}")
```

---

### Solution B2: Cosine Similarity Search
```python
def find_most_similar(query, vectors, labels):
    """Find the most similar vector to query using cosine similarity."""
    query = np.array(query)
    query_norm = np.linalg.norm(query)
    
    best_label = None
    best_similarity = -1  # Cosine similarity ranges from -1 to 1
    
    for vec, label in zip(vectors, labels):
        vec = np.array(vec)
        similarity = np.dot(query, vec) / (query_norm * np.linalg.norm(vec))
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_label = label
    
    return (best_label, best_similarity)

# Test
vectors = [[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0], [0.5, 0.5, 0.707]]
labels = ['x-axis', 'y-axis', 'diagonal-xy', '3d-diagonal']
query = [0.7, 0.7, 0]

result = find_most_similar(query, vectors, labels)
print(f"Most similar to {query}: {result[0]} (similarity: {result[1]:.4f})")
```

---

### Solution B3: Linear Independence Check
```python
def is_linearly_independent(vectors, tolerance=1e-10):
    """Check if a set of vectors is linearly independent."""
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix, tol=tolerance)
    return rank == len(vectors)

# Test
print("Testing linear independence:")
print(f"[1,2], [2,4]: {is_linearly_independent([[1,2], [2,4]])}")  # False
print(f"[1,0], [0,1]: {is_linearly_independent([[1,0], [0,1]])}")  # True
print(f"[1,0,0], [0,1,0], [0,0,1]: {is_linearly_independent([[1,0,0], [0,1,0], [0,0,1]])}")  # True
```

---

### Solution B4: Gram-Schmidt Preview
```python
def orthogonalize_2d(v1, v2):
    """Produce orthogonal vectors from two independent vectors."""
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    
    u1 = v1
    
    # Project v2 onto u1 and subtract
    proj = (np.dot(v2, u1) / np.dot(u1, u1)) * u1
    u2 = v2 - proj
    
    return (u1, u2)

# Test
v1, v2 = [1, 0], [1, 1]
u1, u2 = orthogonalize_2d(v1, v2)
print(f"u1 = {u1}")
print(f"u2 = {u2}")
print(f"u1 · u2 = {np.dot(u1, u2)}")  # Should be ~0
```

---

### Solution B5: Word Embedding Analogy
```python
def word_analogy(embeddings, word_a, word_b, word_c):
    """Solve: word_a is to word_b as word_c is to ???"""
    # Compute target vector
    target = embeddings[word_b] - embeddings[word_a] + embeddings[word_c]
    
    # Find closest word (excluding input words)
    best_word = None
    best_similarity = -2
    
    exclude = {word_a, word_b, word_c}
    
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        
        similarity = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
        if similarity > best_similarity:
            best_similarity = similarity
            best_word = word
    
    return best_word

# Test with simple embeddings
embeddings = {
    'king': np.array([0.9, 0.8]),
    'queen': np.array([0.85, 0.75]),
    'man': np.array([0.7, 0.2]),
    'woman': np.array([0.65, 0.15]),
}

result = word_analogy(embeddings, 'man', 'king', 'woman')
print(f"man:king :: woman:{result}")  # Should be 'queen'
```

---

### Solution B6: Implement Different Norms
```python
def compute_norms(v):
    """Compute L1, L2, and L-infinity norms manually."""
    v = np.array(v)
    
    l1 = sum(abs(x) for x in v)
    l2 = sum(x**2 for x in v) ** 0.5
    linf = max(abs(x) for x in v)
    
    return {'L1': l1, 'L2': l2, 'Linf': linf}

# Test
v = [3, -4, 0]
norms = compute_norms(v)
print(f"v = {v}")
print(f"L1 norm: {norms['L1']}")    # |3| + |-4| + |0| = 7
print(f"L2 norm: {norms['L2']}")    # sqrt(9 + 16) = 5
print(f"L∞ norm: {norms['Linf']}")  # max(3, 4, 0) = 4
```

---

### Solution B7: Basis Transformation
```python
def change_basis(v, old_basis, new_basis):
    """Express vector from old basis in new basis."""
    # First convert to standard coordinates
    old_matrix = np.column_stack(old_basis)
    standard_coords = old_matrix @ v
    
    # Then convert to new basis coordinates
    new_matrix = np.column_stack(new_basis)
    new_coords = np.linalg.solve(new_matrix, standard_coords)
    
    return new_coords

# Test
old_basis = [np.array([1, 0]), np.array([0, 1])]  # Standard
new_basis = [np.array([1, 1]), np.array([1, -1])]  # Rotated

v_old = np.array([3, 1])  # In standard coordinates
v_new = change_basis(v_old, old_basis, new_basis)
print(f"v in standard: {v_old}")
print(f"v in new basis: {v_new}")  # Should be [2, 1]
```

---

### Solution B8: Visualization - Span of Two Vectors
```python
import matplotlib.pyplot as plt

def visualize_span(v1, v2, num_points=20):
    """Visualize the span of two 2D vectors."""
    v1, v2 = np.array(v1), np.array(v2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check independence
    det = v1[0]*v2[1] - v1[1]*v2[0]
    is_independent = abs(det) > 1e-10
    
    # Generate span points
    coeffs = np.linspace(-2, 2, num_points)
    
    if is_independent:
        # Full 2D span - plot grid
        for c1 in coeffs:
            for c2 in coeffs:
                point = c1 * v1 + c2 * v2
                ax.plot(point[0], point[1], 'g.', markersize=3, alpha=0.5)
        title = "Span = R² (Linearly Independent)"
    else:
        # 1D span - just a line
        for c1 in coeffs:
            point = c1 * v1
            ax.plot(point[0], point[1], 'g.', markersize=5)
        title = "Span = Line (Linearly Dependent)"
    
    # Plot basis vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.02, label=f'v1 = {v1}')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.02, label=f'v2 = {v2}')
    
    max_val = max(np.abs(v1).max(), np.abs(v2).max()) * 2.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(title)
    ax.legend()
    plt.show()

# Test with independent vectors
visualize_span([1, 0], [0, 1])

# Test with dependent vectors
visualize_span([1, 2], [2, 4])
```

---

### Solution B9: Simple Neural Layer
```python
def simple_layer_forward(X, W, b):
    """Forward pass of linear layer: y = X @ W + b"""
    return X @ W + b

def simple_layer_backward(X, W, dy):
    """Compute gradients for linear layer."""
    # dL/dX = dL/dy @ W^T (chain rule)
    dX = dy @ W.T
    
    # dL/dW = X^T @ dL/dy (chain rule)
    dW = X.T @ dy
    
    # dL/db = sum of dL/dy over batch
    db = dy.sum(axis=0)
    
    return (dX, dW, db)

# Test
np.random.seed(42)
X = np.random.randn(4, 3)   # 4 samples, 3 features
W = np.random.randn(3, 2)   # 3 inputs, 2 outputs
b = np.random.randn(2)

# Forward
y = simple_layer_forward(X, W, b)
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Backward (assuming some gradient from downstream)
dy = np.random.randn(4, 2)
dX, dW, db = simple_layer_backward(X, W, dy)
print(f"dX shape: {dX.shape}")
print(f"dW shape: {dW.shape}")
print(f"db shape: {db.shape}")
```

---

### Solution B10: Similarity Matrix
```python
def compute_similarity_matrix(vectors):
    """Compute pairwise cosine similarity matrix."""
    vectors = np.array(vectors)
    
    # Normalize each vector
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    
    # Similarity matrix is normalized @ normalized.T
    similarity = normalized @ normalized.T
    
    return similarity

# Test
vectors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1]
])

sim_matrix = compute_similarity_matrix(vectors)
print("Similarity Matrix:")
print(np.round(sim_matrix, 3))
```

---

## Part C: Conceptual Answers

### Answer C1: Why Normalize?

Without normalization:
1. **Magnitude dominates similarity**: A very large vector could have high dot product with everything
2. **Features with larger scales dominate**: If income is in thousands and age in years, income dominates
3. **Unfair comparisons**: Documents with more words appear more similar to everything

Normalization ensures we compare *directions* not magnitudes, which is often what we care about semantically.

---

### Answer C2: L1 vs L2 Regularization

**Geometric explanation:**
- L1 "unit ball" is a diamond with **corners on the axes**
- L2 "unit ball" is a smooth sphere

When the loss function contours intersect the constraint region:
- L1: Most likely to touch at a **corner** (where some coordinates = 0) → sparsity
- L2: Most likely to touch **on the smooth surface** (all coordinates small but nonzero)

**Gradient explanation:**
- L1 gradient is constant (±1) regardless of weight magnitude
- L2 gradient is proportional to weight magnitude
- Small L1 weights get pushed to zero with same force as large weights

---

### Answer C3: High Dimensional Intuition

In high dimensions:
- Most of the "volume" of a hypersphere concentrates near the surface
- Random vectors tend to be nearly orthogonal (dot product ≈ 0)

**Why?** For random unit vectors, E[**u** · **v**] = 0 and Var[**u** · **v**] ≈ 1/d

As d → ∞, dot products concentrate around 0.

**ML implications:**
- Random initializations are nearly orthogonal (good for diversity)
- Need many dimensions for embeddings to capture relationships
- Curse of dimensionality: distances become less meaningful

---

### Answer C4: Embedding Dimension

**Too few dimensions (e.g., 10):**
- Not enough "room" to encode all relationships
- Forced to sacrifice some semantic structure
- Similar words might collide

**Too many dimensions (e.g., 10,000):**
- More parameters = more data needed
- Risk of overfitting
- Computational cost
- Diminishing returns (words don't have infinite distinct properties)

**Sweet spot (100-300):**
- Enough capacity for rich semantics
- Regularization through limited dimensions
- Empirically works well

---

### Answer C5: Dot Product in Attention

Dot product is a good attention score because:

1. **Measures alignment**: High score when query and key "point in same direction"
2. **Fast to compute**: Single matrix multiplication for all pairs
3. **Differentiable**: Smooth gradients for learning
4. **Interpretable**: Score = how much key matches what query is looking for

The query **q** encodes "what am I looking for?" and key **k** encodes "what do I contain?" The dot product measures the match.

(Note: Scaled dot-product attention divides by √d to control variance in high dimensions, addressing the issue from C3.)
