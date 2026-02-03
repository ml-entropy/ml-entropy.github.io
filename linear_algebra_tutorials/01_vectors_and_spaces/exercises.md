# Exercises: Vectors and Vector Spaces

## Part A: Theory Problems

### Exercise A1: Basic Vector Operations 🟢
Given **u** = [3, -1, 2] and **v** = [-1, 4, 1]:

a) Compute **u** + **v**
b) Compute 3**u** - 2**v**
c) Compute **u** · **v**
d) Compute ||**u**||₂ and ||**v**||₂

---

### Exercise A2: Dot Product Properties 🟢
Prove that the dot product is:
a) Commutative: **u** · **v** = **v** · **u**
b) Distributive: **u** · (**v** + **w**) = **u** · **v** + **u** · **w**
c) Homogeneous: (c**u**) · **v** = c(**u** · **v**)

---

### Exercise A3: Angle Calculation 🟡
a) Find the angle between **u** = [1, 2, 2] and **v** = [2, -1, 2]
b) Find a vector perpendicular to [3, 4] in 2D
c) Find all unit vectors perpendicular to [1, 0, 0] in 3D

---

### Exercise A4: Norm Derivation 🟡
Prove that for any vector **v**:
$$\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \|\mathbf{v}\|_1$$

*Hint: Start with explicit formulas and use properties of sums.*

---

### Exercise A5: Linear Independence 🟡
Determine if the following sets of vectors are linearly independent:

a) {[1, 2], [2, 4]}
b) {[1, 0, 0], [0, 1, 0], [0, 0, 1]}
c) {[1, 1, 0], [0, 1, 1], [1, 0, 1]}
d) {[1, 2, 3], [4, 5, 6], [7, 8, 9]}

---

### Exercise A6: Span and Basis 🟡
a) Does the vector [1, 1] lie in span({[1, 0], [0, 1]})?
b) Does [1, 2, 3] lie in span({[1, 0, 0], [0, 1, 0]})?
c) Find a basis for the plane x + 2y + z = 0 in ℝ³

---

### Exercise A7: Cauchy-Schwarz Inequality 🔴
Prove the Cauchy-Schwarz inequality:
$$|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\|$$

*Hint: Consider the vector **u** - t**v** and the fact that ||**u** - t**v**||² ≥ 0 for all t.*

---

### Exercise A8: Triangle Inequality 🔴
Using the Cauchy-Schwarz inequality, prove the triangle inequality:
$$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$$

---

### Exercise A9: Projection Derivation 🔴
Derive the formula for the projection of **u** onto **v**:
$$\text{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$$

*Hint: The projection should be parallel to **v** and **u** - proj should be perpendicular to **v**.*

---

### Exercise A10: Change of Basis 🔴
Let B = {[1, 1], [1, -1]} be a basis for ℝ².

a) Find the coordinates of **v** = [3, 1] in basis B
b) If a vector has coordinates [2, 3] in basis B, what are its standard coordinates?
c) Find the change of basis matrix from standard to B

---

## Part B: Coding Problems

### Exercise B1: Vector Operations 🟢
```python
def vector_operations(u, v):
    """
    Implement basic vector operations.
    
    Args:
        u, v: numpy arrays of same length
    
    Returns:
        dict with keys: 'add', 'dot', 'norm_u', 'norm_v', 'cosine_similarity'
    """
    # Your code here
    pass
```

---

### Exercise B2: Cosine Similarity Search 🟢
```python
def find_most_similar(query, vectors, labels):
    """
    Find the most similar vector to query using cosine similarity.
    
    Args:
        query: numpy array (the query vector)
        vectors: list of numpy arrays (candidate vectors)
        labels: list of strings (names for each vector)
    
    Returns:
        tuple: (best_label, best_similarity)
    """
    # Your code here
    pass
```

---

### Exercise B3: Linear Independence Check 🟡
```python
def is_linearly_independent(vectors, tolerance=1e-10):
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: list of numpy arrays (all same length)
        tolerance: threshold for numerical zero
    
    Returns:
        bool: True if linearly independent
    """
    # Your code here
    # Hint: Use the rank of the matrix formed by stacking vectors
    pass
```

---

### Exercise B4: Gram-Schmidt Preview 🟡
```python
def orthogonalize_2d(v1, v2):
    """
    Given two linearly independent 2D vectors, produce two orthogonal vectors
    that span the same space.
    
    Args:
        v1, v2: numpy arrays of shape (2,)
    
    Returns:
        tuple: (u1, u2) orthogonal vectors where u1 = v1
    """
    # Your code here
    # Hint: u2 = v2 - projection of v2 onto v1
    pass
```

---

### Exercise B5: Word Embedding Analogy 🟡
```python
def word_analogy(embeddings, word_a, word_b, word_c):
    """
    Solve analogy: word_a is to word_b as word_c is to ???
    
    Uses: result = word_b - word_a + word_c
    Then finds closest word.
    
    Args:
        embeddings: dict mapping words to numpy arrays
        word_a, word_b, word_c: strings
    
    Returns:
        str: the word that best completes the analogy
    """
    # Your code here
    pass
```

---

### Exercise B6: Implement Different Norms 🟢
```python
def compute_norms(v):
    """
    Compute L1, L2, and L-infinity norms of a vector.
    
    Args:
        v: numpy array
    
    Returns:
        dict with keys: 'L1', 'L2', 'Linf'
    """
    # Your code here (don't use np.linalg.norm - implement manually)
    pass
```

---

### Exercise B7: Basis Transformation 🔴
```python
def change_basis(v, old_basis, new_basis):
    """
    Express vector v (in old basis coordinates) in new basis coordinates.
    
    Args:
        v: numpy array (coordinates in old basis)
        old_basis: list of numpy arrays (old basis vectors in standard coords)
        new_basis: list of numpy arrays (new basis vectors in standard coords)
    
    Returns:
        numpy array: coordinates of v in new basis
    """
    # Your code here
    pass
```

---

### Exercise B8: Visualization - Span of Two Vectors 🟡
```python
def visualize_span(v1, v2, num_points=100):
    """
    Create a visualization showing the span of two 2D vectors.
    
    Plot:
    1. The two basis vectors
    2. A grid of points c1*v1 + c2*v2 for c1, c2 in [-2, 2]
    3. Indicate whether span is a line (dependent) or plane (independent)
    
    Args:
        v1, v2: numpy arrays of shape (2,)
        num_points: number of points per dimension
    """
    # Your code here
    pass
```

---

### Exercise B9: Simple Neural Layer 🔴
```python
def simple_layer_forward(X, W, b):
    """
    Implement forward pass of a linear neural network layer.
    
    y = X @ W + b
    
    where each row of X is an input vector.
    
    Args:
        X: numpy array of shape (batch_size, input_dim)
        W: numpy array of shape (input_dim, output_dim)
        b: numpy array of shape (output_dim,)
    
    Returns:
        numpy array of shape (batch_size, output_dim)
    """
    # Your code here
    pass

def simple_layer_backward(X, W, dy):
    """
    Compute gradients for the linear layer.
    
    Args:
        X: input from forward pass
        W: weights from forward pass
        dy: gradient of loss w.r.t. output y
    
    Returns:
        tuple: (dX, dW, db) gradients
    """
    # Your code here
    pass
```

---

### Exercise B10: Feature Vector Similarity Matrix 🟡
```python
def compute_similarity_matrix(vectors):
    """
    Compute pairwise cosine similarity matrix for a set of vectors.
    
    Args:
        vectors: numpy array of shape (n, d) where n is number of vectors
    
    Returns:
        numpy array of shape (n, n) with similarity[i,j] = cosine_sim(v_i, v_j)
    """
    # Your code here
    # Bonus: implement without explicit loops
    pass
```

---

## Part C: Conceptual Questions

### Question C1: Why Normalize?
In ML, we often normalize feature vectors before computing similarities. What problems could occur if we don't normalize?

---

### Question C2: L1 vs L2 Regularization
L1 regularization (penalizing ||**w**||₁) produces sparse weights, while L2 regularization (penalizing ||**w**||₂²) produces small but non-zero weights. Why does L1 encourage sparsity?

*Hint: Think about the geometry of the unit circles for different norms.*

---

### Question C3: High Dimensional Intuition
In high dimensions (e.g., d = 1000), most pairs of random vectors are nearly orthogonal. Why is this, and what are the implications for ML?

---

### Question C4: Embedding Dimension
Word embeddings like Word2Vec typically use 100-300 dimensions. Why not use more dimensions? Why not fewer?

---

### Question C5: Dot Product in Attention
The attention mechanism in transformers computes scores using dot products: score(q, k) = **q** · **k**. Why is dot product a reasonable choice for measuring relevance?

---

## Difficulty Legend
- 🟢 Easy: Direct application of concepts
- 🟡 Medium: Requires combining multiple concepts
- 🔴 Hard: Requires proof or deep understanding
