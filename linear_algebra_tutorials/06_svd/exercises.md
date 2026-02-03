# Exercises: Singular Value Decomposition (SVD)

## Part A: Theory Problems

### Exercise A1: Basic SVD Computation 🟢
For the matrix:
$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

a) Find the SVD (U, Σ, V) by inspection.
b) What are the singular values?
c) What is the rank of A?

---

### Exercise A2: SVD of 2×2 Matrix 🟡
Compute the SVD of:
$$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

a) Compute AᵀA and find its eigenvalues.
b) Find the singular values σ₁, σ₂.
c) Find V (eigenvectors of AᵀA).
d) Find U using uᵢ = Avᵢ/σᵢ.
e) Verify that A = UΣVᵀ.

---

### Exercise A3: Properties of Singular Values 🟡
Let A be a 4×3 matrix with singular values σ₁ = 5, σ₂ = 3, σ₃ = 1.

a) What is rank(A)?
b) What is ‖A‖₂ (spectral norm)?
c) What is ‖A‖_F (Frobenius norm)?
d) What is the condition number κ(A)?
e) What are the singular values of 2A? Of Aᵀ?

---

### Exercise A4: Low-Rank Approximation 🟡
Given:
$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 1 & 1 \end{bmatrix}$$

with SVD U, Σ = diag(σ₁, σ₂), Vᵀ where σ₁ = 3, σ₂ = 1.

a) Write the rank-1 approximation A₁ in terms of u₁, v₁, σ₁.
b) What is ‖A - A₁‖₂?
c) What is ‖A - A₁‖_F?

---

### Exercise A5: Geometric Interpretation 🟡
A 2×2 matrix A has SVD with:
- σ₁ = 4, σ₂ = 2
- v₁ = [1/√2, 1/√2]ᵀ, v₂ = [1/√2, -1/√2]ᵀ
- u₁ = [0, 1]ᵀ, u₂ = [1, 0]ᵀ

a) Describe geometrically what A does to a unit circle.
b) What is the image of the point [1, 0]ᵀ under A?
c) Along which input direction is stretching maximal?

---

### Exercise A6: Connection to Eigendecomposition 🟡
For a symmetric matrix S = Sᵀ:

a) Show that if S has eigenvalue λ with eigenvector v, then S has singular value |λ|.
b) What is the relationship between U, V in the SVD of S?
c) For S = [[2, 1], [1, 2]], find the SVD using this relationship.

---

### Exercise A7: Pseudoinverse Computation 🔴
For:
$$A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

a) Compute the SVD of A.
b) Compute the pseudoinverse A⁺.
c) Verify that AA⁺A = A.
d) Use A⁺ to find the least squares solution to Ax = [1, 2, 3]ᵀ.

---

### Exercise A8: Eckart-Young Theorem Proof 🔴
Prove that the rank-k truncated SVD gives the best rank-k approximation in Frobenius norm.

*Hint: Use the fact that for any matrices X, Y: ⟨X, Y⟩_F = tr(XᵀY), and that U and V are orthogonal.*

---

### Exercise A9: SVD of Outer Product 🔴
Let u ∈ ℝᵐ and v ∈ ℝⁿ be unit vectors. Consider A = uvᵀ (rank-1 matrix).

a) What is the rank of A?
b) What are the singular values of A?
c) What are the left and right singular vectors?
d) Generalize: If A = σuvᵀ for scalar σ > 0, what is the SVD?

---

### Exercise A10: SVD and Matrix Norms 🔴
Prove that:

a) ‖A‖₂ = σ₁ (largest singular value)
b) ‖A‖_F = √(Σσᵢ²) (Frobenius norm)
c) For any matrix B with rank(B) ≤ k: ‖A - B‖₂ ≥ σₖ₊₁

---

## Part B: Coding Problems

### Exercise B1: Manual SVD Computation 🟢
```python
def svd_via_eigendecomposition(A):
    """
    Compute SVD by eigendecomposition of A^T A.
    
    Args:
        A: numpy array of shape (m, n)
    
    Returns:
        tuple: (U, S, Vt) where A = U @ diag(S) @ Vt
    """
    # Your code here
    # Steps:
    # 1. Compute A^T A
    # 2. Find eigenvalues and eigenvectors
    # 3. Singular values = sqrt of eigenvalues
    # 4. V = eigenvectors
    # 5. Compute U from A @ V / S
    pass
```

---

### Exercise B2: Image Compression with SVD 🟢
```python
def compress_image_svd(image, k):
    """
    Compress a grayscale image using rank-k SVD approximation.
    
    Args:
        image: numpy array of shape (height, width)
        k: number of singular values to keep
    
    Returns:
        dict with keys:
            'compressed': reconstructed image
            'compression_ratio': original_size / compressed_size
            'relative_error': ||A - A_k||_F / ||A||_F
    """
    # Your code here
    pass

def visualize_compression(image, k_values):
    """
    Visualize image at different compression levels.
    
    Args:
        image: original grayscale image
        k_values: list of k values to try
    """
    # Your code here - show original and compressed versions
    pass
```

---

### Exercise B3: PCA via SVD 🟡
```python
def pca_svd(X, n_components):
    """
    Implement PCA using SVD (more numerically stable than eigendecomposition).
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        n_components: number of components to keep
    
    Returns:
        dict with keys:
            'components': principal components (n_components, n_features)
            'explained_variance': variance explained by each component
            'transformed': projected data (n_samples, n_components)
    """
    # Your code here
    pass
```

---

### Exercise B4: Pseudoinverse and Least Squares 🟡
```python
def solve_least_squares_svd(A, b, tol=1e-10):
    """
    Solve least squares problem Ax ≈ b using SVD pseudoinverse.
    
    Args:
        A: numpy array of shape (m, n)
        b: numpy array of shape (m,)
        tol: threshold for considering singular value as zero
    
    Returns:
        dict with keys:
            'x': least squares solution
            'residual': ||Ax - b||
            'rank': effective rank of A
    """
    # Your code here
    pass
```

---

### Exercise B5: Low-Rank Matrix Approximation 🟡
```python
def best_rank_k_approximation(A, k):
    """
    Find the best rank-k approximation of A using SVD.
    
    Args:
        A: numpy array of shape (m, n)
        k: target rank
    
    Returns:
        dict with keys:
            'A_k': rank-k approximation
            'singular_values_kept': first k singular values
            'approximation_error_frobenius': ||A - A_k||_F
            'approximation_error_spectral': ||A - A_k||_2
            'energy_retained': sum(sigma_k^2) / sum(sigma_all^2)
    """
    # Your code here
    pass
```

---

### Exercise B6: SVD for Recommender Systems 🟡
```python
def matrix_completion_svd(R, k, mask=None, n_iterations=10):
    """
    Complete a rating matrix using iterative SVD.
    
    Args:
        R: numpy array of shape (n_users, n_items) with some entries being NaN
        k: rank for approximation
        mask: boolean array indicating observed entries (True = observed)
        n_iterations: number of iterations
    
    Returns:
        dict with keys:
            'completed': filled matrix
            'U': user factors (n_users, k)
            'V': item factors (n_items, k)
    """
    # Your code here
    pass

def evaluate_recommendations(R_true, R_predicted, mask):
    """
    Evaluate recommendation quality on held-out entries.
    
    Returns:
        dict with RMSE, MAE on unobserved entries
    """
    # Your code here
    pass
```

---

### Exercise B7: Truncated/Randomized SVD 🟡
```python
def randomized_svd(A, k, n_oversamples=10, n_iter=2):
    """
    Compute approximate rank-k SVD using randomized algorithm.
    
    Much faster than full SVD for large matrices!
    
    Args:
        A: numpy array of shape (m, n)
        k: number of singular values to compute
        n_oversamples: extra columns for accuracy
        n_iter: power iterations for better accuracy
    
    Returns:
        tuple: (U, S, Vt) approximate SVD
    """
    # Your code here
    # Steps:
    # 1. Generate random matrix Omega (n x (k + n_oversamples))
    # 2. Form Y = A @ Omega
    # 3. Orthonormalize Y to get Q (using QR)
    # 4. Form B = Q^T @ A
    # 5. Compute SVD of B: B = U_B @ S @ Vt
    # 6. Return U = Q @ U_B, S, Vt
    pass
```

---

### Exercise B8: Latent Semantic Analysis 🔴
```python
def latent_semantic_analysis(documents, n_concepts=10):
    """
    Perform LSA on a collection of documents.
    
    Args:
        documents: list of strings (documents)
        n_concepts: number of latent concepts
    
    Returns:
        dict with keys:
            'term_doc_matrix': original TF-IDF matrix
            'document_concepts': documents in concept space
            'term_concepts': terms in concept space
            'concept_importance': singular values
            'vocabulary': list of terms
    """
    # Your code here
    pass

def find_similar_documents(query, lsa_model, top_k=5):
    """
    Find documents similar to query using LSA representation.
    
    Returns:
        list of (document_index, similarity_score)
    """
    # Your code here
    pass
```

---

### Exercise B9: SVD for Noise Reduction 🔴
```python
def denoise_matrix_svd(A, noise_threshold='auto'):
    """
    Denoise a matrix by removing small singular values.
    
    Args:
        A: noisy matrix
        noise_threshold: cutoff for singular values (or 'auto' for automatic selection)
    
    Returns:
        dict with keys:
            'denoised': denoised matrix
            'rank_original': original rank
            'rank_denoised': rank after denoising
            'singular_values': all singular values
            'threshold_used': actual threshold
    """
    # Your code here
    # For 'auto', consider using the elbow method or 
    # Gavish-Donoho optimal threshold
    pass
```

---

### Exercise B10: Visualize SVD Components 🟢
```python
def visualize_svd_components(A, n_components=5):
    """
    Visualize how SVD decomposes a matrix into rank-1 components.
    
    For each i, plot sigma_i * u_i * v_i^T and show cumulative reconstruction.
    
    Args:
        A: numpy array (ideally an image for visualization)
        n_components: number of components to show
    """
    # Your code here
    pass
```

---

## Part C: Conceptual Questions

### Question C1: SVD vs Eigendecomposition
When would you use SVD vs eigendecomposition? What are the key differences?

---

### Question C2: Choosing k in Low-Rank Approximation
You want to approximate a matrix with a rank-k approximation. How do you choose k? What criteria could you use?

---

### Question C3: SVD and Matrix Rank
Explain how SVD reveals the rank of a matrix. How would you determine the rank in practice given numerical errors?

---

### Question C4: Computational Complexity
SVD computation is O(min(mn², m²n)). Why might you want to use randomized SVD? When is it appropriate?

---

### Question C5: SVD in Deep Learning
How is SVD used in modern deep learning? Consider:
a) Weight initialization
b) Model compression
c) Analyzing learned representations

---

## Difficulty Legend
- 🟢 Easy: Direct application of concepts
- 🟡 Medium: Requires combining multiple concepts or multi-step reasoning
- 🔴 Hard: Requires derivation, proof, or deep understanding
