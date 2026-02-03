# Solutions: Orthogonality and Projections

## Part A: Theory Solutions

### Solution A1: Orthogonality Check

**a)** **u** · **v** = 1(3) + 2(0) + 3(-1) = 3 + 0 - 3 = 0 ✓ **Orthogonal**

**b)** **u** · **v** = 1(1) + 1(-1) + 1(1) + 1(-1) = 1 - 1 + 1 - 1 = 0 ✓ **Orthogonal**

**c)** **u** · **v** = 2(1) + (-1)(2) = 2 - 2 = 0 ✓ **Orthogonal**

All three pairs are orthogonal!

---

### Solution A2: Orthogonal Matrix Verification

**a)** Q = (1/√2)[[1, 1], [1, -1]]

QᵀQ = (1/2)[[1, 1], [1, -1]][[1, 1], [1, -1]] = (1/2)[[2, 0], [0, 2]] = I ✓

det(Q) = (1/2)(1(-1) - 1(1)) = (1/2)(-2) = **-1** → **Reflection**

**b)** R is a standard rotation matrix by construction.

RᵀR = I (can verify)

det(R) = cos²(30°) + sin²(30°) = **1** → **Rotation**

**c)** P swaps first two rows.

PᵀP = PᵀP = P² = [[0,1,0],[1,0,0],[0,0,1]][[0,1,0],[1,0,0],[0,0,1]] = I ✓

det(P) = **-1** (one swap) → **Reflection** (actually a permutation matrix)

---

### Solution A3: Projection onto a Line

**a)** Projection formula: proj = (**a**ᵀ**b** / **a**ᵀ**a**) **a**

**a**ᵀ**b** = 1(3) + 2(4) = 11
**a**ᵀ**a** = 1² + 2² = 5

proj = (11/5)[1, 2]ᵀ = **[11/5, 22/5]ᵀ = [2.2, 4.4]ᵀ**

**b)** Perpendicular component: **b** - proj = [3, 4]ᵀ - [11/5, 22/5]ᵀ

= [15/5 - 11/5, 20/5 - 22/5]ᵀ = **[4/5, -2/5]ᵀ = [0.8, -0.4]ᵀ**

**c)** Verify orthogonality:

proj · perp = (11/5)(4/5) + (22/5)(-2/5) = 44/25 - 44/25 = 0 ✓

---

### Solution A4: Projection Matrix Properties

**a)** P = **aa**ᵀ / ‖**a**‖²

**a** = [1, 1]ᵀ, ‖**a**‖² = 2

$$P = \frac{1}{2}\begin{bmatrix} 1 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & 1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$

**b)** P² = (1/4)[[1,1],[1,1]][[1,1],[1,1]] = (1/4)[[2,2],[2,2]] = (1/2)[[1,1],[1,1]] = P ✓

**c)** P = Pᵀ clearly (symmetric matrix) ✓

**d)** Eigenvalues: tr(P) = 1, det(P) = (1/4)(1 - 1) = 0

Eigenvalues are **λ = 1 and λ = 0**

(Projection matrices always have eigenvalues 0 and 1)

---

### Solution A5: Gram-Schmidt Process

**Step 1:** **q₁** = **v₁**/‖**v₁**‖ = [1, 1, 0]ᵀ/√2 = **[1/√2, 1/√2, 0]ᵀ**

**Step 2:**
**q₁**ᵀ**v₂** = (1/√2)(0) + (1/√2)(1) + (0)(1) = 1/√2

**u₂** = **v₂** - (**q₁**ᵀ**v₂**)**q₁** = [0, 1, 1]ᵀ - (1/√2)[1/√2, 1/√2, 0]ᵀ
     = [0, 1, 1]ᵀ - [1/2, 1/2, 0]ᵀ = [-1/2, 1/2, 1]ᵀ

‖**u₂**‖ = √(1/4 + 1/4 + 1) = √(3/2)

**q₂** = **u₂**/‖**u₂**‖ = **[-1/√6, 1/√6, 2/√6]ᵀ**

**Step 3:**
**q₁**ᵀ**v₃** = (1/√2)(1) + (1/√2)(0) + (0)(1) = 1/√2
**q₂**ᵀ**v₃** = (-1/√6)(1) + (1/√6)(0) + (2/√6)(1) = 1/√6

**u₃** = **v₃** - (1/√2)**q₁** - (1/√6)**q₂**
     = [1, 0, 1]ᵀ - (1/√2)[1/√2, 1/√2, 0]ᵀ - (1/√6)[-1/√6, 1/√6, 2/√6]ᵀ
     = [1, 0, 1]ᵀ - [1/2, 1/2, 0]ᵀ - [-1/6, 1/6, 1/3]ᵀ
     = [1/2 + 1/6, -1/2 - 1/6, 1 - 1/3]ᵀ = [2/3, -2/3, 2/3]ᵀ

‖**u₃**‖ = √(4/9 + 4/9 + 4/9) = √(4/3) = 2/√3

**q₃** = **[1/√3, -1/√3, 1/√3]ᵀ**

---

### Solution A6: QR Decomposition

**a)** Apply Gram-Schmidt to columns [1, 1, 0]ᵀ and [1, 0, 1]ᵀ:

**q₁** = [1, 1, 0]ᵀ/√2

**q₁**ᵀ**a₂** = (1/√2)(1) + (1/√2)(0) + 0(1) = 1/√2

**u₂** = [1, 0, 1]ᵀ - (1/√2)[1/√2, 1/√2, 0]ᵀ = [1/2, -1/2, 1]ᵀ

‖**u₂**‖ = √(1/4 + 1/4 + 1) = √(3/2)

**q₂** = [1/√6, -1/√6, 2/√6]ᵀ

$$Q = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{6} \\ 1/\sqrt{2} & -1/\sqrt{6} \\ 0 & 2/\sqrt{6} \end{bmatrix}$$

**b)** R = QᵀA

$$R = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} & 0 \\ 1/\sqrt{6} & -1/\sqrt{6} & 2/\sqrt{6} \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

R₁₁ = (1/√2)(1) + (1/√2)(1) + 0 = √2
R₁₂ = (1/√2)(1) + (1/√2)(0) + 0 = 1/√2
R₂₁ = (1/√6)(1) + (-1/√6)(1) + (2/√6)(0) = 0
R₂₂ = (1/√6)(1) + (-1/√6)(0) + (2/√6)(1) = 3/√6 = √(3/2)

$$R = \begin{bmatrix} \sqrt{2} & 1/\sqrt{2} \\ 0 & \sqrt{3/2} \end{bmatrix}$$

**c)** Verify: QR = A ✓

---

### Solution A7: Least Squares via Projection

**a)** Projection of **b** onto column space (which is just span{[1,1,1]ᵀ}):

proj = (**a**ᵀ**b** / **a**ᵀ**a**)**a** = ((1+2+3)/3)[1,1,1]ᵀ = **2[1,1,1]ᵀ = [2, 2, 2]ᵀ**

**b)** The least squares solution x satisfies Ax = proj, so:

x[1, 1, 1]ᵀ = [2, 2, 2]ᵀ → **x = 2**

**c)** Residual = **b** - A**x** = [1, 2, 3]ᵀ - [2, 2, 2]ᵀ = **[-1, 0, 1]ᵀ**

Verify: residual · A = (-1)(1) + (0)(1) + (1)(1) = 0 ✓

---

### Solution A8: Orthogonal Complement

V = span{[1, 1, 0, 0]ᵀ, [0, 1, 1, 0]ᵀ}

**a)** Find **w** such that **w**ᵀ**v₁** = 0 and **w**ᵀ**v₂** = 0.

Let **w** = [a, b, c, d]ᵀ.

a + b = 0 → b = -a
b + c = 0 → c = -b = a

So **w** = [a, -a, a, d]ᵀ = a[1, -1, 1, 0]ᵀ + d[0, 0, 0, 1]ᵀ

Basis for V⊥: **{[1, -1, 1, 0]ᵀ, [0, 0, 0, 1]ᵀ}**

**b)** Verify:
[1, -1, 1, 0] · [1, 1, 0, 0] = 1 - 1 + 0 + 0 = 0 ✓
[1, -1, 1, 0] · [0, 1, 1, 0] = 0 - 1 + 1 + 0 = 0 ✓
[0, 0, 0, 1] · [1, 1, 0, 0] = 0 ✓
[0, 0, 0, 1] · [0, 1, 1, 0] = 0 ✓

**c)** dim(V⊥) = 4 - 2 = **2** ✓

---

### Solution A9: Four Fundamental Subspaces

$$A = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \end{bmatrix}$$

Note: Row 2 = 2 × Row 1, so rank(A) = 1.

**a)** Column space C(A): span{[1, 2]ᵀ} (any column works, they're all multiples)

Basis: **{[1, 2]ᵀ}**, dim = 1

**b)** Null space N(A): Solve A**x** = **0**

Row reduce: [[1, 2, 1], [0, 0, 0]]

x₁ + 2x₂ + x₃ = 0 → x₁ = -2x₂ - x₃

**x** = x₂[-2, 1, 0]ᵀ + x₃[-1, 0, 1]ᵀ

Basis: **{[-2, 1, 0]ᵀ, [-1, 0, 1]ᵀ}**, dim = 2

**c)** Row space C(Aᵀ): span of rows = span{[1, 2, 1]}

Basis: **{[1, 2, 1]ᵀ}**, dim = 1

**d)** Left null space N(Aᵀ): Solve Aᵀ**y** = **0**

Aᵀ = [[1, 2], [2, 4], [1, 2]]

Row 2 = 2 × Row 1, Row 3 = Row 1

y₁ + 2y₂ = 0 → y₁ = -2y₂

Basis: **{[-2, 1]ᵀ}**, dim = 1

**e)** Verify orthogonality:

- N(A) ⊥ C(Aᵀ): [-2, 1, 0] · [1, 2, 1] = -2 + 2 + 0 = 0 ✓
- N(Aᵀ) ⊥ C(A): [-2, 1] · [1, 2] = -2 + 2 = 0 ✓

---

### Solution A10: Orthogonal Procrustes Problem

**a)** We want to minimize ‖XQ - Y‖_F² subject to QᵀQ = I.

Expanding: ‖XQ - Y‖_F² = ‖XQ‖_F² - 2⟨XQ, Y⟩_F + ‖Y‖_F²

= ‖X‖_F² - 2tr(QᵀXᵀY) + ‖Y‖_F² (using orthogonality of Q)

To minimize, maximize tr(QᵀXᵀY) = tr(QᵀM) where M = XᵀY.

Let M = UΣVᵀ be the SVD. Then:
tr(QᵀM) = tr(QᵀUΣVᵀ) = tr(VᵀQᵀUΣ)

Let W = VᵀQᵀU. Since Q, U, V are orthogonal, so is W.

tr(WΣ) = Σᵢ wᵢᵢσᵢ ≤ Σᵢ σᵢ (since |wᵢᵢ| ≤ 1 for orthogonal W)

Maximum when W = I, i.e., Q = UVᵀ.

**Answer: Q = UVᵀ where XᵀY = UΣVᵀ**

**b)** Compute XᵀY:

$$X^TY = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$$

SVD of [[1, 2], [2, 1]]:
- Eigenvalues of symmetric matrix: (1-λ)(1-λ) - 4 = λ² - 2λ - 3 = (λ-3)(λ+1)
- λ = 3, -1, so singular values σ₁ = 3, σ₂ = 1

Eigenvector for λ=3: [1, 1]ᵀ/√2
Eigenvector for λ=-1: [1, -1]ᵀ/√2

U = V = (1/√2)[[1, 1], [1, -1]]

Q = UVᵀ = (1/√2)[[1, 1], [1, -1]] × (1/√2)[[1, 1], [-1, 1]]⁻¹ = I

**Q = I** (the matrices are already well-aligned)

---

## Part B: Coding Solutions

### Solution B1: Orthogonality Checker

```python
import numpy as np

def check_orthogonality(vectors, tolerance=1e-10):
    """Check if vectors are orthogonal/orthonormal."""
    vectors = [np.array(v) for v in vectors]
    n = len(vectors)
    
    dot_products = np.zeros((n, n))
    norms = []
    
    for i in range(n):
        norms.append(np.linalg.norm(vectors[i]))
        for j in range(n):
            dot_products[i, j] = np.dot(vectors[i], vectors[j])
    
    # Check orthogonality: off-diagonal should be zero
    off_diagonal = dot_products - np.diag(np.diag(dot_products))
    is_orthogonal = np.allclose(off_diagonal, 0, atol=tolerance)
    
    # Check orthonormality: diagonal should be 1
    is_orthonormal = is_orthogonal and np.allclose(np.diag(dot_products), 1, atol=tolerance)
    
    return {
        'is_orthogonal': is_orthogonal,
        'is_orthonormal': is_orthonormal,
        'dot_products': dot_products,
        'norms': np.array(norms)
    }

# Test
vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
result = check_orthogonality(vectors)
print(f"Orthogonal: {result['is_orthogonal']}, Orthonormal: {result['is_orthonormal']}")
```

---

### Solution B2: Projection Visualization

```python
import matplotlib.pyplot as plt

def visualize_projection(b, a):
    """Visualize projection of b onto line spanned by a."""
    b, a = np.array(b), np.array(a)
    
    # Compute projection
    proj = (np.dot(a, b) / np.dot(a, a)) * a
    perp = b - proj
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw vectors
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.02, label='a (line direction)')
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
              color='green', width=0.02, label='b (original)')
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.02, label='proj (projection)')
    ax.quiver(proj[0], proj[1], perp[0], perp[1], angles='xy', scale_units='xy', scale=1,
              color='orange', width=0.02, label='perp (perpendicular)')
    
    # Draw right angle
    scale = 0.15 * min(np.linalg.norm(proj), np.linalg.norm(perp))
    if np.linalg.norm(proj) > 0.1 and np.linalg.norm(perp) > 0.1:
        proj_unit = proj / np.linalg.norm(proj) * scale
        perp_unit = perp / np.linalg.norm(perp) * scale
        corner = proj - proj_unit + perp_unit
        ax.plot([proj[0], corner[0]], [proj[1], corner[1]], 'k-', linewidth=1)
        ax.plot([corner[0], proj[0] + perp_unit[0]], [corner[1], proj[1] + perp_unit[1]], 'k-', linewidth=1)
    
    max_val = max(np.abs([*a, *b])) * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title(f'Projection of b onto a\nproj · perp = {np.dot(proj, perp):.6f} (should be ≈0)')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# Test
visualize_projection([3, 4], [1, 2])
```

---

### Solution B3: Gram-Schmidt Implementation

```python
def gram_schmidt(vectors, normalize=True):
    """Orthogonalize vectors using Gram-Schmidt."""
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        vectors = [vectors[:, i] for i in range(vectors.shape[1])]
    
    vectors = [np.array(v, dtype=float) for v in vectors]
    orthogonal = []
    
    for v in vectors:
        # Subtract projections onto previous vectors
        for u in orthogonal:
            v = v - (np.dot(u, v) / np.dot(u, u)) * u
        
        if np.linalg.norm(v) > 1e-10:  # Check not zero
            orthogonal.append(v)
    
    result = np.column_stack(orthogonal)
    
    if normalize:
        result = result / np.linalg.norm(result, axis=0)
    
    return result

def modified_gram_schmidt(vectors, normalize=True):
    """Numerically stable Gram-Schmidt."""
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        V = vectors.copy().astype(float)
    else:
        V = np.column_stack([np.array(v, dtype=float) for v in vectors])
    
    n = V.shape[1]
    Q = np.zeros_like(V)
    
    for i in range(n):
        Q[:, i] = V[:, i]
        
        if normalize:
            Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])
        
        # Subtract projection from all remaining vectors
        for j in range(i + 1, n):
            V[:, j] = V[:, j] - np.dot(Q[:, i], V[:, j]) * Q[:, i]
    
    return Q

# Test
V = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]).T
Q = gram_schmidt(V)
print("Gram-Schmidt result:")
print(Q)
print(f"Orthonormal check (Q^T Q):\n{Q.T @ Q}")
```

---

### Solution B4: QR Decomposition from Scratch

```python
def qr_decomposition(A):
    """Compute QR decomposition using Gram-Schmidt."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    # Orthonormalize columns
    Q = modified_gram_schmidt(A)
    
    # R = Q^T A
    R = Q.T @ A
    
    return Q, R

# Test
A = np.array([[1, 1], [1, 0], [0, 1]])
Q, R = qr_decomposition(A)
print("Q:")
print(Q)
print("\nR:")
print(R)
print(f"\nReconstruction error: {np.linalg.norm(A - Q @ R)}")
print(f"Q orthonormal check:\n{Q.T @ Q}")
```

---

### Solution B5: Least Squares via QR

```python
def least_squares_qr(A, b):
    """Solve least squares using QR decomposition."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    Q, R = np.linalg.qr(A)
    
    # Solve Rx = Q^T b by back substitution
    c = Q.T @ b
    n = R.shape[1]
    x = np.linalg.solve(R[:n, :n], c[:n])
    
    # Projection and residual
    projection = A @ x
    residual_norm = np.linalg.norm(b - projection)
    
    return {
        'x': x,
        'residual_norm': residual_norm,
        'projection': projection
    }

# Test
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2.5])
result = least_squares_qr(A, b)
print(f"Solution: {result['x']}")
print(f"Residual norm: {result['residual_norm']:.6f}")
```

---

### Solution B6: Projection onto Subspace

```python
def project_onto_subspace(b, basis_vectors):
    """Project vector onto subspace spanned by basis vectors."""
    if isinstance(basis_vectors, list):
        A = np.column_stack([np.array(v) for v in basis_vectors])
    else:
        A = basis_vectors
    
    b = np.array(b, dtype=float)
    
    # Projection matrix P = A(A^T A)^{-1} A^T
    P = A @ np.linalg.inv(A.T @ A) @ A.T
    
    projection = P @ b
    perpendicular = b - projection
    
    return {
        'projection': projection,
        'perpendicular': perpendicular,
        'projection_matrix': P
    }

# Test
b = np.array([1, 2, 3])
basis = [[1, 0, 0], [0, 1, 0]]
result = project_onto_subspace(b, basis)
print(f"Projection: {result['projection']}")
print(f"Perpendicular: {result['perpendicular']}")
print(f"Verify orthogonality: {np.dot(result['projection'], result['perpendicular']):.6f}")
```

---

## Part C: Conceptual Answers

### Answer C1: Orthogonality vs Linear Independence

**Example:** {[1, 0], [1, 1]} is linearly independent but not orthogonal.

**Why orthogonal is "better":**

1. **Simple coefficients:** To express x = c₁v₁ + c₂v₂ in orthonormal basis:
   cᵢ = x · vᵢ (just dot products!)
   
   For non-orthogonal basis, need to solve a linear system.

2. **Numerical stability:** Orthogonal bases are well-conditioned.

3. **Geometric interpretation:** Each coefficient measures component in that direction.

---

### Answer C2: Gram-Schmidt Numerical Issues

**Problem with classical GS:**
When vectors are nearly parallel, subtracting projections can lead to catastrophic cancellation. The resulting vectors may not be orthogonal due to accumulated rounding errors.

**Modified GS helps:**
Instead of computing all projections using original vectors, MGS reorthogonalizes against the already-computed orthogonal vectors. This prevents error accumulation.

**Even better:** Householder QR uses reflections, which are more stable than projections.

---

### Answer C3: Normal Equations vs QR

**Problems with normal equations:**
1. Computing AᵀA squares the condition number: κ(AᵀA) = κ(A)²
2. Loss of information due to squaring small singular values
3. Can fail completely for ill-conditioned A

**When to use QR:**
- A is ill-conditioned (high condition number)
- Need numerical stability
- Features are nearly collinear

---

### Answer C4: Orthogonal Matrices in Deep Learning

**Why orthogonal init helps:**

1. **Preserves norms:** ‖Qx‖ = ‖x‖, so signals don't explode or vanish through layers initially

2. **Preserves gradients:** Backprop through Q also preserves gradient norms

3. **Maximizes information flow:** Orthogonal matrices don't collapse dimensions

**What happens:** For Q orthogonal, the singular values are all 1. This means the "effective gain" of the layer is 1, leading to stable forward and backward passes at initialization.

---

### Answer C5: Projection Residual

**Geometric explanation:**

The projection p = proj_V(b) is the point in V closest to b. 

If the residual r = b - p were NOT perpendicular to V, we could move p within V to get closer to b:
- The component of r along V points toward where b "wants to go"
- Moving p in that direction decreases distance

Therefore, at the minimum, r must be perpendicular to V.

**Algebraic:** The optimality condition ∂‖Ax - b‖²/∂x = 0 gives:
Aᵀ(Ax - b) = 0, meaning the residual is orthogonal to column space of A.
