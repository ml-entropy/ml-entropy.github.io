# Tutorial 07: Orthogonality and Projections

## Introduction: Why Orthogonality Matters

Orthogonality is the generalization of "perpendicular" to higher dimensions. It's one of the most important concepts in linear algebra because orthogonal objects are:

1. **Independent** - No component of one lies along the other
2. **Easy to decompose** - Projections have simple formulas
3. **Numerically stable** - Orthogonal matrices preserve lengths and angles

**In ML, orthogonality appears in:**
- PCA (principal components are orthogonal)
- Gram-Schmidt and QR decomposition
- Least squares solutions
- Neural network initialization
- Regularization and feature decorrelation

---

## Part 1: Orthogonal Vectors

### Definition

Two vectors **u** and **v** are **orthogonal** (written **u** ⊥ **v**) if:

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T\mathbf{v} = 0$$

**Intuition:** They point in completely independent directions. No component of **u** lies along **v**.

### Geometric Interpretation

In 2D/3D, orthogonal = perpendicular (90° angle).

Using the dot product formula:
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

If dot product = 0 and both vectors are non-zero, then cos(θ) = 0, so θ = 90°.

### Orthogonal Sets

A set of vectors {**v₁**, **v₂**, ..., **vₖ**} is **orthogonal** if every pair is orthogonal:
$$\mathbf{v}_i \cdot \mathbf{v}_j = 0 \quad \text{for } i \neq j$$

If additionally each vector has unit length (‖**vᵢ**‖ = 1), the set is **orthonormal**:
$$\mathbf{v}_i \cdot \mathbf{v}_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### Key Property

**Theorem:** An orthogonal set of non-zero vectors is linearly independent.

**Proof:** Suppose c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **0**.
Take dot product with **vᵢ**:
$$c_1(\mathbf{v}_1 \cdot \mathbf{v}_i) + \cdots + c_i(\mathbf{v}_i \cdot \mathbf{v}_i) + \cdots + c_k(\mathbf{v}_k \cdot \mathbf{v}_i) = 0$$

All terms with j ≠ i vanish, leaving: cᵢ‖**vᵢ**‖² = 0.
Since **vᵢ** ≠ **0**, we have cᵢ = 0. ∎

---

## Part 2: Orthogonal Matrices

### Definition

A square matrix Q is **orthogonal** if:

$$Q^TQ = QQ^T = I$$

Equivalently: Q⁻¹ = Qᵀ (inverse equals transpose!)

### Column Interpretation

Q is orthogonal ⟺ columns of Q form an orthonormal set.

If Q = [**q₁** | **q₂** | ... | **qₙ**], then:
$$Q^TQ = \begin{bmatrix} \mathbf{q}_1^T \\ \vdots \\ \mathbf{q}_n^T \end{bmatrix} [\mathbf{q}_1 | \cdots | \mathbf{q}_n] = \begin{bmatrix} \mathbf{q}_i^T\mathbf{q}_j \end{bmatrix} = I$$

This means **qᵢ**ᵀ**qⱼ** = δᵢⱼ.

### Properties of Orthogonal Matrices

1. **Preserve lengths:** ‖Q**x**‖ = ‖**x**‖
   *Proof:* ‖Q**x**‖² = (Q**x**)ᵀ(Q**x**) = **x**ᵀQᵀQ**x** = **x**ᵀ**x** = ‖**x**‖²

2. **Preserve angles:** ∠(Q**x**, Q**y**) = ∠(**x**, **y**)
   *Proof:* (Q**x**)ᵀ(Q**y**) = **x**ᵀQᵀQ**y** = **x**ᵀ**y**

3. **Preserve dot products:** (Q**x**) · (Q**y**) = **x** · **y**

4. **det(Q) = ±1**
   *Proof:* det(QᵀQ) = det(I) = 1, so det(Q)² = 1.

### Geometric Meaning

Orthogonal matrices represent:
- **Rotations** (det = +1)
- **Reflections** (det = -1)

They are "rigid motions" that preserve shape and size.

### Examples

**2D Rotation:**
$$Q = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**2D Reflection** (across x-axis):
$$Q = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**Permutation matrices** are also orthogonal.

---

## Part 3: Projections

### Motivation

Given a vector **b** and a subspace V, find the closest point in V to **b**.

This is fundamental to:
- Least squares regression
- Feature approximation
- Compression

### Projection onto a Line (1D Subspace)

Given a line spanned by unit vector **u**, the projection of **b** onto this line:

$$\text{proj}_{\mathbf{u}}(\mathbf{b}) = (\mathbf{u}^T\mathbf{b})\mathbf{u}$$

**Derivation:**

The projection must be a scalar multiple of **u**: proj = c**u**

The error (**b** - proj) must be orthogonal to **u**:
$$\mathbf{u}^T(\mathbf{b} - c\mathbf{u}) = 0$$
$$\mathbf{u}^T\mathbf{b} - c(\mathbf{u}^T\mathbf{u}) = 0$$
$$c = \frac{\mathbf{u}^T\mathbf{b}}{\mathbf{u}^T\mathbf{u}} = \mathbf{u}^T\mathbf{b} \quad \text{(if } \|\mathbf{u}\| = 1\text{)}$$

### General Projection Formula

If **u** is not unit length:
$$\text{proj}_{\mathbf{u}}(\mathbf{b}) = \frac{\mathbf{u}^T\mathbf{b}}{\mathbf{u}^T\mathbf{u}}\mathbf{u} = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}\mathbf{b}$$

The **projection matrix** is:
$$P = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}$$

### Properties of Projection Matrices

1. **P² = P** (idempotent) - Projecting twice = projecting once
2. **Pᵀ = P** (symmetric)
3. **Eigenvalues are 0 or 1**

### Projection onto a Subspace

For a subspace V spanned by columns of A (assuming A has full column rank):

$$\text{proj}_V(\mathbf{b}) = A(A^TA)^{-1}A^T\mathbf{b}$$

The projection matrix is:
$$P = A(A^TA)^{-1}A^T$$

**Derivation:** Find **x** such that A**x** is closest to **b**.

Minimize ‖**b** - A**x**‖². The residual (**b** - A**x**) must be orthogonal to column space:
$$A^T(\mathbf{b} - A\mathbf{x}) = \mathbf{0}$$
$$A^T\mathbf{b} = A^TA\mathbf{x}$$
$$\mathbf{x} = (A^TA)^{-1}A^T\mathbf{b}$$

So: proj = A**x** = A(AᵀA)⁻¹Aᵀ**b**

---

## Part 4: Gram-Schmidt Process

### The Problem

Given a set of linearly independent vectors {**v₁**, **v₂**, ..., **vₖ**}, produce an orthonormal set {**q₁**, **q₂**, ..., **qₖ**} that spans the same space.

### The Algorithm

**Step 1:** Normalize first vector
$$\mathbf{q}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}$$

**Step 2:** Subtract projection onto **q₁** and normalize
$$\mathbf{u}_2 = \mathbf{v}_2 - (\mathbf{q}_1^T\mathbf{v}_2)\mathbf{q}_1$$
$$\mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}$$

**Step k:** Subtract projections onto all previous q's and normalize
$$\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1}(\mathbf{q}_j^T\mathbf{v}_k)\mathbf{q}_j$$
$$\mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

### Why It Works

At each step, we remove the components of **vₖ** that lie along the previous directions. What remains is orthogonal to all previous **qⱼ**.

### Example

Given **v₁** = [1, 1, 0]ᵀ and **v₂** = [1, 0, 1]ᵀ:

**Step 1:** **q₁** = [1, 1, 0]ᵀ / √2

**Step 2:**
$$\mathbf{q}_1^T\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1) = \frac{1}{\sqrt{2}}$$
$$\mathbf{u}_2 = [1, 0, 1]^T - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}[1, 1, 0]^T = [1, 0, 1]^T - [1/2, 1/2, 0]^T = [1/2, -1/2, 1]^T$$
$$\mathbf{q}_2 = [1/2, -1/2, 1]^T / \|[1/2, -1/2, 1]^T\| = [1/2, -1/2, 1]^T / \sqrt{3/2}$$

Verify: **q₁** · **q₂** = 0 ✓

---

## Part 5: QR Decomposition

### Definition

Any m×n matrix A with linearly independent columns can be factored as:

$$A = QR$$

where:
- Q is m×n with orthonormal columns
- R is n×n upper triangular with positive diagonal

### Connection to Gram-Schmidt

QR decomposition IS Gram-Schmidt written in matrix form!

If A = [**a₁** | **a₂** | ... | **aₙ**] and Q = [**q₁** | **q₂** | ... | **qₙ**]:

- **q₁**, **q₂**, ... are the orthonormal vectors from Gram-Schmidt
- R captures the coefficients used in the process

$$\mathbf{a}_j = r_{1j}\mathbf{q}_1 + r_{2j}\mathbf{q}_2 + \cdots + r_{jj}\mathbf{q}_j$$

Since **aⱼ** only depends on **q₁** through **qⱼ**, R is upper triangular.

### Computing R

Once we have Q:
$$R = Q^TA$$

This works because QᵀQ = I.

### Why QR is Useful

1. **Least squares:** Solve A**x** = **b** via QR
   - A**x** = **b** → QR**x** = **b** → R**x** = Qᵀ**b**
   - R is upper triangular, so solve by back-substitution!

2. **Numerical stability:** More stable than forming AᵀA

3. **Eigenvalue algorithms:** QR iteration is the standard method

---

## Part 6: Least Squares via QR

### The Problem

Find **x** minimizing ‖A**x** - **b**‖² where A is m×n with m > n.

### Solution via QR

Factor A = QR. Then:
$$\|A\mathbf{x} - \mathbf{b}\|^2 = \|QR\mathbf{x} - \mathbf{b}\|^2 = \|R\mathbf{x} - Q^T\mathbf{b}\|^2$$

(using that Q preserves norms)

Let **c** = Qᵀ**b**. Partition **c** = [**c₁**; **c₂**] where **c₁** has n entries:

$$\|R\mathbf{x} - \mathbf{c}\|^2 = \left\|\begin{bmatrix} R_1 \\ 0 \end{bmatrix}\mathbf{x} - \begin{bmatrix} \mathbf{c}_1 \\ \mathbf{c}_2 \end{bmatrix}\right\|^2 = \|R_1\mathbf{x} - \mathbf{c}_1\|^2 + \|\mathbf{c}_2\|^2$$

Minimum achieved when R₁**x** = **c₁** (upper triangular system!).

The minimum residual is ‖**c₂**‖.

### Advantages over Normal Equations

Normal equations: (AᵀA)**x** = Aᵀ**b**
- Computing AᵀA squares the condition number
- Can be numerically unstable

QR approach:
- Never forms AᵀA
- More stable for ill-conditioned problems

---

## Part 7: Orthogonal Complements

### Definition

The **orthogonal complement** of a subspace V, denoted V⊥, is:

$$V^\perp = \{\mathbf{w} : \mathbf{w} \cdot \mathbf{v} = 0 \text{ for all } \mathbf{v} \in V\}$$

All vectors perpendicular to everything in V.

### Properties

1. V⊥ is a subspace
2. dim(V) + dim(V⊥) = n (for subspaces of ℝⁿ)
3. (V⊥)⊥ = V
4. Every **x** ∈ ℝⁿ can be uniquely written as **x** = **v** + **w** where **v** ∈ V and **w** ∈ V⊥

### The Four Fundamental Subspaces

For an m×n matrix A:
1. **Column space** C(A) ⊆ ℝᵐ - span of columns
2. **Row space** C(Aᵀ) ⊆ ℝⁿ - span of rows
3. **Null space** N(A) ⊆ ℝⁿ - solutions to A**x** = **0**
4. **Left null space** N(Aᵀ) ⊆ ℝᵐ - solutions to Aᵀ**y** = **0**

**Key relationships:**
- C(Aᵀ)⊥ = N(A) (row space ⊥ null space)
- C(A)⊥ = N(Aᵀ) (column space ⊥ left null space)

---

## Part 8: ML Applications

### Application 1: Least Squares Regression

Linear regression: Find **w** minimizing ‖X**w** - **y**‖²

**Via QR:** X = QR, then solve R**w** = Qᵀ**y**

More stable than normal equations when features are correlated.

### Application 2: Orthogonal Weight Initialization

In deep learning, orthogonal initialization helps:
- Preserve gradient magnitudes through layers
- Prevent vanishing/exploding gradients early in training

```python
# PyTorch orthogonal initialization
torch.nn.init.orthogonal_(layer.weight)
```

### Application 3: Decorrelating Features

PCA finds orthogonal directions of maximum variance.

If data has covariance Σ with eigendecomposition Σ = QΛQᵀ:
- Q gives orthogonal directions
- Transforming data by Qᵀ decorrelates it

### Application 4: Regularization via Orthogonality

Soft orthogonality constraints in neural networks:
$$\mathcal{L}_{ortho} = \|W^TW - I\|_F^2$$

Encourages diverse, non-redundant features.

### Application 5: Attention and Orthogonality

In transformers, heads with more orthogonal query/key spaces capture different relationships.

---

## Summary

| Concept | Definition | ML Relevance |
|---------|-----------|--------------|
| Orthogonal vectors | **u**·**v** = 0 | Independent features |
| Orthogonal matrix | QᵀQ = I | Rotations, preserves geometry |
| Projection | (AᵀA)⁻¹Aᵀ | Least squares, approximation |
| Gram-Schmidt | Orthogonalize a basis | QR decomposition |
| QR decomposition | A = QR | Stable least squares |
| Orthogonal complement | V⊥ | Residuals, null space |

---

## Key Takeaways

1. **Orthogonal = independent** - No redundant information
2. **Orthogonal matrices preserve geometry** - Rotations/reflections
3. **Projection = closest point** - Fundamental to least squares
4. **Gram-Schmidt produces QR** - Key algorithm
5. **QR is more stable than normal equations** - Use it for least squares
6. **Orthogonality helps deep learning** - Initialization, regularization

---

*Next: Tutorial 08 - Positive Definite Matrices*
