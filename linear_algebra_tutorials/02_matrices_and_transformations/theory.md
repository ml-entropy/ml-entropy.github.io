# Tutorial 02: Matrices and Linear Transformations

## Introduction: What is a Matrix?

A matrix is not just a table of numbers—it's a **function** that transforms vectors.

### The Big Insight

> **Every matrix represents a linear transformation, and every linear transformation can be represented by a matrix.**

This is one of the most powerful ideas in mathematics.

---

## Part 1: Matrices as Transformations

### What Does a Matrix DO?

When you multiply a matrix A by a vector **x**, you get a new vector **y** = A**x**.

```
     Matrix A          Vector x      Vector y
┌           ┐        ┌     ┐      ┌     ┐
│ a₁₁  a₁₂ │   ×    │ x₁  │  =   │ y₁  │
│ a₂₁  a₂₂ │        │ x₂  │      │ y₂  │
└           ┘        └     ┘      └     ┘
```

**The matrix transforms the input vector into the output vector.**

### 2D Examples: Visualizing Transformations

**1. Rotation by θ:**
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

This rotates every vector counterclockwise by angle θ.

**2. Scaling:**
$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

Stretches x-coordinates by sₓ and y-coordinates by sᵧ.

**3. Reflection (across x-axis):**
$$F = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

Flips the y-coordinate.

**4. Shear:**
$$H = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

Shifts x proportionally to y (makes rectangles into parallelograms).

### The Column View: Where Basis Vectors Go

**Key insight:** The columns of matrix A tell you where the standard basis vectors land!

If A = [**a₁** | **a₂** | ... | **aₙ**], then:
- First basis vector **e₁** = [1,0,...,0]ᵀ maps to **a₁**
- Second basis vector **e₂** = [0,1,...,0]ᵀ maps to **a₂**
- etc.

**Example:** Rotation by 90°
$$R_{90} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

- **e₁** = [1, 0] maps to [0, 1] (first column)
- **e₂** = [0, 1] maps to [-1, 0] (second column)

The whole transformation is determined by where the basis vectors go!

### The Row View: Dot Products

Each row of A**x** is a dot product:
$$y_i = (\text{row } i \text{ of } A) \cdot \mathbf{x}$$

This means each output component measures how much **x** aligns with a "detector" vector (the row).

**In neural networks:** Each row of the weight matrix is a "feature detector."

---

## Part 2: Matrix Multiplication

### Definition and Intuition

For matrices A (m×n) and B (n×p), their product C = AB is (m×p):

$$C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$

**Intuition: Composition of Transformations**

If A transforms vectors, and B transforms vectors, then AB transforms vectors by **first applying B, then applying A**.

$$AB\mathbf{x} = A(B\mathbf{x})$$

### Three Ways to View Matrix Multiplication

**1. Dot Product View:**
$$C_{ij} = \text{row } i \text{ of } A \cdot \text{column } j \text{ of } B$$

**2. Column View:**
Each column of C is A times the corresponding column of B:
$$\mathbf{c}_j = A\mathbf{b}_j$$

**3. Row View:**
Each row of C comes from the corresponding row of A times B.

**4. Outer Product View:**
$$AB = \sum_{k=1}^n (\text{column } k \text{ of } A)(\text{row } k \text{ of } B)$$

### Why Matrix Multiplication is Not Commutative

In general, AB ≠ BA.

**Intuitive reason:** "Rotate then scale" ≠ "Scale then rotate"

**Example:**
$$A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}$$

$$AB = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad BA = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$$

---

## Part 3: Special Matrices

### Identity Matrix

$$I = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**Property:** AI = IA = A for any compatible matrix A.

**As transformation:** Does nothing (every vector maps to itself).

### Diagonal Matrices

$$D = \begin{bmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{bmatrix}$$

**As transformation:** Scales each axis independently.

**Useful property:** D**x** = [d₁x₁, d₂x₂, ..., dₙxₙ]ᵀ (elementwise scaling)

### Transpose

The transpose Aᵀ swaps rows and columns:

$$(A^T)_{ij} = A_{ji}$$

**Properties:**
- (AB)ᵀ = BᵀAᵀ (order reverses!)
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ

### Symmetric Matrices

A matrix is symmetric if Aᵀ = A.

**Examples:**
- Covariance matrices
- Hessian matrices (second derivatives)
- Distance matrices

**Key property:** Symmetric matrices have real eigenvalues and orthogonal eigenvectors.

### Inverse Matrix

A⁻¹ is the inverse of A if:
$$A^{-1}A = AA^{-1} = I$$

**As transformation:** A⁻¹ "undoes" what A does.

**When does A⁻¹ exist?**
- Only for square matrices
- Only if det(A) ≠ 0
- Only if columns are linearly independent

### Orthogonal Matrices

Matrix Q is orthogonal if:
$$Q^T Q = QQ^T = I$$

Equivalently: Q⁻¹ = Qᵀ

**Properties:**
- Columns are orthonormal (mutually perpendicular unit vectors)
- Preserves lengths: ||Q**x**|| = ||**x**||
- Preserves angles: (Q**u**) · (Q**v**) = **u** · **v**

**As transformation:** Rotations and reflections (rigid motions).

**In ML:** Orthogonal matrices in neural networks help avoid vanishing/exploding gradients.

---

## Part 4: The Rank

### Definition

The **rank** of a matrix is:
- The number of linearly independent columns
- = The number of linearly independent rows
- = The dimension of the column space
- = The dimension of the row space

### What Rank Tells Us

| Rank | Meaning |
|------|---------|
| rank(A) = n (full column rank) | Columns are independent; A**x** = **b** has at most one solution |
| rank(A) = m (full row rank) | Rows are independent; A**x** = **b** always has a solution |
| rank(A) = min(m,n) (full rank) | Maximum possible rank |
| rank(A) < min(m,n) (rank deficient) | Some redundancy; transformation "loses dimensions" |

### Geometric Interpretation

If A is m×n with rank r:
- A maps ℝⁿ to an r-dimensional subspace of ℝᵐ
- The "image" of A is r-dimensional
- (n - r) dimensions are "collapsed" to zero

**Example:** A 3×3 matrix with rank 2 squashes 3D space onto a 2D plane.

---

## Part 5: Matrix Decompositions (Preview)

### Why Decompose?

Breaking a matrix into simpler pieces helps us:
- Understand what the transformation does
- Compute things efficiently
- Analyze properties

### Key Decompositions

**1. Eigendecomposition (for square matrices):**
$$A = PDP^{-1}$$
where D is diagonal and P contains eigenvectors.

**2. Singular Value Decomposition (for any matrix):**
$$A = U\Sigma V^T$$
where U, V are orthogonal and Σ is diagonal.

**3. LU Decomposition:**
$$A = LU$$
where L is lower triangular and U is upper triangular.

**4. QR Decomposition:**
$$A = QR$$
where Q is orthogonal and R is upper triangular.

We'll explore these in detail in later tutorials.

---

## Part 6: Applications in Machine Learning

### Neural Network Layers

A linear layer computes:
$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

The weight matrix W transforms input features into output features.

For a batch of inputs (matrix X):
$$Y = XW^T + \mathbf{b}$$

### Data as Matrices

- Training data: X is (N × D) where N = samples, D = features
- Images: (H × W × C) tensors, often flattened to vectors
- Sequences: (T × D) where T = time steps

### Covariance Matrix

For data matrix X (N × D), centered at mean:
$$\Sigma = \frac{1}{N-1} X^T X$$

This D × D matrix captures how features vary together.

### Principal Component Analysis (PCA)

1. Compute covariance matrix Σ
2. Find eigenvectors (principal components)
3. Project data onto top k eigenvectors

The eigenvectors of Σ give directions of maximum variance.

### Attention Mechanism (Transformers)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

- Q: Query matrix (what we're looking for)
- K: Key matrix (what's available)
- V: Value matrix (what we retrieve)
- QKᵀ: Similarity scores between queries and keys

---

## Part 7: Deriving Matrix Calculus Basics

### Derivative of Linear Function

For **y** = A**x**:
$$\frac{\partial y_i}{\partial x_j} = A_{ij}$$

So the Jacobian matrix is just A itself!

### Derivative of Quadratic Form

For f(**x**) = **x**ᵀA**x** (a scalar):

$$\nabla f = (A + A^T)\mathbf{x}$$

If A is symmetric: ∇f = 2A**x**

**Derivation:**
$$f = \sum_{i,j} x_i A_{ij} x_j$$

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i x_i A_{ik} = (A\mathbf{x})_k + (A^T\mathbf{x})_k$$

Therefore: ∇f = A**x** + Aᵀ**x** = (A + Aᵀ)**x**

---

## Summary

| Concept | Definition | ML Connection |
|---------|------------|---------------|
| Matrix | Linear transformation | Neural network weights |
| Matrix multiplication | Composition of transformations | Forward pass through layers |
| Transpose | Swap rows and columns | Backpropagation uses Wᵀ |
| Inverse | Undo transformation | Solving linear systems |
| Rank | Dimension of output space | Model capacity, compression |
| Orthogonal matrix | Preserves lengths and angles | Stable transformations |

---

## Key Takeaways

1. **Matrices ARE transformations** - Think geometrically
2. **Columns show where basis vectors go** - Powerful visualization tool
3. **Matrix multiplication = function composition** - Order matters!
4. **Rank = "effective dimension"** - How much information survives
5. **Neural networks = chains of matrix transformations + nonlinearities**

---

*Next: Tutorial 03 - Systems of Linear Equations*
