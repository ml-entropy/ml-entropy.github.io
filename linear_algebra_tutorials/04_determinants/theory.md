# Tutorial 04: Determinants

## Introduction: What is a Determinant?

The determinant is a single number that captures the "essence" of a square matrix. But what does it really mean?

### Three Key Interpretations

**1. Geometric: Volume Scaling Factor**
- The determinant tells you how much a transformation scales area (2D) or volume (3D)
- |det(A)| = factor by which areas/volumes are multiplied
- sign(det(A)) = whether orientation is preserved (+) or flipped (-)

**2. Algebraic: Solvability Test**
- det(A) ≠ 0 means A is invertible
- det(A) = 0 means A is singular (not invertible)

**3. Computational: Building Block**
- Used in eigenvalue calculations
- Appears in change of variables formulas
- Foundation for Cramer's rule

**In ML:**
- Jacobian determinants in normalizing flows
- Covariance matrix determinants for Gaussians
- Stability analysis of dynamical systems

---

## Part 1: Geometric Meaning - Area and Volume Scaling

### 2D Case: Area Scaling

Consider the unit square with vertices at (0,0), (1,0), (0,1), (1,1).

When we apply a matrix transformation A = [[a, b], [c, d]], the unit square becomes a parallelogram.

```
Before:           After applying A:
  (0,1)───(1,1)      (b, d)───(a+b, c+d)
    │       │           \         \
    │       │            \         \
  (0,0)───(1,0)      (0, 0)───(a, c)
```

**Theorem:** The area of the transformed parallelogram equals |det(A)|.

**Derivation:**

The unit square has vertices at **0**, **e₁** = [1, 0]ᵀ, **e₂** = [0, 1]ᵀ, and **e₁** + **e₂**.

After transformation:
- **0** → **0**
- **e₁** → A**e₁** = [a, c]ᵀ (first column of A)
- **e₂** → A**e₂** = [b, d]ᵀ (second column of A)

The parallelogram is spanned by vectors **u** = [a, c]ᵀ and **v** = [b, d]ᵀ.

Area of parallelogram = base × height = |**u**| × |**v**| × sin(θ)

Using the cross product formula:
$$\text{Area} = |u_1 v_2 - u_2 v_1| = |ad - bc| = |\det(A)|$$

### 3D Case: Volume Scaling

For a 3×3 matrix, det(A) gives the volume scaling factor.

The unit cube becomes a parallelepiped, and:
$$\text{Volume of parallelepiped} = |\det(A)|$$

### Orientation and Sign

- **det(A) > 0**: Transformation preserves orientation (right-handed stays right-handed)
- **det(A) < 0**: Transformation reverses orientation (like a mirror reflection)
- **det(A) = 0**: Dimension collapse (3D → 2D plane, or 2D → 1D line)

**Example:** Reflection matrix across y-axis:
$$R = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$$

det(R) = -1 (area preserved but orientation flipped)

---

## Part 2: The 2×2 Determinant Formula

For a 2×2 matrix:
$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

$$\det(A) = ad - bc$$

### Intuitive Derivation from Area

Consider the parallelogram with sides **u** = [a, c] and **v** = [b, d].

Using the "box method":
- Large rectangle containing parallelogram: (a+b) × (c+d)
- Subtract the parts outside the parallelogram

Actually, the cleanest derivation uses the cross product in 3D:

Embed **u** and **v** in 3D: **u** = [a, c, 0], **v** = [b, d, 0]

$$\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a & c & 0 \\ b & d & 0 \end{vmatrix} = (ad - bc)\mathbf{k}$$

The magnitude |ad - bc| is the area of the parallelogram.

### Properties from the Formula

From det(A) = ad - bc:

1. **Swapping rows negates determinant:**
   $$\det\begin{bmatrix} c & d \\ a & b \end{bmatrix} = cb - da = -(ad - bc)$$

2. **Scaling a row scales determinant:**
   $$\det\begin{bmatrix} ka & kb \\ c & d \end{bmatrix} = kad - kbc = k(ad - bc)$$

3. **det(I) = 1:** det([[1, 0], [0, 1]]) = 1·1 - 0·0 = 1

---

## Part 3: General Definition via Cofactor Expansion

### The 3×3 Case

For:
$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

**Cofactor expansion along first row:**

$$\det(A) = a_{11} \det\begin{bmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{bmatrix} - a_{12} \det\begin{bmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{bmatrix} + a_{13} \det\begin{bmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}$$

**Pattern:** Expand along row i:
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

where $M_{ij}$ is the **minor** (determinant of matrix with row i and column j deleted).

The term $C_{ij} = (-1)^{i+j} M_{ij}$ is called the **cofactor**.

### Why Alternating Signs?

The signs follow a checkerboard pattern:
```
+ - + - ...
- + - + ...
+ - + - ...
...
```

This comes from the definition of determinant as a sum over permutations:
$$\det(A) = \sum_{\sigma \in S_n} \text{sign}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}$$

### The n×n General Case

**Definition (Leibniz formula):**
$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}$$

where $S_n$ is the set of all permutations of {1, 2, ..., n} and sgn(σ) is +1 for even permutations, -1 for odd.

**Computational note:** This definition involves n! terms, so direct computation is O(n!). We need better methods!

---

## Part 4: Key Properties of Determinants

These properties make determinants easier to compute and understand.

### Property 1: Multilinearity in Rows

The determinant is linear in each row separately:

**a) Scaling:** If row i is multiplied by c, determinant is multiplied by c
$$\det(\ldots, c\mathbf{r}_i, \ldots) = c \cdot \det(\ldots, \mathbf{r}_i, \ldots)$$

**b) Additivity:** If one row is a sum, determinant splits
$$\det(\ldots, \mathbf{r}_i + \mathbf{r}'_i, \ldots) = \det(\ldots, \mathbf{r}_i, \ldots) + \det(\ldots, \mathbf{r}'_i, \ldots)$$

### Property 2: Alternating (Antisymmetric)

Swapping any two rows negates the determinant:
$$\det(\ldots, \mathbf{r}_i, \ldots, \mathbf{r}_j, \ldots) = -\det(\ldots, \mathbf{r}_j, \ldots, \mathbf{r}_i, \ldots)$$

**Consequence:** If two rows are identical, det(A) = 0.
*Proof:* Swapping identical rows should negate determinant, but result is the same matrix. So det(A) = -det(A), meaning det(A) = 0.

### Property 3: Determinant of Identity

$$\det(I) = 1$$

### Property 4: Row Operations

**a) Adding multiple of one row to another doesn't change determinant:**
$$\det(\ldots, \mathbf{r}_i + c\mathbf{r}_j, \ldots, \mathbf{r}_j, \ldots) = \det(\ldots, \mathbf{r}_i, \ldots, \mathbf{r}_j, \ldots)$$

This is crucial for Gaussian elimination!

**b) Swapping rows negates determinant** (Property 2)

**c) Scaling a row scales determinant** (Property 1a)

### Property 5: Product Rule

$$\det(AB) = \det(A) \det(B)$$

**Proof sketch:** Both sides are multilinear, alternating functions of the rows of AB that equal 1 when AB = I. Such functions are unique.

**Consequence:** If A is invertible, det(A⁻¹) = 1/det(A)

### Property 6: Transpose

$$\det(A^T) = \det(A)$$

This means all row properties also apply to columns!

### Property 7: Triangular Matrices

For upper or lower triangular matrices:
$$\det(A) = a_{11} \cdot a_{22} \cdot \ldots \cdot a_{nn} = \prod_{i=1}^{n} a_{ii}$$

Determinant of triangular matrix = product of diagonal entries.

---

## Part 5: Computing Determinants Efficiently

### Method 1: Gaussian Elimination (Best for Large Matrices)

Reduce to upper triangular form using row operations:

1. For each row operation:
   - Adding multiple of row to another: det unchanged
   - Swapping rows: det negates
   - Scaling row by c: det multiplied by c

2. For triangular result U: det(U) = product of diagonal

**Example:**
$$A = \begin{bmatrix} 2 & 1 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

Row 2 - 2×Row 1:
$$\begin{bmatrix} 2 & 1 & 3 \\ 0 & 3 & 0 \\ 7 & 8 & 9 \end{bmatrix}$$

Row 3 - 3.5×Row 1:
$$\begin{bmatrix} 2 & 1 & 3 \\ 0 & 3 & 0 \\ 0 & 4.5 & -1.5 \end{bmatrix}$$

Row 3 - 1.5×Row 2:
$$\begin{bmatrix} 2 & 1 & 3 \\ 0 & 3 & 0 \\ 0 & 0 & -1.5 \end{bmatrix}$$

det(A) = 2 × 3 × (-1.5) = -9

**Complexity:** O(n³) - same as matrix multiplication!

### Method 2: Cofactor Expansion (Small Matrices)

Good for 3×3 or matrices with many zeros.

**Strategy:** Expand along row/column with most zeros.

**Example:**
$$A = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 3 & 0 \\ 4 & 0 & 5 \end{bmatrix}$$

Expand along column 2 (two zeros!):
$$\det(A) = -0 \cdot M_{12} + 3 \cdot M_{22} - 0 \cdot M_{32} = 3 \cdot \det\begin{bmatrix} 1 & 2 \\ 4 & 5 \end{bmatrix} = 3(5-8) = -9$$

### Method 3: LU Decomposition

If A = LU (lower × upper triangular):
$$\det(A) = \det(L) \det(U) = \left(\prod_{i} l_{ii}\right) \left(\prod_{i} u_{ii}\right)$$

Since L typically has 1s on diagonal: det(A) = det(U).

---

## Part 6: The Jacobian Determinant

### Change of Variables in Integration

When changing variables in an integral, we need the Jacobian determinant.

**1D Analogy:**
$$\int f(x) dx = \int f(g(u)) \cdot g'(u) du$$

The |g'(u)| factor accounts for how intervals stretch/shrink.

**2D Case:**
For change of variables (x, y) → (u, v) with x = x(u,v), y = y(u,v):

$$\iint f(x,y) \, dx\,dy = \iint f(x(u,v), y(u,v)) \cdot |J| \, du\,dv$$

where the **Jacobian matrix** is:
$$J = \begin{bmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{bmatrix}$$

and |J| = |det(J)| is the **Jacobian determinant**.

### Why the Jacobian?

The Jacobian matrix is the local linear approximation of the transformation:
$$\begin{bmatrix} dx \\ dy \end{bmatrix} \approx J \begin{bmatrix} du \\ dv \end{bmatrix}$$

The determinant |det(J)| tells us how infinitesimal areas transform:
$$dA_{xy} = |det(J)| \cdot dA_{uv}$$

### Example: Polar Coordinates

Transform: x = r cos(θ), y = r sin(θ)

$$J = \begin{bmatrix} \frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\ \frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta} \end{bmatrix} = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}$$

$$|J| = r\cos^2\theta + r\sin^2\theta = r$$

So: $dx\,dy = r \, dr\,d\theta$ ✓

---

## Part 7: ML Applications

### Application 1: Normalizing Flows

Normalizing flows transform a simple distribution (like Gaussian) into a complex one through a sequence of invertible transformations.

**Key Equation (Change of Variables for PDFs):**

If z has density p(z) and x = f(z) where f is invertible:
$$p(x) = p(z) \cdot \left|\det\left(\frac{\partial f^{-1}}{\partial x}\right)\right| = p(z) \cdot \left|\det\left(\frac{\partial f}{\partial z}\right)\right|^{-1}$$

Taking log:
$$\log p(x) = \log p(z) - \log\left|\det\left(\frac{\partial f}{\partial z}\right)\right|$$

**Challenge:** Computing det(Jacobian) is O(n³)!

**Solutions in Practice:**
1. **Triangular Jacobians:** Use transformations where Jacobian is triangular (det = product of diagonal)
2. **Autoregressive flows:** Each output depends only on previous inputs → triangular Jacobian
3. **RealNVP:** Coupling layers with easy determinants

### Application 2: Gaussian Distributions

The multivariate Gaussian PDF is:
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

The term $|\Sigma|^{1/2} = \sqrt{\det(\Sigma)}$ is the **normalization constant**.

**Interpretation:** det(Σ) measures the "spread" of the Gaussian.
- Large det(Σ) → distribution spread out → lower peak
- Small det(Σ) → distribution concentrated → higher peak

**Log-likelihood:**
$$\log p(\mathbf{x}) = -\frac{1}{2}\log|\Sigma| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu}) + \text{const}$$

Computing log|Σ| efficiently is crucial for Gaussian processes and VAEs.

### Application 3: Feature Colinearity Detection

If features are perfectly correlated (multicolinearity):
$$\det(\mathbf{X}^T\mathbf{X}) \approx 0$$

This indicates:
- Redundant features
- Numerical instability in linear regression
- Need for regularization or feature selection

---

## Part 8: Cramer's Rule (Historical/Theoretical Importance)

For the system Ax = b where A is n×n with det(A) ≠ 0:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

where $A_i$ is A with column i replaced by b.

**Example (2×2):**
$$\begin{cases} ax + by = e \\ cx + dy = f \end{cases}$$

$$x = \frac{ed - bf}{ad - bc} = \frac{\det\begin{bmatrix} e & b \\ f & d \end{bmatrix}}{\det\begin{bmatrix} a & b \\ c & d \end{bmatrix}}$$

**Practical use:** Cramer's rule is O(n! × n) vs O(n³) for Gaussian elimination. It's mainly theoretical, but useful for:
- Deriving formulas symbolically
- Understanding when solutions exist
- Computing single components of solution

---

## Summary

| Concept | Formula/Definition | ML Relevance |
|---------|-------------------|--------------|
| 2×2 determinant | ad - bc | Quick invertibility check |
| Geometric meaning | Volume scaling factor | Understanding transformations |
| Cofactor expansion | Σ (-1)^{i+j} aᵢⱼ Mᵢⱼ | Computing small determinants |
| Product rule | det(AB) = det(A)det(B) | Composing transformations |
| Triangular det | Product of diagonal | Efficient computation |
| Jacobian | det(∂f/∂x) | Change of variables, normalizing flows |
| Gaussian normalization | (2π)^{-n/2} |Σ|^{-1/2} | Probability computations |

---

## Key Takeaways

1. **Determinant = volume scaling** - Always think geometrically first
2. **det(A) = 0 means dimension collapse** - Matrix is not invertible
3. **Row operations for computation** - O(n³) via Gaussian elimination
4. **Jacobian for change of variables** - Essential for probability density transformations
5. **In ML, avoid explicit determinants when possible** - Use log-determinants and triangular forms

---

*Next: Tutorial 05 - Eigenvalues and Eigenvectors*
