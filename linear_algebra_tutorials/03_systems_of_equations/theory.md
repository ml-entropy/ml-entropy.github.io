# Tutorial 03: Systems of Linear Equations

## Introduction: Why Systems of Equations?

Almost every ML algorithm boils down to solving some system of equations:
- **Linear regression:** Find weights that minimize error
- **Neural networks:** Backprop solves for gradients
- **Optimization:** Find where gradient = 0

Understanding how to solve A**x** = **b** is fundamental.

---

## Part 1: What is a System of Linear Equations?

### The Basic Setup

A system of m linear equations in n unknowns:

$$a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1$$
$$a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2$$
$$\vdots$$
$$a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m$$

### Matrix Form

$$A\mathbf{x} = \mathbf{b}$$

where:
- A is m×n (coefficient matrix)
- **x** is n×1 (unknowns)
- **b** is m×1 (right-hand side)

### Geometric Interpretation

**2D Example:** Two lines
- Each equation is a line
- Solution = intersection point

**3D Example:** Three planes
- Each equation is a plane
- Solution = point where all three meet

**In general:** Each equation defines a hyperplane. Solution is their intersection.

---

## Part 2: When Do Solutions Exist?

### Three Possibilities

**1. Unique Solution** (determined system)
- Lines/planes intersect at exactly one point
- rank(A) = rank([A|**b**]) = n

**2. No Solution** (inconsistent system)
- Lines are parallel (never meet)
- rank(A) < rank([A|**b**])

**3. Infinitely Many Solutions** (underdetermined system)
- Lines are the same, or planes intersect in a line
- rank(A) = rank([A|**b**]) < n

### The Rank Theorem

A system A**x** = **b** has a solution if and only if:
$$\text{rank}(A) = \text{rank}([A | \mathbf{b}])$$

(The augmented matrix [A|**b**] includes the right-hand side.)

### Geometric Intuition

**b** must lie in the column space of A!

Why? Because A**x** = x₁**a₁** + x₂**a₂** + ... = linear combination of columns.

If **b** isn't in span(columns of A), no solution exists.

---

## Part 3: Gaussian Elimination

### The Algorithm

Transform [A|**b**] to row echelon form using three operations:
1. **Swap** two rows
2. **Multiply** a row by a nonzero constant
3. **Add** a multiple of one row to another

These operations don't change the solution!

### Example

Solve:
$$x + 2y + z = 9$$
$$2x + y + z = 8$$
$$3x + y + 2z = 13$$

**Step 1:** Augmented matrix
$$\begin{bmatrix} 1 & 2 & 1 & | & 9 \\ 2 & 1 & 1 & | & 8 \\ 3 & 1 & 2 & | & 13 \end{bmatrix}$$

**Step 2:** Eliminate below first pivot
- R2 ← R2 - 2R1
- R3 ← R3 - 3R1

$$\begin{bmatrix} 1 & 2 & 1 & | & 9 \\ 0 & -3 & -1 & | & -10 \\ 0 & -5 & -1 & | & -14 \end{bmatrix}$$

**Step 3:** Eliminate below second pivot
- R3 ← R3 - (5/3)R2

$$\begin{bmatrix} 1 & 2 & 1 & | & 9 \\ 0 & -3 & -1 & | & -10 \\ 0 & 0 & 2/3 & | & 4/3 \end{bmatrix}$$

**Step 4:** Back substitution
- From R3: (2/3)z = 4/3 → z = 2
- From R2: -3y - 2 = -10 → y = 8/3
- From R1: x + 16/3 + 2 = 9 → x = 5/3

Solution: **x** = [5/3, 8/3, 2]ᵀ

### Computational Complexity

Gaussian elimination is O(n³) for an n×n system.

---

## Part 4: LU Decomposition

### The Idea

Factor A = LU where:
- L is lower triangular (1s on diagonal)
- U is upper triangular

Then solving A**x** = **b** becomes:
1. Solve L**y** = **b** (forward substitution)
2. Solve U**x** = **y** (back substitution)

### Why LU is Useful

**Same A, different b:** If you solve many systems with the same A but different **b**, you only factor once!

This happens in:
- Iterative methods
- Control systems
- Some ML optimization algorithms

### Derivation

LU decomposition records the steps of Gaussian elimination:
- U = final upper triangular matrix
- L = multipliers used in elimination

**Example:**
$$A = \begin{bmatrix} 2 & 1 \\ 6 & 4 \end{bmatrix}$$

Eliminate: R2 ← R2 - 3R1

$$U = \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}, \quad L = \begin{bmatrix} 1 & 0 \\ 3 & 1 \end{bmatrix}$$

Check: LU = $\begin{bmatrix} 1 & 0 \\ 3 & 1 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 6 & 4 \end{bmatrix}$ = A ✓

---

## Part 5: Special Case - Square Invertible Matrices

### When A is Invertible

If A is n×n and invertible (det(A) ≠ 0):

$$A\mathbf{x} = \mathbf{b} \implies \mathbf{x} = A^{-1}\mathbf{b}$$

Unique solution exists for any **b**.

### Computing A⁻¹ (Don't Do This!)

In practice, **never compute A⁻¹ explicitly** to solve A**x** = **b**.

Why not?
1. Computing A⁻¹ is O(n³), then A⁻¹**b** is O(n²)
2. Direct solve is also O(n³) but with smaller constant
3. A⁻¹ can have numerical issues

Instead: Use LU decomposition or iterative methods.

---

## Part 6: Least Squares (When No Exact Solution Exists)

### The Problem

When m > n (more equations than unknowns), usually no exact solution exists.

**Example:** Fitting a line to 100 data points (2 unknowns, 100 equations).

### The Least Squares Solution

Find **x** that minimizes ||A**x** - **b**||²

**Derivation:**

Let f(**x**) = ||A**x** - **b**||² = (A**x** - **b**)ᵀ(A**x** - **b**)

Expand:
$$f = \mathbf{x}^TA^TA\mathbf{x} - 2\mathbf{b}^TA\mathbf{x} + \mathbf{b}^T\mathbf{b}$$

Take gradient and set to zero:
$$\nabla f = 2A^TA\mathbf{x} - 2A^T\mathbf{b} = 0$$

### The Normal Equations

$$A^TA\mathbf{x} = A^T\mathbf{b}$$

If AᵀA is invertible:
$$\mathbf{x} = (A^TA)^{-1}A^T\mathbf{b}$$

### Geometric Interpretation

The least squares solution projects **b** onto the column space of A.

```
         b
        /|
       / |
      /  | error
     /   |
    /    ↓
   Ax ●──● projection
      ←──→
   Column space of A
```

The error (A**x** - **b**) is perpendicular to the column space.

---

## Part 7: Linear Regression as Least Squares

### Setup

Data: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)

Model: y = w₀ + w₁x

### As a Linear System

$$\begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix} \begin{bmatrix} w_0 \\ w_1 \end{bmatrix} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

This is A**w** = **y** with A being the "design matrix."

### Solution

Normal equations: AᵀA**w** = Aᵀ**y**

For simple linear regression:

$$A^TA = \begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix}$$

$$A^T\mathbf{y} = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix}$$

Solving gives the familiar formulas for slope and intercept!

---

## Part 8: Iterative Methods (Preview)

### When Direct Methods Fail

For very large systems (millions of variables), O(n³) is too slow.

### Iterative Approach

Start with guess **x**⁽⁰⁾, then improve iteratively:

**x**⁽ᵏ⁺¹⁾ = f(**x**⁽ᵏ⁾)

until convergence.

### Gradient Descent for Least Squares

Minimize f(**x**) = ||A**x** - **b**||²

Gradient: ∇f = 2Aᵀ(A**x** - **b**)

Update: **x** ← **x** - α · 2Aᵀ(A**x** - **b**)

This is how neural networks solve for weights!

### Conjugate Gradient

Smarter than gradient descent for linear systems:
- Converges in at most n iterations (exact arithmetic)
- Uses O(n²) memory instead of O(n³)
- Standard for large sparse systems

---

## Part 9: Numerical Considerations

### Condition Number

The condition number κ(A) = ||A|| · ||A⁻¹|| measures sensitivity to perturbations.

**High κ (ill-conditioned):**
- Small changes in **b** cause large changes in solution
- Numerical errors amplified
- Need special care (regularization)

**Low κ (well-conditioned):**
- Stable solution
- Standard methods work fine

### Regularization

Add small value to diagonal: (AᵀA + λI)**x** = Aᵀ**b**

This is **ridge regression** - improves conditioning!

---

## Part 10: ML Applications

### Linear Regression
- Normal equations: (XᵀX)**w** = Xᵀ**y**
- Often add regularization: (XᵀX + λI)**w** = Xᵀ**y**

### Neural Network Training
- Finding optimal weights = solving optimization problem
- Backprop computes gradients for gradient descent
- At each step: approximately solving a linear system

### Solving for Embeddings
- Matrix factorization: find U, V such that UV ≈ R
- Each step involves linear systems

### Gaussian Processes
- Prediction requires solving: K**α** = **y**
- K is the kernel matrix
- Often very large → need iterative methods

---

## Summary

| Situation | Method |
|-----------|--------|
| Small system, one solve | Gaussian elimination |
| Same A, multiple **b** | LU decomposition |
| Overdetermined (m > n) | Least squares (normal equations) |
| Large sparse system | Iterative methods (CG, GMRES) |
| Ill-conditioned | Regularization |

---

## Key Takeaways

1. **A**x** = **b** encodes many ML problems**
2. **Solution exists iff b is in column space of A**
3. **Least squares finds closest solution when exact doesn't exist**
4. **Never invert matrices explicitly** - use decompositions
5. **Condition number determines numerical stability**
6. **Linear regression = least squares = normal equations**

---

*Next: Tutorial 04 - Determinants*
