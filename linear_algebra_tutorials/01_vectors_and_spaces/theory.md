# Tutorial 01: Vectors and Vector Spaces

## Introduction: What is a Vector?

A vector is one of the most fundamental objects in mathematics and ML. But what *is* it?

### Three Perspectives

**1. Physicist's View: An Arrow**
- A vector is an arrow with magnitude and direction
- Position doesn't matter, only length and direction
- Example: velocity (30 mph northeast)

**2. Computer Scientist's View: A List of Numbers**
- A vector is an ordered list: [3, 2, 1]
- Each number is a component or coordinate
- Example: RGB color [255, 128, 0] = orange

**3. Mathematician's View: An Element of a Vector Space**
- A vector is anything that follows certain rules (axioms)
- Could be arrows, functions, matrices...
- Abstract but powerful

**In ML, we use all three:**
- Arrows help us visualize gradients
- Lists represent feature vectors
- Abstract view lets us work in high dimensions

---

## Part 1: Vectors as Arrows

### The Geometric Picture

In 2D, a vector **v** = [3, 2] means:
- Start at origin (0, 0)
- Go 3 units right (x-direction)
- Go 2 units up (y-direction)
- The arrow from origin to (3, 2) is the vector

```
    y
    │
  2 │     ●  (3, 2)
    │    /
  1 │   /
    │  /
    │ /
    └──────────── x
       1   2   3
```

### Vector Operations: Geometric Meaning

#### Addition: Tip-to-Tail

To add **u** + **v**:
1. Draw **u** starting from origin
2. Draw **v** starting from the tip of **u**
3. The sum is the arrow from origin to the tip of **v**

```
        ●  u + v
       /|
      / |
  v  /  | v
    /   |
   /    |
  ●─────●
    u
```

**Why this rule?** It models sequential movements:
- Walk 3 blocks east (**u**)
- Then 2 blocks north (**v**)
- Net displacement = **u** + **v**

#### Scalar Multiplication: Stretching

To compute c**v** (scalar c times vector **v**):
- |c| > 1: stretch the vector
- |c| < 1: shrink the vector
- c < 0: flip direction

```
2v:  ────────────→
v:   ─────→
0.5v: ──→
-v:  ←─────
```

**In ML:** Learning rate multiplies the gradient vector!

---

## Part 2: Vectors as Lists of Numbers

### From Geometry to Algebra

A vector in ℝⁿ is an ordered n-tuple of real numbers:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

### Algebraic Operations

**Addition** (component-wise):
$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$

**Scalar Multiplication**:
$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

### The Dot Product: Where Algebra Meets Geometry

**Definition:**
$$\mathbf{u} \cdot \mathbf{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n = \sum_{i=1}^n u_i v_i$$

**Geometric Interpretation:**
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where θ is the angle between vectors.

**Derivation: Why is dot product related to angle?**

Consider vectors **u** and **v**. Form a triangle with sides **u**, **v**, and **u** - **v**.

By the law of cosines:
$$\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Expand the left side algebraically:
$$\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v})$$
$$= \mathbf{u} \cdot \mathbf{u} - 2\mathbf{u} \cdot \mathbf{v} + \mathbf{v} \cdot \mathbf{v}$$
$$= \|\mathbf{u}\|^2 - 2\mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|^2$$

Comparing:
$$\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Therefore:
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta \quad \blacksquare$$

### What the Dot Product Tells Us

| Value of u·v | Meaning |
|--------------|---------|
| > 0 | Vectors point in similar directions (θ < 90°) |
| = 0 | Vectors are perpendicular (θ = 90°) |
| < 0 | Vectors point in opposite directions (θ > 90°) |
| = ‖u‖‖v‖ | Vectors are parallel (θ = 0°) |

**In ML: Cosine Similarity**
$$\text{similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos\theta$$

Used in:
- Word embeddings (word2vec)
- Recommendation systems
- Document similarity

---

## Part 3: Vector Norms (Length/Magnitude)

### L2 Norm (Euclidean Length)

$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

**Derivation from Pythagorean theorem (2D case):**
- Vector [a, b] forms a right triangle
- Horizontal leg = a, Vertical leg = b
- Hypotenuse = √(a² + b²)

For nD, apply Pythagorean theorem recursively.

### Other Norms

**L1 Norm (Manhattan Distance):**
$$\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

Called "Manhattan" because it's like walking on a grid.

**L∞ Norm (Max Norm):**
$$\|\mathbf{v}\|_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|)$$

**General Lp Norm:**
$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}$$

### Unit Vectors

A unit vector has length 1. To normalize any vector:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**In ML:** We often normalize feature vectors so magnitude doesn't dominate.

---

## Part 4: Vector Spaces (The Abstract View)

### Definition

A **vector space** V over real numbers ℝ is a set with two operations:
1. Vector addition: **u** + **v** ∈ V
2. Scalar multiplication: c**v** ∈ V

that satisfy these axioms:

**Addition Axioms:**
1. **u** + **v** = **v** + **u** (commutativity)
2. (**u** + **v**) + **w** = **u** + (**v** + **w**) (associativity)
3. There exists **0** such that **v** + **0** = **v** (zero vector)
4. For each **v**, there exists -**v** such that **v** + (-**v**) = **0** (additive inverse)

**Scalar Multiplication Axioms:**
5. c(d**v**) = (cd)**v** (associativity)
6. 1**v** = **v** (multiplicative identity)
7. c(**u** + **v**) = c**u** + c**v** (distributive over vectors)
8. (c + d)**v** = c**v** + d**v** (distributive over scalars)

### Examples of Vector Spaces

**1. ℝⁿ** - n-dimensional real vectors ✓

**2. Functions** - All functions f: ℝ → ℝ
- (f + g)(x) = f(x) + g(x)
- (cf)(x) = c · f(x)
- Zero vector: f(x) = 0 for all x

**3. Polynomials** - All polynomials of degree ≤ n

**4. Matrices** - All m × n matrices

### Why Abstract?

The abstract definition lets us:
- Apply the same theorems to different types of objects
- Work in infinite dimensions (functions)
- Reason without coordinates

---

## Part 5: Subspaces, Span, and Basis

### Subspace

A **subspace** W of V is a subset that is itself a vector space:
1. **0** ∈ W (contains zero)
2. **u**, **v** ∈ W ⇒ **u** + **v** ∈ W (closed under addition)
3. **v** ∈ W, c ∈ ℝ ⇒ c**v** ∈ W (closed under scalar multiplication)

**Example:** In ℝ³, any plane through the origin is a subspace.

### Linear Combination

A **linear combination** of vectors **v₁**, **v₂**, ..., **vₖ** is:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

where c₁, c₂, ..., cₖ are scalars.

### Span

The **span** of vectors is the set of all their linear combinations:

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

**Intuition:**
- span({**v**}) = line through origin in direction **v**
- span({**u**, **v**}) = plane through origin (if **u**, **v** not parallel)

### Linear Independence

Vectors **v₁**, ..., **vₖ** are **linearly independent** if:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$

implies c₁ = c₂ = ... = cₖ = 0.

**Intuition:** No vector can be written as a combination of the others.

**Geometric meaning in ℝ³:**
- 1 vector: independent unless it's zero
- 2 vectors: independent unless parallel (same line)
- 3 vectors: independent unless coplanar (same plane)
- 4+ vectors: always dependent (too many for 3D space)

### Basis

A **basis** for V is a set of vectors that are:
1. Linearly independent
2. Span V

**The dimension** of V = number of vectors in any basis.

**Standard Basis for ℝⁿ:**
$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

Any vector **v** = [v₁, v₂, ..., vₙ]ᵀ can be written as:
$$\mathbf{v} = v_1\mathbf{e}_1 + v_2\mathbf{e}_2 + \cdots + v_n\mathbf{e}_n$$

---

## Part 6: Applications in Machine Learning

### Feature Vectors

Each data point is a vector:
$$\mathbf{x} = \begin{bmatrix} \text{age} \\ \text{income} \\ \text{years\_education} \\ \vdots \end{bmatrix}$$

A dataset with N samples and D features is an N × D matrix.

### Word Embeddings

Words become vectors where similar words are close:
- king ≈ [0.2, 0.8, ...]
- queen ≈ [0.3, 0.75, ...]
- apple ≈ [-0.5, 0.1, ...]

Famous result: king - man + woman ≈ queen

This works because vector arithmetic captures semantic relationships!

### Neural Network Weights

Layer i has weight matrix W⁽ⁱ⁾. Each row is a "detector" vector.

Output of neuron j = **w**ⱼ · **x** + bⱼ

High dot product ⟹ input aligns with what neuron detects.

### Gradient Descent

The gradient ∇L is a vector pointing toward steepest increase in loss.

Update: **θ** ← **θ** - η∇L

We move opposite to gradient (toward steepest decrease).

---

## Summary

| Concept | Definition | ML Relevance |
|---------|------------|--------------|
| Vector | Arrow / list of numbers | Data points, weights, gradients |
| Dot product | Σuᵢvᵢ = ‖u‖‖v‖cos θ | Similarity, neural network outputs |
| Norm | Length of vector | Regularization, normalization |
| Span | All linear combinations | Expressiveness of model |
| Basis | Independent spanning set | Dimensionality, representations |
| Linear independence | No redundancy | Feature selection |

---

## Key Takeaways

1. **Vectors have geometry** - Always try to visualize
2. **Dot product = similarity** - Fundamental to neural nets
3. **Basis gives coordinates** - Different bases = different representations
4. **Linear independence = no redundancy** - Important for efficient models
5. **Everything in ML is high-dimensional linear algebra**

---

*Next: Tutorial 02 - Matrices and Linear Transformations*
