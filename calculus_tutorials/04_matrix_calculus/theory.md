# Tutorial 04: Matrix Calculus

## Why Matrix Calculus?

Neural networks operate on vectors and matrices. To derive backpropagation, we need to understand derivatives involving:
- Vectors
- Matrices
- Products of matrices

This tutorial covers the essential identities you need for ML.

---

## Notation Conventions

Two main conventions exist:

### Numerator Layout (we use this)
- $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ has shape (dim $\mathbf{y}$) × (dim $\mathbf{x}$)
- Gradient $\nabla_{\mathbf{x}} f$ is a column vector

### Denominator Layout
- $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ has shape (dim $\mathbf{x}$) × (dim $\mathbf{y}$)
- Gradient is a row vector

**Always check which convention a paper uses!**

---

## Part 1: Scalar-by-Vector Derivatives

### Definition

For $f: \mathbb{R}^n \to \mathbb{R}$ and $\mathbf{x} \in \mathbb{R}^n$:

$$\frac{\partial f}{\partial \mathbf{x}} = \nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Key Identities

**Identity 1:** $\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$

*Derivation:*
$$\mathbf{a}^T \mathbf{x} = \sum_{i=1}^{n} a_i x_i$$
$$\frac{\partial}{\partial x_j} \sum_{i} a_i x_i = a_j$$

Therefore $\nabla_{\mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$.

---

**Identity 2:** $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}$

*Derivation:*
$$\mathbf{x}^T \mathbf{x} = \sum_{i=1}^{n} x_i^2$$
$$\frac{\partial}{\partial x_j} \sum_{i} x_i^2 = 2x_j$$

Therefore $\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}$.

---

**Identity 3:** $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$

*Derivation:*
$$\mathbf{x}^T A \mathbf{x} = \sum_{i,j} A_{ij} x_i x_j$$

$$\frac{\partial}{\partial x_k} \sum_{i,j} A_{ij} x_i x_j = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i = (A\mathbf{x})_k + (A^T\mathbf{x})_k$$

Therefore $\nabla_{\mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = A\mathbf{x} + A^T\mathbf{x} = (A + A^T)\mathbf{x}$.

**Special case:** If $A$ is symmetric, $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x}$.

---

**Identity 4:** $\frac{\partial}{\partial \mathbf{x}}\|\mathbf{x} - \mathbf{a}\|^2 = 2(\mathbf{x} - \mathbf{a})$

*This is the gradient of L2 loss!*

*Derivation:*
$$\|\mathbf{x} - \mathbf{a}\|^2 = (\mathbf{x} - \mathbf{a})^T(\mathbf{x} - \mathbf{a})$$

Let $\mathbf{u} = \mathbf{x} - \mathbf{a}$. Then:
$$\frac{\partial}{\partial \mathbf{x}}\|\mathbf{u}\|^2 = \frac{\partial}{\partial \mathbf{u}}\|\mathbf{u}\|^2 \cdot \frac{\partial \mathbf{u}}{\partial \mathbf{x}} = 2\mathbf{u} \cdot I = 2(\mathbf{x} - \mathbf{a})$$

---

## Part 2: Vector-by-Vector Derivatives (Jacobian)

### Definition

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

### Key Identities

**Identity 5:** $\frac{\partial}{\partial \mathbf{x}}(A\mathbf{x}) = A$

*Derivation:*
$$(A\mathbf{x})_i = \sum_j A_{ij} x_j$$
$$\frac{\partial (A\mathbf{x})_i}{\partial x_k} = A_{ik}$$

The Jacobian has entry $(i, k) = A_{ik}$, so it equals $A$.

---

**Identity 6:** $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A) = A^T$

This follows from Identity 5 transposed.

---

**Identity 7:** For element-wise function $\mathbf{f}(\mathbf{x}) = [f(x_1), \ldots, f(x_n)]^T$:

$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \text{diag}(f'(x_1), \ldots, f'(x_n))$$

The Jacobian is diagonal!

*Example:* For sigmoid $\sigma(\mathbf{x})$:
$$\frac{\partial \sigma(\mathbf{x})}{\partial \mathbf{x}} = \text{diag}(\sigma(\mathbf{x}) \odot (1 - \sigma(\mathbf{x})))$$

---

## Part 3: Scalar-by-Matrix Derivatives

### Definition

For $f: \mathbb{R}^{m \times n} \to \mathbb{R}$ and matrix $X \in \mathbb{R}^{m \times n}$:

$$\frac{\partial f}{\partial X} = \begin{bmatrix}
\frac{\partial f}{\partial X_{11}} & \cdots & \frac{\partial f}{\partial X_{1n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial X_{m1}} & \cdots & \frac{\partial f}{\partial X_{mn}}
\end{bmatrix}$$

Same shape as $X$!

### Key Identities

**Identity 8:** $\frac{\partial}{\partial X}\text{tr}(AX) = A^T$

*Derivation:*
$$\text{tr}(AX) = \sum_{i,j} A_{ij} X_{ji}$$
$$\frac{\partial}{\partial X_{kl}} \sum_{i,j} A_{ij} X_{ji} = A_{lk}$$

So entry $(k, l)$ of the gradient is $A_{lk} = (A^T)_{kl}$.

---

**Identity 9:** $\frac{\partial}{\partial X}\text{tr}(X^T A) = A$

---

**Identity 10:** $\frac{\partial}{\partial X}\text{tr}(X^T X) = 2X$

*This is Frobenius norm squared!*

---

**Identity 11:** $\frac{\partial}{\partial X}\text{tr}(AXB) = A^T B^T$

---

## Part 4: Chain Rule for Matrices

### The Key Formula

If $\mathcal{L} = \mathcal{L}(\mathbf{y})$ where $\mathbf{y} = \mathbf{f}(\mathbf{x})$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J_{\mathbf{f}}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

Or equivalently:
$$\nabla_{\mathbf{x}} \mathcal{L} = J_{\mathbf{f}}^T \nabla_{\mathbf{y}} \mathcal{L}$$

**This is the backprop equation!**

### Example: Linear Layer

Forward: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$

Jacobians:
- $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = W$
- $\frac{\partial \mathbf{y}}{\partial W}$ — need to be careful (matrix by matrix)

**Gradient w.r.t. input:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

**Gradient w.r.t. weights:**
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T$$

*Derivation:*
$$\mathcal{L} = \mathcal{L}(W\mathbf{x})$$
$$\frac{\partial \mathcal{L}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot x_j$$

This gives $\frac{\partial \mathcal{L}}{\partial W} = \nabla_{\mathbf{y}} \mathcal{L} \cdot \mathbf{x}^T$ (outer product).

**Gradient w.r.t. bias:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

---

## Part 5: Backprop Through Common Layers

### 1. Fully Connected Layer

Forward: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$

Backward:
$$\delta_{\mathbf{x}} = W^T \delta_{\mathbf{y}}$$
$$\delta_W = \delta_{\mathbf{y}} \mathbf{x}^T$$
$$\delta_{\mathbf{b}} = \delta_{\mathbf{y}}$$

where $\delta$ denotes upstream gradient.

### 2. Activation (element-wise)

Forward: $\mathbf{y} = \sigma(\mathbf{x})$

Backward:
$$\delta_{\mathbf{x}} = \delta_{\mathbf{y}} \odot \sigma'(\mathbf{x})$$

Element-wise multiplication!

### 3. Softmax + Cross-Entropy

Forward: 
$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \quad \mathcal{L} = -\sum_i y_i \log p_i$$

Backward (combined):
$$\delta_{\mathbf{z}} = \mathbf{p} - \mathbf{y}$$

*This beautiful simplification is why we use softmax + cross-entropy!*

### 4. Matrix Multiplication

Forward: $C = AB$ where $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$

Backward:
$$\delta_A = \delta_C B^T$$
$$\delta_B = A^T \delta_C$$

---

## Summary of Essential Identities

| Expression | Derivative |
|------------|------------|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$ | $(A + A^T)\mathbf{x}$ |
| $A\mathbf{x}$ | $A$ (Jacobian) |
| $\text{tr}(AX)$ | $A^T$ |
| $\text{tr}(X^T X)$ | $2X$ |

---

## Practical Tips

1. **Shape checking**: Output gradient should have same shape as variable
2. **Outer products**: When gradient involves "expanding" dimensions, use outer product
3. **Trace trick**: Convert scalar expressions to trace form to use identities
4. **Index notation**: When in doubt, write out index form and differentiate

---

## Key Takeaways for ML

1. **Backprop** = chain rule with Jacobians
2. For vector inputs: $\nabla_{\mathbf{x}} \mathcal{L} = J^T \nabla_{\mathbf{y}} \mathcal{L}$
3. For matrix weights: $\nabla_W \mathcal{L} = \nabla_{\mathbf{y}} \mathcal{L} \cdot \mathbf{x}^T$
4. Element-wise operations → diagonal Jacobian → element-wise backward
5. **Dimensions must match!** Use this to sanity-check your gradients
