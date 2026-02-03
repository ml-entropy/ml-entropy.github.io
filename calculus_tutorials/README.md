# Calculus for Machine Learning

A tutorial series covering the essential calculus concepts needed for deep learning, with emphasis on **intuition**, **derivations**, and **practical applications**.

## Why Calculus in ML?

Neural networks learn by computing **gradients** — derivatives that tell us how to adjust parameters to reduce loss. Understanding derivatives deeply is essential for:

- Understanding backpropagation
- Debugging gradient issues
- Implementing custom layers
- Reading ML papers

## 📚 Tutorial Curriculum

### [01. Single Variable Derivatives](01_single_variable_derivatives/)
**The foundation: rates of change**
- Definition from first principles (limit definition)
- Geometric interpretation: tangent lines
- Common derivatives (polynomial, exponential, logarithm, trigonometric)
- The chain rule derivation
- Product and quotient rules
- Why derivatives of log and exp are special in ML

### [02. Multivariable Derivatives](02_multivariable_derivatives/)
**Functions of many variables**
- Partial derivatives: holding other variables fixed
- The gradient vector: direction of steepest ascent
- The Jacobian matrix: derivatives of vector functions
- The Hessian matrix: second-order information
- Gradient descent intuition

### [03. Directional Derivatives](03_directional_derivatives/)
**Rates of change in any direction**
- Definition and derivation
- Connection to gradient (dot product!)
- Why gradient is direction of steepest ascent (proof)
- Level curves and gradient perpendicularity
- Applications in optimization

### [04. Matrix Calculus](04_matrix_calculus/)
**Derivatives involving matrices and vectors**
- Scalar-by-vector derivatives
- Vector-by-vector derivatives (Jacobian)
- Scalar-by-matrix derivatives
- Key identities: $\frac{\partial}{\partial x}(Ax)$, $\frac{\partial}{\partial x}(x^TAx)$
- Chain rule for matrices
- Backpropagation through linear layers

## 🔧 Setup

```bash
pip install numpy matplotlib jupyter
```

## 📖 How to Use

Each tutorial contains:
- `theory.md` — Detailed explanations with derivations
- `*.ipynb` — Interactive visualizations
- `exercises.md` — Practice problems (theory + coding)
- `solutions.md` — Complete solutions

## Key Notation

| Symbol | Meaning |
|--------|---------|
| $f'(x)$ or $\frac{df}{dx}$ | Derivative of scalar w.r.t. scalar |
| $\nabla f$ | Gradient (vector of partial derivatives) |
| $\frac{\partial f}{\partial x_i}$ | Partial derivative |
| $D_v f$ | Directional derivative in direction $v$ |
| $J$ | Jacobian matrix |
| $H$ | Hessian matrix |
