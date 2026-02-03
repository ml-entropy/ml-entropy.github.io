# Linear Algebra for Machine Learning

A comprehensive tutorial series covering linear algebra concepts essential for understanding and implementing machine learning algorithms.

## Why Linear Algebra for ML?

**Everything in ML is linear algebra:**
- Neural networks = matrix multiplications + nonlinearities
- Data = matrices (samples × features)
- Images = 3D tensors (height × width × channels)
- PCA = eigenvalue decomposition
- Recommendations = matrix factorization
- Transformers = attention matrices

## Tutorial Overview

| # | Topic | Key ML Applications |
|---|-------|---------------------|
| 01 | [Vectors and Vector Spaces](01_vectors_and_spaces/) | Feature vectors, embeddings, word2vec |
| 02 | [Matrices and Linear Transformations](02_matrices_and_transformations/) | Neural network layers, data transformations |
| 03 | [Systems of Linear Equations](03_systems_of_equations/) | Linear regression, solving Ax=b |
| 04 | [Determinants](04_determinants/) | Change of variables, Jacobians, probability |
| 05 | [Eigenvalues and Eigenvectors](05_eigenvalues_eigenvectors/) | PCA, PageRank, stability analysis |
| 06 | [Singular Value Decomposition](06_svd/) | Dimensionality reduction, recommenders, compression |
| 07 | [Orthogonality and Projections](07_orthogonality_projections/) | Least squares, Gram-Schmidt, QR decomposition |
| 08 | [Positive Definite Matrices](08_positive_definite_matrices/) | Covariance matrices, optimization, Gaussian distributions |

## Learning Path

```
01. Vectors ──────────────────────────────────────────┐
    │                                                 │
02. Matrices ─────────────────┬───────────────────────┤
    │                         │                       │
03. Systems ──────────────────┤                       │
    │                         │                       │
04. Determinants ─────────────┴───────────────────────┤
    │                                                 │
05. Eigenvalues ──────────────┬───────────────────────┤
    │                         │                       │
06. SVD ──────────────────────┤                       │
    │                         │                       │
07. Orthogonality ────────────┴───────────────────────┤
    │                                                 │
08. Positive Definite ────────────────────────────────┘
```

## Each Tutorial Contains

- **theory.md**: Intuitive explanations, derivations from first principles
- **notebook.ipynb**: Interactive visualizations and code implementations
- **exercises.md**: Practice problems (theory + coding)
- **solutions.md**: Detailed solutions with explanations

## Core Philosophy

1. **Geometric Intuition First**: Every concept has a geometric meaning
2. **Build from Simple**: Start with 2D/3D cases you can visualize
3. **Connect to ML**: Every topic links to real ML applications
4. **Derive, Don't Memorize**: Understand where formulas come from

## Prerequisites

- Basic algebra
- Some familiarity with Python/NumPy
- Curiosity about how ML really works

## Installation

```bash
pip install numpy matplotlib seaborn jupyter
```

## Quick Reference: Key Intuitions

| Concept | Intuition |
|---------|-----------|
| Vector | An arrow pointing somewhere in space (or a list of features) |
| Matrix | A transformation that rotates, scales, or shears space |
| Determinant | How much a transformation scales area/volume |
| Eigenvalue | Factor by which eigenvectors get stretched |
| Eigenvector | Directions that don't change under transformation |
| SVD | "Rotate → Scale → Rotate" decomposition of any matrix |
| Projection | Shadow of a vector onto a subspace |
| Positive Definite | Matrix whose quadratic form is always a bowl (convex) |

---

*Part of the ML Fundamentals Tutorial Series*
