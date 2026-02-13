# Tutorial 13: Autoencoders - Exercises

## Part A: Theory Derivations

### Exercise A1 (Easy)
**Linear autoencoder = PCA**

Consider a single-layer linear autoencoder:
- Encoder: $z = Wx$ where $W \in \mathbb{R}^{d \times n}$
- Decoder: $\hat{x} = W'z$ where $W' \in \mathbb{R}^{n \times d}$
- Loss: $\mathcal{L} = \|x - W'Wx\|^2$

1. Show that the optimal $W'$ satisfies $W' = (W^TW)^{-1}W^T$ (the pseudo-inverse of $W$).
2. Show that when $W$ has orthonormal rows ($WW^T = I_d$), the optimal solution spans the same subspace as the top $d$ principal components.
3. What happens if we add a bias term to the encoder and decoder?

---

### Exercise A2 (Easy)
**Bottleneck capacity**

An autoencoder has input dimension $n = 784$ (MNIST) and latent dimension $d$.

1. For $d = 2$, how many bits of information can the bottleneck represent (assuming each latent dimension uses 32-bit float)?
2. How does this compare to the raw input (784 pixels, 8 bits each)?
3. Calculate the compression ratio for $d = 2, 10, 32, 64, 128$.

---

### Exercise A3 (Medium)
**Sparse autoencoder: KL penalty gradient**

For the KL sparsity penalty:
$$\Omega_{sparse} = \sum_{j=1}^d D_{KL}(\rho \| \hat{\rho}_j)$$

where $\hat{\rho}_j = \frac{1}{N}\sum_{i=1}^N z_j^{(i)}$ is the average activation.

1. Derive $\frac{\partial \Omega_{sparse}}{\partial \hat{\rho}_j}$.
2. Using the chain rule, express $\frac{\partial \Omega_{sparse}}{\partial W}$ for a single-layer encoder $z = \sigma(Wx + b)$.
3. What happens to the gradient when $\hat{\rho}_j \to 0$ or $\hat{\rho}_j \to 1$?

---

### Exercise A4 (Medium)
**Contractive autoencoder: Jacobian computation**

For encoder $z = \sigma(Wx + b)$ with sigmoid activation $\sigma$:

1. Compute the Jacobian $J = \frac{\partial z}{\partial x}$.
2. Express $\|J\|_F^2$ in terms of the weights $W$ and activations $z$.
3. Show that the contractive penalty encourages the encoder to be "flat" in directions where data doesn't vary.

---

### Exercise A5 (Hard)
**Denoising autoencoder and score matching**

For Gaussian corruption $\tilde{x} = x + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$:

1. Show that the optimal denoising function is $g^*(\tilde{x}) = \mathbb{E}[X | \tilde{X} = \tilde{x}]$.
2. Using Tweedie's formula, show that $g^*(\tilde{x}) = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})$.
3. Conclude that training a denoising autoencoder implicitly learns the score function $\nabla_x \log p(x)$.

---

## Part B: Coding Exercises

### Exercise B1 (Easy)
**Implement a basic autoencoder**

```python
import numpy as np

class Autoencoder:
    """
    Simple fully-connected autoencoder.

    Architecture: n -> hidden -> d -> hidden -> n

    Args:
        input_dim: Dimension of input data (n)
        hidden_dim: Dimension of hidden layers
        latent_dim: Dimension of latent space (d)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        # Initialize weights with Xavier initialization
        # YOUR CODE HERE
        pass

    def encode(self, x):
        """Encode input to latent representation."""
        # YOUR CODE HERE
        pass

    def decode(self, z):
        """Decode latent representation to reconstruction."""
        # YOUR CODE HERE
        pass

    def forward(self, x):
        """Full forward pass: encode then decode."""
        # YOUR CODE HERE
        pass

    def compute_loss(self, x, x_hat):
        """Compute MSE reconstruction loss."""
        # YOUR CODE HERE
        pass
```

---

### Exercise B2 (Medium)
**Implement a sparse autoencoder**

```python
class SparseAutoencoder(Autoencoder):
    """
    Autoencoder with KL-divergence sparsity penalty.

    Args:
        input_dim, hidden_dim, latent_dim: Architecture params
        sparsity_target: Target average activation (rho)
        sparsity_weight: Weight of sparsity penalty (lambda)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 sparsity_target=0.05, sparsity_weight=1.0):
        # YOUR CODE HERE
        pass

    def kl_divergence(self, rho, rho_hat):
        """
        Compute KL(rho || rho_hat) for Bernoulli distributions.

        Args:
            rho: Target sparsity (scalar)
            rho_hat: Actual average activation (vector of size d)

        Returns:
            KL divergence penalty (scalar)
        """
        # YOUR CODE HERE
        pass

    def compute_loss(self, x, x_hat, z):
        """
        Compute reconstruction loss + sparsity penalty.

        Returns:
            total_loss, reconstruction_loss, sparsity_loss
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B3 (Medium)
**Implement a denoising autoencoder**

```python
class DenoisingAutoencoder(Autoencoder):
    """
    Autoencoder trained to reconstruct clean input from corrupted input.

    Args:
        input_dim, hidden_dim, latent_dim: Architecture params
        noise_type: 'gaussian', 'masking', or 'salt_pepper'
        noise_level: Intensity of corruption
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 noise_type='masking', noise_level=0.3):
        # YOUR CODE HERE
        pass

    def corrupt(self, x):
        """
        Apply corruption to input.

        For masking: randomly zero out pixels with probability noise_level
        For gaussian: add N(0, noise_level^2) noise
        For salt_pepper: randomly set to 0 or 1

        Returns:
            Corrupted input (same shape as x)
        """
        # YOUR CODE HERE
        pass

    def train_step(self, x):
        """
        One training step:
        1. Corrupt input
        2. Forward pass with corrupted input
        3. Compute loss against CLEAN input

        Returns:
            loss, corrupted_input, reconstruction
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B4 (Hard)
**Compare autoencoder with PCA on MNIST**

```python
def compare_ae_pca(X_train, X_test, latent_dims=[2, 5, 10, 32]):
    """
    Compare reconstruction quality of autoencoder vs PCA
    across different latent dimensions.

    For each latent_dim:
    1. Train PCA with latent_dim components
    2. Train autoencoder with latent_dim latent units
    3. Compute reconstruction MSE on test set
    4. Visualize reconstructions side by side

    Args:
        X_train: Training data, shape (N, 784)
        X_test: Test data, shape (M, 784)
        latent_dims: List of latent dimensions to compare

    Returns:
        Dictionary with MSE for each method and latent_dim
    """
    # YOUR CODE HERE
    pass

def visualize_latent_space(encoder, X, labels, method='autoencoder'):
    """
    Visualize 2D latent space colored by digit labels.

    Compare autoencoder latent space vs PCA projection.

    Args:
        encoder: Trained encoder function
        X: Data, shape (N, 784)
        labels: Digit labels, shape (N,)
        method: 'autoencoder' or 'pca'
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B5 (Hard)
**Anomaly detection with autoencoders**

```python
class AnomalyDetector:
    """
    Use a trained autoencoder for anomaly detection.

    Strategy: Train on normal data, flag high reconstruction error as anomalous.
    """

    def __init__(self, autoencoder, threshold_percentile=95):
        # YOUR CODE HERE
        pass

    def fit(self, X_normal):
        """
        Fit the anomaly detector on normal data.

        1. Compute reconstruction errors for all normal samples
        2. Set threshold at the given percentile

        Args:
            X_normal: Normal training data, shape (N, n)
        """
        # YOUR CODE HERE
        pass

    def predict(self, X):
        """
        Predict whether each sample is anomalous.

        Returns:
            anomaly_scores: Reconstruction error for each sample
            is_anomaly: Boolean array (True if anomalous)
        """
        # YOUR CODE HERE
        pass

    def evaluate(self, X_normal_test, X_anomaly_test):
        """
        Evaluate using AUROC and precision-recall.

        Args:
            X_normal_test: Normal test samples
            X_anomaly_test: Anomalous test samples

        Returns:
            auroc, precision, recall, f1
        """
        # YOUR CODE HERE
        pass
```

---

## Part C: Conceptual Questions

### C1 (Easy)
Why does a bottleneck (undercomplete representation) prevent the autoencoder from learning the identity function? Under what conditions could an undercomplete autoencoder still learn a trivial mapping?

### C2 (Easy)
Explain the difference between the encoder being a "compressor" and the decoder being a "decompressor." What determines what information gets preserved vs. discarded?

### C3 (Medium)
A denoising autoencoder with masking noise (30% of pixels zeroed out) achieves better representation quality than a standard autoencoder. Explain why from two perspectives:
1. The manifold learning perspective
2. The information-theoretic perspective

### C4 (Medium)
Compare sparse autoencoders to L1-regularized linear regression (Lasso). What are the similarities and differences? When would you use each?

### C5 (Hard)
From an entropy perspective, explain the trade-off between reconstruction quality and latent space regularity in autoencoders. How does this relate to the rate-distortion function $R(D)$?

### C6 (Hard)
Modern self-supervised learning methods (SimCLR, BYOL, MAE) can be seen as descendants of autoencoders. Compare masked autoencoders (MAE) with denoising autoencoders. What changed, and what fundamental principle stayed the same?
