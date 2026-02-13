# Tutorial 13: Autoencoders - Solutions

## Part A: Theory Solutions

### Solution A1: Linear Autoencoder = PCA

**Part 1: Optimal $W'$**

The loss is $\mathcal{L} = \|x - W'Wx\|^2$. Taking derivative w.r.t. $W'$:

$$\frac{\partial \mathcal{L}}{\partial W'} = -2(x - W'Wx)(Wx)^T = 0$$

$$x(Wx)^T = W'(Wx)(Wx)^T$$

Over the dataset: $\mathbb{E}[x(Wx)^T] = W'\mathbb{E}[(Wx)(Wx)^T]$

$$\mathbb{E}[xz^T] = W'\mathbb{E}[zz^T]$$

$$W' = \mathbb{E}[xz^T](\mathbb{E}[zz^T])^{-1}$$

If $W$ has full row rank, $\mathbb{E}[zz^T] = W\Sigma_x W^T$ where $\Sigma_x$ is the data covariance, and:

$$\boxed{W' = (W^T W)^{-1} W^T = W^\dagger \text{ (pseudo-inverse of } W)}$$

Wait, more precisely: $W' = \Sigma_x W^T (W \Sigma_x W^T)^{-1}$.

For orthonormal $W$ ($WW^T = I_d$): $W' = \Sigma_x W^T (W\Sigma_x W^T)^{-1}$. If additionally $W$ aligns with eigenvectors of $\Sigma_x$, this simplifies to $W' = W^T$.

**Part 2: Connection to PCA**

For orthonormal $W$ ($WW^T = I_d$), the reconstruction is $\hat{x} = W^T W x$, which is the projection onto the row space of $W$.

The loss becomes:
$$\mathcal{L} = \|x - W^T W x\|^2 = \|(I - W^T W)x\|^2$$

This is the **projection error**, minimized when $W$'s rows span the top $d$ eigenvectors of $\Sigma_x = \mathbb{E}[xx^T]$.

This is exactly PCA: project onto the principal subspace.

$$\boxed{W^* = [v_1, v_2, \ldots, v_d]^T \text{ where } v_i \text{ are top eigenvectors of } \Sigma_x}$$

**Part 3: Bias term**

Adding bias: $z = Wx + b$, $\hat{x} = W'z + b'$.

The bias term allows the autoencoder to handle non-zero-mean data. It's equivalent to PCA with mean centering: $b$ absorbs the mean, and the linear transformation acts on centered data.

---

### Solution A2: Bottleneck Capacity

**Part 1:** With $d = 2$ latent dimensions, each using 32-bit float:
$$\text{Capacity} = 2 \times 32 = 64 \text{ bits}$$

**Part 2:** Raw MNIST input: $784 \times 8 = 6{,}272$ bits.

**Part 3:** Compression ratios ($d \times 32$ bits for latent vs $784 \times 8 = 6272$ bits for input):

| $d$ | Latent bits | Compression ratio |
|-----|-------------|-------------------|
| 2   | 64          | 98:1              |
| 10  | 320         | 20:1              |
| 32  | 1,024       | 6:1               |
| 64  | 2,048       | 3:1               |
| 128 | 4,096       | 1.5:1             |

Note: The effective information content is much less than the bit-width suggests, since the latent values are continuous and correlated.

---

### Solution A3: Sparse Autoencoder KL Gradient

**Part 1: Gradient of KL penalty**

$$D_{KL}(\rho \| \hat{\rho}_j) = \rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1 - \hat{\rho}_j}$$

$$\frac{\partial D_{KL}}{\partial \hat{\rho}_j} = -\frac{\rho}{\hat{\rho}_j} + \frac{1 - \rho}{1 - \hat{\rho}_j}$$

$$\boxed{\frac{\partial \Omega_{sparse}}{\partial \hat{\rho}_j} = -\frac{\rho}{\hat{\rho}_j} + \frac{1 - \rho}{1 - \hat{\rho}_j}}$$

**Part 2: Chain rule to weights**

$$\frac{\partial \Omega}{\partial W_{jk}} = \frac{\partial \Omega}{\partial \hat{\rho}_j} \cdot \frac{\partial \hat{\rho}_j}{\partial W_{jk}}$$

Since $\hat{\rho}_j = \frac{1}{N}\sum_i \sigma(w_j^T x_i + b_j)$:

$$\frac{\partial \hat{\rho}_j}{\partial W_{jk}} = \frac{1}{N}\sum_i \sigma'(w_j^T x_i + b_j) \cdot x_{ik}$$

For sigmoid: $\sigma'(a) = \sigma(a)(1 - \sigma(a)) = z_j^{(i)}(1 - z_j^{(i)})$.

**Part 3: Boundary behavior**

- When $\hat{\rho}_j \to 0$: $-\rho/\hat{\rho}_j \to -\infty$ (strong push to increase activation)
- When $\hat{\rho}_j \to 1$: $(1-\rho)/(1-\hat{\rho}_j) \to +\infty$ (strong push to decrease activation)
- At $\hat{\rho}_j = \rho$: gradient is zero (equilibrium)

---

### Solution A4: Contractive Autoencoder Jacobian

**Part 1: Jacobian computation**

For $z_i = \sigma(\sum_k W_{ik} x_k + b_i)$:

$$J_{ij} = \frac{\partial z_i}{\partial x_j} = \sigma'(a_i) \cdot W_{ij}$$

where $a_i = \sum_k W_{ik} x_k + b_i$.

For sigmoid: $\sigma'(a_i) = z_i(1 - z_i)$.

$$\boxed{J = \text{diag}(z \odot (1 - z)) \cdot W}$$

**Part 2: Frobenius norm**

$$\|J\|_F^2 = \sum_{i,j} J_{ij}^2 = \sum_{i,j} [z_i(1-z_i)]^2 W_{ij}^2$$

$$\boxed{\|J\|_F^2 = \sum_{i=1}^d z_i^2(1-z_i)^2 \|w_i\|^2}$$

where $w_i$ is the $i$-th row of $W$.

**Part 3: Flatness interpretation**

The penalty $\|J\|_F^2$ penalizes large derivatives $\partial z / \partial x$. This means:
- If input varies in a direction that doesn't change $z$, no penalty
- If input varies in a direction that does change $z$, high penalty

The encoder learns to be sensitive only to directions of **meaningful variation** in the data (the manifold tangent directions) and invariant to other directions (noise).

---

### Solution A5: Denoising and Score Matching

**Part 1: Optimal denoising function**

We want to minimize $\mathbb{E}_{\tilde{x}, x}[\|g(\tilde{x}) - x\|^2]$.

For each fixed $\tilde{x}$, the optimal $g^*(\tilde{x})$ minimizes:
$$\mathbb{E}_{x|\tilde{x}}[\|g(\tilde{x}) - x\|^2]$$

This is minimized by the conditional mean:
$$\boxed{g^*(\tilde{x}) = \mathbb{E}[X | \tilde{X} = \tilde{x}]}$$

**Part 2: Tweedie's formula**

For the Gaussian corruption model $\tilde{x} = x + \sigma\epsilon$:

$$p(\tilde{x}) = \int p(\tilde{x}|x) p(x) dx = \int \mathcal{N}(\tilde{x}; x, \sigma^2 I) p(x) dx$$

Tweedie's formula states:
$$\mathbb{E}[X | \tilde{X} = \tilde{x}] = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})$$

Proof sketch: Take the gradient of $\log p(\tilde{x})$:
$$\nabla_{\tilde{x}} \log p(\tilde{x}) = \frac{\nabla_{\tilde{x}} p(\tilde{x})}{p(\tilde{x})} = \frac{\int \frac{x - \tilde{x}}{\sigma^2} \mathcal{N}(\tilde{x}; x, \sigma^2 I) p(x) dx}{p(\tilde{x})}$$

$$= \frac{1}{\sigma^2}\left(\mathbb{E}[X|\tilde{X} = \tilde{x}] - \tilde{x}\right)$$

Rearranging: $\boxed{g^*(\tilde{x}) = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})}$

**Part 3: Score function learning**

From Part 2, the residual of the optimal denoising function is:
$$g^*(\tilde{x}) - \tilde{x} = \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})$$

Therefore:
$$\nabla_{\tilde{x}} \log p(\tilde{x}) = \frac{g^*(\tilde{x}) - \tilde{x}}{\sigma^2}$$

A denoising autoencoder trained to minimize $\|g(\tilde{x}) - x\|^2$ learns $g \approx g^*$, which implicitly learns the **score function** $\nabla_x \log p(x)$. This is the fundamental connection to score-based diffusion models.

---

## Part B: Coding Solutions

### Solution B1: Basic Autoencoder

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / latent_dim)

        # Encoder
        self.W1 = np.random.randn(hidden_dim, input_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(latent_dim, hidden_dim) * scale2
        self.b2 = np.zeros(latent_dim)

        # Decoder
        self.W3 = np.random.randn(hidden_dim, latent_dim) * scale3
        self.b3 = np.zeros(hidden_dim)
        self.W4 = np.random.randn(input_dim, hidden_dim) * scale2
        self.b4 = np.zeros(input_dim)

    def encode(self, x):
        self.h1 = relu(self.W1 @ x + self.b1)
        z = self.W2 @ self.h1 + self.b2  # linear latent
        return z

    def decode(self, z):
        self.h2 = relu(self.W3 @ z + self.b3)
        x_hat = sigmoid(self.W4 @ self.h2 + self.b4)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(self, x, x_hat):
        return np.mean((x - x_hat) ** 2)
```

---

### Solution B2: Sparse Autoencoder

```python
class SparseAutoencoder(Autoencoder):
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 sparsity_target=0.05, sparsity_weight=1.0):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.rho = sparsity_target
        self.lam = sparsity_weight

    def kl_divergence(self, rho, rho_hat):
        # Clip to avoid log(0)
        rho_hat = np.clip(rho_hat, 1e-8, 1 - 1e-8)
        kl = rho * np.log(rho / rho_hat) + \
             (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
        return np.sum(kl)

    def compute_loss(self, x_batch, x_hat_batch, z_batch):
        # Reconstruction loss
        recon_loss = np.mean(np.sum((x_batch - x_hat_batch) ** 2, axis=1))

        # Average activation over batch
        rho_hat = np.mean(np.abs(z_batch), axis=0)
        # Use sigmoid to map to [0,1] for KL
        rho_hat_sig = sigmoid(rho_hat)

        sparsity_loss = self.lam * self.kl_divergence(self.rho, rho_hat_sig)

        total_loss = recon_loss + sparsity_loss
        return total_loss, recon_loss, sparsity_loss
```

---

### Solution B3: Denoising Autoencoder

```python
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 noise_type='masking', noise_level=0.3):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.noise_type = noise_type
        self.noise_level = noise_level

    def corrupt(self, x):
        if self.noise_type == 'masking':
            mask = np.random.binomial(1, 1 - self.noise_level, size=x.shape)
            return x * mask

        elif self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_level, size=x.shape)
            return np.clip(x + noise, 0, 1)

        elif self.noise_type == 'salt_pepper':
            corrupted = x.copy()
            # Salt (set to 1)
            salt = np.random.random(x.shape) < self.noise_level / 2
            corrupted[salt] = 1.0
            # Pepper (set to 0)
            pepper = np.random.random(x.shape) < self.noise_level / 2
            corrupted[pepper] = 0.0
            return corrupted

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def train_step(self, x):
        # 1. Corrupt
        x_corrupted = self.corrupt(x)

        # 2. Forward pass with corrupted input
        x_hat, z = self.forward(x_corrupted)

        # 3. Loss against CLEAN input
        loss = self.compute_loss(x, x_hat)

        return loss, x_corrupted, x_hat
```

---

### Solution B4: Compare with PCA

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def compare_ae_pca(X_train, X_test, latent_dims=[2, 5, 10, 32]):
    results = {'pca': {}, 'autoencoder': {}}

    for d in latent_dims:
        # PCA
        pca = PCA(n_components=d)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_test_recon_pca = pca.inverse_transform(X_test_pca)
        mse_pca = np.mean((X_test - X_test_recon_pca) ** 2)
        results['pca'][d] = mse_pca

        # Autoencoder
        ae = Autoencoder(784, 256, d)
        # Simple training loop
        lr = 0.001
        for epoch in range(50):
            for i in range(0, len(X_train), 64):
                batch = X_train[i:i+64]
                for x in batch:
                    x_hat, z = ae.forward(x)
                    # Simplified: just compute loss
                    loss = ae.compute_loss(x, x_hat)

        # Evaluate
        mse_ae = 0
        for x in X_test:
            x_hat, _ = ae.forward(x)
            mse_ae += np.mean((x - x_hat) ** 2)
        mse_ae /= len(X_test)
        results['autoencoder'][d] = mse_ae

        print(f"d={d}: PCA MSE={mse_pca:.4f}, AE MSE={mse_ae:.4f}")

    return results

def visualize_latent_space(encoder, X, labels, method='autoencoder'):
    if method == 'pca':
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
    else:
        Z = np.array([encoder(x) for x in X])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10',
                         alpha=0.5, s=1)
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} - 2D Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
```

---

### Solution B5: Anomaly Detection

```python
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class AnomalyDetector:
    def __init__(self, autoencoder, threshold_percentile=95):
        self.ae = autoencoder
        self.percentile = threshold_percentile
        self.threshold = None

    def _reconstruction_error(self, X):
        errors = []
        for x in X:
            x_hat, _ = self.ae.forward(x)
            error = np.mean((x - x_hat) ** 2)
            errors.append(error)
        return np.array(errors)

    def fit(self, X_normal):
        errors = self._reconstruction_error(X_normal)
        self.threshold = np.percentile(errors, self.percentile)
        print(f"Threshold set at {self.threshold:.6f} "
              f"({self.percentile}th percentile)")

    def predict(self, X):
        anomaly_scores = self._reconstruction_error(X)
        is_anomaly = anomaly_scores > self.threshold
        return anomaly_scores, is_anomaly

    def evaluate(self, X_normal_test, X_anomaly_test):
        scores_normal = self._reconstruction_error(X_normal_test)
        scores_anomaly = self._reconstruction_error(X_anomaly_test)

        # Combine for AUROC
        all_scores = np.concatenate([scores_normal, scores_anomaly])
        all_labels = np.concatenate([
            np.zeros(len(scores_normal)),
            np.ones(len(scores_anomaly))
        ])

        auroc = roc_auc_score(all_labels, all_scores)

        # Binary predictions
        all_preds = (all_scores > self.threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        print(f"AUROC: {auroc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

        return auroc, precision, recall, f1
```

---

## Part C: Conceptual Answers

### C1: Bottleneck prevents identity

With $d < n$, the encoding $z \in \mathbb{R}^d$ cannot contain all information in $x \in \mathbb{R}^n$ (by dimensionality). The mapping through $\mathbb{R}^d$ must lose information, so the identity is impossible.

**Exception:** If the data actually lies on a $d$-dimensional manifold (or lower), the autoencoder can perfectly reconstruct even with $d < n$. For example, if all inputs are identical, even $d = 1$ suffices.

### C2: Encoder vs Decoder roles

The **encoder** decides what information to keep, effectively acting as a "feature selector." The **decoder** determines how to reconstruct from those features.

What gets preserved depends on the **loss function**: MSE prioritizes large-scale structure (pixel intensities), while perceptual losses can prioritize semantic content. The architecture also matters: convolutional encoders naturally preserve spatial information.

### C3: Why denoising helps

**Manifold perspective:** The corrupted input is pushed off the data manifold. The autoencoder learns to project back onto the manifold. This makes the learned representation robust to perturbations and captures the true data structure.

**Information-theoretic perspective:** By corrupting the input, we remove some information. The encoder must learn to extract the remaining useful information from the noisy input. This forces it to learn more robust, generalizable features rather than memorizing pixel values.

### C4: Sparse AE vs Lasso

**Similarities:**
- Both encourage sparse representations via L1-type penalties
- Both select a subset of "active" features for each input
- Both can perform feature selection

**Differences:**
- Lasso: linear model, convex optimization, global optimum
- Sparse AE: nonlinear, non-convex, local optima
- Lasso sparsity is on weights; Sparse AE sparsity is on activations
- Sparse AE can learn hierarchical features; Lasso cannot

**Use Lasso when:** Linear model suffices, interpretability is key, data is low-dimensional.
**Use Sparse AE when:** Data is high-dimensional, nonlinear features needed, representation learning is the goal.

### C5: Entropy and rate-distortion

The autoencoder implements an approximate **rate-distortion** trade-off:

- **Rate $R$:** Measured by the capacity of the bottleneck. With $d$ latent dimensions, the rate is bounded by the entropy $H(Z)$.
- **Distortion $D$:** The reconstruction error $\mathbb{E}[\|X - \hat{X}\|^2]$.

Shannon's rate-distortion theorem says there exists a function $R(D)$ giving the minimum rate needed for distortion $\leq D$. The autoencoder tries to find an encoder/decoder pair that operates near this curve. Reducing the bottleneck dimension (lower rate) inevitably increases distortion, and vice versa.

### C6: DAE vs MAE

**Masked Autoencoders (MAE)** mask large patches of an image and reconstruct them. **Denoising Autoencoders (DAE)** add noise to all pixels and reconstruct the clean version.

**What changed:**
- MAE masks large contiguous patches (75%+) vs DAE's pixel-level noise
- MAE uses Vision Transformers vs DAE's fully connected/conv networks
- MAE operates on patches as tokens vs DAE on raw pixels
- MAE's masking is more aggressive, forcing more global reasoning

**What stayed the same:**
- The fundamental principle: learn representations by reconstructing corrupted input
- The manifold learning insight: the model must understand data structure to fill in missing information
- The implicit score matching connection: both learn to move from corrupted to clean data
