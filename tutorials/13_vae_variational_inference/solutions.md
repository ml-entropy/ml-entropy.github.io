# Tutorial 04: VAE & Variational Inference — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Why Latent Variables?

Latent variables solve the **intractability of complex distributions**.

Real-world data (images, text) lives in high-dimensional space with complex dependencies. Direct modeling is intractable:
- $p(x)$ for a 256×256 image has $2^{256×256×3}$ states (for binary pixels)

**Latent variables** introduce structure:
$$p(x) = \int p(x|z)p(z)dz$$

Benefits:
1. **Disentanglement**: $z$ can capture meaningful factors (pose, lighting, identity)
2. **Compression**: Low-dimensional $z$ captures essence of high-dimensional $x$
3. **Generation**: Sample $z \sim p(z)$, then decode $x \sim p(x|z)$

---

### Solution A2 — ELBO Derivation

Start with log marginal likelihood:
$$\log p(x) = \log \int p(x, z) dz = \log \int p(x|z)p(z) dz$$

Introduce approximate posterior $q(z|x)$:
$$\log p(x) = \log \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz = \log E_{q(z|x)}\left[\frac{p(x|z)p(z)}{q(z|x)}\right]$$

Apply Jensen's inequality ($\log$ is concave):
$$\log p(x) \geq E_{q(z|x)}\left[\log\frac{p(x|z)p(z)}{q(z|x)}\right]$$

Expand:
$$\log p(x) \geq E_{q}[\log p(x|z)] + E_{q}\left[\log\frac{p(z)}{q(z|x)}\right]$$

$$\boxed{\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z)) = \text{ELBO}}$$

---

### Solution A3 — ELBO Interpretation

$$\text{ELBO} = \underbrace{E_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) || p(z))}_{\text{Regularization}}$$

1. **Reconstruction term**: Expected log-likelihood of data given latent codes
   - Encourages encoder to produce $z$'s that decoder can reconstruct well
   - For Gaussian decoder: becomes MSE loss

2. **KL term**: Penalty for posterior deviating from prior
   - Encourages $q(z|x)$ to stay close to $p(z) = N(0, I)$
   - Prevents encoder from "cheating" by encoding each $x$ as a point
   - Enables generation: sample from $p(z)$, decode

**Trade-off**: Too much KL → poor reconstruction. Too little → no regularization.

---

### Solution A4 — Reparameterization Trick

**Problem**: $z \sim q_\phi(z|x) = N(\mu_\phi(x), \sigma^2_\phi(x))$

Can't backprop through sampling! Gradient of sample w.r.t. distribution parameters undefined.

**Solution**: Reparameterize as deterministic function of parameters + noise:
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim N(0, I)$$

Now:
- $\epsilon$ is independent of $\phi$ (can sample once, reuse)
- $z$ is differentiable w.r.t. $\mu, \sigma$
- Gradients flow through!

$$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$

---

### Solution A5 — KL for Gaussian Posterior

$$q(z|x) = N(\mu, \text{diag}(\sigma^2)), \quad p(z) = N(0, I)$$

For univariate Gaussians (then sum over dimensions):
$$D_{KL}(N(\mu, \sigma^2) || N(0, 1)) = \int q(z) \log\frac{q(z)}{p(z)} dz$$

$$= \int q(z) \left[\log q(z) - \log p(z)\right] dz$$

Using $\log N(\mu, \sigma^2) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}$:

$$D_{KL} = -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}E_q[z^2]$$

Since $E_q[z^2] = \mu^2 + \sigma^2$:

$$D_{KL} = \frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$$

For $d$ dimensions:
$$\boxed{D_{KL} = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)}$$

---

### Solution A6 — Posterior Collapse

**What**: Decoder ignores $z$, encoder outputs $q(z|x) \approx p(z)$ for all $x$.

**Why it happens**:
1. Powerful decoders (e.g., autoregressive) can model $p(x)$ without $z$
2. KL term pushes $q \to p$, which minimizes information in $z$
3. Decoder learns to ignore uninformative $z$

**Mitigation**:
1. **KL annealing**: Start with low KL weight, increase over training
2. **Free bits**: Minimum KL per dimension before penalty
3. **Weaker decoders**: Force reliance on $z$
4. **β-VAE**: Use $\beta < 1$ for KL weight

---

## Part B: Coding Solutions

### Solution B1 — Reparameterization

```python
import torch

def reparameterize(mu, log_var):
    """
    Reparameterization trick: z = μ + σ * ε
    
    Args:
        mu: Mean of q(z|x), shape [batch, latent_dim]
        log_var: Log variance of q(z|x), shape [batch, latent_dim]
    
    Returns:
        z: Sampled latent, shape [batch, latent_dim]
    """
    # Standard deviation
    std = torch.exp(0.5 * log_var)
    
    # Sample ε ~ N(0, I)
    eps = torch.randn_like(std)
    
    # z = μ + σ * ε
    z = mu + std * eps
    
    return z

# Test gradient flow
mu = torch.randn(32, 10, requires_grad=True)
log_var = torch.randn(32, 10, requires_grad=True)
z = reparameterize(mu, log_var)
loss = z.sum()
loss.backward()

print(f"mu.grad exists: {mu.grad is not None}")  # True
print(f"log_var.grad exists: {log_var.grad is not None}")  # True
```

### Solution B2 — KL Loss

```python
import torch

def kl_divergence(mu, log_var):
    """
    KL(N(μ, σ²) || N(0, I))
    
    Formula: 0.5 * sum(μ² + σ² - log(σ²) - 1)
    
    Args:
        mu: [batch, latent_dim]
        log_var: [batch, latent_dim]
    
    Returns:
        kl: [batch] KL for each sample
    """
    # σ² = exp(log_var)
    # KL = 0.5 * (μ² + σ² - log(σ²) - 1)
    kl = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=1)
    return kl

# Test
mu = torch.zeros(32, 10)
log_var = torch.zeros(32, 10)  # σ² = 1
kl = kl_divergence(mu, log_var)
print(f"KL for N(0,1) || N(0,1): {kl[0].item():.4f}")  # Should be ~0

mu = torch.ones(32, 10) * 2
kl = kl_divergence(mu, log_var)
print(f"KL for N(2,1) || N(0,1): {kl[0].item():.4f}")  # Should be ~20
```

### Solution B3 — Full VAE

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

def vae_loss(x, x_recon, mu, log_var):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss

# Training loop (simplified)
def train_vae(epochs=10):
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(latent_dim=2).to(device)  # 2D for visualization
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            
            x_recon, mu, log_var = model(x)
            loss = vae_loss(x, x_recon, mu, log_var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_data):.2f}")
    
    return model

# Visualize latent space
def visualize_latent(model, test_loader, device):
    model.eval()
    zs, labels = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)
            mu, _ = model.encode(x)
            zs.append(mu.cpu())
            labels.append(y)
    
    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title('VAE Latent Space (colored by digit)')
    plt.show()

# Run if executed
if __name__ == "__main__":
    model = train_vae(epochs=5)
```

---

## Part C: Conceptual Solutions

### C1
Using $\log\sigma^2$ instead of $\sigma$:
1. **Unconstrained output**: $\log\sigma^2 \in (-\infty, \infty)$, but $\sigma > 0$
2. **Numerical stability**: Avoids issues with very small $\sigma$
3. **Symmetric gradients**: Easier optimization (log-space is more symmetric)

### C2
**Standard Autoencoder**:
- Deterministic: $z = f(x)$
- Latent space can have "holes" (regions not covered by any data)
- No probabilistic interpretation

**VAE**:
- Stochastic: $z \sim q(z|x)$
- KL term forces $q(z|x) \approx p(z) = N(0,I)$
- Fills the latent space continuously
- Can sample from prior and decode → generation!

The KL term is crucial: it prevents "memorization" by spreading encodings and ensuring the decoder must learn smooth interpolations.
