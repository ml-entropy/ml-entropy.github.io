# Tutorial 11: Learning Rate - Solutions

## Part A: Theory Solutions

### Solution A1: Convergence Condition

**Step 1: Gradient**
$$\nabla f(x) = ax$$

**Step 2: Update rule**
$$x_{t+1} = x_t - \eta \cdot ax_t = (1 - \eta a)x_t$$

**Step 3: Unroll recurrence**
$$x_t = (1 - \eta a)x_{t-1} = (1 - \eta a)^2 x_{t-2} = ... = (1 - \eta a)^t x_0$$

**Step 4: Convergence condition**
For $x_t \to 0$, we need $|1 - \eta a| < 1$

$$-1 < 1 - \eta a < 1$$
$$-2 < -\eta a < 0$$
$$0 < \eta a < 2$$
$$\boxed{0 < \eta < \frac{2}{a}}$$

---

### Solution A2: Momentum Derivation

**Step 1: Expand velocity**
$$v_t = \beta v_{t-1} + g_{t-1} = \beta(\beta v_{t-2} + g_{t-2}) + g_{t-1} = ...$$
$$v_t = \sum_{k=0}^{t-1} \beta^k g_{t-1-k}$$

**Step 2: Effective learning rate for old gradient**
A gradient from $k$ steps ago contributes $\beta^k$ to current velocity, so effective learning rate is $\eta \beta^k$.

**Step 3: Total effective learning rate**
Sum of geometric series:
$$\eta \sum_{k=0}^{\infty} \beta^k = \eta \cdot \frac{1}{1-\beta} = \boxed{\frac{\eta}{1-\beta}}$$

For $\beta = 0.9$: effective lr = $10\eta$

---

### Solution A3: RMSprop

**Step 1: What $s_t$ approximates**
$s_t$ is an exponential moving average of squared gradients. It approximates $\mathbb{E}[g^2]$, the second moment.

**Step 2: Why divide by $\sqrt{s_t}$**
- Parameters with large gradients get smaller effective learning rates
- Parameters with small gradients get larger effective learning rates
- This "normalizes" the learning rate per parameter

**Step 3: Effective learning rate**
For constant gradient $g$:
$$s_t = (1-\beta)g^2 \sum_{k=0}^{t-1} \beta^k = (1-\beta)g^2 \cdot \frac{1-\beta^t}{1-\beta} = (1-\beta^t)g^2$$

Effective learning rate:
$$\frac{\eta}{\sqrt{s_t + \epsilon}} = \frac{\eta}{\sqrt{(1-\beta^t)g^2 + \epsilon}}$$

As $t \to \infty$: $\approx \frac{\eta}{|g|}$

---

### Solution A4: Adam Bias Correction

**Step 1: Expected value of $m_t$**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

Assuming constant $\mathbb{E}[g]$:
$$\mathbb{E}[m_t] = (1-\beta_1)\mathbb{E}[g]\sum_{k=0}^{t-1}\beta_1^k = (1-\beta_1)\mathbb{E}[g] \cdot \frac{1-\beta_1^t}{1-\beta_1}$$
$$\boxed{\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]}$$

**Step 2: Bias correction**
We want $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

Verify: $\mathbb{E}[\hat{m}_t] = \frac{(1-\beta_1^t)\mathbb{E}[g]}{1-\beta_1^t} = \mathbb{E}[g]$ ✓

**Step 3: Why important early**
At $t=1$ with $\beta_1 = 0.9$:
- Without correction: $\mathbb{E}[m_1] = 0.1 \cdot \mathbb{E}[g]$ (biased low by 10x!)
- With correction: $\hat{m}_1 = \frac{m_1}{0.1}$ (unbiased)

---

### Solution A5: Optimal Learning Rate

**Step 1: Gradient**
$$\nabla f(x) = Ax$$

**Step 2: Convergence condition**
In the eigenbasis of $A$, each coordinate decays as $(1 - \eta \lambda_i)^t$.

For convergence: $|1 - \eta \lambda_i| < 1$ for all $i$

Most restrictive: largest eigenvalue
$$|1 - \eta \lambda_{max}| < 1 \implies \boxed{\eta < \frac{2}{\lambda_{max}}}$$

**Step 3: Optimal learning rate**
Error in direction $i$ decays as $(1 - \eta \lambda_i)^t$.

Optimal balances fastest/slowest directions:
$$|1 - \eta \lambda_{max}| = |1 - \eta \lambda_{min}|$$

Solving: $\eta^* = \frac{2}{\lambda_{max} + \lambda_{min}}$

**Step 4: Condition number**
$$\kappa = \frac{\lambda_{max}}{\lambda_{min}}$$

Large $\kappa$ means:
- Elongated contours (ravines)
- Optimal learning rate for one direction is bad for another
- Convergence rate: $(1 - \frac{2}{\kappa+1})^t \approx (1 - \frac{2}{\kappa})^t$

For $\kappa = 100$: convergence factor ≈ 0.98 per step (slow!)

---

## Part B: Coding Solutions

### Solution B1: Basic Optimizers

```python
import numpy as np

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self, grads):
        for i, grad in enumerate(grads):
            self.params[i] -= self.lr * grad

class MomentumSGD:
    def __init__(self, params, lr, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in params]
    
    def step(self, grads):
        for i, grad in enumerate(grads):
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            self.params[i] -= self.lr * self.velocities[i]
```

---

### Solution B2: Adam

```python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        
        for i, grad in enumerate(grads):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

### Solution B3: Optimizer Trajectories

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_optimizers_2d():
    # Loss function: f(x, y) = x^2 + 10*y^2
    def loss(pos):
        return pos[0]**2 + 10*pos[1]**2
    
    def grad(pos):
        return np.array([2*pos[0], 20*pos[1]])
    
    # Starting point
    start = np.array([5.0, 4.0])
    
    # Optimizers
    def run_sgd(start, lr, steps):
        pos = start.copy()
        trajectory = [pos.copy()]
        for _ in range(steps):
            pos -= lr * grad(pos)
            trajectory.append(pos.copy())
        return np.array(trajectory)
    
    def run_momentum(start, lr, momentum, steps):
        pos = start.copy()
        vel = np.zeros(2)
        trajectory = [pos.copy()]
        for _ in range(steps):
            vel = momentum * vel + grad(pos)
            pos -= lr * vel
            trajectory.append(pos.copy())
        return np.array(trajectory)
    
    def run_rmsprop(start, lr, beta, eps, steps):
        pos = start.copy()
        s = np.zeros(2)
        trajectory = [pos.copy()]
        for _ in range(steps):
            g = grad(pos)
            s = beta * s + (1 - beta) * g**2
            pos -= lr * g / (np.sqrt(s) + eps)
            trajectory.append(pos.copy())
        return np.array(trajectory)
    
    def run_adam(start, lr, beta1, beta2, eps, steps):
        pos = start.copy()
        m, v = np.zeros(2), np.zeros(2)
        trajectory = [pos.copy()]
        for t in range(1, steps + 1):
            g = grad(pos)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            pos -= lr * m_hat / (np.sqrt(v_hat) + eps)
            trajectory.append(pos.copy())
        return np.array(trajectory)
    
    steps = 50
    trajectories = {
        'SGD (lr=0.05)': run_sgd(start, 0.05, steps),
        'Momentum (lr=0.05, β=0.9)': run_momentum(start, 0.05, 0.9, steps),
        'RMSprop (lr=0.5)': run_rmsprop(start, 0.5, 0.9, 1e-8, steps),
        'Adam (lr=0.5)': run_adam(start, 0.5, 0.9, 0.999, 1e-8, steps),
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Contour plot
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 10*Y**2
    ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
    
    # Trajectories
    colors = ['red', 'blue', 'green', 'orange']
    for (name, traj), color in zip(trajectories.items(), colors):
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, label=name, markersize=3)
    
    ax.scatter([0], [0], marker='*', s=200, c='gold', edgecolor='black', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Optimizer Trajectories on Elongated Bowl: f(x,y) = x² + 10y²')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

visualize_optimizers_2d()
```

---

### Solution B4: Learning Rate Finder

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def lr_range_test(model, train_loader, min_lr=1e-7, max_lr=10, num_steps=100):
    # Save initial weights
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Compute learning rate multiplier
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)
    criterion = nn.CrossEntropyLoss()
    
    lrs = []
    losses = []
    lr = min_lr
    
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        
        # Forward pass
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        
        # Record
        lrs.append(lr)
        losses.append(loss.item())
        
        # Check for explosion
        if loss.item() > 4 * losses[0]:
            break
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Restore initial weights
    model.load_state_dict(initial_state)
    
    # Find suggested learning rate (steepest descent)
    smoothed_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
    gradients = np.gradient(smoothed_losses)
    suggested_idx = np.argmin(gradients)
    suggested_lr = lrs[suggested_idx + 2]  # Account for smoothing offset
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.semilogx(lrs, losses)
    plt.axvline(suggested_lr, color='red', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return lrs, losses, suggested_lr
```

---

### Solution B5: Cosine Annealing with Warm Restarts

```python
import numpy as np
import matplotlib.pyplot as plt

class CosineAnnealingWarmRestarts:
    def __init__(self, lr_max, lr_min=0, T_0=10, T_mult=2):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult
    
    def get_lr(self, epoch):
        # Find which restart period we're in
        T_i = self.T_0
        T_cur = epoch
        
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        # Cosine annealing within period
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * T_cur / T_i))
        return lr
    
    def plot_schedule(self, total_epochs=100):
        epochs = np.arange(total_epochs)
        lrs = [self.get_lr(e) for e in epochs]
        
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Cosine Annealing with Warm Restarts (T_0={self.T_0}, T_mult={self.T_mult})')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return lrs

# Example
scheduler = CosineAnnealingWarmRestarts(lr_max=0.1, lr_min=0.001, T_0=10, T_mult=2)
scheduler.plot_schedule(100)
```

---

## Part C: Conceptual Answers

### C1: Why Learning Rate is Most Important
- **Too high**: Overshoots minima, loss oscillates or explodes, training diverges
- **Too low**: Gets stuck in local minima, training takes forever, may never reach good solution
- **Just right**: Converges efficiently to good solution

Learning rate directly controls the scale of all updates. A bad LR wastes all other tuning effort.

### C2: Momentum Intuition
a) **Ravines**: Oscillations in steep direction get canceled (opposite gradients), while consistent direction accumulates momentum. Accelerates along the valley floor.

b) **Noisy gradients**: Noise tends to be random, canceling out. True gradient direction accumulates. Momentum acts as a low-pass filter.

c) **Local minima**: Momentum provides "inertia" to roll through shallow local minima. Like a ball rolling through small bumps.

### C3: Why Adam "Just Works"
Adam adapts learning rate per parameter based on gradient history:
- Automatic scaling: no need to tune LR for different layers
- Handles varying gradient magnitudes across parameters
- Bias correction handles early training instability

SGD uses same learning rate for all parameters, which may be suboptimal without careful tuning.

### C4: Learning Rate Effects
a) **Large LR and generalization**: Large LR can only converge to "flat" minima (sharp minima would cause overshooting). Flat minima generalize better (more robust to parameter perturbations).

b) **Warmup and stability**: Early gradients are unreliable (random weights). Large LR + bad gradients = bad updates. Warmup lets network "find its footing" before taking big steps.

c) **Decay and convergence**: Near optimum, large steps overshoot. Decreasing LR allows finer adjustments for precise convergence.

### C5: Information-Theoretic View
Learning rate controls the "resolution" of learning:
- **Large LR**: Coarse search, finds broad structures, ignores noise
- **Small LR**: Fine-grained, can fit noise (overfitting)

From minimum description length perspective: large LR finds simpler patterns (lower description length), better for generalization.

### C6: SGD vs Adam Generalization
Adaptive methods like Adam:
- Can have very small effective LR for some parameters
- This allows fitting sharp minima (bad generalization)
- The "effective regularization" of large uniform LR is lost

SGD with large uniform LR:
- Can only converge to flat minima
- Acts as implicit regularizer
- Often generalizes better on image tasks (empirically)

Recent work (SAM, sharpness-aware minimization) tries to get Adam's convenience with SGD's generalization.
