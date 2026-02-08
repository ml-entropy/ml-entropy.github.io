# Tutorial 11: Learning Rate - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Derive the convergence condition**

For gradient descent on a quadratic function $f(x) = \frac{1}{2}ax^2$ with update $x_{t+1} = x_t - \eta \nabla f(x_t)$:

1. Compute $\nabla f(x)$
2. Write the update in terms of $x_t$
3. Show that $x_t = (1 - \eta a)^t x_0$
4. For what values of $\eta$ does this converge to 0?

---

### Exercise A2 🟡 (Medium)
**Momentum derivation**

Momentum uses: $v_{t+1} = \beta v_t + \nabla f(x_t)$, then $x_{t+1} = x_t - \eta v_{t+1}$

1. Expand $v_t$ to show it's an exponentially weighted sum of past gradients
2. What's the effective learning rate for a gradient from $k$ steps ago?
3. Show that the "effective" total learning rate is $\frac{\eta}{1-\beta}$

---

### Exercise A3 🟡 (Medium)
**RMSprop derivation**

RMSprop adapts the learning rate per parameter using: $s_t = \beta s_{t-1} + (1-\beta)g_t^2$

1. What does $s_t$ approximate?
2. Why divide by $\sqrt{s_t}$?
3. Show that for constant gradient $g$, the effective learning rate is $\frac{\eta}{\sqrt{(1-\beta^t)g^2 + \epsilon}} \approx \frac{\eta}{|g|}$

---

### Exercise A4 🔴 (Hard)
**Derive Adam bias correction**

Adam initializes $m_0 = 0$ and $v_0 = 0$.

1. Show that $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$ when gradients are constant
2. Derive the bias correction: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$
3. Why is this correction important early in training?

---

### Exercise A5 🔴 (Hard)
**Optimal learning rate for quadratic bowl**

For $f(x) = \frac{1}{2}x^T A x$ where $A$ is positive definite:

1. The gradient is $\nabla f(x) = Ax$
2. Show that convergence requires $\eta < \frac{2}{\lambda_{max}}$ where $\lambda_{max}$ is the largest eigenvalue
3. Show that the optimal learning rate is $\eta^* = \frac{2}{\lambda_{max} + \lambda_{min}}$
4. Derive the condition number $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$ and explain why large $\kappa$ makes optimization hard

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Implement basic optimizers**

```python
class SGD:
    def __init__(self, params, lr):
        """Simple SGD optimizer."""
        # YOUR CODE HERE
        pass
    
    def step(self, grads):
        """Update parameters using gradients."""
        # YOUR CODE HERE
        pass

class MomentumSGD:
    def __init__(self, params, lr, momentum=0.9):
        """SGD with momentum."""
        # YOUR CODE HERE
        pass
    
    def step(self, grads):
        # YOUR CODE HERE
        pass
```

---

### Exercise B2 🟡 (Medium)
**Implement Adam**

```python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam optimizer with bias correction.
        """
        # YOUR CODE HERE
        pass
    
    def step(self, grads):
        """
        Update parameters.
        Remember: apply bias correction!
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B3 🟡 (Medium)
**Visualize optimizer trajectories**

```python
def visualize_optimizers_2d():
    """
    Compare optimizer trajectories on a 2D loss surface.
    
    Use: f(x, y) = x^2 + 10*y^2 (elongated bowl)
    
    1. Initialize all optimizers at same starting point
    2. Run 50 steps for each
    3. Plot trajectories overlaid on contour plot
    4. Compare: SGD, Momentum, RMSprop, Adam
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B4 🔴 (Hard)
**Implement learning rate finder**

```python
def lr_range_test(model, train_loader, min_lr=1e-7, max_lr=10, num_steps=100):
    """
    Learning rate range test (Smith, 2017).
    
    1. Start with very small learning rate
    2. Increase exponentially each step
    3. Record loss at each step
    4. Plot loss vs learning rate
    5. Suggest optimal learning rate (where loss decreases fastest)
    
    Returns:
        lrs: List of learning rates used
        losses: List of losses at each step
        suggested_lr: Suggested learning rate
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B5 🔴 (Hard)
**Implement cosine annealing with warm restarts**

```python
class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts (SGDR).
    
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * T_cur / T_i))
    
    Where T_cur is current epoch within restart, T_i is restart period.
    After each restart, T_i can be multiplied by T_mult.
    """
    
    def __init__(self, lr_max, lr_min=0, T_0=10, T_mult=2):
        # YOUR CODE HERE
        pass
    
    def get_lr(self, epoch):
        """Return learning rate for given epoch."""
        # YOUR CODE HERE
        pass
    
    def plot_schedule(self, total_epochs=100):
        """Plot the learning rate schedule."""
        # YOUR CODE HERE
        pass
```

---

## Part C: Conceptual Questions

### C1 🟢
Why is the learning rate the most important hyperparameter? What happens if it's too high? Too low?

### C2 🟢
Explain intuitively why momentum helps with:
a) Ravines (elongated valleys)
b) Noisy gradients
c) Local minima

### C3 🟡
Why does Adam often work "out of the box" while SGD requires careful tuning?

### C4 🟡
Explain the connection between:
a) Large learning rates and generalization
b) Learning rate warmup and training stability
c) Learning rate decay and convergence

### C5 🔴
From an information-theoretic perspective, how does the learning rate affect what the network learns? (Hint: Think about sharpness of minima and generalization)

### C6 🔴
Some recent papers argue SGD with momentum generalizes better than Adam. Why might adaptive methods like Adam find "worse" minima? (Hint: Consider the implicit regularization of large learning rates)
