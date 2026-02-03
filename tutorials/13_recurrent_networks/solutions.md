# Tutorial 13: Recurrent Neural Networks - Solutions

## Part A: Theory Solutions

### Solution A1: RNN Forward Equations

**Equation:**
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**Weight matrices:**
- $W_{xh}$: Input to hidden ($H \times D$)
- $W_{hh}$: Hidden to hidden ($H \times H$)
- $W_{hy}$: Hidden to output ($O \times H$)
- Biases: $b_h$, $b_y$

**Why tanh:**
- Output range $[-1, 1]$: bounded, won't explode
- Zero-centered: helps gradient flow
- Non-linear: provides expressiveness
- Derivative max at 0: good gradient flow for active units

---

### Solution A2: Unrolling

**Unrolled graph:**
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T
         ↑           ↑
      (same W)    (same W)
```

**Weight usage:**
Same weights $W_{xh}$, $W_{hh}$ used $T$ times (once per timestep).

**Multiplications:**
Per timestep: $H \times D$ (input) + $H \times H$ (hidden) = $H(D + H)$
Total: $T \cdot H(D + H)$

---

### Solution A3: BPTT Gradients

**Gradient w.r.t. $W_{hh}$:**
$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

**Jacobian product:**
$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

**Why vanishing/exploding:**
Each Jacobian $\frac{\partial h_i}{\partial h_{i-1}}$ has eigenvalues related to $W_{hh}$.
- If eigenvalues < 1: product shrinks exponentially (vanishing)
- If eigenvalues > 1: product grows exponentially (exploding)

For $T$ timesteps: gradient magnitude $\propto \lambda^T$ where $\lambda$ is max eigenvalue.

---

### Solution A4: Gradient Magnitude Analysis

**Step 1: Jacobian**
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(z_t)) \cdot W_{hh}$$

where $z_t = W_{hh} h_{t-1} + W_{xh} x_t + b$.

Since $\tanh'(z) = 1 - \tanh^2(z) = 1 - h_t^2$:
$$\boxed{\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) \cdot W_{hh}}$$

**Step 2: Gradient bound**
The spectral norm: $\|\frac{\partial h_t}{\partial h_{t-1}}\| \leq \|1 - h_t^2\|_\infty \cdot \|W_{hh}\|$

Since $0 \leq 1 - h_t^2 \leq 1$:
$$\|\frac{\partial h_t}{\partial h_{t-1}}\| \leq \|W_{hh}\| = \lambda_{max}$$

**Step 3: Vanishing condition**
Gradients vanish if $\lambda_{max} < 1$ OR if activations saturate ($|h_t| \approx 1$, so $1 - h_t^2 \approx 0$).

Both conditions lead to gradient magnitude decreasing exponentially with sequence length.

---

### Solution A5: LSTM Gradient Flow

**Step 1: Cell state gradient**
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

(Since $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ and the second term doesn't depend on $c_{t-1}$)

**Step 2: Gradient flow when $f_t \approx 1$**
If $f_t = 1$:
$$\frac{\partial c_T}{\partial c_1} = \prod_{t=2}^{T} f_t = 1$$

Gradients flow unchanged through the cell state highway!

**Step 3: LSTM vanishing condition**
Gradients vanish only if $\prod_{t} f_t \to 0$, which requires:
- Many forget gates < 1
- Or some forget gates $\approx 0$

With forget bias = 1, initial $f_t \approx \sigma(1) \approx 0.73$, keeping gradients flowing.

---

## Part B: Coding Solutions

### Solution B1: RNN Cell

```python
import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        self.W_xh = np.random.randn(hidden_size, input_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        
    def forward(self, x, h_prev):
        z = self.W_xh @ x + self.W_hh @ h_prev + self.b_h
        h = np.tanh(z)
        return h
```

---

### Solution B2: Full RNN with BPTT

```python
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        scale_ih = np.sqrt(2.0 / (input_size + hidden_size))
        scale_ho = np.sqrt(2.0 / (hidden_size + output_size))
        
        self.W_xh = np.random.randn(hidden_size, input_size) * scale_ih
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_ih
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_ho
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        
        self.hidden_size = hidden_size
        
    def forward(self, xs):
        """xs: list of input vectors"""
        T = len(xs)
        h = np.zeros(self.hidden_size)
        
        self.hs = [h]
        self.xs = xs
        self.zs = []
        ys = []
        
        for t in range(T):
            z = self.W_xh @ xs[t] + self.W_hh @ h + self.b_h
            h = np.tanh(z)
            y = self.W_hy @ h + self.b_y
            
            self.zs.append(z)
            self.hs.append(h)
            ys.append(y)
        
        return ys
    
    def backward(self, douts):
        """douts: list of upstream gradients"""
        T = len(douts)
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros(self.hidden_size)
        
        for t in reversed(range(T)):
            # Output layer gradient
            dW_hy += np.outer(douts[t], self.hs[t+1])
            db_y += douts[t]
            
            # Hidden state gradient
            dh = self.W_hy.T @ douts[t] + dh_next
            
            # Backprop through tanh
            dz = dh * (1 - self.hs[t+1]**2)
            
            # Weight gradients
            dW_xh += np.outer(dz, self.xs[t])
            dW_hh += np.outer(dz, self.hs[t])
            db_h += dz
            
            # Gradient to previous hidden state
            dh_next = self.W_hh.T @ dz
        
        return {'W_xh': dW_xh, 'W_hh': dW_hh, 'W_hy': dW_hy, 
                'b_h': db_h, 'b_y': db_y}
```

---

### Solution B3: Gradient Clipping

```python
def clip_gradients(grads, max_norm):
    """Clip gradients by global norm."""
    # Compute global norm
    total_norm = 0
    for g in grads:
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        scale = max_norm / total_norm
        clipped = [g * scale for g in grads]
    else:
        clipped = grads
    
    return clipped

# Alternative: clip each gradient individually
def clip_gradients_elementwise(grads, clip_value):
    return [np.clip(g, -clip_value, clip_value) for g in grads]
```

---

### Solution B4: LSTM Cell

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        scale = np.sqrt(2.0 / concat_size)
        
        # Combined weights for efficiency
        self.W = np.random.randn(4 * hidden_size, concat_size) * scale
        self.b = np.zeros(4 * hidden_size)
        # Initialize forget bias to 1
        self.b[hidden_size:2*hidden_size] = 1.0
        
    def forward(self, x, h_prev, c_prev):
        H = self.hidden_size
        
        # Concatenate input and hidden
        concat = np.concatenate([x, h_prev])
        
        # All gates in one matrix multiply
        gates = self.W @ concat + self.b
        
        # Split gates
        i = sigmoid(gates[:H])           # Input gate
        f = sigmoid(gates[H:2*H])        # Forget gate
        o = sigmoid(gates[2*H:3*H])      # Output gate
        g = np.tanh(gates[3*H:])         # Candidate
        
        # Cell state and hidden state
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        
        cache = (x, h_prev, c_prev, i, f, o, g, c, concat)
        return h, c, cache
    
    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, i, f, o, g, c, concat = cache
        H = self.hidden_size
        
        # Gradient through output
        do = dh * np.tanh(c)
        dc += dh * o * (1 - np.tanh(c)**2)
        
        # Gradient through cell state
        df = dc * c_prev
        dc_prev = dc * f
        di = dc * g
        dg = dc * i
        
        # Gradient through gates
        do_pre = do * o * (1 - o)      # sigmoid derivative
        df_pre = df * f * (1 - f)
        di_pre = di * i * (1 - i)
        dg_pre = dg * (1 - g**2)       # tanh derivative
        
        # Stack gradients
        dgates = np.concatenate([di_pre, df_pre, do_pre, dg_pre])
        
        # Weight gradients
        dW = np.outer(dgates, concat)
        db = dgates
        
        # Input gradients
        dconcat = self.W.T @ dgates
        dx = dconcat[:len(x)]
        dh_prev = dconcat[len(x):]
        
        return {'dW': dW, 'db': db, 'dx': dx, 'dh_prev': dh_prev, 'dc_prev': dc_prev}
```

---

### Solution B5: Gradient Flow Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def analyze_gradient_flow(sequence_length, hidden_size):
    # Generate random sequence
    X = torch.randn(1, sequence_length, hidden_size, requires_grad=True)
    
    results = {}
    
    for name, model_class in [('RNN', nn.RNN), ('LSTM', nn.LSTM)]:
        model = model_class(hidden_size, hidden_size, batch_first=True)
        
        # Forward pass
        output, _ = model(X)
        
        # Loss at final timestep
        loss = output[0, -1].sum()
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Measure gradient at each timestep
        grad_mags = []
        for t in range(sequence_length):
            # Gradient of loss w.r.t. hidden state at time t
            model.zero_grad()
            output, _ = model(X)
            output[0, t].sum().backward(retain_graph=True)
            grad_mags.append(X.grad[0, t].norm().item())
            X.grad.zero_()
        
        results[name] = grad_mags
    
    # Plot
    plt.figure(figsize=(10, 5))
    for name, mags in results.items():
        plt.semilogy(mags, label=name)
    plt.xlabel('Timestep')
    plt.ylabel('Gradient magnitude (log scale)')
    plt.title(f'Gradient Flow: Sequence Length = {sequence_length}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Quantify vanishing
    for name, mags in results.items():
        ratio = mags[-1] / mags[0] if mags[0] > 0 else 0
        print(f"{name}: Final/Initial gradient ratio = {ratio:.2e}")
    
    return results

analyze_gradient_flow(50, 64)
```

---

## Part C: Conceptual Answers

### C1: RNN vs Feedforward Vanishing Gradient
Both have vanishing gradients for deep networks, but RNNs are special:
- **Feedforward**: Each layer has different weights, so products of Jacobians are diverse
- **RNN**: Same weight $W_{hh}$ repeated $T$ times → $W_{hh}^T$

The repeated multiplication of the same matrix causes eigenvalues to compound. If $\lambda_{max} < 1$, gradients decay as $\lambda^T$. This is much more severe than varied matrices in feedforward nets.

### C2: LSTM Gate Roles
- **Forget gate ($f_t$)**: Decides what to erase from cell state. "How much of the past do we keep?"
- **Input gate ($i_t$)**: Decides what new information to write. "How much of the new candidate do we add?"
- **Output gate ($o_t$)**: Decides what part of cell state to expose as hidden state. "What do we reveal to the next layer?"

### C3: Forget Gate Bias = 1
If initialized to 0, initial forget gate $\approx 0.5$, which:
- Immediately starts erasing cell state
- Makes early gradients vanish

With bias = 1, $f_t \approx \sigma(1) = 0.73$:
- Cell state mostly preserved initially
- Gradients flow through the cell highway
- Network learns when to forget, rather than defaulting to forgetting

### C4: RNN vs LSTM vs GRU Comparison
| Aspect | RNN | LSTM | GRU |
|--------|-----|------|-----|
| Parameters | Low ($H^2 + HD$) | High ($4(H^2 + HD)$) | Medium ($3(H^2 + HD)$) |
| Gradient flow | Poor | Good (cell highway) | Good (update gate) |
| Gates | 0 | 3 (f, i, o) | 2 (update, reset) |
| When to use | Short sequences | Long dependencies | Similar to LSTM, fewer params |

### C5: Transformers vs LSTMs
**Transformers:**
- Direct attention to any position → O(1) gradient path
- Parallel computation → faster training
- No inherent sequence bias → needs positional encoding
- Quadratic memory in sequence length

**LSTMs:**
- Sequential processing → can't parallelize
- Inductive bias for sequences → good for small data
- Linear memory in sequence length
- Still useful for very long sequences (memory efficient)

### C6: Information-Theoretic View of Cell State
The cell state can be viewed as a **memory buffer** that stores compressed information about the past:

- **Write** (input gate): Add bits of information
- **Erase** (forget gate): Remove irrelevant bits
- **Read** (output gate): Expose relevant bits

This implements an information bottleneck:
- Only task-relevant information is preserved
- Forget gate determines what to compress/discard
- The cell state holds maximum relevant information about past in fixed-size vector

The gating mechanism learns to maximize mutual information $I(c_t; Y)$ while minimizing $I(c_t; X_{irrelevant})$.
