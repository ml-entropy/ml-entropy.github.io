# Tutorial 13: Recurrent Neural Networks — Learning Sequences

## Overview
RNNs process sequential data by maintaining hidden state across time steps. We derive the forward and backward passes (BPTT), understand vanishing gradients, and see how LSTM/GRU solve them.

---

## Part 1: Why Sequences Need Special Treatment

### The Problem with Feedforward Networks

Standard networks assume **fixed-size input**. But many data types are sequential:
- Text: Variable-length sentences
- Audio: Different duration sounds
- Time series: Arbitrary length signals

### What Makes Sequences Special

1. **Variable length**: Can't use fixed architecture
2. **Order matters**: "dog bites man" ≠ "man bites dog"
3. **Long-range dependencies**: Meaning spans many timesteps

### The Solution: Recurrence

Process one element at a time, maintaining **hidden state**:
$$h_t = f(h_{t-1}, x_t)$$

The same function $f$ applied at every timestep.

---

## Part 2: Basic RNN Architecture

### The Equations

At each timestep $t$:

**Hidden state update**:
$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

**Output** (optional, at each timestep):
$$y_t = W_{hy}h_t + b_y$$

### Unrolling Through Time

```
x₁    x₂    x₃    x₄
 ↓     ↓     ↓     ↓
[RNN]→[RNN]→[RNN]→[RNN]
 ↓     ↓     ↓     ↓
h₁    h₂    h₃    h₄
 ↓     ↓     ↓     ↓
y₁    y₂    y₃    y₄
```

**Key insight**: Same weights $(W_{hh}, W_{xh}, W_{hy})$ at every timestep.

### Types of RNN Tasks

**Many-to-Many** (same length): POS tagging
```
[x₁, x₂, x₃] → [y₁, y₂, y₃]
```

**Many-to-One**: Sentiment classification
```
[x₁, x₂, x₃] → y
```

**One-to-Many**: Image captioning
```
x → [y₁, y₂, y₃]
```

**Many-to-Many** (different length): Translation (encoder-decoder)
```
[x₁, x₂] → [y₁, y₂, y₃]
```

---

## Part 3: Backpropagation Through Time (BPTT)

### The Computational Graph

Loss is sum over timesteps:
$$L = \sum_{t=1}^{T} L_t$$

Each $L_t$ depends on $h_t$, which depends on all previous hidden states.

### Forward Pass (for reference)

```python
h[0] = initial_state
for t in range(1, T+1):
    h[t] = tanh(W_hh @ h[t-1] + W_xh @ x[t] + b_h)
    y[t] = W_hy @ h[t] + b_y
    L[t] = loss(y[t], target[t])
L_total = sum(L)
```

### Backward Pass Derivation

**Gradient w.r.t. output weights**:
$$\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} h_t^T$$

**Gradient w.r.t. hidden state** (the key challenge):

Define $\delta_t = \frac{\partial L}{\partial h_t}$ (includes contributions from $L_t, L_{t+1}, ..., L_T$).

At final timestep:
$$\delta_T = W_{hy}^T \frac{\partial L_T}{\partial y_T}$$

For earlier timesteps (recursive!):
$$\delta_t = W_{hy}^T \frac{\partial L_t}{\partial y_t} + W_{hh}^T \delta_{t+1} \odot (1 - h_{t+1}^2)$$

The $(1 - h_{t+1}^2)$ comes from $\tanh'(z) = 1 - \tanh^2(z)$.

**Gradient w.r.t. recurrent weights**:
$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t \odot (1 - h_t^2) \cdot h_{t-1}^T$$

### The BPTT Algorithm

```python
# Backward pass
delta[T] = W_hy.T @ dL_dy[T]

for t in range(T-1, 0, -1):
    delta[t] = W_hy.T @ dL_dy[t] + W_hh.T @ (delta[t+1] * (1 - h[t+1]**2))

# Accumulate gradients
dW_hh = sum(delta[t] * (1 - h[t]**2) @ h[t-1].T for t in range(1, T+1))
dW_xh = sum(delta[t] * (1 - h[t]**2) @ x[t].T for t in range(1, T+1))
dW_hy = sum(dL_dy[t] @ h[t].T for t in range(1, T+1))
```

---

## Part 4: The Vanishing/Exploding Gradient Problem

### The Problem

From the recursion:
$$\delta_t = W_{hh}^T \delta_{t+1} \odot (1 - h_{t+1}^2) + ...$$

After $k$ steps:
$$\delta_t \propto \prod_{i=t}^{t+k} W_{hh}^T \cdot \text{diag}(1 - h_i^2)$$

**If** $\|W_{hh}\| < 1$: Gradient vanishes exponentially
**If** $\|W_{hh}\| > 1$: Gradient explodes exponentially

### Mathematical Analysis

Let $\lambda$ be the largest eigenvalue of $W_{hh}$.

After $T$ timesteps, gradient scaled by $\approx \lambda^T$.

- $\lambda = 0.9$, $T = 100$: $0.9^{100} \approx 10^{-5}$ (vanishing!)
- $\lambda = 1.1$, $T = 100$: $1.1^{100} \approx 10^{4}$ (exploding!)

### The Consequence

**Vanishing gradients**: Can't learn long-range dependencies
- Early timesteps get negligible gradient
- Model only learns short-term patterns

**Exploding gradients**: Training becomes unstable
- Can fix with gradient clipping

### Gradient Clipping

If $\|\nabla\| > \text{threshold}$:
$$\nabla \leftarrow \frac{\text{threshold}}{\|\nabla\|} \cdot \nabla$$

Prevents explosion but doesn't fix vanishing.

---

## Part 5: Long Short-Term Memory (LSTM)

### The Key Idea

Add a **cell state** $c_t$ that can carry information unchanged across many timesteps.

Use **gates** to control information flow.

### The LSTM Equations

**Forget gate**: What to forget from cell state
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$

**Input gate**: What new information to add
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$

**Candidate cell state**: Proposed new information
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$$

**Cell state update**: Forget old + add new
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate**: What to output from cell state
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$$

**Hidden state**: Filtered cell state
$$h_t = o_t \odot \tanh(c_t)$$

### Why LSTMs Solve Vanishing Gradients

**The cell state highway**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

If $f_t \approx 1$ and $i_t \approx 0$:
$$c_t \approx c_{t-1}$$

Gradient flows **unchanged** through the cell state!

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

If forget gate learns to be near 1, gradients don't vanish.

### LSTM Backward Pass (Sketch)

The key equation:
$$\frac{\partial L}{\partial c_{t-1}} = \frac{\partial L}{\partial c_t} \cdot f_t + \frac{\partial L}{\partial h_t} \cdot o_t \cdot (1 - \tanh^2(c_t)) \cdot f_t$$

The $f_t$ factor means gradients can flow when $f_t \approx 1$.

---

## Part 6: Gated Recurrent Unit (GRU)

### Simplified Gating

GRU combines forget and input gates:

**Reset gate**: How much of past to forget
$$r_t = \sigma(W_r[h_{t-1}, x_t] + b_r)$$

**Update gate**: How much to update vs keep
$$z_t = \sigma(W_z[h_{t-1}, x_t] + b_z)$$

**Candidate hidden state**:
$$\tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1}, x_t] + b_h)$$

**Hidden state update**:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### GRU vs LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | 2 (cell + hidden) | 1 (hidden only) |
| Parameters | More | Fewer |
| Performance | Often similar | Often similar |

GRU is simpler; LSTM slightly more flexible.

---

## Part 7: Bidirectional RNNs

### The Idea

Process sequence both forward and backward:

**Forward**: $\overrightarrow{h}_t$ sees $x_1, ..., x_t$
**Backward**: $\overleftarrow{h}_t$ sees $x_T, ..., x_t$

**Combined**: $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$

### Why Bidirectional?

For many tasks, future context helps:
- "The **bank** of the river" vs "The **bank** gave a loan"
- Need to see "river" or "loan" to understand "bank"

### When NOT to Use

Can't use for online/streaming tasks where future isn't available.

---

## Part 8: The Information-Theoretic View

### Hidden State as Compression

The hidden state $h_t$ is a **lossy compression** of $x_1, ..., x_t$.

Information bottleneck:
$$I(h_t; x_1, ..., x_t) \leq H(h_t)$$

Fixed-size $h_t$ can only hold so much information!

### Gates as Information Control

- **Forget gate**: Controls what information to discard
- **Input gate**: Controls what new information to store
- **Output gate**: Controls what information to expose

Gates implement **selective information flow**.

### Long-Range Dependencies and Information

To capture long-range dependency between $x_1$ and $x_T$:
$$I(h_T; x_1) > 0$$

Vanishing gradients mean $I(h_T; x_1) \to 0$.

LSTM cell state maintains:
$$I(c_T; x_1) > I(h_T; x_1)$$

### Entropy and Sequence Modeling

Next-token prediction = entropy minimization:
$$H(x_t | x_1, ..., x_{t-1})$$

Better model = lower conditional entropy = better compression.

**Language modeling IS compression!**

---

## Part 9: Modern Developments

### Attention Mechanism

Instead of fixed-size $h_t$, attend to all past states:
$$\text{context}_t = \sum_{i=1}^{t-1} \alpha_{ti} h_i$$

where $\alpha_{ti}$ are learned attention weights.

### Transformers

Replace recurrence entirely with self-attention:
- Parallel processing (faster training)
- Direct connections to all positions (no vanishing gradient)
- Dominant architecture today

### When to Still Use RNNs?

- Online/streaming applications
- Very long sequences (memory constraints)
- Simple sequential patterns
- When parallelization isn't needed

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| RNN | Same weights, hidden state carries information |
| BPTT | Backprop through unrolled network |
| Vanishing gradient | $\lambda^T$ → 0 for $\lambda < 1$ |
| LSTM | Cell state highway preserves gradients |
| GRU | Simpler gating, similar performance |
| Bidirectional | Use future context when available |

**The key insight**: Sequence modeling requires maintaining and selectively updating a compressed representation of history. Gates (LSTM/GRU) learn to control information flow, enabling long-range dependencies that vanilla RNNs cannot capture.
