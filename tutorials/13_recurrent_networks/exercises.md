# Tutorial 13: Recurrent Neural Networks - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**RNN forward pass equations**

For a vanilla RNN:
1. Write the equation for $h_t$ in terms of $h_{t-1}$, $x_t$, and weights
2. How many weight matrices are needed?
3. Why is tanh commonly used as activation?

---

### Exercise A2 🟢 (Easy)
**Unrolling through time**

For a sequence of length $T$:
1. Draw the unrolled computational graph
2. How many times are the same weights used?
3. What is the total number of multiplications (in terms of hidden size $H$ and input size $D$)?

---

### Exercise A3 🟡 (Medium)
**Derive BPTT gradients**

For the loss $\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$:

1. Derive $\frac{\partial \mathcal{L}}{\partial W_{hh}}$
2. Show this involves products of Jacobians: $\prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$
3. Why does this lead to vanishing/exploding gradients?

---

### Exercise A4 🔴 (Hard)
**Analyze gradient magnitude**

For $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$:

1. Compute the Jacobian $\frac{\partial h_t}{\partial h_{t-1}}$
2. Show this equals $\text{diag}(1 - h_t^2) \cdot W_{hh}$
3. Let $\lambda_{max}$ be the largest eigenvalue of $W_{hh}$. Under what conditions on $\lambda_{max}$ and the activation derivatives will gradients vanish?

---

### Exercise A5 🔴 (Hard)
**LSTM gradient flow**

For LSTM cell state: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$

1. Compute $\frac{\partial c_t}{\partial c_{t-1}}$
2. Why does this allow gradients to flow unchanged when $f_t \approx 1$?
3. Derive the condition for when LSTM gradients vanish

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Implement vanilla RNN cell**

```python
class RNNCell:
    def __init__(self, input_size, hidden_size):
        """
        Initialize RNN cell with random weights.
        
        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, x, h_prev):
        """
        Single step forward.
        
        Args:
            x: Input at current timestep (input_size,)
            h_prev: Previous hidden state (hidden_size,)
        
        Returns:
            h: New hidden state (hidden_size,)
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B2 🟡 (Medium)
**Implement BPTT**

```python
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        """Full RNN with output layer."""
        # YOUR CODE HERE
        pass
    
    def forward(self, xs):
        """
        Process sequence and return outputs at each timestep.
        Store all hidden states for backward pass.
        """
        # YOUR CODE HERE
        pass
    
    def backward(self, douts):
        """
        Backpropagation through time.
        
        Args:
            douts: Upstream gradients at each timestep
        
        Returns:
            Gradients for all weights
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B3 🟡 (Medium)
**Implement gradient clipping**

```python
def clip_gradients(grads, max_norm):
    """
    Clip gradients by global norm.
    
    If ||grads|| > max_norm, scale all gradients by max_norm / ||grads||
    
    Args:
        grads: List of gradient arrays
        max_norm: Maximum allowed norm
    
    Returns:
        Clipped gradients
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B4 🔴 (Hard)
**Implement LSTM cell**

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        """
        LSTM cell with forget, input, and output gates.
        
        Equations:
        f_t = σ(W_f [h_{t-1}, x_t] + b_f)
        i_t = σ(W_i [h_{t-1}, x_t] + b_i)
        c̃_t = tanh(W_c [h_{t-1}, x_t] + b_c)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        o_t = σ(W_o [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(c_t)
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, x, h_prev, c_prev):
        """
        Single LSTM step.
        
        Returns:
            h: New hidden state
            c: New cell state
            cache: Values for backward pass
        """
        # YOUR CODE HERE
        pass
    
    def backward(self, dh, dc, cache):
        """
        Backward pass through LSTM cell.
        
        Returns:
            Gradients for all weights and dx, dh_prev, dc_prev
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B5 🔴 (Hard)
**Compare gradient flow: RNN vs LSTM**

```python
def analyze_gradient_flow(sequence_length, hidden_size):
    """
    Compare how gradients flow through RNN and LSTM.
    
    1. Create random sequences
    2. Compute loss at final timestep
    3. Measure gradient magnitude at each timestep
    4. Plot gradient magnitude vs timestep for both architectures
    5. Quantify the vanishing gradient effect
    
    Returns:
        Dictionary with gradient magnitudes for RNN and LSTM
    """
    # YOUR CODE HERE
    pass
```

---

## Part C: Conceptual Questions

### C1 🟢
Why do RNNs have the vanishing gradient problem but feedforward networks don't (for the same depth)?

### C2 🟢
Explain the role of each LSTM gate:
- Forget gate: ?
- Input gate: ?
- Output gate: ?

### C3 🟡
Why is the forget gate bias often initialized to 1 or a positive value?

### C4 🟡
Compare RNN, LSTM, and GRU:
- Parameters: ?
- Gradient flow: ?
- When to use each: ?

### C5 🔴
How do Transformers solve the long-range dependency problem differently than LSTMs? What are the trade-offs?

### C6 🔴
From an information-theoretic perspective, what is the LSTM cell state storing? How does this relate to the information bottleneck?
